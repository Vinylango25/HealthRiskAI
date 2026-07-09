"""
premium_pricer.py
=================
GLM-based actuarial premium pricing model with HealthRisk AI enhancement.

Architecture:
  1. GLM Baseline  – statsmodels Tweedie/Poisson for frequency & severity
  2. Risk Factor   – age, sex, plan type, HCC score, chronic conditions, geography
  3. AI Uplift     – XGBoost residual model on top of GLM predictions
  4. Outputs       – base_premium, risk_adjusted_premium, ai_enhanced_premium

Target: predictive ratio (actual / expected) in [0.95, 1.05].
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Optional heavy dependencies — degrade gracefully so smoke test always runs
# ---------------------------------------------------------------------------
try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import Tweedie
    from statsmodels.genmod.families.links import log as TweedieLog

    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False
    logger.warning("statsmodels not installed – GLM baseline will use a linear approximation.")

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False
    logger.warning("xgboost not installed – AI uplift layer will be skipped.")

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    HAS_MLFLOW = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PricingResult:
    """Container for premium pricing outputs per member."""

    member_id: np.ndarray
    base_premium: np.ndarray
    risk_adjusted_premium: np.ndarray
    ai_enhanced_premium: np.ndarray
    predictive_ratio: Optional[np.ndarray] = None

    def to_dataframe(self) -> pd.DataFrame:
        d = {
            "member_id": self.member_id,
            "base_premium": self.base_premium,
            "risk_adjusted_premium": self.risk_adjusted_premium,
            "ai_enhanced_premium": self.ai_enhanced_premium,
        }
        if self.predictive_ratio is not None:
            d["predictive_ratio"] = self.predictive_ratio
        return pd.DataFrame(d)


@dataclass
class EvaluationMetrics:
    """Actuarial and ML evaluation metrics."""

    mean_predictive_ratio: float
    pct_within_band: float          # % of members with ratio in [0.95, 1.05]
    rmse: float
    mae: float
    gini_coefficient: float
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Label-encode categorical columns in-place and return updated frame."""
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def _gini(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Normalised Gini coefficient (actuarial lift metric)."""
    n = len(actual)
    if n == 0:
        return 0.0
    order = np.argsort(predicted)
    cum_actual = np.cumsum(actual[order]) / actual.sum()
    gini = 1 - 2 * cum_actual.mean()
    return float(gini)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class PremiumPricer:
    """
    Actuarial premium pricing model combining a GLM baseline with an
    XGBoost AI uplift layer.

    Parameters
    ----------
    base_rate : float
        Community base rate (PMPM) before any risk adjustment.
    tweedie_power : float
        Tweedie power parameter (1 < p < 2 for compound Poisson-Gamma).
    xgb_params : dict, optional
        Override default XGBoost hyperparameters.
    mlflow_experiment : str, optional
        MLflow experiment name.  Tracking disabled if None.
    """

    CATEGORICAL_COLS = ["sex", "plan_type", "geography"]
    FEATURE_COLS = [
        "age", "sex", "plan_type", "hcc_score",
        "chronic_count", "geography", "prior_utilization",
    ]

    def __init__(
        self,
        base_rate: float = 400.0,
        tweedie_power: float = 1.5,
        xgb_params: Optional[dict] = None,
        mlflow_experiment: Optional[str] = "healthrisk_premium_pricing",
    ) -> None:
        self.base_rate = base_rate
        self.tweedie_power = tweedie_power
        self.xgb_params: dict = xgb_params or {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        self.mlflow_experiment = mlflow_experiment

        self._glm_model = None
        self._xgb_model: Optional[xgb.XGBRegressor] = None  # type: ignore[name-defined]
        self._feature_means: dict = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive model features from raw member data.

        Expected raw columns (minimum):
          age, sex, plan_type, hcc_score, chronic_count,
          geography, prior_utilization
        """
        df = df.copy()
        # Age-band interaction
        df["age_sq"] = df["age"] ** 2
        df["age_hcc"] = df["age"] * df["hcc_score"]
        df["chronic_hcc"] = df["chronic_count"] * df["hcc_score"]
        df = _encode_categoricals(df, self.CATEGORICAL_COLS)
        return df

    def _glm_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return GLM predicted PMPM."""
        if self._glm_model is None or not HAS_STATSMODELS:
            # Fallback: simple linear scoring
            score = (
                self.base_rate
                + 3.5 * X["age"].values
                + 80 * X["hcc_score"].values
                + 25 * X.get("chronic_count", pd.Series(np.zeros(len(X)))).values
            )
            return np.clip(score, 50, 5000)
        X_sm = sm.add_constant(X, has_constant="add")
        return self._glm_model.predict(X_sm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, target_col: str = "actual_pmpm") -> "PremiumPricer":
        """
        Fit both the GLM baseline and XGBoost uplift model.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with member risk factors and historical PMPM.
        target_col : str
            Column name for observed per-member per-month cost.

        Returns
        -------
        self
        """
        logger.info("Starting PremiumPricer.fit() on %d members", len(df))
        X = self._build_features(df)
        y = df[target_col].values.astype(float)

        model_cols = self.FEATURE_COLS + ["age_sq", "age_hcc", "chronic_hcc"]
        available = [c for c in model_cols if c in X.columns]

        # --- GLM baseline ---
        if HAS_STATSMODELS:
            try:
                X_sm = sm.add_constant(X[available], has_constant="add")
                family = sm.families.Tweedie(
                    var_power=self.tweedie_power,
                    link=sm.families.links.log(),
                )
                self._glm_model = sm.GLM(y, X_sm, family=family).fit(maxiter=100)
                logger.info("GLM fitted. Deviance: %.4f", self._glm_model.deviance)
            except Exception as exc:
                logger.warning("GLM fitting failed (%s); using fallback scorer.", exc)
                self._glm_model = None
        else:
            self._glm_model = None

        # GLM predictions used as offset for XGBoost
        glm_pred = self._glm_predict(X[available])
        residuals = y - glm_pred

        # --- XGBoost uplift ---
        if HAS_XGB:
            X_xgb = X[available].values
            X_tr, X_val, r_tr, r_val = train_test_split(
                X_xgb, residuals, test_size=0.2, random_state=42
            )
            self._xgb_model = xgb.XGBRegressor(**self.xgb_params)
            self._xgb_model.fit(
                X_tr, r_tr,
                eval_set=[(X_val, r_val)],
                verbose=False,
            )
            logger.info("XGBoost uplift model fitted.")

        self._fitted_cols = available
        self._feature_means = X[available].mean().to_dict()
        self._is_fitted = True

        # --- MLflow tracking ---
        if HAS_MLFLOW and self.mlflow_experiment:
            self._log_to_mlflow(df, y, glm_pred)

        return self

    def predict(self, df: pd.DataFrame) -> PricingResult:
        """
        Generate premium predictions for a member population.

        Returns
        -------
        PricingResult
            base_premium, risk_adjusted_premium, ai_enhanced_premium per member.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        X = self._build_features(df)
        available = self._fitted_cols

        # Fill missing columns with training means
        for col in available:
            if col not in X.columns:
                X[col] = self._feature_means.get(col, 0.0)

        base = np.full(len(df), self.base_rate)

        # Risk-adjusted: GLM output
        risk_adj = self._glm_predict(X[available])
        risk_adj = np.clip(risk_adj, 50, 50_000)

        # AI-enhanced: GLM + XGBoost residual
        if HAS_XGB and self._xgb_model is not None:
            uplift = self._xgb_model.predict(X[available].values)
            ai_enhanced = np.clip(risk_adj + uplift, 50, 50_000)
        else:
            ai_enhanced = risk_adj.copy()

        member_ids = (
            df["member_id"].values if "member_id" in df.columns
            else np.arange(len(df))
        )

        return PricingResult(
            member_id=member_ids,
            base_premium=base,
            risk_adjusted_premium=risk_adj,
            ai_enhanced_premium=ai_enhanced,
        )

    def evaluate(self, df: pd.DataFrame, actual_col: str = "actual_pmpm") -> EvaluationMetrics:
        """
        Evaluate predictive performance against observed costs.

        Parameters
        ----------
        df : pd.DataFrame
            Holdout set with actual PMPM column.
        actual_col : str
            Column name for observed cost.

        Returns
        -------
        EvaluationMetrics
        """
        result = self.predict(df)
        actual = df[actual_col].values.astype(float)
        predicted = result.ai_enhanced_premium

        # Predictive ratio (A/E)
        pred_ratio = np.where(predicted > 0, actual / predicted, 1.0)
        mean_ratio = float(np.mean(pred_ratio))
        pct_within = float(np.mean((pred_ratio >= 0.95) & (pred_ratio <= 1.05)) * 100)

        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mae = float(np.mean(np.abs(actual - predicted)))
        gini = _gini(actual, predicted)

        result.predictive_ratio = pred_ratio

        logger.info(
            "Evaluation — Mean A/E: %.4f | Within-band: %.1f%% | RMSE: %.2f | Gini: %.4f",
            mean_ratio, pct_within, rmse, gini,
        )

        return EvaluationMetrics(
            mean_predictive_ratio=mean_ratio,
            pct_within_band=pct_within,
            rmse=rmse,
            mae=mae,
            gini_coefficient=gini,
            details={"n_members": len(df), "target_band": "[0.95, 1.05]"},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log_to_mlflow(self, df: pd.DataFrame, y: np.ndarray, glm_pred: np.ndarray) -> None:
        """Log training artefacts to MLflow."""
        try:
            mlflow.set_experiment(self.mlflow_experiment)
            with mlflow.start_run(run_name="premium_pricer_train"):
                mlflow.log_param("base_rate", self.base_rate)
                mlflow.log_param("tweedie_power", self.tweedie_power)
                mlflow.log_param("n_members", len(df))
                glm_rmse = float(np.sqrt(np.mean((y - glm_pred) ** 2)))
                mlflow.log_metric("glm_rmse", glm_rmse)
                logger.info("MLflow run logged.")
        except Exception as exc:
            logger.debug("MLflow logging skipped: %s", exc)


# ---------------------------------------------------------------------------
# Synthetic data generator for smoke test
# ---------------------------------------------------------------------------


def _generate_synthetic_members(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 85, size=n)
    hcc = np.clip(rng.gamma(1.2, 0.5, size=n), 0.1, 8.0)
    chronic = rng.integers(0, 7, size=n)
    true_pmpm = (
        200
        + 3.0 * ages
        + 120 * hcc
        + 30 * chronic
        + rng.normal(0, 40, size=n)
    )
    return pd.DataFrame({
        "member_id": np.arange(n),
        "age": ages,
        "sex": rng.choice(["M", "F"], size=n),
        "plan_type": rng.choice(["HMO", "PPO", "EPO", "HDHP"], size=n),
        "hcc_score": hcc.round(3),
        "chronic_count": chronic,
        "geography": rng.choice(["NE", "SE", "MW", "W", "SW"], size=n),
        "prior_utilization": rng.gamma(2, 200, size=n).round(2),
        "actual_pmpm": np.clip(true_pmpm, 50, 6000).round(2),
    })


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== PremiumPricer smoke test ===")

    data = _generate_synthetic_members(n=2000)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    pricer = PremiumPricer(base_rate=400.0, tweedie_power=1.5, mlflow_experiment=None)
    pricer.fit(train_df, target_col="actual_pmpm")

    predictions = pricer.predict(test_df)
    print("\nSample predictions:")
    print(predictions.to_dataframe().head(10).to_string(index=False))

    metrics = pricer.evaluate(test_df, actual_col="actual_pmpm")
    print(f"\nEvaluation Metrics:")
    print(f"  Mean Predictive Ratio (A/E) : {metrics.mean_predictive_ratio:.4f}")
    print(f"  % Members within [0.95-1.05]: {metrics.pct_within_band:.1f}%")
    print(f"  RMSE                        : {metrics.rmse:.2f}")
    print(f"  MAE                         : {metrics.mae:.2f}")
    print(f"  Gini Coefficient            : {metrics.gini_coefficient:.4f}")

    assert 0.5 <= metrics.mean_predictive_ratio <= 2.0, "Predictive ratio outside sanity bounds"
    logger.info("=== Smoke test PASSED ===")
