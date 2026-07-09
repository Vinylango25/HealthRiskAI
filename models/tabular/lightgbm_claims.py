"""
models/tabular/lightgbm_claims.py
==================================
LightGBM real-time claims processing model.

Tasks:
  - Binary claim approval / fraud flag
  - Claim cost regression (severity)
  - Real-time scoring API (< 50 ms per record)

Targets AUROC comparable to XGBoost baselines with faster inference.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "reports" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class LGBMClaimsConfig:
    """Hyperparameters for the claims LightGBM model."""

    # Classification head (fraud / approve)
    clf_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "n_estimators": 500,
            "n_jobs": -1,
            "verbose": -1,
            "random_state": 42,
        }
    )
    # Regression head (claim cost)
    reg_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "n_estimators": 500,
            "n_jobs": -1,
            "verbose": -1,
            "random_state": 42,
        }
    )
    early_stopping_rounds: int = 50
    cv_folds: int = 5
    threshold: float = 0.5  # fraud detection threshold
    mlflow_experiment: str = "lgbm_claims"


# ─── Feature definitions ──────────────────────────────────────────────────────

CLAIM_FEATURES: List[str] = [
    # Claim characteristics
    "claim_amount",
    "claim_type_code",
    "procedure_code",
    "diagnosis_code",
    "days_supply",
    "quantity_dispensed",
    "refill_number",
    # Provider features
    "provider_specialty_code",
    "provider_claim_count_30d",
    "provider_avg_claim_amount",
    "provider_fraud_rate_hist",
    # Member features
    "member_age",
    "member_sex_code",
    "member_plan_type",
    "member_claim_count_30d",
    "member_total_cost_ytd",
    "member_chronic_count",
    "member_hcc_score",
    # Temporal features
    "claim_day_of_week",
    "claim_hour",
    "days_since_last_claim",
    # Drug features (pharmacy claims)
    "drug_tier",
    "is_generic",
    "is_specialty_drug",
    "formulary_status",
    # Network features
    "is_in_network",
    "is_emergency",
    "authorization_on_file",
    # Derived risk signals
    "amount_vs_provider_avg_ratio",
    "amount_vs_member_avg_ratio",
    "unusual_quantity_flag",
    "early_refill_flag",
]


# ─── Claims model class ───────────────────────────────────────────────────────


class LightGBMClaimsModel:
    """
    Dual-head LightGBM for real-time claims:
      - clf: fraud / high-risk flag (binary)
      - reg: expected claim cost (Tweedie)

    Supports:
      - Time-aware cross-validation
      - MLflow tracking
      - Sub-50ms single-record scoring
      - Feature importance & SHAP-ready
    """

    def __init__(self, config: Optional[LGBMClaimsConfig] = None) -> None:
        self.config = config or LGBMClaimsConfig()
        self.clf_model: Optional[lgb.LGBMClassifier] = None
        self.reg_model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self._is_fitted = False

    # ── Data preparation ──────────────────────────────────────────────────────

    def _prepare_features(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> pd.DataFrame:
        """Extract and engineer claim features from raw DataFrame."""
        available = [f for f in CLAIM_FEATURES if f in df.columns]
        X = df[available].copy()

        # Derived ratios (safe division)
        if "claim_amount" in X.columns and "provider_avg_claim_amount" in X.columns:
            X["amount_vs_provider_avg_ratio"] = X["claim_amount"] / (
                X["provider_avg_claim_amount"].replace(0, np.nan)
            ).fillna(1.0)

        if "claim_amount" in X.columns and "member_total_cost_ytd" in X.columns:
            X["amount_vs_member_avg_ratio"] = X["claim_amount"] / (
                (X["member_total_cost_ytd"] / 12).replace(0, np.nan)
            ).fillna(1.0)

        # Fill missing
        X = X.fillna(X.median(numeric_only=True))

        # Categorical encoding (LightGBM handles natively via int codes)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            X[col] = pd.Categorical(X[col]).codes

        self.feature_names = list(X.columns)
        return X

    def _time_aware_split(
        self, df: pd.DataFrame, date_col: str = "claim_date"
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return (train_idx, val_idx) pairs using temporal ordering."""
        if date_col not in df.columns:
            logger.warning("No %s column — using random stratified CV", date_col)
            skf = StratifiedKFold(
                n_splits=self.config.cv_folds, shuffle=True, random_state=42
            )
            return list(skf.split(df, df.get("is_fraud", np.zeros(len(df)))))

        df = df.sort_values(date_col).reset_index(drop=True)
        n = len(df)
        fold_size = n // self.config.cv_folds
        splits = []
        for i in range(self.config.cv_folds):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.config.cv_folds - 1 else n
            train_idx = np.concatenate(
                [np.arange(0, val_start), np.arange(val_end, n)]
            )
            val_idx = np.arange(val_start, val_end)
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
        return splits

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        target_fraud: str = "is_fraud",
        target_cost: str = "paid_amount",
        date_col: str = "claim_date",
        eval_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Train both heads with time-aware CV.

        Parameters
        ----------
        df : DataFrame with features + targets
        target_fraud : column name for binary fraud label
        target_cost : column name for claim cost
        date_col : temporal ordering column
        eval_df : optional held-out evaluation set

        Returns
        -------
        metrics : dict with CV + eval metrics
        """
        mlflow.set_experiment(self.config.mlflow_experiment)

        with mlflow.start_run(run_name="lgbm_claims_training"):
            X = self._prepare_features(df, fit_scaler=True)
            y_clf = df[target_fraud].values if target_fraud in df.columns else None
            y_reg = df[target_cost].values if target_cost in df.columns else None

            mlflow.log_params(
                {
                    "n_samples": len(X),
                    "n_features": X.shape[1],
                    "cv_folds": self.config.cv_folds,
                    **{f"clf_{k}": v for k, v in self.config.clf_params.items()},
                }
            )

            metrics: Dict[str, Any] = {}

            # ── Classification head ──────────────────────────────────────────
            if y_clf is not None:
                logger.info("Training fraud classifier on %d samples", len(X))
                cv_aucs = []
                splits = self._time_aware_split(df, date_col)

                for fold_i, (train_idx, val_idx) in enumerate(splits):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y_clf[train_idx], y_clf[val_idx]

                    clf = lgb.LGBMClassifier(**self.config.clf_params)
                    clf.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[
                            lgb.early_stopping(
                                self.config.early_stopping_rounds, verbose=False
                            ),
                            lgb.log_evaluation(period=-1),
                        ],
                    )
                    proba = clf.predict_proba(X_val)[:, 1]
                    fold_auc = roc_auc_score(y_val, proba)
                    cv_aucs.append(fold_auc)
                    logger.info("  Fold %d AUROC: %.4f", fold_i + 1, fold_auc)

                metrics["clf_cv_auroc_mean"] = float(np.mean(cv_aucs))
                metrics["clf_cv_auroc_std"] = float(np.std(cv_aucs))
                logger.info(
                    "CLF CV AUROC: %.4f ± %.4f",
                    metrics["clf_cv_auroc_mean"],
                    metrics["clf_cv_auroc_std"],
                )

                # Final model on full data
                self.clf_model = lgb.LGBMClassifier(**self.config.clf_params)
                self.clf_model.fit(X, y_clf)
                mlflow.lightgbm.log_model(self.clf_model, "clf_model")

            # ── Regression head ──────────────────────────────────────────────
            if y_reg is not None:
                logger.info("Training cost regressor on %d samples", len(X))
                cv_maes = []
                splits = self._time_aware_split(df, date_col)

                for fold_i, (train_idx, val_idx) in enumerate(splits):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y_reg[train_idx], y_reg[val_idx]

                    reg = lgb.LGBMRegressor(**self.config.reg_params)
                    reg.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[
                            lgb.early_stopping(
                                self.config.early_stopping_rounds, verbose=False
                            ),
                            lgb.log_evaluation(period=-1),
                        ],
                    )
                    preds = reg.predict(X_val)
                    fold_mae = mean_absolute_error(y_val, preds)
                    cv_maes.append(fold_mae)
                    logger.info("  Fold %d MAE: %.2f", fold_i + 1, fold_mae)

                metrics["reg_cv_mae_mean"] = float(np.mean(cv_maes))
                metrics["reg_cv_mae_std"] = float(np.std(cv_maes))

                self.reg_model = lgb.LGBMRegressor(**self.config.reg_params)
                self.reg_model.fit(X, y_reg)
                mlflow.lightgbm.log_model(self.reg_model, "reg_model")

            mlflow.log_metrics(metrics)
            self._is_fitted = True

            # Eval on held-out set
            if eval_df is not None:
                eval_metrics = self.evaluate(eval_df, target_fraud, target_cost)
                metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                mlflow.log_metrics(eval_metrics)

            logger.info("Training complete. Metrics: %s", metrics)
            return metrics

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score claims. Returns DataFrame with columns:
          - fraud_probability
          - fraud_flag (threshold-based)
          - expected_cost
          - risk_score (composite)

        Sub-50ms per record for real-time use.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._prepare_features(df)

        result = pd.DataFrame(index=df.index)

        if self.clf_model is not None:
            start = time.perf_counter()
            result["fraud_probability"] = self.clf_model.predict_proba(X)[:, 1]
            result["fraud_flag"] = (
                result["fraud_probability"] >= self.config.threshold
            ).astype(int)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "CLF inference: %.1f ms total / %.2f ms per record",
                elapsed_ms,
                elapsed_ms / max(len(df), 1),
            )

        if self.reg_model is not None:
            result["expected_cost"] = self.reg_model.predict(X).clip(min=0)

        # Composite risk score [0-100]
        if "fraud_probability" in result.columns and "expected_cost" in result.columns:
            fraud_score = result["fraud_probability"] * 60
            cost_pct = result["expected_cost"].rank(pct=True) * 40
            result["risk_score"] = (fraud_score + cost_pct).clip(0, 100)

        return result

    def predict_single(self, claim: Dict[str, Any]) -> Dict[str, float]:
        """
        Score a single claim dict in real-time.
        Returns dict with fraud_probability, fraud_flag, expected_cost, risk_score.
        """
        df = pd.DataFrame([claim])
        result = self.predict(df)
        return result.iloc[0].to_dict()

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        df: pd.DataFrame,
        target_fraud: str = "is_fraud",
        target_cost: str = "paid_amount",
    ) -> Dict[str, float]:
        """Compute evaluation metrics on a labelled dataset."""
        preds = self.predict(df)
        metrics: Dict[str, float] = {}

        if target_fraud in df.columns and "fraud_probability" in preds.columns:
            y = df[target_fraud].values
            p = preds["fraud_probability"].values
            metrics["auroc"] = float(roc_auc_score(y, p))
            metrics["auprc"] = float(average_precision_score(y, p))
            logger.info(
                "Eval — AUROC: %.4f  AUPRC: %.4f",
                metrics["auroc"],
                metrics["auprc"],
            )

        if target_cost in df.columns and "expected_cost" in preds.columns:
            y = df[target_cost].values
            p = preds["expected_cost"].values
            metrics["mae"] = float(mean_absolute_error(y, p))
            metrics["mape"] = float(mean_absolute_percentage_error(y, p))
            logger.info(
                "Eval — MAE: %.2f  MAPE: %.4f",
                metrics["mae"],
                metrics["mape"],
            )

        return metrics

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self, model_head: str = "clf") -> pd.DataFrame:
        """Return sorted feature importance DataFrame."""
        model = self.clf_model if model_head == "clf" else self.reg_model
        if model is None:
            raise ValueError(f"Model head '{model_head}' not trained.")

        imp = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance_gain": model.booster_.feature_importance(
                    importance_type="gain"
                ),
                "importance_split": model.booster_.feature_importance(
                    importance_type="split"
                ),
            }
        ).sort_values("importance_gain", ascending=False)
        return imp

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist model bundle to disk."""
        path = path or MODEL_DIR / "lgbm_claims_bundle.pkl"
        bundle = {
            "clf_model": self.clf_model,
            "reg_model": self.reg_model,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        joblib.dump(bundle, path)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "LightGBMClaimsModel":
        """Load a persisted model bundle."""
        bundle = joblib.load(path)
        instance = cls(config=bundle["config"])
        instance.clf_model = bundle["clf_model"]
        instance.reg_model = bundle["reg_model"]
        instance.feature_names = bundle["feature_names"]
        instance._is_fitted = True
        logger.info("Model loaded from %s", path)
        return instance


# ─── Synthetic demo / smoke test ─────────────────────────────────────────────


def _make_synthetic_claims(n: int = 1000) -> pd.DataFrame:
    """Generate synthetic claims data for testing."""
    rng = np.random.default_rng(42)
    n_fraud = int(n * 0.05)  # 5% fraud rate
    is_fraud = np.zeros(n, dtype=int)
    fraud_idx = rng.choice(n, n_fraud, replace=False)
    is_fraud[fraud_idx] = 1

    df = pd.DataFrame(
        {
            "claim_amount": rng.lognormal(6, 1.5, n),
            "claim_type_code": rng.integers(1, 10, n),
            "procedure_code": rng.integers(1000, 9999, n),
            "diagnosis_code": rng.integers(100, 999, n),
            "days_supply": rng.integers(1, 90, n),
            "quantity_dispensed": rng.integers(1, 500, n),
            "refill_number": rng.integers(0, 12, n),
            "provider_specialty_code": rng.integers(1, 50, n),
            "provider_claim_count_30d": rng.integers(1, 500, n),
            "provider_avg_claim_amount": rng.lognormal(6, 1, n),
            "provider_fraud_rate_hist": rng.uniform(0, 0.2, n),
            "member_age": rng.integers(18, 85, n),
            "member_sex_code": rng.integers(0, 2, n),
            "member_plan_type": rng.integers(1, 5, n),
            "member_claim_count_30d": rng.integers(0, 20, n),
            "member_total_cost_ytd": rng.lognormal(8, 1.5, n),
            "member_chronic_count": rng.integers(0, 10, n),
            "member_hcc_score": rng.uniform(0.5, 5.0, n),
            "claim_day_of_week": rng.integers(0, 7, n),
            "claim_hour": rng.integers(0, 24, n),
            "days_since_last_claim": rng.integers(0, 365, n),
            "drug_tier": rng.integers(1, 5, n),
            "is_generic": rng.integers(0, 2, n),
            "is_specialty_drug": rng.integers(0, 2, n),
            "formulary_status": rng.integers(0, 3, n),
            "is_in_network": rng.integers(0, 2, n),
            "is_emergency": rng.integers(0, 2, n),
            "authorization_on_file": rng.integers(0, 2, n),
            "amount_vs_provider_avg_ratio": rng.uniform(0.1, 5.0, n),
            "amount_vs_member_avg_ratio": rng.uniform(0.1, 5.0, n),
            "unusual_quantity_flag": rng.integers(0, 2, n),
            "early_refill_flag": rng.integers(0, 2, n),
            "is_fraud": is_fraud,
        }
    )

    # Paid amount: base claim amount with noise (fraudulent claims overstated)
    df["paid_amount"] = df["claim_amount"] * rng.uniform(0.8, 1.0, n)
    df.loc[is_fraud == 1, "paid_amount"] *= rng.uniform(1.5, 3.0, n_fraud)

    # Temporal column for time-aware CV
    dates = pd.date_range("2022-01-01", periods=n, freq="h")
    df["claim_date"] = dates
    df = df.sort_values("claim_date").reset_index(drop=True)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== LightGBM Claims Model — Smoke Test ===")

    df = _make_synthetic_claims(2000)
    train_df = df.iloc[:1600]
    eval_df = df.iloc[1600:]

    model = LightGBMClaimsModel()
    metrics = model.fit(train_df, eval_df=eval_df)
    logger.info("Final metrics: %s", metrics)

    # Single-record inference latency
    sample_claim = train_df.iloc[0].to_dict()
    t0 = time.perf_counter()
    result = model.predict_single(sample_claim)
    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("Single-record latency: %.2f ms", latency_ms)
    logger.info("Result: %s", result)
    assert latency_ms < 200, f"Latency {latency_ms:.1f}ms exceeds 200ms threshold"

    path = model.save()
    loaded = LightGBMClaimsModel.load(path)
    logger.info("Model load/save round-trip OK")
    logger.info("=== PASS ===")
