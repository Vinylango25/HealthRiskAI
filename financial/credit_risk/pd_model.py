"""
pd_model.py
===========
Probability of Default (PD) model for hospital bonds.

Architecture
------------
1. Baseline  : Logistic Regression on scaled features
2. Enhanced  : XGBoost gradient-boosted classifier
3. Ensemble  : Weighted average of baseline + enhanced predictions

PD horizons : 1-year, 3-year, 5-year (via hazard-rate term structure)
LGD         : Recovery-adjusted Loss Given Default estimate
EL          : Expected Loss = PD × LGD × EAD

Through-the-cycle (TTC) PD : blended toward long-run average default rate
Point-in-time (PIT) PD     : current-conditions ensemble prediction

Features
--------
Financial ratios  : debt_to_ebitda, operating_margin, days_cash_on_hand,
                    current_ratio, debt_service_coverage, revenue_growth
Market position   : market_share, system_affiliation, bed_count_log
Payor mix         : medicare_pct, medicaid_pct, commercial_pct
Regulatory flags  : cms_penalty_flag, joint_commission_flag, dsh_flag
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn("xgboost not installed — enhanced model disabled.", stacklevel=2)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

FINANCIAL_FEATURES = [
    "debt_to_ebitda", "operating_margin", "days_cash_on_hand",
    "current_ratio", "debt_service_coverage", "revenue_growth",
]
MARKET_FEATURES = ["market_share", "system_affiliation", "bed_count_log"]
PAYOR_FEATURES  = ["medicare_pct", "medicaid_pct", "commercial_pct"]
REG_FEATURES    = ["cms_penalty_flag", "joint_commission_flag", "dsh_flag"]
ALL_FEATURES    = FINANCIAL_FEATURES + MARKET_FEATURES + PAYOR_FEATURES + REG_FEATURES

# Cumulative PD term-structure multipliers (hazard-rate model)
TTC_MULTIPLIERS: Dict[int, float] = {1: 1.0, 3: 2.4, 5: 3.6}

# Long-run average 1-yr hospital bond default rate
LRA_DEFAULT_RATE = 0.025

# LGD parameters for senior secured hospital bonds
LGD_BASE              = 0.40
LGD_UNSECURED_ADD     = 0.15
LGD_NON_PROFIT_DISC   = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PDOutput:
    """Structured PD output for a single obligor."""
    hospital_id: str
    pd_1yr: float
    pd_3yr: float
    pd_5yr: float
    lgd_estimate: float
    expected_loss_1yr: float
    pd_ttc: float
    pd_pit: float
    model_version: str = "v1.0"


# ---------------------------------------------------------------------------
# LGD estimator
# ---------------------------------------------------------------------------

class LGDEstimator:
    """
    Rule-based LGD model for hospital bonds.

    Parameters are calibrated to historical healthcare sector recoveries.
    Senior secured non-profit bonds recover ~60-70 cents on the dollar.
    """

    def estimate(
        self,
        secured: bool = True,
        non_profit: bool = True,
        debt_to_ebitda: float = 4.5,
    ) -> float:
        """
        Estimate LGD for a single bond.

        Parameters
        ----------
        secured        : True if bond has first-lien security pledge.
        non_profit     : Non-profits benefit from asset protection covenants.
        debt_to_ebitda : Higher leverage → lower recovery.

        Returns
        -------
        float : LGD in [0.10, 0.90].
        """
        lgd = LGD_BASE
        if not secured:
            lgd += LGD_UNSECURED_ADD
        if non_profit:
            lgd -= LGD_NON_PROFIT_DISC
        # Leverage stress: +2 pp per turn of leverage above 4x
        lgd += max(0.0, (debt_to_ebitda - 4.0) * 0.02)
        return float(np.clip(lgd, 0.10, 0.90))


# ---------------------------------------------------------------------------
# PD Model
# ---------------------------------------------------------------------------

class HospitalPDModel:
    """
    Probability of Default model for hospital bonds.

    Parameters
    ----------
    use_xgboost         : Enable XGBoost enhanced model (requires xgboost package).
    ensemble_weight_xgb : Weight given to XGBoost in final ensemble (0-1).
    ttc_blend           : 0 = pure PIT, 1 = pure TTC blend.
    random_state        : Reproducibility seed.

    Usage
    -----
    >>> model = HospitalPDModel()
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict_pd(X_new)
    >>> metrics = model.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        use_xgboost: bool = True,
        ensemble_weight_xgb: float = 0.6,
        ttc_blend: float = 0.3,
        random_state: int = 42,
    ):
        self.use_xgboost = use_xgboost and _XGB_AVAILABLE
        self.ensemble_weight_xgb = ensemble_weight_xgb
        self.ttc_blend = ttc_blend
        self.random_state = random_state
        self._fitted = False
        self._scaler = StandardScaler()
        self._lr: Optional[CalibratedClassifierCV] = None
        self._xgb: Optional[CalibratedClassifierCV] = None
        self._lgd_estimator = LGDEstimator()

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HospitalPDModel":
        """
        Fit baseline logistic regression and (optionally) XGBoost models.

        Parameters
        ----------
        X : DataFrame with columns in ALL_FEATURES.
        y : Binary series — 1 = default, 0 = performing.
        """
        logger.info(
            "Fitting HospitalPDModel on %d samples (default rate=%.3f) …",
            len(X), y.mean(),
        )
        X_fit = X[ALL_FEATURES].copy().fillna(X[ALL_FEATURES].median())
        X_scaled = self._scaler.fit_transform(X_fit)

        # Logistic regression with isotonic calibration
        self._lr = CalibratedClassifierCV(
            LogisticRegression(
                C=0.5, solver="lbfgs", max_iter=1000,
                random_state=self.random_state,
            ),
            method="isotonic", cv=3,
        )
        self._lr.fit(X_scaled, y)
        logger.info("Logistic baseline fitted.")

        # XGBoost enhanced model
        if self.use_xgboost:
            self._xgb = CalibratedClassifierCV(
                xgb.XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=self.random_state,
                    verbosity=0,
                ),
                method="isotonic", cv=3,
            )
            self._xgb.fit(X_fit, y)
            logger.info("XGBoost enhanced model fitted.")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _raw_pit_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Return 1-yr PIT PD from the ensemble."""
        X_proc = X[ALL_FEATURES].copy().fillna(0.0)
        X_scaled = self._scaler.transform(X_proc)
        lr_pd = self._lr.predict_proba(X_scaled)[:, 1]

        if self.use_xgboost and self._xgb is not None:
            xgb_pd = self._xgb.predict_proba(X_proc)[:, 1]
            w = self.ensemble_weight_xgb
            return w * xgb_pd + (1.0 - w) * lr_pd
        return lr_pd

    # ------------------------------------------------------------------
    def _pit_to_ttc(self, pd_pit: np.ndarray) -> np.ndarray:
        """
        Blend PIT estimate toward TTC long-run average.
        TTC_PD = alpha * LRA + (1 - alpha) * PIT_PD
        """
        return self.ttc_blend * LRA_DEFAULT_RATE + (1.0 - self.ttc_blend) * pd_pit

    # ------------------------------------------------------------------
    @staticmethod
    def _term_structure(pd_1yr: np.ndarray, horizon: int) -> np.ndarray:
        """
        Multi-year cumulative PD via simple hazard-rate model.
        Cumulative PD_n ≈ 1 - (1 - h)^n  where h ≈ annual PD.
        """
        mult = TTC_MULTIPLIERS.get(horizon, float(horizon))
        return np.clip(1.0 - (1.0 - pd_1yr) ** mult, 0.0, 1.0)

    # ------------------------------------------------------------------
    def predict_pd(
        self,
        X: pd.DataFrame,
        secured: bool = True,
        non_profit: bool = True,
        ead: float = 1.0,
    ) -> pd.DataFrame:
        """
        Predict PD, LGD and Expected Loss for each row.

        Parameters
        ----------
        X          : Feature DataFrame.
        secured    : Bond seniority for LGD estimation.
        non_profit : Non-profit status for LGD estimation.
        ead        : Exposure at default (fraction of par, default 1.0).

        Returns
        -------
        DataFrame with columns: pd_1yr, pd_3yr, pd_5yr, pd_pit, pd_ttc,
                                 lgd_estimate, expected_loss_1yr.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_pd().")

        pd_pit = self._raw_pit_pd(X)
        pd_ttc = self._pit_to_ttc(pd_pit)
        pd_3yr = self._term_structure(pd_pit, 3)
        pd_5yr = self._term_structure(pd_pit, 5)

        d2e_vals = X.get(
            "debt_to_ebitda",
            pd.Series(np.full(len(X), 4.5), index=X.index),
        ).values
        lgd = np.array([
            self._lgd_estimator.estimate(
                secured=secured, non_profit=non_profit, debt_to_ebitda=float(d)
            )
            for d in d2e_vals
        ])
        el = pd_pit * lgd * ead

        return pd.DataFrame(
            {
                "pd_1yr":            pd_pit.round(6),
                "pd_3yr":            pd_3yr.round(6),
                "pd_5yr":            pd_5yr.round(6),
                "pd_pit":            pd_pit.round(6),
                "pd_ttc":            pd_ttc.round(6),
                "lgd_estimate":      lgd.round(4),
                "expected_loss_1yr": el.round(6),
            },
            index=X.index,
        )

    # ------------------------------------------------------------------
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate discrimination and calibration on a labelled dataset.

        Returns dict with keys: auc, gini, brier.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluate().")

        results = self.predict_pd(X)
        pd_hat  = results["pd_1yr"].values
        y_arr   = y.values

        auc   = roc_auc_score(y_arr, pd_hat)
        gini  = 2.0 * auc - 1.0
        brier = brier_score_loss(y_arr, pd_hat)

        metrics = {
            "auc":   round(auc,   4),
            "gini":  round(gini,  4),
            "brier": round(brier, 4),
        }
        logger.info(
            "PD Evaluate | AUC=%.4f  Gini=%.4f  Brier=%.4f",
            auc, gini, brier,
        )
        return metrics

    # ------------------------------------------------------------------
    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_folds: int = 5
    ) -> Dict[str, float]:
        """
        K-fold cross-validation returning mean AUC and Gini.
        Uses the logistic baseline for speed.
        """
        X_proc   = X[ALL_FEATURES].fillna(X[ALL_FEATURES].median())
        X_scaled = self._scaler.fit_transform(X_proc)
        lr_base  = LogisticRegression(
            C=0.5, solver="lbfgs", max_iter=1000, random_state=self.random_state
        )
        cv    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        proba = cross_val_predict(lr_base, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        auc   = roc_auc_score(y, proba)
        return {"cv_auc": round(auc, 4), "cv_gini": round(2.0 * auc - 1.0, 4)}


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def _generate_synthetic_hospitals(
    n: int = 600, seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic hospital bond dataset for testing."""
    rng = np.random.default_rng(seed)
    data: Dict[str, np.ndarray] = {
        "debt_to_ebitda":        rng.normal(4.5, 2.0, n).clip(0, 15),
        "operating_margin":      rng.normal(0.04, 0.05, n).clip(-0.20, 0.25),
        "days_cash_on_hand":     rng.normal(150, 60, n).clip(10, 400),
        "current_ratio":         rng.normal(1.8, 0.5, n).clip(0.5, 5.0),
        "debt_service_coverage": rng.normal(2.5, 1.0, n).clip(0.5, 8.0),
        "revenue_growth":        rng.normal(0.03, 0.04, n).clip(-0.15, 0.20),
        "market_share":          rng.beta(2, 5, n),
        "system_affiliation":    rng.binomial(1, 0.6, n).astype(float),
        "bed_count_log":         rng.normal(5.5, 0.8, n).clip(3, 8),
        "medicare_pct":          rng.beta(4, 4, n),
        "medicaid_pct":          rng.beta(2, 6, n),
        "commercial_pct":        rng.beta(3, 5, n),
        "cms_penalty_flag":      rng.binomial(1, 0.15, n).astype(float),
        "joint_commission_flag": rng.binomial(1, 0.85, n).astype(float),
        "dsh_flag":              rng.binomial(1, 0.30, n).astype(float),
    }
    X = pd.DataFrame(data)
    log_p = (
        -3.0
        + 0.25 * X["debt_to_ebitda"]
        - 8.0  * X["operating_margin"]
        - 0.004 * X["days_cash_on_hand"]
        - 0.3  * X["debt_service_coverage"]
        - 1.0  * X["system_affiliation"]
        + 0.6  * X["cms_penalty_flag"]
        + 0.2  * X["dsh_flag"]
        - 0.15 * X["bed_count_log"]
    )
    prob = 1.0 / (1.0 + np.exp(-log_p.values))
    y = pd.Series((rng.uniform(size=n) < prob).astype(int), name="default")
    return X, y


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HospitalPDModel — Smoke Test")
    print("=" * 60)

    X, y = _generate_synthetic_hospitals(n=800)
    print(f"Dataset: {len(X)} hospitals | Default rate: {y.mean():.2%}")

    split = int(0.7 * len(X))
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test,  y_test  = X.iloc[split:], y.iloc[split:]

    model = HospitalPDModel(use_xgboost=_XGB_AVAILABLE, ttc_blend=0.3)
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    print("\nHold-out evaluation:")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v:.4f}")

    preds = model.predict_pd(X_test.head(5))
    print("\nSample PD predictions:")
    print(preds.to_string())

    cv = model.cross_validate(X, y, n_folds=5)
    print(f"\nCross-val: AUC={cv['cv_auc']:.4f}  Gini={cv['cv_gini']:.4f}")

    assert metrics["auc"] > 0.60, f"AUC below threshold: {metrics['auc']}"
    print("\n✓ HospitalPDModel smoke test passed.")
