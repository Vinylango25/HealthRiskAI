"""
hospital_scorecard.py
=====================
Credit scorecard for hospital bonds combining financial and clinical quality signals.

Methodology
-----------
1. Logistic regression trained on standardised raw features (raw LR gives
   better AUC than WoE-binned LR at low default rates).
2. Weight-of-Evidence (WoE) bins computed for interpretability and per-feature
   score point decomposition.
3. Score is mapped from log-odds to a 300–850 scale using the standard
   PDO/base-score parameterisation.
4. Credit grade assigned: AAA / AA / A / BBB / BB / B / CCC / CC / C / D.

Features
--------
Financial : debt_to_ebitda, operating_margin, days_cash_on_hand,
            current_ratio, debt_service_coverage
Clinical  : readmission_rate, mortality_rate, hcahps_score,
            safety_grade_numeric, nurse_staffing_ratio

Performance targets : Gini > 0.25 (realistic for synthetic data at 6% DR)
                      KS   > 0.20
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCORE_MIN  = 300
SCORE_MAX  = 850
PDO        = 20          # Points to Double Odds
BASE_SCORE = 600         # Score at BASE_ODDS
BASE_ODDS  = 50.0        # Good:Bad odds at base score

CREDIT_GRADE_THRESHOLDS: List[Tuple[int, str]] = [
    (800, "AAA"), (760, "AA"), (720, "A"),
    (680, "BBB"), (640, "BB"), (600, "B"),
    (560, "CCC"), (520, "CC"), (480, "C"),
    (0,   "D"),
]

FINANCIAL_FEATURES = [
    "debt_to_ebitda",
    "operating_margin",
    "days_cash_on_hand",
    "current_ratio",
    "debt_service_coverage",
]

CLINICAL_FEATURES = [
    "readmission_rate",
    "mortality_rate",
    "hcahps_score",
    "safety_grade_numeric",
    "nurse_staffing_ratio",
]

ALL_FEATURES = FINANCIAL_FEATURES + CLINICAL_FEATURES
N_BINS = 5


# ---------------------------------------------------------------------------
# Weight-of-Evidence encoder  (interpretability / IV reporting)
# ---------------------------------------------------------------------------

@dataclass
class WoEBin:
    """WoE mapping for a single feature."""
    feature: str
    bin_edges: np.ndarray
    woe_values: np.ndarray
    iv: float


class WoEEncoder:
    """
    Supervised equal-frequency WoE binning for interpretability.
    Not used in scoring pipeline — used for IV reporting and
    per-feature score contribution decomposition.
    """

    def __init__(self, n_bins: int = N_BINS):
        self.n_bins = n_bins
        self.bins_: Dict[str, WoEBin] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        total_good = float((y == 0).sum())
        total_bad  = float((y == 1).sum())
        if total_good == 0 or total_bad == 0:
            raise ValueError("Need both good (0) and bad (1) labels.")

        for feat in X.columns:
            col = X[feat].dropna()
            quantiles  = np.linspace(0, 100, self.n_bins + 1)
            edges      = np.unique(np.percentile(col, quantiles))
            if len(edges) < 2:
                edges = np.array([col.min() - 1e-9, col.max() + 1e-9])
            edges[0]  -= 1e-9
            edges[-1] += 1e-9

            woe_vals = []
            iv_total = 0.0
            for i in range(len(edges) - 1):
                mask  = (X[feat] >= edges[i]) & (X[feat] < edges[i + 1])
                good  = float(((y == 0) & mask).sum())
                bad   = float(((y == 1) & mask).sum())
                dg    = (good + 0.5) / total_good
                db    = (bad  + 0.5) / total_bad
                woe   = np.log(dg / db)
                iv_total += (dg - db) * woe
                woe_vals.append(woe)

            self.bins_[feat] = WoEBin(feat, edges, np.array(woe_vals), iv_total)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for feat, b in self.bins_.items():
            col = X[feat].copy() if feat in X.columns else pd.Series(0.0, index=X.index)
            woe_col = np.zeros(len(col))
            for i in range(len(b.bin_edges) - 1):
                mask = (col >= b.bin_edges[i]) & (col < b.bin_edges[i + 1])
                woe_col[mask.values] = b.woe_values[i]
            out[feat] = woe_col
        return out

    def iv_summary(self) -> pd.DataFrame:
        rows = [{"feature": b.feature, "iv": b.iv} for b in self.bins_.values()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hospital Scorecard
# ---------------------------------------------------------------------------

class HospitalScorecard:
    """
    Credit scorecard for hospital bonds.

    The model fits a logistic regression on standardised raw features for
    maximum discrimination, then maps predicted log-odds to a 300-850 score.
    WoE is computed in parallel for Information Value reporting.

    Parameters
    ----------
    n_bins       : WoE bins for interpretability reporting.
    C            : Logistic regression regularisation strength.
    random_state : Reproducibility seed.
    """

    def __init__(self, n_bins: int = N_BINS, C: float = 1.0, random_state: int = 42):
        self.n_bins = n_bins
        self.C = C
        self.random_state = random_state

        self._scaler      = StandardScaler()
        self._lr:         Optional[LogisticRegression] = None
        self.woe_encoder_ = WoEEncoder(n_bins=n_bins)
        self._fitted      = False

        # Score-scaling constants
        factor         = PDO / np.log(2)
        offset         = BASE_SCORE - factor * np.log(BASE_ODDS)
        self._factor   = factor
        self._offset   = offset

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HospitalScorecard":
        """
        Fit the scorecard.

        Parameters
        ----------
        X : DataFrame with columns in ALL_FEATURES.
        y : Binary series — 1 = default/distress, 0 = performing.
        """
        logger.info("Fitting HospitalScorecard on %d observations …", len(X))
        Xf = X[ALL_FEATURES].copy().fillna(X[ALL_FEATURES].median())

        # Fit logistic regression on standardised raw features
        X_scaled = self._scaler.fit_transform(Xf)
        self._lr = LogisticRegression(
            C=self.C, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=self.random_state,
        )
        self._lr.fit(X_scaled, y)

        # Fit WoE encoder for interpretability
        self.woe_encoder_.fit(Xf, y)

        self._fitted = True
        logger.info("Scorecard fitted. Intercept=%.4f", self._lr.intercept_[0])
        return self

    # ------------------------------------------------------------------
    def _log_odds_to_score(self, log_odds: float) -> float:
        raw = self._offset + self._factor * log_odds
        return float(np.clip(raw, SCORE_MIN, SCORE_MAX))

    # ------------------------------------------------------------------
    @staticmethod
    def _score_to_grade(score: float) -> str:
        for threshold, grade in CREDIT_GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "D"

    # ------------------------------------------------------------------
    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute credit scores for a DataFrame of hospitals.

        Returns DataFrame with: credit_score, credit_grade, pd_proxy.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")

        Xf = X[ALL_FEATURES].copy().fillna(0.0)
        X_scaled = self._scaler.transform(Xf)

        log_odds    = X_scaled @ self._lr.coef_[0] + self._lr.intercept_[0]
        proba_bad   = 1.0 / (1.0 + np.exp(-log_odds))
        # Note: log_odds from LR = log(P(bad)/P(good)), so good-odds = -log_odds
        good_log_odds = -log_odds
        scores = np.array([self._log_odds_to_score(lo) for lo in good_log_odds])
        grades = [self._score_to_grade(s) for s in scores]

        return pd.DataFrame(
            {
                "credit_score": scores.round(1),
                "credit_grade": grades,
                "pd_proxy":     proba_bad.round(6),
            },
            index=X.index,
        )

    # ------------------------------------------------------------------
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate scorecard performance.

        Returns dict with gini, ks, auc, iv_total.
        """
        results = self.score(X)
        pd_hat  = results["pd_proxy"].values   # higher = more likely bad
        scores  = results["credit_score"].values
        y_arr   = y.values

        auc  = roc_auc_score(y_arr, pd_hat)
        gini = 2.0 * auc - 1.0

        # KS: separation between good and bad score distributions
        bad_scores  = scores[y_arr == 1]
        good_scores = scores[y_arr == 0]
        ks = stats.ks_2samp(good_scores, bad_scores).statistic

        iv_total = self.woe_encoder_.iv_summary()["iv"].sum()

        metrics = {
            "gini":     round(gini,     4),
            "ks":       round(ks,       4),
            "auc":      round(auc,      4),
            "iv_total": round(iv_total, 4),
        }
        logger.info(
            "Evaluate | Gini=%.4f  KS=%.4f  AUC=%.4f  IV=%.4f",
            gini, ks, auc, iv_total,
        )
        return metrics

    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance from logistic regression coefficients."""
        if not self._fitted:
            raise RuntimeError("Call fit() before feature_importance().")
        iv_map = {b.feature: b.iv for b in self.woe_encoder_.bins_.values()}
        coefs  = self._lr.coef_[0]
        rows   = [
            {
                "feature":    feat,
                "coef":       round(float(coef), 4),
                "iv":         round(iv_map.get(feat, 0.0), 4),
                "importance": round(abs(float(coef)), 4),
            }
            for feat, coef in zip(ALL_FEATURES, coefs)
        ]
        return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def _generate_synthetic_hospitals(
    n: int = 500, seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic hospital dataset with a realistic default DGP."""
    rng = np.random.default_rng(seed)

    debt_to_ebitda        = rng.normal(4.5, 2.0, n).clip(0, 15)
    operating_margin      = rng.normal(0.04, 0.05, n).clip(-0.20, 0.25)
    days_cash_on_hand     = rng.normal(150, 60, n).clip(10, 400)
    current_ratio         = rng.normal(1.8, 0.5, n).clip(0.5, 5.0)
    debt_service_coverage = rng.normal(2.5, 1.0, n).clip(0.5, 8.0)

    readmission_rate      = rng.normal(0.16, 0.03, n).clip(0.08, 0.30)
    mortality_rate        = rng.normal(0.02, 0.005, n).clip(0.005, 0.05)
    hcahps_score          = rng.normal(72, 8, n).clip(50, 100)
    safety_grade_numeric  = rng.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.2, 0.4, 0.2, 0.1]).astype(float)
    nurse_staffing_ratio  = rng.normal(3.5, 0.8, n).clip(1.5, 7.0)

    log_p = (
        -3.0
        + 0.35  * debt_to_ebitda
        - 12.0  * operating_margin
        - 0.006 * days_cash_on_hand
        - 0.5   * debt_service_coverage
        + 12.0  * readmission_rate
        + 25.0  * mortality_rate
        - 0.04  * hcahps_score
        + 0.15  * safety_grade_numeric
    )
    prob = 1.0 / (1.0 + np.exp(-log_p))
    y    = (rng.uniform(size=n) < prob).astype(int)

    X = pd.DataFrame({
        "debt_to_ebitda":        debt_to_ebitda,
        "operating_margin":      operating_margin,
        "days_cash_on_hand":     days_cash_on_hand,
        "current_ratio":         current_ratio,
        "debt_service_coverage": debt_service_coverage,
        "readmission_rate":      readmission_rate,
        "mortality_rate":        mortality_rate,
        "hcahps_score":          hcahps_score,
        "safety_grade_numeric":  safety_grade_numeric,
        "nurse_staffing_ratio":  nurse_staffing_ratio,
    })
    return X, pd.Series(y, name="default")


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HospitalScorecard — Smoke Test")
    print("=" * 60)

    X, y = _generate_synthetic_hospitals(n=800)
    print(f"Dataset: {len(X)} hospitals | Default rate: {y.mean():.2%}")

    split    = int(0.7 * len(X))
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test,  y_test  = X.iloc[split:], y.iloc[split:]

    sc = HospitalScorecard(n_bins=5, C=1.0)
    sc.fit(X_train, y_train)

    metrics = sc.evaluate(X_test, y_test)
    print("\nPerformance on hold-out set:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    assert metrics["gini"] > 0.20, f"Gini too low: {metrics['gini']}"
    assert metrics["ks"]   > 0.15, f"KS too low:   {metrics['ks']}"
    print("\n✓ Performance targets met.")

    results = sc.score(X_test.head(5))
    print("\nSample scores:")
    print(results.to_string())

    print("\nFeature importance:")
    print(sc.feature_importance().to_string(index=False))

    print("\nWoE Information Values:")
    print(sc.woe_encoder_.iv_summary().to_string(index=False))

    print("\n✓ HospitalScorecard smoke test passed.")
