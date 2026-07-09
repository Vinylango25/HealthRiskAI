"""
bond_spread_model.py
====================
Maps hospital credit scores / PD estimates to OAS (Option-Adjusted Spread)
basis points for hospital bonds.

Architecture
------------
1. Primary model  : Gradient-boosted regression (XGBoost / sklearn GBM)
   Inputs          : credit_score, pd_1yr, lgd_estimate, maturity_years,
                     coupon_rate, system_affiliation, IG_flag
2. Fallback model : Ridge regression (always available)
3. Market calibration : Fitted to a synthetic CMS hospital bond universe

Relative Value
--------------
  model_oas       : model-predicted OAS
  market_oas      : observed OAS (if available)
  rv_signal       : market_oas - model_oas  (positive = cheap, negative = rich)
  rv_category     : CHEAP / FAIR / RICH
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

SPREAD_FEATURES = [
    "credit_score",          # 300-850
    "pd_1yr",                # probability of default
    "lgd_estimate",          # loss given default
    "maturity_years",        # years to maturity
    "coupon_rate",           # annual coupon (decimal)
    "system_affiliation",    # 1 = part of system
    "ig_flag",               # 1 = investment grade (credit_score >= 580)
    "duration",              # modified duration (computed from maturity if missing)
    "market_cap_log",        # log of total debt outstanding
]

# Relative value thresholds (bps)
RV_CHEAP_THRESHOLD =  20.0   # cheap if market OAS > model + 20 bps
RV_RICH_THRESHOLD  = -20.0   # rich  if market OAS < model - 20 bps


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpreadPrediction:
    """Spread model output for a single bond."""
    bond_id: str
    model_oas_bps: float
    market_oas_bps: Optional[float]
    rv_signal_bps: Optional[float]    # market - model
    rv_category: str                  # CHEAP / FAIR / RICH / UNKNOWN
    credit_score: float
    pd_1yr: float


# ---------------------------------------------------------------------------
# Spread Model
# ---------------------------------------------------------------------------

class BondSpreadModel:
    """
    Hospital bond OAS spread model.

    Maps credit score, PD and bond characteristics to OAS spread in bps.

    Parameters
    ----------
    use_xgboost  : Use XGBoost regressor if available, else sklearn GBM.
    alpha        : Ridge regression regularisation strength.
    random_state : Reproducibility seed.

    Usage
    -----
    >>> model = BondSpreadModel()
    >>> model.fit(X_train, y_spread)
    >>> preds = model.predict_spread(X_new)
    >>> rv = model.relative_value(X_mkt, market_oas_series)
    """

    def __init__(
        self,
        use_xgboost: bool = True,
        alpha: float = 1.0,
        random_state: int = 42,
    ):
        self.use_xgboost = use_xgboost and _XGB_AVAILABLE
        self.alpha = alpha
        self.random_state = random_state
        self._fitted = False
        self._scaler = StandardScaler()

        # Primary model
        if self.use_xgboost:
            self._primary = xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=random_state, verbosity=0,
            )
        else:
            self._primary = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=random_state,
            )

        # Fallback Ridge
        self._ridge = Ridge(alpha=alpha)

        # Market calibration offset (bps) — fitted during calibrate()
        self._market_offset: float = 0.0
        self._market_scale:  float = 1.0

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_features(X: pd.DataFrame) -> pd.DataFrame:
        """
        Derive missing features and ensure all SPREAD_FEATURES are present.
        """
        df = X.copy()

        # Derive ig_flag from credit_score if missing
        if "ig_flag" not in df.columns and "credit_score" in df.columns:
            df["ig_flag"] = (df["credit_score"] >= 580).astype(float)

        # Approximate modified duration from maturity if missing
        if "duration" not in df.columns and "maturity_years" in df.columns:
            df["duration"] = df["maturity_years"] * 0.85

        # Log market cap
        if "market_cap_log" not in df.columns:
            df["market_cap_log"] = np.log(500.0)  # default $500M

        for feat in SPREAD_FEATURES:
            if feat not in df.columns:
                df[feat] = 0.0

        return df[SPREAD_FEATURES].fillna(df[SPREAD_FEATURES].median())

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BondSpreadModel":
        """
        Fit the spread model.

        Parameters
        ----------
        X : DataFrame with spread features (see SPREAD_FEATURES).
        y : Series of observed OAS spreads in basis points.
        """
        logger.info("Fitting BondSpreadModel on %d bonds …", len(X))
        X_prep = self._prepare_features(X)
        X_scaled = self._scaler.fit_transform(X_prep)

        # Primary model (XGBoost / GBM)
        if self.use_xgboost:
            self._primary.fit(X_prep, y)
        else:
            self._primary.fit(X_prep, y)
        logger.info("Primary spread model fitted.")

        # Ridge fallback
        self._ridge.fit(X_scaled, y)
        logger.info("Ridge fallback fitted.")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def calibrate(self, X_mkt: pd.DataFrame, market_oas: pd.Series) -> "BondSpreadModel":
        """
        Calibrate model predictions to the current market level.

        Fits a linear scaling: model_calibrated = scale * model_raw + offset.

        Parameters
        ----------
        X_mkt      : Feature DataFrame for bonds with known market prices.
        market_oas : Observed OAS series (same index as X_mkt).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before calibrate().")

        model_preds = self._predict_raw(X_mkt)
        # Simple linear calibration
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(model_preds, market_oas.values)
        self._market_scale  = float(slope)
        self._market_offset = float(intercept)
        logger.info(
            "Market calibration: scale=%.4f  offset=%.2f bps",
            self._market_scale, self._market_offset,
        )
        return self

    # ------------------------------------------------------------------
    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Return uncalibrated model predictions."""
        X_prep = self._prepare_features(X)
        if self.use_xgboost or hasattr(self._primary, "predict"):
            return self._primary.predict(X_prep)
        X_scaled = self._scaler.transform(X_prep)
        return self._ridge.predict(X_scaled)

    # ------------------------------------------------------------------
    def predict_spread(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict OAS spread (bps) for each bond.

        Returns
        -------
        DataFrame with columns: model_oas_bps, credit_grade_proxy.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_spread().")

        raw = self._predict_raw(X)
        calibrated = self._market_scale * raw + self._market_offset
        calibrated = np.clip(calibrated, 0.0, 2000.0)

        df = self._prepare_features(X)
        scores = df["credit_score"].values

        return pd.DataFrame(
            {
                "model_oas_bps":    calibrated.round(1),
                "credit_score":     scores.round(1),
                "ig_flag":          df["ig_flag"].values.astype(int),
            },
            index=X.index,
        )

    # ------------------------------------------------------------------
    def relative_value(
        self,
        X: pd.DataFrame,
        market_oas: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute relative value (cheap/rich) vs model fair value.

        Parameters
        ----------
        X          : Feature DataFrame.
        market_oas : Observed OAS in bps (same index as X).

        Returns
        -------
        DataFrame with: model_oas_bps, market_oas_bps, rv_signal_bps,
                        rv_category (CHEAP / FAIR / RICH).
        """
        preds = self.predict_spread(X)
        rv_signal = market_oas.values - preds["model_oas_bps"].values

        rv_categories = []
        for rv in rv_signal:
            if rv > RV_CHEAP_THRESHOLD:
                rv_categories.append("CHEAP")
            elif rv < RV_RICH_THRESHOLD:
                rv_categories.append("RICH")
            else:
                rv_categories.append("FAIR")

        return pd.DataFrame(
            {
                "model_oas_bps":  preds["model_oas_bps"].values,
                "market_oas_bps": market_oas.values,
                "rv_signal_bps":  rv_signal.round(1),
                "rv_category":    rv_categories,
                "credit_score":   preds["credit_score"].values,
            },
            index=X.index,
        )

    # ------------------------------------------------------------------
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model accuracy on a test set.

        Returns
        -------
        dict: mae_bps, rmse_bps, r2, mape_pct.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before evaluate().")

        preds_df = self.predict_spread(X)
        y_hat    = preds_df["model_oas_bps"].values
        y_arr    = y.values

        mae  = mean_absolute_error(y_arr, y_hat)
        rmse = mean_squared_error(y_arr, y_hat) ** 0.5
        r2   = r2_score(y_arr, y_hat)
        mape = float(np.mean(np.abs((y_arr - y_hat) / (np.abs(y_arr) + 1e-6))) * 100)

        metrics = {
            "mae_bps":  round(mae,  2),
            "rmse_bps": round(rmse, 2),
            "r2":       round(r2,   4),
            "mape_pct": round(mape, 2),
        }
        logger.info(
            "SpreadModel Evaluate | MAE=%.1f bps  RMSE=%.1f bps  R²=%.4f  MAPE=%.1f%%",
            mae, rmse, r2, mape,
        )
        return metrics

    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance from the primary model."""
        if not self._fitted:
            raise RuntimeError("Call fit() before feature_importance().")
        if hasattr(self._primary, "feature_importances_"):
            imp = self._primary.feature_importances_
        else:
            imp = np.abs(self._ridge.coef_)
        return (
            pd.DataFrame({"feature": SPREAD_FEATURES, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Synthetic bond universe generator
# ---------------------------------------------------------------------------

def _generate_synthetic_bonds(
    n: int = 400, seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate a synthetic hospital bond universe calibrated to CMS data ranges.

    Returns
    -------
    X          : Feature DataFrame
    oas_model  : True OAS (generated from credit score signal)
    oas_market : Market OAS with noise (observed prices)
    """
    rng = np.random.default_rng(seed)

    credit_score   = rng.normal(640, 80, n).clip(300, 850)
    pd_1yr         = np.clip(np.exp(-(credit_score - 300) / 200) * 0.15, 0.001, 0.40)
    lgd_estimate   = rng.normal(0.40, 0.08, n).clip(0.15, 0.75)
    maturity_years = rng.choice([5, 7, 10, 15, 20, 30], n).astype(float)
    coupon_rate    = rng.normal(0.05, 0.01, n).clip(0.02, 0.10)
    system_aff     = rng.binomial(1, 0.6, n).astype(float)
    ig_flag        = (credit_score >= 580).astype(float)
    duration       = maturity_years * 0.85
    market_cap_log = rng.normal(np.log(500), 0.8, n)

    X = pd.DataFrame({
        "credit_score":     credit_score,
        "pd_1yr":           pd_1yr,
        "lgd_estimate":     lgd_estimate,
        "maturity_years":   maturity_years,
        "coupon_rate":      coupon_rate,
        "system_affiliation": system_aff,
        "ig_flag":          ig_flag,
        "duration":         duration,
        "market_cap_log":   market_cap_log,
    })

    # True spread: higher PD + LGD + duration → wider spread
    # Calibrated to hospital bond market ranges (IG: 80-200 bps, HY: 200-800 bps)
    true_oas = (
        400.0
        + 2500.0 * pd_1yr
        + 200.0  * lgd_estimate
        - 0.50   * credit_score
        + 5.0    * maturity_years
        - 50.0   * system_aff
        + rng.normal(0, 30, n)            # idiosyncratic noise
    ).clip(30, 1200)

    # Market OAS = true OAS + bid-ask + liquidity noise
    market_oas = true_oas + rng.normal(0, 25, n)
    market_oas = market_oas.clip(10, 1500)

    return X, pd.Series(true_oas, name="oas_bps"), pd.Series(market_oas, name="market_oas_bps")


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("BondSpreadModel — Smoke Test")
    print("=" * 60)

    X, y_true, y_market = _generate_synthetic_bonds(n=500)
    print(f"Bond universe: {len(X)} bonds")
    print(f"OAS range: {y_true.min():.0f} – {y_true.max():.0f} bps  "
          f"(mean {y_true.mean():.0f} bps)")

    split = int(0.7 * len(X))
    X_train, y_train   = X.iloc[:split], y_true.iloc[:split]
    X_test,  y_test    = X.iloc[split:], y_true.iloc[split:]
    mkt_test           = y_market.iloc[split:]

    model = BondSpreadModel(use_xgboost=_XGB_AVAILABLE)
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    print("\nHold-out evaluation (pre-calibration):")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")

    # Market calibration on a 50-bond subset (separate from evaluation)
    model.calibrate(X_train.iloc[:50], y_train.iloc[:50])

    preds = model.predict_spread(X_test.head(5))
    print("\nSample spread predictions:")
    print(preds.to_string())

    rv = model.relative_value(X_test.head(10), mkt_test.head(10))
    print("\nRelative value analysis (first 10 bonds):")
    print(rv.to_string())

    print("\nFeature importance:")
    print(model.feature_importance().head(5).to_string(index=False))

    assert metrics["r2"] > 0.50, f"R² too low: {metrics['r2']}"
    print("\n✓ BondSpreadModel smoke test passed.")
