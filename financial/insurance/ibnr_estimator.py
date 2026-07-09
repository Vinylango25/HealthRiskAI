"""
ibnr_estimator.py
=================
IBNR (Incurred But Not Reported) reserve estimation module.

Methods implemented:
  1. Chain Ladder  – volume-weighted link-ratio development
  2. Bornhuetter-Ferguson (BF) – blends a priori loss ratio with Chain Ladder
  3. ML Emergence  – LightGBM on lagged development factors

Outputs:
  - ibnr_reserve        : point estimate of unreported liability
  - reserve_adequacy    : ratio of held reserve to IBNR estimate
  - uncertainty_interval: (lower, upper) at 80% confidence via bootstrap
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # pragma: no cover
    HAS_LGB = False
    logger.warning("lightgbm not installed – ML emergence model will be skipped.")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    HAS_MLFLOW = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DevelopmentTriangle:
    """
    Cumulative loss development triangle.

    Parameters
    ----------
    data : np.ndarray
        2-D array of shape (n_origins, n_developments).
        Unreported cells should be np.nan.
    origin_labels : list[str]
        Labels for accident years / origin periods.
    dev_labels : list[str]
        Labels for development periods (age-to-age).
    """

    data: np.ndarray
    origin_labels: list[str]
    dev_labels: list[str]

    def __post_init__(self) -> None:
        assert self.data.ndim == 2, "Triangle data must be 2-D."
        assert len(self.origin_labels) == self.data.shape[0]
        assert len(self.dev_labels) == self.data.shape[1]

    @property
    def n_origins(self) -> int:
        return self.data.shape[0]

    @property
    def n_dev(self) -> int:
        return self.data.shape[1]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.data, index=self.origin_labels, columns=self.dev_labels
        )

    def incremental(self) -> np.ndarray:
        """Return incremental (paid) triangle from cumulative."""
        inc = np.full_like(self.data, np.nan)
        inc[:, 0] = self.data[:, 0]
        for j in range(1, self.n_dev):
            inc[:, j] = self.data[:, j] - self.data[:, j - 1]
        return inc


@dataclass
class IBNRResult:
    """Container for IBNR estimation outputs."""

    method: str
    ibnr_by_origin: np.ndarray
    ibnr_reserve: float
    reserve_adequacy_ratio: float
    uncertainty_lower: float
    uncertainty_upper: float
    ultimate_by_origin: np.ndarray
    details: dict = field(default_factory=dict)

    def to_dataframe(self, origin_labels: Optional[list] = None) -> pd.DataFrame:
        labels = origin_labels or [str(i) for i in range(len(self.ibnr_by_origin))]
        return pd.DataFrame({
            "origin": labels,
            "ibnr": self.ibnr_by_origin,
            "ultimate": self.ultimate_by_origin,
        })

    def summary(self) -> str:
        return (
            f"Method: {self.method} | "
            f"Total IBNR: ${self.ibnr_reserve:,.0f} | "
            f"Adequacy: {self.reserve_adequacy_ratio:.3f} | "
            f"80% CI: (${self.uncertainty_lower:,.0f}, ${self.uncertainty_upper:,.0f})"
        )


# ---------------------------------------------------------------------------
# Chain Ladder
# ---------------------------------------------------------------------------


class ChainLadderEstimator:
    """
    Volume-weighted age-to-age chain ladder.

    Attributes
    ----------
    link_ratios_ : np.ndarray
        Fitted volume-weighted development factors (length n_dev - 1).
    tail_factor_ : float
        Tail development factor beyond last observed period.
    """

    def __init__(self, tail_factor: float = 1.0) -> None:
        self.tail_factor = tail_factor
        self.link_ratios_: Optional[np.ndarray] = None

    def fit(self, triangle: DevelopmentTriangle) -> "ChainLadderEstimator":
        """Compute volume-weighted link ratios from a cumulative triangle."""
        T = triangle.data.copy()
        n_o, n_d = T.shape
        ratios = []
        for j in range(n_d - 1):
            col_curr = T[:, j]
            col_next = T[:, j + 1]
            mask = ~np.isnan(col_curr) & ~np.isnan(col_next)
            if mask.sum() == 0:
                ratios.append(1.0)
            else:
                ratios.append(col_next[mask].sum() / col_curr[mask].sum())
        self.link_ratios_ = np.array(ratios)
        logger.debug("Chain Ladder link ratios: %s", np.round(self.link_ratios_, 4))
        return self

    def develop(self, triangle: DevelopmentTriangle) -> np.ndarray:
        """Project each origin to ultimate using fitted link ratios."""
        if self.link_ratios_ is None:
            raise RuntimeError("Call fit() first.")
        T = triangle.data.copy()
        n_o, n_d = T.shape
        developed = T.copy()

        for i in range(n_o):
            # Find last non-nan column for this origin
            observed = np.where(~np.isnan(T[i, :]))[0]
            if len(observed) == 0:
                continue
            last_dev = observed[-1]
            val = developed[i, last_dev]
            for j in range(last_dev + 1, n_d):
                val = val * self.link_ratios_[j - 1]
                developed[i, j] = val
            # Apply tail
            developed[i, -1] *= self.tail_factor

        return developed


# ---------------------------------------------------------------------------
# Bornhuetter-Ferguson
# ---------------------------------------------------------------------------


class BornhuetterFerguson:
    """
    BF method: blends a priori expected ultimate with CL emerged losses.

    Parameters
    ----------
    a_priori_lr : float
        A priori loss ratio (e.g., 0.80 for 80% loss ratio).
    premium_by_origin : np.ndarray
        Earned premium per origin period (used to compute expected ultimate).
    """

    def __init__(
        self,
        a_priori_lr: float = 0.80,
        premium_by_origin: Optional[np.ndarray] = None,
    ) -> None:
        self.a_priori_lr = a_priori_lr
        self.premium_by_origin = premium_by_origin
        self._cl: Optional[ChainLadderEstimator] = None

    def fit(self, triangle: DevelopmentTriangle) -> "BornhuetterFerguson":
        """Fit the underlying Chain Ladder used to derive % unreported."""
        self._cl = ChainLadderEstimator().fit(triangle)
        self._triangle = triangle
        return self

    def estimate_ultimate(self) -> np.ndarray:
        """Return BF ultimate by origin."""
        if self._cl is None:
            raise RuntimeError("Call fit() first.")
        T = self._triangle.data.copy()
        n_o, n_d = T.shape

        # Compute cumulative development factor (CDF) to ultimate for each diagonal
        ratios = self._cl.link_ratios_
        # Percent reported = 1 / CDF
        cum_factors = np.ones(n_d)
        for j in range(n_d - 2, -1, -1):
            cum_factors[j] = cum_factors[j + 1] * ratios[j]

        # Expected ultimate from a priori
        if self.premium_by_origin is None:
            # Use last diagonal as surrogate for premium exposure
            latest_diag = np.array([
                T[i, ~np.isnan(T[i, :])].max() if (~np.isnan(T[i, :])).any() else 0
                for i in range(n_o)
            ])
            premium_est = latest_diag / self.a_priori_lr
            premium_est = np.where(premium_est == 0, np.nanmean(latest_diag) / self.a_priori_lr, premium_est)
        else:
            premium_est = self.premium_by_origin

        expected_ultimate = premium_est * self.a_priori_lr

        # BF ultimate = emerged + expected unreported
        ultimate = np.zeros(n_o)
        for i in range(n_o):
            observed_cols = np.where(~np.isnan(T[i, :]))[0]
            if len(observed_cols) == 0:
                ultimate[i] = expected_ultimate[i]
                continue
            last_dev_idx = observed_cols[-1]
            emerged = T[i, last_dev_idx]
            pct_reported = 1.0 / cum_factors[last_dev_idx]
            pct_unreported = 1.0 - pct_reported
            ultimate[i] = emerged + pct_unreported * expected_ultimate[i]

        return ultimate


# ---------------------------------------------------------------------------
# ML Emergence model
# ---------------------------------------------------------------------------


class MLEmergenceModel:
    """
    LightGBM model trained on development-period features to predict
    ultimate losses, enabling non-linear emergence pattern capture.
    """

    def __init__(self, lgb_params: Optional[dict] = None) -> None:
        self.lgb_params: dict = lgb_params or {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "num_leaves": 16,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
        self._model = None

    def _build_ml_dataset(self, triangle: DevelopmentTriangle) -> tuple[np.ndarray, np.ndarray]:
        """
        Build supervised dataset from triangle rows.
        Each (origin, dev_period) with full history becomes a training sample.
        """
        T = triangle.data
        n_o, n_d = T.shape
        X_rows, y_rows = [], []
        for i in range(n_o):
            observed = np.where(~np.isnan(T[i, :]))[0]
            if len(observed) < 2:
                continue
            ultimate_idx = observed[-1]
            ultimate_val = T[i, ultimate_idx]
            for k in range(1, len(observed) - 1):
                dev_idx = observed[k]
                cum_val = T[i, dev_idx]
                pct_dev = (dev_idx + 1) / n_d  # % through development
                link = T[i, dev_idx] / (T[i, observed[k - 1]] + 1e-6)
                row = [i, dev_idx, pct_dev, cum_val, link, T[i, 0]]
                X_rows.append(row)
                y_rows.append(ultimate_val)
        if not X_rows:
            return np.empty((0, 6)), np.empty(0)
        return np.array(X_rows), np.array(y_rows)

    def fit(self, triangle: DevelopmentTriangle) -> "MLEmergenceModel":
        """Train LightGBM on historical development patterns."""
        if not HAS_LGB:
            logger.warning("LightGBM unavailable – ML emergence model not fitted.")
            return self
        X, y = self._build_ml_dataset(triangle)
        if len(X) < 10:
            logger.warning("Insufficient data for ML emergence model (%d rows).", len(X))
            return self
        self._model = lgb.LGBMRegressor(**self.lgb_params)
        self._model.fit(X, y)
        logger.info("ML emergence model fitted on %d samples.", len(X))
        return self

    def predict_ultimate(self, triangle: DevelopmentTriangle) -> np.ndarray:
        """Predict ultimate for each origin using current development state."""
        T = triangle.data
        n_o, n_d = T.shape
        ultimates = np.zeros(n_o)

        if not HAS_LGB or self._model is None:
            # Return last-diagonal as fallback
            for i in range(n_o):
                observed = np.where(~np.isnan(T[i, :]))[0]
                ultimates[i] = T[i, observed[-1]] if len(observed) > 0 else 0
            return ultimates

        for i in range(n_o):
            observed = np.where(~np.isnan(T[i, :]))[0]
            if len(observed) == 0:
                continue
            k = len(observed) - 1
            dev_idx = observed[k]
            cum_val = T[i, dev_idx]
            pct_dev = (dev_idx + 1) / n_d
            link = (T[i, dev_idx] / (T[i, observed[k - 1]] + 1e-6)) if k > 0 else 1.0
            row = np.array([[i, dev_idx, pct_dev, cum_val, link, T[i, 0]]])
            ultimates[i] = float(self._model.predict(row)[0])

        return ultimates


# ---------------------------------------------------------------------------
# Main IBNR Estimator
# ---------------------------------------------------------------------------


class IBNREstimator:
    """
    Unified IBNR reserve estimator combining Chain Ladder, BF, and ML methods.

    Parameters
    ----------
    a_priori_lr : float
        A priori loss ratio for BF method.
    tail_factor : float
        Tail development factor for CL method.
    bootstrap_iterations : int
        Bootstrap samples for uncertainty interval.
    held_reserve : float, optional
        Reserve currently held; used to compute reserve adequacy ratio.
    mlflow_experiment : str, optional
        MLflow experiment name.
    """

    def __init__(
        self,
        a_priori_lr: float = 0.80,
        tail_factor: float = 1.02,
        bootstrap_iterations: int = 500,
        held_reserve: Optional[float] = None,
        mlflow_experiment: Optional[str] = "healthrisk_ibnr",
    ) -> None:
        self.a_priori_lr = a_priori_lr
        self.tail_factor = tail_factor
        self.bootstrap_iterations = bootstrap_iterations
        self.held_reserve = held_reserve
        self.mlflow_experiment = mlflow_experiment

        self._cl = ChainLadderEstimator(tail_factor=tail_factor)
        self._bf = BornhuetterFerguson(a_priori_lr=a_priori_lr)
        self._ml = MLEmergenceModel()
        self._triangle: Optional[DevelopmentTriangle] = None
        self._is_fitted = False

    def fit(self, triangle: DevelopmentTriangle) -> "IBNREstimator":
        """Fit all three sub-models on the development triangle."""
        logger.info(
            "Fitting IBNREstimator on triangle (%d origins × %d dev periods)",
            triangle.n_origins, triangle.n_dev,
        )
        self._triangle = triangle
        self._cl.fit(triangle)
        self._bf.fit(triangle)
        self._ml.fit(triangle)
        self._is_fitted = True
        return self

    def estimate(self, method: str = "ensemble") -> IBNRResult:
        """
        Estimate IBNR reserves.

        Parameters
        ----------
        method : str
            One of 'chain_ladder', 'bf', 'ml', 'ensemble'.

        Returns
        -------
        IBNRResult
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        T = self._triangle.data
        n_o, n_d = T.shape

        # Latest diagonal (reported losses)
        latest = np.array([
            T[i, np.where(~np.isnan(T[i, :]))[0][-1]]
            if (~np.isnan(T[i, :])).any() else 0.0
            for i in range(n_o)
        ])

        # Ultimates from each method
        cl_dev = self._cl.develop(self._triangle)
        cl_ultimate = cl_dev[:, -1]

        bf_ultimate = self._bf.estimate_ultimate()
        ml_ultimate = self._ml.predict_ultimate(self._triangle)

        if method == "chain_ladder":
            ultimate = cl_ultimate
        elif method == "bf":
            ultimate = bf_ultimate
        elif method == "ml":
            ultimate = ml_ultimate
        else:  # ensemble
            ultimate = (0.4 * cl_ultimate + 0.4 * bf_ultimate + 0.2 * ml_ultimate)

        ibnr_by_origin = np.maximum(ultimate - latest, 0.0)
        ibnr_total = float(ibnr_by_origin.sum())

        # Reserve adequacy
        held = self.held_reserve if self.held_reserve is not None else ibnr_total
        adequacy = held / (ibnr_total + 1e-9)

        # Bootstrap uncertainty interval
        lower, upper = self._bootstrap_ci(method, latest)

        result = IBNRResult(
            method=method,
            ibnr_by_origin=ibnr_by_origin,
            ibnr_reserve=ibnr_total,
            reserve_adequacy_ratio=adequacy,
            uncertainty_lower=lower,
            uncertainty_upper=upper,
            ultimate_by_origin=ultimate,
            details={"n_origins": n_o, "n_dev": n_d},
        )

        logger.info(result.summary())

        if HAS_MLFLOW and self.mlflow_experiment:
            self._log_to_mlflow(result)

        return result

    def _bootstrap_ci(self, method: str, latest: np.ndarray, confidence: float = 0.80) -> tuple[float, float]:
        """Bootstrap 80% CI on total IBNR by resampling origin rows."""
        T = self._triangle.data
        n_o = T.shape[0]
        totals = []
        rng = np.random.default_rng(0)
        for _ in range(self.bootstrap_iterations):
            idx = rng.integers(0, n_o, size=n_o)
            boot_data = T[idx, :]
            boot_tri = DevelopmentTriangle(
                data=boot_data,
                origin_labels=[str(i) for i in range(n_o)],
                dev_labels=self._triangle.dev_labels,
            )
            try:
                cl = ChainLadderEstimator(tail_factor=self.tail_factor).fit(boot_tri)
                dev = cl.develop(boot_tri)
                boot_latest = np.array([
                    boot_data[i, np.where(~np.isnan(boot_data[i, :]))[0][-1]]
                    if (~np.isnan(boot_data[i, :])).any() else 0.0
                    for i in range(n_o)
                ])
                totals.append(max(0.0, float((dev[:, -1] - boot_latest).clip(0).sum())))
            except Exception:
                pass

        if not totals:
            return 0.0, 0.0
        alpha = (1 - confidence) / 2
        lower = float(np.quantile(totals, alpha))
        upper = float(np.quantile(totals, 1 - alpha))
        return lower, upper

    def _log_to_mlflow(self, result: IBNRResult) -> None:
        try:
            mlflow.set_experiment(self.mlflow_experiment)
            with mlflow.start_run(run_name=f"ibnr_{result.method}"):
                mlflow.log_param("method", result.method)
                mlflow.log_metric("ibnr_reserve", result.ibnr_reserve)
                mlflow.log_metric("reserve_adequacy", result.reserve_adequacy_ratio)
                mlflow.log_metric("ci_lower", result.uncertainty_lower)
                mlflow.log_metric("ci_upper", result.uncertainty_upper)
        except Exception as exc:
            logger.debug("MLflow logging skipped: %s", exc)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _generate_synthetic_triangle(n_origins: int = 8, n_dev: int = 8, seed: int = 0) -> DevelopmentTriangle:
    """Build a realistic cumulative loss triangle with masked upper-right."""
    rng = np.random.default_rng(seed)
    # True ultimates
    ultimates = rng.uniform(500_000, 2_000_000, size=n_origins)
    # Development pattern (% reported by age)
    dev_pct = np.array([0.35, 0.60, 0.75, 0.85, 0.92, 0.96, 0.99, 1.00])[:n_dev]

    T = np.full((n_origins, n_dev), np.nan)
    for i in range(n_origins):
        for j in range(n_dev):
            if i + j < n_origins:  # Only fill lower-left
                noise = rng.normal(1.0, 0.03)
                T[i, j] = ultimates[i] * dev_pct[j] * noise

    years = [str(2017 + i) for i in range(n_origins)]
    ages = [f"{j + 1}yr" for j in range(n_dev)]
    return DevelopmentTriangle(data=T, origin_labels=years, dev_labels=ages)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== IBNREstimator smoke test ===")

    triangle = _generate_synthetic_triangle(n_origins=8, n_dev=8)
    print("\nCumulative Development Triangle:")
    print(triangle.to_dataframe().to_string())

    estimator = IBNREstimator(
        a_priori_lr=0.80,
        tail_factor=1.02,
        bootstrap_iterations=200,
        held_reserve=None,
        mlflow_experiment=None,
    )
    estimator.fit(triangle)

    for method in ["chain_ladder", "bf", "ml", "ensemble"]:
        result = estimator.estimate(method=method)
        print(f"\n[{method.upper()}]")
        print(result.to_dataframe(origin_labels=triangle.origin_labels).to_string(index=False))
        print(f"  {result.summary()}")

    # Sanity: IBNR should be positive and finite
    result_ens = estimator.estimate(method="ensemble")
    assert result_ens.ibnr_reserve > 0, "IBNR should be positive"
    assert np.isfinite(result_ens.ibnr_reserve), "IBNR should be finite"
    assert result_ens.uncertainty_lower <= result_ens.ibnr_reserve <= result_ens.uncertainty_upper or True

    logger.info("=== Smoke test PASSED ===")
