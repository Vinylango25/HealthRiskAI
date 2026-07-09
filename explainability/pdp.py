"""
pdp.py
======
Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE)
for HealthRiskAI model explainability.

Provides:
- PDPAnalyzer.compute_pdp()  – average marginal effect (1-way and 2-way)
- PDPAnalyzer.compute_ice()  – per-instance marginal effect
- Centered ICE (c-ICE)       – removes instance-level intercept for cleaner
                                visualisation of heterogeneous effects

All outputs are structured dicts suitable for Angular charting components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PDPResult:
    """
    Result of a 1-way or 2-way Partial Dependence Plot computation.

    For 1-way PDP:
        grid     : list of grid values for the feature
        pdp      : list of average predictions at each grid point

    For 2-way PDP:
        grid     : list of (val_feat1, val_feat2) tuples
        pdp      : flat list of average predictions (row-major: feat1 × feat2)
        grid_shape : (n_feat1, n_feat2)
    """
    feature: Union[str, Tuple[str, str]]
    grid: List[Any]
    pdp: List[float]
    grid_shape: Optional[Tuple[int, int]] = None   # set for 2-way PDP
    feature_index: Union[int, Tuple[int, int]] = -1
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature if isinstance(self.feature, str)
                       else list(self.feature),
            "grid": self.grid,
            "pdp": self.pdp,
            "grid_shape": list(self.grid_shape) if self.grid_shape else None,
            "feature_index": self.feature_index
                             if isinstance(self.feature_index, int)
                             else list(self.feature_index),
            "n_samples": self.n_samples,
            "type": "2way" if self.grid_shape else "1way",
        }


@dataclass
class ICEResult:
    """
    Result of an Individual Conditional Expectation computation.

    ice_lines   : (n_instances, n_grid_points) matrix of raw ICE curves
    centered_ice: (n_instances, n_grid_points) matrix of c-ICE curves
    grid        : list of grid values
    pdp         : list of average values across instances (i.e. the PDP)
    """
    feature: str
    grid: List[float]
    ice_lines: List[List[float]]        # shape: (n_instances, n_grid_points)
    centered_ice: List[List[float]]     # c-ICE: ice_line - ice_line[0]
    pdp: List[float]                    # mean ICE == PDP
    n_instances: int
    feature_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "grid": self.grid,
            "ice_lines": self.ice_lines,
            "centered_ice": self.centered_ice,
            "pdp": self.pdp,
            "n_instances": self.n_instances,
            "feature_index": self.feature_index,
        }


# ---------------------------------------------------------------------------
# PDPAnalyzer
# ---------------------------------------------------------------------------

class PDPAnalyzer:
    """
    Compute PDP and ICE curves for any black-box model.

    Parameters
    ----------
    predict_fn : callable
        predict_fn(X: np.ndarray) → np.ndarray, shape (n_samples,) or (n_samples, n_classes).
        For classifiers, pass predict_proba and set target_class.
    feature_names : list of str
    target_class : int, optional
        For classifiers with multiple output columns, use this column index.
        Defaults to 1 (positive class probability).
    n_grid_points : int
        Number of evenly-spaced grid points per feature (default 50).
    sample_size : int
        Subsample X to this many rows for speed. None = use full dataset.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: List[str],
        target_class: int = 1,
        n_grid_points: int = 50,
        sample_size: Optional[int] = 500,
    ) -> None:
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.target_class = target_class
        self.n_grid_points = n_grid_points
        self.sample_size = sample_size
        self._feature_index: Dict[str, int] = {n: i for i, n in enumerate(feature_names)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_pdp(
        self,
        X: np.ndarray,
        feature: Union[str, Tuple[str, str]],
        grid: Optional[Union[List[float], Tuple[List[float], List[float]]]] = None,
    ) -> PDPResult:
        """
        Compute a 1-way or 2-way Partial Dependence Plot.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        feature : str for 1-way; (str, str) for 2-way
        grid : optional pre-defined grid values

        Returns
        -------
        PDPResult
        """
        X_sub = self._subsample(X)

        if isinstance(feature, (list, tuple)) and not isinstance(feature, str):
            return self._compute_pdp_2way(X_sub, tuple(feature), grid)
        return self._compute_pdp_1way(X_sub, feature, grid)

    def compute_ice(
        self,
        X: np.ndarray,
        feature: str,
        grid: Optional[List[float]] = None,
        n_ice_samples: int = 100,
    ) -> ICEResult:
        """
        Compute Individual Conditional Expectation curves.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        feature : str
        grid : optional pre-defined grid values
        n_ice_samples : int
            Number of instances to compute ICE for (subsample of X).

        Returns
        -------
        ICEResult
        """
        fi = self._get_feature_index(feature)
        grid_values = self._make_grid(X[:, fi], grid)
        X_ice = X[:n_ice_samples] if len(X) > n_ice_samples else X
        n = len(X_ice)

        logger.info("Computing ICE for feature '%s' over %d grid points, %d instances.",
                    feature, len(grid_values), n)

        ice_matrix = np.zeros((n, len(grid_values)))
        for j, gv in enumerate(grid_values):
            X_mod = X_ice.copy()
            X_mod[:, fi] = gv
            preds = self._predict(X_mod)
            ice_matrix[:, j] = preds

        # c-ICE: subtract the value at the first grid point
        centered = ice_matrix - ice_matrix[:, [0]]
        pdp = ice_matrix.mean(axis=0).tolist()

        return ICEResult(
            feature=feature,
            grid=grid_values,
            ice_lines=ice_matrix.tolist(),
            centered_ice=centered.tolist(),
            pdp=pdp,
            n_instances=n,
            feature_index=fi,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_pdp_1way(
        self,
        X: np.ndarray,
        feature: str,
        grid: Optional[List[float]],
    ) -> PDPResult:
        fi = self._get_feature_index(feature)
        grid_values = self._make_grid(X[:, fi], grid)

        logger.info("Computing 1-way PDP for feature '%s' over %d grid points.",
                    feature, len(grid_values))

        pdp_values = []
        for gv in grid_values:
            X_mod = X.copy()
            X_mod[:, fi] = gv
            pdp_values.append(float(self._predict(X_mod).mean()))

        return PDPResult(
            feature=feature,
            grid=grid_values,
            pdp=pdp_values,
            feature_index=fi,
            n_samples=len(X),
        )

    def _compute_pdp_2way(
        self,
        X: np.ndarray,
        features: Tuple[str, str],
        grid: Optional[Tuple[List[float], List[float]]],
    ) -> PDPResult:
        f1, f2 = features
        fi1 = self._get_feature_index(f1)
        fi2 = self._get_feature_index(f2)

        if grid is not None:
            g1, g2 = grid
        else:
            g1 = self._make_grid(X[:, fi1], None)
            g2 = self._make_grid(X[:, fi2], None)

        logger.info("Computing 2-way PDP for features '%s' x '%s' (%dx%d grid).",
                    f1, f2, len(g1), len(g2))

        grid_pairs = []
        pdp_values = []
        for v1 in g1:
            for v2 in g2:
                X_mod = X.copy()
                X_mod[:, fi1] = v1
                X_mod[:, fi2] = v2
                grid_pairs.append([float(v1), float(v2)])
                pdp_values.append(float(self._predict(X_mod).mean()))

        return PDPResult(
            feature=(f1, f2),
            grid=grid_pairs,
            pdp=pdp_values,
            grid_shape=(len(g1), len(g2)),
            feature_index=(fi1, fi2),
            n_samples=len(X),
        )

    def _predict(self, X: np.ndarray) -> np.ndarray:
        out = self.predict_fn(X)
        if out.ndim == 2:
            return out[:, self.target_class]
        return out.flatten()

    def _make_grid(
        self,
        values: np.ndarray,
        grid: Optional[List[float]],
    ) -> List[float]:
        if grid is not None:
            return [float(v) for v in grid]
        return np.linspace(values.min(), values.max(), self.n_grid_points).tolist()

    def _subsample(self, X: np.ndarray) -> np.ndarray:
        if self.sample_size is not None and len(X) > self.sample_size:
            idx = np.random.default_rng(0).choice(len(X), self.sample_size, replace=False)
            return X[idx]
        return X

    def _get_feature_index(self, feature: str) -> int:
        if feature not in self._feature_index:
            raise ValueError(f"Feature '{feature}' not in feature_names: {self.feature_names}")
        return self._feature_index[feature]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.ensemble import GradientBoostingRegressor

    logger.info("=== PDPAnalyzer smoke test ===")

    np.random.seed(42)
    X, y = make_regression(n_samples=400, n_features=8, noise=0.1, random_state=42)
    feature_names = ["hcc_score", "days_cash", "er_visits", "readmissions",
                     "bed_occupancy", "staff_ratio", "lab_cost", "drug_spend"]

    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    def predict_fn(X_: np.ndarray) -> np.ndarray:
        return model.predict(X_)

    analyzer = PDPAnalyzer(
        predict_fn=predict_fn,
        feature_names=feature_names,
        n_grid_points=30,
        sample_size=300,
    )

    # 1-way PDP
    pdp_result = analyzer.compute_pdp(X, feature="hcc_score")
    logger.info("1-way PDP for 'hcc_score': %d grid points, range [%.3f, %.3f]",
                len(pdp_result.grid), min(pdp_result.pdp), max(pdp_result.pdp))

    # 2-way PDP
    pdp2_result = analyzer.compute_pdp(X, feature=("hcc_score", "days_cash"))
    logger.info("2-way PDP: grid_shape=%s, n_values=%d",
                pdp2_result.grid_shape, len(pdp2_result.pdp))

    # ICE
    ice_result = analyzer.compute_ice(X, feature="er_visits", n_ice_samples=50)
    logger.info("ICE for 'er_visits': %d instances, %d grid points",
                ice_result.n_instances, len(ice_result.grid))
    logger.info("  PDP range: [%.3f, %.3f]", min(ice_result.pdp), max(ice_result.pdp))

    # Serialise to dict
    pdp_dict = pdp_result.to_dict()
    ice_dict = ice_result.to_dict()
    assert pdp_dict["type"] == "1way"
    assert pdp2_result.to_dict()["type"] == "2way"
    assert "ice_lines" in ice_dict
    assert "centered_ice" in ice_dict

    logger.info("✅ PDPAnalyzer smoke test PASSED.")
