"""
shap_analyzer.py
================
SHAP-based model explainability for HealthRiskAI.

Supports:
- TreeExplainer  (XGBoost / LightGBM / sklearn tree ensembles)
- DeepExplainer  (PyTorch neural nets)
- KernelExplainer (model-agnostic fallback)

Outputs structured dicts ready for Angular frontend rendering.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GlobalExplanation:
    """Container for global SHAP explanation data."""
    feature_names: List[str]
    mean_abs_shap: List[float]          # bar chart: mean |SHAP| per feature
    beeswarm: Dict[str, Any]            # per-feature value/shap pairs for beeswarm
    base_value: float
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean_abs_shap": self.mean_abs_shap,
            "beeswarm": self.beeswarm,
            "base_value": self.base_value,
            "n_samples": self.n_samples,
        }


@dataclass
class LocalExplanation:
    """Container for a single-prediction SHAP explanation."""
    feature_names: List[str]
    shap_values: List[float]            # signed SHAP values
    feature_values: List[float]         # original feature values
    base_value: float
    prediction: float
    waterfall: Dict[str, Any]           # structured waterfall data
    force_plot: Dict[str, Any]          # structured force-plot data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "shap_values": self.shap_values,
            "feature_values": self.feature_values,
            "base_value": self.base_value,
            "prediction": self.prediction,
            "waterfall": self.waterfall,
            "force_plot": self.force_plot,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_waterfall(
    feature_names: List[str],
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    base_value: float,
) -> Dict[str, Any]:
    """Build waterfall plot data sorted by |SHAP| descending."""
    order = np.argsort(np.abs(shap_values))[::-1]
    steps = []
    running = float(base_value)
    for idx in order:
        steps.append({
            "feature": feature_names[idx],
            "feature_value": float(feature_values[idx]),
            "shap_value": float(shap_values[idx]),
            "cumulative": running + float(shap_values[idx]),
        })
        running += float(shap_values[idx])
    return {"base_value": float(base_value), "steps": steps, "final_value": running}


def _build_force_plot(
    feature_names: List[str],
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    base_value: float,
    prediction: float,
) -> Dict[str, Any]:
    """Build force-plot data (positive vs negative contributions)."""
    positive = []
    negative = []
    for name, sv, fv in zip(feature_names, shap_values, feature_values):
        entry = {"feature": name, "feature_value": float(fv), "shap_value": float(sv)}
        if sv >= 0:
            positive.append(entry)
        else:
            negative.append(entry)
    positive.sort(key=lambda x: -x["shap_value"])
    negative.sort(key=lambda x: x["shap_value"])
    return {
        "base_value": float(base_value),
        "prediction": float(prediction),
        "positive_contributions": positive,
        "negative_contributions": negative,
    }


def _build_beeswarm(
    feature_names: List[str],
    shap_matrix: np.ndarray,
    feature_matrix: np.ndarray,
) -> Dict[str, Any]:
    """Build beeswarm data: per-feature list of (feature_value, shap_value) pairs."""
    beeswarm: Dict[str, Any] = {}
    for i, name in enumerate(feature_names):
        beeswarm[name] = {
            "feature_values": feature_matrix[:, i].tolist(),
            "shap_values": shap_matrix[:, i].tolist(),
            "mean_abs_shap": float(np.mean(np.abs(shap_matrix[:, i]))),
        }
    return beeswarm


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SHAPAnalyzer:
    """
    Unified SHAP explainability for HealthRiskAI models.

    Parameters
    ----------
    model : Any
        Trained model (XGBoost Booster, sklearn estimator, PyTorch Module, etc.)
    feature_names : list of str
        Names of input features.
    explainer_type : str
        One of 'tree', 'deep', 'kernel', or 'auto' (default).
    background_data : np.ndarray, optional
        Background dataset required by KernelExplainer / DeepExplainer.
    """

    SUPPORTED_TYPES = ("auto", "tree", "deep", "kernel")

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        explainer_type: str = "auto",
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        if explainer_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"explainer_type must be one of {self.SUPPORTED_TYPES}")

        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self._explainer: Any = None

        self._explainer = self._init_explainer(explainer_type)
        logger.info("SHAPAnalyzer initialised with %s explainer (%d features).",
                    explainer_type, len(feature_names))

    # ------------------------------------------------------------------
    # Explainer initialisation
    # ------------------------------------------------------------------

    def _init_explainer(self, explainer_type: str) -> Any:
        try:
            import shap  # lazy import — shap is optional at module level
        except ImportError as exc:
            raise ImportError("Install shap: pip install shap") from exc

        etype = explainer_type
        if etype == "auto":
            etype = self._detect_explainer_type()

        logger.info("Creating %s explainer.", etype)
        if etype == "tree":
            return shap.TreeExplainer(self.model)
        if etype == "deep":
            if self.background_data is None:
                raise ValueError("DeepExplainer requires background_data.")
            import torch
            bg = torch.tensor(self.background_data, dtype=torch.float32)
            return shap.DeepExplainer(self.model, bg)
        # kernel fallback
        if self.background_data is None:
            raise ValueError("KernelExplainer requires background_data.")
        bg_summary = shap.kmeans(self.background_data, min(50, len(self.background_data)))
        return shap.KernelExplainer(self.model.predict_proba
                                    if hasattr(self.model, "predict_proba")
                                    else self.model.predict,
                                    bg_summary)

    def _detect_explainer_type(self) -> str:
        model_type = type(self.model).__name__.lower()
        tree_keywords = ("xgb", "lgbm", "lightgbm", "gradientboosting",
                         "randomforest", "decisiontree", "extratrees")
        if any(k in model_type for k in tree_keywords):
            return "tree"
        try:
            import torch.nn as nn
            if isinstance(self.model, nn.Module):
                return "deep"
        except ImportError:
            pass
        return "kernel"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_global(self, X: np.ndarray) -> GlobalExplanation:
        """
        Compute global SHAP explanation over a dataset.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        GlobalExplanation
        """
        logger.info("Computing global SHAP explanation for %d samples.", len(X))
        shap_values = self._compute_shap_values(X)

        # For multi-class, take class-1 (positive risk) slice
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        mean_abs = np.mean(np.abs(shap_values), axis=0).tolist()
        base_value = float(np.mean(self._get_base_value()))
        beeswarm = _build_beeswarm(self.feature_names, shap_values, X)

        return GlobalExplanation(
            feature_names=self.feature_names,
            mean_abs_shap=mean_abs,
            beeswarm=beeswarm,
            base_value=base_value,
            n_samples=len(X),
        )

    def explain_local(self, x: np.ndarray) -> LocalExplanation:
        """
        Compute SHAP explanation for a single prediction.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,) or (1, n_features)

        Returns
        -------
        LocalExplanation
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        logger.info("Computing local SHAP explanation for single instance.")
        shap_values = self._compute_shap_values(x)

        if shap_values.ndim == 3:
            shap_values = shap_values[0, :, 1]
        else:
            shap_values = shap_values[0]

        feature_values = x[0]
        base_value = float(np.mean(self._get_base_value()))

        # Prediction = base_value + sum(shap_values)
        prediction = base_value + float(np.sum(shap_values))

        waterfall = _build_waterfall(self.feature_names, shap_values, feature_values, base_value)
        force_plot = _build_force_plot(self.feature_names, shap_values, feature_values,
                                       base_value, prediction)

        return LocalExplanation(
            feature_names=self.feature_names,
            shap_values=shap_values.tolist(),
            feature_values=feature_values.tolist(),
            base_value=base_value,
            prediction=prediction,
            waterfall=waterfall,
            force_plot=force_plot,
        )

    def get_top_features(self, X: np.ndarray, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Return the top-N most important features ranked by mean |SHAP|.

        Returns
        -------
        list of (feature_name, mean_abs_shap) tuples, descending order.
        """
        explanation = self.explain_global(X)
        pairs = sorted(
            zip(explanation.feature_names, explanation.mean_abs_shap),
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs[:top_n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._explainer.shap_values(X)
        return np.array(raw)

    def _get_base_value(self) -> Union[float, np.ndarray]:
        bv = getattr(self._explainer, "expected_value", 0.0)
        return np.atleast_1d(bv)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    logger.info("=== SHAPAnalyzer smoke test ===")

    try:
        import xgboost as xgb
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
        params = {"objective": "binary:logistic", "max_depth": 3,
                  "n_estimators": 50, "eval_metric": "logloss"}
        model = xgb.train(params, dtrain, num_boost_round=50, verbose_eval=False)

        analyzer = SHAPAnalyzer(model, feature_names=feature_names, explainer_type="tree")

        # Global
        global_exp = analyzer.explain_global(X)
        logger.info("Global explanation — top feature: %s (mean |SHAP|=%.4f)",
                    global_exp.feature_names[np.argmax(global_exp.mean_abs_shap)],
                    max(global_exp.mean_abs_shap))

        # Local
        local_exp = analyzer.explain_local(X[0])
        logger.info("Local explanation — prediction=%.4f, base=%.4f",
                    local_exp.prediction, local_exp.base_value)

        # Top features
        top = analyzer.get_top_features(X, top_n=5)
        logger.info("Top 5 features: %s", [(n, round(v, 4)) for n, v in top])

        logger.info("✅ SHAPAnalyzer smoke test PASSED.")

    except ImportError as e:
        logger.warning("Skipping XGBoost smoke test (missing dependency: %s).", e)
        logger.info("Install with: pip install xgboost shap scikit-learn")
