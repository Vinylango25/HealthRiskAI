"""
lime_analyzer.py
================
LIME-based local model explainability for HealthRiskAI.

Supports:
- Classification and regression tasks via LimeTabularExplainer
- Per-instance explanations (dict or raw Explanation object)
- Batch explanations (sequential or parallel via joblib)
- Global pseudo-importance aggregated from local LIME weights
- Side-by-side instance comparison as a DataFrame
- Matplotlib bar-chart plotting with save support
- JSON report generation

Usage
-----
    analyzer = LIMEAnalyzer(feature_names=feature_names).fit(X_train)
    exp = analyzer.explain_as_dict(X_test[0], model.predict_proba)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# LIMEAnalyzer
# ---------------------------------------------------------------------------

class LIMEAnalyzer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) wrapper for
    tabular health-risk models.

    Parameters
    ----------
    feature_names : list[str]
        Names of input features in the order they appear in training arrays.
    class_names : list[str] | None
        Human-readable class labels (e.g. ['No Risk', 'High Risk']).
        Required for classification; ignored for regression.
    mode : str
        'classification' (default) or 'regression'.
    kernel_width : float | None
        Kernel width for the LIME exponential kernel.  None uses the LIME
        default heuristic (sqrt(n_features) * 0.75).
    random_state : int
        Seed for LIME's internal random sampler.
    n_samples : int
        Number of neighbourhood samples LIME generates per explanation.
    """

    def __init__(
        self,
        feature_names: list[str],
        class_names: list[str] | None = None,
        mode: str = "classification",
        kernel_width: float | None = None,
        random_state: int = 42,
        n_samples: int = 5000,
    ) -> None:
        if mode not in ("classification", "regression"):
            raise ValueError("mode must be 'classification' or 'regression'.")

        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.kernel_width = kernel_width
        self.random_state = random_state
        self.n_samples = n_samples

        # Populated by fit()
        self._explainer: object | None = None
        self._X_train: np.ndarray | None = None
        self._categorical_features: list[int] | None = None
        self._training_labels: np.ndarray | None = None

        logger.info(
            "LIMEAnalyzer created — mode=%s, %d features, n_samples=%d.",
            mode,
            len(feature_names),
            n_samples,
        )

    # ------------------------------------------------------------------
    # Initialisation / fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        categorical_features: list[int] | None = None,
        training_labels: np.ndarray | None = None,
    ) -> "LIMEAnalyzer":
        """
        Build a LimeTabularExplainer from training data.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
            Representative training data used as the background distribution.
        categorical_features : list[int] | None
            Column indices of categorical features.
        training_labels : np.ndarray | None
            Target labels; passed to LIME when available (only used for
            training data statistics).

        Returns
        -------
        self  (for method chaining)
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as exc:
            raise ImportError("Install lime: pip install lime==0.2.0.1") from exc

        self._X_train = X_train
        self._categorical_features = categorical_features
        self._training_labels = training_labels

        kwargs: dict = {
            "training_data": X_train,
            "feature_names": self.feature_names,
            "mode": self.mode,
            "random_state": self.random_state,
        }
        if self.class_names is not None:
            kwargs["class_names"] = self.class_names
        if categorical_features is not None:
            kwargs["categorical_features"] = categorical_features
        if training_labels is not None:
            kwargs["training_labels"] = training_labels
        if self.kernel_width is not None:
            kwargs["kernel_width"] = self.kernel_width

        self._explainer = LimeTabularExplainer(**kwargs)
        logger.info(
            "LimeTabularExplainer fitted on %d training samples.",
            len(X_train),
        )
        return self

    def _require_explainer(self) -> None:
        """Raise a clear error if fit() has not been called yet."""
        if self._explainer is None:
            raise RuntimeError(
                "LIMEAnalyzer has not been fitted. Call .fit(X_train) first."
            )

    # ------------------------------------------------------------------
    # Core explanation methods
    # ------------------------------------------------------------------

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        n_features: int = 10,
        labels: tuple[int, ...] = (1,),
    ) -> object:
        """
        Explain a single instance and return the raw LIME Explanation object.

        If the explainer has not been initialised yet (fit() not called),
        a RuntimeError is raised with a descriptive message.

        Parameters
        ----------
        instance : np.ndarray, shape (n_features,)
            The input sample to explain.
        predict_fn : Callable
            Model prediction function.  For classification it should return a
            probability matrix of shape (n, n_classes); for regression a
            1-D array of shape (n,).
        n_features : int
            Maximum number of features included in the explanation.
        labels : tuple[int, ...]
            Class indices to explain (classification only).

        Returns
        -------
        lime.explanation.Explanation
        """
        self._require_explainer()
        instance_1d = np.asarray(instance).ravel()

        kwargs: dict = {"num_features": n_features, "num_samples": self.n_samples}
        if self.mode == "classification":
            kwargs["labels"] = labels

        explanation = self._explainer.explain_instance(  # type: ignore[union-attr]
            instance_1d, predict_fn, **kwargs
        )
        return explanation

    def explain_as_dict(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
        n_features: int = 10,
        label: int = 1,
    ) -> dict[str, float]:
        """
        Return a ``{feature_name: weight}`` mapping for a single instance,
        sorted by absolute weight descending.

        Parameters
        ----------
        instance : np.ndarray, shape (n_features,)
        predict_fn : Callable
        n_features : int
        label : int
            Class index to extract weights for (classification only).

        Returns
        -------
        dict[str, float]
        """
        labels = (label,) if self.mode == "classification" else (0,)
        exp = self.explain_instance(instance, predict_fn, n_features=n_features, labels=labels)

        if self.mode == "classification":
            raw: list[tuple[str, float]] = exp.as_list(label=label)
        else:
            raw = exp.as_list()

        # Sort by |weight| descending
        sorted_pairs = sorted(raw, key=lambda x: abs(x[1]), reverse=True)
        return {name: weight for name, weight in sorted_pairs}

    # ------------------------------------------------------------------
    # Batch explanation
    # ------------------------------------------------------------------

    def explain_batch(
        self,
        instances: np.ndarray,
        predict_fn: Callable,
        n_features: int = 10,
        label: int = 1,
        n_jobs: int = 1,
    ) -> list[dict[str, float]]:
        """
        Explain multiple instances.

        Parameters
        ----------
        instances : np.ndarray, shape (n_instances, n_features)
        predict_fn : Callable
        n_features : int
        label : int
        n_jobs : int
            Number of parallel workers.  1 = sequential; -1 = all CPUs.
            Parallel mode uses joblib's loky backend.

        Returns
        -------
        list[dict[str, float]]
            One dict per instance, each sorted by absolute weight descending.
        """
        self._require_explainer()
        n = len(instances)
        logger.info("explain_batch: explaining %d instances (n_jobs=%d).", n, n_jobs)

        if n_jobs == 1:
            return [
                self.explain_as_dict(instances[i], predict_fn, n_features=n_features, label=label)
                for i in range(n)
            ]

        # Parallel path ----------------------------------------------------------------
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError("Install joblib: pip install joblib") from exc

        results: list[dict[str, float]] = Parallel(n_jobs=n_jobs)(
            delayed(self.explain_as_dict)(
                instances[i], predict_fn, n_features, label
            )
            for i in range(n)
        )
        return results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_explanation(
        self,
        explanation: object,
        label: int = 1,
        title: str = "LIME Explanation",
        save_path: Path | None = None,
    ) -> None:
        """
        Plot a horizontal bar chart of LIME feature weights.

        Positive weights are rendered in green, negative in red.

        Parameters
        ----------
        explanation : lime.explanation.Explanation
            Object returned by :meth:`explain_instance`.
        label : int
            Class index to visualise (classification only).
        title : str
            Chart title.
        save_path : Path | None
            If provided, the figure is saved to this path instead of
            (or in addition to) being shown interactively.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("Install matplotlib: pip install matplotlib") from exc

        if self.mode == "classification":
            pairs: list[tuple[str, float]] = explanation.as_list(label=label)  # type: ignore[union-attr]
        else:
            pairs = explanation.as_list()  # type: ignore[union-attr]

        # Sort by weight for readability (most negative at bottom, most positive at top)
        pairs = sorted(pairs, key=lambda x: x[1])

        feature_labels = [p[0] for p in pairs]
        weights = [p[1] for p in pairs]
        colours = ["#2ca02c" if w >= 0 else "#d62728" for w in weights]

        fig, ax = plt.subplots(figsize=(9, max(4, len(pairs) * 0.45)))
        bars = ax.barh(feature_labels, weights, color=colours, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("LIME weight", fontsize=11)
        ax.tick_params(axis="y", labelsize=9)

        # Annotate bar values
        for bar, w in zip(bars, weights):
            ax.text(
                w + (0.001 if w >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{w:+.4f}",
                va="center",
                ha="left" if w >= 0 else "right",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("LIME plot saved to %s.", save_path)
        else:
            plt.show()

        plt.close(fig)

    # ------------------------------------------------------------------
    # Comparison & aggregation
    # ------------------------------------------------------------------

    def compare_instances(
        self,
        instances: np.ndarray,
        predict_fn: Callable,
        instance_labels: list[str] | None = None,
        n_features: int = 10,
        label: int = 1,
    ) -> pd.DataFrame:
        """
        Build a comparison DataFrame with features as rows and instances as columns.

        Useful for analysing explanation differences across patient subgroups.

        Parameters
        ----------
        instances : np.ndarray, shape (n_instances, n_features)
        predict_fn : Callable
        instance_labels : list[str] | None
            Column names for the instances.  Defaults to 'Instance_0', etc.
        n_features : int
        label : int

        Returns
        -------
        pd.DataFrame
            Shape (n_features, n_instances).  Missing features are 0.0.
        """
        n = len(instances)
        if instance_labels is None:
            instance_labels = [f"Instance_{i}" for i in range(n)]

        explanations = self.explain_batch(
            instances, predict_fn, n_features=n_features, label=label
        )

        # Union of all feature names encountered
        all_features: list[str] = list(
            dict.fromkeys(f for exp in explanations for f in exp)
        )

        data = {
            col: [exp.get(feat, 0.0) for feat in all_features]
            for col, exp in zip(instance_labels, explanations)
        }
        df = pd.DataFrame(data, index=all_features)
        return df

    def global_feature_importance(
        self,
        instances: np.ndarray,
        predict_fn: Callable,
        n_features: int = 10,
        label: int = 1,
        aggregation: str = "mean_abs",
    ) -> pd.Series:
        """
        Aggregate local LIME weights across all instances to derive a
        pseudo-global feature importance ranking.

        Parameters
        ----------
        instances : np.ndarray, shape (n_instances, n_features)
        predict_fn : Callable
        n_features : int
        label : int
        aggregation : str
            One of:
            * ``'mean_abs'``   — mean of |weight| (default)
            * ``'mean'``       — signed mean
            * ``'median_abs'`` — median of |weight|

        Returns
        -------
        pd.Series
            Feature names as index, importance scores as values, sorted
            descending.
        """
        valid_aggs = ("mean_abs", "mean", "median_abs")
        if aggregation not in valid_aggs:
            raise ValueError(f"aggregation must be one of {valid_aggs}.")

        logger.info(
            "global_feature_importance: %d instances, aggregation=%s.",
            len(instances),
            aggregation,
        )

        explanations = self.explain_batch(
            instances, predict_fn, n_features=n_features, label=label
        )

        # Collect weights per feature across all instances
        all_features: list[str] = list(
            dict.fromkeys(f for exp in explanations for f in exp)
        )
        weight_matrix = np.array(
            [[exp.get(feat, 0.0) for feat in all_features] for exp in explanations]
        )  # shape: (n_instances, n_features)

        if aggregation == "mean_abs":
            scores = np.mean(np.abs(weight_matrix), axis=0)
        elif aggregation == "mean":
            scores = np.mean(weight_matrix, axis=0)
        else:  # median_abs
            scores = np.median(np.abs(weight_matrix), axis=0)

        series = pd.Series(scores, index=all_features, name=f"lime_{aggregation}")
        return series.sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def create_lime_report(
    analyzer: LIMEAnalyzer,
    instances: np.ndarray,
    predict_fn: Callable,
    instance_ids: list[str] | None = None,
    output_path: Path | None = None,
    n_features: int = 10,
) -> dict:
    """
    Generate a structured LIME report for a set of instances.

    Runs :meth:`LIMEAnalyzer.explain_batch` and wraps results into a
    serialisable dictionary.  Optionally persists the report as JSON.

    Parameters
    ----------
    analyzer : LIMEAnalyzer
        A fitted LIMEAnalyzer instance.
    instances : np.ndarray, shape (n_instances, n_features)
        Instances to explain (e.g., patient feature vectors).
    predict_fn : Callable
        Model prediction function.
    instance_ids : list[str] | None
        Identifiers for each instance (e.g. patient IDs).
        Defaults to 'instance_0', 'instance_1', …
    output_path : Path | None
        If provided, save the report as a JSON file at this path.
    n_features : int
        Maximum features per explanation.

    Returns
    -------
    dict
        ``{
            "meta": {feature_names, mode, n_features, n_instances},
            "instances": {
                "<id>": {
                    "feature_weights": {feature: weight, ...},
                    "top_feature": str,
                    "top_weight": float,
                }
            }
        }``
    """
    n = len(instances)
    if instance_ids is None:
        instance_ids = [f"instance_{i}" for i in range(n)]
    if len(instance_ids) != n:
        raise ValueError(
            f"instance_ids length ({len(instance_ids)}) != instances rows ({n})."
        )

    logger.info("create_lime_report: generating report for %d instances.", n)

    explanations = analyzer.explain_batch(
        instances, predict_fn, n_features=n_features
    )

    instances_section: dict = {}
    for id_, exp in zip(instance_ids, explanations):
        if exp:
            top_feat = max(exp, key=lambda k: abs(exp[k]))
            top_weight = exp[top_feat]
        else:
            top_feat, top_weight = "", 0.0

        instances_section[id_] = {
            "feature_weights": exp,
            "top_feature": top_feat,
            "top_weight": top_weight,
        }

    report: dict = {
        "meta": {
            "feature_names": analyzer.feature_names,
            "mode": analyzer.mode,
            "n_features": n_features,
            "n_instances": n,
        },
        "instances": instances_section,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=float)
        logger.info("LIME report saved to %s.", output_path)

    return report


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    logger.info("=== LIMEAnalyzer smoke test ===")

    # 1. Synthetic data --------------------------------------------------------
    N_SAMPLES, N_FEATURES = 300, 12
    feature_names = [f"feature_{i}" for i in range(N_FEATURES)]
    class_names = ["Low Risk", "High Risk"]

    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test = X[:250], X[250:]
    y_train = y[:250]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Train model -----------------------------------------------------------
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    logger.info("LogisticRegression trained on %d samples.", len(X_train_scaled))

    # 3. Fit LIMEAnalyzer ------------------------------------------------------
    analyzer = LIMEAnalyzer(
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=42,
        n_samples=1000,  # reduced for speed in smoke test
    ).fit(X_train_scaled)

    # 4. Explain 3 instances ---------------------------------------------------
    test_instances = X_test_scaled[:3]
    explanations = analyzer.explain_batch(test_instances, clf.predict_proba)

    for i, exp in enumerate(explanations):
        top = max(exp, key=lambda k: abs(exp[k]))
        logger.info(
            "Instance %d — top feature: %s (weight=%.4f), n_features=%d",
            i,
            top,
            exp[top],
            len(exp),
        )
        assert len(exp) > 0, f"Empty explanation for instance {i}!"

    # 5. Global importance -----------------------------------------------------
    global_imp = analyzer.global_feature_importance(
        test_instances, clf.predict_proba, aggregation="mean_abs"
    )
    logger.info("Global importance (top 5):\n%s", global_imp.head())
    assert isinstance(global_imp, pd.Series), "Expected pd.Series"
    assert len(global_imp) > 0, "Empty global importance!"

    # 6. compare_instances ----------------------------------------------------
    df_compare = analyzer.compare_instances(
        test_instances,
        clf.predict_proba,
        instance_labels=["Patient_A", "Patient_B", "Patient_C"],
    )
    logger.info("Comparison DataFrame shape: %s", df_compare.shape)
    assert df_compare.shape[1] == 3, f"Expected 3 columns, got {df_compare.shape[1]}"

    # 7. create_lime_report ---------------------------------------------------
    report = create_lime_report(
        analyzer,
        test_instances,
        clf.predict_proba,
        instance_ids=["P001", "P002", "P003"],
    )
    assert len(report["instances"]) == 3, "Report instance count mismatch!"
    logger.info("Report keys: %s", list(report["instances"].keys()))

    logger.info("✅ LIMEAnalyzer smoke test PASSED.")
