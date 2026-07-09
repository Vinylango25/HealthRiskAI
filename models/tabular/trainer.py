"""
models/tabular/trainer.py
=========================
Shared training utilities for all tabular models in the HealthRisk AI platform.

Provides time-aware cross-validation, MLflow experiment tracking, and a suite
of evaluation metrics for both classification and regression tasks.

Author: HealthRisk AI Team
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "model_config.yaml"


def _load_config() -> dict:
    """Load model configuration from YAML file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    logger.warning(f"Config file not found at {CONFIG_PATH}. Using defaults.")
    return {}


# ---------------------------------------------------------------------------
# TimeAwareCrossValidator
# ---------------------------------------------------------------------------


class TimeAwareCrossValidator:
    """
    Chronological cross-validator that respects temporal ordering.

    Unlike standard k-fold, each fold uses strictly earlier data for training
    and later data for testing — preventing data leakage from the future.

    Parameters
    ----------
    n_splits : int
        Number of folds. Default is 5.
    time_col : str
        Name of the datetime/date column used for sorting. Default is 'date'.

    Example
    -------
    >>> cv = TimeAwareCrossValidator(n_splits=5, time_col='admission_date')
    >>> for train_idx, test_idx in cv.split(df):
    ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """

    def __init__(self, n_splits: int = 5, time_col: str = "date") -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.time_col = time_col

    def split(
        self, df: pd.DataFrame
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-aware train/test index splits.

        Sorts the DataFrame by ``time_col`` and partitions it into
        ``n_splits + 1`` consecutive chunks. Fold k uses chunks [0..k] as
        training data and chunk [k+1] as the test set, guaranteeing no
        temporal leakage.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing ``time_col``.

        Returns
        -------
        list of (train_idx, test_idx) tuples
            Each element is a pair of integer arrays (positional indices into
            the sorted DataFrame).

        Raises
        ------
        KeyError
            If ``time_col`` is not found in ``df``.
        ValueError
            If ``df`` is too small to produce the requested number of splits.
        """
        if self.time_col not in df.columns:
            raise KeyError(
                f"time_col='{self.time_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        n_samples = len(df)
        min_required = self.n_splits + 1
        if n_samples < min_required:
            raise ValueError(
                f"DataFrame has {n_samples} rows but at least {min_required} "
                f"are required for {self.n_splits} splits."
            )

        # Sort chronologically; keep track of original positions
        sorted_df = df.sort_values(self.time_col, kind="mergesort")
        sorted_positions = sorted_df.index  # original index labels (unused below)

        # Work with positional indices (0-based) into the *sorted* array
        sorted_positional = np.arange(n_samples)

        # Split the sorted array into (n_splits + 1) roughly equal chunks
        chunks = np.array_split(sorted_positional, self.n_splits + 1)

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(self.n_splits):
            train_chunks = chunks[: fold + 1]
            test_chunk = chunks[fold + 1]
            train_idx = np.concatenate(train_chunks)
            test_idx = test_chunk
            splits.append((train_idx, test_idx))
            logger.debug(
                f"Fold {fold + 1}/{self.n_splits}: "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )

        return splits


# ---------------------------------------------------------------------------
# MLflowTracker
# ---------------------------------------------------------------------------


class MLflowTracker:
    """
    Wrapper around MLflow for structured experiment tracking.

    Handles experiment creation, run lifecycle, parameter/metric logging,
    model artefacts, and feature-importance plots.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment. Created if it does not exist.
    tracking_uri : str, optional
        MLflow tracking server URI. Falls back to ``MLFLOW_TRACKING_URI``
        environment variable or local ``./mlruns`` directory.

    Example
    -------
    >>> tracker = MLflowTracker("healthrisk/cost_prediction")
    >>> with tracker.start_run("fold_1"):
    ...     tracker.log_params({"lr": 0.01, "max_depth": 6})
    ...     tracker.log_metrics({"rmse": 1234.5})
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name

        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
            logger.info(f"MLflow tracking URI set to: {uri}")
        else:
            logger.info("MLflow tracking URI not set; using local ./mlruns")

        mlflow.set_experiment(experiment_name)
        self._active_run: Optional[mlflow.ActiveRun] = None
        logger.info(f"MLflowTracker initialised for experiment: {experiment_name}")

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_name: str) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Parameters
        ----------
        run_name : str
            Human-readable name for this run.

        Returns
        -------
        mlflow.ActiveRun
            The active MLflow run context.
        """
        if self._active_run is not None:
            logger.warning(
                "A run is already active. Ending it before starting a new one."
            )
            self.end_run()

        self._active_run = mlflow.start_run(run_name=run_name)
        logger.info(
            f"Started MLflow run: {run_name} "
            f"(id={self._active_run.info.run_id[:8]})"
        )
        return self._active_run

    def end_run(self) -> None:
        """End the currently active MLflow run."""
        if self._active_run is not None:
            mlflow.end_run()
            logger.info(
                f"Ended MLflow run: {self._active_run.info.run_id[:8]}"
            )
            self._active_run = None
        else:
            logger.warning("end_run called but no active run exists.")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict) -> None:
        """
        Log a dictionary of hyperparameters to MLflow.

        Parameters
        ----------
        params : dict
            Key-value pairs of parameter names and values.
        """
        # MLflow requires string values; cast safely
        safe_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(safe_params)
        logger.debug(f"Logged {len(safe_params)} params to MLflow.")

    def log_metrics(
        self, metrics: dict, step: Optional[int] = None
    ) -> None:
        """
        Log a dictionary of metrics to MLflow.

        Parameters
        ----------
        metrics : dict
            Key-value pairs of metric names and numeric values.
        step : int, optional
            Training step / fold number for the metrics.
        """
        # Filter out non-finite values to avoid MLflow errors
        clean_metrics = {}
        for k, v in metrics.items():
            try:
                fv = float(v)
                if np.isfinite(fv):
                    clean_metrics[k] = fv
                else:
                    logger.warning(
                        f"Metric '{k}' = {v} is non-finite; skipping."
                    )
            except (TypeError, ValueError):
                logger.warning(f"Metric '{k}' = {v} cannot be cast to float; skipping.")

        mlflow.log_metrics(clean_metrics, step=step)
        logger.debug(
            f"Logged {len(clean_metrics)} metrics to MLflow"
            + (f" at step={step}" if step is not None else "")
        )

    def log_model(self, model: Any, model_name: str) -> None:
        """
        Log a trained model artefact to MLflow.

        Attempts XGBoost-native logging first, falls back to sklearn flavour.

        Parameters
        ----------
        model : Any
            Trained model object.
        model_name : str
            Artefact path under the run.
        """
        try:
            import xgboost as xgb

            if isinstance(model, xgb.XGBModel):
                mlflow.xgboost.log_model(model, model_name)
                logger.info(f"Logged XGBoost model as '{model_name}'.")
                return
        except ImportError:
            pass

        mlflow.sklearn.log_model(model, model_name)
        logger.info(f"Logged sklearn model as '{model_name}'.")

    def log_feature_importance(
        self, feature_names: list[str], importances: np.ndarray
    ) -> None:
        """
        Log feature importances as a CSV artefact and as individual metrics.

        Parameters
        ----------
        feature_names : list of str
            Ordered list of feature names.
        importances : np.ndarray
            Importance scores aligned with ``feature_names``.
        """
        if len(feature_names) != len(importances):
            raise ValueError(
                f"Length mismatch: {len(feature_names)} names vs "
                f"{len(importances)} importances."
            )

        fi_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "feature_importance.csv"
            fi_df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), "feature_importance")

        # Log top-20 as individual metrics for quick dashboarding
        for _, row in fi_df.head(20).iterrows():
            safe_name = str(row["feature"]).replace(" ", "_")[:250]
            mlflow.log_metric(f"fi_{safe_name}", float(row["importance"]))

        logger.info(
            f"Logged feature importances for {len(feature_names)} features."
        )


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_binary: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Compute a comprehensive set of binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.
    y_pred_binary : np.ndarray, optional
        Hard predictions. Derived from ``y_pred_proba >= threshold`` if None.
    threshold : float
        Decision threshold for deriving hard predictions. Default 0.5.

    Returns
    -------
    dict with keys:
        auroc, auprc, f1, precision, recall, brier_score, ks_statistic
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

    auroc = float(roc_auc_score(y_true, y_pred_proba))
    auprc = float(average_precision_score(y_true, y_pred_proba))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
    precision = float(precision_score(y_true, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_true, y_pred_binary, zero_division=0))
    brier = float(brier_score_loss(y_true, y_pred_proba))

    # KS statistic = max(TPR - FPR) from the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ks_statistic = float(np.max(tpr - fpr))

    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "brier_score": brier,
        "ks_statistic": ks_statistic,
    }

    logger.debug(
        f"Classifier metrics — AUROC={auroc:.4f}, AUPRC={auprc:.4f}, "
        f"KS={ks_statistic:.4f}, F1={f1:.4f}"
    )
    return metrics


def evaluate_regressor(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """
    Compute a comprehensive set of regression metrics.

    MAPE infinite values (caused by zero actuals) are clipped before averaging.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth continuous values.
    y_pred : np.ndarray
        Model predictions.

    Returns
    -------
    dict with keys:
        rmse, mae, mape, r2, median_ae, predictive_ratio
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # MAPE: clip infinity from zero-denominator entries
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_pct_err = np.abs((y_true - y_pred) / y_true)
    abs_pct_err = np.clip(abs_pct_err, 0.0, 1e6)  # clip +inf
    mape = float(np.nanmean(abs_pct_err))

    r2 = float(r2_score(y_true, y_pred))
    median_ae = float(median_absolute_error(y_true, y_pred))

    # Predictive ratio: mean predicted / mean actual (should be ≈ 1.0)
    mean_actual = float(np.mean(y_true))
    predictive_ratio = float(np.mean(y_pred) / mean_actual) if mean_actual != 0 else float("nan")

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "median_ae": median_ae,
        "predictive_ratio": predictive_ratio,
    }

    logger.debug(
        f"Regressor metrics — RMSE={rmse:.2f}, MAE={mae:.2f}, "
        f"MAPE={mape:.4f}, R²={r2:.4f}, PR={predictive_ratio:.4f}"
    )
    return metrics


def compute_gini(auroc: float) -> float:
    """
    Compute the Gini coefficient from AUROC.

    Gini = 2 * AUROC - 1. Ranges from -1 (perfect inversion) to 1 (perfect
    discrimination), with 0 being a random classifier.

    Parameters
    ----------
    auroc : float
        Area Under the ROC Curve.

    Returns
    -------
    float
        Gini coefficient.
    """
    return 2.0 * auroc - 1.0


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.

    PSI measures how much a score distribution has shifted between training
    (expected) and deployment (actual). Interpretation:
      - PSI < 0.10 : No significant shift
      - 0.10–0.25  : Moderate shift; monitor
      - PSI > 0.25 : Significant shift; re-train

    Parameters
    ----------
    expected : np.ndarray
        Reference (training) distribution.
    actual : np.ndarray
        Production distribution to compare against.
    buckets : int
        Number of quantile bins. Default is 10.

    Returns
    -------
    float
        PSI value.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Define bin edges from the *expected* distribution
    breakpoints = np.nanpercentile(
        expected, np.linspace(0, 100, buckets + 1)
    )
    # Remove duplicate edges (e.g. from highly discrete distributions)
    breakpoints = np.unique(breakpoints)

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Convert to proportions with Laplace smoothing to avoid log(0)
    eps = 1e-6
    expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
    actual_pct = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

    psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = float(np.sum(psi_components))

    logger.debug(
        f"PSI = {psi:.4f} (expected_n={len(expected)}, actual_n={len(actual)}, "
        f"buckets={len(expected_counts)})"
    )
    return psi
