"""
models/ensemble/evaluator.py
==============================
Comprehensive model evaluation across all HealthRiskAI models.

Produces:
  - Per-model metric tables (AUROC, AUPRC, F1, MAPE, C-index)
  - Ensemble vs single-model comparison
  - Calibration analysis
  - Threshold analysis (precision-recall trade-off)
  - Model comparison report (CSV + HTML)
  - MLflow logging of all metrics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Single model evaluation ─────────────────────────────────────────────────


@dataclass
class ModelEvaluation:
    """Holds all evaluation results for a single model."""

    model_name: str
    task: str  # "binary_classification" | "multiclass" | "regression"
    metrics: Dict[str, float]
    curves: Dict[str, Any]  # roc_curve, pr_curve data
    confusion: Optional[np.ndarray] = None
    threshold_analysis: Optional[pd.DataFrame] = None
    calibration: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "task": self.task,
            **self.metrics,
        }


# ─── Evaluator class ─────────────────────────────────────────────────────────


class ModelEvaluator:
    """
    Evaluates one or more models and produces comparison reports.

    Usage
    -----
    ev = ModelEvaluator()
    ev.evaluate_binary("XGBoost", y_true, y_pred_proba)
    ev.evaluate_regression("CostPredictor", y_true, y_pred)
    ev.compare_all()
    ev.save_report()
    """

    def __init__(
        self,
        mlflow_experiment: str = "model_evaluation",
        report_dir: Optional[Path] = None,
    ) -> None:
        self.mlflow_experiment = mlflow_experiment
        self.report_dir = report_dir or REPORTS_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._evaluations: List[ModelEvaluation] = []

    # ── Binary classification ─────────────────────────────────────────────────

    def evaluate_binary(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        positive_label: int = 1,
    ) -> ModelEvaluation:
        """
        Full binary classification evaluation.
        Returns ModelEvaluation with all metrics + curves.
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        metrics: Dict[str, float] = {}

        # Core metrics
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_pred_proba))
        except ValueError:
            metrics["auroc"] = float("nan")

        try:
            metrics["auprc"] = float(average_precision_score(y_true, y_pred_proba))
        except ValueError:
            metrics["auprc"] = float("nan")

        metrics["accuracy"] = float(accuracy_score(y_true, y_pred_binary))
        metrics["f1"] = float(f1_score(y_true, y_pred_binary, zero_division=0))
        metrics["precision"] = float(precision_score(y_true, y_pred_binary, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred_binary, zero_division=0))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_pred_proba))
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
        except ValueError:
            metrics["log_loss"] = float("nan")

        # Prevalence
        metrics["prevalence"] = float(y_true.mean())
        metrics["threshold"] = threshold

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics["tn"] = float(tn)
        metrics["fp"] = float(fp)
        metrics["fn"] = float(fn)
        metrics["tp"] = float(tp)
        metrics["specificity"] = float(tn / max(tn + fp, 1))

        # Curves
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)

        curves = {
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()},
            "pr": {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()},
        }

        # Calibration
        cal = self._compute_calibration(y_true, y_pred_proba)

        # Threshold analysis
        threshold_df = self._threshold_analysis(y_true, y_pred_proba)

        eval_result = ModelEvaluation(
            model_name=model_name,
            task="binary_classification",
            metrics=metrics,
            curves=curves,
            confusion=cm,
            threshold_analysis=threshold_df,
            calibration=cal,
        )
        self._evaluations.append(eval_result)

        logger.info(
            "%s — AUROC: %.4f | AUPRC: %.4f | F1: %.4f | Brier: %.4f",
            model_name,
            metrics.get("auroc", float("nan")),
            metrics.get("auprc", float("nan")),
            metrics.get("f1", float("nan")),
            metrics.get("brier_score", float("nan")),
        )
        return eval_result

    # ── Regression evaluation ─────────────────────────────────────────────────

    def evaluate_regression(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ModelEvaluation:
        """Full regression evaluation."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        metrics: Dict[str, float] = {}
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

        try:
            metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
        except Exception:
            metrics["mape"] = float("nan")

        metrics["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        metrics["median_ae"] = float(np.median(np.abs(y_true - y_pred)))

        # Bias
        metrics["mean_bias"] = float((y_pred - y_true).mean())
        metrics["predictive_ratio"] = float(y_pred.sum() / max(y_true.sum(), 1e-8))

        logger.info(
            "%s — MAE: %.4f | MAPE: %.4f | R²: %.4f | Pred. ratio: %.4f",
            model_name,
            metrics["mae"],
            metrics.get("mape", float("nan")),
            metrics["r2"],
            metrics["predictive_ratio"],
        )

        eval_result = ModelEvaluation(
            model_name=model_name,
            task="regression",
            metrics=metrics,
            curves={},
        )
        self._evaluations.append(eval_result)
        return eval_result

    # ── Calibration ───────────────────────────────────────────────────────────

    def _compute_calibration(
        self, y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        """Compute calibration curve and ECE."""
        try:
            frac_pos, mean_pred = calibration_curve(
                y_true, y_pred, n_bins=n_bins, strategy="quantile"
            )
            ece = float(np.mean(np.abs(frac_pos - mean_pred)))
            return {
                "frac_positives": frac_pos.tolist(),
                "mean_predicted": mean_pred.tolist(),
                "ece": ece,
                "well_calibrated": ece < 0.05,
            }
        except Exception as e:
            logger.warning("Calibration failed: %s", e)
            return {}

    # ── Threshold analysis ────────────────────────────────────────────────────

    def _threshold_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Compute precision/recall/F1/specificity at multiple thresholds."""
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)

        rows = []
        for t in thresholds:
            y_pred = (y_pred_proba >= t).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])
            rows.append({
                "threshold": t,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "flag_rate": y_pred.mean(),
            })
        return pd.DataFrame(rows)

    # ── Multi-model comparison ────────────────────────────────────────────────

    def compare_all(self, task: str = "binary_classification") -> pd.DataFrame:
        """Build side-by-side comparison table for all evaluated models."""
        rows = [e.to_dict() for e in self._evaluations if e.task == task]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("model")

        # Sort by primary metric
        primary = "auroc" if task == "binary_classification" else "mae"
        ascending = task == "regression"
        if primary in df.columns:
            df = df.sort_values(primary, ascending=ascending)

        # Add target met flags
        targets = {
            "binary_classification": {
                "auroc": 0.78,  # project target
                "f1": 0.60,
                "brier_score": 0.20,
            },
            "regression": {
                "r2": 0.25,
                "mape": 0.52,
            },
        }
        for metric, threshold in targets.get(task, {}).items():
            if metric in df.columns:
                if metric in ("brier_score", "mape", "mae"):
                    df[f"{metric}_target_met"] = df[metric] <= threshold
                else:
                    df[f"{metric}_target_met"] = df[metric] >= threshold

        logger.info("\n=== Model Comparison (%s) ===\n%s", task, df.to_string())
        return df

    # ── Report generation ─────────────────────────────────────────────────────

    def save_report(
        self,
        name: str = "model_comparison",
        log_to_mlflow: bool = True,
    ) -> Path:
        """Save CSV + JSON report of all evaluations."""
        clf_df = self.compare_all("binary_classification")
        reg_df = self.compare_all("regression")

        output = {}

        if not clf_df.empty:
            clf_path = self.report_dir / f"{name}_classification.csv"
            clf_df.to_csv(clf_path)
            output["classification_report"] = str(clf_path)
            logger.info("Classification report saved: %s", clf_path)

        if not reg_df.empty:
            reg_path = self.report_dir / f"{name}_regression.csv"
            reg_df.to_csv(reg_path)
            output["regression_report"] = str(reg_path)
            logger.info("Regression report saved: %s", reg_path)

        # Full JSON dump
        all_evals = [e.to_dict() for e in self._evaluations]
        json_path = self.report_dir / f"{name}_full.json"
        with open(json_path, "w") as f:
            json.dump(all_evals, f, indent=2, default=str)

        if log_to_mlflow:
            mlflow.set_experiment(self.mlflow_experiment)
            try:
                with mlflow.start_run(run_name=f"eval_{name}"):
                    for e in self._evaluations:
                        for metric, value in e.metrics.items():
                            try:
                                mlflow.log_metric(f"{e.model_name}_{metric}", float(value))
                            except Exception:
                                pass
                    mlflow.log_artifact(str(json_path))
            except Exception as ex:
                logger.warning("MLflow logging failed: %s", ex)

        return json_path

    def print_summary(self) -> None:
        """Print formatted summary of all evaluations."""
        for ev in self._evaluations:
            print(f"\n{'─' * 50}")
            print(f"Model: {ev.model_name} | Task: {ev.task}")
            print(f"{'─' * 50}")
            for k, v in ev.metrics.items():
                if isinstance(v, float):
                    print(f"  {k:30s}: {v:.4f}")
                else:
                    print(f"  {k:30s}: {v}")


# ─── Convenience functions ────────────────────────────────────────────────────


def evaluate_all_models(
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    report_name: str = "full_model_comparison",
) -> pd.DataFrame:
    """
    Evaluate all models at once from a predictions dict.

    model_predictions format:
    {
        "XGBoost_readmission": {
            "y_true": array, "y_pred": array, "task": "binary_classification"
        },
        "CostPredictor": {
            "y_true": array, "y_pred": array, "task": "regression"
        },
        ...
    }
    """
    ev = ModelEvaluator()

    for model_name, pred_data in model_predictions.items():
        task = pred_data.get("task", "binary_classification")
        y_true = pred_data["y_true"]
        y_pred = pred_data["y_pred"]

        if task == "binary_classification":
            ev.evaluate_binary(model_name, y_true, y_pred)
        elif task == "regression":
            ev.evaluate_regression(model_name, y_true, y_pred)

    ev.save_report(report_name)
    return ev.compare_all("binary_classification")


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Ensemble Evaluator Smoke Test ===")

    rng = np.random.default_rng(42)
    N = 500
    y_true = rng.binomial(1, 0.25, N)

    # Simulate different model qualities
    good_model = np.clip(0.7 * y_true + rng.normal(0, 0.2, N), 0, 1)
    ok_model = np.clip(0.5 * y_true + rng.normal(0, 0.3, N), 0, 1)
    random_model = rng.uniform(0, 1, N)

    ev = ModelEvaluator()
    ev.evaluate_binary("XGBoost_Readmission", y_true, good_model)
    ev.evaluate_binary("LightGBM_Claims", y_true, ok_model)
    ev.evaluate_binary("RandomBaseline", y_true, random_model)

    # Regression evaluation
    y_cost = rng.lognormal(8, 1.5, N)
    cost_pred = y_cost * rng.lognormal(0, 0.2, N)
    ev.evaluate_regression("CostPredictor", y_cost, cost_pred)

    ev.print_summary()
    comparison = ev.compare_all("binary_classification")

    assert "auroc" in comparison.columns
    assert comparison.loc["XGBoost_Readmission", "auroc"] > comparison.loc["RandomBaseline", "auroc"]
    logger.info("\nComparison table:\n%s", comparison[["auroc", "auprc", "f1"]].to_string())

    path = ev.save_report("smoke_test", log_to_mlflow=False)
    logger.info("Report saved to %s", path)
    logger.info("=== PASS ===")
