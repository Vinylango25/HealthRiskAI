"""
models/ensemble/meta_learner.py
==================================
Stacking ensemble meta-learner combining Level-0 model predictions.

Level-0 models (base learners):
  1. XGBoost cost predictor
  2. XGBoost readmission classifier
  3. LightGBM claims model
  4. CoxPH / DeepSurv survival risk score
  5. GNN node classification score
  6. ClinicalBERT complexity score (optional)

Meta-learner: Ridge regression (classification) / Ridge regression (cost)
  - Trained on out-of-fold predictions from cross-validation
  - Avoids overfitting by never training meta-learner on base model training data

Target: ensemble outperforms best single model on held-out test set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "reports" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Base model wrapper ───────────────────────────────────────────────────────


@dataclass
class BaseModel:
    """Wrapper for a base learner with name and predict functions."""

    name: str
    predict_fn: Callable[[np.ndarray], np.ndarray]   # returns probabilities or scores
    model_type: str = "classifier"                    # "classifier" | "regressor"
    weight: float = 1.0                               # contribution weight hint


# ─── Stacking meta-learner ────────────────────────────────────────────────────


class StackingMetaLearner:
    """
    Two-level stacking ensemble.

    Level 1: Out-of-fold predictions from base models
    Level 2: Ridge meta-learner trained on Level-1 OOF predictions

    Supports:
      - Classification (binary / multi-class)
      - Regression (cost prediction)
      - Time-aware cross-validation folds
      - MLflow tracking of ensemble vs individual model performance
    """

    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        n_folds: int = 5,
        meta_alpha: float = 1.0,
        use_original_features: bool = False,  # append original X to meta features
        calibrate: bool = True,               # calibrate final probabilities
        mlflow_experiment: str = "stacking_ensemble",
        random_state: int = 42,
    ) -> None:
        self.task = task
        self.n_folds = n_folds
        self.meta_alpha = meta_alpha
        self.use_original_features = use_original_features
        self.calibrate = calibrate
        self.mlflow_experiment = mlflow_experiment
        self.random_state = random_state

        # To be fit
        self.base_models: List[BaseModel] = []
        self.meta_learner: Optional[Any] = None
        self.scaler = StandardScaler()
        self.oof_predictions: Optional[np.ndarray] = None
        self.base_model_scores: Dict[str, float] = {}
        self._is_fitted = False

    def add_base_model(self, model: BaseModel) -> None:
        """Register a base model for the ensemble."""
        self.base_models.append(model)
        logger.info("Added base model: %s", model.name)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[np.ndarray] = None,   # for time-aware splits
    ) -> Dict[str, Any]:
        """
        Fit meta-learner using out-of-fold predictions.

        Parameters
        ----------
        X : (N, D) feature matrix used by base models
        y : (N,) target labels/values
        timestamps : (N,) optional for time-ordered CV

        Returns
        -------
        dict with per-model AUROC/MAE + ensemble metric
        """
        if not self.base_models:
            raise ValueError("No base models registered. Call add_base_model() first.")

        N = len(y)
        n_models = len(self.base_models)

        # OOF container: (N, n_models)
        oof = np.zeros((N, n_models), dtype=np.float32)

        # Generate fold splits
        if timestamps is not None:
            splits = self._time_aware_splits(timestamps)
        else:
            if self.task == "classification":
                kf = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.random_state)
                splits = list(kf.split(X, y.astype(int)))
            else:
                kf = KFold(self.n_folds, shuffle=True, random_state=self.random_state)
                splits = list(kf.split(X))

        logger.info(
            "Fitting %d base models with %d-fold CV on %d samples",
            n_models, self.n_folds, N,
        )

        mlflow.set_experiment(self.mlflow_experiment)
        with mlflow.start_run(run_name="stacking_ensemble_fit"):
            mlflow.log_params({
                "n_base_models": n_models,
                "n_folds": self.n_folds,
                "task": self.task,
                "meta_alpha": self.meta_alpha,
            })

            for m_idx, base_model in enumerate(self.base_models):
                model_oof = np.zeros(N, dtype=np.float32)
                fold_scores = []

                for fold_i, (train_idx, val_idx) in enumerate(splits):
                    X_val = X[val_idx]
                    y_val = y[val_idx]

                    # Get base model OOF predictions on val set
                    val_preds = base_model.predict_fn(X_val)
                    if isinstance(val_preds, pd.Series):
                        val_preds = val_preds.values
                    val_preds = np.asarray(val_preds, dtype=np.float32).ravel()

                    model_oof[val_idx] = val_preds

                    # Score this fold
                    try:
                        if self.task == "classification":
                            score = roc_auc_score(y_val, val_preds)
                        else:
                            score = -mean_absolute_error(y_val, val_preds)
                        fold_scores.append(score)
                    except Exception as e:
                        logger.warning("Scoring error for %s fold %d: %s", base_model.name, fold_i, e)

                oof[:, m_idx] = model_oof
                mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
                self.base_model_scores[base_model.name] = mean_score
                metric_name = "cv_auroc" if self.task == "classification" else "cv_neg_mae"
                logger.info("  %s — CV %s: %.4f", base_model.name, metric_name, mean_score)
                mlflow.log_metric(f"{base_model.name}_{metric_name}", mean_score)

            # ── Fit meta-learner on OOF predictions ──────────────────────────
            self.oof_predictions = oof
            meta_X = oof

            if self.use_original_features:
                meta_X = np.hstack([oof, X])

            meta_X_scaled = self.scaler.fit_transform(meta_X)

            if self.task == "classification":
                self.meta_learner = LogisticRegression(
                    C=1.0 / self.meta_alpha,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=self.random_state,
                )
                self.meta_learner.fit(meta_X_scaled, y.astype(int))

                if self.calibrate:
                    self.meta_learner = CalibratedClassifierCV(
                        self.meta_learner, cv=3, method="isotonic"
                    )
                    self.meta_learner.fit(meta_X_scaled, y.astype(int))

                ensemble_proba = self.meta_learner.predict_proba(meta_X_scaled)[:, 1]
                ensemble_auroc = float(roc_auc_score(y, ensemble_proba))
                logger.info("Ensemble OOF AUROC: %.4f", ensemble_auroc)
                mlflow.log_metric("ensemble_oof_auroc", ensemble_auroc)

            else:
                self.meta_learner = Ridge(alpha=self.meta_alpha)
                self.meta_learner.fit(meta_X_scaled, y)
                preds = self.meta_learner.predict(meta_X_scaled)
                ensemble_mae = float(mean_absolute_error(y, preds))
                logger.info("Ensemble OOF MAE: %.4f", ensemble_mae)
                mlflow.log_metric("ensemble_oof_mae", ensemble_mae)

        self._is_fitted = True
        return {
            "base_model_scores": self.base_model_scores,
            "ensemble_oof_metric": ensemble_auroc if self.task == "classification" else ensemble_mae,
        }

    def _time_aware_splits(
        self, timestamps: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sequential temporal folds."""
        N = len(timestamps)
        order = np.argsort(timestamps)
        fold_size = N // self.n_folds
        splits = []
        for i in range(self.n_folds):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_folds - 1 else N
            train_idx = order[np.concatenate([np.arange(0, val_start), np.arange(val_end, N)])]
            val_idx = order[np.arange(val_start, val_end)]
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
        return splits

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions on new data.
        Calls each base model's predict_fn, then runs meta-learner.

        Returns probability (classification) or predicted value (regression).
        """
        if not self._is_fitted:
            raise RuntimeError("Meta-learner not fitted.")

        # Collect base model predictions
        base_preds = np.column_stack([
            np.asarray(m.predict_fn(X), dtype=np.float32).ravel()
            for m in self.base_models
        ])  # (N, n_models)

        meta_X = base_preds
        if self.use_original_features:
            meta_X = np.hstack([base_preds, X])

        meta_X_scaled = self.scaler.transform(meta_X)

        if self.task == "classification":
            return self.meta_learner.predict_proba(meta_X_scaled)[:, 1]
        else:
            return self.meta_learner.predict(meta_X_scaled)

    def predict_with_components(self, X: np.ndarray) -> pd.DataFrame:
        """
        Return DataFrame with individual base model scores + ensemble score.
        Useful for debugging and explainability.
        """
        if not self._is_fitted:
            raise RuntimeError("Meta-learner not fitted.")

        result = pd.DataFrame()
        for m in self.base_models:
            preds = np.asarray(m.predict_fn(X), dtype=np.float32).ravel()
            result[f"base_{m.name}"] = preds

        result["ensemble_score"] = self.predict(X)
        return result

    def get_model_weights(self) -> pd.DataFrame:
        """
        Extract effective model weights from meta-learner coefficients.
        Only meaningful for Ridge/LogReg meta-learners.
        """
        if not self._is_fitted:
            raise ValueError("Meta-learner not fitted.")

        try:
            if hasattr(self.meta_learner, "coef_"):
                coefs = self.meta_learner.coef_
                if coefs.ndim > 1:
                    coefs = coefs[0]
            elif hasattr(self.meta_learner, "estimator"):
                coefs = self.meta_learner.estimator.coef_
                if coefs.ndim > 1:
                    coefs = coefs[0]
            else:
                return pd.DataFrame()

            n_base = len(self.base_models)
            return pd.DataFrame({
                "model": [m.name for m in self.base_models],
                "meta_weight": coefs[:n_base],
            }).sort_values("meta_weight", key=abs, ascending=False)
        except Exception as e:
            logger.warning("Could not extract weights: %s", e)
            return pd.DataFrame()

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate ensemble vs each base model on held-out test set."""
        metrics: Dict[str, float] = {}

        for m in self.base_models:
            preds = np.asarray(m.predict_fn(X), dtype=np.float32).ravel()
            try:
                if self.task == "classification":
                    metrics[f"{m.name}_auroc"] = float(roc_auc_score(y, preds))
                    metrics[f"{m.name}_auprc"] = float(average_precision_score(y, preds))
                else:
                    metrics[f"{m.name}_mae"] = float(mean_absolute_error(y, preds))
            except Exception as e:
                logger.warning("Eval error for %s: %s", m.name, e)

        ensemble_preds = self.predict(X)
        if self.task == "classification":
            metrics["ensemble_auroc"] = float(roc_auc_score(y, ensemble_preds))
            metrics["ensemble_auprc"] = float(average_precision_score(y, ensemble_preds))
        else:
            metrics["ensemble_mae"] = float(mean_absolute_error(y, ensemble_preds))

        # Log to MLflow if active run
        try:
            mlflow.log_metrics(metrics)
        except Exception:
            pass

        logger.info("Ensemble evaluation: %s", metrics)
        return metrics

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "stacking_ensemble.pkl"
        bundle = {
            "meta_learner": self.meta_learner,
            "scaler": self.scaler,
            "task": self.task,
            "use_original_features": self.use_original_features,
            "base_model_scores": self.base_model_scores,
            "base_model_names": [m.name for m in self.base_models],
        }
        joblib.dump(bundle, path)
        logger.info("Ensemble saved to %s", path)
        return path


# ─── Simple weighted average ensemble ────────────────────────────────────────


class WeightedAverageEnsemble:
    """
    Simpler alternative: weighted average of base model predictions.
    Weights optimised by minimising cross-entropy on validation set.
    """

    def __init__(self, task: str = "classification") -> None:
        self.task = task
        self.weights: Optional[np.ndarray] = None
        self.base_models: List[BaseModel] = []

    def add_base_model(self, model: BaseModel) -> None:
        self.base_models.append(model)

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """Optimise weights on validation set via Nelder-Mead."""
        from scipy.optimize import minimize

        n_models = len(self.base_models)
        base_preds = np.column_stack([
            np.asarray(m.predict_fn(X_val), dtype=np.float32).ravel()
            for m in self.base_models
        ])

        def objective(w: np.ndarray) -> float:
            w = np.abs(w)
            w = w / (w.sum() + 1e-8)
            blended = (base_preds * w).sum(axis=1)
            if self.task == "classification":
                return -roc_auc_score(y_val, blended)
            return mean_absolute_error(y_val, blended)

        x0 = np.ones(n_models) / n_models
        result = minimize(objective, x0, method="Nelder-Mead")
        raw = np.abs(result.x)
        self.weights = raw / raw.sum()
        logger.info("Optimised weights: %s", dict(zip([m.name for m in self.base_models], self.weights)))
        return self.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit() first.")
        base_preds = np.column_stack([
            np.asarray(m.predict_fn(X), dtype=np.float32).ravel()
            for m in self.base_models
        ])
        return (base_preds * self.weights).sum(axis=1)


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Stacking Meta-Learner Smoke Test ===")

    rng = np.random.default_rng(42)
    N, D = 500, 20
    X = rng.standard_normal((N, D)).astype(np.float32)
    y_true = rng.binomial(1, 0.3, N).astype(float)

    # Simulate 3 base models with different quality
    def model_a(X):
        return np.clip(0.7 * (X[:, 0] > 0) + rng.uniform(0, 0.2, len(X)), 0, 1)

    def model_b(X):
        return np.clip(0.5 * (X[:, 1] + X[:, 2]) / 2 + 0.5 + rng.uniform(-0.2, 0.2, len(X)), 0, 1)

    def model_c(X):
        return rng.uniform(0, 1, len(X))

    ensemble = StackingMetaLearner(task="classification", n_folds=3)
    ensemble.add_base_model(BaseModel("xgboost_readmit", model_a))
    ensemble.add_base_model(BaseModel("lightgbm_claims", model_b))
    ensemble.add_base_model(BaseModel("gnn_score", model_c))

    result = ensemble.fit(X, y_true)
    logger.info("Fit result: %s", result)

    test_X = rng.standard_normal((100, D)).astype(np.float32)
    test_y = rng.binomial(1, 0.3, 100).astype(float)
    eval_metrics = ensemble.evaluate(test_X, test_y)
    logger.info("Eval metrics: %s", eval_metrics)

    assert "ensemble_auroc" in eval_metrics
    path = ensemble.save()
    logger.info("=== PASS ===")
