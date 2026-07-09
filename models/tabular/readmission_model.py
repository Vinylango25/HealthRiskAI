"""
models/tabular/readmission_model.py
=====================================
XGBoost 30-day hospital readmission classifier.

Predicts the probability that a patient will be readmitted to hospital within
30 days of discharge. Handles class imbalance via ``scale_pos_weight``,
supports Platt-scaling calibration, and provides cost-sensitive threshold
optimisation.

Author: HealthRisk AI Team
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import xgboost as xgb

from models.tabular.trainer import (
    MLflowTracker,
    TimeAwareCrossValidator,
    compute_gini,
    evaluate_classifier,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "model_config.yaml"

#: Feature columns grouped by semantic category.
FEATURE_COLS: list[str] = [
    # ── Demographics ────────────────────────────────────────────────────────
    "age",
    "gender_male",
    "race_white",
    "race_black",
    "race_hispanic",
    "race_asian",
    "race_other",
    "dual_eligible",
    # ── Diagnoses (ICD flags / counts) ──────────────────────────────────────
    "n_diagnoses_primary",
    "n_diagnoses_secondary",
    "dx_chf",
    "dx_copd",
    "dx_ckd",
    "dx_diabetes",
    "dx_pneumonia",
    "dx_sepsis",
    "dx_ami",
    "dx_stroke",
    "dx_depression",
    "dx_substance_abuse",
    # ── Labs at discharge ────────────────────────────────────────────────────
    "sodium_discharge",
    "bun_discharge",
    "creatinine_discharge",
    "wbc_discharge",
    "hemoglobin_discharge",
    "albumin_discharge",
    "bnp_discharge",
    "glucose_discharge",
    # ── Medications ──────────────────────────────────────────────────────────
    "n_discharge_meds",
    "polypharmacy_flag",
    "high_risk_med_flag",       # anticoagulants, insulin, opioids, etc.
    "med_reconciliation_done",
    # ── Prior utilisation ────────────────────────────────────────────────────
    "ed_visits_6m",
    "ed_visits_12m",
    "ip_admissions_12m",
    "ip_admissions_24m",
    "prior_readmission_flag",
    "snf_days_12m",
    # ── Comorbidity scores ────────────────────────────────────────────────────
    "charlson_score",
    "hcc_risk_score",
    "elixhauser_score",
    # ── Current hospitalisation ───────────────────────────────────────────────
    "los_days",
    "icu_flag",
    "surgical_flag",
    "discharge_disposition",    # home, SNF, rehab, etc. (ordinal-encoded)
    "weekend_discharge_flag",
    # ── Social / functional ───────────────────────────────────────────────────
    "lives_alone_flag",
    "adl_score",
    "insurance_type",           # Medicare, Medicaid, commercial, etc.
]

TARGET_COL = "readmitted_30d"


def _load_config() -> dict:
    """Load model configuration YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    logger.warning(f"Config not found at {CONFIG_PATH}. Using defaults.")
    return {}


# ---------------------------------------------------------------------------
# ReadmissionPredictor
# ---------------------------------------------------------------------------


class ReadmissionPredictor:
    """
    XGBoost binary classifier for 30-day hospital readmission.

    Targets AUROC > 0.78. Supports:
    - Time-aware 5-fold cross-validation
    - Automatic class-imbalance handling via ``scale_pos_weight``
    - Platt-scaling probability calibration
    - Cost-sensitive optimal threshold selection

    Parameters
    ----------
    config : dict, optional
        Override for the ``xgboost.readmission`` config section.

    Attributes
    ----------
    feature_cols : list[str]
        Expected input feature column names.
    target_col : str
        Target column name (``'readmitted_30d'``).
    model : xgb.XGBClassifier or CalibratedClassifierCV or None
        Fitted model (set after training or calibration).
    transformer : ColumnTransformer or None
        Fitted preprocessing pipeline.
    scale_pos_weight : float
        Class-imbalance weight computed from training data.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        raw_config = _load_config()
        xgb_defaults = raw_config.get("xgboost", {}).get("readmission", {})

        if config:
            xgb_defaults.update(config)

        self.config: dict = xgb_defaults

        self.model: Optional[Any] = None
        self.transformer: Optional[ColumnTransformer] = None
        self.scale_pos_weight: float = float(
            self.config.get("scale_pos_weight", 8.0)
        )

        self.feature_cols: list[str] = FEATURE_COLS
        self.target_col: str = TARGET_COL

        mlflow_uri = raw_config.get("training", {}).get("mlflow_tracking_uri")
        self.tracker = MLflowTracker(
            experiment_name="healthrisk/readmission",
            tracking_uri=mlflow_uri,
        )

        logger.info(
            f"ReadmissionPredictor initialised. scale_pos_weight={self.scale_pos_weight}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_transformer(self, df: pd.DataFrame) -> ColumnTransformer:
        """Build ColumnTransformer based on column dtypes."""
        feature_df = df[[c for c in self.feature_cols if c in df.columns]]
        numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        transformers = []
        if numeric_cols:
            num_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", num_pipe, numeric_cols))

        if categorical_cols:
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipe, categorical_cols))

        if not transformers:
            raise ValueError("No feature columns with supported dtypes found.")

        return ColumnTransformer(
            transformers=transformers, remainder="drop", sparse_threshold=0
        )

    def _build_model(self, spw: Optional[float] = None) -> xgb.XGBClassifier:
        """Instantiate a fresh XGBClassifier with current config."""
        params = dict(self.config)
        for extra in ["early_stopping_rounds", "eval_metric"]:
            params.pop(extra, None)

        # Dynamic scale_pos_weight takes precedence
        params["scale_pos_weight"] = spw if spw is not None else self.scale_pos_weight
        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", 0)
        params.setdefault("use_label_encoder", False)

        return xgb.XGBClassifier(**params)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self, df: pd.DataFrame, fit: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features and target for training or inference.

        Computes ``scale_pos_weight`` from the label distribution when
        ``fit=True`` so the model automatically accounts for class imbalance.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing feature columns and (when fitting) the
            target column.
        fit : bool
            Fit the ColumnTransformer on this data if True.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        # Fill genuinely missing feature columns with zeros
        for col in self.feature_cols:
            if col not in df.columns:
                df = df.copy()
                df[col] = 0

        if fit:
            # Compute class-imbalance weight from data
            if TARGET_COL in df.columns:
                n_neg = (df[TARGET_COL] == 0).sum()
                n_pos = (df[TARGET_COL] == 1).sum()
                if n_pos > 0:
                    self.scale_pos_weight = float(n_neg / n_pos)
                    logger.info(
                        f"Computed scale_pos_weight={self.scale_pos_weight:.2f} "
                        f"(neg={n_neg}, pos={n_pos})"
                    )

            self.transformer = self._build_transformer(df)
            X = self.transformer.fit_transform(df[self.feature_cols])
        else:
            if self.transformer is None:
                raise RuntimeError(
                    "Transformer not fitted. Call preprocess(fit=True) first."
                )
            X = self.transformer.transform(df[self.feature_cols])

        y = df[TARGET_COL].values.astype(int) if TARGET_COL in df.columns else np.zeros(len(df))
        logger.debug(f"preprocess: X.shape={X.shape}, y.shape={y.shape}")
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self, df: pd.DataFrame, run_name: str = "readmission_v1"
    ) -> dict:
        """
        Train with time-aware 5-fold CV and full-data retraining.

        Per fold metrics logged to MLflow: AUROC, AUPRC, KS statistic,
        Gini coefficient. Targets AUROC > 0.78.

        Parameters
        ----------
        df : pd.DataFrame
            Training data. Must contain all feature columns and
            ``'readmitted_30d'``. Must contain a ``'date'`` column.
        run_name : str
            Base MLflow run name.

        Returns
        -------
        dict
            CV results: ``fold_results``, ``mean_auroc``, ``mean_auprc``,
            ``mean_ks``, ``mean_gini``, ``target_met`` (bool, AUROC > 0.78).
        """
        logger.info(f"Training ReadmissionPredictor on {len(df)} samples.")

        cv = TimeAwareCrossValidator(n_splits=5)
        sorted_df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
        splits = cv.split(sorted_df)

        early_stopping = self.config.get("early_stopping_rounds", 50)
        fold_results: list[dict] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            fold_num = fold_idx + 1
            logger.info(
                f"Fold {fold_num}/5 — train={len(train_idx)}, test={len(test_idx)}"
            )

            df_train = sorted_df.iloc[train_idx]
            df_test = sorted_df.iloc[test_idx]

            X_train, y_train = self.preprocess(df_train, fit=True)
            X_test, y_test = self.preprocess(df_test, fit=False)

            model = self._build_model(spw=self.scale_pos_weight)

            fit_kwargs: dict[str, Any] = {
                "eval_set": [(X_test, y_test)],
                "verbose": False,
            }
            if early_stopping:
                fit_kwargs["early_stopping_rounds"] = early_stopping

            model.fit(X_train, y_train, **fit_kwargs)

            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_classifier(y_test, y_proba)
            metrics["gini"] = compute_gini(metrics["auroc"])
            metrics["fold"] = fold_num
            fold_results.append(metrics)

            self.tracker.start_run(f"{run_name}_fold_{fold_num}")
            self.tracker.log_params(
                {
                    "fold": fold_num,
                    "scale_pos_weight": self.scale_pos_weight,
                    **{k: v for k, v in self.config.items()
                       if k not in ("eval_metric", "scale_pos_weight")},
                }
            )
            self.tracker.log_metrics(
                {k: v for k, v in metrics.items() if k != "fold"},
                step=fold_num,
            )
            self.tracker.end_run()

            logger.info(
                f"Fold {fold_num} — AUROC={metrics['auroc']:.4f}, "
                f"AUPRC={metrics['auprc']:.4f}, KS={metrics['ks_statistic']:.4f}, "
                f"Gini={metrics['gini']:.4f}"
            )

        # Aggregate
        metric_keys = ["auroc", "auprc", "f1", "precision", "recall",
                       "brier_score", "ks_statistic", "gini"]
        agg = {
            f"mean_{k}": float(np.mean([f[k] for f in fold_results]))
            for k in metric_keys
        }
        agg.update(
            {
                f"std_{k}": float(np.std([f[k] for f in fold_results]))
                for k in metric_keys
            }
        )
        target_met = agg["mean_auroc"] > 0.78
        agg["target_met"] = target_met
        logger.info(
            f"CV summary — mean_AUROC={agg['mean_auroc']:.4f}, "
            f"mean_Gini={agg['mean_gini']:.4f}, "
            f"target_AUROC_met={target_met}"
        )
        if not target_met:
            logger.warning(
                f"Target AUROC (>0.78) not met: {agg['mean_auroc']:.4f}. "
                "Consider additional feature engineering or hyperparameter tuning."
            )

        # Retrain on full data
        logger.info("Retraining on full dataset…")
        X_full, y_full = self.preprocess(sorted_df, fit=True)
        self.model = self._build_model(spw=self.scale_pos_weight)
        self.model.fit(X_full, y_full, verbose=False)

        self.tracker.start_run(f"{run_name}_final")
        self.tracker.log_params(
            {
                "scale_pos_weight": self.scale_pos_weight,
                **{k: v for k, v in self.config.items()
                   if k not in ("eval_metric", "scale_pos_weight")},
            }
        )
        self.tracker.log_metrics(
            {k: v for k, v in agg.items() if isinstance(v, float)}
        )
        self.tracker.log_model(self.model, "readmission_final")
        fi_df = self._get_feature_importance_df()
        if not fi_df.empty:
            self.tracker.log_feature_importance(
                fi_df["feature"].tolist(), fi_df["importance"].values
            )
        self.tracker.end_run()

        cv_results = {"fold_results": fold_results, **agg}
        logger.info("Training complete.")
        return cv_results

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict 30-day readmission probabilities.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with all feature columns.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Probability of readmission for each patient.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X, _ = self.preprocess(df, fit=False)

        if isinstance(self.model, CalibratedClassifierCV):
            proba = self.model.predict_proba(X)[:, 1]
        else:
            proba = self.model.predict_proba(X)[:, 1]

        return proba

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, df: pd.DataFrame) -> None:
        """
        Calibrate model probabilities using Platt scaling (sigmoid).

        Wraps the current model in a ``CalibratedClassifierCV`` with
        ``method='sigmoid'`` (Platt scaling), fitted on the provided
        calibration dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Hold-out calibration set. Must contain feature columns and
            ``target_col``.
        """
        if self.model is None:
            raise RuntimeError("Train the model before calibrating.")

        logger.info("Calibrating model with Platt scaling…")
        X_cal, y_cal = self.preprocess(df, fit=False)

        calibrated = CalibratedClassifierCV(
            estimator=self.model,
            method="sigmoid",
            cv="prefit",        # use the already-fitted model
        )
        calibrated.fit(X_cal, y_cal)
        self.model = calibrated
        logger.info("Calibration complete. Model replaced with calibrated wrapper.")

    # ------------------------------------------------------------------
    # Threshold optimisation
    # ------------------------------------------------------------------

    def get_optimal_threshold(
        self,
        df: pd.DataFrame,
        cost_fn_ratio: float = 3.0,
    ) -> float:
        """
        Find the decision threshold that minimises expected misclassification cost.

        Cost function:
            total_cost(t) = FP(t) * cost_fp + FN(t) * cost_fn
            where cost_fn = cost_fn_ratio * cost_fp

        A higher ``cost_fn_ratio`` shifts the threshold lower (flag more
        positives) to avoid missed readmissions.

        Parameters
        ----------
        df : pd.DataFrame
            Validation set with feature columns and ``target_col``.
        cost_fn_ratio : float
            Relative cost of a false negative vs a false positive.
            Default 3.0 (missing a readmission is 3× worse than a false alarm).

        Returns
        -------
        float
            Optimal probability threshold in [0, 1].
        """
        y_true = df[TARGET_COL].values.astype(int)
        y_proba = self.predict_proba(df)

        thresholds = np.linspace(0.01, 0.99, 199)
        best_threshold = 0.5
        best_cost = float("inf")

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            cost = fp + cost_fn_ratio * fn
            if cost < best_cost:
                best_cost = cost
                best_threshold = float(t)

        logger.info(
            f"Optimal threshold={best_threshold:.3f} "
            f"(cost_fn_ratio={cost_fn_ratio}, min_cost={best_cost:.0f})"
        )
        return best_threshold

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def _get_feature_importance_df(self) -> pd.DataFrame:
        """Internal: extract feature importances from the fitted model."""
        base = self.model
        if isinstance(base, CalibratedClassifierCV):
            base = base.estimator

        if base is None or not hasattr(base, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])

        importances = base.feature_importances_
        try:
            names = self.transformer.get_feature_names_out().tolist()
        except AttributeError:
            names = [f"f{i}" for i in range(len(importances))]

        return (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist model and transformer to pickle.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "transformer": self.transformer,
            "config": self.config,
            "scale_pos_weight": self.scale_pos_weight,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"ReadmissionPredictor saved to '{path}'.")

    def load(self, path: str) -> None:
        """
        Load model and transformer from pickle.

        Parameters
        ----------
        path : str
            Source file path.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.transformer = payload["transformer"]
        self.config = payload.get("config", self.config)
        self.scale_pos_weight = payload.get(
            "scale_pos_weight", self.scale_pos_weight
        )
        self.feature_cols = payload.get("feature_cols", self.feature_cols)
        self.target_col = payload.get("target_col", self.target_col)
        logger.info(f"ReadmissionPredictor loaded from '{path}'.")
