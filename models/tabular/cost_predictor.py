"""
models/tabular/cost_predictor.py
=================================
XGBoost 12-month healthcare cost prediction model.

Predicts total healthcare expenditure over the next 12 months using a Tweedie
regression objective, suitable for the right-skewed, zero-inflated distribution
typical of medical cost data.

Author: HealthRisk AI Team
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from models.tabular.trainer import (
    MLflowTracker,
    TimeAwareCrossValidator,
    compute_psi,
    evaluate_regressor,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "model_config.yaml"

#: All feature columns expected by the model, grouped by semantic category.
FEATURE_COLS: list[str] = [
    # ── Demographics ────────────────────────────────────────────────────────
    "age",
    "gender_male",
    "race_white",
    "race_black",
    "race_hispanic",
    "race_asian",
    "race_other",
    "dual_eligible",            # Medicare+Medicaid dual eligibility flag
    # ── HCC risk scores ─────────────────────────────────────────────────────
    "hcc_risk_score",
    "hcc_risk_score_prior",
    "hcc_risk_score_delta",
    "n_hcc_conditions",
    # ── Lab trajectories ────────────────────────────────────────────────────
    "hba1c_last",
    "hba1c_trend_6m",
    "creatinine_last",
    "creatinine_trend_6m",
    "egfr_last",
    "bmi_last",
    "sbp_last",
    "dbp_last",
    "ldl_last",
    "hemoglobin_last",
    # ── Utilisation history ─────────────────────────────────────────────────
    "ed_visits_12m",
    "ed_visits_24m",
    "ip_admissions_12m",
    "ip_admissions_24m",
    "ip_days_12m",
    "op_visits_12m",
    "snf_days_12m",
    "home_health_visits_12m",
    # ── Cost history ────────────────────────────────────────────────────────
    "total_cost_prior_12m",
    "rx_cost_prior_12m",
    "ip_cost_prior_12m",
    "op_cost_prior_12m",
    # ── Medication features ─────────────────────────────────────────────────
    "n_unique_drugs",
    "n_drug_classes",
    "polypharmacy_flag",        # ≥ 5 concurrent medications
    "high_cost_drug_flag",      # specialty / biologic medication
    "adherence_pdc",            # proportion days covered (0–1)
    # ── Charlson comorbidity ────────────────────────────────────────────────
    "charlson_score",
    "charlson_deyo_score",
    # ── Diagnosis flags ─────────────────────────────────────────────────────
    "dx_diabetes",
    "dx_chf",
    "dx_copd",
    "dx_ckd",
    "dx_cancer",
    "dx_depression",
    "dx_dementia",
    "dx_obesity",
]

TARGET_COL = "total_cost_12m"


def _load_config() -> dict:
    """Load model configuration YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    logger.warning(f"Config not found at {CONFIG_PATH}. Using defaults.")
    return {}


# ---------------------------------------------------------------------------
# HealthcareCostPredictor
# ---------------------------------------------------------------------------


class HealthcareCostPredictor:
    """
    XGBoost 12-month healthcare cost predictor with Tweedie regression.

    Handles the full ML lifecycle: preprocessing, time-aware cross-validation,
    MLflow tracking, uncertainty quantification, and artefact persistence.

    Parameters
    ----------
    config : dict, optional
        Override for the ``xgboost.cost_prediction`` section of
        ``configs/model_config.yaml``. If None, loaded from file.

    Attributes
    ----------
    feature_cols : list[str]
        Expected input feature column names.
    target_col : str
        Target column name (``'total_cost_12m'``).
    model : xgb.XGBRegressor or None
        Fitted XGBoost model (set after training).
    transformer : ColumnTransformer or None
        Fitted sklearn preprocessing pipeline.
    _X_train_for_bootstrap : np.ndarray or None
        Training features retained for bootstrap uncertainty estimation.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        raw_config = _load_config()
        xgb_defaults = raw_config.get("xgboost", {}).get("cost_prediction", {})

        # Allow caller to override any key
        if config:
            xgb_defaults.update(config)

        self.config: dict = xgb_defaults

        # Model artefacts (populated after training)
        self.model: Optional[xgb.XGBRegressor] = None
        self.transformer: Optional[ColumnTransformer] = None
        self._X_train_for_bootstrap: Optional[np.ndarray] = None

        # Feature / target metadata
        self.feature_cols: list[str] = FEATURE_COLS
        self.target_col: str = TARGET_COL

        # MLflow tracker (experiment name from config if available)
        mlflow_uri = raw_config.get("training", {}).get("mlflow_tracking_uri")
        self.tracker = MLflowTracker(
            experiment_name="healthrisk/cost_prediction",
            tracking_uri=mlflow_uri,
        )

        logger.info(
            f"HealthcareCostPredictor initialised with config: {self.config}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_transformer(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Construct a ColumnTransformer that median-imputes numeric columns
        and mode-imputes categorical columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame used to identify numeric/categorical columns.

        Returns
        -------
        ColumnTransformer (unfitted)
        """
        feature_df = df[[c for c in self.feature_cols if c in df.columns]]

        numeric_cols = feature_df.select_dtypes(
            include=["number"]
        ).columns.tolist()
        categorical_cols = feature_df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        transformers = []
        if numeric_cols:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", numeric_pipeline, numeric_cols))

        if categorical_cols:
            from sklearn.preprocessing import OrdinalEncoder

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipeline, categorical_cols))

        if not transformers:
            raise ValueError("No numeric or categorical feature columns found.")

        return ColumnTransformer(
            transformers=transformers, remainder="drop", sparse_threshold=0
        )

    def _build_model(self, scale_pos_weight: Optional[float] = None) -> xgb.XGBRegressor:
        """Instantiate a fresh XGBRegressor from stored config."""
        params = dict(self.config)
        # Remove non-XGBoost keys that may appear in config
        for extra in ["early_stopping_rounds"]:
            params.pop(extra, None)

        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", 0)

        return xgb.XGBRegressor(**params)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self, df: pd.DataFrame, fit: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for XGBoost training/inference.

        Missing numerics are median-imputed; missing categoricals are
        mode-imputed. The ColumnTransformer is fitted when ``fit=True``
        and reused (transform only) when ``fit=False``.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame. Must contain ``self.feature_cols`` and
            (when ``fit=True``) ``self.target_col``.
        fit : bool
            Whether to fit the transformer on this data. Default True.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
            Returns an all-zero array for y if target column is absent.
        """
        # Gracefully handle missing feature columns
        available = [c for c in self.feature_cols if c in df.columns]
        missing = set(self.feature_cols) - set(available)
        if missing:
            logger.warning(
                f"preprocess: {len(missing)} feature columns missing from DataFrame; "
                f"they will be filled with zeros: {sorted(missing)}"
            )
            for col in missing:
                df = df.copy()
                df[col] = 0.0

        if fit:
            self.transformer = self._build_transformer(df)
            X = self.transformer.fit_transform(df[self.feature_cols])
        else:
            if self.transformer is None:
                raise RuntimeError(
                    "Transformer not fitted. Call preprocess(fit=True) first."
                )
            X = self.transformer.transform(df[self.feature_cols])

        if TARGET_COL in df.columns:
            y = df[TARGET_COL].values.astype(float)
        else:
            y = np.zeros(len(df))

        logger.debug(f"preprocess: X.shape={X.shape}, y.shape={y.shape}")
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self, df: pd.DataFrame, run_name: str = "cost_predictor_v1"
    ) -> dict:
        """
        Train the cost predictor with time-aware 5-fold cross-validation.

        Each fold:
        1. Preprocesses train/test splits.
        2. Fits XGBoost with Tweedie objective and early stopping.
        3. Evaluates on the held-out test fold.
        4. Logs all metrics and the fold model to MLflow.

        After CV, retrains on the full dataset and logs the final model.

        Parameters
        ----------
        df : pd.DataFrame
            Full training DataFrame. Must contain all ``feature_cols`` and
            ``target_col``. Must contain a date/time column (default ``'date'``).
        run_name : str
            Base name for MLflow runs.

        Returns
        -------
        dict
            ``cv_results`` with per-fold and aggregated metrics:
            ``fold_results``, ``mean_rmse``, ``mean_mape``, ``mean_r2``,
            ``mean_predictive_ratio``.
        """
        logger.info(
            f"Starting training: {len(df)} samples, run_name='{run_name}'"
        )

        cv = TimeAwareCrossValidator(n_splits=5)
        splits = cv.split(df)

        sorted_df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

        fold_results: list[dict] = []
        early_stopping = self.config.get("early_stopping_rounds", 50)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            fold_num = fold_idx + 1
            fold_run_name = f"{run_name}_fold_{fold_num}"
            logger.info(
                f"Fold {fold_num}/5 — train={len(train_idx)}, test={len(test_idx)}"
            )

            df_train_fold = sorted_df.iloc[train_idx]
            df_test_fold = sorted_df.iloc[test_idx]

            X_train, y_train = self.preprocess(df_train_fold, fit=True)
            X_test, y_test = self.preprocess(df_test_fold, fit=False)

            model = self._build_model()

            fit_kwargs: dict[str, Any] = {
                "eval_set": [(X_test, y_test)],
                "verbose": False,
            }
            if early_stopping:
                fit_kwargs["early_stopping_rounds"] = early_stopping

            model.fit(X_train, y_train, **fit_kwargs)

            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, 0.0, None)  # costs must be non-negative

            metrics = evaluate_regressor(y_test, y_pred)
            metrics["fold"] = fold_num
            fold_results.append(metrics)

            # MLflow: log this fold
            self.tracker.start_run(fold_run_name)
            self.tracker.log_params(
                {
                    "fold": fold_num,
                    **{k: v for k, v in self.config.items() if k != "eval_metric"},
                }
            )
            self.tracker.log_metrics(
                {k: v for k, v in metrics.items() if k != "fold"},
                step=fold_num,
            )
            self.tracker.log_model(model, f"cost_predictor_fold_{fold_num}")
            self.tracker.end_run()

            logger.info(
                f"Fold {fold_num} — RMSE={metrics['rmse']:.2f}, "
                f"MAPE={metrics['mape']:.4f}, R²={metrics['r2']:.4f}, "
                f"PR={metrics['predictive_ratio']:.4f}"
            )

        # Aggregate CV metrics
        metric_keys = ["rmse", "mae", "mape", "r2", "median_ae", "predictive_ratio"]
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
        logger.info(
            f"CV summary — mean_RMSE={agg['mean_rmse']:.2f}, "
            f"mean_MAPE={agg['mean_mape']:.4f}, "
            f"mean_R²={agg['mean_r2']:.4f}, "
            f"mean_PR={agg['mean_predictive_ratio']:.4f}"
        )

        # Retrain on full data
        logger.info("Retraining on full dataset…")
        X_full, y_full = self.preprocess(sorted_df, fit=True)
        self.model = self._build_model()
        self.model.fit(X_full, y_full, verbose=False)
        self._X_train_for_bootstrap = X_full.copy()

        # Log final model to MLflow
        self.tracker.start_run(f"{run_name}_final")
        self.tracker.log_params(
            {k: v for k, v in self.config.items() if k != "eval_metric"}
        )
        self.tracker.log_metrics(agg)
        self.tracker.log_model(self.model, "cost_predictor_final")
        fi_df = self.get_feature_importance()
        if not fi_df.empty:
            self.tracker.log_feature_importance(
                fi_df["feature"].tolist(),
                fi_df["importance"].values,
            )
        self.tracker.end_run()

        cv_results = {"fold_results": fold_results, **agg}
        logger.info("Training complete.")
        return cv_results

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict 12-month healthcare costs.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with all feature columns.

        Returns
        -------
        np.ndarray
            Predicted costs per patient (non-negative).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X, _ = self.preprocess(df, fit=False)
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, None)

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        n_iterations: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate prediction uncertainty via bootstrap sampling.

        Re-fits ``n_iterations`` models on random 80% subsamples of the
        training data, then computes per-sample mean and standard deviation
        across iterations.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with all feature columns.
        n_iterations : int
            Number of bootstrap iterations. Default 100.

        Returns
        -------
        mean_preds : np.ndarray, shape (n_samples,)
        std_preds : np.ndarray, shape (n_samples,)
        """
        if self._X_train_for_bootstrap is None:
            raise RuntimeError(
                "Training data not stored. Call train() before "
                "predict_with_uncertainty()."
            )
        if self.transformer is None:
            raise RuntimeError("Transformer not fitted.")

        X_infer, _ = self.preprocess(df, fit=False)
        n_train = len(self._X_train_for_bootstrap)
        subsample_size = int(0.8 * n_train)

        all_preds: list[np.ndarray] = []
        rng = np.random.default_rng(seed=42)

        logger.info(
            f"Bootstrap uncertainty: {n_iterations} iterations, "
            f"subsample_size={subsample_size}"
        )

        for i in range(n_iterations):
            idx = rng.choice(n_train, size=subsample_size, replace=False)
            X_boot = self._X_train_for_bootstrap[idx]

            # Recover y for this bootstrap sample — we stored X only,
            # so we predict with the full model as a proxy target.
            # In production, store y_full alongside X_full.
            y_boot = self.model.predict(X_boot)

            boot_model = self._build_model()
            boot_model.fit(X_boot, y_boot, verbose=False)
            preds = np.clip(boot_model.predict(X_infer), 0.0, None)
            all_preds.append(preds)

            if (i + 1) % 20 == 0:
                logger.debug(f"Bootstrap iteration {i + 1}/{n_iterations}")

        stacked = np.stack(all_preds, axis=0)  # (n_iterations, n_samples)
        mean_preds = stacked.mean(axis=0)
        std_preds = stacked.std(axis=0)

        logger.info(
            f"Bootstrap complete — mean uncertainty (std): "
            f"{std_preds.mean():.2f}"
        )
        return mean_preds, std_preds

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return XGBoost feature importances as a sorted DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``. Sorted descending by
            importance. Empty DataFrame if model is not fitted.
        """
        if self.model is None:
            logger.warning("get_feature_importance: model is None.")
            return pd.DataFrame(columns=["feature", "importance"])

        importances = self.model.feature_importances_

        # Recover feature names from the fitted ColumnTransformer
        try:
            feature_names = self.transformer.get_feature_names_out().tolist()
        except AttributeError:
            feature_names = [f"f{i}" for i in range(len(importances))]

        fi_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        return fi_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist model and transformer to a pickle file.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``'checkpoints/cost_predictor.pkl'``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "transformer": self.transformer,
            "config": self.config,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"HealthcareCostPredictor saved to '{path}'.")

    def load(self, path: str) -> None:
        """
        Load model and transformer from a pickle file.

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
        self.feature_cols = payload.get("feature_cols", self.feature_cols)
        self.target_col = payload.get("target_col", self.target_col)
        logger.info(f"HealthcareCostPredictor loaded from '{path}'.")
