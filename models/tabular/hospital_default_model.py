"""
models/tabular/hospital_default_model.py
==========================================
Hospital bond default probability model.

Predicts the probability of financial distress or default for healthcare
providers using a blend of financial, operational, and clinical quality metrics.
Produces credit-implied ratings and bond-spread estimates for fixed-income risk
management, and flags early clinical quality deterioration before it manifests
as financial distress.

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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import xgboost as xgb

from models.tabular.trainer import (
    MLflowTracker,
    TimeAwareCrossValidator,
    compute_gini,
    compute_psi,
    evaluate_classifier,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "model_config.yaml"

#: Feature columns — mix of financial and clinical quality KPIs.
FEATURE_COLS: list[str] = [
    # ── Financial metrics ──────────────────────────────────────────────────
    "operating_margin",                  # EBITDA / revenue
    "dscr",                              # Debt Service Coverage Ratio
    "days_cash_on_hand",                 # liquidity runway
    "debt_to_capitalization",            # leverage ratio
    "revenue_per_adjusted_discharge",    # productivity proxy
    # ── Payer mix ──────────────────────────────────────────────────────────
    "government_payer_concentration",    # Medicare + Medicaid % of revenue
    # ── Clinical quality indicators ────────────────────────────────────────
    "readmission_rate_excess",           # excess readmission ratio vs national avg
    "hcahps_star_rating",                # CMS patient satisfaction (1–5)
    "hai_sir",                           # Hospital-Acquired Infection SIR
    "cmi_trend",                         # Case Mix Index 12-month trend
    "ed_boarding_hours_avg",             # avg hours patients board in ED
    # ── Facility characteristics ───────────────────────────────────────────
    "bed_count",
    "teaching_hospital_flag",
    "critical_access_flag",
    "urban_rural_code",                  # RUCA code (ordinal)
    "system_affiliation_flag",
    # ── Historical default signals ─────────────────────────────────────────
    "prior_covenant_violation_flag",
    "credit_watch_flag",
    "rating_downgrade_12m_flag",
]

TARGET_COL = "defaulted"

# Bond spread mapping: PD threshold → (rating label, spread bps)
# Thresholds represent cumulative probability breakpoints
_RATING_SCHEDULE = [
    (0.001, "AAA",  10),
    (0.003, "AA",   25),
    (0.010, "A",    50),
    (0.030, "BBB", 100),
    (0.080, "BB",  200),
    (0.200, "B",   400),
    (1.000, "CCC", 800),
]


def _load_config() -> dict:
    """Load model configuration YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    logger.warning(f"Config not found at {CONFIG_PATH}. Using defaults.")
    return {}


def _pd_to_rating(pd_1yr: float) -> tuple[str, int]:
    """
    Map a 1-year probability of default to a credit rating and bond spread.

    Parameters
    ----------
    pd_1yr : float
        1-year probability of default (0–1).

    Returns
    -------
    (rating, spread_bps) : tuple[str, int]
    """
    for threshold, rating, spread in _RATING_SCHEDULE:
        if pd_1yr <= threshold:
            return rating, spread
    return "CCC", 800


# ---------------------------------------------------------------------------
# HospitalDefaultPredictor
# ---------------------------------------------------------------------------


class HospitalDefaultPredictor:
    """
    XGBoost binary classifier for hospital bond default / financial distress.

    Combines financial KPIs (operating margin, DSCR, liquidity) with clinical
    quality indicators (readmission rates, HCAHPS, HAI SIR) to produce:

    - 1-year probability of default (PD)
    - Credit-implied rating (AAA → CCC)
    - Estimated bond spread (bps)
    - Early-warning clinical quality scores

    Target metrics:
    - Gini coefficient > 0.50
    - KS statistic > 0.30

    Parameters
    ----------
    config : dict, optional
        Override for the ``xgboost.hospital_default`` config section.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        raw_config = _load_config()
        xgb_defaults = raw_config.get("xgboost", {}).get("hospital_default", {})

        if config:
            xgb_defaults.update(config)

        self.config: dict = xgb_defaults
        self.model: Optional[xgb.XGBClassifier] = None
        self.transformer: Optional[ColumnTransformer] = None
        self.scale_pos_weight: float = float(
            self.config.get("scale_pos_weight", 5.0)
        )

        self.feature_cols: list[str] = FEATURE_COLS
        self.target_col: str = TARGET_COL

        # Store training score distribution for PSI monitoring
        self._train_scores: Optional[np.ndarray] = None

        mlflow_uri = raw_config.get("training", {}).get("mlflow_tracking_uri")
        self.tracker = MLflowTracker(
            experiment_name="healthrisk/hospital_default",
            tracking_uri=mlflow_uri,
        )

        logger.info(
            f"HospitalDefaultPredictor initialised with config: {self.config}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_transformer(self, df: pd.DataFrame) -> ColumnTransformer:
        """Build a ColumnTransformer for feature preprocessing."""
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
            raise ValueError("No usable feature columns found.")

        return ColumnTransformer(
            transformers=transformers, remainder="drop", sparse_threshold=0
        )

    def _build_model(self, spw: Optional[float] = None) -> xgb.XGBClassifier:
        """Instantiate a fresh XGBClassifier."""
        params = dict(self.config)
        for extra in ["early_stopping_rounds", "eval_metric"]:
            params.pop(extra, None)

        params["scale_pos_weight"] = spw if spw is not None else self.scale_pos_weight
        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", 0)
        params.setdefault("use_label_encoder", False)

        return xgb.XGBClassifier(**params)

    def _preprocess(
        self, df: pd.DataFrame, fit: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess and return (X, y)."""
        for col in self.feature_cols:
            if col not in df.columns:
                df = df.copy()
                df[col] = 0.0

        if fit:
            n_neg = (df[TARGET_COL] == 0).sum() if TARGET_COL in df.columns else 1
            n_pos = (df[TARGET_COL] == 1).sum() if TARGET_COL in df.columns else 1
            if n_pos > 0:
                self.scale_pos_weight = float(n_neg / n_pos)
                logger.info(
                    f"scale_pos_weight={self.scale_pos_weight:.2f} "
                    f"(neg={n_neg}, pos={n_pos})"
                )

            self.transformer = self._build_transformer(df)
            X = self.transformer.fit_transform(df[self.feature_cols])
        else:
            if self.transformer is None:
                raise RuntimeError("Transformer not fitted.")
            X = self.transformer.transform(df[self.feature_cols])

        y = df[TARGET_COL].values.astype(int) if TARGET_COL in df.columns else np.zeros(len(df))
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self, df: pd.DataFrame, run_name: str = "hospital_default_v1"
    ) -> dict:
        """
        Train with time-aware 5-fold cross-validation.

        Per-fold metrics: AUROC, Gini (target > 0.50), KS statistic
        (target > 0.30), AUPRC, F1. Also computes PSI between training
        and each test fold's score distribution.

        Parameters
        ----------
        df : pd.DataFrame
            Training data. Must contain all feature columns, ``'defaulted'``,
            and a ``'date'`` column.
        run_name : str
            Base MLflow run name.

        Returns
        -------
        dict
            ``fold_results``, aggregated mean/std metrics, ``gini_target_met``,
            ``ks_target_met``.
        """
        logger.info(
            f"Training HospitalDefaultPredictor on {len(df)} samples."
        )

        cv = TimeAwareCrossValidator(n_splits=5)
        sorted_df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
        splits = cv.split(sorted_df)

        early_stopping = self.config.get("early_stopping_rounds", 40)
        fold_results: list[dict] = []
        train_scores_all: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            fold_num = fold_idx + 1
            logger.info(
                f"Fold {fold_num}/5 — train={len(train_idx)}, test={len(test_idx)}"
            )

            df_train = sorted_df.iloc[train_idx]
            df_test = sorted_df.iloc[test_idx]

            X_train, y_train = self._preprocess(df_train, fit=True)
            X_test, y_test = self._preprocess(df_test, fit=False)

            model = self._build_model(spw=self.scale_pos_weight)

            fit_kwargs: dict[str, Any] = {
                "eval_set": [(X_test, y_test)],
                "verbose": False,
            }
            if early_stopping:
                fit_kwargs["early_stopping_rounds"] = early_stopping

            model.fit(X_train, y_train, **fit_kwargs)

            # Predictions
            train_proba = model.predict_proba(X_train)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]
            train_scores_all.append(train_proba)

            # Classifier metrics
            metrics = evaluate_classifier(y_test, test_proba)
            metrics["gini"] = compute_gini(metrics["auroc"])
            metrics["fold"] = fold_num

            # PSI: score distribution shift between train and test fold
            psi = compute_psi(train_proba, test_proba, buckets=10)
            metrics["psi"] = psi

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
                {k: v for k, v in metrics.items()
                 if k not in ("fold",) and isinstance(v, (int, float))},
                step=fold_num,
            )
            self.tracker.end_run()

            logger.info(
                f"Fold {fold_num} — AUROC={metrics['auroc']:.4f}, "
                f"Gini={metrics['gini']:.4f}, KS={metrics['ks_statistic']:.4f}, "
                f"PSI={psi:.4f}"
            )

        # Aggregate CV metrics
        metric_keys = [
            "auroc", "auprc", "f1", "precision", "recall",
            "brier_score", "ks_statistic", "gini", "psi",
        ]
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
        agg["gini_target_met"] = agg["mean_gini"] > 0.50
        agg["ks_target_met"] = agg["mean_ks_statistic"] > 0.30
        logger.info(
            f"CV summary — mean_AUROC={agg['mean_auroc']:.4f}, "
            f"mean_Gini={agg['mean_gini']:.4f} (target >0.50: {agg['gini_target_met']}), "
            f"mean_KS={agg['mean_ks_statistic']:.4f} (target >0.30: {agg['ks_target_met']})"
        )

        # Retrain on full dataset
        logger.info("Retraining on full dataset…")
        X_full, y_full = self._preprocess(sorted_df, fit=True)
        self.model = self._build_model(spw=self.scale_pos_weight)
        self.model.fit(X_full, y_full, verbose=False)
        self._train_scores = self.model.predict_proba(X_full)[:, 1]

        # Final MLflow run
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
        self.tracker.log_model(self.model, "hospital_default_final")

        # Feature importance
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

    def predict_pd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict 1-year probability of default with credit rating and spread.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame. Must contain feature columns and a
            ``'provider_id'`` column.

        Returns
        -------
        pd.DataFrame
            Columns: ``provider_id``, ``pd_1yr``, ``credit_implied_rating``,
            ``bond_spread_bps``.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X, _ = self._preprocess(df, fit=False)
        pd_scores = self.model.predict_proba(X)[:, 1]

        ratings, spreads = zip(*[_pd_to_rating(p) for p in pd_scores])

        provider_ids = (
            df["provider_id"].values
            if "provider_id" in df.columns
            else np.arange(len(df))
        )

        result = pd.DataFrame(
            {
                "provider_id": provider_ids,
                "pd_1yr": pd_scores,
                "credit_implied_rating": list(ratings),
                "bond_spread_bps": list(spreads),
            }
        )

        # Optionally compute PSI vs training distribution
        if self._train_scores is not None:
            psi = compute_psi(self._train_scores, pd_scores)
            logger.info(f"Inference PSI vs training distribution: {psi:.4f}")
            if psi > 0.25:
                logger.warning(
                    f"PSI={psi:.4f} > 0.25. Score distribution has shifted "
                    "significantly — consider model retraining."
                )

        logger.info(
            f"predict_pd: {len(result)} hospitals. "
            f"Median PD={float(np.median(pd_scores)):.4f}, "
            f"Mean spread={float(np.mean(list(spreads))):.0f} bps"
        )
        return result

    # ------------------------------------------------------------------
    # Early warning
    # ------------------------------------------------------------------

    def compute_early_warning_score(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect hospitals showing clinical quality deterioration that may
        precede financial distress.

        Flags are triggered when a hospital meets the primary condition
        AND at least one secondary clinical deterioration indicator:

        Primary:
            ``readmission_rate_excess > 0.02``

        Secondary (any one):
            - ``cmi_trend < -0.05``  (declining case mix index)
            - ``hcahps_star_rating <= 2``  (poor patient satisfaction)
            - ``hai_sir > 1.5``  (high hospital-acquired infections)

        Severity is determined by the number of secondary conditions met:
            - 0 secondary: ``'low'``
            - 1 secondary: ``'medium'``
            - 2 secondary: ``'high'``
            - 3 secondary: ``'critical'``

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with clinical quality columns and ``'provider_id'``.

        Returns
        -------
        pd.DataFrame
            Columns: ``provider_id``, ``early_warning_flag``,
            ``warning_reasons`` (list), ``severity``.
        """
        required_cols = [
            "readmission_rate_excess",
            "cmi_trend",
            "hcahps_star_rating",
            "hai_sir",
        ]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(
                    f"compute_early_warning_score: column '{col}' missing. "
                    "Filling with 0."
                )
                df = df.copy()
                df[col] = 0.0

        provider_ids = (
            df["provider_id"].values
            if "provider_id" in df.columns
            else np.arange(len(df))
        )

        rows = []
        severity_map = {0: "low", 1: "medium", 2: "high", 3: "critical"}

        for i, (_, row) in enumerate(df.iterrows()):
            primary_flag = bool(row["readmission_rate_excess"] > 0.02)
            reasons: list[str] = []

            if primary_flag:
                reasons.append(
                    f"readmission_rate_excess={row['readmission_rate_excess']:.4f} > 0.02"
                )

                # Evaluate secondary conditions
                secondary_triggers = 0
                if row["cmi_trend"] < -0.05:
                    reasons.append(
                        f"cmi_trend={row['cmi_trend']:.4f} < -0.05"
                    )
                    secondary_triggers += 1

                if row["hcahps_star_rating"] <= 2:
                    reasons.append(
                        f"hcahps_star_rating={row['hcahps_star_rating']:.1f} <= 2"
                    )
                    secondary_triggers += 1

                if row["hai_sir"] > 1.5:
                    reasons.append(
                        f"hai_sir={row['hai_sir']:.4f} > 1.5"
                    )
                    secondary_triggers += 1

                # Only flag if at least one secondary condition is met
                flagged = secondary_triggers > 0
                severity = severity_map.get(secondary_triggers, "critical")
            else:
                flagged = False
                severity = "low"
                reasons = []

            rows.append(
                {
                    "provider_id": provider_ids[i],
                    "early_warning_flag": flagged,
                    "warning_reasons": reasons,
                    "severity": severity,
                }
            )

        result = pd.DataFrame(rows)
        flagged_count = result["early_warning_flag"].sum()
        logger.info(
            f"Early warning: {flagged_count}/{len(result)} hospitals flagged. "
            f"Severity breakdown: "
            + str(result[result["early_warning_flag"]]["severity"].value_counts().to_dict())
        )
        return result

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def _get_feature_importance_df(self) -> pd.DataFrame:
        """Return sorted feature importances as a DataFrame."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])

        importances = self.model.feature_importances_
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
        """Persist model, transformer, and train scores to pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "transformer": self.transformer,
            "config": self.config,
            "scale_pos_weight": self.scale_pos_weight,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "_train_scores": self._train_scores,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"HospitalDefaultPredictor saved to '{path}'.")

    def load(self, path: str) -> None:
        """Load model and transformer from pickle."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.transformer = payload["transformer"]
        self.config = payload.get("config", self.config)
        self.scale_pos_weight = payload.get("scale_pos_weight", self.scale_pos_weight)
        self.feature_cols = payload.get("feature_cols", self.feature_cols)
        self.target_col = payload.get("target_col", self.target_col)
        self._train_scores = payload.get("_train_scores")
        logger.info(f"HospitalDefaultPredictor loaded from '{path}'.")
