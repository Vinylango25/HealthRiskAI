"""
models/survival/cox_ph.py
==========================
Cox Proportional Hazards survival model for hospital readmission.

Uses the ``lifelines`` library to fit a penalised CoxPHFitter, provides
per-patient survival function prediction at arbitrary time points, and
includes proportional hazards assumption testing via Schoenfeld residuals.

Author: HealthRisk AI Team
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for server environments
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not available; plot_survival_curves will be disabled.")

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import proportional_hazard_test
    _LIFELINES_AVAILABLE = True
except ImportError:
    _LIFELINES_AVAILABLE = False
    logger.error(
        "lifelines not installed. Install with: pip install lifelines"
    )

from models.tabular.trainer import MLflowTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "model_config.yaml"


def _load_config() -> dict:
    """Load model configuration YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}


# ---------------------------------------------------------------------------
# CoxReadmissionModel
# ---------------------------------------------------------------------------


class CoxReadmissionModel:
    """
    Cox Proportional Hazards model for patient time-to-readmission.

    Wraps ``lifelines.CoxPHFitter`` with MLflow tracking, survival probability
    prediction at arbitrary time horizons, median survival estimation,
    Kaplan-Meier visualisation, and proportional hazards assumption testing.

    Parameters
    ----------
    penalizer : float
        L2 regularisation coefficient for the Cox model. Higher values
        produce a more regularised (smoother) fit. Default 0.1.

    Attributes
    ----------
    fitter : lifelines.CoxPHFitter or None
        Fitted Cox model (set after calling ``train``).
    duration_col : str
        Name of the duration column used during training.
    event_col : str
        Name of the event column used during training.
    covariate_cols : list[str]
        Covariate column names used during training.
    """

    def __init__(self, penalizer: float = 0.1) -> None:
        if not _LIFELINES_AVAILABLE:
            raise ImportError(
                "lifelines is required. Install with: pip install lifelines"
            )

        raw_config = _load_config()
        cox_cfg = raw_config.get("survival", {}).get("cox_ph", {})

        self.penalizer: float = penalizer or float(cox_cfg.get("penalizer", 0.1))
        self.l1_ratio: float = float(cox_cfg.get("l1_ratio", 0.0))

        self.fitter: Optional[CoxPHFitter] = None
        self.duration_col: str = "days_to_readmission"
        self.event_col: str = "readmitted_30d"
        self.covariate_cols: list[str] = []

        mlflow_uri = raw_config.get("training", {}).get("mlflow_tracking_uri")
        self.tracker = MLflowTracker(
            experiment_name="healthrisk/survival_cox_ph",
            tracking_uri=mlflow_uri,
        )

        logger.info(
            f"CoxReadmissionModel initialised "
            f"(penalizer={self.penalizer}, l1_ratio={self.l1_ratio})"
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_survival_data(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        covariate_cols: list[str],
    ) -> pd.DataFrame:
        """
        Validate and prepare a DataFrame for lifelines survival fitting.

        Checks for:
        - Negative durations (invalid; raised as ``ValueError``)
        - Tied event times (handled by Breslow method inside lifelines)
        - Missing values in covariates (median-imputed)

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        duration_col : str
            Column containing time-to-event (or censoring) in days.
        event_col : str
            Binary indicator column: 1 = event occurred, 0 = censored.
        covariate_cols : list[str]
            Feature columns to include in the Cox model.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with ``[duration_col, event_col] + covariate_cols``.

        Raises
        ------
        KeyError
            If required columns are absent.
        ValueError
            If negative durations are present.
        """
        required = {duration_col, event_col} | set(covariate_cols)
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise KeyError(
                f"Missing columns in DataFrame: {sorted(missing_cols)}"
            )

        work = df[[duration_col, event_col] + covariate_cols].copy()

        # Validate durations
        neg_mask = work[duration_col] < 0
        if neg_mask.any():
            n_neg = int(neg_mask.sum())
            raise ValueError(
                f"{n_neg} rows have negative durations in '{duration_col}'. "
                "Durations must be non-negative."
            )

        # Clip zero durations to a small positive value to avoid ties at 0
        zero_mask = work[duration_col] == 0
        if zero_mask.any():
            n_zero = int(zero_mask.sum())
            logger.warning(
                f"{n_zero} rows have duration=0. Clipping to 0.001 to avoid "
                "degenerate tied times at baseline."
            )
            work.loc[zero_mask, duration_col] = 0.001

        # Check for tied event times (informational)
        event_times = work.loc[work[event_col] == 1, duration_col]
        n_ties = int(event_times.duplicated().sum())
        if n_ties > 0:
            logger.info(
                f"{n_ties} tied event times detected — "
                "Breslow approximation will be used."
            )

        # Median-impute missing covariate values
        for col in covariate_cols:
            n_missing = int(work[col].isna().sum())
            if n_missing > 0:
                median_val = work[col].median()
                work[col].fillna(median_val, inplace=True)
                logger.debug(
                    f"Imputed {n_missing} missing values in '{col}' "
                    f"with median={median_val:.4f}"
                )

        logger.info(
            f"Survival data prepared: {len(work)} samples, "
            f"{int(work[event_col].sum())} events, "
            f"{n_ties} tied event times."
        )
        return work

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        duration_col: str = "days_to_readmission",
        event_col: str = "readmitted_30d",
        covariate_cols: Optional[list[str]] = None,
        run_name: str = "cox_ph_v1",
    ) -> dict:
        """
        Fit the CoxPHFitter and log results to MLflow.

        Logs:
        - ``concordance_index``: Harrell's C-index
        - ``aic``: Akaike Information Criterion
        - ``log_likelihood``: Model log-likelihood
        - Schoenfeld residuals test p-values per covariate

        Parameters
        ----------
        df : pd.DataFrame
            Training data containing duration, event, and covariate columns.
        duration_col : str
            Time-to-event column.
        event_col : str
            Event indicator column.
        covariate_cols : list[str], optional
            Covariates to include. If None, uses all numeric columns except
            duration and event.
        run_name : str
            MLflow run name.

        Returns
        -------
        dict with keys:
            ``c_index`` (float), ``summary_df`` (pd.DataFrame),
            ``baseline_hazard`` (pd.DataFrame).
        """
        # Infer covariates if not specified
        if covariate_cols is None:
            exclude = {duration_col, event_col}
            covariate_cols = [
                c for c in df.select_dtypes(include="number").columns
                if c not in exclude
            ]
            logger.info(
                f"No covariate_cols specified; using {len(covariate_cols)} "
                "numeric columns."
            )

        self.duration_col = duration_col
        self.event_col = event_col
        self.covariate_cols = covariate_cols

        survival_df = self.prepare_survival_data(
            df, duration_col, event_col, covariate_cols
        )

        logger.info(
            f"Fitting CoxPHFitter (penalizer={self.penalizer}, "
            f"n_covariates={len(covariate_cols)})…"
        )

        self.fitter = CoxPHFitter(
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
            baseline_estimation_method="breslow",
        )
        self.fitter.fit(
            survival_df,
            duration_col=duration_col,
            event_col=event_col,
        )

        c_index = float(self.fitter.concordance_index_)
        aic = float(self.fitter.AIC_)
        log_lik = float(self.fitter.log_likelihood_)

        logger.info(
            f"Fit complete — C-index={c_index:.4f}, AIC={aic:.2f}, "
            f"log-likelihood={log_lik:.2f}"
        )

        # Proportional hazards test
        ph_test = self.check_ph_assumption()
        logger.info(f"PH assumption test — verdict: {ph_test.get('verdict', 'N/A')}")

        # MLflow logging
        self.tracker.start_run(run_name)
        self.tracker.log_params(
            {
                "penalizer": self.penalizer,
                "l1_ratio": self.l1_ratio,
                "n_covariates": len(covariate_cols),
                "n_samples": len(survival_df),
                "n_events": int(survival_df[event_col].sum()),
            }
        )
        self.tracker.log_metrics(
            {
                "concordance_index": c_index,
                "aic": aic,
                "log_likelihood": log_lik,
            }
        )

        # Log per-covariate PH test p-values
        for cov, pval in ph_test.get("p_values", {}).items():
            safe = cov.replace(" ", "_")[:240]
            mlflow.log_metric(f"ph_pvalue_{safe}", float(pval))

        # Log model summary as CSV artefact
        summary_df = self.fitter.summary
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "cox_summary.csv"
            summary_df.to_csv(csv_path)
            mlflow.log_artifact(str(csv_path), "cox_summary")

        self.tracker.end_run()

        baseline_hazard = self.fitter.baseline_hazard_

        return {
            "c_index": c_index,
            "summary_df": summary_df,
            "baseline_hazard": baseline_hazard,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_survival(
        self,
        df: pd.DataFrame,
        times: list[float] = None,
    ) -> pd.DataFrame:
        """
        Predict survival probability for each patient at given time points.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with covariate columns.
        times : list[float]
            Time points (days) at which to evaluate survival probability.
            Default: [7, 14, 30, 60, 90].

        Returns
        -------
        pd.DataFrame
            Long-format table with columns:
            ``patient_id``, ``time``, ``survival_prob``.
        """
        if times is None:
            times = [7.0, 14.0, 30.0, 60.0, 90.0]

        if self.fitter is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Restrict to covariate columns only
        covariates_present = [
            c for c in self.covariate_cols if c in df.columns
        ]
        cov_df = df[covariates_present].copy()

        # Impute missing
        for col in covariates_present:
            cov_df[col].fillna(cov_df[col].median(), inplace=True)

        # lifelines returns (n_times, n_patients) survival matrix
        sf_matrix = self.fitter.predict_survival_function(
            cov_df, times=times
        )  # index = times, columns = patient position

        patient_ids = (
            df["patient_id"].values
            if "patient_id" in df.columns
            else np.arange(len(df))
        )

        rows = []
        for col_idx, pid in enumerate(patient_ids):
            for t in times:
                if t in sf_matrix.index:
                    sp = float(sf_matrix.loc[t, sf_matrix.columns[col_idx]])
                else:
                    sp = float("nan")
                rows.append({"patient_id": pid, "time": t, "survival_prob": sp})

        result = pd.DataFrame(rows)
        logger.debug(
            f"predict_survival: {len(patient_ids)} patients × {len(times)} time points."
        )
        return result

    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate median survival time (days-to-readmission) per patient.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with covariate columns.

        Returns
        -------
        pd.Series
            Median survival time per patient, indexed by patient position.
            NaN if the survival curve never crosses 0.5.
        """
        if self.fitter is None:
            raise RuntimeError("Model not trained. Call train() first.")

        covariates_present = [
            c for c in self.covariate_cols if c in df.columns
        ]
        cov_df = df[covariates_present].copy()
        for col in covariates_present:
            cov_df[col].fillna(cov_df[col].median(), inplace=True)

        median_times = self.fitter.predict_median(cov_df)
        logger.debug(
            f"predict_median_survival: {len(median_times)} patients. "
            f"Median of medians={float(median_times.median()):.1f} days"
        )
        return median_times

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_survival_curves(
        self,
        df: pd.DataFrame,
        group_col: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot Kaplan-Meier survival curves, optionally stratified by group.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``duration_col``, ``event_col``, and optionally
            ``group_col``.
        group_col : str, optional
            Column to stratify curves by. If None, plots overall curve.
        save_path : str, optional
            File path to save the figure (PNG). If None, displays interactively.
        """
        if not _MPL_AVAILABLE:
            logger.error("matplotlib is required for plot_survival_curves.")
            return

        if self.duration_col not in df.columns or self.event_col not in df.columns:
            raise KeyError(
                f"DataFrame must contain '{self.duration_col}' and '{self.event_col}'."
            )

        fig, ax = plt.subplots(figsize=(10, 6))

        if group_col and group_col in df.columns:
            groups = df[group_col].unique()
            for group in sorted(groups):
                mask = df[group_col] == group
                kmf = KaplanMeierFitter()
                kmf.fit(
                    durations=df.loc[mask, self.duration_col],
                    event_observed=df.loc[mask, self.event_col],
                    label=str(group),
                )
                kmf.plot_survival_function(ax=ax, ci_show=True)
            ax.set_title(f"Kaplan-Meier Survival Curves by {group_col}")
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=df[self.duration_col],
                event_observed=df[self.event_col],
                label="Overall",
            )
            kmf.plot_survival_function(ax=ax, ci_show=True)
            ax.set_title("Overall Kaplan-Meier Survival Curve")

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival Probability")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Survival curve saved to '{save_path}'.")
        else:
            plt.show()

        plt.close(fig)

    # ------------------------------------------------------------------
    # Proportional hazards assumption check
    # ------------------------------------------------------------------

    def check_ph_assumption(self) -> dict:
        """
        Test the proportional hazards assumption via Schoenfeld residuals.

        Uses ``lifelines.statistics.proportional_hazard_test``.
        A p-value < 0.05 for a covariate indicates violation of the PH
        assumption for that covariate.

        Returns
        -------
        dict with keys:
            ``p_values`` (dict: covariate → p-value),
            ``verdict`` ('assumption_met' or 'assumption_violated'),
            ``violations`` (list of covariate names with p < 0.05).
        """
        if self.fitter is None:
            raise RuntimeError("Model not trained. Call train() first.")

        try:
            ph_test_result = proportional_hazard_test(
                self.fitter,
                training_data=self.fitter._training_data,  # stored by lifelines
                time_transform="rank",
            )
        except Exception as exc:
            logger.warning(f"PH assumption test failed: {exc}")
            return {"p_values": {}, "verdict": "unknown", "violations": []}

        p_values: dict[str, float] = {}
        for _, row in ph_test_result.summary.iterrows():
            cov = str(row.name)
            pval = float(row.get("p", 1.0))
            p_values[cov] = pval

        violations = [cov for cov, p in p_values.items() if p < 0.05]
        verdict = "assumption_violated" if violations else "assumption_met"

        if violations:
            logger.warning(
                f"PH assumption violated for {len(violations)} covariates: "
                f"{violations}. Consider time-varying coefficients."
            )
        else:
            logger.info("PH assumption holds for all covariates (p ≥ 0.05).")

        return {
            "p_values": p_values,
            "verdict": verdict,
            "violations": violations,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the fitted CoxPH model to pickle.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fitter": self.fitter,
            "penalizer": self.penalizer,
            "l1_ratio": self.l1_ratio,
            "duration_col": self.duration_col,
            "event_col": self.event_col,
            "covariate_cols": self.covariate_cols,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"CoxReadmissionModel saved to '{path}'.")

    def load(self, path: str) -> None:
        """
        Load a fitted CoxPH model from pickle.

        Parameters
        ----------
        path : str
            Source file path.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.fitter = payload["fitter"]
        self.penalizer = payload.get("penalizer", self.penalizer)
        self.l1_ratio = payload.get("l1_ratio", self.l1_ratio)
        self.duration_col = payload.get("duration_col", self.duration_col)
        self.event_col = payload.get("event_col", self.event_col)
        self.covariate_cols = payload.get("covariate_cols", self.covariate_cols)
        logger.info(f"CoxReadmissionModel loaded from '{path}'.")
