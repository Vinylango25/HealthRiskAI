"""
tests/test_tabular.py
=====================
Unit tests for tabular model data preparation and metrics logic.

Heavy ML model classes (XGBoost, LightGBM, etc.) are NOT imported directly.
Only data-logic utilities and sklearn metric helpers are tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_percentage_error,
    roc_auc_score,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LAB_COLS = [
    "hemoglobin", "creatinine", "bun", "sodium", "potassium",
    "glucose", "hba1c", "ldl_cholesterol", "hdl_cholesterol", "troponin",
]
NUMERIC_FEATURES = ["age", "los_days"] + LAB_COLS


def _build_feature_matrix(df: pd.DataFrame):
    """Return (X, y_mortality, y_readmit, y_cost) for downstream tests."""
    feat_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    X = df[feat_cols].fillna(df[feat_cols].median())
    y_mortality = df["mortality_30d"].astype(int).values
    y_readmit = df["readmission_30d"].astype(int).values
    y_cost = df["total_cost"].values
    return X, y_mortality, y_readmit, y_cost


# ---------------------------------------------------------------------------
# TestTabularDataPrep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTabularDataPrep:
    """Data preparation logic for tabular patient models."""

    def test_feature_matrix_shape(self, sample_patient_df):
        """Feature matrix has (n_rows, n_feature_cols) shape."""
        X, y_mortality, _, _ = _build_feature_matrix(sample_patient_df)
        assert X.shape[0] == len(sample_patient_df), "Row count must match"
        assert X.shape[1] >= 5, "At least 5 feature columns expected"
        assert len(y_mortality) == X.shape[0], "Label length must match feature rows"

    def test_no_nulls_after_prep(self, sample_patient_df):
        """Feature matrix has zero NaN after median fillna."""
        X, _, _, _ = _build_feature_matrix(sample_patient_df)
        null_count = X.isna().sum().sum()
        assert null_count == 0, f"Expected 0 NaN after fillna, found {null_count}"

    def test_label_distribution(self, sample_patient_df):
        """Binary labels contain only 0 and 1."""
        _, y_mortality, y_readmit, _ = _build_feature_matrix(sample_patient_df)
        assert set(np.unique(y_mortality)).issubset({0, 1}), "mortality_30d must be binary"
        assert set(np.unique(y_readmit)).issubset({0, 1}), "readmission_30d must be binary"

    def test_cost_target_positive(self, sample_patient_df):
        """total_cost values are all strictly positive."""
        _, _, _, y_cost = _build_feature_matrix(sample_patient_df)
        assert (y_cost > 0).all(), "All total_cost values must be positive"


# ---------------------------------------------------------------------------
# TestTabularMetrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTabularMetrics:
    """Metric calculation helpers used by tabular models."""

    def test_auroc_range(self):
        """AUROC computed on synthetic predictions is strictly in (0, 1)."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        # Mildly informative: noise dominates so AUROC stays well below 1
        y_pred = y_true * 0.3 + rng.uniform(0, 0.7, size=200)
        y_pred = y_pred.clip(0, 1)
        auroc = roc_auc_score(y_true, y_pred)
        assert 0 < auroc < 1, f"AUROC must be in (0, 1), got {auroc:.4f}"

    def test_mape_calculation(self):
        """MAPE formula: mean(|y_true - y_pred| / |y_true|) matches sklearn."""
        rng = np.random.default_rng(7)
        y_true = rng.uniform(100, 1000, size=100)
        y_pred = y_true * rng.uniform(0.8, 1.2, size=100)
        # Manual calculation
        mape_manual = np.mean(np.abs((y_true - y_pred) / y_true))
        # sklearn
        mape_sklearn = mean_absolute_percentage_error(y_true, y_pred)
        assert abs(mape_manual - mape_sklearn) < 1e-10, (
            f"Manual MAPE {mape_manual:.6f} != sklearn {mape_sklearn:.6f}"
        )

    def test_brier_score_bounds(self):
        """Brier score is in [0, 1] for valid binary predictions."""
        rng = np.random.default_rng(13)
        y_true = rng.integers(0, 2, size=150)
        y_pred_proba = rng.uniform(0, 1, size=150)
        brier = brier_score_loss(y_true, y_pred_proba)
        assert 0 <= brier <= 1, f"Brier score must be in [0, 1], got {brier:.4f}"


# ---------------------------------------------------------------------------
# TestHospitalDataPrep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHospitalDataPrep:
    """Data preparation checks for hospital default model inputs."""

    def test_hospital_feature_count(self, sample_hospital_df):
        """sample_hospital_df contains at least 8 numeric columns."""
        numeric_cols = sample_hospital_df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) >= 8, (
            f"Expected >= 8 numeric cols, found {len(numeric_cols)}: {list(numeric_cols)}"
        )

    def test_default_label(self, sample_hospital_df):
        """default_probability values are in [0, 1]."""
        dp = sample_hospital_df["default_probability"]
        assert dp.between(0.0, 1.0).all(), (
            f"default_probability out of [0, 1]: min={dp.min()}, max={dp.max()}"
        )
