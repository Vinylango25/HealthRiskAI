"""
tests/test_processing.py
========================
Unit tests for data/processing modules:
  DataCleaner, LabNormaliser, ICDNormaliser, CohortBuilder,
  DataSplitter, SchemaValidator, ValidationResult
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

try:
    from data.processing import (
        CohortBuilder,
        DataCleaner,
        DataSplitter,
        ICDNormaliser,
        LabNormaliser,
        SchemaValidator,
        ValidationResult,
    )
except ImportError as exc:
    pytest.skip(f"data.processing not importable: {exc}", allow_module_level=True)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# TestDataCleaner
# ---------------------------------------------------------------------------


class TestDataCleaner:
    """Tests for DataCleaner."""

    def test_handle_missing_median(self):
        """NaN in numeric column is filled with the column median."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, np.nan]})
        cleaner = DataCleaner(verbose=False)
        result = cleaner.handle_missing(df, strategy="median", numeric_cols=["a"])
        assert result["a"].isna().sum() == 0, "Expected no NaN after median imputation"
        # Median of [1, 3, 4] = 3; NaN slots should be 3
        assert result["a"].iloc[1] == pytest.approx(3.0)

    def test_remove_duplicates(self):
        """Duplicate rows are removed, unique rows are preserved."""
        df = pd.DataFrame({"id": [1, 2, 2, 3, 3], "val": [10, 20, 20, 30, 30]})
        cleaner = DataCleaner(verbose=False)
        result = cleaner.remove_duplicates(df)
        assert len(result) == 3, "Expected 3 unique rows after dedup"
        assert result["id"].tolist() == [1, 2, 3]

    def test_clip_outliers_iqr(self):
        """Extreme outliers are clipped to IQR fences."""
        rng = np.random.default_rng(0)
        vals = rng.normal(50, 5, 100).tolist()
        vals[0] = 1_000.0   # extreme high
        vals[1] = -1_000.0  # extreme low
        df = pd.DataFrame({"x": vals})
        cleaner = DataCleaner(verbose=False)
        result = cleaner.clip_outliers(df, cols=["x"], method="iqr", multiplier=1.5)
        assert result["x"].max() < 1_000.0, "High outlier should be clipped"
        assert result["x"].min() > -1_000.0, "Low outlier should be clipped"

    def test_fix_dtypes(self):
        """fix_dtypes casts columns according to the provided schema dict."""
        df = pd.DataFrame({"age": ["25", "30", "45"], "flag": ["1", "0", "1"]})
        cleaner = DataCleaner(verbose=False)
        schema = {"age": "float64", "flag": "int32"}
        result = cleaner.fix_dtypes(df, schema)
        assert result["age"].dtype == np.float64, "age should be cast to float64"
        assert result["flag"].dtype == np.int32, "flag should be cast to int32"

    def test_clean_patient_df(self, sample_patient_df):
        """clean_patient_df completes without raising on numeric columns of the session fixture."""
        cleaner = DataCleaner(verbose=False)
        # Drop list-typed column (diagnoses) which clean_patient_df doesn't handle
        df = sample_patient_df.drop(columns=["diagnoses"], errors="ignore")
        result = cleaner.clean_patient_df(df)
        assert isinstance(result, pd.DataFrame), "Expected a DataFrame"
        assert len(result) > 0, "Result should be non-empty"


# ---------------------------------------------------------------------------
# TestLabNormaliser
# ---------------------------------------------------------------------------


class TestLabNormaliser:
    """Tests for LabNormaliser."""

    def test_normalise_hemoglobin(self, sample_patient_df):
        """Normalised hemoglobin values are in [0, 1]."""
        ln = LabNormaliser(clip=True)
        normed = ln.normalise(sample_patient_df, "hemoglobin")
        assert normed.between(0.0, 1.0).all(), "All normalised Hgb values must be in [0, 1]"

    def test_normalise_all_labs(self, sample_patient_df):
        """normalise_all_labs appends *_norm columns and preserves row count."""
        ln = LabNormaliser(clip=True)
        result = ln.normalise_all_labs(sample_patient_df)
        assert result.shape[0] == len(sample_patient_df), "Row count must be preserved"
        norm_cols = [c for c in result.columns if c.endswith("_norm")]
        assert len(norm_cols) > 0, "At least one _norm column should be created"

    def test_flag_abnormal(self, sample_patient_df):
        """flag_abnormal returns a Series with values in {'normal', 'high', 'low'}."""
        ln = LabNormaliser()
        flags = ln.flag_abnormal(sample_patient_df, "hemoglobin")
        valid = {"normal", "high", "low"}
        assert set(flags.dropna().unique()).issubset(valid), (
            f"Expected only {valid}, got {set(flags.dropna().unique())}"
        )


# ---------------------------------------------------------------------------
# TestICDNormaliser
# ---------------------------------------------------------------------------


class TestICDNormaliser:
    """Tests for ICDNormaliser."""

    def test_to_category_cardiovascular(self):
        """ICD code I10 maps to 'cardiovascular' category."""
        norm = ICDNormaliser()
        result = norm.to_category("I10")
        assert result == "cardiovascular", f"Expected 'cardiovascular', got '{result}'"

    def test_to_category_diabetes(self):
        """ICD code E11.9 maps to the endocrine/metabolic category."""
        norm = ICDNormaliser()
        result = norm.to_category("E11.9")
        # The normaliser uses 'endocrine_metabolic' as the category label
        assert result in ("metabolic", "endocrine_metabolic"), (
            f"Expected metabolic/endocrine_metabolic, got '{result}'"
        )

    def test_normalise_codes(self):
        """normalise_codes strips whitespace and upper-cases ICD codes."""
        norm = ICDNormaliser()
        raw = ["  i10 ", "E11.9", " j44.1  "]
        result = norm.normalise_codes(raw)
        assert result[0] == "I10", "Code should be trimmed and uppercased"
        assert result[2] == "J44.1", "Code should be trimmed and uppercased"

    def test_code_to_chapter(self):
        """Spot-check code_to_chapter for a handful of ICD-10 codes."""
        norm = ICDNormaliser()
        # I10 — Chapter IX: Diseases of the circulatory system
        ch_i10 = norm.code_to_chapter("I10")
        assert ch_i10 is not None, "I10 should map to a known chapter"
        # E11.9 — Chapter IV: Endocrine / metabolic
        ch_e11 = norm.code_to_chapter("E11.9")
        assert ch_e11 is not None, "E11.9 should map to a known chapter"


# ---------------------------------------------------------------------------
# TestCohortBuilder
# ---------------------------------------------------------------------------


class TestCohortBuilder:
    """Tests for CohortBuilder."""

    def test_age_range(self, sample_patient_df):
        """age_range keeps only rows where age is in [30, 60]."""
        cb = CohortBuilder(verbose=False)
        result = cb.age_range(sample_patient_df, min_age=30, max_age=60)
        assert result["age"].between(30, 60).all(), "All ages must be in [30, 60]"

    def test_diagnosis_filter_include(self, sample_patient_df):
        """diagnosis_filter with include_codes returns a subset."""
        cb = CohortBuilder(verbose=False)
        before = len(sample_patient_df)
        result = cb.diagnosis_filter(
            sample_patient_df,
            diagnoses_col="diagnoses",
            include_codes=["I10"],
        )
        assert len(result) <= before, "Filtered result cannot exceed original size"
        assert len(result) > 0, "At least one patient should have I10"

    def test_los_filter(self, sample_patient_df):
        """los_filter with min_los=5 removes rows with los_days < 5."""
        cb = CohortBuilder(verbose=False)
        result = cb.los_filter(sample_patient_df, min_los=5, los_col="los_days")
        assert (result["los_days"] >= 5).all(), "All remaining LOS must be >= 5"

    def test_date_range(self, sample_patient_df):
        """date_range returns only rows within the specified window."""
        cb = CohortBuilder(verbose=False)
        result = cb.date_range(
            sample_patient_df,
            date_col="admission_date",
            start="2020-01-01",
            end="2021-12-31",
        )
        dates = pd.to_datetime(result["admission_date"])
        assert (dates >= pd.Timestamp("2020-01-01")).all()
        assert (dates <= pd.Timestamp("2021-12-31")).all()

    def test_build_survival_cohort(self, sample_patient_df):
        """build_survival_cohort returns rows with time>0 and event in {0,1}."""
        cb = CohortBuilder(verbose=False)
        df = sample_patient_df.copy()
        df["survival_time"] = df["los_days"].clip(lower=1)
        df["event"] = df["mortality_30d"]
        result = cb.build_survival_cohort(df, time_col="survival_time", event_col="event")
        assert (result["survival_time"] > 0).all(), "Survival time must be > 0"
        assert result["event"].isin([0, 1]).all(), "Event indicator must be 0 or 1"

    def test_summary(self, sample_patient_df):
        """summary returns a dict containing at minimum n_rows and n_cols."""
        cb = CohortBuilder(verbose=False)
        s = cb.summary(sample_patient_df)
        assert isinstance(s, dict), "summary must return a dict"
        assert "n_rows" in s, "'n_rows' key must be present"
        assert "n_cols" in s, "'n_cols' key must be present"
        assert s["n_rows"] == len(sample_patient_df)


# ---------------------------------------------------------------------------
# TestDataSplitter
# ---------------------------------------------------------------------------


class TestDataSplitter:
    """Tests for DataSplitter."""

    def test_random_split_sizes(self, sample_patient_df):
        """Random split produces ~70/10/20 row counts from 100-row input."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1, strategy="random")
        train, val, test = splitter.split(sample_patient_df, target_col="mortality_30d")
        total = len(train) + len(val) + len(test)
        assert total == len(sample_patient_df), "All rows must be accounted for"
        assert len(train) >= 60, f"Train too small: {len(train)}"
        assert len(val) >= 5, f"Val too small: {len(val)}"
        assert len(test) >= 10, f"Test too small: {len(test)}"

    def test_temporal_split_no_leakage(self, sample_patient_df):
        """Temporal split ensures max(train dates) <= min(test dates)."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.temporal_split(
            sample_patient_df, timestamp_col="admission_date"
        )
        train_max = pd.to_datetime(train["admission_date"]).max()
        test_min = pd.to_datetime(test["admission_date"]).min()
        assert train_max <= test_min, (
            f"Temporal leakage: train max {train_max} > test min {test_min}"
        )

    def test_stratified_split_ratio(self, sample_patient_df):
        """Stratified split preserves class ratio within 15 percentage points."""
        splitter = DataSplitter(test_size=0.2, val_size=0.1)
        train, val, test = splitter.stratified_split(
            sample_patient_df, target_col="mortality_30d"
        )
        overall_rate = sample_patient_df["mortality_30d"].mean()
        test_rate = test["mortality_30d"].mean()
        assert abs(test_rate - overall_rate) < 0.15, (
            f"Stratification off: overall={overall_rate:.3f}, test={test_rate:.3f}"
        )

    def test_get_fold_indices_kfold(self, sample_patient_df):
        """get_fold_indices with 5 folds returns 5 non-overlapping tuples."""
        splitter = DataSplitter()
        folds = splitter.get_fold_indices(sample_patient_df, n_splits=5, strategy="kfold")
        assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
        all_val_indices: set = set()
        for train_idx, val_idx in folds:
            assert len(np.intersect1d(train_idx, val_idx)) == 0, "Train/val overlap"
            all_val_indices.update(val_idx.tolist())
        assert len(all_val_indices) == len(sample_patient_df), "All rows covered by val folds"


# ---------------------------------------------------------------------------
# TestSchemaValidator
# ---------------------------------------------------------------------------


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_validate_patient_df_valid(self, sample_patient_df):
        """A custom schema that matches the fixture passes validation."""
        sv = SchemaValidator()
        # Build a schema that matches the conftest fixture columns
        from data.processing.validator import _col
        schema = {
            "subject_id": _col(required=True, dtype="numeric", unique=True),
            "age": _col(required=True, dtype="numeric", min=0.0, max=130.0),
            "gender": _col(required=True, dtype="object"),
            "los_days": _col(required=True, dtype="numeric", min=0.0),
            "mortality_30d": _col(required=False, dtype="numeric"),
        }
        result = sv.validate(sample_patient_df, schema)
        assert isinstance(result, ValidationResult), "Expected ValidationResult"
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_validate_hospital_df_valid(self, sample_hospital_df):
        """A custom schema matching the hospital fixture passes validation."""
        sv = SchemaValidator()
        from data.processing.validator import _col
        schema = {
            "hospital_id": _col(required=True, dtype="object", unique=True),
            "state": _col(required=True, dtype="object"),
            "bed_count": _col(required=True, dtype="numeric", min=1.0),
            "operating_margin": _col(required=False, dtype="numeric"),
            "default_probability": _col(required=False, dtype="numeric", min=0.0, max=1.0),
        }
        result = sv.validate(sample_hospital_df, schema)
        assert isinstance(result, ValidationResult), "Expected ValidationResult"
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_validate_missing_column(self, sample_patient_df):
        """Dropping a required column produces at least one error."""
        sv = SchemaValidator()
        from data.processing.validator import _col
        schema = {
            "subject_id": _col(required=True, dtype="numeric", unique=True),
            "age": _col(required=True, dtype="numeric", min=0.0, max=130.0),
            "los_days": _col(required=True, dtype="numeric", min=0.0),
        }
        # Drop a required column
        df_bad = sample_patient_df.drop(columns=["subject_id"])
        result = sv.validate(df_bad, schema)
        assert not result.is_valid, "Validation should fail when required column is missing"
        assert len(result.errors) > 0, "Expected at least one error"

    def test_validate_no_leakage_clean(self, sample_patient_df):
        """Disjoint subject_id sets pass the no-leakage check."""
        sv = SchemaValidator()
        n = len(sample_patient_df)
        train = sample_patient_df.iloc[: n // 2].copy()
        test = sample_patient_df.iloc[n // 2 :].copy()
        assert sv.validate_no_leakage(train, test, id_col="subject_id") is True

    def test_validate_no_leakage_overlap(self, sample_patient_df):
        """Overlapping subject_id sets fail the no-leakage check."""
        sv = SchemaValidator()
        train = sample_patient_df.copy()
        test = sample_patient_df.copy()
        assert sv.validate_no_leakage(train, test, id_col="subject_id") is False
