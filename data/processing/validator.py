"""
data.processing.validator — Schema validation for clinical DataFrames.

Provides :class:`SchemaValidator` and the :class:`ValidationResult` dataclass.

Validation checks
-----------------
- Required columns present
- Correct dtypes (coercible or exact)
- Value range constraints (min / max)
- Allowed value sets for categorical columns
- Null-rate thresholds
- Unique-key constraints
- No train/test leakage on a specified ID column

:class:`ValidationResult` captures blocking **errors** and non-blocking
**warnings** separately so that callers can decide whether to abort or
proceed with caution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a schema validation run.

    Attributes
    ----------
    is_valid:
        ``True`` when no blocking *errors* were found.
    errors:
        Blocking issues that should prevent downstream use of the DataFrame.
    warnings:
        Non-blocking issues that may indicate data quality problems.
    n_rows:
        Number of rows in the validated DataFrame.
    n_cols:
        Number of columns in the validated DataFrame.
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0

    def __str__(self) -> str:  # noqa: D401
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"ValidationResult [{status}] — {self.n_rows} rows × {self.n_cols} cols",
            f"  Errors   ({len(self.errors)}): "
            + (", ".join(self.errors) if self.errors else "none"),
            f"  Warnings ({len(self.warnings)}): "
            + (", ".join(self.warnings) if self.warnings else "none"),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema definition helpers
# ---------------------------------------------------------------------------

# Column spec dict keys:
#   required  : bool   — if True, column must be present
#   dtype     : str    — expected pandas dtype family ('numeric', 'category',
#                        'object', 'datetime', 'bool', or exact dtype string)
#   min       : float  — minimum allowed value (numeric columns)
#   max       : float  — maximum allowed value (numeric columns)
#   allowed   : list   — exhaustive set of allowed values (categorical)
#   max_null_rate : float — maximum fraction of nulls tolerated (0–1)
#   unique    : bool   — if True, all values must be unique


def _col(
    required: bool = True,
    dtype: str = "numeric",
    min: Optional[float] = None,
    max: Optional[float] = None,
    allowed: Optional[List[Any]] = None,
    max_null_rate: float = 1.0,
    unique: bool = False,
) -> Dict[str, Any]:
    """Convenience factory for a column-spec dict."""
    return {
        "required": required,
        "dtype": dtype,
        "min": min,
        "max": max,
        "allowed": allowed,
        "max_null_rate": max_null_rate,
        "unique": unique,
    }


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Validate patient and hospital DataFrames against expected schemas.

    Parameters
    ----------
    strict:
        When *True*, dtype mismatches are treated as **errors** rather than
        warnings.  Default ``False`` (lenient — coercible dtypes pass with
        a warning).

    Examples
    --------
    >>> validator = SchemaValidator()
    >>> result = validator.validate_patient_df(df)
    >>> if not result.is_valid:
    ...     raise ValueError(f"Invalid data: {result.errors}")
    """

    # ------------------------------------------------------------------
    # Class-level schemas
    # ------------------------------------------------------------------

    PATIENT_SCHEMA: ClassVar[Dict[str, Dict[str, Any]]] = {
        "subject_id": _col(required=True, dtype="numeric", unique=True, max_null_rate=0.0),
        "hadm_id": _col(required=True, dtype="numeric", unique=True, max_null_rate=0.0),
        "age": _col(required=True, dtype="numeric", min=0.0, max=130.0, max_null_rate=0.05),
        "gender": _col(
            required=True,
            dtype="category",
            allowed=["M", "F", "m", "f", "Male", "Female", "male", "female", "Unknown"],
            max_null_rate=0.02,
        ),
        "los": _col(required=True, dtype="numeric", min=0.0, max=365.0, max_null_rate=0.05),
        "hospital_expire_flag": _col(
            required=False,
            dtype="numeric",
            allowed=[0, 1],
            max_null_rate=0.10,
        ),
        "readmit_30d": _col(
            required=False,
            dtype="numeric",
            allowed=[0, 1],
            max_null_rate=0.15,
        ),
        "insurance": _col(
            required=False,
            dtype="category",
            allowed=["Medicare", "Medicaid", "Private", "Self Pay", "Government", "Unknown"],
            max_null_rate=0.10,
        ),
        "ethnicity": _col(required=False, dtype="category", max_null_rate=0.20),
        "num_diagnoses": _col(required=False, dtype="numeric", min=0.0, max_null_rate=0.10),
        "num_procedures": _col(required=False, dtype="numeric", min=0.0, max_null_rate=0.10),
    }

    HOSPITAL_SCHEMA: ClassVar[Dict[str, Dict[str, Any]]] = {
        "provider_id": _col(required=True, dtype="object", unique=True, max_null_rate=0.0),
        "hospital_name": _col(required=True, dtype="object", max_null_rate=0.0),
        "state": _col(required=True, dtype="object", max_null_rate=0.0),
        "beds": _col(required=False, dtype="numeric", min=1.0, max=5000.0, max_null_rate=0.05),
        "total_discharges": _col(required=False, dtype="numeric", min=0.0, max_null_rate=0.10),
        "average_covered_charges": _col(
            required=False, dtype="numeric", min=0.0, max_null_rate=0.15
        ),
        "average_total_payments": _col(
            required=False, dtype="numeric", min=0.0, max_null_rate=0.15
        ),
        "average_medicare_payments": _col(
            required=False, dtype="numeric", min=0.0, max_null_rate=0.15
        ),
        "cms_certification_number": _col(
            required=False, dtype="object", max_null_rate=0.05
        ),
        "hospital_type": _col(
            required=False,
            dtype="category",
            allowed=[
                "Acute Care Hospitals",
                "Critical Access Hospitals",
                "Childrens",
                "Psychiatric",
                "Other",
            ],
            max_null_rate=0.10,
        ),
    }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def _dtype_family(self, series: pd.Series) -> str:
        """Return a normalised dtype family string."""
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_bool_dtype(series):
            return "bool"
        if isinstance(series.dtype, pd.CategoricalDtype):
            return "category"
        return "object"

    def _check_column(
        self,
        df: pd.DataFrame,
        col: str,
        spec: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate a single column against its spec, appending to errors/warnings."""
        # Required presence
        if col not in df.columns:
            if spec["required"]:
                errors.append(f"Required column '{col}' is missing.")
            return  # Nothing else to check if absent

        series = df[col]

        # --- Null rate ---
        null_rate = series.isna().mean()
        if null_rate > spec["max_null_rate"]:
            msg = (
                f"Column '{col}' null rate {null_rate:.1%} exceeds "
                f"threshold {spec['max_null_rate']:.1%}."
            )
            if spec["required"] and spec["max_null_rate"] == 0.0:
                errors.append(msg)
            else:
                warnings.append(msg)

        # --- Dtype ---
        actual_family = self._dtype_family(series)
        expected_family = spec["dtype"]
        dtype_ok = actual_family == expected_family or (
            # numeric covers int and float
            expected_family == "numeric" and actual_family == "numeric"
        ) or (
            # category and object are often interchangeable
            expected_family == "category" and actual_family in ("category", "object")
        )
        if not dtype_ok:
            msg = (
                f"Column '{col}' has dtype family '{actual_family}'; "
                f"expected '{expected_family}'."
            )
            if self.strict:
                errors.append(msg)
            else:
                warnings.append(msg)

        # --- Uniqueness ---
        if spec["unique"] and series.dropna().duplicated().any():
            n_dup = series.dropna().duplicated().sum()
            errors.append(
                f"Column '{col}' must be unique but has {n_dup} duplicate values."
            )

        # --- Numeric range ---
        if spec["dtype"] == "numeric" and pd.api.types.is_numeric_dtype(series):
            if spec["min"] is not None:
                below = (series < spec["min"]).sum()
                if below > 0:
                    warnings.append(
                        f"Column '{col}': {below} values below minimum {spec['min']}."
                    )
            if spec["max"] is not None:
                above = (series > spec["max"]).sum()
                if above > 0:
                    warnings.append(
                        f"Column '{col}': {above} values above maximum {spec['max']}."
                    )

        # --- Allowed values ---
        if spec["allowed"] is not None:
            non_null = series.dropna()
            if len(non_null) > 0:
                # For numeric, compare as values; for others, compare as strings
                if pd.api.types.is_numeric_dtype(non_null):
                    unexpected = ~non_null.isin(spec["allowed"])
                else:
                    unexpected = ~non_null.astype(str).isin(
                        [str(v) for v in spec["allowed"]]
                    )
                n_unexpected = unexpected.sum()
                if n_unexpected > 0:
                    sample = non_null[unexpected].unique()[:5].tolist()
                    warnings.append(
                        f"Column '{col}': {n_unexpected} values not in allowed set. "
                        f"Sample: {sample}"
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Dict[str, Any]],
    ) -> ValidationResult:
        """Validate *df* against *schema*.

        Parameters
        ----------
        df:
            DataFrame to validate.
        schema:
            Column specification dict.  Use the class-level
            :attr:`PATIENT_SCHEMA` / :attr:`HOSPITAL_SCHEMA` or a custom
            mapping built with :func:`_col`.

        Returns
        -------
        ValidationResult
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(df, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                errors=["Input is not a pandas DataFrame."],
                n_rows=0,
                n_cols=0,
            )

        if df.empty:
            warnings.append("DataFrame is empty (0 rows).")

        for col, spec in schema.items():
            self._check_column(df, col, spec, errors, warnings)

        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            n_rows=len(df),
            n_cols=df.shape[1],
        )

        if is_valid:
            logger.info(
                "validate: PASSED — %d rows, %d cols, %d warning(s)",
                result.n_rows, result.n_cols, len(warnings),
            )
        else:
            logger.warning(
                "validate: FAILED — %d error(s), %d warning(s)",
                len(errors), len(warnings),
            )
        return result

    def validate_patient_df(self, df: pd.DataFrame) -> ValidationResult:
        """Validate *df* against the built-in :attr:`PATIENT_SCHEMA`.

        Parameters
        ----------
        df:
            Patient-level DataFrame.

        Returns
        -------
        ValidationResult
        """
        logger.info("validate_patient_df: starting validation.")
        return self.validate(df, self.PATIENT_SCHEMA)

    def validate_hospital_df(self, df: pd.DataFrame) -> ValidationResult:
        """Validate *df* against the built-in :attr:`HOSPITAL_SCHEMA`.

        Parameters
        ----------
        df:
            Hospital-level DataFrame.

        Returns
        -------
        ValidationResult
        """
        logger.info("validate_hospital_df: starting validation.")
        return self.validate(df, self.HOSPITAL_SCHEMA)

    def validate_no_leakage(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        id_col: str,
    ) -> bool:
        """Check that no IDs from *df_test* appear in *df_train*.

        Parameters
        ----------
        df_train:
            Training set DataFrame.
        df_test:
            Test set DataFrame.
        id_col:
            Column holding entity ID (e.g. ``'subject_id'``).

        Returns
        -------
        bool
            ``True`` if no overlap is found (no leakage), ``False`` otherwise.

        Raises
        ------
        KeyError
            If *id_col* is absent from either DataFrame.
        """
        for name, df in [("df_train", df_train), ("df_test", df_test)]:
            if id_col not in df.columns:
                raise KeyError(f"validate_no_leakage: '{id_col}' not in {name}.")

        train_ids: Set = set(df_train[id_col].dropna().unique())
        test_ids: Set = set(df_test[id_col].dropna().unique())
        overlap = train_ids & test_ids

        if overlap:
            logger.error(
                "validate_no_leakage: DATA LEAKAGE DETECTED — %d IDs appear in both "
                "train and test sets.  Sample: %s",
                len(overlap),
                list(overlap)[:5],
            )
            return False

        logger.info(
            "validate_no_leakage: PASSED — train=%d unique IDs, test=%d unique IDs, "
            "overlap=0.",
            len(train_ids),
            len(test_ids),
        )
        return True


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    rng = np.random.default_rng(0)
    n = 300

    # --- Build a mostly-valid patient DataFrame ---
    patient_df = pd.DataFrame(
        {
            "subject_id": np.arange(1, n + 1),
            "hadm_id": np.arange(1001, 1001 + n),
            "age": rng.uniform(18, 90, n),
            "gender": rng.choice(["M", "F"], n),
            "los": rng.exponential(4, n),
            "hospital_expire_flag": rng.integers(0, 2, n),
            "readmit_30d": rng.integers(0, 2, n),
            "insurance": rng.choice(["Medicare", "Medicaid", "Private"], n),
            "ethnicity": rng.choice(["White", "Black", "Hispanic", "Asian", "Other"], n),
        }
    )

    # Inject some problems
    patient_df.loc[0:4, "age"] = -5           # below minimum
    patient_df.loc[5:9, "gender"] = None      # nulls
    patient_df.loc[10, "subject_id"] = 5      # duplicate ID

    validator = SchemaValidator(strict=False)

    print("=== validate_patient_df ===")
    result = validator.validate_patient_df(patient_df)
    print(result)

    # --- Hospital DataFrame ---
    hosp_df = pd.DataFrame(
        {
            "provider_id": [f"H{i:04d}" for i in range(50)],
            "hospital_name": [f"Hospital {i}" for i in range(50)],
            "state": rng.choice(["CA", "NY", "TX", "FL"], 50),
            "beds": rng.integers(10, 800, 50).astype(float),
            "total_discharges": rng.integers(100, 5000, 50).astype(float),
            "average_covered_charges": rng.uniform(5000, 150000, 50),
            "average_total_payments": rng.uniform(3000, 80000, 50),
        }
    )

    print("\n=== validate_hospital_df ===")
    h_result = validator.validate_hospital_df(hosp_df)
    print(h_result)

    # --- Leakage check ---
    train_df = patient_df.iloc[:200]
    test_df_clean = patient_df.iloc[200:]
    test_df_leak = patient_df.iloc[190:]  # 10-row overlap

    print("\n=== validate_no_leakage ===")
    print(f"  No overlap: {validator.validate_no_leakage(train_df, test_df_clean, 'subject_id')}")
    print(f"  With overlap: {validator.validate_no_leakage(train_df, test_df_leak, 'subject_id')}")
