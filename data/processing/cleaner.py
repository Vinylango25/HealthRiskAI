"""
data.processing.cleaner — Clinical data cleaning pipeline.

Provides :class:`DataCleaner`, an opinionated, production-grade cleaning
utility for patient and hospital DataFrames sourced from MIMIC-IV, CMS, and
other health data repositories.

Responsibilities
----------------
- Missing-value imputation (median / mode / constant / forward-fill)
- Duplicate record removal
- Numeric outlier clipping (IQR or z-score)
- Schema-driven dtype casting
- End-to-end ``clean_patient_df`` pipeline
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Columns that are always treated as categorical when auto-detecting dtype.
_ALWAYS_CATEGORICAL: frozenset[str] = frozenset(
    {
        "gender",
        "sex",
        "race",
        "ethnicity",
        "admission_type",
        "discharge_disposition",
        "insurance",
        "marital_status",
        "language",
        "religion",
    }
)

#: Columns that are always treated as numeric when auto-detecting dtype.
_ALWAYS_NUMERIC: frozenset[str] = frozenset(
    {
        "age",
        "los",
        "length_of_stay",
        "num_diagnoses",
        "num_procedures",
        "icu_los",
        "weight_kg",
        "height_cm",
        "bmi",
    }
)

#: Patient-level dtype schema used by ``clean_patient_df``.
PATIENT_DTYPE_SCHEMA: Dict[str, str] = {
    "subject_id": "int64",
    "hadm_id": "int64",
    "age": "float64",
    "gender": "category",
    "los": "float64",
    "num_diagnoses": "Int64",
    "num_procedures": "Int64",
    "insurance": "category",
    "marital_status": "category",
    "ethnicity": "category",
    "admission_type": "category",
    "discharge_disposition": "category",
    "hospital_expire_flag": "Int64",
    "readmit_30d": "Int64",
}


# ---------------------------------------------------------------------------
# DataCleaner
# ---------------------------------------------------------------------------


class DataCleaner:
    """Production-grade cleaning utility for clinical DataFrames.

    All public methods accept a :class:`pandas.DataFrame` and return a
    *new* DataFrame — the original is never mutated.

    Parameters
    ----------
    verbose:
        When *True*, INFO-level log messages are emitted for every
        significant operation performed.

    Examples
    --------
    >>> cleaner = DataCleaner()
    >>> df_clean = cleaner.clean_patient_df(raw_df)
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str, *args: Any) -> None:
        if self.verbose:
            logger.info(msg, *args)

    @staticmethod
    def _infer_numeric_cols(df: pd.DataFrame) -> List[str]:
        """Return column names that are numeric (float/int) in *df*."""
        return df.select_dtypes(include=[np.number]).columns.tolist()

    @staticmethod
    def _infer_categorical_cols(df: pd.DataFrame) -> List[str]:
        """Return column names that are object or category in *df*."""
        return df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_missing(
        self,
        df: pd.DataFrame,
        strategy: str = "median",
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        fill_value: Any = "Unknown",
    ) -> pd.DataFrame:
        """Impute missing values in *df*.

        Parameters
        ----------
        df:
            Input DataFrame.
        strategy:
            Imputation strategy for **numeric** columns.
            One of ``'median'``, ``'mean'``, ``'zero'``, ``'ffill'``,
            ``'bfill'``.
        numeric_cols:
            Explicit list of numeric columns to impute.  When *None*,
            all numeric columns are selected automatically.
        categorical_cols:
            Explicit list of categorical / object columns to impute.
            When *None*, all object / category columns are selected.
            These always receive ``fill_value`` regardless of *strategy*.
        fill_value:
            Constant used to fill categorical nulls (default ``'Unknown'``).

        Returns
        -------
        pd.DataFrame
            A copy of *df* with missing values imputed.

        Raises
        ------
        ValueError
            If *strategy* is not one of the supported options.
        """
        supported = {"median", "mean", "zero", "ffill", "bfill"}
        if strategy not in supported:
            raise ValueError(
                f"Unknown strategy '{strategy}'.  Choose from {supported}."
            )

        df = df.copy()

        num_cols: List[str] = (
            numeric_cols if numeric_cols is not None else self._infer_numeric_cols(df)
        )
        cat_cols: List[str] = (
            categorical_cols
            if categorical_cols is not None
            else self._infer_categorical_cols(df)
        )

        # --- numeric imputation ---
        missing_before = df[num_cols].isna().sum().sum()
        if strategy == "median":
            for col in num_cols:
                if df[col].isna().any():
                    fill = df[col].median()
                    df[col] = df[col].fillna(fill)
        elif strategy == "mean":
            for col in num_cols:
                if df[col].isna().any():
                    fill = df[col].mean()
                    df[col] = df[col].fillna(fill)
        elif strategy == "zero":
            df[num_cols] = df[num_cols].fillna(0)
        elif strategy == "ffill":
            df[num_cols] = df[num_cols].fillna(method="ffill")
        elif strategy == "bfill":
            df[num_cols] = df[num_cols].fillna(method="bfill")

        missing_after = df[num_cols].isna().sum().sum()
        self._log(
            "handle_missing [numeric/%s]: %d → %d nulls across %d cols",
            strategy,
            missing_before,
            missing_after,
            len(num_cols),
        )

        # --- categorical imputation ---
        cat_missing_before = df[cat_cols].isna().sum().sum()
        for col in cat_cols:
            if df[col].isna().any():
                # For category dtype we need to add the new category first.
                if hasattr(df[col], "cat"):
                    if fill_value not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories(fill_value)
                df[col] = df[col].fillna(fill_value)
        cat_missing_after = df[cat_cols].isna().sum().sum()
        self._log(
            "handle_missing [categorical/'%s']: %d → %d nulls across %d cols",
            fill_value,
            cat_missing_before,
            cat_missing_after,
            len(cat_cols),
        )

        return df

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[Sequence[str]] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """Remove duplicate rows from *df*.

        Parameters
        ----------
        df:
            Input DataFrame.
        subset:
            Column(s) to consider for identifying duplicates.  *None*
            uses all columns (pandas default).
        keep:
            Which duplicate to keep: ``'first'``, ``'last'``, or
            ``False`` (drop all).

        Returns
        -------
        pd.DataFrame
            Deduplicated copy of *df*.
        """
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        removed = before - len(df)
        self._log(
            "remove_duplicates: removed %d / %d rows (subset=%s)",
            removed,
            before,
            subset,
        )
        return df.reset_index(drop=True)

    def clip_outliers(
        self,
        df: pd.DataFrame,
        cols: Sequence[str],
        method: str = "iqr",
        multiplier: float = 1.5,
    ) -> pd.DataFrame:
        """Clip outliers in numeric columns using IQR fences or z-score.

        Values outside the computed bounds are *clipped* (not dropped),
        preserving the number of rows.

        Parameters
        ----------
        df:
            Input DataFrame.
        cols:
            List of numeric column names to clip.
        method:
            ``'iqr'``  — fences = Q1 - *multiplier*·IQR, Q3 + *multiplier*·IQR.
            ``'zscore'`` — fences = μ ± *multiplier*·σ.
        multiplier:
            Scaling factor for the fence.  Default ``1.5`` (Tukey fences).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with clipped values.

        Raises
        ------
        ValueError
            If *method* is not ``'iqr'`` or ``'zscore'``.
        """
        if method not in {"iqr", "zscore"}:
            raise ValueError(f"method must be 'iqr' or 'zscore', got '{method}'.")

        df = df.copy()
        total_clipped = 0

        for col in cols:
            if col not in df.columns:
                logger.warning("clip_outliers: column '%s' not found — skipped.", col)
                continue

            series = df[col].dropna()

            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lo = q1 - multiplier * iqr
                hi = q3 + multiplier * iqr
            else:  # zscore
                mu = series.mean()
                sigma = series.std(ddof=0)
                lo = mu - multiplier * sigma
                hi = mu + multiplier * sigma

            clipped = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            total_clipped += clipped

        self._log(
            "clip_outliers [%s × %.2f]: %d values clipped across %d cols",
            method,
            multiplier,
            total_clipped,
            len(cols),
        )
        return df

    def fix_dtypes(
        self,
        df: pd.DataFrame,
        schema: Dict[str, str],
    ) -> pd.DataFrame:
        """Cast columns to the dtypes specified in *schema*.

        Columns absent from *df* are silently skipped.  Conversion
        errors for individual cells are coerced to ``NaN`` / ``NaT``.

        Parameters
        ----------
        df:
            Input DataFrame.
        schema:
            Mapping of ``{column_name: dtype_string}``.  Dtype strings
            are any value accepted by :func:`pandas.api.types.pandas_dtype`,
            plus ``'category'`` and nullable integer types like ``'Int64'``.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with columns recast.

        Examples
        --------
        >>> cleaner.fix_dtypes(df, {"age": "float64", "gender": "category"})
        """
        df = df.copy()
        for col, dtype in schema.items():
            if col not in df.columns:
                logger.debug("fix_dtypes: column '%s' not in DataFrame — skipped.", col)
                continue
            try:
                if dtype in ("int64", "Int64", "int32", "Int32"):
                    # Use pandas nullable integer to survive NaNs.
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                        dtype if dtype.startswith("I") else pd.Int64Dtype()
                        if dtype == "int64"
                        else dtype
                    )
                elif dtype.startswith("float"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                elif dtype == "category":
                    df[col] = df[col].astype("category")
                elif dtype in ("datetime64[ns]", "datetime64"):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif dtype == "bool":
                    df[col] = df[col].astype(bool)
                else:
                    df[col] = df[col].astype(dtype)
                logger.debug("fix_dtypes: cast '%s' → %s", col, dtype)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "fix_dtypes: failed to cast '%s' → %s: %s", col, dtype, exc
                )
        return df

    def clean_patient_df(
        self,
        df: pd.DataFrame,
        outlier_cols: Optional[List[str]] = None,
        id_subset: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Opinionated end-to-end cleaning pipeline for patient DataFrames.

        The pipeline applies the following steps in order:

        1. **Dtype fixing** — casts known columns using
           :data:`PATIENT_DTYPE_SCHEMA`.
        2. **Duplicate removal** — deduplicates on ``subject_id`` +
           ``hadm_id`` when both columns exist; falls back to *id_subset*
           or all columns.
        3. **Outlier clipping** — IQR clipping on numeric lab/vital
           columns (defaults to ``age`` and ``los`` if present).
        4. **Missing-value imputation** — median for numeric, 'Unknown'
           for categorical.

        Parameters
        ----------
        df:
            Raw patient-level DataFrame.
        outlier_cols:
            Override the default list of columns to clip for outliers.
        id_subset:
            Columns to use for duplicate detection when ``subject_id``
            / ``hadm_id`` are absent.

        Returns
        -------
        pd.DataFrame
            Cleaned patient DataFrame.
        """
        self._log("clean_patient_df: starting pipeline on %d rows × %d cols", *df.shape)

        # Step 1 — dtype casting
        df = self.fix_dtypes(df, PATIENT_DTYPE_SCHEMA)

        # Step 2 — deduplication
        dup_keys: Optional[List[str]] = None
        if "subject_id" in df.columns and "hadm_id" in df.columns:
            dup_keys = ["subject_id", "hadm_id"]
        elif id_subset is not None:
            dup_keys = id_subset
        df = self.remove_duplicates(df, subset=dup_keys)

        # Step 3 — outlier clipping
        if outlier_cols is None:
            default_clip = ["age", "los", "length_of_stay", "bmi", "weight_kg"]
            outlier_cols = [c for c in default_clip if c in df.columns]
        if outlier_cols:
            df = self.clip_outliers(df, cols=outlier_cols, method="iqr", multiplier=1.5)

        # Step 4 — missing-value imputation
        df = self.handle_missing(df, strategy="median")

        self._log("clean_patient_df: pipeline complete → %d rows × %d cols", *df.shape)
        return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    rng = np.random.default_rng(42)
    n = 200

    raw = pd.DataFrame(
        {
            "subject_id": np.arange(1, n + 1),
            "hadm_id": np.arange(1001, 1001 + n),
            "age": rng.uniform(18, 95, n),
            "gender": rng.choice(["M", "F", None], n),
            "los": rng.exponential(5, n),
            "num_diagnoses": rng.integers(1, 30, n).astype(float),
            "insurance": rng.choice(["Medicare", "Medicaid", "Private", None], n),
            "hospital_expire_flag": rng.integers(0, 2, n),
        }
    )

    # Inject some missing values and outliers
    raw.loc[rng.choice(n, 20, replace=False), "age"] = np.nan
    raw.loc[rng.choice(n, 10, replace=False), "los"] = 999.0  # extreme outlier
    raw = pd.concat([raw, raw.iloc[:5]], ignore_index=True)   # add duplicates

    cleaner = DataCleaner(verbose=True)
    clean = cleaner.clean_patient_df(raw)

    print(f"\nRaw:   {raw.shape[0]} rows")
    print(f"Clean: {clean.shape[0]} rows")
    print("\nMissing values after cleaning:")
    print(clean.isna().sum()[clean.isna().sum() > 0])
    print("\ndtypes:")
    print(clean.dtypes)
    print("\nAge stats (post-clip):")
    print(clean["age"].describe())
