"""
data.processing.cohort — Patient cohort construction and filtering.

Provides :class:`CohortBuilder`, a declarative API for applying inclusion /
exclusion criteria to patient DataFrames and producing analysis-ready cohorts
for clinical ML tasks including survival analysis.

Design philosophy
-----------------
- Every method accepts and returns a ``pd.DataFrame`` — methods are composable.
- Filters are non-destructive (always return copies).
- ``from_patient_df`` supports a declarative ``filters`` list for pipeline
  configuration via YAML / JSON.
- ``summary`` always returns a consistent dict structure regardless of which
  demographic columns are present.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FilterSpec = Dict[str, Any]  # e.g. {"type": "age_range", "min_age": 18, "max_age": 90}


# ---------------------------------------------------------------------------
# CohortBuilder
# ---------------------------------------------------------------------------


class CohortBuilder:
    """Build analysis-ready patient cohorts with declarative filter pipelines.

    All filtering methods return a *new* DataFrame; the source is never
    mutated.  Methods can be chained manually or driven via the
    :meth:`from_patient_df` declarative interface.

    Parameters
    ----------
    verbose:
        Emit INFO-level log messages after each filter step.

    Examples
    --------
    >>> cb = CohortBuilder()
    >>> cohort = (
    ...     cb.age_range(df, min_age=18, max_age=80)
    ...       .pipe(cb.los_filter, min_los=1)
    ... )
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_filter(
        self,
        method: str,
        before: int,
        after: int,
        **kwargs: Any,
    ) -> None:
        if self.verbose:
            removed = before - after
            logger.info(
                "%s: %d → %d rows (removed %d, kept %.1f%%) | params=%s",
                method,
                before,
                after,
                removed,
                100.0 * after / before if before else 0,
                kwargs,
            )

    # ------------------------------------------------------------------
    # Declarative interface
    # ------------------------------------------------------------------

    def from_patient_df(
        self,
        df: pd.DataFrame,
        filters: List[FilterSpec],
    ) -> pd.DataFrame:
        """Apply a list of filter specifications to *df*.

        Each specification is a dict with a ``'type'`` key.  Remaining
        keys are forwarded as keyword arguments to the corresponding
        method.

        Supported types
        ---------------
        ``age_range``
            Keys: ``min_age``, ``max_age``, ``age_col`` (default ``'age'``).
        ``diagnosis_filter``
            Keys: ``diagnoses_col``, ``include_codes``, ``exclude_codes``.
        ``los_filter``
            Keys: ``min_los``, ``max_los``, ``los_col`` (default ``'los'``).
        ``date_range``
            Keys: ``date_col``, ``start``, ``end``.

        Parameters
        ----------
        df:
            Input patient DataFrame.
        filters:
            Ordered list of filter specification dicts.

        Returns
        -------
        pd.DataFrame
            Filtered cohort.

        Raises
        ------
        ValueError
            If a filter ``'type'`` is not recognised.

        Examples
        --------
        >>> filters = [
        ...     {"type": "age_range", "min_age": 18, "max_age": 80},
        ...     {"type": "los_filter", "min_los": 1},
        ... ]
        >>> cohort = cb.from_patient_df(df, filters)
        """
        logger.info(
            "from_patient_df: applying %d filter(s) to %d rows", len(filters), len(df)
        )
        for spec in filters:
            spec = dict(spec)  # make a copy so we don't mutate caller's dict
            filter_type = spec.pop("type")

            if filter_type == "age_range":
                df = self.age_range(df, **spec)
            elif filter_type == "diagnosis_filter":
                df = self.diagnosis_filter(df, **spec)
            elif filter_type == "los_filter":
                df = self.los_filter(df, **spec)
            elif filter_type == "date_range":
                df = self.date_range(df, **spec)
            else:
                raise ValueError(
                    f"Unknown filter type '{filter_type}'.  "
                    "Supported: age_range, diagnosis_filter, los_filter, date_range."
                )

        logger.info("from_patient_df: final cohort = %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Filter methods
    # ------------------------------------------------------------------

    def age_range(
        self,
        df: pd.DataFrame,
        min_age: Optional[float] = None,
        max_age: Optional[float] = None,
        age_col: str = "age",
    ) -> pd.DataFrame:
        """Restrict cohort to patients within an age range.

        Parameters
        ----------
        df:
            Input DataFrame.
        min_age:
            Minimum age (inclusive).  *None* = no lower bound.
        max_age:
            Maximum age (inclusive).  *None* = no upper bound.
        age_col:
            Column name holding patient age.

        Returns
        -------
        pd.DataFrame
            Filtered copy.
        """
        if age_col not in df.columns:
            logger.warning("age_range: column '%s' not found — filter skipped.", age_col)
            return df

        mask = pd.Series(True, index=df.index)
        if min_age is not None:
            mask &= df[age_col] >= min_age
        if max_age is not None:
            mask &= df[age_col] <= max_age

        result = df[mask].copy()
        self._log_filter(
            "age_range", len(df), len(result), min_age=min_age, max_age=max_age
        )
        return result.reset_index(drop=True)

    def diagnosis_filter(
        self,
        df: pd.DataFrame,
        diagnoses_col: str,
        include_codes: Optional[Sequence[str]] = None,
        exclude_codes: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Filter cohort by ICD code membership.

        *inclusion* and *exclusion* can be used together.  Exclusion is
        applied **after** inclusion.  Code matching is prefix-based: a
        filter code ``'I21'`` matches ``'I21.3'``, ``'I210'``, etc.

        Parameters
        ----------
        df:
            Input DataFrame.
        diagnoses_col:
            Column containing ICD code strings (one code per cell) **or**
            a column containing lists / comma-separated strings of codes.
        include_codes:
            Codes (or prefixes) to *include*.  *None* = include all.
        exclude_codes:
            Codes (or prefixes) to *exclude*.  *None* = exclude none.

        Returns
        -------
        pd.DataFrame
            Filtered copy.
        """
        if diagnoses_col not in df.columns:
            logger.warning(
                "diagnosis_filter: column '%s' not found — filter skipped.",
                diagnoses_col,
            )
            return df

        def _matches_any(code_value: Any, prefixes: Sequence[str]) -> bool:
            """Return True if any prefix in *prefixes* matches *code_value*."""
            if code_value is None or (isinstance(code_value, float) and np.isnan(code_value)):
                return False
            # Support list-like or comma-separated
            if isinstance(code_value, (list, np.ndarray)):
                codes = [str(c).strip().upper() for c in code_value]
            else:
                codes = [str(code_value).strip().upper()]
            for c in codes:
                for p in prefixes:
                    if c.startswith(p.upper()):
                        return True
            return False

        mask = pd.Series(True, index=df.index)

        if include_codes:
            include_mask = df[diagnoses_col].apply(
                lambda v: _matches_any(v, include_codes)
            )
            mask &= include_mask

        if exclude_codes:
            exclude_mask = df[diagnoses_col].apply(
                lambda v: _matches_any(v, exclude_codes)
            )
            mask &= ~exclude_mask

        result = df[mask].copy()
        self._log_filter(
            "diagnosis_filter",
            len(df),
            len(result),
            include_codes=include_codes,
            exclude_codes=exclude_codes,
        )
        return result.reset_index(drop=True)

    def los_filter(
        self,
        df: pd.DataFrame,
        min_los: Optional[float] = None,
        max_los: Optional[float] = None,
        los_col: str = "los",
    ) -> pd.DataFrame:
        """Filter by length-of-stay (LOS).

        Parameters
        ----------
        df:
            Input DataFrame.
        min_los:
            Minimum LOS in days (inclusive).  *None* = no lower bound.
        max_los:
            Maximum LOS in days (inclusive).  *None* = no upper bound.
        los_col:
            Column name holding LOS values.

        Returns
        -------
        pd.DataFrame
            Filtered copy.
        """
        if los_col not in df.columns:
            logger.warning("los_filter: column '%s' not found — filter skipped.", los_col)
            return df

        mask = pd.Series(True, index=df.index)
        if min_los is not None:
            mask &= df[los_col] >= min_los
        if max_los is not None:
            mask &= df[los_col] <= max_los

        result = df[mask].copy()
        self._log_filter(
            "los_filter", len(df), len(result), min_los=min_los, max_los=max_los
        )
        return result.reset_index(drop=True)

    def date_range(
        self,
        df: pd.DataFrame,
        date_col: str,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Filter rows by a datetime column.

        Parameters
        ----------
        df:
            Input DataFrame.
        date_col:
            Column name with datetime-like values.
        start:
            Lower bound (inclusive).  Accepts ISO 8601 strings or
            :class:`pandas.Timestamp`.
        end:
            Upper bound (inclusive).

        Returns
        -------
        pd.DataFrame
            Filtered copy.
        """
        if date_col not in df.columns:
            logger.warning("date_range: column '%s' not found — filter skipped.", date_col)
            return df

        col = pd.to_datetime(df[date_col], errors="coerce")
        mask = pd.Series(True, index=df.index)

        if start is not None:
            mask &= col >= pd.Timestamp(start)
        if end is not None:
            mask &= col <= pd.Timestamp(end)

        result = df[mask].copy()
        self._log_filter(
            "date_range", len(df), len(result), start=str(start), end=str(end)
        )
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return a summary dictionary of cohort demographics.

        Keys always present:  ``n_rows``, ``n_cols``.
        Keys present when matching column exists:
        ``age_mean``, ``age_std``, ``age_min``, ``age_max``,
        ``gender_distribution``, ``ethnicity_distribution``,
        ``mortality_rate``, ``readmit_30d_rate``, ``los_mean``.

        Parameters
        ----------
        df:
            Cohort DataFrame.

        Returns
        -------
        dict
            Summary statistics.
        """
        stats: Dict[str, Any] = {
            "n_rows": len(df),
            "n_cols": df.shape[1],
        }

        # --- Age ---
        for age_col in ("age", "age_at_admission"):
            if age_col in df.columns:
                age = df[age_col].dropna()
                stats.update(
                    {
                        "age_mean": round(float(age.mean()), 2),
                        "age_std": round(float(age.std()), 2),
                        "age_min": float(age.min()),
                        "age_max": float(age.max()),
                    }
                )
                break

        # --- Gender ---
        for g_col in ("gender", "sex"):
            if g_col in df.columns:
                stats["gender_distribution"] = (
                    df[g_col].value_counts(normalize=True).round(4).to_dict()
                )
                break

        # --- Ethnicity ---
        for e_col in ("ethnicity", "race"):
            if e_col in df.columns:
                stats["ethnicity_distribution"] = (
                    df[e_col].value_counts(normalize=True).round(4).to_dict()
                )
                break

        # --- Outcomes ---
        if "hospital_expire_flag" in df.columns:
            stats["mortality_rate"] = round(
                float(df["hospital_expire_flag"].mean()), 4
            )
        if "readmit_30d" in df.columns:
            stats["readmit_30d_rate"] = round(float(df["readmit_30d"].mean()), 4)

        # --- LOS ---
        for los_col in ("los", "length_of_stay"):
            if los_col in df.columns:
                stats["los_mean"] = round(float(df[los_col].mean()), 2)
                break

        logger.info("summary: cohort has %d rows, %d cols", stats["n_rows"], stats["n_cols"])
        return stats

    # ------------------------------------------------------------------
    # Survival analysis helper
    # ------------------------------------------------------------------

    def build_survival_cohort(
        self,
        df: pd.DataFrame,
        time_col: str,
        event_col: str,
        min_follow_up: float = 0.0,
    ) -> pd.DataFrame:
        """Filter to rows with valid survival data.

        Rows are kept when:
        - *time_col* is present, positive, and non-null.
        - *event_col* is present, non-null, and binary (0 / 1).

        Parameters
        ----------
        df:
            Input DataFrame.
        time_col:
            Column holding follow-up time (e.g. ``'time_to_death_days'``).
        event_col:
            Column holding event indicator (1 = event occurred, 0 = censored).
        min_follow_up:
            Minimum required follow-up time (exclusive lower bound).

        Returns
        -------
        pd.DataFrame
            Filtered copy suitable for survival modelling.

        Raises
        ------
        KeyError
            If *time_col* or *event_col* are absent from *df*.
        """
        for col in (time_col, event_col):
            if col not in df.columns:
                raise KeyError(f"build_survival_cohort: required column '{col}' not found.")

        mask = (
            df[time_col].notna()
            & (df[time_col] > min_follow_up)
            & df[event_col].notna()
            & df[event_col].isin([0, 1])
        )
        result = df[mask].copy()

        n_removed = len(df) - len(result)
        logger.info(
            "build_survival_cohort: %d rows → %d (removed %d invalid rows)",
            len(df),
            len(result),
            n_removed,
        )

        # Basic integrity check
        event_counts = result[event_col].value_counts().to_dict()
        logger.info("build_survival_cohort: event distribution: %s", event_counts)

        return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    rng = np.random.default_rng(7)
    n = 500

    df = pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "age": rng.uniform(0, 100, n),
            "gender": rng.choice(["M", "F"], n),
            "ethnicity": rng.choice(["White", "Black", "Hispanic", "Asian", "Other"], n),
            "los": rng.exponential(5, n),
            "primary_icd": rng.choice(["I21.3", "J18.9", "E11.9", "F32.1", "C50.9", "K35.2"], n),
            "admit_date": pd.date_range("2018-01-01", periods=n, freq="6h"),
            "hospital_expire_flag": rng.integers(0, 2, n),
            "readmit_30d": rng.integers(0, 2, n),
            "time_to_event": rng.uniform(-1, 1000, n),  # some negative / zero intentionally
            "event": rng.choice([0, 1, np.nan], n),
        }
    )

    cb = CohortBuilder()

    # Declarative pipeline
    filters: List[FilterSpec] = [
        {"type": "age_range", "min_age": 18, "max_age": 85},
        {"type": "los_filter", "min_los": 1.0},
        {"type": "date_range", "date_col": "admit_date", "start": "2019-01-01"},
        {"type": "diagnosis_filter", "diagnoses_col": "primary_icd",
         "include_codes": ["I", "J", "E"], "exclude_codes": ["E11"]},
    ]
    cohort = cb.from_patient_df(df, filters)
    print("=== Cohort summary ===")
    import json
    print(json.dumps(cb.summary(cohort), indent=2))

    # Survival cohort
    surv = cb.build_survival_cohort(df, "time_to_event", "event")
    print(f"\nSurvival cohort: {len(surv)} rows")
