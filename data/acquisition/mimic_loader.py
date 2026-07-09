"""
MIMIC-IV data loader for HealthRiskAI.

Connects to a PostgreSQL instance hosting MIMIC-IV and extracts structured
clinical data (patient cohorts, lab events, prescriptions, clinical notes,
diagnoses) into pandas DataFrames for downstream modelling.

Environment variables
---------------------
DATABASE_URL : str
    SQLAlchemy-compatible PostgreSQL connection string, e.g.
    postgresql+psycopg2://user:pass@host:5432/mimic4
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import OperationalError

import os

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logger.bind(module="mimic_loader")

# Load .env file so DATABASE_URL is available via os.getenv
load_dotenv()


# ---------------------------------------------------------------------------
# MIMICLoader
# ---------------------------------------------------------------------------
class MIMICLoader:
    """High-level interface for extracting cohorts from MIMIC-IV."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        db_url:
            SQLAlchemy connection string. Falls back to the ``DATABASE_URL``
            environment variable when not provided.
        """
        resolved_url = db_url or os.getenv("DATABASE_URL")
        if not resolved_url:
            raise ValueError(
                "No database URL supplied. Set DATABASE_URL in your environment "
                "or pass db_url= explicitly."
            )

        logger.info("Creating SQLAlchemy engine …")
        self.engine: Engine = create_engine(
            resolved_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={"connect_timeout": 30},
        )
        logger.success("Engine created for host={}", self.engine.url.host)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> Connection:
        """Return an active SQLAlchemy connection, retrying up to 3 times
        with exponential back-off (1 s, 2 s, 4 s) on ``OperationalError``."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                conn = self.engine.connect()
                logger.debug("DB connection established on attempt {}", attempt)
                return conn
            except OperationalError as exc:
                wait = 2 ** (attempt - 1)  # 1, 2, 4 seconds
                if attempt == max_attempts:
                    logger.error(
                        "Could not connect after {} attempts: {}", max_attempts, exc
                    )
                    raise
                logger.warning(
                    "Connection attempt {} failed — retrying in {}s …", attempt, wait
                )
                time.sleep(wait)

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    def extract_patient_cohort(
        self,
        icd_codes: list[str],
        min_age: int = 18,
        max_records: int = 10_000,
    ) -> pd.DataFrame:
        """Return a cohort of hospitalisations matching the given ICD-10 codes.

        Parameters
        ----------
        icd_codes:
            List of ICD-10 codes (prefix matching is applied, e.g. ``["E11"]``
            will match all sub-codes starting with *E11*).
        min_age:
            Minimum ``anchor_age`` (inclusive).
        max_records:
            Maximum number of rows to return (LIMIT).

        Returns
        -------
        pd.DataFrame
            Columns: subject_id, hadm_id, gender, anchor_age, admittime,
            dischtime, hospital_expire_flag, icd_code, icd_version
        """
        if not icd_codes:
            raise ValueError("icd_codes must be a non-empty list.")

        # Build a PostgreSQL array literal of LIKE patterns, e.g. '{"E11%","I10%"}'
        like_patterns = [f"{code}%" for code in icd_codes]
        array_literal = "ARRAY[" + ", ".join(f"'{p}'" for p in like_patterns) + "]"

        sql = text(
            f"""
            SELECT
                p.subject_id,
                a.hadm_id,
                p.gender,
                p.anchor_age,
                a.admittime,
                a.dischtime,
                a.hospital_expire_flag,
                d.icd_code,
                d.icd_version
            FROM mimiciv_hosp.patients   p
            JOIN mimiciv_hosp.admissions a ON p.subject_id = a.subject_id
            JOIN mimiciv_hosp.diagnoses_icd d
                 ON a.hadm_id = d.hadm_id
            WHERE d.icd_code  LIKE ANY({array_literal})
              AND d.icd_version = 10
              AND p.anchor_age >= :min_age
            ORDER BY a.admittime DESC
            LIMIT :max_records
            """
        )

        logger.info(
            "Extracting patient cohort | icd_codes={} min_age={} max_records={}",
            icd_codes,
            min_age,
            max_records,
        )

        with self._get_connection() as conn:
            df = pd.read_sql(sql, conn, params={"min_age": min_age, "max_records": max_records})

        logger.success("Cohort extracted: {} rows", len(df))
        return df

    def extract_lab_events(
        self,
        subject_ids: list[int],
        loinc_codes: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Extract lab-event records for a list of subjects.

        Parameters
        ----------
        subject_ids:
            List of MIMIC subject_id values to filter on.
        loinc_codes:
            Optional list of LOINC codes; when supplied only labs with
            matching LOINC codes are returned (requires a join to
            ``mimiciv_hosp.d_labitems``).

        Returns
        -------
        pd.DataFrame
            Columns: subject_id, hadm_id, itemid, charttime, value,
            valuenum, valueuom, flag  (plus loinc_code when filtered)
        """
        if not subject_ids:
            raise ValueError("subject_ids must be a non-empty list.")

        subject_array = "ARRAY[" + ", ".join(str(i) for i in subject_ids) + "]"

        if loinc_codes:
            loinc_array = (
                "ARRAY[" + ", ".join(f"'{c}'" for c in loinc_codes) + "]"
            )
            sql = text(
                f"""
                SELECT
                    le.subject_id,
                    le.hadm_id,
                    le.itemid,
                    dl.loinc_code,
                    le.charttime,
                    le.value,
                    le.valuenum,
                    le.valueuom,
                    le.flag
                FROM mimiciv_hosp.labevents le
                JOIN mimiciv_hosp.d_labitems dl ON le.itemid = dl.itemid
                WHERE le.subject_id = ANY({subject_array})
                  AND dl.loinc_code  = ANY({loinc_array})
                ORDER BY le.charttime
                """
            )
            logger.info(
                "Extracting lab events | subjects={} loinc_codes={}",
                len(subject_ids),
                loinc_codes,
            )
        else:
            sql = text(
                f"""
                SELECT
                    le.subject_id,
                    le.hadm_id,
                    le.itemid,
                    le.charttime,
                    le.value,
                    le.valuenum,
                    le.valueuom,
                    le.flag
                FROM mimiciv_hosp.labevents le
                WHERE le.subject_id = ANY({subject_array})
                ORDER BY le.charttime
                """
            )
            logger.info(
                "Extracting lab events | subjects={} (all LOINC codes)",
                len(subject_ids),
            )

        with self._get_connection() as conn:
            df = pd.read_sql(sql, conn)

        logger.success("Lab events extracted: {} rows", len(df))
        return df

    def extract_prescriptions(self, subject_ids: list[int]) -> pd.DataFrame:
        """Extract prescription records for a list of subjects.

        Returns
        -------
        pd.DataFrame
            Columns: subject_id, hadm_id, drug, drug_type,
            formulary_drug_cd, gsn, ndc, prod_strength, dose_val_rx,
            dose_unit_rx, route, starttime, stoptime
        """
        if not subject_ids:
            raise ValueError("subject_ids must be a non-empty list.")

        subject_array = "ARRAY[" + ", ".join(str(i) for i in subject_ids) + "]"

        sql = text(
            f"""
            SELECT
                subject_id,
                hadm_id,
                drug,
                drug_type,
                formulary_drug_cd,
                gsn,
                ndc,
                prod_strength,
                dose_val_rx,
                dose_unit_rx,
                route,
                starttime,
                stoptime
            FROM mimiciv_hosp.prescriptions
            WHERE subject_id = ANY({subject_array})
            ORDER BY starttime
            """
        )

        logger.info(
            "Extracting prescriptions | subjects={}", len(subject_ids)
        )

        with self._get_connection() as conn:
            df = pd.read_sql(sql, conn)

        logger.success("Prescriptions extracted: {} rows", len(df))
        return df

    def extract_clinical_notes(
        self,
        hadm_ids: list[int],
        note_types: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Extract free-text clinical notes for a list of admissions.

        Parameters
        ----------
        hadm_ids:
            List of hospital admission IDs.
        note_types:
            Optional list of ``note_type`` values to restrict results, e.g.
            ``["Discharge summary", "Radiology"]``.

        Returns
        -------
        pd.DataFrame
            Columns: subject_id, hadm_id, note_type, note_seq, charttime, text
        """
        if not hadm_ids:
            raise ValueError("hadm_ids must be a non-empty list.")

        hadm_array = "ARRAY[" + ", ".join(str(i) for i in hadm_ids) + "]"

        if note_types:
            type_array = (
                "ARRAY[" + ", ".join(f"'{t}'" for t in note_types) + "]"
            )
            type_filter = f"AND n.note_type = ANY({type_array})"
        else:
            type_filter = ""

        sql = text(
            f"""
            SELECT
                n.subject_id,
                n.hadm_id,
                n.note_type,
                n.note_seq,
                n.charttime,
                n.text
            FROM mimiciv_note.noteevents n
            WHERE n.hadm_id = ANY({hadm_array})
            {type_filter}
            ORDER BY n.charttime, n.note_seq
            """
        )

        logger.info(
            "Extracting clinical notes | hadm_ids={} note_types={}",
            len(hadm_ids),
            note_types,
        )

        with self._get_connection() as conn:
            df = pd.read_sql(sql, conn)

        logger.success("Clinical notes extracted: {} rows", len(df))
        return df

    def extract_diagnoses(self, subject_ids: list[int]) -> pd.DataFrame:
        """Extract all ICD-10 diagnosis codes for the given subjects.

        Returns
        -------
        pd.DataFrame
            Columns: subject_id, hadm_id, seq_num, icd_code, icd_version,
            long_title
        """
        if not subject_ids:
            raise ValueError("subject_ids must be a non-empty list.")

        subject_array = "ARRAY[" + ", ".join(str(i) for i in subject_ids) + "]"

        sql = text(
            f"""
            SELECT
                d.subject_id,
                d.hadm_id,
                d.seq_num,
                d.icd_code,
                d.icd_version,
                di.long_title
            FROM mimiciv_hosp.diagnoses_icd d
            LEFT JOIN mimiciv_hosp.d_icd_diagnoses di
                   ON d.icd_code    = di.icd_code
                  AND d.icd_version = di.icd_version
            WHERE d.subject_id = ANY({subject_array})
              AND d.icd_version = 10
            ORDER BY d.hadm_id, d.seq_num
            """
        )

        logger.info("Extracting diagnoses | subjects={}", len(subject_ids))

        with self._get_connection() as conn:
            df = pd.read_sql(sql, conn)

        logger.success("Diagnoses extracted: {} rows", len(df))
        return df

    def get_readmission_labels(
        self,
        admissions_df: pd.DataFrame,
        days: int = 30,
    ) -> pd.Series:
        """Compute 30-day (or *days*-day) unplanned readmission labels.

        For each admission in *admissions_df* the method checks whether the
        same patient has a subsequent admission that starts within *days* days
        of the current discharge time.

        Parameters
        ----------
        admissions_df:
            DataFrame with at minimum columns:
            ``hadm_id``, ``subject_id``, ``admittime``, ``dischtime``.
        days:
            Readmission window in calendar days (default 30).

        Returns
        -------
        pd.Series
            Boolean Series indexed by ``hadm_id``.
            ``True`` → patient was readmitted within the window.
        """
        required_cols = {"hadm_id", "subject_id", "admittime", "dischtime"}
        missing = required_cols - set(admissions_df.columns)
        if missing:
            raise ValueError(f"admissions_df is missing columns: {missing}")

        df = admissions_df[["hadm_id", "subject_id", "admittime", "dischtime"]].copy()
        df["admittime"] = pd.to_datetime(df["admittime"])
        df["dischtime"] = pd.to_datetime(df["dischtime"])
        df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

        # For each admission find the next admission for the same subject
        df["next_admittime"] = (
            df.groupby("subject_id")["admittime"].shift(-1)
        )

        threshold = pd.Timedelta(days=days)
        df["readmitted"] = (
            (df["next_admittime"] - df["dischtime"]) <= threshold
        ) & (df["next_admittime"].notna())

        labels = df.set_index("hadm_id")["readmitted"].astype(bool)
        readmit_count = labels.sum()
        logger.info(
            "Readmission labels computed | window={}d positive={}/{} ({:.1f}%)",
            days,
            readmit_count,
            len(labels),
            100 * readmit_count / len(labels) if len(labels) else 0,
        )
        return labels

    def save_cohort(self, df: pd.DataFrame, path: str) -> None:
        """Persist *df* to Parquet, creating parent directories as needed.

        Parameters
        ----------
        df:
            DataFrame to save.
        path:
            Destination file path (should end in ``.parquet``).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest, index=False, engine="pyarrow", compression="snappy")
        logger.success(
            "Cohort saved to {} | rows={} columns={}",
            dest,
            len(df),
            list(df.columns),
        )
