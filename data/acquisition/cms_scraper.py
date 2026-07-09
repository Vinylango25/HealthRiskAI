"""
CMS Hospital Compare Data Client
==================================
Production-quality client for fetching hospital quality, readmission, HCAHPS,
complications, payment, and financial data from CMS Provider Data Catalog and
CMS Cost Report public use files.

Usage
-----
    from data.acquisition.cms_scraper import CMSDataClient

    client = CMSDataClient()
    hospitals = client.get_hospital_general_info()
    merged = client.merge_all_hospital_data()

Environment variables
---------------------
CMS_BASE_URL : str
    Override the default CMS Provider Data API base URL.
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from requests.exceptions import RequestException

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_BASE_URL = "https://data.cms.gov/provider-data/api/1/datastore/sql"
_COST_REPORT_URL = (
    "https://www.cms.gov/files/zip/fy2022-mdcr-prvdr-cost-rpt-fh-202.zip"
)
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_MIN_REQUEST_INTERVAL = 0.2   # 5 req/s max

logger = logger.bind(module="cms_scraper")


class CMSDataClient:
    """
    Client for CMS Hospital Compare datasets via the Provider Data Catalog API
    and CMS Hospital Cost Report public use files.

    Parameters
    ----------
    base_url:
        Base URL for the CMS Provider Data SQL API.
        Falls back to ``CMS_BASE_URL`` env var, then to the public default.
    cache_dir:
        Local directory for cached parquet files.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        cache_dir: str = "data/raw/cms",
    ) -> None:
        self.base_url: str = (
            base_url
            or os.getenv("CMS_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "HealthRiskAI/1.0 (cms-scraper-client)",
            }
        )
        self._last_request_ts: float = 0.0

        logger.info(
            "CMSDataClient initialised | base_url={} cache_dir={}",
            self.base_url,
            self.cache_dir,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    def _get_json(self, url: str, params: dict) -> list[dict]:
        """Single GET with retry/back-off. Returns parsed JSON list."""
        for attempt in range(1, _MAX_RETRIES + 1):
            self._rate_wait()
            try:
                resp = self._session.get(url, params=params, timeout=60)
                self._last_request_ts = time.monotonic()

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", _BACKOFF_BASE ** attempt))
                    logger.warning("Rate-limited (429). Sleeping {}s …", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "Server error {} attempt {}. Retrying in {}s …",
                        resp.status_code, attempt, wait,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                # CMS API wraps rows under 'results' or returns a list directly
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("results", data.get("data", []))
                return []

            except RequestException as exc:
                wait = _BACKOFF_BASE ** attempt
                if attempt == _MAX_RETRIES:
                    logger.error("GET {} failed after {} retries: {}", url, _MAX_RETRIES, exc)
                    raise
                logger.warning("Attempt {} error: {}. Retrying in {}s …", attempt, exc, wait)
                time.sleep(wait)

        return []

    def _api_get(
        self,
        dataset_id: str,
        limit: int = 10_000,
        offset: int = 0,
    ) -> pd.DataFrame:
        """
        Paginate through a CMS Provider Data Catalog dataset using the
        SQL-style datastore endpoint.

        Parameters
        ----------
        dataset_id:
            CMS dataset identifier (4x4 or UUID string).
        limit:
            Rows per page.
        offset:
            Starting row offset (used when called recursively/externally).

        Returns
        -------
        pd.DataFrame
            All rows concatenated across pages.
        """
        all_frames: list[pd.DataFrame] = []
        current_offset = offset

        while True:
            query = (
                f"[SELECT * FROM {dataset_id} "
                f"LIMIT {limit} OFFSET {current_offset}][]"
            )
            params = {"query": query}

            logger.debug(
                "CMS API | dataset={} offset={} limit={}", dataset_id, current_offset, limit
            )
            rows = self._get_json(self.base_url, params)

            if not rows:
                break

            all_frames.append(pd.DataFrame(rows))

            if len(rows) < limit:
                break

            current_offset += limit

        if not all_frames:
            logger.warning("No data returned for dataset_id={}", dataset_id)
            return pd.DataFrame()

        df = pd.concat(all_frames, ignore_index=True)
        logger.success("dataset={} total_rows={}", dataset_id, len(df))
        return df

    @staticmethod
    def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase + strip column names."""
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Dataset methods
    # ------------------------------------------------------------------

    def get_hospital_general_info(self) -> pd.DataFrame:
        """
        Fetch Hospital General Information from CMS dataset ``29c8-io3p``.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, hospital_name, address, city, state, zip,
            county, phone, hospital_type, hospital_ownership,
            emergency_services
        """
        logger.info("Fetching hospital general info (29c8-io3p)…")
        raw = self._api_get("29c8-io3p")
        if raw.empty:
            return raw

        raw = self._normalise_cols(raw)

        col_map = {
            "facility_id": "provider_id",
            "provider_number": "provider_id",
            "provider_id": "provider_id",
            "facility_name": "hospital_name",
            "hospital_name": "hospital_name",
            "address": "address",
            "street_address": "address",
            "city": "city",
            "city_town": "city",
            "state": "state",
            "zip_code": "zip",
            "zip": "zip",
            "county_name": "county",
            "county": "county",
            "phone_number": "phone",
            "phone": "phone",
            "hospital_type": "hospital_type",
            "hospital_ownership": "hospital_ownership",
            "emergency_services": "emergency_services",
            "meets_criteria_for_promoting_interoperability_of_ehrs": "ehr_interoperability",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = [
            "provider_id", "hospital_name", "address", "city", "state",
            "zip", "county", "phone", "hospital_type", "hospital_ownership",
            "emergency_services",
        ]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()
        logger.success("Hospital general info: {} rows", len(df))
        return df

    def get_readmission_rates(self) -> pd.DataFrame:
        """
        Fetch Hospital Readmissions Reduction Program data from dataset
        ``9n3s-kdb3``.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, measure_name, score,
            compared_to_national, footnote
        """
        logger.info("Fetching readmission rates (9n3s-kdb3)…")
        raw = self._api_get("9n3s-kdb3")
        if raw.empty:
            return raw

        raw = self._normalise_cols(raw)

        col_map = {
            "facility_id": "provider_id",
            "provider_id": "provider_id",
            "measure_name": "measure_name",
            "score": "score",
            "compared_to_national": "compared_to_national",
            "footnote": "footnote",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = ["provider_id", "measure_name", "score", "compared_to_national", "footnote"]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()

        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce")

        logger.success("Readmission rates: {} rows", len(df))
        return df

    def get_hcahps_scores(self) -> pd.DataFrame:
        """
        Fetch Patient Survey (HCAHPS) scores from CMS dataset ``dgck-syfz``.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, hcahps_measure_id, hcahps_question,
            hcahps_answer_pct, num_completed_surveys, survey_response_rate
        """
        logger.info("Fetching HCAHPS scores (dgck-syfz)…")
        raw = self._api_get("dgck-syfz")
        if raw.empty:
            return raw

        raw = self._normalise_cols(raw)

        col_map = {
            "facility_id": "provider_id",
            "provider_id": "provider_id",
            "hcahps_measure_id": "hcahps_measure_id",
            "hcahps_question": "hcahps_question",
            "hcahps_answer_percent": "hcahps_answer_pct",
            "hcahps_answer_pct": "hcahps_answer_pct",
            "number_of_completed_surveys": "num_completed_surveys",
            "num_completed_surveys": "num_completed_surveys",
            "survey_response_rate_percent": "survey_response_rate",
            "survey_response_rate": "survey_response_rate",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = [
            "provider_id", "hcahps_measure_id", "hcahps_question",
            "hcahps_answer_pct", "num_completed_surveys", "survey_response_rate",
        ]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()

        for col in ("hcahps_answer_pct", "num_completed_surveys", "survey_response_rate"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.success("HCAHPS scores: {} rows", len(df))
        return df

    def get_complications_safety(self) -> pd.DataFrame:
        """
        Fetch Complications and Deaths data from CMS dataset ``ynj2-r877``.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, measure_id, measure_name, score,
            compared_to_national
        """
        logger.info("Fetching complications & safety (ynj2-r877)…")
        raw = self._api_get("ynj2-r877")
        if raw.empty:
            return raw

        raw = self._normalise_cols(raw)

        col_map = {
            "facility_id": "provider_id",
            "provider_id": "provider_id",
            "measure_id": "measure_id",
            "measure_name": "measure_name",
            "score": "score",
            "compared_to_national": "compared_to_national",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = ["provider_id", "measure_id", "measure_name", "score", "compared_to_national"]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()

        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce")

        logger.success("Complications & safety: {} rows", len(df))
        return df

    def get_payment_efficiency(self) -> pd.DataFrame:
        """
        Fetch Payment and Value of Care data from CMS dataset ``arex-3axe``.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, payment_category, payment,
            payment_compared_to_national, value_of_care
        """
        logger.info("Fetching payment & efficiency (arex-3axe)…")
        raw = self._api_get("arex-3axe")
        if raw.empty:
            return raw

        raw = self._normalise_cols(raw)

        col_map = {
            "facility_id": "provider_id",
            "provider_id": "provider_id",
            "payment_category": "payment_category",
            "measure_name": "payment_category",
            "payment": "payment",
            "payment_compared_to_national": "payment_compared_to_national",
            "compared_to_national": "payment_compared_to_national",
            "value_of_care_display": "value_of_care",
            "value_of_care": "value_of_care",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = [
            "provider_id", "payment_category", "payment",
            "payment_compared_to_national", "value_of_care",
        ]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()

        if "payment" in df.columns:
            df["payment"] = pd.to_numeric(
                df["payment"].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce",
            )

        logger.success("Payment & efficiency: {} rows", len(df))
        return df

    def get_hospital_financial_data(self) -> pd.DataFrame:
        """
        Download and parse CMS Hospital Cost Report public use file (FY 2022).

        Attempts to download the ZIP from CMS, parse ALPHA.CSV and NMRC.CSV
        worksheets to extract key financial fields.

        If the download fails (network error, access restriction), returns a
        DataFrame with the correct schema populated with NaN values so that
        downstream pipelines do not crash.

        Returns
        -------
        pd.DataFrame
            Columns: provider_id, total_beds, total_revenue,
            net_patient_revenue, total_expenses, net_income,
            medicare_pct, medicaid_pct, total_discharges,
            adjusted_discharges
        """
        logger.info("Downloading CMS Hospital Cost Report (FY2022)…")

        # --- Worksheet line/column codes for key financial items ----------
        # NMRC worksheet codes (line_num, col_num) → field name
        # These are approximate HCRIS mappings for the 2552-10 cost report.
        NMRC_MAP: dict[tuple[str, str], str] = {
            ("S-3", "1"): "total_beds",
            ("G-3", "1"): "total_revenue",
            ("G-3", "3"): "net_patient_revenue",
            ("G-3", "2"): "total_expenses",
        }

        try:
            resp = self._session.get(_COST_REPORT_URL, timeout=120, stream=True)
            resp.raise_for_status()

            zip_bytes = io.BytesIO(resp.content)
            with zipfile.ZipFile(zip_bytes) as zf:
                names_upper = {n.upper(): n for n in zf.namelist()}

                # --- ALPHA.CSV: report header (provider_id, fiscal year, etc.)
                alpha_name = names_upper.get("ALPHA.CSV") or next(
                    (n for n in zf.namelist() if "ALPHA" in n.upper()), None
                )
                nmrc_name = names_upper.get("NMRC.CSV") or next(
                    (n for n in zf.namelist() if "NMRC" in n.upper()), None
                )

                if not alpha_name or not nmrc_name:
                    raise FileNotFoundError(
                        f"Expected ALPHA.CSV and NMRC.CSV inside ZIP. "
                        f"Found: {zf.namelist()}"
                    )

                alpha_df = pd.read_csv(
                    io.BytesIO(zf.read(alpha_name)),
                    dtype=str,
                    low_memory=False,
                )
                nmrc_df = pd.read_csv(
                    io.BytesIO(zf.read(nmrc_name)),
                    dtype=str,
                    low_memory=False,
                )

            # Normalise column names
            alpha_df.columns = [c.lower().strip() for c in alpha_df.columns]
            nmrc_df.columns = [c.lower().strip() for c in nmrc_df.columns]

            # ALPHA: identify provider_id column (rpt_rec_num links to NMRC)
            rpt_col = next(
                (c for c in alpha_df.columns if "rpt_rec_num" in c or "provider_number" in c),
                alpha_df.columns[0],
            )
            prvdr_col = next(
                (c for c in alpha_df.columns if "prvdr_num" in c or "provider_number" in c),
                None,
            )

            alpha_slim = alpha_df[[rpt_col]].copy()
            if prvdr_col and prvdr_col != rpt_col:
                alpha_slim["provider_id"] = alpha_df[prvdr_col]
            else:
                alpha_slim["provider_id"] = alpha_df[rpt_col]

            alpha_slim = alpha_slim.rename(columns={rpt_col: "rpt_rec_num"})

            # NMRC: pivot worksheet_code / line_num / col_num → value
            nmrc_rpt_col = next(
                (c for c in nmrc_df.columns if "rpt_rec_num" in c), nmrc_df.columns[0]
            )
            wksht_col = next(
                (c for c in nmrc_df.columns if "wksht_cd" in c or "worksheet" in c),
                nmrc_df.columns[1],
            )
            line_col = next(
                (c for c in nmrc_df.columns if "line_num" in c or "line" in c),
                nmrc_df.columns[2],
            )
            col_col = next(
                (c for c in nmrc_df.columns if "clmn_num" in c or "col" in c),
                nmrc_df.columns[3],
            )
            val_col = next(
                (c for c in nmrc_df.columns if "itm_val_num" in c or "value" in c),
                nmrc_df.columns[4],
            )

            records: list[dict] = []
            for _, row in alpha_slim.iterrows():
                rpt = row["rpt_rec_num"]
                prvdr = row["provider_id"]
                subset = nmrc_df[nmrc_df[nmrc_rpt_col] == rpt]
                rec: dict = {"provider_id": prvdr}
                for (wksht, line), field in NMRC_MAP.items():
                    match = subset[
                        (subset[wksht_col].str.strip() == wksht)
                        & (subset[line_col].str.strip() == line)
                    ]
                    if not match.empty:
                        rec[field] = pd.to_numeric(
                            match[val_col].iloc[0], errors="coerce"
                        )
                    else:
                        rec[field] = float("nan")
                records.append(rec)

            df = pd.DataFrame(records)

            # Add remaining expected columns as NaN if not derived above
            for col in (
                "net_income", "medicare_pct", "medicaid_pct",
                "total_discharges", "adjusted_discharges",
            ):
                if col not in df.columns:
                    df[col] = float("nan")

            logger.success("Hospital financial data: {} providers parsed", len(df))
            return df

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not download/parse CMS cost report: {}. "
                "Returning empty schema DataFrame.",
                exc,
            )
            note_cols = [
                "provider_id", "total_beds", "total_revenue",
                "net_patient_revenue", "total_expenses", "net_income",
                "medicare_pct", "medicaid_pct", "total_discharges",
                "adjusted_discharges", "_note",
            ]
            empty = pd.DataFrame(columns=note_cols)
            # One row flagging the issue so callers can detect it
            empty.loc[0] = [None] * (len(note_cols) - 1) + [
                "Real data requires downloading the CMS FY2022 cost report ZIP. "
                f"Download failed: {exc}"
            ]
            return empty

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def compute_hospital_credit_features(
        self,
        hospitals_df: pd.DataFrame,
        quality_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute derived financial and quality features for credit analysis.

        Joins hospital general info with financial data (and optionally quality
        data) on ``provider_id``, then computes:

        * ``operating_margin`` = (net_patient_revenue − total_expenses)
          / net_patient_revenue
        * ``beds_per_1000`` = total_beds / (population_served / 1 000)
          (only when ``population_served`` column is present)
        * ``financial_stress_flag`` = operating_margin < 0
          OR days_cash_on_hand < 30

        Parameters
        ----------
        hospitals_df:
            DataFrame with at least ``provider_id`` and financial columns
            (net_patient_revenue, total_expenses, total_beds).
        quality_df:
            Optional DataFrame with quality metrics keyed on ``provider_id``.

        Returns
        -------
        pd.DataFrame
            Merged and enriched DataFrame.
        """
        df = hospitals_df.copy()

        # Coerce numeric fields
        for col in ("net_patient_revenue", "total_expenses", "total_beds"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Operating margin
        if "net_patient_revenue" in df.columns and "total_expenses" in df.columns:
            df["operating_margin"] = (
                df["net_patient_revenue"] - df["total_expenses"]
            ) / df["net_patient_revenue"].replace(0, float("nan"))
        else:
            df["operating_margin"] = float("nan")

        # Beds per 1 000 population
        if "total_beds" in df.columns and "population_served" in df.columns:
            df["population_served"] = pd.to_numeric(df["population_served"], errors="coerce")
            df["beds_per_1000"] = df["total_beds"] / (df["population_served"] / 1_000).replace(
                0, float("nan")
            )
        else:
            df["beds_per_1000"] = float("nan")

        # Financial stress flag
        days_cash = (
            pd.to_numeric(df["days_cash_on_hand"], errors="coerce")
            if "days_cash_on_hand" in df.columns
            else pd.Series(float("nan"), index=df.index)
        )
        df["financial_stress_flag"] = (df["operating_margin"] < 0) | (days_cash < 30)

        # Merge quality data if provided
        if quality_df is not None and not quality_df.empty:
            if "provider_id" in quality_df.columns:
                df = df.merge(quality_df, on="provider_id", how="left", suffixes=("", "_quality"))
            else:
                logger.warning("quality_df has no 'provider_id' column; skipping quality merge.")

        logger.success(
            "compute_hospital_credit_features: {} rows, {} columns",
            len(df), len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Merge helper
    # ------------------------------------------------------------------

    def merge_all_hospital_data(self) -> pd.DataFrame:
        """
        Fetch all CMS datasets, merge them on ``provider_id``, and return a
        single comprehensive DataFrame.

        Returns
        -------
        pd.DataFrame
            Wide DataFrame with all hospital metrics joined on provider_id.
        """
        logger.info("merge_all_hospital_data: fetching all CMS datasets…")

        general = self.get_hospital_general_info()
        readmissions = self.get_readmission_rates()
        hcahps = self.get_hcahps_scores()
        complications = self.get_complications_safety()
        payments = self.get_payment_efficiency()
        financials = self.get_hospital_financial_data()

        if general.empty:
            logger.warning("Hospital general info is empty; returning empty DataFrame.")
            return pd.DataFrame()

        merged = general.copy()

        def _safe_merge(left: pd.DataFrame, right: pd.DataFrame, label: str) -> pd.DataFrame:
            if right.empty or "provider_id" not in right.columns:
                logger.warning("{} DataFrame is empty or missing provider_id; skipping.", label)
                return left
            return left.merge(right, on="provider_id", how="left", suffixes=("", f"_{label}"))

        # Pivot readmissions and HCAHPS to wide before merging to avoid row explosion
        if not readmissions.empty and "provider_id" in readmissions.columns:
            try:
                readm_wide = readmissions.pivot_table(
                    index="provider_id",
                    columns="measure_name",
                    values="score",
                    aggfunc="mean",
                ).reset_index()
                readm_wide.columns = [
                    f"readm_{c}" if c != "provider_id" else c for c in readm_wide.columns
                ]
                merged = _safe_merge(merged, readm_wide, "readmissions")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not pivot readmissions: {}", exc)
                merged = _safe_merge(merged, readmissions, "readmissions")

        if not hcahps.empty and "provider_id" in hcahps.columns:
            try:
                hcahps_wide = hcahps.pivot_table(
                    index="provider_id",
                    columns="hcahps_measure_id",
                    values="hcahps_answer_pct",
                    aggfunc="mean",
                ).reset_index()
                hcahps_wide.columns = [
                    f"hcahps_{c}" if c != "provider_id" else c for c in hcahps_wide.columns
                ]
                merged = _safe_merge(merged, hcahps_wide, "hcahps")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not pivot HCAHPS: {}", exc)
                merged = _safe_merge(merged, hcahps, "hcahps")

        merged = _safe_merge(merged, complications, "complications")
        merged = _safe_merge(merged, payments, "payments")
        merged = _safe_merge(merged, financials, "financials")

        # Compute credit features on the merged result
        merged = self.compute_hospital_credit_features(merged)

        logger.success("merge_all_hospital_data: {} rows, {} columns", len(merged), len(merged.columns))
        return merged
