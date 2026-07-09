"""
CDC WONDER & Socrata API Client
================================
Production-quality client for fetching public health mortality, disease
surveillance, and prevalence data from the CDC's Socrata-powered open data
portal (data.cdc.gov) and CDC WONDER APIs.

Datasets accessed
-----------------
* 65eu-2qhd  Drug Overdose Mortality by State
* 9dzk-7tez  Heart Disease Mortality
* er7h-3asm  Cancer Incidence
* x9gk-5huc  National Notifiable Disease Surveillance System (NNDSS)
* bi63-dtpu  Underlying Cause of Death
* rqg5-mkef  Diagnosed Diabetes Prevalence

Usage
-----
    from data.acquisition.cdc_wonder import CDCWonderClient

    client = CDCWonderClient()
    df = client.get_drug_overdose_mortality()
    client.download_all()

Environment variables
---------------------
CDC_SOCRATA_APP_TOKEN : str
    Optional Socrata app token to raise the anonymous rate limit.
    Without it the API is throttled to ~1 000 rows/request.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from requests.exceptions import HTTPError, RequestException

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SOCRATA_BASE: str = "https://data.cdc.gov/resource"
_DEFAULT_PAGE_SIZE: int = 50_000      # rows per page request
_MIN_REQUEST_INTERVAL: float = 0.2    # ≤ 5 req/s → be conservative
_MAX_RETRIES: int = 3
_BACKOFF_BASE: float = 2.0            # seconds

logger = logger.bind(module="cdc_wonder")


class CDCWonderClient:
    """
    Client for CDC public health datasets served via the Socrata ODP API.

    Parameters
    ----------
    cache_dir:
        Local directory used to persist downloaded parquet files so that
        repeated calls skip the network round-trip.
    """

    def __init__(self, cache_dir: str = "data/raw/cdc") -> None:
        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._app_token: Optional[str] = os.getenv("CDC_SOCRATA_APP_TOKEN")

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "HealthRiskAI/1.0 (cdc-wonder-client)",
            }
        )
        if self._app_token:
            self._session.headers["X-App-Token"] = self._app_token

        self._last_request_ts: float = 0.0

        logger.info(
            "CDCWonderClient initialised | cache_dir={} app_token={}",
            self.cache_dir,
            "set" if self._app_token else "not set (anonymous)",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_wait(self) -> None:
        """Sleep if needed to stay within the rate limit."""
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    def _socrata_get(
        self,
        dataset_id: str,
        where_clause: Optional[str] = None,
        limit: int = _DEFAULT_PAGE_SIZE,
    ) -> pd.DataFrame:
        """
        Fetch all rows from a Socrata dataset, handling pagination transparently.

        Parameters
        ----------
        dataset_id:
            Socrata 4x4 dataset identifier (e.g. ``'65eu-2qhd'``).
        where_clause:
            Optional SQL-style ``$where`` filter (Socrata Query Language).
        limit:
            Page size per request (max 50 000 for anonymous callers).

        Returns
        -------
        pd.DataFrame
            All rows concatenated across pages.
        """
        url = f"{_SOCRATA_BASE}/{dataset_id}.json"
        all_frames: list[pd.DataFrame] = []
        offset = 0

        while True:
            params: dict = {"$limit": limit, "$offset": offset}
            if where_clause:
                params["$where"] = where_clause

            data = self._get_json(url, params)
            if not data:
                break

            all_frames.append(pd.DataFrame(data))
            logger.debug(
                "dataset={} offset={} rows_fetched={}",
                dataset_id,
                offset,
                len(data),
            )

            if len(data) < limit:
                # Last page — no more rows
                break

            offset += limit

        if not all_frames:
            logger.warning("No data returned for dataset_id={}", dataset_id)
            return pd.DataFrame()

        df = pd.concat(all_frames, ignore_index=True)
        logger.success(
            "dataset={} total_rows={} columns={}",
            dataset_id,
            len(df),
            list(df.columns),
        )
        return df

    def _get_json(self, url: str, params: dict) -> list[dict]:
        """
        Perform a single GET with retry / back-off logic.

        Returns
        -------
        list[dict]
            Parsed JSON response (always a list for Socrata endpoints).
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            self._rate_wait()
            try:
                resp = self._session.get(url, params=params, timeout=60)
                self._last_request_ts = time.monotonic()

                if resp.status_code == 429:
                    wait = _BACKOFF_BASE ** attempt
                    retry_after = int(resp.headers.get("Retry-After", wait))
                    logger.warning("Rate-limited (429). Sleeping {}s …", retry_after)
                    time.sleep(retry_after)
                    continue

                if resp.status_code >= 500:
                    wait = _BACKOFF_BASE ** attempt
                    logger.warning(
                        "Server error {} on attempt {}. Retrying in {}s …",
                        resp.status_code, attempt, wait,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()  # type: ignore[return-value]

            except RequestException as exc:
                wait = _BACKOFF_BASE ** attempt
                if attempt == _MAX_RETRIES:
                    logger.error("GET {} failed after {} retries: {}", url, _MAX_RETRIES, exc)
                    raise
                logger.warning("Attempt {} failed: {}. Retrying in {}s …", attempt, exc, wait)
                time.sleep(wait)

        return []

    # ------------------------------------------------------------------
    # Public dataset methods
    # ------------------------------------------------------------------

    def get_drug_overdose_mortality(self) -> pd.DataFrame:
        """
        Fetch drug overdose mortality data by state from CDC Socrata dataset
        ``65eu-2qhd`` (Drug Overdose Mortality by State).

        Returns
        -------
        pd.DataFrame
            Columns: state, year, month, cause_of_death, deaths,
            population, age_adjusted_rate
        """
        logger.info("Fetching drug overdose mortality (65eu-2qhd)…")
        raw = self._socrata_get("65eu-2qhd")

        if raw.empty:
            return raw

        # Flexible column mapping: Socrata may return snake_case or Title Case
        col_map = {
            "state": "state",
            "year": "year",
            "month": "month",
            "cause_of_death": "cause_of_death",
            "indicator": "cause_of_death",         # alternate column name
            "deaths": "deaths",
            "population": "population",
            "age_adjusted_rate": "age_adjusted_rate",
            "age-adjusted rate": "age_adjusted_rate",
        }
        raw.columns = [c.lower().strip() for c in raw.columns]
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = ["state", "year", "month", "cause_of_death", "deaths",
                   "population", "age_adjusted_rate"]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()

        # Coerce numeric columns
        for col in ("deaths", "population", "age_adjusted_rate"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("year", "month"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        logger.success("Drug overdose mortality: {} rows", len(df))
        return df

    def get_heart_disease_mortality(self) -> pd.DataFrame:
        """
        Fetch heart disease mortality data from CDC dataset ``9dzk-7tez``.

        Returns
        -------
        pd.DataFrame
            Columns: state, year, cause, deaths, rate
        """
        logger.info("Fetching heart disease mortality (9dzk-7tez)…")
        raw = self._socrata_get("9dzk-7tez")

        if raw.empty:
            return raw

        raw.columns = [c.lower().strip() for c in raw.columns]

        col_map = {
            "locationabbr": "state",
            "locationdesc": "state",
            "yearstart": "year",
            "year": "year",
            "topic": "cause",
            "question": "cause",
            "datavalue": "rate",
            "data_value": "rate",
            "death_rate": "rate",
            "deaths": "deaths",
            "datavaluealt": "rate",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = ["state", "year", "cause", "deaths", "rate"]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()

        for col in ("deaths", "rate"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        logger.success("Heart disease mortality: {} rows", len(df))
        return df

    def get_cancer_incidence(self) -> pd.DataFrame:
        """
        Fetch cancer incidence data from CDC dataset ``er7h-3asm``.

        Returns
        -------
        pd.DataFrame
            Columns: state, year, cancer_type, incidence_rate, deaths
        """
        logger.info("Fetching cancer incidence (er7h-3asm)…")
        raw = self._socrata_get("er7h-3asm")

        if raw.empty:
            return raw

        raw.columns = [c.lower().strip() for c in raw.columns]

        col_map = {
            "area": "state",
            "state": "state",
            "year": "year",
            "site": "cancer_type",
            "cancer_site": "cancer_type",
            "cancer_type": "cancer_type",
            "incidence_rate": "incidence_rate",
            "rate": "incidence_rate",
            "count": "deaths",
            "deaths": "deaths",
            "age_adjusted_incidence_rate": "incidence_rate",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = ["state", "year", "cancer_type", "incidence_rate", "deaths"]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()

        for col in ("incidence_rate", "deaths"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        logger.success("Cancer incidence: {} rows", len(df))
        return df

    def get_infectious_disease_surveillance(self) -> pd.DataFrame:
        """
        Fetch National Notifiable Disease Surveillance System (NNDSS) data
        from CDC dataset ``x9gk-5huc``.

        Returns
        -------
        pd.DataFrame
            Columns: disease, year, week, state, cases
        """
        logger.info("Fetching infectious disease surveillance (x9gk-5huc)…")
        raw = self._socrata_get("x9gk-5huc")

        if raw.empty:
            return raw

        raw.columns = [c.lower().strip() for c in raw.columns]

        col_map = {
            "disease_name": "disease",
            "disease": "disease",
            "condition": "disease",
            "year": "year",
            "mmwr_year": "year",
            "week": "week",
            "mmwr_week": "week",
            "reporting_area": "state",
            "state": "state",
            "current_week": "cases",
            "cases": "cases",
            "cum_2023": "cases",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = ["disease", "year", "week", "state", "cases"]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()

        for col in ("cases",):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("year", "week"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        logger.success("Infectious disease surveillance: {} rows", len(df))
        return df

    def get_mortality_data(
        self,
        cause_codes: Optional[list[str]] = None,
        year_start: int = 2015,
        year_end: int = 2023,
    ) -> pd.DataFrame:
        """
        Fetch underlying cause-of-death data from CDC dataset ``bi63-dtpu``.

        Parameters
        ----------
        cause_codes:
            Optional list of ICD-10 cause-of-death codes to filter on.
            If ``None``, returns all causes.
        year_start:
            Earliest year (inclusive).
        year_end:
            Latest year (inclusive).

        Returns
        -------
        pd.DataFrame
            Columns: year, state, cause_code, cause_name, deaths, population,
            crude_rate, age_adjusted_rate
        """
        logger.info(
            "Fetching underlying cause-of-death mortality (bi63-dtpu) "
            "years={}-{} cause_codes={}",
            year_start, year_end, cause_codes,
        )

        where_parts: list[str] = [
            f"year >= '{year_start}'",
            f"year <= '{year_end}'",
        ]
        if cause_codes:
            quoted = ", ".join(f"'{c}'" for c in cause_codes)
            where_parts.append(f"cause_of_death_code IN ({quoted})")

        where_clause = " AND ".join(where_parts)
        raw = self._socrata_get("bi63-dtpu", where_clause=where_clause)

        if raw.empty:
            return raw

        raw.columns = [c.lower().strip() for c in raw.columns]

        col_map = {
            "year": "year",
            "state": "state",
            "cause_of_death_code": "cause_code",
            "icd_10_code": "cause_code",
            "cause_of_death": "cause_name",
            "cause": "cause_name",
            "deaths": "deaths",
            "population": "population",
            "crude_rate": "crude_rate",
            "age_adjusted_rate": "age_adjusted_rate",
            "age-adjusted rate": "age_adjusted_rate",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = ["year", "state", "cause_code", "cause_name", "deaths",
                   "population", "crude_rate", "age_adjusted_rate"]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()

        for col in ("deaths", "population", "crude_rate", "age_adjusted_rate"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        logger.success("Mortality data: {} rows", len(df))
        return df

    def get_diabetes_prevalence(self) -> pd.DataFrame:
        """
        Fetch diagnosed diabetes prevalence data from CDC dataset ``rqg5-mkef``.

        Returns state-level diabetes prevalence by year.

        Returns
        -------
        pd.DataFrame
            Columns: state, year, percentage, lower_ci, upper_ci, sample_size
        """
        logger.info("Fetching diabetes prevalence (rqg5-mkef)…")
        raw = self._socrata_get("rqg5-mkef")

        if raw.empty:
            return raw

        raw.columns = [c.lower().strip() for c in raw.columns]

        col_map = {
            "locationabbr": "state",
            "locationdesc": "state_name",
            "year": "year",
            "yearend": "year",
            "data_value": "percentage",
            "datavalue": "percentage",
            "prevalence": "percentage",
            "low_confidence_limit": "lower_ci",
            "high_confidence_limit": "upper_ci",
            "sample_size": "sample_size",
            "samplesize": "sample_size",
        }
        raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

        desired = ["state", "year", "percentage", "lower_ci", "upper_ci", "sample_size"]
        available = [c for c in desired if c in raw.columns]
        df = raw[available].copy()

        for col in ("percentage", "lower_ci", "upper_ci", "sample_size"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        logger.success("Diabetes prevalence: {} rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Batch download
    # ------------------------------------------------------------------

    def download_all(self) -> dict[str, str]:
        """
        Download all datasets and persist each to a parquet file under
        ``cache_dir``.

        Returns
        -------
        dict[str, str]
            Mapping of dataset name to parquet file path (or ``'FAILED'``
            if the fetch raised an exception).
        """
        datasets: dict[str, object] = {
            "drug_overdose_mortality": self.get_drug_overdose_mortality,
            "heart_disease_mortality": self.get_heart_disease_mortality,
            "cancer_incidence": self.get_cancer_incidence,
            "infectious_disease_surveillance": self.get_infectious_disease_surveillance,
            "mortality_data": self.get_mortality_data,
            "diabetes_prevalence": self.get_diabetes_prevalence,
        }

        results: dict[str, str] = {}
        for name, fetch_fn in datasets.items():
            try:
                logger.info("download_all: fetching '{}'…", name)
                df: pd.DataFrame = fetch_fn()  # type: ignore[operator]
                out_path = self.cache_dir / f"{name}.parquet"
                df.to_parquet(out_path, index=False)
                logger.success("Saved {} rows to {}", len(df), out_path)
                results[name] = str(out_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to download '{}': {}", name, exc)
                results[name] = "FAILED"

        logger.info("download_all complete: {}", results)
        return results
