"""
WHO Global Health Observatory (GHO) OData REST API client for HealthRiskAI.

Fetches epidemiological and health-system indicators from the WHO GHO API
(https://www.who.int/data/gho/info/gho-odata-api) and caches results locally
as JSON files to avoid redundant network calls.

Environment variables
---------------------
WHO_GHO_BASE_URL : str
    Base URL for the GHO OData endpoint.
    Defaults to ``https://ghoapi.azureedge.net/api``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from requests.exceptions import HTTPError, RequestException

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logger.bind(module="who_gho")

# Load .env so WHO_GHO_BASE_URL is available via os.getenv
load_dotenv()

# Default public GHO OData endpoint
_DEFAULT_BASE_URL = "https://ghoapi.azureedge.net/api"

# Minimum interval between requests to stay within 10 req/sec
_MIN_REQUEST_INTERVAL: float = 0.1  # seconds

# ---------------------------------------------------------------------------
# Indicator catalogue used by convenience methods
# ---------------------------------------------------------------------------

_DISEASE_BURDEN_INDICATORS: dict[str, str] = {
    "WHOSIS_000001": "life_expectancy",
    "NCD_BMI_30A": "obesity_prevalence",
    "SA_0000001688": "alcohol_use_disorder",
    "M_Est_cigs_curr": "tobacco_smoking",
    "MDG_0000000026": "under5_mortality",
    "SDGPM25": "pm25_exposure",
    "NCD_DIABETES": "diabetes_prevalence",
}

_HEALTH_EXPENDITURE_INDICATORS: dict[str, str] = {
    "GHED_CHEGDP_SHA2011": "health_spend_pct_gdp",
    "GHED_CHEPCUSD_SHA2011": "health_spend_per_capita_usd",
}

_RISK_FACTOR_INDICATORS: dict[str, str] = {
    "NCD_BMI_30A": "obesity_prevalence",
    "M_Est_cigs_curr": "tobacco_smoking",
    "SA_0000001688": "alcohol_use_disorder",
    "NCD_PAC": "physical_inactivity",
}


# ---------------------------------------------------------------------------
# WHOGHOClient
# ---------------------------------------------------------------------------


class WHOGHOClient:
    """Client for the WHO Global Health Observatory OData REST API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        cache_dir: str = "data/raw/who_gho",
    ) -> None:
        """
        Parameters
        ----------
        base_url:
            Root URL of the GHO OData API. Falls back to the
            ``WHO_GHO_BASE_URL`` environment variable, then to the public
            default ``https://ghoapi.azureedge.net/api``.
        cache_dir:
            Directory used to cache downloaded JSON responses.
        """
        self.base_url: str = (
            base_url
            or os.getenv("WHO_GHO_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")

        self.cache_dir: Path = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_request_time: float = 0.0
        self._session: requests.Session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "HealthRiskAI-WHOGHOClient/1.0",
            }
        )

        logger.info(
            "WHOGHOClient initialised | base_url={} cache_dir={}",
            self.base_url,
            self.cache_dir,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limited_get(
        self,
        url: str,
        params: Optional[dict] = None,
    ) -> dict:
        """Perform a rate-limited HTTP GET request with retry logic.

        Enforces a maximum of 10 requests per second by inserting
        ``time.sleep`` where necessary.  Retries up to 3 times on HTTP
        429 (Too Many Requests) or 5xx server errors with exponential
        back-off.

        Parameters
        ----------
        url:
            Full request URL.
        params:
            Optional query-string parameters dict.

        Returns
        -------
        dict
            Parsed JSON response body.
        """
        max_attempts = 3
        backoff_base = 2  # seconds

        for attempt in range(1, max_attempts + 1):
            # Enforce rate limit (max 10 req/s → min 0.1 s between calls)
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < _MIN_REQUEST_INTERVAL:
                time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

            logger.debug("GET {} params={} attempt={}", url, params, attempt)

            try:
                response = self._session.get(url, params=params, timeout=30)
                self._last_request_time = time.monotonic()

                if response.status_code == 429:
                    # Respect Retry-After header if present
                    retry_after = int(
                        response.headers.get("Retry-After", backoff_base ** attempt)
                    )
                    logger.warning(
                        "Rate limited (429). Waiting {}s before retry …",
                        retry_after,
                    )
                    time.sleep(retry_after)
                    continue

                if response.status_code >= 500:
                    wait = backoff_base ** attempt
                    logger.warning(
                        "Server error {} on attempt {}. Retrying in {}s …",
                        response.status_code,
                        attempt,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

            except RequestException as exc:
                wait = backoff_base ** attempt
                if attempt == max_attempts:
                    logger.error(
                        "Request failed after {} attempts: {}", max_attempts, exc
                    )
                    raise
                logger.warning(
                    "Request error on attempt {}: {}. Retrying in {}s …",
                    attempt,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"Failed to GET {url} after {max_attempts} attempts.")

    def _cache_path(self, indicator_code: str, country: Optional[str]) -> str:
        """Return the local cache file path for an indicator/country pair.

        Parameters
        ----------
        indicator_code:
            WHO GHO indicator code, e.g. ``"WHOSIS_000001"``.
        country:
            ISO 3-letter country code, or ``None`` for all countries.

        Returns
        -------
        str
            Path string under ``data/raw/who_gho/``.
        """
        country_tag = country if country else "all"
        return str(self.cache_dir / f"{indicator_code}_{country_tag}.json")

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_indicator(
        self,
        indicator_code: str,
        country: Optional[str] = None,
        year_start: int = 2015,
        year_end: int = 2024,
    ) -> pd.DataFrame:
        """Fetch data for a single GHO indicator.

        Results are cached to a JSON file and reused on subsequent calls
        with the same parameters.

        Parameters
        ----------
        indicator_code:
            WHO GHO indicator code.
        country:
            ISO 3-letter country code to restrict results (e.g. ``"KEN"``).
            Pass ``None`` to retrieve all countries.
        year_start:
            Earliest year (inclusive) via OData ``$filter``.
        year_end:
            Latest year (inclusive) via OData ``$filter``.

        Returns
        -------
        pd.DataFrame
            Columns: indicator_code, country_code, year, value, low, high
        """
        cache_file = Path(self._cache_path(indicator_code, country))

        # --- Serve from cache if available ---
        if cache_file.exists():
            logger.debug(
                "Cache hit for indicator={} country={}", indicator_code, country
            )
            with cache_file.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        else:
            url = f"{self.base_url}/{indicator_code}"

            # Build $filter clause
            filter_parts = [
                f"TimeDim ge {year_start}",
                f"TimeDim le {year_end}",
            ]
            if country:
                filter_parts.append(f"SpatialDim eq '{country}'")

            params: dict = {
                "$filter": " and ".join(filter_parts),
                "$select": "SpatialDim,TimeDim,NumericValue,Low,High",
                "$top": 50_000,
            }

            logger.info(
                "Fetching indicator={} country={} years={}-{}",
                indicator_code,
                country or "ALL",
                year_start,
                year_end,
            )
            raw = self._rate_limited_get(url, params=params)

            # Persist to cache
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("w", encoding="utf-8") as fh:
                json.dump(raw, fh, ensure_ascii=False)
            logger.debug("Cached response to {}", cache_file)

        # --- Parse OData response ---
        records = raw.get("value", [])
        if not records:
            logger.warning(
                "No data returned for indicator={} country={}",
                indicator_code,
                country,
            )
            return pd.DataFrame(
                columns=["indicator_code", "country_code", "year", "value", "low", "high"]
            )

        rows = [
            {
                "indicator_code": indicator_code,
                "country_code": r.get("SpatialDim"),
                "year": r.get("TimeDim"),
                "value": r.get("NumericValue"),
                "low": r.get("Low"),
                "high": r.get("High"),
            }
            for r in records
        ]
        df = pd.DataFrame(rows)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")
        df["high"] = pd.to_numeric(df["high"], errors="coerce")

        logger.success(
            "Indicator={} fetched: {} rows", indicator_code, len(df)
        )
        return df

    def get_all_indicators(
        self,
        indicator_codes: list[str],
        country: Optional[str] = None,
        year_start: int = 2015,
        year_end: int = 2024,
    ) -> pd.DataFrame:
        """Fetch and concatenate data for multiple GHO indicators.

        Parameters
        ----------
        indicator_codes:
            List of WHO GHO indicator codes.
        country:
            Optional ISO 3-letter country code to restrict results.
        year_start:
            Earliest year (inclusive).
        year_end:
            Latest year (inclusive).

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all indicators stacked row-wise.
            Columns: indicator_code, country_code, year, value, low, high
        """
        frames: list[pd.DataFrame] = []
        for code in indicator_codes:
            try:
                df = self.get_indicator(
                    code,
                    country=country,
                    year_start=year_start,
                    year_end=year_end,
                )
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to fetch indicator={}: {}", code, exc
                )

        if not frames:
            logger.warning("No indicator data retrieved.")
            return pd.DataFrame(
                columns=["indicator_code", "country_code", "year", "value", "low", "high"]
            )

        combined = pd.concat(frames, ignore_index=True)
        logger.success(
            "All indicators fetched: {} rows across {} indicators",
            len(combined),
            combined["indicator_code"].nunique(),
        )
        return combined

    def get_disease_burden(
        self,
        country: Optional[str] = None,
        year_start: int = 2015,
        year_end: int = 2024,
    ) -> pd.DataFrame:
        """Fetch a pre-defined set of disease-burden indicators and pivot wide.

        Indicators fetched
        ------------------
        - WHOSIS_000001  – Life expectancy at birth
        - NCD_BMI_30A    – Obesity prevalence (BMI ≥ 30), adults
        - SA_0000001688  – Alcohol use disorders (prevalence)
        - M_Est_cigs_curr – Current tobacco smoking prevalence
        - MDG_0000000026 – Under-5 mortality rate
        - SDGPM25        – Population exposed to harmful PM2.5 levels
        - NCD_DIABETES   – Diabetes mellitus prevalence

        Parameters
        ----------
        country:
            Optional ISO 3-letter country code.
        year_start / year_end:
            Temporal range filter.

        Returns
        -------
        pd.DataFrame
            Wide format: one row per (country_code, year) with each
            indicator as a separate column.
        """
        frames: list[pd.DataFrame] = []
        for code, col_name in _DISEASE_BURDEN_INDICATORS.items():
            try:
                df = self.get_indicator(
                    code,
                    country=country,
                    year_start=year_start,
                    year_end=year_end,
                )
                if df.empty:
                    continue
                df = df[["country_code", "year", "value"]].rename(
                    columns={"value": col_name}
                )
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to fetch disease burden indicator={}: {}", code, exc)

        if not frames:
            logger.warning("No disease burden data retrieved.")
            return pd.DataFrame()

        # Merge all indicator frames on (country_code, year)
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["country_code", "year"], how="outer")

        merged = merged.sort_values(["country_code", "year"]).reset_index(drop=True)
        logger.success(
            "Disease burden pivot: {} rows, {} columns",
            len(merged),
            len(merged.columns),
        )
        return merged

    def get_health_expenditure(
        self,
        country: Optional[str] = None,
        year_start: int = 2015,
        year_end: int = 2024,
    ) -> pd.DataFrame:
        """Fetch health-expenditure indicators.

        Indicators fetched
        ------------------
        - GHED_CHEGDP_SHA2011    – Current health expenditure as % of GDP
        - GHED_CHEPCUSD_SHA2011  – Current health expenditure per capita (USD)

        Returns
        -------
        pd.DataFrame
            Wide format: one row per (country_code, year) with
            ``health_spend_pct_gdp`` and ``health_spend_per_capita_usd``.
        """
        frames: list[pd.DataFrame] = []
        for code, col_name in _HEALTH_EXPENDITURE_INDICATORS.items():
            try:
                df = self.get_indicator(
                    code,
                    country=country,
                    year_start=year_start,
                    year_end=year_end,
                )
                if df.empty:
                    continue
                df = df[["country_code", "year", "value"]].rename(
                    columns={"value": col_name}
                )
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to fetch health expenditure indicator={}: {}", code, exc
                )

        if not frames:
            logger.warning("No health expenditure data retrieved.")
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["country_code", "year"], how="outer")

        merged = merged.sort_values(["country_code", "year"]).reset_index(drop=True)
        logger.success(
            "Health expenditure pivot: {} rows", len(merged)
        )
        return merged

    def get_risk_factors(
        self,
        country: Optional[str] = None,
        year_start: int = 2015,
        year_end: int = 2024,
    ) -> pd.DataFrame:
        """Fetch modifiable NCD risk-factor indicators.

        Indicators fetched
        ------------------
        - NCD_BMI_30A    – Obesity prevalence
        - M_Est_cigs_curr – Tobacco smoking prevalence
        - SA_0000001688  – Alcohol use disorders
        - NCD_PAC        – Insufficient physical activity

        Returns
        -------
        pd.DataFrame
            Wide format: one row per (country_code, year).
        """
        frames: list[pd.DataFrame] = []
        for code, col_name in _RISK_FACTOR_INDICATORS.items():
            try:
                df = self.get_indicator(
                    code,
                    country=country,
                    year_start=year_start,
                    year_end=year_end,
                )
                if df.empty:
                    continue
                df = df[["country_code", "year", "value"]].rename(
                    columns={"value": col_name}
                )
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to fetch risk factor indicator={}: {}", code, exc
                )

        if not frames:
            logger.warning("No risk factor data retrieved.")
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["country_code", "year"], how="outer")

        merged = merged.sort_values(["country_code", "year"]).reset_index(drop=True)
        logger.success("Risk factors pivot: {} rows", len(merged))
        return merged
