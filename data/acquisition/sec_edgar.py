"""
SEC EDGAR Pharmaceutical Filings Client
=========================================
Production-quality client for fetching pharmaceutical company filings and
financial data from the SEC EDGAR XBRL/submissions APIs.

SEC Fair Access Policy requires a descriptive User-Agent header identifying
the application and a contact email. A ValueError is raised at initialisation
if the user-agent is not configured.

Usage
-----
    from data.acquisition.sec_edgar import SECEdgarClient

    client = SECEdgarClient()
    companies = client.search_pharma_companies()
    financials = client.get_financial_statements(cik="0000002178")

Environment variables
---------------------
SEC_EDGAR_BASE_URL  : str   Override default https://data.sec.gov
SEC_EDGAR_USER_AGENT: str   REQUIRED — e.g. "MyApp/1.0 contact@example.com"
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from requests.exceptions import RequestException

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_BASE_URL = "https://data.sec.gov"
_EDGAR_SEARCH_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
_MIN_REQUEST_INTERVAL = 0.1   # 10 req/s max per SEC Fair Access Policy
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0

_DEFAULT_SIC_CODES: list[str] = ["2836", "2835", "2830", "8011", "8049"]

_KEY_CONCEPTS: list[str] = [
    "Revenues",
    "ResearchAndDevelopmentExpense",
    "OperatingIncomeLoss",
    "CashAndCashEquivalentsAtCarryingValue",
    "LongTermDebt",
]

logger = logger.bind(module="sec_edgar")


class SECEdgarClient:
    """
    Client for the SEC EDGAR XBRL, submissions, and company-search APIs.

    Parameters
    ----------
    base_url:
        Base URL for the EDGAR data API. Falls back to ``SEC_EDGAR_BASE_URL``
        env var, then ``https://data.sec.gov``.
    user_agent:
        User-Agent string required by SEC Fair Access Policy.
        Falls back to ``SEC_EDGAR_USER_AGENT`` env var.
        Raises ``ValueError`` if neither is provided.
    cache_dir:
        Local directory for cached parquet files.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        user_agent: Optional[str] = None,
        cache_dir: str = "data/raw/sec_edgar",
    ) -> None:
        self.base_url: str = (
            base_url or os.getenv("SEC_EDGAR_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")

        resolved_ua = user_agent or os.getenv("SEC_EDGAR_USER_AGENT")
        if not resolved_ua:
            raise ValueError(
                "SEC EDGAR User-Agent is required by the SEC Fair Access Policy. "
                "Set SEC_EDGAR_USER_AGENT env var or pass user_agent= to the constructor. "
                "Example: 'HealthRiskAI/1.0 contact@example.com'"
            )
        self.user_agent: str = resolved_ua

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            }
        )
        self._last_request_ts: float = 0.0

        logger.info(
            "SECEdgarClient initialised | base_url={} user_agent='{}'",
            self.base_url,
            self.user_agent,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """
        GET request with User-Agent header, rate limiting, and retry logic.

        Parameters
        ----------
        url:
            Full request URL.
        params:
            Optional query-string parameters.

        Returns
        -------
        dict
            Parsed JSON response body.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            self._rate_wait()
            try:
                resp = self._session.get(url, params=params, timeout=30)
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
                return resp.json()  # type: ignore[return-value]

            except RequestException as exc:
                wait = _BACKOFF_BASE ** attempt
                if attempt == _MAX_RETRIES:
                    logger.error("GET {} failed after {} retries: {}", url, _MAX_RETRIES, exc)
                    raise
                logger.warning("Attempt {} error: {}. Retrying in {}s …", attempt, exc, wait)
                time.sleep(wait)

        return {}

    def _get_xml(self, url: str, params: Optional[dict] = None) -> str:
        """Fetch raw XML text with the same retry logic as _get."""
        for attempt in range(1, _MAX_RETRIES + 1):
            self._rate_wait()
            try:
                resp = self._session.get(url, params=params, timeout=30)
                self._last_request_ts = time.monotonic()

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", _BACKOFF_BASE ** attempt))
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    time.sleep(_BACKOFF_BASE ** attempt)
                    continue

                resp.raise_for_status()
                return resp.text

            except RequestException as exc:
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(_BACKOFF_BASE ** attempt)
                logger.warning("XML GET attempt {} error: {}", attempt, exc)

        return ""

    # ------------------------------------------------------------------
    # Company search
    # ------------------------------------------------------------------

    def search_pharma_companies(
        self,
        sic_codes: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Search EDGAR for pharmaceutical companies by SIC code.

        Queries the EDGAR full-text search ``browse-edgar`` endpoint (Atom XML
        output) for each SIC code and parses the Atom feed.

        Parameters
        ----------
        sic_codes:
            List of SIC codes to query. Defaults to
            ``['2836', '2835', '2830', '8011', '8049']``.

        Returns
        -------
        pd.DataFrame
            Columns: cik, company_name, sic_code, state, fiscal_year_end
        """
        codes = sic_codes or _DEFAULT_SIC_CODES
        all_records: list[dict] = []

        # Atom XML namespace
        atom_ns = "http://www.w3.org/2005/Atom"

        for sic in codes:
            logger.info("Searching EDGAR for SIC={}…", sic)
            params = {
                "action": "getcompany",
                "SIC": sic,
                "type": "",
                "dateb": "",
                "owner": "include",
                "count": "100",
                "search_text": "",
                "output": "atom",
            }
            try:
                xml_text = self._get_xml(_EDGAR_SEARCH_URL, params=params)
                if not xml_text:
                    continue

                root = ET.fromstring(xml_text)

                for entry in root.findall(f"{{{atom_ns}}}entry"):
                    # Extract fields from Atom entry
                    cik_el = entry.find(f"{{{atom_ns}}}id")
                    name_el = entry.find(f"{{{atom_ns}}}company-name")
                    if name_el is None:
                        # Fall back: look inside title
                        title_el = entry.find(f"{{{atom_ns}}}title")
                        name_text = title_el.text if title_el is not None else None
                    else:
                        name_text = name_el.text

                    # CIK is in the <id> or content field
                    cik_text = None
                    if cik_el is not None and cik_el.text:
                        # Typical format: .../CIK0000012345
                        for part in cik_el.text.split("/"):
                            if "CIK" in part.upper():
                                cik_text = part.upper().replace("CIK", "").lstrip("0") or "0"
                                break

                    # State and fiscal year end may appear in content
                    content_el = entry.find(f"{{{atom_ns}}}content")
                    state_text = None
                    fiscal_ye = None
                    if content_el is not None and content_el.text:
                        for token in content_el.text.split():
                            if len(token) == 2 and token.isupper():
                                state_text = token
                            if len(token) == 4 and token.isdigit():
                                fiscal_ye = token

                    all_records.append(
                        {
                            "cik": cik_text,
                            "company_name": name_text,
                            "sic_code": sic,
                            "state": state_text,
                            "fiscal_year_end": fiscal_ye,
                        }
                    )

            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to parse EDGAR Atom for SIC={}: {}", sic, exc)

        if not all_records:
            logger.warning("No pharma companies found for SIC codes={}", codes)
            return pd.DataFrame(
                columns=["cik", "company_name", "sic_code", "state", "fiscal_year_end"]
            )

        df = pd.DataFrame(all_records).drop_duplicates(subset=["cik"])
        logger.success("search_pharma_companies: {} companies found", len(df))
        return df

    # ------------------------------------------------------------------
    # Filings
    # ------------------------------------------------------------------

    def get_company_filings(
        self,
        cik: str,
        filing_type: str = "10-K",
        count: int = 8,
    ) -> list[dict]:
        """
        Retrieve recent filings for a company from the EDGAR submissions API.

        Parameters
        ----------
        cik:
            Company CIK (numeric string, zero-padded to 10 digits internally).
        filing_type:
            SEC form type filter (e.g. ``'10-K'``, ``'10-Q'``).
        count:
            Maximum number of filings to return.

        Returns
        -------
        list[dict]
            Each dict contains: accessionNumber, filingDate, form,
            primaryDocument, reportDate
        """
        padded_cik = cik.lstrip("0").zfill(10)
        url = f"{self.base_url}/submissions/CIK{padded_cik}.json"
        logger.info("Fetching filings for CIK={} type={}", cik, filing_type)

        try:
            data = self._get(url)
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not fetch submissions for CIK={}: {}", cik, exc)
            return []

        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            logger.warning("No recent filings found for CIK={}", cik)
            return []

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        results: list[dict] = []
        for i, form in enumerate(forms):
            if form != filing_type:
                continue
            results.append(
                {
                    "accessionNumber": accessions[i] if i < len(accessions) else None,
                    "filingDate": dates[i] if i < len(dates) else None,
                    "form": form,
                    "primaryDocument": primary_docs[i] if i < len(primary_docs) else None,
                    "reportDate": report_dates[i] if i < len(report_dates) else None,
                }
            )
            if len(results) >= count:
                break

        logger.success("CIK={} | {} {} filings found", cik, len(results), filing_type)
        return results

    # ------------------------------------------------------------------
    # XBRL concept data
    # ------------------------------------------------------------------

    def get_company_concept(
        self,
        cik: str,
        concept: str,
        taxonomy: str = "us-gaap",
    ) -> pd.DataFrame:
        """
        Fetch time-series data for a single XBRL concept for a company.

        Parameters
        ----------
        cik:
            Company CIK.
        concept:
            XBRL concept name (e.g. ``'Revenues'``,
            ``'ResearchAndDevelopmentExpense'``).
        taxonomy:
            XBRL taxonomy namespace, defaults to ``'us-gaap'``.

        Returns
        -------
        pd.DataFrame
            Annual (10-K) values only.
            Columns: accn, end, val, form, filed
        """
        padded_cik = cik.lstrip("0").zfill(10)
        url = (
            f"{self.base_url}/api/xbrl/companyconcept/"
            f"CIK{padded_cik}/{taxonomy}/{concept}.json"
        )
        logger.debug("get_company_concept | CIK={} concept={}", cik, concept)

        try:
            data = self._get(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch concept {} for CIK={}: {}", concept, cik, exc)
            return pd.DataFrame(columns=["accn", "end", "val", "form", "filed"])

        units = data.get("units", {})
        usd_records = units.get("USD", [])

        if not usd_records:
            # Some concepts use shares or pure units
            for unit_key, records in units.items():
                if records:
                    usd_records = records
                    break

        if not usd_records:
            logger.warning("No unit data for concept={} CIK={}", concept, cik)
            return pd.DataFrame(columns=["accn", "end", "val", "form", "filed"])

        df = pd.DataFrame(usd_records)[["accn", "end", "val", "form", "filed"]]
        # Filter to annual (10-K) values only
        df = df[df["form"].isin(["10-K", "10-K/A"])].copy()
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
        df = df.sort_values("end").drop_duplicates(subset=["end"], keep="last")
        df["concept"] = concept

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------

    def get_financial_statements(self, cik: str) -> pd.DataFrame:
        """
        Build a wide-format annual financial statement DataFrame for a company
        by fetching each of the five key XBRL concepts.

        Parameters
        ----------
        cik:
            Company CIK.

        Returns
        -------
        pd.DataFrame
            One row per fiscal year.
            Columns: year, revenue, rd_expense, operating_income, cash,
            long_term_debt, rd_intensity, operating_margin
        """
        logger.info("get_financial_statements | CIK={}", cik)

        concept_col_map = {
            "Revenues": "revenue",
            "ResearchAndDevelopmentExpense": "rd_expense",
            "OperatingIncomeLoss": "operating_income",
            "CashAndCashEquivalentsAtCarryingValue": "cash",
            "LongTermDebt": "long_term_debt",
        }

        frames: list[pd.DataFrame] = []
        for concept, col_name in concept_col_map.items():
            df = self.get_company_concept(cik, concept)
            if df.empty:
                continue
            slim = df[["end", "val"]].rename(columns={"val": col_name, "end": "period_end"})
            slim["year"] = pd.DatetimeIndex(slim["period_end"]).year
            slim = slim.drop_duplicates(subset=["year"], keep="last")[["year", col_name]]
            frames.append(slim)

        if not frames:
            logger.warning("No financial statement data for CIK={}", cik)
            return pd.DataFrame()

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="year", how="outer")

        merged = merged.sort_values("year").reset_index(drop=True)

        # Derived metrics
        if "revenue" in merged.columns and "rd_expense" in merged.columns:
            merged["rd_intensity"] = (
                merged["rd_expense"] / merged["revenue"].replace(0, np.nan)
            )
        else:
            merged["rd_intensity"] = np.nan

        if "operating_income" in merged.columns and "revenue" in merged.columns:
            merged["operating_margin"] = (
                merged["operating_income"] / merged["revenue"].replace(0, np.nan)
            )
        else:
            merged["operating_margin"] = np.nan

        merged["cik"] = cik
        logger.success("Financial statements CIK={}: {} annual periods", cik, len(merged))
        return merged

    # ------------------------------------------------------------------
    # Pharma metrics
    # ------------------------------------------------------------------

    def extract_pharma_metrics(self, cik: str) -> dict:
        """
        Compute high-level pharmaceutical financial metrics from XBRL data.

        Parameters
        ----------
        cik:
            Company CIK.

        Returns
        -------
        dict
            Keys: cik, revenue_concentration (placeholder), rd_productivity,
            revenue_cagr_5yr, rd_cagr_5yr, operating_margin_avg_3yr,
            data_years, metadata
        """
        logger.info("extract_pharma_metrics | CIK={}", cik)
        fs = self.get_financial_statements(cik)

        metrics: dict = {
            "cik": cik,
            "revenue_concentration": 0.4,  # Placeholder — requires product-level data
            "_revenue_concentration_note": (
                "Placeholder value 0.4. True revenue concentration requires "
                "product-segment disclosures not available in structured XBRL."
            ),
            "rd_productivity": None,
            "revenue_cagr_5yr": None,
            "rd_cagr_5yr": None,
            "operating_margin_avg_3yr": None,
            "data_years": [],
            "metadata": {"source": "SEC EDGAR XBRL", "cik": cik},
        }

        if fs.empty:
            logger.warning("No financial data for CIK={}; returning partial metrics.", cik)
            return metrics

        metrics["data_years"] = sorted(fs["year"].dropna().astype(int).tolist())

        # R&D productivity: total revenue / total R&D over available history (up to 10yr)
        fs_10 = fs.tail(10)
        if "revenue" in fs_10.columns and "rd_expense" in fs_10.columns:
            total_rev = fs_10["revenue"].sum(skipna=True)
            total_rd = fs_10["rd_expense"].sum(skipna=True)
            if total_rd and total_rd != 0:
                metrics["rd_productivity"] = float(total_rev / total_rd)

        def _cagr(series: pd.Series, n: int) -> Optional[float]:
            """Compute CAGR over last n years."""
            valid = series.dropna()
            if len(valid) < 2:
                return None
            tail = valid.tail(n + 1)
            if len(tail) < 2:
                return None
            start_val, end_val = float(tail.iloc[0]), float(tail.iloc[-1])
            years = len(tail) - 1
            if start_val <= 0 or end_val <= 0:
                return None
            return float((end_val / start_val) ** (1 / years) - 1)

        if "revenue" in fs.columns:
            metrics["revenue_cagr_5yr"] = _cagr(fs["revenue"], 5)

        if "rd_expense" in fs.columns:
            metrics["rd_cagr_5yr"] = _cagr(fs["rd_expense"], 5)

        if "operating_margin" in fs.columns:
            recent_3 = fs["operating_margin"].dropna().tail(3)
            if not recent_3.empty:
                metrics["operating_margin_avg_3yr"] = float(recent_3.mean())

        logger.success("Pharma metrics for CIK={}: {}", cik, metrics)
        return metrics

    # ------------------------------------------------------------------
    # Universe builder
    # ------------------------------------------------------------------

    def get_pharma_universe(
        self,
        sic_codes: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Build a comprehensive financial dataset for all pharma companies
        found via ``search_pharma_companies``.

        For each company, calls ``get_financial_statements`` and appends the
        result. Companies where XBRL data is unavailable are skipped.

        Parameters
        ----------
        sic_codes:
            Optional list of SIC codes to include.

        Returns
        -------
        pd.DataFrame
            Stacked financial statements for all pharma companies, with
            ``company_name`` and ``sic_code`` columns prepended.
        """
        logger.info("get_pharma_universe | sic_codes={}", sic_codes or _DEFAULT_SIC_CODES)
        companies = self.search_pharma_companies(sic_codes=sic_codes)

        if companies.empty:
            logger.warning("No companies found; returning empty DataFrame.")
            return pd.DataFrame()

        all_frames: list[pd.DataFrame] = []

        for _, row in companies.iterrows():
            cik = row.get("cik")
            name = row.get("company_name", "")
            sic = row.get("sic_code", "")

            if not cik:
                continue

            try:
                fs = self.get_financial_statements(str(cik))
                if fs.empty:
                    logger.debug("No financial data for {} (CIK={})", name, cik)
                    continue
                fs["company_name"] = name
                fs["sic_code"] = sic
                all_frames.append(fs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed financial statements for CIK={}: {}", cik, exc)

        if not all_frames:
            logger.warning("No financial data retrieved across pharma universe.")
            return pd.DataFrame()

        universe = pd.concat(all_frames, ignore_index=True)
        logger.success("Pharma universe: {} rows across {} companies", len(universe), len(all_frames))
        return universe
