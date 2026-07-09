"""
FDA FAERS openFDA API Client
=============================
Production-quality client for querying FDA Adverse Event Reporting System (FAERS)
data via the openFDA public REST API and performing pharmacovigilance signal detection.

Usage:
    from data.acquisition.fda_faers import FDAFAERSClient

    client = FDAFAERSClient()
    df = client.get_adverse_events(drug_name="aspirin", serious=1, limit=500)
    signals = client.detect_safety_signals(["aspirin", "ibuprofen"])
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from scipy import stats as scipy_stats

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://api.fda.gov/drug"
MAX_RETRIES = 3
BACKOFF_BASE = 2.0          # seconds, doubled on each retry
OPENFDA_PAGE_SIZE = 100     # hard max enforced by openFDA
# Rate limits
RATE_LIMIT_WITH_KEY = 240   # requests / minute
RATE_LIMIT_NO_KEY = 40      # requests / minute


class FDAFAERSClient:
    """Client for the FDA openFDA Drug Adverse Event API (FAERS).

    Environment variables:
        OPENFDA_API_KEY  – openFDA API key (optional; raises rate limits).
        OPENFDA_BASE_URL – override the default API base URL.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str = "data/raw/faers",
    ) -> None:
        self.api_key: str | None = api_key or os.getenv("OPENFDA_API_KEY")
        self.base_url: str = (
            base_url or os.getenv("OPENFDA_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Sliding-window rate-limit state
        self._request_timestamps: list[float] = []

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "HealthRiskAI/1.0 (fda-faers-client)",
            }
        )

        key_status = "with API key" if self.api_key else "without API key"
        logger.info(
            "FDAFAERSClient initialised | {} | base_url={} | cache_dir={}",
            key_status,
            self.base_url,
            self.cache_dir,
        )

    # ------------------------------------------------------------------
    # Core HTTP helper
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict:
        """Rate-limited GET with automatic retry on 429 / 5xx.

        The API key is injected automatically if available.

        Args:
            endpoint: Path relative to ``base_url``, e.g. ``/event.json``.
            params:   Query-string parameters (do NOT include ``api_key`` here).

        Returns:
            Parsed JSON dict from the API response.

        Raises:
            requests.HTTPError: After all retries are exhausted.
        """
        url = f"{self.base_url}{endpoint}"
        params = dict(params or {})
        if self.api_key:
            params["api_key"] = self.api_key

        for attempt in range(MAX_RETRIES + 1):
            self._enforce_rate_limit()
            logger.debug(
                "GET {} | params_keys={} | attempt={}", url, list(params.keys()), attempt
            )
            try:
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < MAX_RETRIES:
                        wait = BACKOFF_BASE * (2 ** attempt)
                        logger.warning(
                            "HTTP {} — retrying in {:.1f}s (attempt {}/{})",
                            resp.status_code,
                            wait,
                            attempt + 1,
                            MAX_RETRIES,
                        )
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()

                # 404 from openFDA typically means zero results — return empty structure
                if resp.status_code == 404:
                    logger.debug("404 from openFDA — returning empty results")
                    return {"results": [], "meta": {"results": {"total": 0}}}

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    logger.warning("Timeout — retrying in {:.1f}s", wait)
                    time.sleep(wait)
                else:
                    raise
            except requests.exceptions.ConnectionError:
                if attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    logger.warning("ConnectionError — retrying in {:.1f}s", wait)
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"_get failed after {MAX_RETRIES} retries for {url}")

    def _enforce_rate_limit(self) -> None:
        """Block until the per-minute rate limit window allows a new request."""
        limit = RATE_LIMIT_WITH_KEY if self.api_key else RATE_LIMIT_NO_KEY
        window = 60.0  # one minute

        now = time.monotonic()
        # Prune timestamps outside the window
        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < window
        ]

        if len(self._request_timestamps) >= limit:
            oldest = self._request_timestamps[0]
            sleep_for = window - (now - oldest) + 0.05  # small buffer
            if sleep_for > 0:
                logger.debug("Rate limit — sleeping {:.2f}s", sleep_for)
                time.sleep(sleep_for)

        self._request_timestamps.append(time.monotonic())

    # ------------------------------------------------------------------
    # Public API – adverse events
    # ------------------------------------------------------------------

    def get_adverse_events(
        self,
        drug_name: str | None = None,
        reaction: str | None = None,
        serious: int | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Retrieve adverse event reports matching the given criteria.

        Args:
            drug_name:  Medicinal product name to filter by.
            reaction:   MedDRA reaction term to filter by.
            serious:    1 = serious reports only; 0 = non-serious; None = all.
            date_start: Start of receive date range, format ``YYYYMMDD``.
            date_end:   End of receive date range, format ``YYYYMMDD``.
            limit:      Maximum number of records to return.

        Returns:
            Flat :class:`pd.DataFrame`, one row per adverse event report.
        """
        search_query = self._build_search_query(
            drug_name=drug_name,
            reaction=reaction,
            serious=serious,
            date_start=date_start,
            date_end=date_end,
        )

        records: list[dict] = []
        fetched = 0
        skip = 0

        logger.info(
            "get_adverse_events | query='{}' | limit={}", search_query, limit
        )

        while fetched < limit:
            batch_size = min(OPENFDA_PAGE_SIZE, limit - fetched)
            params: dict[str, Any] = {
                "search": search_query,
                "limit": batch_size,
                "skip": skip,
            }

            data = self._get("/event.json", params=params)
            results = data.get("results", [])

            if not results:
                logger.debug("No more results at skip={}", skip)
                break

            for report in results:
                records.append(self._parse_adverse_event(report))

            fetched += len(results)
            skip += len(results)

            total_available = (
                data.get("meta", {}).get("results", {}).get("total", 0)
            )
            logger.debug(
                "Fetched {}/{} (total_available={})", fetched, limit, total_available
            )

            if fetched >= total_available:
                break

        if not records:
            logger.warning("get_adverse_events returned no records")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info("get_adverse_events complete | {} records", len(df))
        return df

    # ------------------------------------------------------------------
    # Public API – pharmacovigilance signal detection
    # ------------------------------------------------------------------

    def compute_disproportionality(self, drug: str, reaction: str) -> dict:
        """Compute standard 2×2 pharmacovigilance disproportionality metrics.

        The contingency table is:

        +-----------------------+------------+-----------------+
        |                       | Reaction   | No Reaction     |
        +-----------------------+------------+-----------------+
        | Drug                  | a          | b               |
        | No Drug               | c          | d               |
        +-----------------------+------------+-----------------+

        Metrics computed:
            - PRR (Proportional Reporting Ratio)
            - ROR (Reporting Odds Ratio)
            - Chi-square (Yates-corrected via scipy)
            - p-value
            - significant: True if p < 0.05 AND ROR > threshold_ror

        Args:
            drug:     Drug name.
            reaction: MedDRA reaction PT term.

        Returns:
            Dict with all contingency counts, PRR, ROR, chi_square, p_value,
            significant.
        """
        logger.debug("compute_disproportionality | drug='{}' reaction='{}'", drug, reaction)

        # a = drug AND reaction
        a = self._count_reports(drug_name=drug, reaction=reaction)
        # b = drug AND NOT reaction (drug reports - a)
        n_drug = self._count_reports(drug_name=drug)
        b = max(n_drug - a, 0)
        # c = NOT drug AND reaction
        n_reaction = self._count_reports(reaction=reaction)
        c = max(n_reaction - a, 0)
        # d = total - a - b - c
        n_total = self._count_reports()
        d = max(n_total - a - b - c, 0)

        n_drug_safe = a + b
        n_no_drug = c + d

        # PRR = (a / n_drug) / (c / n_no_drug)
        prr: float | None = None
        if n_drug_safe > 0 and n_no_drug > 0 and c > 0:
            prr = round((a / n_drug_safe) / (c / n_no_drug), 4)

        # ROR = (a * d) / (b * c)
        ror: float | None = None
        if b > 0 and c > 0:
            ror = round((a * d) / (b * c), 4)
        elif a > 0 and b == 0:
            # Infinite ROR — use a large sentinel
            ror = float("inf")

        # Chi-square via scipy 2×2 contingency table
        chi_square: float | None = None
        p_value: float | None = None
        contingency = [[a, b], [c, d]]
        try:
            chi2_result = scipy_stats.chi2_contingency(contingency, correction=True)
            chi_square = round(float(chi2_result.statistic), 4)
            p_value = round(float(chi2_result.pvalue), 6)
        except Exception as exc:
            logger.warning("chi2_contingency failed: {}", exc)

        significant = (
            p_value is not None
            and p_value < 0.05
            and ror is not None
            and ror != float("inf")
            and ror > 2.0
        )

        return {
            "drug": drug,
            "reaction": reaction,
            "n_drug_reaction": a,
            "n_drug_no_reaction": b,
            "n_no_drug_reaction": c,
            "n_neither": d,
            "n_drug_total": n_drug_safe,
            "n_total": n_total,
            "prr": prr,
            "ror": ror,
            "chi_square": chi_square,
            "p_value": p_value,
            "significant": significant,
        }

    def detect_safety_signals(
        self,
        drugs: list[str],
        threshold_ror: float = 2.0,
    ) -> pd.DataFrame:
        """Detect disproportionality signals across a list of drugs.

        For each drug, the top 20 most-reported reactions are identified via the
        openFDA count endpoint, then disproportionality metrics are computed for
        each drug-reaction pair.

        Args:
            drugs:         List of drug names to analyse.
            threshold_ror: ROR threshold used in the ``significant`` flag.

        Returns:
            DataFrame sorted by ROR descending, with columns: drug, reaction,
            ror, prr, chi_square, p_value, significant, n_reports.
        """
        rows: list[dict] = []

        for drug in drugs:
            logger.info("detect_safety_signals | drug='{}'", drug)
            top_reactions = self._get_top_reactions(drug, top_n=20)

            for reaction, n_reports in top_reactions:
                try:
                    disp = self.compute_disproportionality(drug, reaction)
                    rows.append(
                        {
                            "drug": drug,
                            "reaction": reaction,
                            "ror": disp["ror"],
                            "prr": disp["prr"],
                            "chi_square": disp["chi_square"],
                            "p_value": disp["p_value"],
                            "significant": disp["significant"],
                            "n_reports": n_reports,
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "Skipping {}/{} — {}", drug, reaction, exc
                    )

        if not rows:
            return pd.DataFrame(
                columns=["drug", "reaction", "ror", "prr", "chi_square",
                         "p_value", "significant", "n_reports"]
            )

        df = pd.DataFrame(rows)
        # Replace inf for sort stability, then put them at the top
        df["_ror_sort"] = df["ror"].apply(
            lambda v: 1e12 if v == float("inf") or (isinstance(v, float) and math.isinf(v)) else (v or 0)
        )
        df = df.sort_values("_ror_sort", ascending=False).drop(columns=["_ror_sort"])
        df = df.reset_index(drop=True)
        logger.info("detect_safety_signals complete | {} drug-reaction pairs", len(df))
        return df

    # ------------------------------------------------------------------
    # Public API – pharmacovigilance report
    # ------------------------------------------------------------------

    def build_pharmacovigilance_report(self, drug_list: list[str]) -> pd.DataFrame:
        """Build a summary pharmacovigilance report across a list of drugs.

        For each drug, the following are computed:
            - total_reports
            - death_reports
            - serious_reports
            - top_5_reactions (comma-separated string)
            - safety_signals_count (disproportionality-significant pairs)

        Args:
            drug_list: List of drug names.

        Returns:
            DataFrame with one row per drug.
        """
        rows: list[dict] = []

        for drug in drug_list:
            logger.info("build_pharmacovigilance_report | drug='{}'", drug)

            total_reports = self._count_reports(drug_name=drug)
            death_reports = self._count_reports(drug_name=drug, serious_death=True)
            serious_reports = self._count_reports(drug_name=drug, serious=1)

            top_reactions = self._get_top_reactions(drug, top_n=5)
            top_5_reactions = ", ".join(r for r, _ in top_reactions)

            # Count significant safety signals from detect_safety_signals
            signals_df = self.detect_safety_signals([drug], threshold_ror=2.0)
            safety_signals_count = int(signals_df["significant"].sum()) if not signals_df.empty else 0

            rows.append(
                {
                    "drug": drug,
                    "total_reports": total_reports,
                    "death_reports": death_reports,
                    "serious_reports": serious_reports,
                    "top_5_reactions": top_5_reactions,
                    "safety_signals_count": safety_signals_count,
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(
            "build_pharmacovigilance_report complete | {} drugs", len(df)
        )
        return df

    # ------------------------------------------------------------------
    # Public API – drug label
    # ------------------------------------------------------------------

    def get_drug_label(self, drug_name: str) -> dict:
        """Retrieve FDA drug label information for a given brand name.

        Args:
            drug_name: Brand name of the drug.

        Returns:
            Structured dict with keys: brand_name, generic_name, warnings,
            contraindications, adverse_reactions, boxed_warnings.
        """
        logger.info("get_drug_label | drug='{}'", drug_name)
        params = {"search": f'openfda.brand_name:"{drug_name}"', "limit": 1}
        data = self._get("/label.json", params=params)
        results = data.get("results", [])

        if not results:
            logger.warning("No label found for drug='{}'", drug_name)
            return {
                "brand_name": drug_name,
                "generic_name": None,
                "warnings": None,
                "contraindications": None,
                "adverse_reactions": None,
                "boxed_warnings": None,
            }

        label = results[0]
        openfda = label.get("openfda", {})

        def first_or_none(lst: list | None) -> str | None:
            if lst and isinstance(lst, list):
                return lst[0]
            return None

        brand_name = first_or_none(openfda.get("brand_name"))
        generic_name = first_or_none(openfda.get("generic_name"))

        # openFDA returns these as lists of text strings
        warnings = label.get("warnings") or label.get("warnings_and_cautions")
        if isinstance(warnings, list):
            warnings = " ".join(warnings)

        contraindications = label.get("contraindications")
        if isinstance(contraindications, list):
            contraindications = " ".join(contraindications)

        adverse_reactions = label.get("adverse_reactions")
        if isinstance(adverse_reactions, list):
            adverse_reactions = " ".join(adverse_reactions)

        boxed_warnings = label.get("boxed_warning")
        if isinstance(boxed_warnings, list):
            boxed_warnings = " ".join(boxed_warnings)

        return {
            "brand_name": brand_name or drug_name,
            "generic_name": generic_name,
            "warnings": warnings,
            "contraindications": contraindications,
            "adverse_reactions": adverse_reactions,
            "boxed_warnings": boxed_warnings,
        }

    # ------------------------------------------------------------------
    # Query builder
    # ------------------------------------------------------------------

    def _build_search_query(
        self,
        drug_name: str | None = None,
        reaction: str | None = None,
        serious: int | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> str:
        """Build an openFDA search query string from individual filter parameters.

        All terms are ANDed together. Phrase values are quoted.

        Args:
            drug_name:  Filter by medicinal product name.
            reaction:   Filter by MedDRA reaction PT term.
            serious:    1 = serious; 0 = non-serious.
            date_start: Start of receive-date range (``YYYYMMDD``).
            date_end:   End of receive-date range (``YYYYMMDD``).

        Returns:
            openFDA query string, e.g.:
            ``'patient.drug.medicinalproduct:"aspirin"+AND+serious:1'``
        """
        parts: list[str] = []

        if drug_name:
            # Search across both brand and generic name fields
            escaped = _escape_openfda_value(drug_name)
            parts.append(
                f'(patient.drug.medicinalproduct:"{escaped}"+OR+'
                f'patient.drug.openfda.brand_name:"{escaped}"+OR+'
                f'patient.drug.openfda.generic_name:"{escaped}")'
            )

        if reaction:
            escaped_r = _escape_openfda_value(reaction)
            parts.append(f'patient.reaction.reactionmeddrapt:"{escaped_r}"')

        if serious is not None:
            parts.append(f"serious:{int(serious)}")

        if date_start and date_end:
            parts.append(f"receivedate:[{date_start}+TO+{date_end}]")
        elif date_start:
            parts.append(f"receivedate:[{date_start}+TO+99991231]")
        elif date_end:
            parts.append(f"receivedate:[19000101+TO+{date_end}]")

        if not parts:
            return "*:*"

        return "+AND+".join(parts)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> str:
        """Return absolute cache file path for the given key.

        Args:
            key: Arbitrary string key.

        Returns:
            Absolute path string under ``cache_dir``.
        """
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return str(self.cache_dir / f"{safe_key}.json")

    def _load_cache(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read cache {}: {}", path, exc)
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except OSError as exc:
            logger.warning("Failed to write cache {}: {}", path, exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_reports(
        self,
        drug_name: str | None = None,
        reaction: str | None = None,
        serious: int | None = None,
        serious_death: bool = False,
    ) -> int:
        """Return the total number of reports matching the given criteria.

        Uses the openFDA ``count`` parameter to fetch only metadata.

        Args:
            drug_name:    Optional drug name filter.
            reaction:     Optional reaction term filter.
            serious:      Optional seriousness filter (1 = serious).
            serious_death: If True, filter for ``seriousnessdeath=1``.

        Returns:
            Integer count of matching reports, or 0 on error.
        """
        search_parts: list[str] = []

        if drug_name:
            escaped = _escape_openfda_value(drug_name)
            search_parts.append(
                f'(patient.drug.medicinalproduct:"{escaped}"+OR+'
                f'patient.drug.openfda.brand_name:"{escaped}")'
            )
        if reaction:
            escaped_r = _escape_openfda_value(reaction)
            search_parts.append(f'patient.reaction.reactionmeddrapt:"{escaped_r}"')
        if serious is not None:
            search_parts.append(f"serious:{int(serious)}")
        if serious_death:
            search_parts.append("seriousnessdeath:1")

        search_query = "+AND+".join(search_parts) if search_parts else "*:*"
        params: dict[str, Any] = {
            "search": search_query,
            "limit": 1,
        }

        try:
            data = self._get("/event.json", params=params)
            total = data.get("meta", {}).get("results", {}).get("total", 0)
            return int(total)
        except Exception as exc:
            logger.warning("_count_reports failed for query='{}': {}", search_query, exc)
            return 0

    def _get_top_reactions(
        self, drug_name: str, top_n: int = 20
    ) -> list[tuple[str, int]]:
        """Return the top N most-reported MedDRA reactions for a drug.

        Uses the openFDA ``count`` endpoint to get reaction frequencies.

        Args:
            drug_name: Drug name to filter by.
            top_n:     Number of top reactions to retrieve.

        Returns:
            List of (reaction_term, count) tuples sorted by count descending.
        """
        escaped = _escape_openfda_value(drug_name)
        search_query = (
            f'(patient.drug.medicinalproduct:"{escaped}"+OR+'
            f'patient.drug.openfda.brand_name:"{escaped}")'
        )
        params: dict[str, Any] = {
            "search": search_query,
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": top_n,
        }

        try:
            data = self._get("/event.json", params=params)
            results = data.get("results", [])
            return [(r["term"], r["count"]) for r in results]
        except Exception as exc:
            logger.warning(
                "_get_top_reactions failed for drug='{}': {}", drug_name, exc
            )
            return []

    @staticmethod
    def _parse_adverse_event(report: dict) -> dict:
        """Flatten a single FAERS adverse event report into a dict.

        Extracts the most clinically relevant fields. Lists (drugs, reactions)
        are serialised to ``|``-separated strings for tabular representation.
        """
        patient = report.get("patient", {})

        # Patient demographics
        age_raw = patient.get("patientonsetage")
        age: float | None = None
        try:
            age = float(age_raw) if age_raw is not None else None
        except (TypeError, ValueError):
            pass

        sex_code = patient.get("patientsex")
        sex_map = {"1": "male", "2": "female", "0": "unknown"}
        sex: str = sex_map.get(str(sex_code), str(sex_code) if sex_code else "")

        # Drugs
        drugs_raw = patient.get("drug", []) or []
        drug_names: list[str] = []
        drug_indications: list[str] = []
        for d in drugs_raw:
            name = d.get("medicinalproduct", "")
            if name:
                drug_names.append(name.strip())
            indication = d.get("drugindication", "")
            if indication:
                drug_indications.append(indication.strip())

        # Reactions
        reactions_raw = patient.get("reaction", []) or []
        reactions: list[str] = [
            r.get("reactionmeddrapt", "").strip()
            for r in reactions_raw
            if r.get("reactionmeddrapt")
        ]

        return {
            "safetyreportid": report.get("safetyreportid", ""),
            "receivedate": report.get("receivedate", ""),
            "serious": report.get("serious"),
            "seriousnessdeath": report.get("seriousnessdeath"),
            "patient_age": age,
            "patient_sex": sex,
            "drugs": " | ".join(drug_names),
            "drug_indications": " | ".join(drug_indications),
            "reactions": " | ".join(reactions),
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _escape_openfda_value(value: str) -> str:
    """Escape special characters in an openFDA query value.

    openFDA uses Lucene query syntax; this escapes characters that have
    special meaning in that syntax but preserves the readability of the value.

    Args:
        value: Raw string value to escape.

    Returns:
        Escaped string safe to embed in an openFDA search query.
    """
    # Characters with special meaning in Lucene: + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
    special_chars = r'+-&|!(){}[]^~*?:\\/'
    result = []
    for ch in value:
        if ch in special_chars:
            result.append(f"\\{ch}")
        else:
            result.append(ch)
    return "".join(result)
