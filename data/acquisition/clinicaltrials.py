"""
ClinicalTrials.gov v2 API Client
=================================
Production-quality client for fetching, parsing, and analyzing clinical trial data
from the ClinicalTrials.gov v2 REST API.

Usage:
    from data.acquisition.clinicaltrials import ClinicalTrialsClient

    client = ClinicalTrialsClient()
    df = client.search_trials(query="diabetes", phase=["PHASE3"], status=["RECRUITING"])
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger
from ratelimit import limits, sleep_and_retry

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://clinicaltrials.gov/api/v2"
MAX_RETRIES = 3
BACKOFF_BASE = 2.0          # seconds – will be multiplied by 2^attempt
REQUESTS_PER_SECOND = 10


class ClinicalTrialsClient:
    """Client for the ClinicalTrials.gov v2 REST API.

    Environment variables:
        CLINICALTRIALS_BASE_URL – override the default API base URL.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        base_url: str | None = None,
        cache_dir: str = "data/raw/clinicaltrials",
    ) -> None:
        self.base_url: str = (
            base_url
            or os.getenv("CLINICALTRIALS_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "HealthRiskAI/1.0 (clinical-trials-client)",
            }
        )
        logger.info(
            "ClinicalTrialsClient initialised | base_url={} cache_dir={}",
            self.base_url,
            self.cache_dir,
        )

    # ------------------------------------------------------------------
    # Core HTTP helper
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict:
        """Rate-limited GET with retry on 429 / 5xx responses.

        Args:
            endpoint: Path relative to base_url (e.g. ``/studies``).
            params:   Query-string parameters.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            requests.HTTPError: When all retries are exhausted.
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        for attempt in range(MAX_RETRIES + 1):
            try:
                # Honour 10 req/s rate limit via simple token-bucket approach
                self._rate_limit()
                logger.debug("GET {} | params={} attempt={}", url, params, attempt)
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < MAX_RETRIES:
                        wait = BACKOFF_BASE * (2 ** attempt)
                        logger.warning(
                            "HTTP {} — retrying in {:.1f}s (attempt {}/{})",
                            response.status_code,
                            wait,
                            attempt + 1,
                            MAX_RETRIES,
                        )
                        time.sleep(wait)
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

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

        # Should never reach here
        raise RuntimeError(f"_get failed after {MAX_RETRIES} retries: {url}")

    # Minimal token-bucket state
    _last_request_times: list[float] = []

    def _rate_limit(self) -> None:
        """Enforce ≤10 requests per second using a sliding window."""
        now = time.monotonic()
        # Keep only timestamps within the last second
        self._last_request_times = [
            t for t in self._last_request_times if now - t < 1.0
        ]
        if len(self._last_request_times) >= REQUESTS_PER_SECOND:
            sleep_for = 1.0 - (now - self._last_request_times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._last_request_times.append(time.monotonic())

    # ------------------------------------------------------------------
    # Public API – search
    # ------------------------------------------------------------------

    def search_trials(
        self,
        query: str | None = None,
        phase: list[str] | None = None,
        status: list[str] | None = None,
        page_size: int = 100,
        max_pages: int = 50,
    ) -> pd.DataFrame:
        """Search for clinical trials with pagination.

        Args:
            query:     Free-text search term (maps to ``query.term``).
            phase:     List of phase filters, e.g. ``["PHASE2", "PHASE3"]``.
            status:    List of overall-status filters, e.g. ``["RECRUITING"]``.
            page_size: Trials per page (max 1000 per API docs).
            max_pages: Safety cap on the number of pages fetched.

        Returns:
            Flat :class:`pd.DataFrame`, one row per trial.
        """
        params: dict[str, Any] = {"pageSize": page_size}
        if query:
            params["query.term"] = query
        if phase:
            params["filter.phase"] = "|".join(phase)
        if status:
            params["filter.overallStatus"] = "|".join(status)

        records: list[dict] = []
        page_token: str | None = None
        page_num = 0

        logger.info(
            "search_trials | query={} phase={} status={} page_size={}",
            query,
            phase,
            status,
            page_size,
        )

        while page_num < max_pages:
            if page_token:
                params["pageToken"] = page_token
            elif "pageToken" in params:
                del params["pageToken"]

            data = self._get("/studies", params=params)
            studies = data.get("studies", [])
            logger.debug("Page {} — {} studies returned", page_num + 1, len(studies))

            for study in studies:
                records.append(self._parse_study(study))

            page_token = data.get("nextPageToken")
            page_num += 1

            if not page_token:
                logger.info("Pagination complete after {} page(s)", page_num)
                break

        if not records:
            logger.warning("search_trials returned no results")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info("search_trials complete | {} trials", len(df))
        return df

    # ------------------------------------------------------------------
    # Public API – detail / stats / results
    # ------------------------------------------------------------------

    def get_trial_details(self, nct_id: str) -> dict:
        """Fetch the full protocolSection for a single trial.

        Args:
            nct_id: ClinicalTrials.gov identifier, e.g. ``"NCT04280705"``.

        Returns:
            The ``protocolSection`` dict from the API response.
        """
        nct_id = nct_id.upper().strip()
        logger.info("get_trial_details | nct_id={}", nct_id)
        data = self._get(f"/studies/{nct_id}")
        return data.get("protocolSection", data)

    def get_enrollment_stats(self, nct_id: str) -> dict:
        """Extract enrollment statistics for a single trial.

        Args:
            nct_id: ClinicalTrials.gov identifier.

        Returns:
            Dict with keys: nct_id, enrollment_actual, enrollment_estimated,
            enrollment_velocity, pct_enrolled, sites_count.
        """
        nct_id = nct_id.upper().strip()
        details = self.get_trial_details(nct_id)

        # Enrollment info
        design_module = details.get("designModule", {})
        enrollment_info = design_module.get("enrollmentInfo", {})
        enrollment_actual: int | None = None
        enrollment_estimated: int | None = None
        enroll_count = enrollment_info.get("count")
        enroll_type = enrollment_info.get("type", "").upper()
        if enroll_count is not None:
            if enroll_type == "ACTUAL":
                enrollment_actual = int(enroll_count)
            else:
                enrollment_estimated = int(enroll_count)

        # Start date → months since start
        status_module = details.get("statusModule", {})
        start_date_str = status_module.get("startDateStruct", {}).get("date")
        months_since_start: float | None = None
        if start_date_str:
            try:
                start_dt = _parse_date(start_date_str)
                now_dt = datetime.now(tz=timezone.utc)
                delta_days = (now_dt - start_dt).days
                months_since_start = delta_days / 30.44
            except Exception as exc:
                logger.warning("Could not parse start date '{}': {}", start_date_str, exc)

        # Velocity = actual / months_since_start
        enrollment_velocity: float | None = None
        if enrollment_actual is not None and months_since_start and months_since_start > 0:
            enrollment_velocity = round(enrollment_actual / months_since_start, 4)

        # Percent enrolled
        pct_enrolled: float | None = None
        if enrollment_actual is not None and enrollment_estimated and enrollment_estimated > 0:
            pct_enrolled = round(enrollment_actual / enrollment_estimated * 100, 2)

        # Sites count (contactsLocationsModule)
        locations_module = details.get("contactsLocationsModule", {})
        sites_count = len(locations_module.get("locations", []))

        return {
            "nct_id": nct_id,
            "enrollment_actual": enrollment_actual,
            "enrollment_estimated": enrollment_estimated,
            "enrollment_velocity": enrollment_velocity,
            "pct_enrolled": pct_enrolled,
            "sites_count": sites_count,
        }

    def get_trial_results(self, nct_id: str) -> dict:
        """Fetch results section data for a trial.

        Args:
            nct_id: ClinicalTrials.gov identifier.

        Returns:
            Dict with keys: nct_id, primary_outcomes, adverse_events_total,
            has_results.
        """
        nct_id = nct_id.upper().strip()
        logger.info("get_trial_results | nct_id={}", nct_id)
        data = self._get(f"/studies/{nct_id}", params={"fields": "resultsSection"})
        results_section = data.get("resultsSection", {})
        has_results = bool(results_section)

        # Primary outcome measures
        outcome_module = results_section.get("outcomeMeasuresModule", {})
        primary_outcomes: list[dict] = []
        for om in outcome_module.get("outcomeMeasures", []):
            if om.get("type", "").upper() == "PRIMARY":
                primary_outcomes.append(
                    {
                        "title": om.get("title"),
                        "description": om.get("description"),
                        "time_frame": om.get("timeFrame"),
                    }
                )

        # Adverse events
        ae_module = results_section.get("adverseEventsModule", {})
        ae_total: int | None = None
        total_subjects = ae_module.get("eventGroups", [{}])
        if total_subjects:
            # Sum serious + other AE totals across all groups
            serious_totals = sum(
                g.get("seriousNumAffected", 0) or 0 for g in total_subjects
            )
            other_totals = sum(
                g.get("otherNumAffected", 0) or 0 for g in total_subjects
            )
            ae_total = serious_totals + other_totals

        return {
            "nct_id": nct_id,
            "primary_outcomes": primary_outcomes,
            "adverse_events_total": ae_total,
            "has_results": has_results,
        }

    # ------------------------------------------------------------------
    # Public API – pipeline dataset
    # ------------------------------------------------------------------

    def build_pipeline_dataset(self, company_names: list[str]) -> pd.DataFrame:
        """Build a consolidated pipeline dataset for a list of companies.

        Searches ClinicalTrials.gov using each company name as the sponsor
        query term and collects key fields.

        Args:
            company_names: List of pharmaceutical/biotech company names.

        Returns:
            DataFrame with columns: company, nct_id, title, phase, status,
            condition, start_date, completion_date, enrollment_actual,
            enrollment_estimated, enrollment_velocity, has_results.
        """
        all_rows: list[dict] = []

        for company in company_names:
            logger.info("build_pipeline_dataset | querying company='{}'", company)
            try:
                df_company = self.search_trials(query=company, page_size=100, max_pages=10)
            except Exception as exc:
                logger.error("Failed to fetch trials for '{}': {}", company, exc)
                continue

            if df_company.empty:
                logger.warning("No trials found for company='{}'", company)
                continue

            for _, row in df_company.iterrows():
                nct_id = row.get("nct_id", "")

                # Attempt to get results flag
                has_results = False
                try:
                    results = self.get_trial_results(nct_id)
                    has_results = results.get("has_results", False)
                except Exception:
                    pass

                # Enrollment velocity from individual stats
                enrollment_velocity: float | None = None
                try:
                    stats = self.get_enrollment_stats(nct_id)
                    enrollment_velocity = stats.get("enrollment_velocity")
                except Exception:
                    pass

                all_rows.append(
                    {
                        "company": company,
                        "nct_id": nct_id,
                        "title": row.get("brief_title"),
                        "phase": row.get("phase"),
                        "status": row.get("overall_status"),
                        "condition": row.get("conditions"),
                        "start_date": row.get("start_date"),
                        "completion_date": row.get("primary_completion_date"),
                        "enrollment_actual": row.get("enrollment_actual"),
                        "enrollment_estimated": row.get("enrollment_estimated"),
                        "enrollment_velocity": enrollment_velocity,
                        "has_results": has_results,
                    }
                )

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        logger.info(
            "build_pipeline_dataset complete | {} rows for {} companies",
            len(df),
            len(company_names),
        )
        return df

    # ------------------------------------------------------------------
    # Public API – enrollment velocity
    # ------------------------------------------------------------------

    def compute_enrollment_velocity(self, trial_df: pd.DataFrame) -> pd.DataFrame:
        """Annotate a trials DataFrame with enrollment velocity metrics.

        Adds the following columns:
            - ``months_active``: months from start_date to today
            - ``patients_per_month``: enrollment_actual / months_active
            - ``expected_duration_months``: months from start_date to
              completion_date (or median fallback of 24 months)
            - ``velocity_vs_target``: patients_per_month /
              (enrollment_estimated / expected_duration_months)
            - ``velocity_signal``: 'POSITIVE' (>1.05), 'NEGATIVE' (<0.85),
              or 'NEUTRAL'

        Args:
            trial_df: DataFrame containing at minimum the columns
                ``start_date``, ``enrollment_actual``,
                ``enrollment_estimated``, and optionally
                ``completion_date``.

        Returns:
            Input DataFrame with the new columns appended.
        """
        df = trial_df.copy()
        today = datetime.now(tz=timezone.utc)

        def calc_months_active(start_date_val: Any) -> float | None:
            if pd.isna(start_date_val) or start_date_val is None:
                return None
            try:
                dt = _parse_date(str(start_date_val))
                delta = (today - dt).days
                return max(delta / 30.44, 0.0)
            except Exception:
                return None

        def calc_expected_duration(
            start_date_val: Any, completion_date_val: Any
        ) -> float:
            """Return expected duration in months; fallback = 24."""
            try:
                if pd.isna(start_date_val) or pd.isna(completion_date_val):
                    return 24.0
                dt_start = _parse_date(str(start_date_val))
                dt_end = _parse_date(str(completion_date_val))
                months = (dt_end - dt_start).days / 30.44
                return max(months, 1.0)
            except Exception:
                return 24.0

        df["months_active"] = df["start_date"].apply(calc_months_active)

        # Patients per month
        def _ppm(row: pd.Series) -> float | None:
            actual = row.get("enrollment_actual")
            months = row.get("months_active")
            if actual is None or months is None or months <= 0:
                return None
            try:
                return float(actual) / float(months)
            except (TypeError, ZeroDivisionError):
                return None

        df["patients_per_month"] = df.apply(_ppm, axis=1)

        # Expected duration
        completion_col = "completion_date" if "completion_date" in df.columns else None
        if completion_col:
            df["expected_duration_months"] = df.apply(
                lambda r: calc_expected_duration(r["start_date"], r[completion_col]),
                axis=1,
            )
        else:
            df["expected_duration_months"] = 24.0

        # Velocity vs target
        def _vvt(row: pd.Series) -> float | None:
            ppm = row.get("patients_per_month")
            est = row.get("enrollment_estimated")
            dur = row.get("expected_duration_months")
            if ppm is None or est is None or dur is None or dur <= 0:
                return None
            try:
                target_ppm = float(est) / float(dur)
                if target_ppm <= 0:
                    return None
                return round(ppm / target_ppm, 4)
            except (TypeError, ZeroDivisionError):
                return None

        df["velocity_vs_target"] = df.apply(_vvt, axis=1)

        def _signal(vvt: Any) -> str | None:
            if vvt is None or (isinstance(vvt, float) and math.isnan(vvt)):
                return None
            if vvt > 1.05:
                return "POSITIVE"
            if vvt < 0.85:
                return "NEGATIVE"
            return "NEUTRAL"

        df["velocity_signal"] = df["velocity_vs_target"].apply(_signal)
        logger.info("compute_enrollment_velocity complete | {} rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> str:
        """Return an absolute path to a cache file for the given key.

        The key is hashed so filenames remain filesystem-safe.

        Args:
            key: Arbitrary string key (e.g. an NCT ID or search fingerprint).

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
    # Private parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_study(study: dict) -> dict:
        """Flatten a raw study dict into a single-level record."""
        proto = study.get("protocolSection", {})

        # ---- identification ----
        id_module = proto.get("identificationModule", {})
        nct_id: str = id_module.get("nctId", "")
        brief_title: str = id_module.get("briefTitle", "")
        official_title: str = id_module.get("officialTitle", "")

        # ---- status ----
        status_module = proto.get("statusModule", {})
        overall_status: str = status_module.get("overallStatus", "")
        start_date: str = status_module.get("startDateStruct", {}).get("date", "")
        primary_completion_date: str = status_module.get(
            "primaryCompletionDateStruct", {}
        ).get("date", "")

        # ---- design ----
        design_module = proto.get("designModule", {})
        phase_list: list[str] = design_module.get("phases", [])
        phase: str = "|".join(phase_list) if phase_list else ""
        study_type: str = design_module.get("studyType", "")
        enrollment_info = design_module.get("enrollmentInfo", {})
        enroll_count = enrollment_info.get("count")
        enroll_type = (enrollment_info.get("type") or "").upper()
        enrollment_actual: int | None = (
            int(enroll_count) if enroll_count is not None and enroll_type == "ACTUAL" else None
        )
        enrollment_estimated: int | None = (
            int(enroll_count)
            if enroll_count is not None and enroll_type != "ACTUAL"
            else None
        )

        # ---- conditions ----
        conditions_module = proto.get("conditionsModule", {})
        conditions: str = ", ".join(conditions_module.get("conditions", []))

        # ---- interventions ----
        arms_module = proto.get("armsInterventionsModule", {})
        interventions_raw = arms_module.get("interventions", [])
        interventions: str = ", ".join(
            i.get("name", "") for i in interventions_raw if i.get("name")
        )

        # ---- sponsors ----
        sponsor_module = proto.get("sponsorsCollaboratorsModule", {})
        lead_sponsor: str = sponsor_module.get("leadSponsor", {}).get("name", "")
        collaborators: list[str] = [
            c.get("name", "") for c in sponsor_module.get("collaborators", [])
        ]
        sponsors: str = lead_sponsor
        if collaborators:
            sponsors += "; " + ", ".join(collaborators)

        # ---- eligibility ----
        eligibility_module = proto.get("eligibilityModule", {})
        min_age: str = eligibility_module.get("minimumAge", "")
        max_age: str = eligibility_module.get("maximumAge", "")
        sex: str = eligibility_module.get("sex", "")

        return {
            "nct_id": nct_id,
            "brief_title": brief_title,
            "official_title": official_title,
            "overall_status": overall_status,
            "phase": phase,
            "study_type": study_type,
            "conditions": conditions,
            "interventions": interventions,
            "sponsors": sponsors,
            "enrollment_actual": enrollment_actual,
            "enrollment_estimated": enrollment_estimated,
            "start_date": start_date or None,
            "primary_completion_date": primary_completion_date or None,
            "min_age": min_age,
            "max_age": max_age,
            "sex": sex,
        }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> datetime:
    """Parse a ClinicalTrials date string into a timezone-aware datetime.

    Accepts formats: ``YYYY-MM-DD``, ``YYYY-MM``, ``YYYY``.
    """
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: '{date_str}'")
