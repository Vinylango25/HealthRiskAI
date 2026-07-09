"""
pipeline_monitor.py
===================
Automated pharmaceutical pipeline monitoring system.

Tracks clinical trial status changes sourced from ClinicalTrials.gov and
integrates FDA FAERS (Adverse Event Reporting System) data for safety signal
monitoring. Generates structured signals for investment and risk decisions.

Data Sources (production):
    - ClinicalTrials.gov REST API v2: https://clinicaltrials.gov/api/v2/
    - FDA FAERS API: https://api.fda.gov/drug/event.json
    - SEC EDGAR for company filings
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False
    logger.warning("pandas not installed; DataFrame outputs will be dict lists.")


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class TrialStatus(str, Enum):
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"
    SUSPENDED = "SUSPENDED"
    RESULTS_POSTED = "RESULTS_POSTED"
    UNKNOWN = "UNKNOWN"


class SignalType(str, Enum):
    POSITIVE_READOUT = "POSITIVE_READOUT"
    NEGATIVE_READOUT = "NEGATIVE_READOUT"
    ENROLLMENT_DELAY = "ENROLLMENT_DELAY"
    TRIAL_FAILURE = "TRIAL_FAILURE"
    SAFETY_CONCERN = "SAFETY_CONCERN"
    COMPETITOR_THREAT = "COMPETITOR_THREAT"
    COMPETITOR_FAILURE = "COMPETITOR_FAILURE"
    RESULTS_POSTED = "RESULTS_POSTED"
    STATUS_CHANGE = "STATUS_CHANGE"


class SignalSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


ENROLLMENT_DELAY_THRESHOLD_DAYS = 90  # flag if > 3 months past expected start
FAERS_SIGNAL_ROR_THRESHOLD = 2.0      # Reporting Odds Ratio threshold


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClinicalTrial:
    """Represents a single clinical trial being monitored."""
    nct_id: str
    title: str
    sponsor: str
    indication: str
    phase: str
    status: TrialStatus
    primary_completion_date: Optional[date] = None
    enrollment_target: int = 0
    enrollment_actual: int = 0
    drug_name: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineEvent:
    """A discrete event detected in pipeline monitoring."""
    event_id: str
    nct_id: str
    drug_name: str
    sponsor: str
    event_type: SignalType
    severity: SignalSeverity
    description: str
    detected_at: datetime = field(default_factory=datetime.utcnow)
    previous_status: Optional[TrialStatus] = None
    current_status: Optional[TrialStatus] = None
    estimated_value_impact_mm: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertSignal:
    """Actionable signal for portfolio decision-making."""
    signal_id: str
    signal_type: SignalType
    severity: SignalSeverity
    affected_drugs: List[str]
    affected_sponsors: List[str]
    headline: str
    detail: str
    confidence: float  # 0–1
    action_required: bool = False
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CompetitiveIntelligence:
    """Competitive landscape snapshot for a given indication."""
    indication: str
    as_of: date
    active_trials: int
    late_stage_count: int  # Phase 3 + NDA
    nearest_readout: Optional[str] = None  # NCT ID
    nearest_readout_date: Optional[date] = None
    market_threat_score: float = 0.0  # 0–10
    key_competitors: List[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Pipeline Monitor
# ---------------------------------------------------------------------------

class PipelineMonitor:
    """
    Monitors pharmaceutical clinical trial pipelines and generates signals.

    In production this connects to ClinicalTrials.gov v2 API and FDA FAERS.
    In this implementation, HTTP calls are mocked with synthetic data to allow
    offline testing and CI execution without API keys.

    Parameters
    ----------
    company_nct_ids : dict
        Mapping of company name → list of NCT IDs to monitor.
    use_live_api : bool
        If True, attempt real HTTP calls (requires network). Default: False.
    """

    def __init__(
        self,
        company_nct_ids: Optional[Dict[str, List[str]]] = None,
        use_live_api: bool = False,
    ):
        self.company_nct_ids = company_nct_ids or {}
        self.use_live_api = use_live_api

        self._trials: Dict[str, ClinicalTrial] = {}           # nct_id → trial
        self._events: List[PipelineEvent] = []
        self._signals: List[AlertSignal] = []
        self._competitor_map: Dict[str, List[str]] = {}       # indication → NCT IDs
        self._prev_statuses: Dict[str, TrialStatus] = {}

        logger.info("PipelineMonitor initialised (live_api=%s)", use_live_api)

    # ------------------------------------------------------------------
    # Synthetic data generation (offline / test mode)
    # ------------------------------------------------------------------

    def _synthetic_trial(self, nct_id: str, company: str) -> ClinicalTrial:
        """Generate a deterministic synthetic trial from an NCT ID."""
        rng = random.Random(int(hashlib.md5(nct_id.encode()).hexdigest(), 16) % (2**32))
        phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 2/3"]
        indications = ["Oncology", "Cardiovascular", "CNS", "Infectious Disease", "Rare Disease"]
        statuses = list(TrialStatus)

        status = rng.choice([
            TrialStatus.RECRUITING,
            TrialStatus.ACTIVE_NOT_RECRUITING,
            TrialStatus.COMPLETED,
            TrialStatus.RECRUITING,
        ])

        return ClinicalTrial(
            nct_id=nct_id,
            title=f"A Study of Drug-{nct_id[-4:]} in {rng.choice(indications)}",
            sponsor=company,
            indication=rng.choice(indications),
            phase=rng.choice(phases),
            status=status,
            primary_completion_date=date.today() + timedelta(days=rng.randint(-365, 730)),
            enrollment_target=rng.randint(50, 800),
            enrollment_actual=rng.randint(0, 600),
            drug_name=f"DRUG-{nct_id[-5:].upper()}",
            last_updated=datetime.utcnow() - timedelta(days=rng.randint(0, 180)),
        )

    def _fetch_trial(self, nct_id: str, company: str) -> ClinicalTrial:
        """Fetch trial data — live API or synthetic fallback."""
        if self.use_live_api:
            try:
                import urllib.request, urllib.error
                url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}?format=json"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = json.loads(resp.read())
                proto = data.get("protocolSection", {})
                id_mod = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                design_mod = proto.get("designModule", {})
                raw_status = status_mod.get("overallStatus", "UNKNOWN").upper().replace(" ", "_")
                return ClinicalTrial(
                    nct_id=nct_id,
                    title=id_mod.get("briefTitle", "Unknown"),
                    sponsor=company,
                    indication="",
                    phase=design_mod.get("phases", ["Unknown"])[0] if design_mod.get("phases") else "Unknown",
                    status=TrialStatus(raw_status) if raw_status in TrialStatus._value2member_map_ else TrialStatus.UNKNOWN,
                    drug_name=id_mod.get("acronym", nct_id),
                )
            except Exception as exc:
                logger.warning("Live API fetch failed for %s: %s. Using synthetic data.", nct_id, exc)
        return self._synthetic_trial(nct_id, company)

    def _fetch_faers_signals(self, drug_name: str) -> List[Dict[str, Any]]:
        """Query FDA FAERS for safety signals. Returns synthetic data in offline mode."""
        if self.use_live_api:
            try:
                import urllib.request
                url = (
                    f"https://api.fda.gov/drug/event.json"
                    f"?search=patient.drug.medicinalproduct:{drug_name}&limit=10"
                )
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = json.loads(resp.read())
                return data.get("results", [])
            except Exception as exc:
                logger.warning("FAERS fetch failed for %s: %s", drug_name, exc)
        # Synthetic: random safety score
        rng = random.Random(hash(drug_name) % (2**31))
        return [{"ror": rng.uniform(0.5, 3.5), "term": "adverse_event_synthetic"}]

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def update(self) -> List[PipelineEvent]:
        """
        Refresh all monitored trials and detect status changes.

        Returns
        -------
        list of PipelineEvent
            New events detected since the last update call.
        """
        new_events: List[PipelineEvent] = []

        for company, nct_ids in self.company_nct_ids.items():
            for nct_id in nct_ids:
                trial = self._fetch_trial(nct_id, company)
                self._trials[nct_id] = trial

                prev_status = self._prev_statuses.get(nct_id)

                if prev_status and prev_status != trial.status:
                    events = self._classify_status_change(trial, prev_status)
                    new_events.extend(events)
                    self._events.extend(events)

                # Check enrollment pace
                if trial.status == TrialStatus.RECRUITING:
                    enroll_event = self._check_enrollment_delay(trial)
                    if enroll_event:
                        new_events.append(enroll_event)
                        self._events.append(enroll_event)

                # Check FAERS safety signals
                if trial.drug_name:
                    safety_events = self._check_safety_signals(trial)
                    new_events.extend(safety_events)
                    self._events.extend(safety_events)

                self._prev_statuses[nct_id] = trial.status

        new_signals = self._generate_alert_signals(new_events)
        self._signals.extend(new_signals)

        logger.info("Update complete: %d new events, %d new signals", len(new_events), len(new_signals))
        return new_events

    def _classify_status_change(
        self, trial: ClinicalTrial, prev_status: TrialStatus
    ) -> List[PipelineEvent]:
        """Map a status transition to one or more PipelineEvents."""
        events: List[PipelineEvent] = []
        eid = f"EVT-{trial.nct_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        transition = (prev_status, trial.status)

        if trial.status == TrialStatus.RESULTS_POSTED:
            # Simulate positive/negative based on hash
            is_positive = int(hashlib.md5(trial.nct_id.encode()).hexdigest()[0], 16) > 7
            stype = SignalType.POSITIVE_READOUT if is_positive else SignalType.NEGATIVE_READOUT
            severity = SignalSeverity.HIGH if "Phase 3" in trial.phase else SignalSeverity.MEDIUM
            events.append(PipelineEvent(
                event_id=eid, nct_id=trial.nct_id, drug_name=trial.drug_name,
                sponsor=trial.sponsor, event_type=stype, severity=severity,
                description=f"{'Positive' if is_positive else 'Negative'} results posted for {trial.title}",
                previous_status=prev_status, current_status=trial.status,
                estimated_value_impact_mm=500.0 if is_positive else -300.0,
            ))

        elif trial.status == TrialStatus.TERMINATED:
            events.append(PipelineEvent(
                event_id=eid, nct_id=trial.nct_id, drug_name=trial.drug_name,
                sponsor=trial.sponsor, event_type=SignalType.TRIAL_FAILURE,
                severity=SignalSeverity.CRITICAL,
                description=f"Trial terminated: {trial.title}",
                previous_status=prev_status, current_status=trial.status,
                estimated_value_impact_mm=-200.0,
            ))

        elif transition == (TrialStatus.ACTIVE_NOT_RECRUITING, TrialStatus.COMPLETED):
            events.append(PipelineEvent(
                event_id=eid, nct_id=trial.nct_id, drug_name=trial.drug_name,
                sponsor=trial.sponsor, event_type=SignalType.STATUS_CHANGE,
                severity=SignalSeverity.MEDIUM,
                description=f"Trial completed, awaiting results: {trial.title}",
                previous_status=prev_status, current_status=trial.status,
            ))
        else:
            events.append(PipelineEvent(
                event_id=eid, nct_id=trial.nct_id, drug_name=trial.drug_name,
                sponsor=trial.sponsor, event_type=SignalType.STATUS_CHANGE,
                severity=SignalSeverity.LOW,
                description=f"Status change {prev_status.value} → {trial.status.value}: {trial.title}",
                previous_status=prev_status, current_status=trial.status,
            ))

        return events

    def _check_enrollment_delay(self, trial: ClinicalTrial) -> Optional[PipelineEvent]:
        """Flag enrollment delays based on target vs actual ratio and time elapsed."""
        if trial.enrollment_target == 0:
            return None
        ratio = trial.enrollment_actual / trial.enrollment_target
        # Flag if <30% enrolled and primary completion is within 12 months
        if ratio < 0.30 and trial.primary_completion_date:
            days_to_completion = (trial.primary_completion_date - date.today()).days
            if 0 < days_to_completion < 365:
                return PipelineEvent(
                    event_id=f"ENRL-{trial.nct_id}-{date.today().isoformat()}",
                    nct_id=trial.nct_id, drug_name=trial.drug_name,
                    sponsor=trial.sponsor, event_type=SignalType.ENROLLMENT_DELAY,
                    severity=SignalSeverity.MEDIUM,
                    description=(
                        f"Enrollment at {ratio*100:.0f}% ({trial.enrollment_actual}/"
                        f"{trial.enrollment_target}) with {days_to_completion}d to completion."
                    ),
                    current_status=trial.status,
                )
        return None

    def _check_safety_signals(self, trial: ClinicalTrial) -> List[PipelineEvent]:
        """Check FAERS for elevated adverse event reporting odds ratio."""
        faers = self._fetch_faers_signals(trial.drug_name)
        events: List[PipelineEvent] = []
        for record in faers:
            ror = record.get("ror", 1.0)
            if ror >= FAERS_SIGNAL_ROR_THRESHOLD:
                events.append(PipelineEvent(
                    event_id=f"SAFE-{trial.nct_id}-{record.get('term','')}",
                    nct_id=trial.nct_id, drug_name=trial.drug_name,
                    sponsor=trial.sponsor, event_type=SignalType.SAFETY_CONCERN,
                    severity=SignalSeverity.HIGH if ror > 3.0 else SignalSeverity.MEDIUM,
                    description=f"FAERS ROR={ror:.2f} for term '{record.get('term')}' on {trial.drug_name}",
                    current_status=trial.status,
                    metadata={"ror": ror, "term": record.get("term")},
                ))
        return events

    def _generate_alert_signals(self, events: List[PipelineEvent]) -> List[AlertSignal]:
        """Aggregate events into actionable AlertSignals."""
        signals: List[AlertSignal] = []
        for evt in events:
            if evt.severity in (SignalSeverity.HIGH, SignalSeverity.CRITICAL):
                sig = AlertSignal(
                    signal_id=f"SIG-{evt.event_id}",
                    signal_type=evt.event_type,
                    severity=evt.severity,
                    affected_drugs=[evt.drug_name],
                    affected_sponsors=[evt.sponsor],
                    headline=f"[{evt.severity.value}] {evt.event_type.value}: {evt.drug_name}",
                    detail=evt.description,
                    confidence=0.80 if evt.event_type != SignalType.SAFETY_CONCERN else 0.65,
                    action_required=evt.severity == SignalSeverity.CRITICAL,
                )
                signals.append(sig)
        return signals

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_signals(
        self,
        severity: Optional[SignalSeverity] = None,
        signal_type: Optional[SignalType] = None,
    ) -> List[AlertSignal]:
        """
        Retrieve filtered alert signals.

        Parameters
        ----------
        severity : SignalSeverity, optional
            Filter to specific severity level.
        signal_type : SignalType, optional
            Filter to specific signal type.

        Returns
        -------
        list of AlertSignal
        """
        result = self._signals
        if severity:
            result = [s for s in result if s.severity == severity]
        if signal_type:
            result = [s for s in result if s.signal_type == signal_type]
        return result

    def track_competitor(self, indication: str, competitor_nct_ids: List[str]) -> CompetitiveIntelligence:
        """
        Build competitive intelligence for a given indication.

        Parameters
        ----------
        indication : str
            Disease area to analyse.
        competitor_nct_ids : list of str
            NCT IDs of competitor trials to track.

        Returns
        -------
        CompetitiveIntelligence
        """
        self._competitor_map[indication] = competitor_nct_ids

        comp_trials: List[ClinicalTrial] = []
        for nct_id in competitor_nct_ids:
            trial = self._fetch_trial(nct_id, "competitor")
            comp_trials.append(trial)

        late_stage = [t for t in comp_trials if "Phase 3" in t.phase or "NDA" in t.phase]
        active = [t for t in comp_trials if t.status in (TrialStatus.RECRUITING, TrialStatus.ACTIVE_NOT_RECRUITING)]

        # Find nearest readout
        nearest: Optional[ClinicalTrial] = None
        for t in active:
            if t.primary_completion_date and (
                nearest is None or t.primary_completion_date < nearest.primary_completion_date
            ):
                nearest = t

        threat_score = min(10.0, len(late_stage) * 2.5 + len(active) * 0.5)

        ci = CompetitiveIntelligence(
            indication=indication,
            as_of=date.today(),
            active_trials=len(active),
            late_stage_count=len(late_stage),
            nearest_readout=nearest.nct_id if nearest else None,
            nearest_readout_date=nearest.primary_completion_date if nearest else None,
            market_threat_score=round(threat_score, 1),
            key_competitors=list({t.sponsor for t in comp_trials}),
        )

        logger.info(
            "Competitive intel for '%s': %d active, %d late-stage, threat=%.1f",
            indication, ci.active_trials, ci.late_stage_count, ci.market_threat_score,
        )
        return ci

    def get_pipeline_events_dataframe(self) -> Any:
        """Return all pipeline events as a pandas DataFrame (or list of dicts)."""
        rows = [
            {
                "event_id": e.event_id,
                "nct_id": e.nct_id,
                "drug_name": e.drug_name,
                "sponsor": e.sponsor,
                "event_type": e.event_type.value,
                "severity": e.severity.value,
                "description": e.description,
                "detected_at": e.detected_at.isoformat(),
                "value_impact_mm": e.estimated_value_impact_mm,
            }
            for e in self._events
        ]
        if _PANDAS_AVAILABLE:
            import pandas as pd
            return pd.DataFrame(rows)
        return rows


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline Monitor — Synthetic Smoke Test")
    print("=" * 60)

    monitor = PipelineMonitor(
        company_nct_ids={
            "PharmaAlpha": ["NCT04001001", "NCT04001002", "NCT04001003"],
            "BetaBiotech": ["NCT05002001", "NCT05002002"],
        },
        use_live_api=False,
    )

    # Seed some prior statuses to trigger change detection
    monitor._prev_statuses["NCT04001001"] = TrialStatus.ACTIVE_NOT_RECRUITING
    monitor._prev_statuses["NCT05002001"] = TrialStatus.RECRUITING

    print("\n--- Running update() ---")
    events = monitor.update()
    print(f"  Detected {len(events)} pipeline event(s)")
    for evt in events[:5]:
        print(f"  [{evt.severity.value}] {evt.event_type.value}: {evt.description[:80]}")

    print("\n--- Alert signals ---")
    signals = monitor.get_signals()
    print(f"  Total signals: {len(signals)}")
    for sig in signals[:3]:
        print(f"  {sig.headline}")

    print("\n--- Competitive intelligence: Oncology ---")
    ci = monitor.track_competitor(
        "Oncology",
        ["NCT06100001", "NCT06100002", "NCT06100003", "NCT06100004"],
    )
    print(f"  Active trials  : {ci.active_trials}")
    print(f"  Late-stage     : {ci.late_stage_count}")
    print(f"  Threat score   : {ci.market_threat_score}/10")
    print(f"  Competitors    : {ci.key_competitors}")

    print("\n--- Pipeline events DataFrame ---")
    df = monitor.get_pipeline_events_dataframe()
    if _PANDAS_AVAILABLE:
        print(df[["drug_name", "event_type", "severity", "value_impact_mm"]].to_string(index=False))
    else:
        for row in df[:5]:
            print(f"  {row}")

    # Assertions
    assert isinstance(events, list), "update() must return list"
    assert isinstance(signals, list), "get_signals() must return list"
    assert ci.market_threat_score >= 0, "Threat score must be non-negative"
    print("\n✓ All smoke-test assertions passed.")
