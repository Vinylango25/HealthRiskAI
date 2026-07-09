"""
HealthRisk Lab – Scenario Generator
=====================================
Generates stochastic scenarios for the simulation.  Four scenario families
are supported, each with its own mechanics:

* **pandemic**          – epidemic dynamics modelled via SIR/SEIR ODEs
* **drug_safety_crisis** – FAERS adverse-event signal spike
* **regulatory_change** – sudden policy / reimbursement shift
* **hospital_merger**   – consolidation event affecting credit / market risk

Every scenario exposes a uniform interface so the simulation engine can
apply financial and clinical impacts without knowing the underlying model.
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ScenarioType(Enum):
    PANDEMIC          = "pandemic"
    DRUG_SAFETY_CRISIS = "drug_safety_crisis"
    REGULATORY_CHANGE  = "regulatory_change"
    HOSPITAL_MERGER    = "hospital_merger"


class Severity(Enum):
    MILD         = "mild"
    MODERATE     = "moderate"
    SEVERE       = "severe"
    CATASTROPHIC = "catastrophic"

    @property
    def multiplier(self) -> float:
        return {"mild": 0.25, "moderate": 0.55, "severe": 0.80, "catastrophic": 1.0}[self.value]


# ---------------------------------------------------------------------------
# SIR / SEIR epidemic model
# ---------------------------------------------------------------------------

@dataclass
class EpidemicState:
    """State vector for the SIR / SEIR compartmental model."""
    S: float   # Susceptible
    E: float   # Exposed (SEIR only; 0 for SIR)
    I: float   # Infectious
    R: float   # Recovered / Removed
    N: float   # Total population

    @property
    def attack_rate(self) -> float:
        """Final epidemic attack rate (fraction of population infected)."""
        return self.R / self.N if self.N > 0 else 0.0

    @property
    def peak_prevalence(self) -> float:
        return self.I / self.N if self.N > 0 else 0.0


def run_sir_model(
    population: float = 1_000_000,
    beta: float = 0.3,
    gamma: float = 0.1,
    initial_infected: int = 100,
    days: int = 180,
    dt: float = 1.0,
) -> Tuple[EpidemicState, List[EpidemicState]]:
    """
    Euler-integration SIR model.

    Parameters
    ----------
    population       : Total susceptible population.
    beta             : Transmission rate (contacts × probability per day).
    gamma            : Recovery rate (1 / infectious_period).
    initial_infected : Seed cases at t=0.
    days             : Simulation horizon in days.
    dt               : Time step in days.

    Returns
    -------
    (final_state, trajectory) – final EpidemicState and daily trajectory list.
    """
    S = float(population - initial_infected)
    E = 0.0
    I = float(initial_infected)
    R = 0.0
    N = float(population)
    R0 = beta / gamma

    trajectory: List[EpidemicState] = []
    steps = int(days / dt)

    for _ in range(steps):
        new_infections = beta * S * I / N
        new_recoveries = gamma * I
        S -= new_infections * dt
        I += (new_infections - new_recoveries) * dt
        R += new_recoveries * dt
        S = max(S, 0.0)
        I = max(I, 0.0)
        trajectory.append(EpidemicState(S=S, E=E, I=I, R=R, N=N))

    final = EpidemicState(S=S, E=E, I=I, R=R, N=N)
    logger.debug("SIR model complete | R0=%.2f | attack_rate=%.1f%%", R0, final.attack_rate * 100)
    return final, trajectory


def run_seir_model(
    population: float = 1_000_000,
    beta: float = 0.4,
    sigma: float = 0.2,      # 1 / incubation_period
    gamma: float = 0.1,
    initial_exposed: int = 500,
    days: int = 270,
    dt: float = 1.0,
) -> Tuple[EpidemicState, List[EpidemicState]]:
    """
    Euler-integration SEIR model with an Exposed compartment.

    Returns
    -------
    (final_state, trajectory)
    """
    S = float(population - initial_exposed)
    E = float(initial_exposed)
    I = 0.0
    R = 0.0
    N = float(population)

    trajectory: List[EpidemicState] = []
    steps = int(days / dt)

    for _ in range(steps):
        new_exposures  = beta * S * I / N
        new_infectious = sigma * E
        new_recoveries = gamma * I
        S -= new_exposures  * dt
        E += (new_exposures - new_infectious) * dt
        I += (new_infectious - new_recoveries) * dt
        R += new_recoveries * dt
        S = max(S, 0.0)
        E = max(E, 0.0)
        I = max(I, 0.0)
        trajectory.append(EpidemicState(S=S, E=E, I=I, R=R, N=N))

    final = EpidemicState(S=S, E=E, I=I, R=R, N=N)
    return final, trajectory


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """
    Fully-specified simulation scenario.

    Attributes
    ----------
    scenario_id          : Unique identifier.
    scenario_type        : ScenarioType enum value.
    severity             : Severity enum value.
    name                 : Human-readable display name.
    description          : Narrative description for UI / training.
    impact_duration      : Number of quarters the scenario persists.
    financial_impact_range : (min_pct, max_pct) portfolio impact.
    clinical_impact_range  : (min_per_100k, max_per_100k) adverse events.
    metadata             : Extra model-specific data (e.g. epidemic trajectory).
    """
    scenario_id:             str
    scenario_type:           ScenarioType
    severity:                Severity
    name:                    str
    description:             str
    impact_duration:         int                     # quarters
    financial_impact_range:  Tuple[float, float]     # fraction, e.g. (-0.15, -0.05)
    clinical_impact_range:   Tuple[float, float]     # adverse events per 100k
    metadata:                Dict                    = field(default_factory=dict)

    def sample_financial_impact(self, rng: random.Random) -> float:
        """Draw a single realisation of the financial impact."""
        lo, hi = self.financial_impact_range
        return rng.uniform(lo, hi)

    def sample_clinical_impact(self, rng: random.Random) -> float:
        lo, hi = self.clinical_impact_range
        return rng.uniform(lo, hi)

    def __str__(self) -> str:
        lo, hi = self.financial_impact_range
        return (
            f"[{self.severity.value.upper()}] {self.name}  "
            f"duration={self.impact_duration}q  "
            f"fin_impact=[{lo*100:.1f}%, {hi*100:.1f}%]"
        )


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------

# Structure: { ScenarioType: { Severity: (name, description, duration, fin_lo, fin_hi, clin_lo, clin_hi) } }
_TEMPLATES: Dict[ScenarioType, Dict[Severity, tuple]] = {
    ScenarioType.PANDEMIC: {
        Severity.MILD:         ("Seasonal Flu Surge",       "Elevated ILI activity strains outpatient capacity.",       1, -0.02, -0.005, 500, 2000),
        Severity.MODERATE:     ("Regional Outbreak",         "Novel pathogen spreads across 3 states.",                  2, -0.06, -0.02,  2000, 8000),
        Severity.SEVERE:       ("National Epidemic",         "High-transmissibility variant triggers emergency measures.", 3, -0.14, -0.06, 8000, 25000),
        Severity.CATASTROPHIC: ("Global Pandemic",           "Pandemic declaration; widespread lockdowns and excess mortality.", 5, -0.30, -0.12, 25000, 80000),
    },
    ScenarioType.DRUG_SAFETY_CRISIS: {
        Severity.MILD:         ("Post-Market Signal",        "FAERS disproportionality flag on a second-line agent.",    1, -0.015, 0.0,  10,  50),
        Severity.MODERATE:     ("Class-Wide Safety Review",  "FDA requests REMS update for blockbuster drug class.",     2, -0.05, -0.01, 50,  200),
        Severity.SEVERE:       ("Voluntary Recall",          "Major manufacturer initiates market withdrawal.",          3, -0.10, -0.04, 200, 800),
        Severity.CATASTROPHIC: ("Market-Wide Drug Crisis",   "Multi-drug contamination triggers congressional hearings.", 4, -0.22, -0.10, 800, 3000),
    },
    ScenarioType.REGULATORY_CHANGE: {
        Severity.MILD:         ("Coding Update",             "CMS ICD-10 revisions alter reimbursement slightly.",       1, -0.01,  0.005, 0, 5),
        Severity.MODERATE:     ("Reimbursement Cuts",        "10% across-the-board Medicare rate reduction.",            2, -0.06, -0.02,  0, 5),
        Severity.SEVERE:       ("Drug Pricing Legislation",  "Price negotiation law passed; pharma equities hit hard.",  3, -0.12, -0.05,  0, 5),
        Severity.CATASTROPHIC: ("ACA Repeal",                "Landmark coverage legislation struck down; mass dis-enrollment.", 4, -0.25, -0.12, 5, 30),
    },
    ScenarioType.HOSPITAL_MERGER: {
        Severity.MILD:         ("Small System Merger",       "Two community hospitals combine back-office functions.",   1,  0.00,  0.02,  0, 2),
        Severity.MODERATE:     ("Regional Network Deal",     "5-hospital system acquired; credit spread widens.",        2, -0.03,  0.02,  0, 5),
        Severity.SEVERE:       ("Mega-Merger Blocked",       "DOJ challenges proposed merger; deal collapses.",          2, -0.08, -0.02,  0, 5),
        Severity.CATASTROPHIC: ("System Bankruptcy",         "Largest regional IDN files Chapter 11.",                   4, -0.18, -0.08,  0, 20),
    },
}


# ---------------------------------------------------------------------------
# Scenario Generator
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """
    Factory that produces ``Scenario`` objects from the template library,
    optionally augmented with epidemic model outputs.

    Parameters
    ----------
    seed : Optional RNG seed for reproducibility.
    """

    _SEVERITY_WEIGHTS = [0.40, 0.30, 0.20, 0.10]  # mild, moderate, severe, catastrophic

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        scenario_type: Optional[ScenarioType] = None,
        severity: Optional[Severity] = None,
    ) -> Scenario:
        """
        Generate a Scenario.

        Parameters
        ----------
        scenario_type : If None, chosen at random.
        severity      : If None, sampled by weighted distribution.

        Returns
        -------
        Scenario instance.
        """
        if scenario_type is None:
            scenario_type = self._rng.choice(list(ScenarioType))
        if severity is None:
            severity = self._rng.choices(list(Severity), weights=self._SEVERITY_WEIGHTS, k=1)[0]

        name, description, duration, fin_lo, fin_hi, clin_lo, clin_hi = \
            _TEMPLATES[scenario_type][severity]

        # Scale impact by severity multiplier (adds slight randomness)
        m = severity.multiplier
        fin_lo  = fin_lo  * (0.85 + 0.30 * self._rng.random())
        fin_hi  = fin_hi  * (0.85 + 0.30 * self._rng.random())
        clin_lo = clin_lo * m
        clin_hi = clin_hi * m

        metadata: Dict = {}

        # Augment pandemic scenarios with epidemic model output
        if scenario_type == ScenarioType.PANDEMIC:
            metadata = self._build_epidemic_metadata(severity)

        scenario = Scenario(
            scenario_id=str(uuid.uuid4()),
            scenario_type=scenario_type,
            severity=severity,
            name=name,
            description=description,
            impact_duration=duration,
            financial_impact_range=(min(fin_lo, fin_hi), max(fin_lo, fin_hi)),
            clinical_impact_range=(clin_lo, clin_hi),
            metadata=metadata,
        )
        logger.info("Generated scenario: %s", scenario)
        return scenario

    def generate_batch(self, n: int = 10) -> List[Scenario]:
        """Generate *n* random scenarios."""
        return [self.generate() for _ in range(n)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_epidemic_metadata(self, severity: Severity) -> Dict:
        """Run an epidemic model and embed summary statistics."""
        params = {
            Severity.MILD:         dict(beta=0.20, gamma=0.15, days=90,  population=500_000),
            Severity.MODERATE:     dict(beta=0.30, gamma=0.10, days=180, population=2_000_000),
            Severity.SEVERE:       dict(beta=0.45, gamma=0.09, days=270, population=10_000_000),
            Severity.CATASTROPHIC: dict(beta=0.60, gamma=0.07, days=365, population=50_000_000),
        }[severity]

        use_seir = severity in (Severity.SEVERE, Severity.CATASTROPHIC)

        if use_seir:
            final, traj = run_seir_model(
                population=params["population"],
                beta=params["beta"],
                sigma=0.18,
                gamma=params["gamma"],
                days=params["days"],
            )
        else:
            final, traj = run_sir_model(
                population=params["population"],
                beta=params["beta"],
                gamma=params["gamma"],
                days=params["days"],
            )

        peak_I = max(s.I for s in traj)
        return {
            "model":          "SEIR" if use_seir else "SIR",
            "R0":             round(params["beta"] / params["gamma"], 2),
            "attack_rate":    round(final.attack_rate, 4),
            "peak_infected":  int(peak_I),
            "total_infected": int(final.R),
            "population":     int(params["population"]),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    gen = ScenarioGenerator(seed=7)

    print("=== Random scenario batch ===")
    for s in gen.generate_batch(6):
        print(f"  {s}")
        if s.metadata:
            print(f"    epidemic_meta: {s.metadata}")

    print("\n=== Forced catastrophic pandemic ===")
    cat = gen.generate(ScenarioType.PANDEMIC, Severity.CATASTROPHIC)
    print(f"  {cat}")
    print(f"  SIR/SEIR metadata: {cat.metadata}")
    print(f"  Sampled fin impact : {cat.sample_financial_impact(random.Random())*100:.2f}%")
    print(f"  Sampled clin impact: {cat.sample_clinical_impact(random.Random()):.0f} adverse events/100k")
