"""
rnpv_calculator.py
==================
Risk-adjusted Net Present Value (rNPV) calculator for pharmaceutical drug pipelines.

Uses Monte Carlo simulation with indication-adjusted phase success probabilities
to model the full distribution of a drug program's value, incorporating realistic
development costs, peak sales projections, and patent cliff dynamics.

References:
    - Hay et al. (2014) Clinical Development Success Rates for Investigational Drugs
    - ClinicalTrials.gov aggregate success rate data
    - DiMasi et al. (2016) Cost of Drug Development
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Constants: baseline phase success probabilities (Hay et al. 2014)
# ---------------------------------------------------------------------------
BASELINE_SUCCESS_PROBS: Dict[str, float] = {
    "phase1": 0.65,
    "phase2": 0.40,
    "phase3": 0.60,
    "nda": 0.85,
}

# Indication multipliers derived from ClinicalTrials.gov aggregate data
# Values > 1 indicate higher-than-average success; < 1 lower-than-average
INDICATION_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "oncology": {"phase1": 0.90, "phase2": 0.75, "phase3": 0.80, "nda": 0.95},
    "cardiovascular": {"phase1": 1.05, "phase2": 1.10, "phase3": 1.05, "nda": 1.00},
    "cns": {"phase1": 0.95, "phase2": 0.70, "phase3": 0.75, "nda": 0.90},
    "infectious_disease": {"phase1": 1.10, "phase2": 1.15, "phase3": 1.15, "nda": 1.05},
    "rare_disease": {"phase1": 1.00, "phase2": 0.85, "phase3": 0.90, "nda": 1.10},
    "metabolic": {"phase1": 1.05, "phase2": 1.00, "phase3": 0.95, "nda": 1.00},
    "default": {"phase1": 1.00, "phase2": 1.00, "phase3": 1.00, "nda": 1.00},
}

# Average development costs per phase (USD millions, DiMasi 2016 adjusted to 2024)
PHASE_COSTS_MM: Dict[str, float] = {
    "phase1": 25.0,
    "phase2": 58.0,
    "phase3": 255.0,
    "nda": 35.0,
}

# Average phase durations in years
PHASE_DURATIONS_YR: Dict[str, float] = {
    "phase1": 1.5,
    "phase2": 2.5,
    "phase3": 3.5,
    "nda": 1.5,
}


@dataclass
class DrugProgram:
    """Represents a single drug development program."""

    name: str
    indication: str = "default"
    current_phase: str = "phase1"  # phase1 | phase2 | phase3 | nda | approved

    # Commercial parameters (USD millions)
    peak_sales_mm: float = 1_000.0
    peak_sales_year: int = 5          # years post-approval
    commercial_life_yr: int = 12      # years of patent-protected sales
    patent_cliff_erosion: float = 0.70  # fraction of sales lost at cliff

    # Financial parameters
    discount_rate: float = 0.10       # WACC
    royalty_rate: float = 0.0         # if partnered; 0 = fully owned

    # Optional overrides
    cost_overrides: Dict[str, float] = field(default_factory=dict)
    duration_overrides: Dict[str, float] = field(default_factory=dict)

    # Market ramp shape: fraction of peak sales each year 1..peak_sales_year
    market_ramp: Optional[List[float]] = None


@dataclass
class RNPVResult:
    """Holds results from an rNPV calculation run."""

    program_name: str
    n_simulations: int

    rnpv_mean: float = 0.0
    rnpv_std: float = 0.0
    rnpv_p5: float = 0.0
    rnpv_p50: float = 0.0
    rnpv_p95: float = 0.0

    irr_mean: float = 0.0
    irr_p5: float = 0.0
    irr_p50: float = 0.0
    irr_p95: float = 0.0

    probability_of_success: float = 0.0
    expected_value: float = 0.0      # PoS-weighted NPV

    npv_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    irr_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        return (
            f"=== rNPV Summary: {self.program_name} ===\n"
            f"  N simulations  : {self.n_simulations:,}\n"
            f"  rNPV mean      : ${self.rnpv_mean:.1f}M\n"
            f"  rNPV std       : ${self.rnpv_std:.1f}M\n"
            f"  NPV P5/P50/P95 : ${self.rnpv_p5:.1f}M / ${self.rnpv_p50:.1f}M / ${self.rnpv_p95:.1f}M\n"
            f"  IRR P50        : {self.irr_p50*100:.1f}%\n"
            f"  Prob of success: {self.probability_of_success*100:.1f}%\n"
            f"  Expected value : ${self.expected_value:.1f}M\n"
        )


class RNPVCalculator:
    """
    Risk-adjusted NPV calculator using Monte Carlo simulation.

    For each simulation path:
      1. Draw Bernoulli outcomes for each remaining phase (phase success/fail).
      2. Build a cash-flow timeline: costs during development, revenues after approval.
      3. Discount all cash flows back to today.
      4. Aggregate results across 10,000 paths.
    """

    def __init__(self, n_simulations: int = 10_000, seed: Optional[int] = 42):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)
        logger.info("RNPVCalculator initialised (n=%d, seed=%s)", n_simulations, seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adjusted_probs(self, indication: str) -> Dict[str, float]:
        """Return indication-adjusted success probabilities, clipped to [0.01, 0.99]."""
        mults = INDICATION_MULTIPLIERS.get(indication, INDICATION_MULTIPLIERS["default"])
        return {
            phase: float(np.clip(BASELINE_SUCCESS_PROBS[phase] * mults[phase], 0.01, 0.99))
            for phase in BASELINE_SUCCESS_PROBS
        }

    def _phase_sequence(self, current_phase: str) -> List[str]:
        """Return phases still to be completed, starting from current_phase."""
        all_phases = ["phase1", "phase2", "phase3", "nda"]
        try:
            idx = all_phases.index(current_phase)
            return all_phases[idx:]
        except ValueError:
            return []  # already approved

    def _build_cashflows(
        self,
        program: DrugProgram,
        phases_remaining: List[str],
        approved: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build arrays of (years, cashflows_mm) for a single simulation path.

        Returns two arrays of equal length: time points (years from now) and
        cash flows (USD millions, negative = costs).
        """
        times: List[float] = []
        cfs: List[float] = []

        t = 0.0
        for phase in phases_remaining:
            cost = program.cost_overrides.get(phase, PHASE_COSTS_MM[phase])
            dur = program.duration_overrides.get(phase, PHASE_DURATIONS_YR[phase])
            # Cost paid at midpoint of phase
            times.append(t + dur / 2)
            cfs.append(-cost)
            t += dur

        if approved:
            ramp = program.market_ramp or np.linspace(0.1, 1.0, program.peak_sales_year).tolist()
            # Ramp-up years
            for i, frac in enumerate(ramp):
                times.append(t + i + 1)
                revenue = program.peak_sales_mm * frac * (1 - program.royalty_rate)
                cfs.append(revenue)
            # Plateau + cliff
            plateau_years = program.commercial_life_yr - program.peak_sales_year
            for yr in range(plateau_years):
                times.append(t + program.peak_sales_year + yr + 1)
                cfs.append(program.peak_sales_mm * (1 - program.royalty_rate))
            # Post-cliff (generic erosion), assume 5 more years
            for yr in range(5):
                times.append(t + program.commercial_life_yr + yr + 1)
                post_cliff = program.peak_sales_mm * (1 - program.patent_cliff_erosion)
                cfs.append(post_cliff * (1 - program.royalty_rate))

        return np.array(times), np.array(cfs)

    def _npv_from_cashflows(
        self, times: np.ndarray, cfs: np.ndarray, rate: float
    ) -> float:
        """Discount cash flows to time 0."""
        if len(cfs) == 0:
            return 0.0
        discount_factors = (1 + rate) ** (-times)
        return float(np.sum(cfs * discount_factors))

    def _irr_from_cashflows(
        self, times: np.ndarray, cfs: np.ndarray
    ) -> float:
        """Estimate IRR via bisection on NPV(r) = 0. Returns NaN if no solution."""
        if len(cfs) == 0 or np.all(cfs <= 0):
            return float("nan")
        lo, hi = -0.99, 10.0
        for _ in range(100):
            mid = (lo + hi) / 2
            npv = self._npv_from_cashflows(times, cfs, mid)
            if abs(npv) < 1e-3:
                return mid
            if npv > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, program: DrugProgram) -> RNPVResult:
        """
        Run Monte Carlo rNPV calculation for a drug program.

        Parameters
        ----------
        program : DrugProgram
            The drug development program to value.

        Returns
        -------
        RNPVResult
            Full distribution of NPV and IRR outcomes.
        """
        logger.info("Starting rNPV calculation for '%s' (%s)", program.name, program.indication)

        adj_probs = self._adjusted_probs(program.indication)
        phases = self._phase_sequence(program.current_phase)

        npv_paths: List[float] = []
        irr_paths: List[float] = []
        successes = 0

        for _ in range(self.n_simulations):
            phases_completed: List[str] = []
            approved = True

            for phase in phases:
                p_success = adj_probs[phase]
                # Add noise to probability (~5% std, reflecting estimation uncertainty)
                p_noisy = float(np.clip(
                    p_success + self.rng.normal(0, 0.05), 0.01, 0.99
                ))
                if self.rng.random() > p_noisy:
                    approved = False
                    phases_completed.append(phase)  # costs incurred up to failure
                    break
                phases_completed.append(phase)

            times, cfs = self._build_cashflows(program, phases_completed, approved)

            # Add stochastic noise to peak sales (log-normal, ~30% CV)
            if approved and len(cfs) > 0:
                sales_mult = float(self.rng.lognormal(mean=0.0, sigma=0.30))
                # Revenue CFs come after development phase index
                n_dev = len(phases_completed)
                if len(cfs) > n_dev:
                    cfs[n_dev:] *= sales_mult
                successes += 1

            npv = self._npv_from_cashflows(times, cfs, program.discount_rate)
            irr = self._irr_from_cashflows(times, cfs)
            npv_paths.append(npv)
            irr_paths.append(irr)

        npv_arr = np.array(npv_paths)
        irr_arr = np.array(irr_paths)
        valid_irr = irr_arr[np.isfinite(irr_arr)]

        result = RNPVResult(
            program_name=program.name,
            n_simulations=self.n_simulations,
            rnpv_mean=float(np.mean(npv_arr)),
            rnpv_std=float(np.std(npv_arr)),
            rnpv_p5=float(np.percentile(npv_arr, 5)),
            rnpv_p50=float(np.percentile(npv_arr, 50)),
            rnpv_p95=float(np.percentile(npv_arr, 95)),
            irr_mean=float(np.mean(valid_irr)) if len(valid_irr) else float("nan"),
            irr_p5=float(np.percentile(valid_irr, 5)) if len(valid_irr) else float("nan"),
            irr_p50=float(np.percentile(valid_irr, 50)) if len(valid_irr) else float("nan"),
            irr_p95=float(np.percentile(valid_irr, 95)) if len(valid_irr) else float("nan"),
            probability_of_success=successes / self.n_simulations,
            expected_value=float(np.mean(npv_arr[npv_arr > 0])) * (successes / self.n_simulations)
            if successes > 0 else 0.0,
            npv_distribution=npv_arr,
            irr_distribution=irr_arr,
        )

        logger.info("Completed rNPV: mean=$%.1fM, PoS=%.1f%%", result.rnpv_mean, result.probability_of_success * 100)
        return result

    def sensitivity_analysis(
        self,
        program: DrugProgram,
        parameters: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, List[float]]:
        """
        One-at-a-time sensitivity analysis across key parameters.

        Parameters
        ----------
        program : DrugProgram
            Base-case program.
        parameters : dict, optional
            Mapping of parameter name → list of values to test.
            Defaults to testing peak_sales_mm and discount_rate.

        Returns
        -------
        dict
            Mapping parameter_name → list of rNPV means for each value.
        """
        if parameters is None:
            parameters = {
                "peak_sales_mm": [500, 750, 1000, 1500, 2000],
                "discount_rate": [0.06, 0.08, 0.10, 0.12, 0.15],
                "patent_cliff_erosion": [0.50, 0.60, 0.70, 0.80, 0.90],
            }

        results: Dict[str, List[float]] = {}

        for param, values in parameters.items():
            rnpvs: List[float] = []
            for val in values:
                import copy
                prog_copy = copy.deepcopy(program)
                setattr(prog_copy, param, val)
                res = self.calculate(prog_copy)
                rnpvs.append(res.rnpv_mean)
                logger.debug("  sensitivity %s=%.2f → rNPV=$%.1fM", param, val, res.rnpv_mean)
            results[param] = rnpvs
            logger.info("Sensitivity on '%s': range $%.1fM – $%.1fM", param, min(rnpvs), max(rnpvs))

        return results


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("rNPV Calculator — Synthetic Smoke Test")
    print("=" * 60)

    calc = RNPVCalculator(n_simulations=10_000, seed=2024)

    # Program 1: Early-stage oncology asset
    oncology_drug = DrugProgram(
        name="ONC-001 (Phase 1 NSCLC)",
        indication="oncology",
        current_phase="phase1",
        peak_sales_mm=1_200.0,
        peak_sales_year=6,
        commercial_life_yr=14,
        patent_cliff_erosion=0.65,
        discount_rate=0.12,
    )

    result1 = calc.calculate(oncology_drug)
    print(result1.summary())

    # Program 2: Late-stage cardiovascular asset
    cardio_drug = DrugProgram(
        name="CV-042 (Phase 3 Heart Failure)",
        indication="cardiovascular",
        current_phase="phase3",
        peak_sales_mm=2_500.0,
        peak_sales_year=4,
        commercial_life_yr=10,
        patent_cliff_erosion=0.72,
        discount_rate=0.09,
    )

    result2 = calc.calculate(cardio_drug)
    print(result2.summary())

    # Sensitivity analysis on oncology asset
    print("\nRunning sensitivity analysis on ONC-001...")
    sens = calc.sensitivity_analysis(
        oncology_drug,
        parameters={
            "peak_sales_mm": [600, 900, 1200, 1800],
            "discount_rate": [0.08, 0.10, 0.12, 0.15],
        },
    )
    for param, rnpvs in sens.items():
        print(f"  {param}: {[f'${v:.0f}M' for v in rnpvs]}")

    # Assertions
    assert result1.n_simulations == 10_000, "Simulation count mismatch"
    assert result1.rnpv_p5 < result1.rnpv_p50 < result1.rnpv_p95, "Percentile ordering violated"
    assert 0.0 <= result1.probability_of_success <= 1.0, "PoS out of range"
    assert result2.rnpv_p50 > result1.rnpv_p50, "Late-stage should have higher rNPV than Phase 1"
    print("\n✓ All smoke-test assertions passed.")
