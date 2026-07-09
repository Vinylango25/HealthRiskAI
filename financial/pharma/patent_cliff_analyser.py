"""
patent_cliff_analyser.py
========================
Patent cliff analysis for pharmaceutical companies.

Models the financial impact when key drug patents expire, factoring in:
  - Generic entry erosion curves (small-molecule drugs)
  - Biosimilar competition dynamics (biologics/large-molecule)
  - Pipeline offset: revenue from upcoming products filling the gap
  - Geographic variation in patent protection and generic penetration

Key metrics:
  - revenue_at_risk (USD millions)
  - cliff_severity_score (0–10)
  - recovery_timeline (years to regain pre-cliff revenue baseline)

References:
  - Grabowski et al. (2007) The Market for Follow-On Biologics
  - FDA Generic Drug User Fee Amendments (GDUFA) data
  - IMS Health/IQVIA generic erosion benchmarks
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
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


# ---------------------------------------------------------------------------
# Constants — erosion curves from IQVIA / Grabowski benchmarks
# ---------------------------------------------------------------------------

# Generic erosion: fraction of branded sales remaining by year post-patent expiry
# Year 0 = expiry year, Year 1 = first full year of generic competition, etc.
GENERIC_EROSION_CURVE: Dict[int, float] = {
    0: 1.00,  # expiry year: still full sales
    1: 0.45,  # ~55% erosion in year 1 (FDA first-filer exclusivity effect)
    2: 0.25,
    3: 0.15,
    4: 0.10,
    5: 0.08,
}

# Biosimilar erosion: slower, less complete (30–50% penetration typical)
BIOSIMILAR_EROSION_CURVE: Dict[int, float] = {
    0: 1.00,
    1: 0.80,  # 20% erosion; biosimilar uptake slower
    2: 0.65,
    3: 0.55,
    4: 0.48,
    5: 0.42,
}

# Geographic patent protection adjustments (multiplier on US erosion)
GEO_EROSION_MULTIPLIER: Dict[str, float] = {
    "US": 1.00,
    "EU": 0.90,   # data exclusivity extends effective protection
    "Japan": 0.85,
    "China": 1.10,  # faster generic entry
    "RoW": 1.15,
}

# Revenue concentration by geography for a typical global pharma
DEFAULT_GEO_MIX: Dict[str, float] = {
    "US": 0.45,
    "EU": 0.25,
    "Japan": 0.10,
    "China": 0.10,
    "RoW": 0.10,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DrugPatent:
    """A single drug's patent information."""
    drug_name: str
    molecule_type: str           # "small_molecule" | "biologic"
    peak_annual_sales_mm: float  # USD millions at patent expiry
    patent_expiry_date: date
    geographies: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GEO_MIX))
    # Geography → expiry date overrides (some drugs have different expiry dates by region)
    geo_expiry_overrides: Dict[str, date] = field(default_factory=dict)
    indication: str = ""
    is_orphan: bool = False      # Orphan drugs have additional exclusivity
    has_pediatric_exclusivity: bool = False  # +6 months US exclusivity


@dataclass
class PipelineDrug:
    """An in-development drug that offsets patent cliff revenue."""
    drug_name: str
    indication: str
    phase: str                   # "phase1"|"phase2"|"phase3"|"approved"
    probability_of_success: float
    expected_launch_year: int
    peak_sales_mm: float         # expected peak annual sales
    ramp_years: int = 4          # years to reach peak sales


@dataclass
class CliffAnalysisResult:
    """Results of patent cliff analysis for a company."""
    company_name: str
    analysis_date: date
    revenue_at_risk_mm: float         # cumulative 5-year revenue lost to generics/biosimilars
    cliff_severity_score: float       # 0–10 scale
    recovery_timeline_yr: float       # years to recover to pre-cliff revenue level
    peak_cliff_year: int              # year with highest single-year revenue loss
    peak_cliff_loss_mm: float         # USD millions lost in worst year
    pipeline_offset_mm: float         # cumulative pipeline revenue offsets
    net_revenue_gap_mm: float         # revenue_at_risk - pipeline_offset
    annual_projections: List[Dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"=== Patent Cliff Analysis: {self.company_name} ===\n"
            f"  Revenue at risk (5yr)  : ${self.revenue_at_risk_mm:,.0f}M\n"
            f"  Cliff severity score   : {self.cliff_severity_score:.1f}/10\n"
            f"  Recovery timeline      : {self.recovery_timeline_yr:.1f} years\n"
            f"  Peak cliff year        : {self.peak_cliff_year} (−${self.peak_cliff_loss_mm:,.0f}M)\n"
            f"  Pipeline offset (5yr)  : ${self.pipeline_offset_mm:,.0f}M\n"
            f"  Net revenue gap (5yr)  : ${self.net_revenue_gap_mm:,.0f}M\n"
        )


# ---------------------------------------------------------------------------
# Patent Cliff Analyser
# ---------------------------------------------------------------------------

class PatentCliffAnalyser:
    """
    Analyses patent cliff exposure for a pharmaceutical company.

    Combines erosion curve modelling for each expiring drug with pipeline
    offset forecasting to compute net revenue impact over a 10-year horizon.

    Parameters
    ----------
    company_name : str
        Name of the company being analysed.
    base_year : int
        Starting year for projections (default: current year).
    forecast_horizon : int
        Number of years to project (default: 10).
    """

    def __init__(
        self,
        company_name: str,
        base_year: Optional[int] = None,
        forecast_horizon: int = 10,
    ):
        self.company_name = company_name
        self.base_year = base_year or date.today().year
        self.forecast_horizon = forecast_horizon
        self._patents: List[DrugPatent] = []
        self._pipeline: List[PipelineDrug] = []
        logger.info(
            "PatentCliffAnalyser ready: %s (base_year=%d, horizon=%d)",
            company_name, self.base_year, forecast_horizon,
        )

    def add_patent(self, patent: DrugPatent) -> None:
        """Register a drug patent for tracking."""
        self._patents.append(patent)
        logger.debug("Added patent: %s (expiry %s)", patent.drug_name, patent.patent_expiry_date)

    def add_pipeline_drug(self, drug: PipelineDrug) -> None:
        """Register a pipeline drug as a potential cliff offset."""
        self._pipeline.append(drug)
        logger.debug("Added pipeline drug: %s (phase %s, launch %d)", drug.drug_name, drug.phase, drug.expected_launch_year)

    def _get_erosion_curve(self, patent: DrugPatent) -> Dict[int, float]:
        """Return the appropriate erosion curve for a drug."""
        base = BIOSIMILAR_EROSION_CURVE if patent.molecule_type == "biologic" else GENERIC_EROSION_CURVE
        result = dict(base)
        # Orphan drug: extend exclusivity by 7 years (US) → shift cliff by 1 year
        if patent.is_orphan:
            result = {k + 1: v for k, v in result.items()}
            result[0] = 1.0
        # Paediatric: +0.5 year (approximate as shifting cliff 1 year for simplicity)
        if patent.has_pediatric_exclusivity:
            result = {k + 1: v for k, v in result.items()}
            result[0] = 1.0
        return result

    def _annual_branded_revenue(self, patent: DrugPatent, year: int) -> float:
        """
        Compute branded revenue for a drug in a given projection year,
        accounting for geographic patent expiry variation.

        Returns USD millions.
        """
        erosion_curve = self._get_erosion_curve(patent)
        total_revenue = 0.0

        for geo, geo_fraction in patent.geographies.items():
            # Get expiry date for this geography
            expiry = patent.geo_expiry_overrides.get(geo, patent.patent_expiry_date)
            years_post_expiry = year - expiry.year

            geo_mult = GEO_EROSION_MULTIPLIER.get(geo, 1.0)

            if years_post_expiry < 0:
                # Patent still in force
                fraction_remaining = 1.0
            elif years_post_expiry in erosion_curve:
                # Apply geographic multiplier: faster erosion in some markets
                base_fraction = erosion_curve[years_post_expiry]
                # Higher geo_mult → faster erosion → lower remaining fraction
                adjusted_erosion = 1 - (1 - base_fraction) * geo_mult
                fraction_remaining = max(0.0, adjusted_erosion)
            else:
                max_key = max(erosion_curve.keys())
                fraction_remaining = erosion_curve[max_key]

            total_revenue += patent.peak_annual_sales_mm * geo_fraction * fraction_remaining

        return total_revenue

    def _pipeline_revenue(self, year: int) -> float:
        """Compute probability-weighted pipeline revenue for a given year."""
        total = 0.0
        for drug in self._pipeline:
            if year < drug.expected_launch_year:
                continue
            years_post_launch = year - drug.expected_launch_year
            if years_post_launch < drug.ramp_years:
                # Linear ramp to peak
                ramp_frac = (years_post_launch + 1) / drug.ramp_years
            else:
                ramp_frac = 1.0
            total += drug.peak_sales_mm * ramp_frac * drug.probability_of_success
        return total

    def analyse(self) -> CliffAnalysisResult:
        """
        Run full patent cliff analysis.

        Returns
        -------
        CliffAnalysisResult
        """
        if not self._patents:
            raise ValueError("No patents added. Call add_patent() first.")

        years = list(range(self.base_year, self.base_year + self.forecast_horizon))
        projections: List[Dict[str, float]] = []

        # Baseline: revenue if patents never expired (counterfactual)
        baseline_revenue = sum(p.peak_annual_sales_mm for p in self._patents)

        cumulative_loss_5yr = 0.0
        cumulative_pipeline_5yr = 0.0
        worst_year, worst_loss = self.base_year, 0.0

        prev_total_branded = baseline_revenue

        for yr in years:
            branded = sum(self._annual_branded_revenue(p, yr) for p in self._patents)
            pipeline = self._pipeline_revenue(yr)
            annual_loss = max(0.0, prev_total_branded - branded)
            net_revenue = branded + pipeline

            row = {
                "year": float(yr),
                "branded_revenue_mm": round(branded, 1),
                "pipeline_revenue_mm": round(pipeline, 1),
                "total_revenue_mm": round(net_revenue, 1),
                "annual_cliff_loss_mm": round(annual_loss, 1),
                "cumulative_loss_mm": 0.0,  # filled below
            }
            projections.append(row)
            prev_total_branded = branded

            if yr <= self.base_year + 4:
                cumulative_loss_5yr += annual_loss
                cumulative_pipeline_5yr += pipeline

            if annual_loss > worst_loss:
                worst_loss = annual_loss
                worst_year = yr

        # Fill cumulative loss column
        running = 0.0
        for row in projections:
            running += row["annual_cliff_loss_mm"]
            row["cumulative_loss_mm"] = round(running, 1)

        # Recovery timeline: years until total_revenue ≥ baseline
        recovery_yr = None
        for row in projections:
            if row["total_revenue_mm"] >= baseline_revenue * 0.95:  # within 5% of baseline
                recovery_yr = row["year"] - self.base_year
                break
        recovery_timeline = float(recovery_yr) if recovery_yr is not None else float(self.forecast_horizon)

        # Severity score (0–10): based on % of revenue at risk over 5 years
        if baseline_revenue > 0:
            pct_at_risk = cumulative_loss_5yr / (baseline_revenue * 5)
            severity = float(np.clip(pct_at_risk * 10, 0, 10))
        else:
            severity = 0.0

        net_gap = cumulative_loss_5yr - cumulative_pipeline_5yr

        result = CliffAnalysisResult(
            company_name=self.company_name,
            analysis_date=date.today(),
            revenue_at_risk_mm=round(cumulative_loss_5yr, 1),
            cliff_severity_score=round(severity, 2),
            recovery_timeline_yr=round(recovery_timeline, 1),
            peak_cliff_year=worst_year,
            peak_cliff_loss_mm=round(worst_loss, 1),
            pipeline_offset_mm=round(cumulative_pipeline_5yr, 1),
            net_revenue_gap_mm=round(net_gap, 1),
            annual_projections=projections,
        )

        logger.info(
            "Cliff analysis complete: severity=%.1f/10, 5yr loss=$%.0fM, "
            "pipeline offset=$%.0fM",
            severity, cumulative_loss_5yr, cumulative_pipeline_5yr,
        )
        return result

    def forecast_revenue(self, n_years: int = 10) -> Any:
        """
        Return year-by-year revenue forecast as a DataFrame or list of dicts.

        Parameters
        ----------
        n_years : int
            Number of years to forecast.

        Returns
        -------
        DataFrame or list of dicts
        """
        self.forecast_horizon = n_years
        result = self.analyse()
        rows = result.annual_projections

        if _PANDAS_AVAILABLE:
            df = pd.DataFrame(rows)
            df["year"] = df["year"].astype(int)
            return df
        return rows

    def peer_comparison(
        self,
        peers: List["PatentCliffAnalyser"],
    ) -> Any:
        """
        Compare cliff severity and recovery metrics across peer companies.

        Parameters
        ----------
        peers : list of PatentCliffAnalyser
            Peer company analysers (each must have patents added).

        Returns
        -------
        DataFrame or list of dicts with peer comparison metrics.
        """
        all_analysers = [self] + peers
        rows = []

        for analyser in all_analysers:
            try:
                res = analyser.analyse()
                rows.append({
                    "company": analyser.company_name,
                    "revenue_at_risk_5yr_mm": res.revenue_at_risk_mm,
                    "cliff_severity_score": res.cliff_severity_score,
                    "recovery_timeline_yr": res.recovery_timeline_yr,
                    "pipeline_offset_5yr_mm": res.pipeline_offset_mm,
                    "net_gap_mm": res.net_revenue_gap_mm,
                    "peak_cliff_year": res.peak_cliff_year,
                })
            except ValueError as exc:
                logger.warning("Peer %s skipped: %s", analyser.company_name, exc)

        if _PANDAS_AVAILABLE:
            df = pd.DataFrame(rows)
            return df.sort_values("cliff_severity_score", ascending=False).reset_index(drop=True)
        return sorted(rows, key=lambda r: -r["cliff_severity_score"])


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Patent Cliff Analyser — Synthetic Smoke Test")
    print("=" * 60)

    # ---- Company A: Large pharma with significant near-term cliff ----
    analyser_a = PatentCliffAnalyser("PharmaAlpha", base_year=2024, forecast_horizon=10)

    analyser_a.add_patent(DrugPatent(
        drug_name="AlphaBloc",
        molecule_type="small_molecule",
        peak_annual_sales_mm=4_200.0,
        patent_expiry_date=date(2025, 6, 30),
        indication="Cardiovascular",
        has_pediatric_exclusivity=True,
    ))
    analyser_a.add_patent(DrugPatent(
        drug_name="ImmunoBio",
        molecule_type="biologic",
        peak_annual_sales_mm=2_800.0,
        patent_expiry_date=date(2026, 12, 31),
        indication="Oncology",
    ))
    analyser_a.add_patent(DrugPatent(
        drug_name="NeuroPlex",
        molecule_type="small_molecule",
        peak_annual_sales_mm=1_500.0,
        patent_expiry_date=date(2029, 3, 15),
        indication="CNS",
        is_orphan=True,
    ))

    # Pipeline drugs to offset the cliff
    analyser_a.add_pipeline_drug(PipelineDrug(
        drug_name="AlphaNext",
        indication="Cardiovascular",
        phase="phase3",
        probability_of_success=0.60 * 0.85,  # phase3 × NDA
        expected_launch_year=2027,
        peak_sales_mm=2_100.0,
        ramp_years=5,
    ))
    analyser_a.add_pipeline_drug(PipelineDrug(
        drug_name="OncoGene-X",
        indication="Oncology",
        phase="phase2",
        probability_of_success=0.40 * 0.60 * 0.85,
        expected_launch_year=2030,
        peak_sales_mm=3_500.0,
        ramp_years=6,
    ))

    print("\n--- Analyse: PharmaAlpha ---")
    result_a = analyser_a.analyse()
    print(result_a.summary())

    print("\nYear-by-year projections:")
    for row in result_a.annual_projections[:7]:
        print(
            f"  {int(row['year'])}: branded=${row['branded_revenue_mm']:,.0f}M  "
            f"pipeline=${row['pipeline_revenue_mm']:,.0f}M  "
            f"cliff_loss=${row['annual_cliff_loss_mm']:,.0f}M"
        )

    # ---- Revenue forecast ----
    print("\n--- Revenue forecast (10 years) ---")
    forecast = analyser_a.forecast_revenue(n_years=10)
    if _PANDAS_AVAILABLE:
        print(forecast[["year", "branded_revenue_mm", "pipeline_revenue_mm", "total_revenue_mm"]].to_string(index=False))
    else:
        for row in forecast[:5]:
            print(f"  {row}")

    # ---- Peer comparison ----
    print("\n--- Peer comparison ---")
    analyser_b = PatentCliffAnalyser("BetaPharma", base_year=2024, forecast_horizon=10)
    analyser_b.add_patent(DrugPatent(
        drug_name="BetaCore",
        molecule_type="small_molecule",
        peak_annual_sales_mm=3_000.0,
        patent_expiry_date=date(2027, 9, 30),
        indication="Metabolic",
    ))
    analyser_b.add_pipeline_drug(PipelineDrug(
        drug_name="BetaNext",
        indication="Metabolic",
        phase="phase3",
        probability_of_success=0.51,
        expected_launch_year=2028,
        peak_sales_mm=2_800.0,
        ramp_years=4,
    ))

    peers = analyser_a.peer_comparison([analyser_b])
    if _PANDAS_AVAILABLE:
        print(peers.to_string(index=False))
    else:
        for row in peers:
            print(f"  {row}")

    # ---- Assertions ----
    assert result_a.revenue_at_risk_mm >= 0, "Revenue at risk must be non-negative"
    assert 0.0 <= result_a.cliff_severity_score <= 10.0, "Severity out of range"
    assert result_a.recovery_timeline_yr >= 0, "Recovery timeline must be positive"
    assert len(result_a.annual_projections) == 10, "Should have 10 projection rows"
    assert result_a.pipeline_offset_mm >= 0, "Pipeline offset must be non-negative"

    print("\n✓ All smoke-test assertions passed.")
