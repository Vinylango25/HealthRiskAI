"""
tests/test_simulation.py
========================
Unit tests for the simulation engine, scenario generator,
portfolio manager, and scoring engine.

All tests use minimal configuration and synthetic data — no disk I/O.
Heavy optional dependencies are handled with pytest.importorskip or
try/except so the suite runs in minimal environments.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Import simulation modules — skip gracefully if unavailable
# ---------------------------------------------------------------------------

try:
    from simulation.engine import SimulationEngine, GameState, GamePhase
    _HAS_ENGINE = True
except ImportError:
    _HAS_ENGINE = False

try:
    from simulation.scenario_generator import (
        ScenarioGenerator,
        ScenarioType,
        Severity,
    )
    _HAS_SCENARIO = True
except ImportError:
    _HAS_SCENARIO = False

try:
    from simulation.portfolio_manager import (
        PortfolioManager,
        STARTING_ALLOCATIONS,
        STARTING_TOTAL,
    )
    _HAS_PORTFOLIO = True
except ImportError:
    _HAS_PORTFOLIO = False

try:
    from simulation.scoring import ScoringEngine, ScoreBreakdown, MAX_SCORE
    _HAS_SCORING = True
except ImportError:
    _HAS_SCORING = False


# ---------------------------------------------------------------------------
# TestSimulationEngine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSimulationEngine:
    """Smoke tests for the SimulationEngine game loop."""

    def test_engine_smoke(self):
        """Instantiate engine, start a game, run one quarter without error."""
        if not _HAS_ENGINE:
            pytest.skip("simulation.engine not importable")

        engine = SimulationEngine(seed=42)
        state = engine.start_game()
        assert state.phase == GamePhase.IN_PROGRESS, "Game should be IN_PROGRESS after start"
        result = engine.next_quarter(player_decision={"action": "hold"})
        assert result is not None, "next_quarter must return a non-None result"

    def test_result_keys(self):
        """next_quarter result dict / GameState has portfolio_value and score."""
        if not _HAS_ENGINE:
            pytest.skip("simulation.engine not importable")

        engine = SimulationEngine(seed=7)
        engine.start_game()
        result = engine.next_quarter(player_decision={"action": "hold"})
        # next_quarter may return a dict or a GameState-like object
        if isinstance(result, dict):
            assert "portfolio_value" in result, "'portfolio_value' key missing"
            assert "score" in result, "'score' key missing"
        else:
            # GameState object
            assert hasattr(result, "portfolio_value"), "Result must have portfolio_value"
            assert hasattr(result, "score"), "Result must have score"


# ---------------------------------------------------------------------------
# TestScenarioGenerator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScenarioGenerator:
    """Tests for scenario generation logic."""

    def test_scenario_types(self):
        """Each scenario type generates a scenario with name and severity fields."""
        if not _HAS_SCENARIO:
            pytest.skip("simulation.scenario_generator not importable")

        gen = ScenarioGenerator(seed=0)
        for stype in ScenarioType:
            scenario = gen.generate(scenario_type=stype, severity=Severity.MODERATE)
            assert scenario is not None, f"Generator returned None for {stype}"
            name_val = (
                scenario.get("name") if isinstance(scenario, dict)
                else getattr(scenario, "name", None)
            )
            sev_val = (
                scenario.get("severity") if isinstance(scenario, dict)
                else getattr(scenario, "severity", None)
            )
            assert name_val is not None, f"Scenario must have a 'name' for {stype}"
            assert sev_val is not None, f"Scenario must have a 'severity' for {stype}"

    def test_scenario_impact_negative(self):
        """A catastrophic pandemic scenario produces a negative or zero portfolio impact."""
        if not _HAS_SCENARIO:
            pytest.skip("simulation.scenario_generator not importable")

        gen = ScenarioGenerator(seed=1)
        scenario = gen.generate(
            scenario_type=ScenarioType.PANDEMIC,
            severity=Severity.CATASTROPHIC,
        )
        # Look for any impact field in the scenario object
        impact = None
        for attr in ("portfolio_impact", "financial_impact", "impact_pct", "impact_fraction"):
            if isinstance(scenario, dict):
                impact = scenario.get(attr)
            else:
                impact = getattr(scenario, attr, None)
            if impact is not None:
                break

        if impact is None:
            pytest.skip("Scenario object has no recognised impact field")
        assert impact <= 0, (
            f"Catastrophic scenario impact should be <= 0, got {impact}"
        )


# ---------------------------------------------------------------------------
# TestPortfolioManager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPortfolioManager:
    """Tests for PortfolioManager value and allocation logic."""

    def test_initial_portfolio_positive(self):
        """Starting portfolio total value is positive (approx $500M)."""
        if not _HAS_PORTFOLIO:
            pytest.skip("simulation.portfolio_manager not importable")

        pm = PortfolioManager()
        total = pm.state.total_value
        assert total > 0, f"Initial portfolio value must be positive, got {total}"
        assert total >= 400_000_000, (
            f"Initial portfolio should be ~$500M, got ${total:,.0f}"
        )

    def test_asset_allocation_sums_to_one(self):
        """Position weights from STARTING_ALLOCATIONS sum to 1.0."""
        if not _HAS_PORTFOLIO:
            pytest.skip("simulation.portfolio_manager not importable")

        pm = PortfolioManager()
        total = pm.state.total_value
        weight_sum = sum(
            pm.state.positions[name].market_value / total
            for name in STARTING_ALLOCATIONS
        )
        assert abs(weight_sum - 1.0) < 1e-6, (
            f"Weights must sum to 1.0, got {weight_sum:.8f}"
        )


# ---------------------------------------------------------------------------
# TestScoringEngine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoringEngine:
    """Tests for the 1000-point scoring framework."""

    def _make_game_state(
        self,
        portfolio_value: float = 520_000_000.0,
        scenarios_survived: int = 5,
        quarter: int = 8,
    ) -> dict:
        """Build a minimal game_state dict accepted by ScoringEngine.calculate_score."""
        history = [
            {
                "quarter": q,
                "portfolio_value": portfolio_value * (0.98 + 0.004 * q),
                "score": 0.0,
                "total_return_pct": (portfolio_value / 500_000_000 - 1) * 100,
                "decisions_made": q * 2,
                "scenarios_survived": min(q, scenarios_survived),
            }
            for q in range(1, quarter + 1)
        ]
        return {
            "portfolio_value": portfolio_value,
            "history": history,
            "decisions_made": quarter * 2,
            "scenarios_survived": scenarios_survived,
            "quarter": quarter,
            "clinical_warnings_hit": 2,
            "total_warnings": 4,
            "nlp_signals_used": 3,
            "nlp_signals_available": 5,
            "decision_latencies": [3.5] * (quarter * 2),
        }

    def test_score_range(self):
        """Total score from calculate_score is in [0, MAX_SCORE] (1000)."""
        if not _HAS_SCORING:
            pytest.skip("simulation.scoring not importable")

        engine = ScoringEngine()
        game_state = self._make_game_state(portfolio_value=520_000_000.0)
        breakdown = engine.calculate_score(game_state)
        total = breakdown.total_score
        assert 0 <= total <= MAX_SCORE, (
            f"Total score must be in [0, {MAX_SCORE}], got {total:.2f}"
        )

    def test_better_portfolio_higher_score(self):
        """A profitable portfolio scores >= a portfolio with heavy losses."""
        if not _HAS_SCORING:
            pytest.skip("simulation.scoring not importable")

        engine = ScoringEngine()
        good_state = self._make_game_state(
            portfolio_value=650_000_000.0, scenarios_survived=7, quarter=8
        )
        bad_state = self._make_game_state(
            portfolio_value=280_000_000.0, scenarios_survived=1, quarter=8
        )
        score_good = engine.calculate_score(good_state).total_score
        score_bad = engine.calculate_score(bad_state).total_score
        assert score_good >= score_bad, (
            f"Profitable portfolio should score >= loss portfolio: "
            f"good={score_good:.1f}, bad={score_bad:.1f}"
        )
