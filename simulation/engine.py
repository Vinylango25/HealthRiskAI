"""
HealthRisk Lab Simulation Engine
=================================
Drives 10-year (40-quarter) portfolio simulation cycles for the HealthRisk AI
training platform.  Each quarter executes the canonical game loop:

    generate_scenario → player_decision → ai_decision
    → apply_outcomes → update_scores

Events are emitted at every state change so UI layers or loggers can react
without polling the engine directly.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    PORTFOLIO = auto()
    SCENARIO  = auto()
    SCORING   = auto()
    GAME      = auto()


@dataclass
class BaseEvent:
    """Common envelope for all simulation events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: EventType = EventType.GAME
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioEvent(BaseEvent):
    """Fired whenever the portfolio value changes."""
    event_type: EventType = EventType.PORTFOLIO

    @classmethod
    def create(cls, quarter: int, old_value: float, new_value: float,
               reason: str) -> "PortfolioEvent":
        return cls(payload={
            "quarter":   quarter,
            "old_value": old_value,
            "new_value": new_value,
            "change":    new_value - old_value,
            "reason":    reason,
        })


@dataclass
class ScenarioEvent(BaseEvent):
    """Fired when a scenario is generated or resolved."""
    event_type: EventType = EventType.SCENARIO

    @classmethod
    def create(cls, quarter: int, scenario_name: str, severity: str,
               phase: str) -> "ScenarioEvent":
        return cls(payload={
            "quarter":       quarter,
            "scenario_name": scenario_name,
            "severity":      severity,
            "phase":         phase,
        })


@dataclass
class ScoringEvent(BaseEvent):
    """Fired when the score is updated."""
    event_type: EventType = EventType.SCORING

    @classmethod
    def create(cls, quarter: int, old_score: float, new_score: float,
               breakdown: Dict[str, float]) -> "ScoringEvent":
        return cls(payload={
            "quarter":   quarter,
            "old_score": old_score,
            "new_score": new_score,
            "delta":     new_score - old_score,
            "breakdown": breakdown,
        })


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class GamePhase(Enum):
    NOT_STARTED  = "not_started"
    IN_PROGRESS  = "in_progress"
    ENDED        = "ended"


@dataclass
class GameState:
    """
    Immutable snapshot of the simulation at a given quarter.

    Attributes
    ----------
    quarter           : Current quarter number (1–40).
    portfolio_value   : Mark-to-market portfolio value in USD.
    score             : Cumulative score (0–1000).
    decisions_made    : Total decisions recorded (player + AI combined).
    scenarios_survived: Scenarios where portfolio loss < 10%.
    phase             : Life-cycle phase of the game.
    history           : Per-quarter record for replay / charts.
    active_scenario   : The scenario being processed this quarter (if any).
    player_decision   : Most recent player action dict.
    ai_decision       : Most recent AI action dict.
    """
    quarter:            int   = 0
    portfolio_value:    float = 500_000_000.0   # $500 M starting value
    score:              float = 0.0
    decisions_made:     int   = 0
    scenarios_survived: int   = 0
    phase:              GamePhase = GamePhase.NOT_STARTED
    history:            List[Dict[str, Any]] = field(default_factory=list)
    active_scenario:    Optional[Dict[str, Any]] = None
    player_decision:    Optional[Dict[str, Any]] = None
    ai_decision:        Optional[Dict[str, Any]] = None

    # Derived helpers
    @property
    def total_return(self) -> float:
        """Percentage return from starting $500M."""
        return (self.portfolio_value - 500_000_000.0) / 500_000_000.0 * 100

    @property
    def is_active(self) -> bool:
        return self.phase == GamePhase.IN_PROGRESS

    def snapshot(self) -> Dict[str, Any]:
        """Return a dict suitable for history logging."""
        return {
            "quarter":          self.quarter,
            "portfolio_value":  self.portfolio_value,
            "score":            self.score,
            "total_return_pct": round(self.total_return, 4),
            "decisions_made":   self.decisions_made,
            "scenarios_survived": self.scenarios_survived,
        }


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

TOTAL_QUARTERS   = 40          # 10 years
STARTING_VALUE   = 500_000_000 # $500M
SCENARIO_CHANCE  = 0.60        # probability a scenario fires each quarter
BANKRUPTCY_FLOOR = 50_000_000  # < $50M → game over


class SimulationEngine:
    """
    Orchestrates the HealthRisk Lab turn-based simulation.

    Usage
    -----
    >>> engine = SimulationEngine()
    >>> engine.start_game()
    >>> while engine.state.is_active:
    ...     result = engine.next_quarter(
    ...         player_decision={"action": "increase_insurance_reserves"},
    ...     )
    >>> summary = engine.end_game()
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._state = GameState()
        self._listeners: Dict[EventType, List[Callable[[BaseEvent], None]]] = {
            et: [] for et in EventType
        }
        logger.info("SimulationEngine initialised (seed=%s)", seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_game(self) -> GameState:
        """Initialise a fresh game and return the opening state."""
        self._state = GameState(
            quarter=0,
            portfolio_value=float(STARTING_VALUE),
            score=0.0,
            decisions_made=0,
            scenarios_survived=0,
            phase=GamePhase.IN_PROGRESS,
        )
        self._emit(BaseEvent(payload={"message": "Game started", "quarters": TOTAL_QUARTERS}))
        logger.info("Game started: $%,.0f starting portfolio, %d quarters", STARTING_VALUE, TOTAL_QUARTERS)
        return self._state

    def next_quarter(
        self,
        player_decision: Optional[Dict[str, Any]] = None,
        decision_latency_s: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Advance simulation by one quarter.

        Parameters
        ----------
        player_decision     : Dict describing the player's action this turn.
        decision_latency_s  : Seconds the player took to decide (for speed bonus).

        Returns
        -------
        Quarter result dict with scenario, decisions, outcome, and new state.
        """
        if not self._state.is_active:
            raise RuntimeError("Game is not in progress. Call start_game() first.")

        self._state.quarter += 1
        q = self._state.quarter
        logger.info("--- Quarter %d / %d ---", q, TOTAL_QUARTERS)

        # Step 1 – generate scenario
        scenario = self._generate_scenario()

        # Step 2 – record player decision
        p_decision = self._record_player_decision(player_decision or {"action": "do_nothing"})

        # Step 3 – AI decision
        ai_dec = self._ai_decision(scenario)

        # Step 4 – apply outcomes
        old_value = self._state.portfolio_value
        outcome = self._apply_outcomes(scenario, p_decision, ai_dec)

        # Step 5 – update scores
        old_score = self._state.score
        score_delta, breakdown = self._update_scores(
            scenario, outcome, decision_latency_s
        )

        # Persist snapshot
        self._state.history.append(self._state.snapshot())

        result = {
            "quarter":         q,
            "scenario":        scenario,
            "player_decision": p_decision,
            "ai_decision":     ai_dec,
            "outcome":         outcome,
            "score_delta":     score_delta,
            "score_breakdown": breakdown,
            "portfolio_value": self._state.portfolio_value,
            "score":           self._state.score,
        }

        # Check end conditions
        if q >= TOTAL_QUARTERS or self._state.portfolio_value < BANKRUPTCY_FLOOR:
            self._state.phase = GamePhase.ENDED
            logger.info("Game over triggered at quarter %d", q)

        return result

    def end_game(self) -> Dict[str, Any]:
        """Finalise the game and return the summary report."""
        self._state.phase = GamePhase.ENDED
        summary = {
            "final_quarter":      self._state.quarter,
            "final_portfolio":    self._state.portfolio_value,
            "final_score":        self._state.score,
            "total_return_pct":   round(self._state.total_return, 2),
            "decisions_made":     self._state.decisions_made,
            "scenarios_survived": self._state.scenarios_survived,
            "history":            self._state.history,
            "grade":              self._compute_grade(),
        }
        self._emit(BaseEvent(payload={"message": "Game ended", **summary}))
        logger.info(
            "Game ended | Final score: %.1f | Return: %.2f%% | Grade: %s",
            self._state.score, self._state.total_return, summary["grade"],
        )
        return summary

    def get_state(self) -> GameState:
        """Return the current (mutable) game state."""
        return self._state

    def register_listener(
        self, event_type: EventType, callback: Callable[[BaseEvent], None]
    ) -> None:
        """Subscribe a callable to a specific event type."""
        self._listeners[event_type].append(callback)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_scenario(self) -> Optional[Dict[str, Any]]:
        """Probabilistically generate a scenario for this quarter."""
        if self._rng.random() > SCENARIO_CHANCE:
            logger.debug("Quarter %d: no scenario this quarter", self._state.quarter)
            self._state.active_scenario = None
            return None

        types = ["pandemic", "drug_safety_crisis", "regulatory_change", "hospital_merger"]
        severities = ["mild", "moderate", "severe", "catastrophic"]
        weights = [0.40, 0.30, 0.20, 0.10]

        scenario_type = self._rng.choice(types)
        severity      = self._rng.choices(severities, weights=weights, k=1)[0]

        scenario = {
            "type":     scenario_type,
            "severity": severity,
            "quarter":  self._state.quarter,
            "name":     f"{scenario_type.replace('_', ' ').title()} ({severity})",
        }
        self._state.active_scenario = scenario

        self._emit(ScenarioEvent.create(
            self._state.quarter, scenario["name"], severity, "generated"
        ))
        logger.info("Scenario: %s | Severity: %s", scenario["name"], severity)
        return scenario

    def _record_player_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and store the player's decision."""
        valid_actions = {
            "increase_insurance_reserves", "reduce_pharma_exposure",
            "buy_credit_protection", "do_nothing", "rebalance_portfolio",
        }
        action = decision.get("action", "do_nothing")
        if action not in valid_actions:
            logger.warning("Unknown player action '%s'; defaulting to do_nothing", action)
            action = "do_nothing"

        recorded = {**decision, "action": action, "quarter": self._state.quarter}
        self._state.player_decision = recorded
        self._state.decisions_made += 1
        return recorded

    def _ai_decision(self, scenario: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a simple rule-based AI response (full model in ai_opponent.py)."""
        action = "do_nothing"
        if scenario:
            severity = scenario.get("severity", "mild")
            stype    = scenario.get("type", "")
            if severity in ("severe", "catastrophic"):
                action = "increase_insurance_reserves" if "pandemic" in stype else "buy_credit_protection"
            elif severity == "moderate":
                action = "reduce_pharma_exposure" if "drug" in stype else "do_nothing"

        ai_dec = {"action": action, "source": "engine_fallback", "quarter": self._state.quarter}
        self._state.ai_decision = ai_dec
        self._state.decisions_made += 1
        return ai_dec

    def _apply_outcomes(
        self,
        scenario: Optional[Dict[str, Any]],
        player_dec: Dict[str, Any],
        ai_dec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply scenario and decision effects to portfolio value."""
        old_value = self._state.portfolio_value

        # Base quarterly drift: ~2% annual = 0.5% / quarter
        drift = self._rng.gauss(0.005, 0.015)

        # Scenario impact
        impact_pct = 0.0
        if scenario:
            severity_impacts = {
                "mild":         (-0.02, 0.01),
                "moderate":     (-0.05, -0.01),
                "severe":       (-0.12, -0.04),
                "catastrophic": (-0.25, -0.10),
            }
            lo, hi = severity_impacts.get(scenario["severity"], (-0.03, 0.0))
            impact_pct = self._rng.uniform(lo, hi)

            # Mitigation from decisions
            mitigation = 0.0
            good_responses = {
                "pandemic":          "increase_insurance_reserves",
                "drug_safety_crisis":"reduce_pharma_exposure",
                "regulatory_change": "buy_credit_protection",
                "hospital_merger":   "rebalance_portfolio",
            }
            best = good_responses.get(scenario.get("type", ""), "")
            if player_dec.get("action") == best:
                mitigation += 0.40   # 40% impact reduction
            if ai_dec.get("action") == best:
                mitigation += 0.20   # additional 20% from AI
            impact_pct *= (1 - min(mitigation, 0.70))

        total_change_pct = drift + impact_pct
        new_value = old_value * (1 + total_change_pct)
        new_value = max(new_value, 0.0)
        self._state.portfolio_value = new_value

        # Track survival
        if scenario and impact_pct > -0.10:
            self._state.scenarios_survived += 1

        self._emit(PortfolioEvent.create(
            self._state.quarter, old_value, new_value,
            scenario["name"] if scenario else "quarterly drift"
        ))

        outcome = {
            "drift_pct":   round(drift * 100, 3),
            "impact_pct":  round(impact_pct * 100, 3),
            "total_pct":   round(total_change_pct * 100, 3),
            "old_value":   old_value,
            "new_value":   new_value,
        }
        logger.info(
            "Portfolio: $%,.0f → $%,.0f  (%.2f%%)",
            old_value, new_value, total_change_pct * 100,
        )
        return outcome

    def _update_scores(
        self,
        scenario: Optional[Dict[str, Any]],
        outcome: Dict[str, Any],
        latency_s: float,
    ) -> tuple[float, Dict[str, float]]:
        """Compute incremental score for this quarter."""
        breakdown: Dict[str, float] = {}

        # Portfolio performance contribution (max 10 pts/quarter = 400 total)
        return_pts = max(0.0, outcome["total_pct"] * 2)
        return_pts = min(return_pts, 10.0)
        breakdown["portfolio_performance"] = round(return_pts, 2)

        # Risk management (max 7.5 pts/quarter = 300 total)
        survived = int(scenario is not None and outcome["impact_pct"] > -10.0)
        risk_pts = 5.0 * survived + (2.5 if self._state.portfolio_value > STARTING_VALUE * 0.80 else 0.0)
        breakdown["risk_management"] = round(risk_pts, 2)

        # Clinical intelligence (max 5 pts/quarter = 200 total)
        clinical_pts = 3.0 if scenario and self._state.player_decision and \
                       self._state.player_decision.get("action") != "do_nothing" else 0.0
        breakdown["clinical_intelligence"] = round(clinical_pts, 2)

        # Speed bonus (max 2.5 pts/quarter = 100 total)
        speed_pts = max(0.0, 2.5 - (latency_s / 10.0))
        breakdown["speed_bonus"] = round(speed_pts, 2)

        delta = sum(breakdown.values())
        old_score = self._state.score
        self._state.score = min(self._state.score + delta, 1000.0)

        self._emit(ScoringEvent.create(
            self._state.quarter, old_score, self._state.score, breakdown
        ))
        return delta, breakdown

    def _compute_grade(self) -> str:
        s = self._state.score
        if s >= 900: return "S"
        if s >= 800: return "A"
        if s >= 650: return "B"
        if s >= 500: return "C"
        if s >= 300: return "D"
        return "F"

    def _emit(self, event: BaseEvent) -> None:
        for cb in self._listeners.get(event.event_type, []):
            try:
                cb(event)
            except Exception as exc:
                logger.error("Event listener error: %s", exc)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    def on_portfolio(evt: BaseEvent) -> None:
        p = evt.payload
        print(f"  💰 Portfolio Q{p['quarter']}: ${p['new_value']:,.0f}  ({p['change']:+,.0f})")

    def on_scoring(evt: BaseEvent) -> None:
        p = evt.payload
        print(f"  🏆 Score Q{p['quarter']}: {p['new_score']:.1f}  (+{p['delta']:.2f})")

    engine = SimulationEngine(seed=42)
    engine.register_listener(EventType.PORTFOLIO, on_portfolio)
    engine.register_listener(EventType.SCORING, on_scoring)

    engine.start_game()
    actions = [
        "increase_insurance_reserves", "reduce_pharma_exposure",
        "buy_credit_protection", "do_nothing", "rebalance_portfolio",
    ]
    for _ in range(8):   # run 8 quarters as demo
        result = engine.next_quarter(
            player_decision={"action": random.choice(actions)},
            decision_latency_s=random.uniform(1, 30),
        )
        print(f"Q{result['quarter']:02d} | scenario={result['scenario']['name'] if result['scenario'] else 'none'!r}")
        if not engine.get_state().is_active:
            break

    summary = engine.end_game()
    print("\n=== FINAL SUMMARY ===")
    for k, v in summary.items():
        if k != "history":
            print(f"  {k}: {v}")
