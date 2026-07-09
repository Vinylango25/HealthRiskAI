"""
HealthRisk Lab – AI Opponent
==============================
Implements an AI agent that competes against (or alongside) the player in the
HealthRisk Lab simulation.  The AI reads risk signals from the current
GameState and active Scenario to produce portfolio allocation decisions.

Three strategy profiles are supported:

* **conservative** – prioritises capital preservation; reacts early and strongly.
* **balanced**     – balances return and risk; proportional responses.
* **aggressive**   – targets maximum return; tolerates higher drawdowns.

The AI is intentionally non-trivial so the player has a credible benchmark.
Signal weights, thresholds, and allocation rules are tunable at construction
time, making the opponent suitable for reinforcement-learning experiments.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations & constants
# ---------------------------------------------------------------------------

class StrategyLevel(Enum):
    CONSERVATIVE = "conservative"
    BALANCED     = "balanced"
    AGGRESSIVE   = "aggressive"


class AIAction(Enum):
    INCREASE_INSURANCE_RESERVES = "increase_insurance_reserves"
    REDUCE_PHARMA_EXPOSURE      = "reduce_pharma_exposure"
    BUY_CREDIT_PROTECTION       = "buy_credit_protection"
    REBALANCE_PORTFOLIO         = "rebalance_portfolio"
    DO_NOTHING                  = "do_nothing"


# Severity-to-numeric mapping shared across the module
_SEVERITY_SCORE: Dict[str, float] = {
    "mild": 0.25, "moderate": 0.50, "severe": 0.75, "catastrophic": 1.00
}

# Default allocation deltas (fraction of total portfolio) per action
_BASE_ALLOCATION_CHANGES: Dict[AIAction, Dict[str, float]] = {
    AIAction.INCREASE_INSURANCE_RESERVES: {
        "insurance_book":   +0.05,
        "bond_portfolio":   -0.03,
        "pharma_equities":  -0.01,
        "credit_facility":  -0.01,
    },
    AIAction.REDUCE_PHARMA_EXPOSURE: {
        "insurance_book":   +0.02,
        "bond_portfolio":   +0.03,
        "pharma_equities":  -0.06,
        "credit_facility":  +0.01,
    },
    AIAction.BUY_CREDIT_PROTECTION: {
        "insurance_book":   +0.01,
        "bond_portfolio":   +0.04,
        "pharma_equities":  -0.02,
        "credit_facility":  -0.03,
    },
    AIAction.REBALANCE_PORTFOLIO: {
        "insurance_book":   +0.00,
        "bond_portfolio":   +0.00,
        "pharma_equities":  +0.00,
        "credit_facility":  +0.00,
    },
    AIAction.DO_NOTHING: {
        "insurance_book":   +0.00,
        "bond_portfolio":   +0.00,
        "pharma_equities":  +0.00,
        "credit_facility":  +0.00,
    },
}


# ---------------------------------------------------------------------------
# Risk signal computation
# ---------------------------------------------------------------------------

@dataclass
class RiskSignal:
    """
    Aggregated risk signal derived from game state + scenario.

    Attributes
    ----------
    overall_score    : 0–1 composite risk score.
    pandemic_score   : 0–1 pandemic component.
    drug_score       : 0–1 drug-safety component.
    regulatory_score : 0–1 regulatory component.
    credit_score     : 0–1 credit / merger component.
    drawdown         : Current portfolio drawdown from $500M peak.
    signal_age       : Quarters since the signal was generated.
    """
    overall_score:    float = 0.0
    pandemic_score:   float = 0.0
    drug_score:       float = 0.0
    regulatory_score: float = 0.0
    credit_score:     float = 0.0
    drawdown:         float = 0.0
    signal_age:       int   = 0

    def dominant_component(self) -> str:
        components = {
            "pandemic":    self.pandemic_score,
            "drug":        self.drug_score,
            "regulatory":  self.regulatory_score,
            "credit":      self.credit_score,
        }
        return max(components, key=components.get)


def compute_risk_signal(
    game_state_dict: Dict[str, Any],
    scenario: Optional[Dict[str, Any]],
) -> RiskSignal:
    """
    Derive a RiskSignal from the current game state and active scenario.

    Parameters
    ----------
    game_state_dict : Plain dict representation of GameState.
    scenario        : Active scenario dict (may be None).
    """
    portfolio_value = float(game_state_dict.get("portfolio_value", 500_000_000))
    starting_value  = 500_000_000.0
    drawdown = max(0.0, (starting_value - portfolio_value) / starting_value)

    pandemic_score   = 0.0
    drug_score       = 0.0
    regulatory_score = 0.0
    credit_score     = 0.0

    if scenario:
        stype    = scenario.get("type", "")
        severity = scenario.get("severity", "mild")
        sev_val  = _SEVERITY_SCORE.get(severity, 0.25)

        if stype == "pandemic":
            pandemic_score = sev_val
        elif stype == "drug_safety_crisis":
            drug_score = sev_val
        elif stype == "regulatory_change":
            regulatory_score = sev_val
        elif stype == "hospital_merger":
            credit_score = sev_val

    # Drawdown contributes universally
    drawdown_signal = min(drawdown * 2.0, 1.0)   # 50% drawdown → score 1.0

    overall = max(pandemic_score, drug_score, regulatory_score, credit_score, drawdown_signal)

    return RiskSignal(
        overall_score    = round(overall, 4),
        pandemic_score   = round(pandemic_score, 4),
        drug_score       = round(drug_score, 4),
        regulatory_score = round(regulatory_score, 4),
        credit_score     = round(credit_score, 4),
        drawdown         = round(drawdown, 4),
        signal_age       = 0,
    )


# ---------------------------------------------------------------------------
# AI Decision dataclass
# ---------------------------------------------------------------------------

@dataclass
class AIDecision:
    """
    Output of the AI decision engine for a single quarter.

    Attributes
    ----------
    decision_id       : Unique identifier.
    action            : Chosen AIAction.
    allocation_changes: Dict of component → fractional change.
    reasoning         : Human-readable explanation of the decision.
    confidence        : 0–1 confidence score.
    risk_signal       : The RiskSignal that drove this decision.
    strategy          : StrategyLevel used.
    latency_ms        : Wall-clock time to produce the decision.
    """
    decision_id:        str             = field(default_factory=lambda: str(uuid.uuid4()))
    action:             AIAction        = AIAction.DO_NOTHING
    allocation_changes: Dict[str, float] = field(default_factory=dict)
    reasoning:          str             = ""
    confidence:         float           = 0.5
    risk_signal:        Optional[RiskSignal] = None
    strategy:           StrategyLevel   = StrategyLevel.BALANCED
    latency_ms:         float           = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id":        self.decision_id,
            "action":             self.action.value,
            "allocation_changes": self.allocation_changes,
            "reasoning":          self.reasoning,
            "confidence":         self.confidence,
            "strategy":           self.strategy.value,
            "latency_ms":         self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Strategy configurations
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """
    Tunable parameters for each AI strategy.

    Attributes
    ----------
    reaction_threshold  : overall_score above which the AI acts (not do_nothing).
    severity_multiplier : scales allocation_changes.
    noise_scale         : random noise added to signal scores (simulates uncertainty).
    exploration_rate    : probability of taking a random action (ε-greedy).
    """
    reaction_threshold:  float = 0.40
    severity_multiplier: float = 1.0
    noise_scale:         float = 0.05
    exploration_rate:    float = 0.10


_STRATEGY_CONFIGS: Dict[StrategyLevel, StrategyConfig] = {
    StrategyLevel.CONSERVATIVE: StrategyConfig(
        reaction_threshold  = 0.20,
        severity_multiplier = 1.40,
        noise_scale         = 0.02,
        exploration_rate    = 0.05,
    ),
    StrategyLevel.BALANCED: StrategyConfig(
        reaction_threshold  = 0.40,
        severity_multiplier = 1.00,
        noise_scale         = 0.05,
        exploration_rate    = 0.10,
    ),
    StrategyLevel.AGGRESSIVE: StrategyConfig(
        reaction_threshold  = 0.65,
        severity_multiplier = 0.60,
        noise_scale         = 0.10,
        exploration_rate    = 0.20,
    ),
}


# ---------------------------------------------------------------------------
# AI Opponent
# ---------------------------------------------------------------------------

class AIOpponent:
    """
    AI agent for the HealthRisk Lab simulation.

    Parameters
    ----------
    strategy : Strategy profile (conservative / balanced / aggressive).
    seed     : Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        strategy: StrategyLevel = StrategyLevel.BALANCED,
        seed: Optional[int] = None,
    ) -> None:
        self._strategy   = strategy
        self._config     = _STRATEGY_CONFIGS[strategy]
        self._rng        = random.Random(seed)
        self._history:   List[AIDecision] = []
        logger.info("AIOpponent initialised | strategy=%s", strategy.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> StrategyLevel:
        return self._strategy

    @property
    def decision_history(self) -> List[AIDecision]:
        return list(self._history)

    def decide(
        self,
        game_state: Dict[str, Any],
        scenario:   Optional[Dict[str, Any]] = None,
    ) -> AIDecision:
        """
        Produce an AIDecision for the current quarter.

        Parameters
        ----------
        game_state : Plain dict representation of GameState (or GameState.snapshot()).
        scenario   : Active scenario dict, or None.

        Returns
        -------
        AIDecision with action, allocation_changes, and reasoning.
        """
        t_start = time.perf_counter()

        signal = compute_risk_signal(game_state, scenario)

        # Add noise to scores
        noisy_score = self._add_noise(signal.overall_score)

        # Determine action
        action, confidence, reasoning = self._select_action(signal, noisy_score, scenario)

        # Scale allocation changes by strategy multiplier
        raw_changes = _BASE_ALLOCATION_CHANGES[action]
        scaled_changes = {
            k: round(v * self._config.severity_multiplier, 4)
            for k, v in raw_changes.items()
        }

        latency_ms = (time.perf_counter() - t_start) * 1000

        decision = AIDecision(
            action             = action,
            allocation_changes = scaled_changes,
            reasoning          = reasoning,
            confidence         = round(confidence, 3),
            risk_signal        = signal,
            strategy           = self._strategy,
            latency_ms         = round(latency_ms, 2),
        )

        self._history.append(decision)
        logger.info(
            "AI decision [%s]: %s  (conf=%.2f, signal=%.2f)",
            self._strategy.value, action.value, confidence, signal.overall_score,
        )
        return decision

    def reset(self) -> None:
        """Clear decision history."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_noise(self, score: float) -> float:
        noise = self._rng.gauss(0, self._config.noise_scale)
        return max(0.0, min(1.0, score + noise))

    def _select_action(
        self,
        signal:      RiskSignal,
        noisy_score: float,
        scenario:    Optional[Dict[str, Any]],
    ) -> Tuple[AIAction, float, str]:
        """
        Rule-based action selection with ε-greedy exploration.

        Returns
        -------
        (action, confidence, reasoning)
        """
        # Exploration: random action
        if self._rng.random() < self._config.exploration_rate:
            action     = self._rng.choice(list(AIAction))
            confidence = 0.30
            reasoning  = (
                f"[{self._strategy.value}] Exploratory random action "
                f"(ε={self._config.exploration_rate:.0%})."
            )
            return action, confidence, reasoning

        # Below threshold → do nothing
        if noisy_score < self._config.reaction_threshold:
            return (
                AIAction.DO_NOTHING,
                1.0 - noisy_score,
                f"[{self._strategy.value}] Risk score {noisy_score:.2f} below "
                f"threshold {self._config.reaction_threshold:.2f}; maintaining current allocation.",
            )

        # Map dominant risk component to optimal action
        dominant = signal.dominant_component()
        action_map = {
            "pandemic":   AIAction.INCREASE_INSURANCE_RESERVES,
            "drug":       AIAction.REDUCE_PHARMA_EXPOSURE,
            "regulatory": AIAction.BUY_CREDIT_PROTECTION,
            "credit":     AIAction.BUY_CREDIT_PROTECTION,
        }
        action = action_map.get(dominant, AIAction.REBALANCE_PORTFOLIO)

        # Drawdown override: if portfolio down >15%, rebalance regardless
        if signal.drawdown > 0.15:
            action = AIAction.REBALANCE_PORTFOLIO

        # Compute confidence based on signal clarity
        confidence = 0.50 + 0.50 * noisy_score

        # Build reasoning text
        scenario_label = scenario["name"] if scenario and "name" in scenario else "background risk"
        reasoning = (
            f"[{self._strategy.value}] Responding to '{scenario_label}'. "
            f"Dominant risk: {dominant} (score={noisy_score:.2f}). "
            f"Action: {action.value}. "
            f"Drawdown={signal.drawdown*100:.1f}%."
        )

        return action, confidence, reasoning

    def performance_summary(self) -> Dict[str, Any]:
        """Summarise AI decision history statistics."""
        if not self._history:
            return {"total_decisions": 0}
        action_counts: Dict[str, int] = {}
        for d in self._history:
            action_counts[d.action.value] = action_counts.get(d.action.value, 0) + 1
        avg_conf = sum(d.confidence for d in self._history) / len(self._history)
        avg_latency = sum(d.latency_ms for d in self._history) / len(self._history)
        return {
            "total_decisions": len(self._history),
            "action_distribution": action_counts,
            "avg_confidence":  round(avg_conf, 3),
            "avg_latency_ms":  round(avg_latency, 3),
            "strategy":        self._strategy.value,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    fake_state = {
        "quarter":         5,
        "portfolio_value": 460_000_000,   # ~8% drawdown
        "score":           180.0,
    }

    scenarios_to_test = [
        None,
        {"type": "pandemic",         "name": "National Epidemic (severe)",    "severity": "severe"},
        {"type": "drug_safety_crisis","name": "Voluntary Recall (severe)",     "severity": "severe"},
        {"type": "regulatory_change", "name": "Drug Pricing Legislation",      "severity": "severe"},
        {"type": "hospital_merger",   "name": "System Bankruptcy (catastrophic)", "severity": "catastrophic"},
    ]

    for strat in StrategyLevel:
        print(f"\n=== Strategy: {strat.value.upper()} ===")
        ai = AIOpponent(strategy=strat, seed=99)
        for sc in scenarios_to_test:
            decision = ai.decide(fake_state, sc)
            print(f"  scenario={sc['name'] if sc else 'none'!r:<45}  "
                  f"action={decision.action.value:<35}  conf={decision.confidence:.2f}")

    print("\n=== Performance summary (last strategy) ===")
    for k, v in ai.performance_summary().items():
        print(f"  {k}: {v}")
