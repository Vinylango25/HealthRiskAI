"""
HealthRisk Lab – Scoring Engine
==================================
Implements the 1000-point scoring framework for the HealthRisk Lab simulation.

Score pillars
-------------
1. Portfolio Performance  (max 400 pts)
   - Absolute return vs. $500M benchmark       (200 pts)
   - Sharpe ratio                               (100 pts)
   - Maximum drawdown penalty                  (100 pts)

2. Risk Management         (max 300 pts)
   - Scenario survival rate                    (120 pts)
   - Reserve adequacy                          (100 pts)
   - VaR compliance                             (80 pts)

3. Clinical Intelligence   (max 200 pts)
   - Early warning accuracy                    (100 pts)
   - NLP signal utilisation                    (100 pts)

4. Speed Bonus             (max 100 pts)
   - Decision speed                             (60 pts)
   - Decision efficiency                        (40 pts)

Total: 1000 pts
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score constants
# ---------------------------------------------------------------------------

MAX_SCORE           = 1000.0
STARTING_PORTFOLIO  = 500_000_000.0
TOTAL_QUARTERS      = 40

# Pillar maxima
MAX_PORTFOLIO_PERF  = 400.0
MAX_RISK_MGMT       = 300.0
MAX_CLINICAL_INTEL  = 200.0
MAX_SPEED_BONUS     = 100.0

# Sub-component maxima within pillars
MAX_ABS_RETURN      = 200.0
MAX_SHARPE          = 100.0
MAX_DRAWDOWN_COMP   = 100.0

MAX_SCENARIO_SURV   = 120.0
MAX_RESERVE_ADEQUACY= 100.0
MAX_VAR_COMPLIANCE  = 80.0

MAX_EARLY_WARNING   = 100.0
MAX_NLP_SIGNAL      = 100.0

MAX_DECISION_SPEED  = 60.0
MAX_EFFICIENCY      = 40.0


# ---------------------------------------------------------------------------
# Score breakdown dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    """
    Full 1000-point score decomposition for one game.

    Attributes
    ----------
    total_score         : Capped at MAX_SCORE.
    portfolio_performance : Pillar 1 total (0–400).
    risk_management     : Pillar 2 total (0–300).
    clinical_intelligence : Pillar 3 total (0–200).
    speed_bonus         : Pillar 4 total (0–100).
    details             : Per-component scores.
    computed_at         : ISO timestamp of computation.
    grade               : Letter grade S / A / B / C / D / F.
    """
    total_score:            float = 0.0
    portfolio_performance:  float = 0.0
    risk_management:        float = 0.0
    clinical_intelligence:  float = 0.0
    speed_bonus:            float = 0.0
    details:                Dict[str, float] = field(default_factory=dict)
    computed_at:            str = field(default_factory=lambda: datetime.utcnow().isoformat())
    grade:                  str = "F"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score":           round(self.total_score, 2),
            "grade":                 self.grade,
            "pillars": {
                "portfolio_performance": round(self.portfolio_performance, 2),
                "risk_management":       round(self.risk_management, 2),
                "clinical_intelligence": round(self.clinical_intelligence, 2),
                "speed_bonus":           round(self.speed_bonus, 2),
            },
            "details":     {k: round(v, 2) for k, v in self.details.items()},
            "computed_at": self.computed_at,
        }

    def __str__(self) -> str:
        return (
            f"Score: {self.total_score:.1f}/1000  Grade: {self.grade}  |  "
            f"Port={self.portfolio_performance:.1f}  Risk={self.risk_management:.1f}  "
            f"Clin={self.clinical_intelligence:.1f}  Speed={self.speed_bonus:.1f}"
        )


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

@dataclass
class LeaderboardEntry:
    rank:           int
    player_name:    str
    total_score:    float
    grade:          str
    final_portfolio: float
    quarters_played: int
    achieved_at:    str = field(default_factory=lambda: datetime.utcnow().isoformat())


class Leaderboard:
    """In-memory leaderboard with top-N tracking."""

    def __init__(self, capacity: int = 100) -> None:
        self._entries: List[LeaderboardEntry] = []
        self._capacity = capacity

    def submit(
        self, player_name: str, breakdown: ScoreBreakdown,
        final_portfolio: float, quarters_played: int
    ) -> LeaderboardEntry:
        entry = LeaderboardEntry(
            rank            = 0,
            player_name     = player_name,
            total_score     = breakdown.total_score,
            grade           = breakdown.grade,
            final_portfolio = final_portfolio,
            quarters_played = quarters_played,
        )
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.total_score, reverse=True)
        self._entries = self._entries[:self._capacity]
        for i, e in enumerate(self._entries, start=1):
            e.rank = i
        logger.info("Leaderboard updated | %s: %.1f pts (%s)", player_name, breakdown.total_score, breakdown.grade)
        return entry

    def top(self, n: int = 10) -> List[LeaderboardEntry]:
        return self._entries[:n]

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Scoring Engine
# ---------------------------------------------------------------------------

class ScoringEngine:
    """
    Computes the full 1000-point score from a completed (or in-progress)
    game state.

    Parameters
    ----------
    leaderboard : Optional shared Leaderboard instance.
    """

    def __init__(self, leaderboard: Optional[Leaderboard] = None) -> None:
        self._leaderboard = leaderboard or Leaderboard()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_score(self, game_state: Dict[str, Any]) -> ScoreBreakdown:
        """
        Compute a ScoreBreakdown from the provided game state dict.

        Expected keys in game_state
        ---------------------------
        portfolio_value       : Current mark-to-market value.
        history               : List of per-quarter snapshots.
        decisions_made        : Total decisions recorded.
        scenarios_survived    : Scenarios where loss < 10%.
        quarter               : Quarters completed.
        clinical_warnings_hit : (optional) Count of early warning triggers.
        total_warnings        : (optional) Total warning opportunities.
        nlp_signals_used      : (optional) Count of NLP signals acted upon.
        nlp_signals_available : (optional) Total NLP signals generated.
        decision_latencies_s  : (optional) List of per-decision latency seconds.
        var_breaches          : (optional) Number of VaR limit breaches.
        """
        t0 = time.perf_counter()

        pillar1, d1 = self._portfolio_performance(game_state)
        pillar2, d2 = self._risk_management(game_state)
        pillar3, d3 = self._clinical_intelligence(game_state)
        pillar4, d4 = self._speed_bonus(game_state)

        total = min(pillar1 + pillar2 + pillar3 + pillar4, MAX_SCORE)
        grade = self._grade(total)

        breakdown = ScoreBreakdown(
            total_score           = round(total, 2),
            portfolio_performance = round(pillar1, 2),
            risk_management       = round(pillar2, 2),
            clinical_intelligence = round(pillar3, 2),
            speed_bonus           = round(pillar4, 2),
            details               = {**d1, **d2, **d3, **d4},
            grade                 = grade,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Score computed in %.1f ms: %s", elapsed, breakdown)
        return breakdown

    def submit_to_leaderboard(
        self, player_name: str, game_state: Dict[str, Any]
    ) -> LeaderboardEntry:
        breakdown = self.calculate_score(game_state)
        return self._leaderboard.submit(
            player_name     = player_name,
            breakdown       = breakdown,
            final_portfolio = float(game_state.get("portfolio_value", STARTING_PORTFOLIO)),
            quarters_played = int(game_state.get("quarter", 0)),
        )

    @property
    def leaderboard(self) -> Leaderboard:
        return self._leaderboard

    # ------------------------------------------------------------------
    # Pillar 1: Portfolio Performance (400 pts)
    # ------------------------------------------------------------------

    def _portfolio_performance(
        self, gs: Dict[str, Any]
    ) -> tuple[float, Dict[str, float]]:
        portfolio_value = float(gs.get("portfolio_value", STARTING_PORTFOLIO))
        history         = gs.get("history", [])

        # 1a. Absolute return (0–200 pts)
        total_return = (portfolio_value - STARTING_PORTFOLIO) / STARTING_PORTFOLIO
        # Map [-30%, +30%] → [0, 200]
        abs_return_score = self._sigmoid_scale(total_return, centre=0.05, scale=0.15, max_pts=MAX_ABS_RETURN)

        # 1b. Sharpe ratio (0–100 pts)
        sharpe = self._compute_sharpe_from_history(history)
        # Map Sharpe [-1, 3] → [0, 100]
        sharpe_score = max(0.0, min(MAX_SHARPE, (sharpe + 1.0) / 4.0 * MAX_SHARPE))

        # 1c. Max drawdown (0–100 pts — penalty for deep drawdowns)
        max_dd = self._compute_max_drawdown(history, portfolio_value)
        # 0% drawdown → 100 pts; 40%+ drawdown → 0 pts
        dd_score = max(0.0, MAX_DRAWDOWN_COMP * (1.0 - max_dd / 0.40))

        total = abs_return_score + sharpe_score + dd_score
        details = {
            "abs_return_score": round(abs_return_score, 2),
            "sharpe_score":     round(sharpe_score, 2),
            "drawdown_score":   round(dd_score, 2),
            "total_return_pct": round(total_return * 100, 3),
            "sharpe_ratio":     round(sharpe, 4),
            "max_drawdown_pct": round(max_dd * 100, 3),
        }
        return min(total, MAX_PORTFOLIO_PERF), details

    # ------------------------------------------------------------------
    # Pillar 2: Risk Management (300 pts)
    # ------------------------------------------------------------------

    def _risk_management(
        self, gs: Dict[str, Any]
    ) -> tuple[float, Dict[str, float]]:
        scenarios_survived = int(gs.get("scenarios_survived", 0))
        quarter            = max(int(gs.get("quarter", 1)), 1)
        portfolio_value    = float(gs.get("portfolio_value", STARTING_PORTFOLIO))
        var_breaches       = int(gs.get("var_breaches", 0))

        # 2a. Scenario survival (0–120 pts)
        # Assume ~60% of quarters have a scenario
        expected_scenarios = max(1, int(quarter * 0.60))
        survival_rate = min(scenarios_survived / expected_scenarios, 1.0)
        survival_score = survival_rate * MAX_SCENARIO_SURV

        # 2b. Reserve adequacy (0–100 pts)
        # Portfolio > 80% of starting value → full marks; scales to 0 at 40%
        adequacy = max(0.0, (portfolio_value / STARTING_PORTFOLIO - 0.40) / 0.40)
        reserve_score = min(adequacy * MAX_RESERVE_ADEQUACY, MAX_RESERVE_ADEQUACY)

        # 2c. VaR compliance (0–80 pts)
        # Each breach deducts 10 pts; floor at 0
        var_score = max(0.0, MAX_VAR_COMPLIANCE - var_breaches * 10.0)

        total = survival_score + reserve_score + var_score
        details = {
            "survival_score":   round(survival_score, 2),
            "reserve_score":    round(reserve_score, 2),
            "var_score":        round(var_score, 2),
            "survival_rate":    round(survival_rate, 4),
            "var_breaches":     var_breaches,
        }
        return min(total, MAX_RISK_MGMT), details

    # ------------------------------------------------------------------
    # Pillar 3: Clinical Intelligence (200 pts)
    # ------------------------------------------------------------------

    def _clinical_intelligence(
        self, gs: Dict[str, Any]
    ) -> tuple[float, Dict[str, float]]:
        warnings_hit       = int(gs.get("clinical_warnings_hit", 0))
        total_warnings     = max(int(gs.get("total_warnings", 1)), 1)
        nlp_signals_used   = int(gs.get("nlp_signals_used", 0))
        nlp_available      = max(int(gs.get("nlp_signals_available", 1)), 1)

        # 3a. Early warning accuracy (0–100 pts)
        warn_accuracy = warnings_hit / total_warnings
        early_warn_score = warn_accuracy * MAX_EARLY_WARNING

        # 3b. NLP signal utilisation (0–100 pts)
        nlp_util = nlp_signals_used / nlp_available
        nlp_score = nlp_util * MAX_NLP_SIGNAL

        total = early_warn_score + nlp_score
        details = {
            "early_warning_score": round(early_warn_score, 2),
            "nlp_score":           round(nlp_score, 2),
            "warning_accuracy":    round(warn_accuracy, 4),
            "nlp_utilisation":     round(nlp_util, 4),
        }
        return min(total, MAX_CLINICAL_INTEL), details

    # ------------------------------------------------------------------
    # Pillar 4: Speed Bonus (100 pts)
    # ------------------------------------------------------------------

    def _speed_bonus(
        self, gs: Dict[str, Any]
    ) -> tuple[float, Dict[str, float]]:
        latencies: List[float] = gs.get("decision_latencies_s", [])
        decisions_made         = max(int(gs.get("decisions_made", 1)), 1)
        quarter                = max(int(gs.get("quarter", 1)), 1)

        # 4a. Decision speed (0–60 pts)
        if latencies:
            median_lat = statistics.median(latencies)
            # < 5s → 60 pts; 60s+ → 0 pts
            speed_score = max(0.0, MAX_DECISION_SPEED * (1.0 - median_lat / 60.0))
        else:
            speed_score = MAX_DECISION_SPEED * 0.5   # default 50% if no timing data

        # 4b. Efficiency: decisions per quarter (0–40 pts)
        # Optimal ≈ 2 decisions/quarter (player + AI)
        decisions_per_q = decisions_made / quarter
        efficiency = 1.0 - abs(decisions_per_q - 2.0) / 4.0
        efficiency_score = max(0.0, min(MAX_EFFICIENCY, efficiency * MAX_EFFICIENCY))

        total = speed_score + efficiency_score
        details = {
            "speed_score":      round(speed_score, 2),
            "efficiency_score": round(efficiency_score, 2),
            "median_latency_s": round(statistics.median(latencies) if latencies else 0.0, 2),
            "decisions_per_q":  round(decisions_per_q, 2),
        }
        return min(total, MAX_SPEED_BONUS), details

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sharpe_from_history(
        history: List[Dict[str, Any]],
        risk_free_quarterly: float = 0.0125,   # ~5% annual
    ) -> float:
        """Compute annualised Sharpe ratio from quarterly portfolio history."""
        if len(history) < 2:
            return 0.0
        values = [h.get("portfolio_value", STARTING_PORTFOLIO) for h in history]
        returns = [
            (values[i] - values[i - 1]) / values[i - 1]
            for i in range(1, len(values))
        ]
        if len(returns) < 2:
            return 0.0
        try:
            mean_r  = statistics.mean(returns)
            std_r   = statistics.stdev(returns)
            if std_r == 0:
                return 0.0
            sharpe_q = (mean_r - risk_free_quarterly) / std_r
            return sharpe_q * math.sqrt(4)   # annualise (4 quarters)
        except statistics.StatisticsError:
            return 0.0

    @staticmethod
    def _compute_max_drawdown(
        history: List[Dict[str, Any]], current_value: float
    ) -> float:
        """Maximum peak-to-trough drawdown over the history."""
        values = [h.get("portfolio_value", STARTING_PORTFOLIO) for h in history]
        values.append(current_value)
        if not values:
            return 0.0
        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def _sigmoid_scale(
        x: float, centre: float, scale: float, max_pts: float
    ) -> float:
        """Map x through a sigmoid centred at `centre` with given `scale`."""
        z = (x - centre) / scale
        sigmoid = 1.0 / (1.0 + math.exp(-z))
        return max(0.0, min(max_pts, sigmoid * max_pts))

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 900: return "S"
        if score >= 800: return "A"
        if score >= 650: return "B"
        if score >= 500: return "C"
        if score >= 300: return "D"
        return "F"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    import random

    rng = random.Random(42)
    # Build a fake 20-quarter history
    value = STARTING_PORTFOLIO
    history = []
    for q in range(1, 21):
        value *= (1 + rng.gauss(0.005, 0.015))
        history.append({
            "quarter":         q,
            "portfolio_value": value,
            "score":           q * 10.0,
            "decisions_made":  q * 2,
            "scenarios_survived": max(0, q - 2),
        })

    fake_state = {
        "quarter":              20,
        "portfolio_value":      value,
        "history":              history,
        "decisions_made":       40,
        "scenarios_survived":   10,
        "var_breaches":         1,
        "clinical_warnings_hit": 7,
        "total_warnings":       10,
        "nlp_signals_used":     8,
        "nlp_signals_available": 10,
        "decision_latencies_s": [rng.uniform(3, 45) for _ in range(20)],
    }

    engine  = ScoringEngine()
    bd      = engine.calculate_score(fake_state)
    print("\n=== Score Breakdown ===")
    for k, v in bd.to_dict().items():
        print(f"  {k}: {v}")

    # Leaderboard demo
    for name, portfolio_mult in [("Alice", 1.12), ("Bob", 0.95), ("Carol", 1.05)]:
        fake_state["portfolio_value"] = STARTING_PORTFOLIO * portfolio_mult
        engine.submit_to_leaderboard(name, fake_state)

    print("\n=== Leaderboard Top 5 ===")
    for entry in engine.leaderboard.top(5):
        print(f"  #{entry.rank}  {entry.player_name:<8}  {entry.total_score:>7.1f} pts  Grade: {entry.grade}")
