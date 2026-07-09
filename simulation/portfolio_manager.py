"""
HealthRisk Lab – Portfolio Manager
=====================================
Manages the $500M simulation portfolio composed of four asset-class buckets.
Supports mark-to-market valuation, P&L tracking, scenario shock application,
VaR estimation, and Sharpe ratio calculation.

Portfolio composition (starting values)
---------------------------------------
insurance_book    $200M  – healthcare insurance liabilities & reserves
bond_portfolio    $150M  – investment-grade bonds (duration ~5y)
pharma_equities   $100M  – listed pharmaceutical equities
credit_facility    $50M  – revolving credit / CLO exposure
─────────────────────────
Total             $500M
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STARTING_TOTAL        = 500_000_000.0
STARTING_ALLOCATIONS: Dict[str, float] = {
    "insurance_book":  200_000_000.0,
    "bond_portfolio":  150_000_000.0,
    "pharma_equities": 100_000_000.0,
    "credit_facility":  50_000_000.0,
}

# Volatility (quarterly σ) and expected return per component
COMPONENT_PARAMS: Dict[str, Dict[str, float]] = {
    "insurance_book":  {"mu": 0.0050, "sigma": 0.0200, "beta": 0.30},
    "bond_portfolio":  {"mu": 0.0075, "sigma": 0.0120, "beta": 0.15},
    "pharma_equities": {"mu": 0.0150, "sigma": 0.0600, "beta": 1.20},
    "credit_facility": {"mu": 0.0100, "sigma": 0.0350, "beta": 0.60},
}

# Scenario impact multipliers by component (fraction of announced impact)
SCENARIO_SENSITIVITY: Dict[str, Dict[str, float]] = {
    "pandemic": {
        "insurance_book":  1.50,   # primary loss source
        "bond_portfolio":  0.40,
        "pharma_equities": 0.60,
        "credit_facility": 0.30,
    },
    "drug_safety_crisis": {
        "insurance_book":  0.40,
        "bond_portfolio":  0.20,
        "pharma_equities": 2.00,   # primary loss source
        "credit_facility": 0.50,
    },
    "regulatory_change": {
        "insurance_book":  0.80,
        "bond_portfolio":  0.60,
        "pharma_equities": 1.20,
        "credit_facility": 0.40,
    },
    "hospital_merger": {
        "insurance_book":  0.30,
        "bond_portfolio":  0.50,
        "pharma_equities": 0.40,
        "credit_facility": 1.80,   # primary loss source
    },
}

VAR_CONFIDENCE  = 0.95   # 95% 1-quarter VaR
VAR_LIMIT_PCT   = 0.08   # breach if realised loss > 8% of total portfolio
REBALANCE_COST  = 0.0005 # 5 bps transaction cost per rebalance


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    A single portfolio component position.

    Attributes
    ----------
    name         : Component identifier.
    market_value : Current mark-to-market value.
    book_value   : Original cost basis.
    unrealised_pnl : market_value - book_value.
    realised_pnl : Cumulative realised gains/losses.
    weight       : Fraction of total portfolio.
    returns      : Quarterly return history for risk calculations.
    """
    name:           str
    market_value:   float
    book_value:     float
    unrealised_pnl: float = 0.0
    realised_pnl:   float = 0.0
    weight:         float = 0.0
    returns:        List[float] = field(default_factory=list)

    @property
    def total_pnl(self) -> float:
        return self.unrealised_pnl + self.realised_pnl


@dataclass
class PortfolioState:
    """
    Full mark-to-market snapshot of the portfolio.

    Attributes
    ----------
    state_id         : Unique snapshot identifier.
    quarter          : Simulation quarter at which this state was recorded.
    positions        : Per-component positions.
    total_value      : Sum of all market values.
    total_book_value : Sum of all book values.
    total_pnl        : Aggregate P&L.
    var_95           : 1-quarter 95% Value-at-Risk (parametric).
    sharpe_ratio     : Rolling Sharpe ratio.
    max_drawdown     : Peak-to-trough drawdown from $500M.
    rebalance_count  : Number of rebalances performed.
    var_breaches     : Number of quarters where realised loss exceeded VAR_LIMIT_PCT.
    snapshot_time    : ISO timestamp.
    """
    state_id:         str   = field(default_factory=lambda: str(uuid.uuid4()))
    quarter:          int   = 0
    positions:        Dict[str, Position] = field(default_factory=dict)
    total_value:      float = STARTING_TOTAL
    total_book_value: float = STARTING_TOTAL
    total_pnl:        float = 0.0
    var_95:           float = 0.0
    sharpe_ratio:     float = 0.0
    max_drawdown:     float = 0.0
    rebalance_count:  int   = 0
    var_breaches:     int   = 0
    snapshot_time:    str   = field(default_factory=lambda: datetime.utcnow().isoformat())

    def allocation_weights(self) -> Dict[str, float]:
        if self.total_value <= 0:
            return {k: 0.0 for k in self.positions}
        return {k: p.market_value / self.total_value for k, p in self.positions.items()}

    def summary(self) -> Dict[str, Any]:
        return {
            "quarter":        self.quarter,
            "total_value":    round(self.total_value, 2),
            "total_pnl":      round(self.total_pnl, 2),
            "return_pct":     round((self.total_value - STARTING_TOTAL) / STARTING_TOTAL * 100, 3),
            "var_95":         round(self.var_95, 2),
            "sharpe_ratio":   round(self.sharpe_ratio, 4),
            "max_drawdown":   round(self.max_drawdown * 100, 3),
            "var_breaches":   self.var_breaches,
            "rebalances":     self.rebalance_count,
            "allocations":    {k: round(v * 100, 2) for k, v in self.allocation_weights().items()},
        }


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------

class PortfolioManager:
    """
    Manages the HealthRisk Lab $500M simulation portfolio.

    Parameters
    ----------
    seed : Optional RNG seed for reproducible returns.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng     = random.Random(seed)
        self._history: List[PortfolioState] = []
        self._state   = self._initialise_state()
        logger.info(
            "PortfolioManager initialised | total=$%,.0f | components=%d",
            self._state.total_value, len(self._state.positions),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> PortfolioState:
        return self._state

    @property
    def history(self) -> List[PortfolioState]:
        return list(self._history)

    def advance_quarter(self) -> PortfolioState:
        """
        Apply stochastic returns for one quarter (no scenario shock).
        Records a snapshot to history and returns it.
        """
        self._state.quarter += 1
        old_total = self._state.total_value

        for name, pos in self._state.positions.items():
            params = COMPONENT_PARAMS[name]
            r = self._rng.gauss(params["mu"], params["sigma"])
            old_mv = pos.market_value
            pos.market_value = old_mv * (1 + r)
            pos.unrealised_pnl = pos.market_value - pos.book_value
            pos.returns.append(r)

        self._refresh_aggregate_metrics(old_total)
        snapshot = self._take_snapshot()
        self._history.append(snapshot)
        logger.debug(
            "Q%d advance | total=$%,.0f | Δ=$%+,.0f",
            self._state.quarter, self._state.total_value,
            self._state.total_value - old_total,
        )
        return snapshot

    def apply_scenario_impact(
        self,
        scenario_type: str,
        aggregate_impact_pct: float,
    ) -> Dict[str, float]:
        """
        Apply a scenario shock to the portfolio.

        Parameters
        ----------
        scenario_type        : One of the 4 scenario type strings.
        aggregate_impact_pct : Expected portfolio-level impact (e.g. -0.08 = -8%).

        Returns
        -------
        Dict mapping component name → dollar impact.
        """
        sensitivity = SCENARIO_SENSITIVITY.get(scenario_type, {k: 1.0 for k in STARTING_ALLOCATIONS})
        total_weight = sum(sensitivity.values())

        component_impacts: Dict[str, float] = {}
        old_total = self._state.total_value

        for name, pos in self._state.positions.items():
            s = sensitivity.get(name, 1.0)
            # Weight the aggregate impact by relative sensitivity
            component_pct = aggregate_impact_pct * (s / total_weight) * len(self._state.positions)
            # Add noise per component
            component_pct += self._rng.gauss(0, abs(aggregate_impact_pct) * 0.10)
            dollar_impact  = pos.market_value * component_pct
            pos.market_value     = max(0.0, pos.market_value + dollar_impact)
            pos.unrealised_pnl   = pos.market_value - pos.book_value
            component_impacts[name] = round(dollar_impact, 2)

        self._refresh_aggregate_metrics(old_total)

        # Check VaR breach
        realised_loss_pct = (self._state.total_value - old_total) / old_total
        if realised_loss_pct < -VAR_LIMIT_PCT:
            self._state.var_breaches += 1
            logger.warning(
                "VaR breach in Q%d: loss=%.2f%% > limit=%.2f%%",
                self._state.quarter, abs(realised_loss_pct) * 100, VAR_LIMIT_PCT * 100,
            )

        logger.info(
            "Scenario impact [%s]: aggregate=%.2f%% | total=$%,.0f → $%,.0f",
            scenario_type, aggregate_impact_pct * 100, old_total, self._state.total_value,
        )
        return component_impacts

    def rebalance(
        self,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Rebalance the portfolio to target weights.

        Parameters
        ----------
        target_weights : Desired weight per component (must sum to ≤1).
                         Defaults to original 40/30/20/10 split.

        Returns
        -------
        Dict of component → dollar trade (positive = buy, negative = sell).
        """
        if target_weights is None:
            target_weights = {
                "insurance_book":  0.40,
                "bond_portfolio":  0.30,
                "pharma_equities": 0.20,
                "credit_facility": 0.10,
            }

        total = self._state.total_value
        trades: Dict[str, float] = {}
        rebalance_volume = 0.0

        for name, pos in self._state.positions.items():
            target_mv  = total * target_weights.get(name, 0.0)
            trade      = target_mv - pos.market_value
            rebalance_volume += abs(trade)
            pos.market_value  = target_mv
            pos.unrealised_pnl = pos.market_value - pos.book_value
            trades[name] = round(trade, 2)

        # Apply transaction costs
        cost = rebalance_volume * REBALANCE_COST
        self._state.total_value -= cost
        # Spread cost proportionally
        for pos in self._state.positions.values():
            pos.market_value *= (1 - REBALANCE_COST)

        self._state.rebalance_count += 1
        self._refresh_aggregate_metrics(total)

        logger.info(
            "Rebalance #%d | volume=$%,.0f | cost=$%,.0f",
            self._state.rebalance_count, rebalance_volume, cost,
        )
        return trades

    def compute_var(
        self,
        confidence: float = VAR_CONFIDENCE,
        horizon_quarters: int = 1,
    ) -> float:
        """
        Parametric Value-at-Risk at the requested confidence level.

        Uses a variance-covariance approach with per-component σ and
        a simplified correlation matrix (off-diagonal ρ = 0.30).

        Returns
        -------
        Dollar VaR (positive = potential loss).
        """
        positions = list(self._state.positions.values())
        names     = [p.name for p in positions]
        weights   = [p.market_value / self._state.total_value for p in positions]
        sigmas    = [COMPONENT_PARAMS[n]["sigma"] * math.sqrt(horizon_quarters) for n in names]

        # Correlation matrix: 1 on diagonal, 0.30 off-diagonal
        rho = 0.30
        n   = len(names)
        portfolio_variance = 0.0
        for i in range(n):
            for j in range(n):
                corr = 1.0 if i == j else rho
                portfolio_variance += weights[i] * weights[j] * sigmas[i] * sigmas[j] * corr

        portfolio_sigma = math.sqrt(portfolio_variance)

        # z-score for confidence level (standard normal)
        # 95% → 1.6449,  99% → 2.3263
        z = self._normal_ppf(confidence)
        var_pct = z * portfolio_sigma
        var_dollar = var_pct * self._state.total_value

        self._state.var_95 = var_dollar
        logger.debug(
            "VaR(%d%%) = $%,.0f  (σ_portfolio=%.4f, z=%.4f)",
            int(confidence * 100), var_dollar, portfolio_sigma, z,
        )
        return var_dollar

    def compute_sharpe(self, risk_free_quarterly: float = 0.0125) -> float:
        """
        Compute the rolling Sharpe ratio from quarterly total-portfolio returns.

        Returns
        -------
        Annualised Sharpe ratio (float).  Returns 0.0 if < 2 data points.
        """
        if len(self._history) < 2:
            return 0.0

        values  = [h.total_value for h in self._history]
        returns = [
            (values[i] - values[i - 1]) / values[i - 1]
            for i in range(1, len(values))
        ]
        if len(returns) < 2:
            return 0.0
        try:
            mean_r = statistics.mean(returns)
            std_r  = statistics.stdev(returns)
            if std_r == 0:
                return 0.0
            sharpe_q = (mean_r - risk_free_quarterly) / std_r
            annualised = sharpe_q * math.sqrt(4)
            self._state.sharpe_ratio = round(annualised, 4)
            return annualised
        except statistics.StatisticsError:
            return 0.0

    def get_pnl_report(self) -> Dict[str, Any]:
        """Return a full P&L breakdown by component."""
        report: Dict[str, Any] = {
            "quarter":           self._state.quarter,
            "total_value":       round(self._state.total_value, 2),
            "total_return_pct":  round(
                (self._state.total_value - STARTING_TOTAL) / STARTING_TOTAL * 100, 3),
            "components": {},
        }
        for name, pos in self._state.positions.items():
            report["components"][name] = {
                "market_value":   round(pos.market_value, 2),
                "book_value":     round(pos.book_value, 2),
                "unrealised_pnl": round(pos.unrealised_pnl, 2),
                "realised_pnl":   round(pos.realised_pnl, 2),
                "total_pnl":      round(pos.total_pnl, 2),
                "weight_pct":     round(pos.market_value / self._state.total_value * 100, 2),
            }
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialise_state(self) -> PortfolioState:
        positions = {
            name: Position(
                name          = name,
                market_value  = mv,
                book_value    = mv,
                weight        = mv / STARTING_TOTAL,
            )
            for name, mv in STARTING_ALLOCATIONS.items()
        }
        return PortfolioState(
            quarter          = 0,
            positions        = positions,
            total_value      = STARTING_TOTAL,
            total_book_value = STARTING_TOTAL,
        )

    def _refresh_aggregate_metrics(self, old_total: float) -> None:
        """Recompute aggregate stats after any value change."""
        total = sum(p.market_value for p in self._state.positions.values())
        self._state.total_value  = total
        self._state.total_pnl    = total - STARTING_TOTAL

        for pos in self._state.positions.values():
            pos.weight = pos.market_value / total if total > 0 else 0.0

        # Max drawdown update
        peak = max(
            (h.total_value for h in self._history),
            default=STARTING_TOTAL,
        )
        peak = max(peak, STARTING_TOTAL)
        self._state.max_drawdown = max(0.0, (peak - total) / peak)

        # Compute VaR
        self.compute_var()

        # Compute Sharpe (requires history)
        if len(self._history) >= 2:
            self.compute_sharpe()

    def _take_snapshot(self) -> PortfolioState:
        """Create a lightweight history snapshot."""
        snap_positions = {
            name: Position(
                name           = pos.name,
                market_value   = pos.market_value,
                book_value     = pos.book_value,
                unrealised_pnl = pos.unrealised_pnl,
                realised_pnl   = pos.realised_pnl,
                weight         = pos.weight,
                returns        = list(pos.returns),
            )
            for name, pos in self._state.positions.items()
        }
        return PortfolioState(
            quarter          = self._state.quarter,
            positions        = snap_positions,
            total_value      = self._state.total_value,
            total_book_value = self._state.total_book_value,
            total_pnl        = self._state.total_pnl,
            var_95           = self._state.var_95,
            sharpe_ratio     = self._state.sharpe_ratio,
            max_drawdown     = self._state.max_drawdown,
            rebalance_count  = self._state.rebalance_count,
            var_breaches     = self._state.var_breaches,
        )

    @staticmethod
    def _normal_ppf(p: float) -> float:
        """
        Rational approximation of the inverse normal CDF (Abramowitz & Stegun).
        Accurate to ~3 decimal places for 0.90 ≤ p ≤ 0.9999.
        """
        assert 0 < p < 1
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c = (2.515517, 0.802853, 0.010328)
        d = (1.432788, 0.189269, 0.001308)
        return t - (c[0] + c[1] * t + c[2] * t**2) / (1 + d[0] * t + d[1] * t**2 + d[2] * t**3)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    pm = PortfolioManager(seed=21)

    print("=== Initial State ===")
    print(f"  Total: ${pm.state.total_value:,.0f}")
    for name, pos in pm.state.positions.items():
        print(f"  {name:<20}: ${pos.market_value:>15,.0f}  ({pos.weight*100:.1f}%)")

    print("\n=== Running 8 quarters ===")
    for q in range(1, 9):
        snap = pm.advance_quarter()
        if q % 3 == 0:
            print(f"  Q{q:02d} — rebalancing …")
            pm.rebalance()

    print("\n=== Applying pandemic scenario (severe, -10% aggregate) ===")
    impacts = pm.apply_scenario_impact("pandemic", -0.10)
    for comp, impact in impacts.items():
        print(f"  {comp:<20}: ${impact:>+15,.0f}")

    print("\n=== P&L Report ===")
    report = pm.get_pnl_report()
    print(f"  Total value : ${report['total_value']:,.0f}")
    print(f"  Total return: {report['total_return_pct']:.2f}%")
    for comp, data in report["components"].items():
        print(f"  {comp:<20}: MV=${data['market_value']:>14,.0f}  PnL=${data['total_pnl']:>+13,.0f}")

    print("\n=== Risk Metrics ===")
    print(f"  95% VaR      : ${pm.compute_var():,.0f}")
    print(f"  Sharpe ratio : {pm.compute_sharpe():.4f}")
    print(f"  Max drawdown : {pm.state.max_drawdown*100:.2f}%")
    print(f"  VaR breaches : {pm.state.var_breaches}")
