"""
early_warning.py
================
Early Warning System (EWS) for hospital credit deterioration.

Philosophy
----------
Clinical quality signals (HCAHPS, readmission rates, staffing) tend to
deteriorate 6-12 months *before* financial metrics reflect distress.
This module detects those leading signals and issues tiered alerts.

Alert tiers
-----------
  WATCH              : Mild deterioration — monitor closely
  NEGATIVE_OUTLOOK   : Sustained negative trend — potential downgrade
  CREDITWATCH_NEG    : Severe / accelerating deterioration — imminent action

Signal groups
-------------
  Clinical quality  : hcahps_trend, readmission_change, mortality_change,
                      safety_grade_change
  Operational       : staffing_ratio_trend, volume_trend, er_wait_trend
  Financial leading : revenue_growth_trend, expense_ratio_change,
                      days_cash_change

Change-point detection uses a CUSUM (cumulative sum) algorithm on rolling
z-scores of each signal to identify structural breaks in time series.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Signals and their directionality:
# +1  = higher value is worse (readmission, mortality, expense ratio)
# -1  = lower value is worse  (HCAHPS, staffing, cash, volume)
SIGNAL_DIRECTIONS: Dict[str, int] = {
    "hcahps_score":          -1,
    "readmission_rate":      +1,
    "mortality_rate":        +1,
    "safety_grade_numeric":  +1,   # 1=A (best), 5=F (worst)
    "nurse_staffing_ratio":  -1,
    "inpatient_volume":      -1,
    "er_wait_minutes":       +1,
    "revenue_growth":        -1,
    "expense_ratio":         +1,
    "days_cash_on_hand":     -1,
}

ALL_SIGNALS = list(SIGNAL_DIRECTIONS.keys())

# CUSUM threshold: alert when cumulative sum exceeds k * sigma
CUSUM_K        = 0.5   # reference value (slack)
CUSUM_H_WATCH  = 3.0   # threshold for Watch
CUSUM_H_NEG    = 5.0   # threshold for Negative Outlook
CUSUM_H_CW     = 8.0   # threshold for CreditWatch Negative

# Minimum number of periods required for trend detection
MIN_PERIODS = 4


# ---------------------------------------------------------------------------
# Alert types
# ---------------------------------------------------------------------------

class AlertLevel(str, Enum):
    NONE               = "NONE"
    WATCH              = "WATCH"
    NEGATIVE_OUTLOOK   = "NEGATIVE_OUTLOOK"
    CREDITWATCH_NEG    = "CREDITWATCH_NEG"


@dataclass
class SignalAlert:
    """Alert for a single signal."""
    signal: str
    cusum_value: float
    alert_level: AlertLevel
    trend_direction: str        # "deteriorating" | "stable" | "improving"
    periods_in_trend: int
    description: str


@dataclass
class HospitalAlert:
    """Aggregated alert for a hospital entity."""
    hospital_id: str
    alert_level: AlertLevel
    composite_score: float          # 0-100, higher = more at risk
    signal_alerts: List[SignalAlert] = field(default_factory=list)
    triggered_signals: List[str]    = field(default_factory=list)
    generated_at: str               = field(default_factory=lambda: datetime.utcnow().isoformat())
    report_text: str                = ""


# ---------------------------------------------------------------------------
# CUSUM change-point detector
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """
    One-sided CUSUM for detecting upward shifts in a signal.

    The signal is first normalised to z-scores using a rolling baseline,
    then the cumulative sum tracks persistent deviations.

    Parameters
    ----------
    k : float
        Reference / allowance value (slack). Typical range 0.25-1.0.
    h_watch, h_neg, h_cw : float
        Thresholds for Watch / Negative Outlook / CreditWatch Negative.
    """

    def __init__(
        self,
        k: float = CUSUM_K,
        h_watch: float = CUSUM_H_WATCH,
        h_neg: float = CUSUM_H_NEG,
        h_cw: float = CUSUM_H_CW,
    ):
        self.k = k
        self.h_watch = h_watch
        self.h_neg   = h_neg
        self.h_cw    = h_cw

    def run(self, series: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run CUSUM on a 1-D series (already z-scored or normalised).

        Returns
        -------
        cusum_path : cumulative sum array
        final_value : last CUSUM value
        """
        cusum = np.zeros(len(series))
        for t in range(1, len(series)):
            cusum[t] = max(0.0, cusum[t - 1] + series[t] - self.k)
        return cusum, float(cusum[-1])

    def classify(self, cusum_val: float) -> AlertLevel:
        if cusum_val >= self.h_cw:
            return AlertLevel.CREDITWATCH_NEG
        elif cusum_val >= self.h_neg:
            return AlertLevel.NEGATIVE_OUTLOOK
        elif cusum_val >= self.h_watch:
            return AlertLevel.WATCH
        return AlertLevel.NONE


# ---------------------------------------------------------------------------
# Early Warning System
# ---------------------------------------------------------------------------

class EarlyWarningSystem:
    """
    Detects hospital credit deterioration from time-series signals.

    Parameters
    ----------
    window        : Rolling window for z-score normalisation (periods).
    cusum_k       : CUSUM slack parameter.
    anomaly_contamination : IsolationForest contamination for outlier baseline.

    Usage
    -----
    >>> ews = EarlyWarningSystem()
    >>> ews.fit(panel_df)                     # panel_df: (hospital_id, period, signals)
    >>> alerts = ews.detect_alerts(panel_df)  # returns List[HospitalAlert]
    >>> report = ews.generate_report(alerts)
    """

    def __init__(
        self,
        window: int = 4,
        cusum_k: float = CUSUM_K,
        anomaly_contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.window = window
        self.cusum_k = cusum_k
        self.anomaly_contamination = anomaly_contamination
        self.random_state = random_state
        self._detector = CUSUMDetector(k=cusum_k)
        self._scaler   = StandardScaler()
        self._iso_forest: Optional[IsolationForest] = None
        self._baseline_stats: Dict[str, Tuple[float, float]] = {}  # signal -> (mean, std)
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, panel: pd.DataFrame) -> "EarlyWarningSystem":
        """
        Learn baseline statistics from a historical panel.

        Parameters
        ----------
        panel : DataFrame with columns [hospital_id, period] + ALL_SIGNALS.
                'period' should be sortable (int, date, etc.).
        """
        logger.info(
            "Fitting EarlyWarningSystem on %d rows, %d hospitals …",
            len(panel),
            panel["hospital_id"].nunique(),
        )
        signal_cols = [c for c in ALL_SIGNALS if c in panel.columns]

        # Per-signal baseline: mean and std across full history
        for sig in signal_cols:
            vals = panel[sig].dropna().values
            self._baseline_stats[sig] = (float(vals.mean()), float(vals.std()) + 1e-9)

        # IsolationForest on cross-sectional snapshot for anomaly detection
        snap = panel.groupby("hospital_id")[signal_cols].last().dropna()
        if len(snap) >= 10:
            self._iso_forest = IsolationForest(
                contamination=self.anomaly_contamination,
                random_state=self.random_state,
            )
            self._iso_forest.fit(snap.values)
            logger.info("IsolationForest fitted on %d hospital snapshots.", len(snap))

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _zscore_series(self, signal: str, values: np.ndarray) -> np.ndarray:
        """Z-score a signal series using fitted baseline stats."""
        mean, std = self._baseline_stats.get(signal, (0.0, 1.0))
        return (values - mean) / std

    # ------------------------------------------------------------------
    def _analyse_signal(
        self, signal: str, values: np.ndarray
    ) -> SignalAlert:
        """Run CUSUM on a single signal time series for one hospital."""
        direction = SIGNAL_DIRECTIONS.get(signal, 1)
        z = self._zscore_series(signal, values) * direction  # flip so up = bad

        if len(z) < MIN_PERIODS:
            return SignalAlert(
                signal=signal, cusum_value=0.0,
                alert_level=AlertLevel.NONE,
                trend_direction="stable", periods_in_trend=0,
                description="Insufficient history.",
            )

        _, cusum_val = self._detector.run(z)
        level = self._detector.classify(cusum_val)

        # Trend direction from last 3 periods
        recent = z[-3:] if len(z) >= 3 else z
        if recent.mean() > 0.3:
            trend = "deteriorating"
        elif recent.mean() < -0.3:
            trend = "improving"
        else:
            trend = "stable"

        # Count consecutive deteriorating periods
        consec = 0
        for val in reversed(z):
            if val > 0:
                consec += 1
            else:
                break

        desc_map = {
            AlertLevel.NONE:             f"{signal} within normal range.",
            AlertLevel.WATCH:            f"{signal} showing mild deterioration trend.",
            AlertLevel.NEGATIVE_OUTLOOK: f"{signal} sustained negative trend (CUSUM={cusum_val:.2f}).",
            AlertLevel.CREDITWATCH_NEG:  f"{signal} severe deterioration — immediate review required.",
        }

        return SignalAlert(
            signal=signal,
            cusum_value=round(cusum_val, 3),
            alert_level=level,
            trend_direction=trend,
            periods_in_trend=consec,
            description=desc_map[level],
        )

    # ------------------------------------------------------------------
    def _composite_score(self, signal_alerts: List[SignalAlert]) -> float:
        """
        Aggregate signal alerts into a 0-100 composite risk score.
        CreditWatch = 25 pts, Negative Outlook = 10 pts, Watch = 4 pts.
        """
        weights = {
            AlertLevel.CREDITWATCH_NEG:  25.0,
            AlertLevel.NEGATIVE_OUTLOOK: 10.0,
            AlertLevel.WATCH:             4.0,
            AlertLevel.NONE:              0.0,
        }
        raw = sum(weights[a.alert_level] for a in signal_alerts)
        return float(min(raw, 100.0))

    # ------------------------------------------------------------------
    @staticmethod
    def _agg_alert_level(signal_alerts: List[SignalAlert]) -> AlertLevel:
        """Return the highest alert level across all signals."""
        order = [
            AlertLevel.CREDITWATCH_NEG,
            AlertLevel.NEGATIVE_OUTLOOK,
            AlertLevel.WATCH,
            AlertLevel.NONE,
        ]
        for level in order:
            if any(a.alert_level == level for a in signal_alerts):
                return level
        return AlertLevel.NONE

    # ------------------------------------------------------------------
    def detect_alerts(self, panel: pd.DataFrame) -> List[HospitalAlert]:
        """
        Detect alerts for all hospitals in the panel.

        Parameters
        ----------
        panel : Same schema as fit() — must include hospital_id, period columns.

        Returns
        -------
        List of HospitalAlert objects, one per hospital.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before detect_alerts().")

        signal_cols = [c for c in ALL_SIGNALS if c in panel.columns]
        results: List[HospitalAlert] = []

        for hosp_id, grp in panel.sort_values("period").groupby("hospital_id"):
            sig_alerts: List[SignalAlert] = []
            for sig in signal_cols:
                vals = grp[sig].values.astype(float)
                sig_alerts.append(self._analyse_signal(sig, vals))

            composite = self._composite_score(sig_alerts)
            agg_level  = self._agg_alert_level(sig_alerts)
            triggered  = [a.signal for a in sig_alerts if a.alert_level != AlertLevel.NONE]

            alert = HospitalAlert(
                hospital_id=str(hosp_id),
                alert_level=agg_level,
                composite_score=round(composite, 1),
                signal_alerts=sig_alerts,
                triggered_signals=triggered,
            )
            alert.report_text = self._build_report_text(alert)
            results.append(alert)

        n_alerts = sum(1 for a in results if a.alert_level != AlertLevel.NONE)
        logger.info(
            "detect_alerts: %d hospitals | %d with alerts (%d Watch, %d NegOutlook, %d CW-)",
            len(results),
            n_alerts,
            sum(1 for a in results if a.alert_level == AlertLevel.WATCH),
            sum(1 for a in results if a.alert_level == AlertLevel.NEGATIVE_OUTLOOK),
            sum(1 for a in results if a.alert_level == AlertLevel.CREDITWATCH_NEG),
        )
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _build_report_text(alert: HospitalAlert) -> str:
        lines = [
            f"Hospital: {alert.hospital_id}",
            f"Alert Level: {alert.alert_level.value}",
            f"Composite Risk Score: {alert.composite_score}/100",
            f"Generated: {alert.generated_at}",
            "",
            "Triggered Signals:",
        ]
        for sa in alert.signal_alerts:
            if sa.alert_level != AlertLevel.NONE:
                lines.append(
                    f"  [{sa.alert_level.value:20s}] {sa.signal:30s} "
                    f"CUSUM={sa.cusum_value:.2f}  Trend={sa.trend_direction}"
                )
        if not alert.triggered_signals:
            lines.append("  No signals triggered — within normal range.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def generate_report(self, alerts: List[HospitalAlert]) -> str:
        """
        Generate a summary report string across all hospital alerts.

        Parameters
        ----------
        alerts : Output of detect_alerts().

        Returns
        -------
        Formatted report string.
        """
        lines = [
            "=" * 70,
            "HOSPITAL CREDIT EARLY WARNING REPORT",
            f"Generated: {datetime.utcnow().isoformat()} UTC",
            "=" * 70,
            "",
            f"Total hospitals monitored : {len(alerts)}",
            f"CreditWatch Negative      : {sum(1 for a in alerts if a.alert_level == AlertLevel.CREDITWATCH_NEG)}",
            f"Negative Outlook          : {sum(1 for a in alerts if a.alert_level == AlertLevel.NEGATIVE_OUTLOOK)}",
            f"Watch                     : {sum(1 for a in alerts if a.alert_level == AlertLevel.WATCH)}",
            f"No Alert                  : {sum(1 for a in alerts if a.alert_level == AlertLevel.NONE)}",
            "",
            "-" * 70,
            "DETAIL (alerts only):",
            "-" * 70,
        ]
        for a in sorted(alerts, key=lambda x: -x.composite_score):
            if a.alert_level != AlertLevel.NONE:
                lines.append(a.report_text)
                lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic panel data generator
# ---------------------------------------------------------------------------

def _generate_synthetic_panel(
    n_hospitals: int = 30,
    n_periods: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic quarterly panel data for n_hospitals over n_periods.
    5 hospitals will have injected deterioration signals.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for hid in range(n_hospitals):
        deteriorating = hid < 5   # first 5 hospitals are distressed
        for t in range(n_periods):
            drift = t * 0.05 if deteriorating else 0.0
            rows.append({
                "hospital_id":        f"HOSP_{hid:03d}",
                "period":             t,
                "hcahps_score":       rng.normal(72 - drift * 3, 2),
                "readmission_rate":   rng.normal(0.16 + drift * 0.01, 0.005),
                "mortality_rate":     rng.normal(0.020 + drift * 0.002, 0.002),
                "safety_grade_numeric": float(np.clip(rng.normal(2.5 + drift * 0.3, 0.3), 1, 5)),
                "nurse_staffing_ratio": rng.normal(3.5 - drift * 0.1, 0.2),
                "inpatient_volume":   rng.normal(5000 - drift * 200, 100),
                "er_wait_minutes":    rng.normal(45 + drift * 5, 5),
                "revenue_growth":     rng.normal(0.03 - drift * 0.01, 0.01),
                "expense_ratio":      rng.normal(0.95 + drift * 0.02, 0.01),
                "days_cash_on_hand":  rng.normal(150 - drift * 5, 10),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("EarlyWarningSystem — Smoke Test")
    print("=" * 60)

    panel = _generate_synthetic_panel(n_hospitals=30, n_periods=12)
    print(f"Panel shape: {panel.shape} | Hospitals: {panel['hospital_id'].nunique()}")

    # Split: first 8 periods for training, last 4 for detection
    train_panel = panel[panel["period"] < 8]
    test_panel  = panel[panel["period"] >= 8]

    ews = EarlyWarningSystem(window=4, cusum_k=0.5)
    ews.fit(train_panel)

    alerts = ews.detect_alerts(test_panel)

    n_triggered = sum(1 for a in alerts if a.alert_level != AlertLevel.NONE)
    print(f"\nAlerts generated: {n_triggered} / {len(alerts)}")

    # Expect the 5 injected deteriorating hospitals to trigger
    triggered_ids = {a.hospital_id for a in alerts if a.alert_level != AlertLevel.NONE}
    expected_ids  = {f"HOSP_{i:03d}" for i in range(5)}
    overlap = len(triggered_ids & expected_ids)
    print(f"Deteriorating hospitals detected: {overlap}/5")

    report = ews.generate_report(alerts)
    print("\n" + report[:1500])

    assert n_triggered > 0, "No alerts generated — check signal logic."
    print("\n✓ EarlyWarningSystem smoke test passed.")
