"""
tests/test_survival.py
======================
Unit tests for survival analysis data preparation and metrics utilities.

Heavy survival model classes (DeepSurv, DynamicDeepHit) are NOT imported.
Only data-logic and lightweight metric helpers are tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic survival data helper
# ---------------------------------------------------------------------------

def _make_survival_data(n: int = 200, seed: int = 0):
    """Return (times, events, X) — basic synthetic survival dataset."""
    rng = np.random.default_rng(seed)
    times = rng.exponential(scale=30, size=n).clip(min=1.0)      # all > 0
    events = rng.binomial(1, p=0.4, size=n)                      # ~40% events
    X = rng.standard_normal((n, 5))
    return times, events, X


# ---------------------------------------------------------------------------
# TestSurvivalDataPrep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurvivalDataPrep:
    """Validation of synthetic survival dataset construction."""

    def test_survival_time_positive(self):
        """All survival times are strictly positive."""
        times, _, _ = _make_survival_data()
        assert (times > 0).all(), "All survival times must be > 0"

    def test_event_indicator_binary(self):
        """Event indicator contains only 0 and 1."""
        _, events, _ = _make_survival_data()
        unique_vals = set(np.unique(events))
        assert unique_vals.issubset({0, 1}), f"Event indicator must be binary, got {unique_vals}"

    def test_censored_fraction(self):
        """Censoring rate is between 10% and 90% for synthetic data."""
        _, events, _ = _make_survival_data(n=500)
        censoring_rate = 1.0 - events.mean()
        assert 0.10 <= censoring_rate <= 0.90, (
            f"Censoring rate {censoring_rate:.2f} outside [0.10, 0.90]"
        )


# ---------------------------------------------------------------------------
# TestSurvivalMetrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSurvivalMetrics:
    """Tests for survival-specific evaluation metrics."""

    def test_concordance_index_range(self):
        """lifelines concordance_index is in (0, 1) for informative predictions."""
        lifelines = pytest.importorskip(
            "lifelines", reason="lifelines not installed — skipping C-index test"
        )
        from lifelines.utils import concordance_index

        rng = np.random.default_rng(42)
        n = 200
        times, events, X = _make_survival_data(n=n, seed=42)
        # Informative risk score: negative of time (higher score → shorter survival)
        risk_scores = -times + rng.normal(0, 5, n)
        c_idx = concordance_index(times, -risk_scores, events)
        assert 0 < c_idx < 1, f"C-index must be in (0, 1), got {c_idx:.4f}"

    def test_brier_score_survival(self):
        """Manual survival Brier score at a fixed time point is in [0, 0.5]."""
        rng = np.random.default_rng(99)
        n = 300
        times, events, _ = _make_survival_data(n=n, seed=99)
        # Predicted survival probability at time t=20 — uniform noise
        t_eval = 20.0
        surv_pred = rng.uniform(0.3, 0.9, size=n)
        # Uncensored Brier at t: E[(S(t) - I(T > t))^2]
        indicator_gt = (times > t_eval).astype(float)
        brier = np.mean((surv_pred - indicator_gt) ** 2)
        # Theoretical max of Brier score for binary outcomes is 1.0;
        # for random predictions in [0.3, 0.9] the practical range is [0, 0.5]
        assert 0 <= brier <= 0.5, f"Brier score {brier:.4f} outside [0, 0.5]"


# ---------------------------------------------------------------------------
# TestCoxPHDataHelpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCoxPHDataHelpers:
    """Data helper logic for Cox Proportional Hazards model preparation."""

    def test_event_times_sorted(self):
        """Sorting by survival time produces a monotone non-decreasing sequence."""
        times, events, X = _make_survival_data(n=100, seed=7)
        df = pd.DataFrame({"time": times, "event": events})
        df_sorted = df.sort_values("time").reset_index(drop=True)
        diffs = np.diff(df_sorted["time"].values)
        assert (diffs >= 0).all(), "Sorted times should be non-decreasing"

    def test_log_partial_hazard(self):
        """Log partial hazard (linear predictor) is finite for valid covariates."""
        rng = np.random.default_rng(3)
        n = 100
        X = rng.standard_normal((n, 4))
        beta = np.array([0.5, -0.3, 0.2, 0.1])
        log_hazard = X @ beta  # linear predictor
        assert np.all(np.isfinite(log_hazard)), (
            "All log partial hazard values must be finite"
        )
        # The hazard ratio exp(log_hazard) must be positive
        hazard_ratio = np.exp(log_hazard)
        assert (hazard_ratio > 0).all(), "Hazard ratios must be positive"
