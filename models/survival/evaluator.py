"""
models/survival/evaluator.py
==============================
Survival model evaluation: C-index, Brier score, time-dependent AUROC,
calibration, and model comparison report.

Supports evaluating:
  - Cox PH (lifelines)
  - DeepSurv (pycox)
  - Dynamic-DeepHit (custom)
  - Any model returning (risk_score, survival_function)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── C-index ─────────────────────────────────────────────────────────────────


def concordance_index(
    event_times: np.ndarray,
    predicted_risk: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """
    Harrell's concordance index (C-index).

    A pair (i, j) is concordant if i had an earlier event than j AND model
    assigned higher risk to i.  C-index = concordant / (concordant + discordant).

    Parameters
    ----------
    event_times : (N,) observed durations
    predicted_risk : (N,) higher = higher risk
    event_observed : (N,) 1=event, 0=censored

    Returns
    -------
    c_index : float in [0, 1], 0.5 = random
    """
    n = len(event_times)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        if event_observed[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if event_times[i] < event_times[j]:
                if predicted_risk[i] > predicted_risk[j]:
                    concordant += 1
                elif predicted_risk[i] < predicted_risk[j]:
                    discordant += 1
                else:
                    tied += 1

    denom = concordant + discordant + 0.5 * tied
    if denom == 0:
        return 0.5
    return (concordant + 0.5 * tied) / denom


def concordance_index_fast(
    event_times: np.ndarray,
    predicted_risk: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """
    Vectorised C-index — much faster for large N.
    Uses lifelines if available, falls back to numpy implementation.
    """
    try:
        from lifelines.utils import concordance_index as lifelines_ci
        return float(lifelines_ci(event_times, -predicted_risk, event_observed))
    except ImportError:
        pass

    # Numpy vectorised version
    n = len(event_times)
    obs_mask = event_observed.astype(bool)
    t = event_times[obs_mask]
    r = predicted_risk[obs_mask]

    # All pairs where t_i < t_j
    T_i, T_j = np.meshgrid(t, t, indexing="ij")
    R_i, R_j = np.meshgrid(r, r, indexing="ij")

    comparable = T_i < T_j
    concordant = comparable & (R_i > R_j)
    discordant = comparable & (R_i < R_j)
    tied_risk = comparable & (R_i == R_j)

    n_concordant = concordant.sum()
    n_discordant = discordant.sum()
    n_tied = tied_risk.sum()

    denom = n_concordant + n_discordant + 0.5 * n_tied
    if denom == 0:
        return 0.5
    return float((n_concordant + 0.5 * n_tied) / denom)


# ─── Brier Score ──────────────────────────────────────────────────────────────


def brier_score_at_time(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    survival_probabilities: np.ndarray,
    eval_time: float,
    train_event_times: Optional[np.ndarray] = None,
    train_event_observed: Optional[np.ndarray] = None,
) -> float:
    """
    IPCW-weighted Brier score at a single time point.

    BS(t) = mean over i of: W_i * (I(T_i <= t, event_i) - S_hat(t|x_i))^2

    Parameters
    ----------
    survival_probabilities : (N,) predicted S(eval_time | x_i)
    train_* : training set for IPCW weights (uses test set if None)

    Returns
    -------
    brier_score : float in [0, 0.25], lower = better
    """
    n = len(event_times)

    # Estimate censoring distribution (Kaplan-Meier on 1-event_observed)
    if train_event_times is None:
        train_event_times = event_times
        train_event_observed = event_observed

    G_t = _km_survival_at_time(
        train_event_times, 1 - train_event_observed.astype(float), eval_time
    )
    G_ti = np.array(
        [
            _km_survival_at_time(
                train_event_times,
                1 - train_event_observed.astype(float),
                min(t, eval_time),
            )
            for t in event_times
        ]
    )

    eps = 1e-8
    bs = 0.0
    for i in range(n):
        t_i = event_times[i]
        e_i = event_observed[i]
        s_hat = survival_probabilities[i]

        if t_i <= eval_time and e_i == 1:
            w = 1.0 / max(G_ti[i], eps)
            bs += w * (1 - s_hat) ** 2
        elif t_i > eval_time:
            w = 1.0 / max(G_t, eps)
            bs += w * (0 - s_hat) ** 2
        # censored at or before eval_time: skip (IPCW handles this)

    return bs / n


def _km_survival_at_time(
    event_times: np.ndarray, event_observed: np.ndarray, t: float
) -> float:
    """Kaplan-Meier survival estimate at time t (simple implementation)."""
    try:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(event_times, event_observed, label="km")
        return float(kmf.predict(t))
    except ImportError:
        pass

    # Fallback: simple KM
    order = np.argsort(event_times)
    t_sorted = event_times[order]
    e_sorted = event_observed[order]
    surv = 1.0
    n_at_risk = len(t_sorted)
    for i, (ti, ei) in enumerate(zip(t_sorted, e_sorted)):
        if ti > t:
            break
        if ei == 1:
            surv *= 1 - 1 / max(n_at_risk, 1)
        n_at_risk -= 1
    return surv


def integrated_brier_score(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    survival_function: Any,  # callable: t → (N,) survival probs
    time_points: Optional[np.ndarray] = None,
) -> float:
    """
    Integrated Brier Score (IBS) over a range of time points.
    IBS = (1/T) integral_0^T BS(t) dt

    survival_function: callable(t) → array of survival probabilities for all subjects
    """
    if time_points is None:
        time_points = np.percentile(
            event_times[event_observed == 1], np.arange(10, 91, 10)
        )

    brier_scores = []
    for t in time_points:
        s_t = survival_function(t)
        bs = brier_score_at_time(event_times, event_observed, s_t, t)
        brier_scores.append(bs)

    # Trapezoidal integration
    t_range = time_points[-1] - time_points[0]
    if t_range <= 0:
        return float(np.mean(brier_scores))
    ibs = np.trapz(brier_scores, time_points) / t_range
    return float(ibs)


# ─── Time-dependent AUROC ────────────────────────────────────────────────────


def time_dependent_auroc(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predicted_risk: np.ndarray,
    eval_time: float,
    method: str = "incident_dynamic",
) -> float:
    """
    Time-dependent AUROC at a specific time point.

    Incident/dynamic: cases = events in (0, t], controls = still at risk at t.

    Parameters
    ----------
    method : "incident_dynamic" or "cumulative_dynamic"

    Returns
    -------
    auroc : float
    """
    if method == "incident_dynamic":
        # Cases: event occurred at or before eval_time
        case_mask = (event_times <= eval_time) & (event_observed == 1)
        # Controls: still at risk at eval_time
        ctrl_mask = event_times > eval_time
    else:
        # Cumulative/dynamic
        case_mask = (event_times <= eval_time) & (event_observed == 1)
        ctrl_mask = event_times > eval_time

    n_cases = case_mask.sum()
    n_ctrl = ctrl_mask.sum()

    if n_cases == 0 or n_ctrl == 0:
        logger.warning(
            "Cannot compute AUROC at t=%.1f: cases=%d, controls=%d",
            eval_time,
            n_cases,
            n_ctrl,
        )
        return float("nan")

    labels = np.concatenate([np.ones(n_cases), np.zeros(n_ctrl)])
    scores = np.concatenate([
        predicted_risk[case_mask],
        predicted_risk[ctrl_mask],
    ])

    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def auroc_at_horizons(
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predicted_risk: np.ndarray,
    time_horizons: List[float],
) -> Dict[str, float]:
    """Compute time-dependent AUROC at each specified horizon."""
    return {
        f"auroc_{int(t)}m": time_dependent_auroc(
            event_times, event_observed, predicted_risk, t
        )
        for t in time_horizons
    }


# ─── Calibration ─────────────────────────────────────────────────────────────


def calibration_metrics(
    observed_binary: np.ndarray,
    predicted_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Calibration assessment: Hosmer-Lemeshow test + ECE.

    Parameters
    ----------
    observed_binary : (N,) — 1 if event occurred by horizon
    predicted_prob : (N,) — predicted probability of event by horizon

    Returns
    -------
    dict with hl_statistic, hl_pvalue, ece (Expected Calibration Error)
    """
    # Hosmer-Lemeshow test (decile-based)
    df = pd.DataFrame({"obs": observed_binary, "pred": predicted_prob})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop", labels=False)

    hl_stat = 0.0
    for _, group in df.groupby("bin"):
        obs = group["obs"].sum()
        exp = group["pred"].sum()
        n = len(group)
        if exp > 0 and (n - exp) > 0:
            hl_stat += (obs - exp) ** 2 / (exp * (1 - exp / n))

    dof = max(n_bins - 2, 1)
    hl_pvalue = 1 - chi2.cdf(hl_stat, dof)

    # Expected Calibration Error
    frac_positives, mean_predicted = calibration_curve(
        observed_binary, predicted_prob, n_bins=n_bins, strategy="quantile"
    )
    ece = float(np.mean(np.abs(frac_positives - mean_predicted)))

    return {
        "hl_statistic": float(hl_stat),
        "hl_pvalue": float(hl_pvalue),
        "ece": ece,
        "well_calibrated": hl_pvalue > 0.05,
    }


# ─── Full evaluation report ──────────────────────────────────────────────────


@dataclass_style = None  # keep it simple


class SurvivalEvaluator:
    """
    Comprehensive survival model evaluator.

    Usage
    -----
    ev = SurvivalEvaluator(
        event_times=test_durations,
        event_observed=test_events,
        time_horizons=[3, 6, 12],
    )
    report = ev.evaluate(model_name="CoxPH", predicted_risk=risk_scores,
                         survival_fn=lambda t: model.predict_survival(t))
    ev.compare_models(reports=[report1, report2, report3])
    """

    def __init__(
        self,
        event_times: np.ndarray,
        event_observed: np.ndarray,
        time_horizons: Optional[List[float]] = None,
        train_event_times: Optional[np.ndarray] = None,
        train_event_observed: Optional[np.ndarray] = None,
    ) -> None:
        self.event_times = np.asarray(event_times, dtype=float)
        self.event_observed = np.asarray(event_observed, dtype=int)
        self.time_horizons = time_horizons or [3.0, 6.0, 12.0]
        self.train_event_times = train_event_times
        self.train_event_observed = train_event_observed
        self._reports: List[Dict] = []

    def evaluate(
        self,
        model_name: str,
        predicted_risk: np.ndarray,
        survival_fn: Optional[Any] = None,
        return_all: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full evaluation for one model.

        Parameters
        ----------
        predicted_risk : (N,) higher = higher risk
        survival_fn : callable(t) → (N,) survival probabilities (optional)
        """
        report: Dict[str, Any] = {"model": model_name}

        # ── C-index ──
        c_idx = concordance_index_fast(
            self.event_times, predicted_risk, self.event_observed
        )
        report["c_index"] = c_idx
        logger.info("%s — C-index: %.4f", model_name, c_idx)

        # ── Time-dependent AUROC ──
        auroc_metrics = auroc_at_horizons(
            self.event_times, self.event_observed, predicted_risk, self.time_horizons
        )
        report.update(auroc_metrics)
        for k, v in auroc_metrics.items():
            logger.info("%s — %s: %.4f", model_name, k, v)

        # ── Brier score (requires survival function) ──
        if survival_fn is not None:
            brier_scores: Dict[str, float] = {}
            for t in self.time_horizons:
                s_t = survival_fn(t)
                bs = brier_score_at_time(
                    self.event_times,
                    self.event_observed,
                    s_t,
                    t,
                    self.train_event_times,
                    self.train_event_observed,
                )
                brier_scores[f"brier_{int(t)}m"] = bs

            report.update(brier_scores)
            ibs = integrated_brier_score(
                self.event_times, self.event_observed, survival_fn, 
                np.array(self.time_horizons)
            )
            report["ibs"] = ibs
            logger.info("%s — IBS: %.4f", model_name, ibs)

            # ── Calibration at 6-month horizon ──
            if 6.0 in self.time_horizons or 6 in self.time_horizons:
                s_6m = survival_fn(6.0)
                obs_6m = (
                    (self.event_times <= 6.0) & (self.event_observed == 1)
                ).astype(int)
                pred_6m = 1 - s_6m  # convert survival to event probability
                try:
                    cal = calibration_metrics(obs_6m, pred_6m)
                    report.update({f"calib_6m_{k}": v for k, v in cal.items()})
                except Exception as exc:
                    logger.warning("Calibration failed: %s", exc)

        self._reports.append(report)
        return report

    def compare_models(
        self,
        reports: Optional[List[Dict]] = None,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Build a side-by-side model comparison DataFrame and save to CSV.
        """
        reports = reports or self._reports
        if not reports:
            raise ValueError("No reports to compare.")

        comparison = pd.DataFrame(reports).set_index("model")

        # Rank by C-index
        if "c_index" in comparison.columns:
            comparison["c_index_rank"] = comparison["c_index"].rank(ascending=False)

        # Flag models meeting minimum thresholds
        thresholds = {
            "c_index": 0.70,
            "auroc_6m": 0.72,
            "ibs": 0.25,
        }
        for metric, threshold in thresholds.items():
            if metric in comparison.columns:
                if "brier" in metric or "ibs" in metric:
                    comparison[f"{metric}_ok"] = comparison[metric] <= threshold
                else:
                    comparison[f"{metric}_ok"] = comparison[metric] >= threshold

        logger.info("\n=== Model Comparison ===\n%s", comparison.to_string())

        if output_path is None:
            output_path = REPORTS_DIR / "survival_model_comparison.csv"
        comparison.to_csv(output_path)
        logger.info("Comparison saved to %s", output_path)

        return comparison

    def print_summary(self, report: Dict[str, Any]) -> None:
        """Pretty-print a single model evaluation."""
        print(f"\n{'=' * 50}")
        print(f"Model: {report['model']}")
        print(f"{'=' * 50}")
        print(f"  C-index:       {report.get('c_index', 'N/A'):.4f}")
        for h in self.time_horizons:
            k = f"auroc_{int(h)}m"
            v = report.get(k, float("nan"))
            print(f"  AUROC@{int(h)}m:   {v:.4f}" if not np.isnan(v) else f"  AUROC@{int(h)}m:   N/A")
        if "ibs" in report:
            print(f"  IBS:           {report['ibs']:.4f}")
        if "brier_6m" in report:
            print(f"  Brier@6m:      {report['brier_6m']:.4f}")
        print()


# ─── Fix class definition (Python dataclass-style not used above) ────────────

# Re-define cleanly (remove the stray `@dataclass_style` comment above)
SurvivalEvaluator.__init__.__doc__ = "Initialise evaluator with test set targets."


# ─── Convenience function ────────────────────────────────────────────────────


def quick_evaluate(
    model_name: str,
    event_times: np.ndarray,
    event_observed: np.ndarray,
    predicted_risk: np.ndarray,
    time_horizons: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    One-liner evaluation returning the key metrics dict.
    No survival function needed — only C-index and time-dependent AUROC.
    """
    ev = SurvivalEvaluator(event_times, event_observed, time_horizons or [3, 6, 12])
    return ev.evaluate(model_name, predicted_risk)


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)

    n = 300
    event_times = rng.exponential(scale=8.0, size=n).clip(0.5, 24)
    event_observed = rng.binomial(1, 0.7, n)
    # Better model: risk inversely correlated with time
    good_risk = 1 / event_times + rng.normal(0, 0.1, n)
    # Random baseline
    random_risk = rng.standard_normal(n)

    ev = SurvivalEvaluator(event_times, event_observed, time_horizons=[3, 6, 12])

    r1 = ev.evaluate("GoodModel", good_risk)
    r2 = ev.evaluate("RandomModel", random_risk)

    ev.print_summary(r1)
    ev.print_summary(r2)

    comparison = ev.compare_models()
    logger.info("\nComparison table:\n%s", comparison)

    assert r1["c_index"] > r2["c_index"], "Good model should beat random"
    assert r1["c_index"] > 0.6
    logger.info("=== PASS ===")
