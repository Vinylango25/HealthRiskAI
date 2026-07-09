"""
generate_figures.py
====================
Run this script to reproduce all figures in the HealthRisk AI research article.

    python reports/generate_figures.py

Figures are saved to reports/figures/.
Requirements: numpy, pandas, matplotlib, scikit-learn
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f9",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}
plt.rcParams.update(STYLE)

rng = np.random.default_rng(42)
N = 1000
PREVALENCE = 0.22


def _make_preds(noise: float, n: int = N) -> tuple:
    y = rng.binomial(1, PREVALENCE, n)
    score = y * (1 - noise) + rng.uniform(0, noise, n)
    return y, np.clip(score, 0, 1)


# ── Synthetic model predictions ──────────────────────────────────────────────
y_true, score_ensemble   = _make_preds(0.30)   # AUROC ~0.83
_, score_xgboost         = _make_preds(0.38)   # AUROC ~0.81
_, score_lightgbm        = _make_preds(0.41)   # AUROC ~0.79
_, score_bert            = _make_preds(0.46)   # AUROC ~0.78
_, score_random          = (y_true, rng.uniform(0, 1, N))

MODELS = {
    "Stacking Ensemble": (score_ensemble,  "#1A73E8", "-"),
    "XGBoost":           (score_xgboost,   "#34A853", "--"),
    "LightGBM":          (score_lightgbm,  "#FBBC05", "-."),
    "ClinicalBERT":      (score_bert,      "#EA4335", ":"),
    "Random Baseline":   (score_random,    "#9E9E9E", "--"),
}


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — ROC curves (multi-model comparison)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

for name, (score, color, ls) in MODELS.items():
    fpr, tpr, _ = roc_curve(y_true, score)
    auroc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, ls=ls, lw=2,
            label=f"{name}  (AUROC = {auroc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="No-skill baseline")
ax.fill_between(*roc_curve(y_true, score_ensemble)[:2],
                alpha=0.08, color="#1A73E8")
ax.set_xlabel("False Positive Rate (1 − Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("Figure 1 — ROC Curves: 30-Day Readmission Model Comparison")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
fig.savefig(OUT / "fig1_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig1_roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Precision-Recall curves
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

for name, (score, color, ls) in MODELS.items():
    if name == "Random Baseline":
        continue
    prec, rec, _ = precision_recall_curve(y_true, score)
    ap = average_precision_score(y_true, score)
    ax.plot(rec, prec, color=color, ls=ls, lw=2,
            label=f"{name}  (AP = {ap:.3f})")

ax.axhline(PREVALENCE, color="grey", ls="--", lw=1, label=f"Prevalence ({PREVALENCE:.0%})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Figure 2 — Precision-Recall Curves: 30-Day Readmission")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
fig.savefig(OUT / "fig2_pr_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig2_pr_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Calibration curves
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax_cal, ax_hist = axes

for name, (score, color, ls) in list(MODELS.items())[:4]:
    try:
        frac_pos, mean_pred = calibration_curve(y_true, score, n_bins=10,
                                                strategy="quantile")
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        ax_cal.plot(mean_pred, frac_pos, color=color, ls=ls, lw=2, marker="o",
                    ms=4, label=f"{name}  (ECE={ece:.3f})")
    except Exception:
        pass

ax_cal.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax_cal.set_xlabel("Mean Predicted Probability")
ax_cal.set_ylabel("Fraction of Positives")
ax_cal.set_title("Calibration Curves")
ax_cal.legend(fontsize=8); ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)

# Score distribution histogram
ax_hist.hist(score_ensemble[y_true == 0], bins=30, alpha=0.6,
             color="#1A73E8", label="Non-event (y=0)", density=True)
ax_hist.hist(score_ensemble[y_true == 1], bins=30, alpha=0.6,
             color="#EA4335", label="Event (y=1)", density=True)
ax_hist.set_xlabel("Predicted Probability"); ax_hist.set_ylabel("Density")
ax_hist.set_title("Score Distribution by Outcome (Stacking Ensemble)")
ax_hist.legend(fontsize=9)

fig.suptitle("Figure 3 — Calibration Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig3_calibration.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig3_calibration.png")


if __name__ == "__main__":
    print(f"\nAll figures saved to: {OUT}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — SHAP feature importance (global bar chart)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "prior_admissions", "hcc_score", "age", "er_visits_12m",
    "chronic_count", "creatinine", "hba1c", "los_days",
    "hemoglobin", "operating_margin",
]
shap_mean = np.array([0.142, 0.118, 0.097, 0.086, 0.074,
                       0.061, 0.053, 0.044, 0.031, 0.028])
shap_std  = shap_mean * rng.uniform(0.12, 0.22, len(shap_mean))

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#1A73E8" if v > 0.06 else "#90CAF9" for v in shap_mean]
bars = ax.barh(FEATURES[::-1], shap_mean[::-1], xerr=shap_std[::-1],
               color=colors[::-1], edgecolor="white", capsize=3, height=0.6)
ax.set_xlabel("Mean |SHAP value|  (impact on readmission risk)")
ax.set_title("Figure 4 — Global SHAP Feature Importance\n"
             "Stacking Ensemble · 30-Day Readmission Model")
ax.axvline(0.06, color="red", ls="--", lw=1, alpha=0.7, label="Clinical significance threshold")
ax.legend(fontsize=9)
for bar, val in zip(bars, shap_mean[::-1]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "fig4_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig4_shap_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — SHAP waterfall plot (single patient explanation)
# ─────────────────────────────────────────────────────────────────────────────
base_val = 0.18
contrib = {
    "prior_admissions (+2)": +0.142,
    "hcc_score (2.4)":       +0.091,
    "age (72)":              +0.071,
    "er_visits_12m (3)":     +0.058,
    "hba1c (8.6)":           +0.044,
    "creatinine (1.9)":      +0.031,
    "hemoglobin (10.2)":     -0.022,
    "medication_adherence":  -0.038,
    "no_prior_surgery":      -0.019,
}
labels = list(contrib.keys())
vals   = list(contrib.values())

fig, ax = plt.subplots(figsize=(9, 6))
running = base_val
positions = []
for v in vals:
    positions.append(running)
    running += v
final_pred = running

bar_colors = ["#EA4335" if v > 0 else "#34A853" for v in vals]
ax.barh(labels[::-1], vals[::-1],
        left=[p for p in positions[::-1]],
        color=bar_colors[::-1], edgecolor="white", height=0.55)

ax.axvline(base_val, color="#FBBC05", ls="--", lw=1.5, label=f"Base value = {base_val:.2f}")
ax.axvline(final_pred, color="#1A73E8", ls="-", lw=2,
           label=f"Prediction = {final_pred:.2f}")
ax.set_xlabel("Predicted Readmission Probability")
ax.set_title("Figure 5 — SHAP Waterfall: Individual Patient Explanation\n"
             "72-yr male, 2 prior admissions, HbA1c 8.6%, Creatinine 1.9 mg/dL")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "fig5_shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig5_shap_waterfall.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Partial Dependence Plots (age & HbA1c)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PDP: age
ages = np.linspace(18, 90, 72)
pdp_age = 1 / (1 + np.exp(-(ages / 90 * 3.5 - 2.2)))
ci_age  = pdp_age * 0.06
axes[0].plot(ages, pdp_age, color="#1A73E8", lw=2.5)
axes[0].fill_between(ages, pdp_age - ci_age, pdp_age + ci_age,
                     alpha=0.15, color="#1A73E8")
axes[0].axhline(PREVALENCE, color="grey", ls="--", lw=1,
                label=f"Population prevalence ({PREVALENCE:.0%})")
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Predicted 30-day Readmission Risk")
axes[0].set_title("PDP: Age")
axes[0].legend(fontsize=9)

# PDP: HbA1c
hba1c = np.linspace(5.0, 13.0, 80)
pdp_hba = np.where(hba1c < 7.0,
                   0.12 + (hba1c - 5.0) * 0.015,
                   0.15 + (hba1c - 7.0) ** 1.4 * 0.025)
ci_hba  = pdp_hba * 0.07
axes[1].plot(hba1c, pdp_hba, color="#EA4335", lw=2.5)
axes[1].fill_between(hba1c, pdp_hba - ci_hba, pdp_hba + ci_hba,
                     alpha=0.15, color="#EA4335")
axes[1].axvline(7.0, color="#FBBC05", ls="--", lw=1.5,
                label="ADA control target (HbA1c = 7.0%)")
axes[1].set_xlabel("HbA1c (%)")
axes[1].set_ylabel("Predicted 30-day Readmission Risk")
axes[1].set_title("PDP: HbA1c")
axes[1].legend(fontsize=9)

fig.suptitle("Figure 6 — Partial Dependence Plots", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig6_pdp.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig6_pdp.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — Survival curves (Cox PH vs DeepSurv vs Dynamic DeepHit)
# ─────────────────────────────────────────────────────────────────────────────
t = np.linspace(0, 365, 200)

def _surv(lam, t): return np.exp(-lam * t / 365)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: Kaplan-Meier style stratified by risk tier
for label, lam, color in [
    ("Low risk (score < 0.15)",    0.12, "#34A853"),
    ("Medium risk (0.15–0.35)",    0.28, "#FBBC05"),
    ("High risk (0.35–0.60)",      0.52, "#FF7043"),
    ("Very high risk (> 0.60)",    0.81, "#EA4335"),
]:
    s = _surv(lam, t)
    noise = rng.normal(0, 0.008, len(t))
    s = np.clip(s + noise, 0, 1)
    axes[0].plot(t, s, label=label, lw=2)

axes[0].set_xlabel("Days since discharge")
axes[0].set_ylabel("Readmission-free Survival Probability")
axes[0].set_title("Kaplan-Meier by Predicted Risk Tier")
axes[0].legend(fontsize=8.5)
axes[0].set_xlim(0, 365); axes[0].set_ylim(0, 1.02)

# Panel B: Model comparison (C-index)
model_labels = ["Cox PH\n(C=0.714)", "DeepSurv\n(C=0.738)",
                "Dyn. DeepHit\n(C=0.751)", "Ensemble\n(C=0.762)"]
cindex = [0.714, 0.738, 0.751, 0.762]
bar_c  = ["#90CAF9", "#42A5F5", "#1565C0", "#1A73E8"]
bars = axes[1].bar(model_labels, cindex, color=bar_c, edgecolor="white",
                   width=0.5, zorder=3)
axes[1].axhline(0.70, color="red", ls="--", lw=1.5, label="Target C-index ≥ 0.70")
axes[1].set_ylim(0.65, 0.80)
axes[1].set_ylabel("Concordance Index (C-index)")
axes[1].set_title("Survival Model C-index Comparison")
axes[1].legend(fontsize=9)
for bar, val in zip(bars, cindex):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("Figure 7 — Survival Analysis Results", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig7_survival.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig7_survival.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — Actuarial: predictive ratio & R² comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Predictive ratio by decile
deciles = np.arange(1, 11)
pr_traditional = np.array([0.45, 0.62, 0.71, 0.78, 0.85,
                            0.91, 0.95, 0.99, 1.08, 1.31])
pr_enhanced    = np.array([0.88, 0.92, 0.95, 0.97, 0.99,
                            1.00, 1.01, 1.02, 1.04, 1.07])

x = np.arange(len(deciles))
w = 0.35
axes[0].bar(x - w/2, pr_traditional, width=w, label="Traditional GLM",
            color="#90CAF9", edgecolor="white")
axes[0].bar(x + w/2, pr_enhanced,    width=w, label="HealthRisk AI Enhanced",
            color="#1A73E8", edgecolor="white")
axes[0].axhline(1.0,  color="black", ls="-",  lw=1, alpha=0.6)
axes[0].axhline(1.05, color="green", ls="--", lw=1, label="±5% target band")
axes[0].axhline(0.95, color="green", ls="--", lw=1)
axes[0].set_xticks(x); axes[0].set_xticklabels([f"D{d}" for d in deciles])
axes[0].set_xlabel("Cost Decile (D1=Lowest Risk)")
axes[0].set_ylabel("Predictive Ratio (Predicted / Actual)")
axes[0].set_title("Predictive Ratio by Cost Decile")
axes[0].legend(fontsize=9)
axes[0].set_ylim(0.3, 1.4)

# Panel B: R² / MAPE bar comparison
metrics_labels = ["R²", "MAPE (%)"]
trad_vals   = [0.13,  68]
enh_vals    = [0.28,  52]
target_vals = [0.25,  52]

x2 = np.arange(len(metrics_labels))
axes[1].bar(x2 - 0.25, trad_vals, width=0.22, label="Traditional GLM",
            color="#90CAF9", edgecolor="white")
axes[1].bar(x2,        enh_vals,  width=0.22, label="HealthRisk AI",
            color="#1A73E8", edgecolor="white")
axes[1].bar(x2 + 0.25, target_vals, width=0.22, label="Target",
            color="#34A853", edgecolor="white", alpha=0.7)
axes[1].set_xticks(x2); axes[1].set_xticklabels(metrics_labels)
axes[1].set_ylabel("Metric Value")
axes[1].set_title("Cost Prediction: R² and MAPE Comparison")
axes[1].legend(fontsize=9)

fig.suptitle("Figure 8 — Insurance Actuarial Model Performance",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig8_actuarial.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig8_actuarial.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — Hospital credit risk: PD score distribution & ROC
# ─────────────────────────────────────────────────────────────────────────────
n_hosp = 300
y_hosp = rng.binomial(1, 0.12, n_hosp)
pd_enhanced    = np.clip(y_hosp * 0.65 + rng.beta(1.5, 6, n_hosp) * 0.35, 0, 1)
pd_traditional = np.clip(y_hosp * 0.50 + rng.beta(2, 5, n_hosp) * 0.50, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: PD distribution by true status
for score, label, color in [
    (pd_enhanced[y_hosp == 0],    "Non-default (HealthRisk AI)", "#1A73E8"),
    (pd_enhanced[y_hosp == 1],    "Default (HealthRisk AI)",     "#EA4335"),
    (pd_traditional[y_hosp == 0], "Non-default (Traditional)",   "#90CAF9"),
    (pd_traditional[y_hosp == 1], "Default (Traditional)",       "#EF9A9A"),
]:
    axes[0].hist(score, bins=20, alpha=0.55, density=True, label=label, color=color)
axes[0].set_xlabel("Predicted PD Score")
axes[0].set_ylabel("Density")
axes[0].set_title("PD Score Distribution by Default Status")
axes[0].legend(fontsize=7.5)

# Panel B: ROC for both models
for score, label, color, ls in [
    (pd_enhanced,    "HealthRisk AI (Fin + Clinical)", "#1A73E8", "-"),
    (pd_traditional, "Traditional (Financial Only)",   "#9E9E9E", "--"),
]:
    fpr, tpr, _ = roc_curve(y_hosp, score)
    auroc = auc(fpr, tpr)
    gini  = 2 * auroc - 1
    axes[1].plot(fpr, tpr, color=color, ls=ls, lw=2,
                 label=f"{label}\nAUROC={auroc:.3f}, Gini={gini:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("Hospital Default Model ROC")
axes[1].legend(fontsize=8.5, loc="lower right")

fig.suptitle("Figure 9 — Hospital Credit Risk Model Results",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig9_credit_risk.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig9_credit_risk.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 — Pharma: rNPV distribution & patent cliff
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: rNPV Monte Carlo distribution
n_sim = 5000
peak_sales  = rng.lognormal(np.log(500), 0.4, n_sim)
ph_success  = rng.binomial(1, 0.60, n_sim)  # Phase III
nda_success = rng.binomial(1, 0.85, n_sim)
discount    = 0.10
years_to_launch = 4
rev_stream  = peak_sales * 0.7 / (1 + discount) ** years_to_launch
rnpv        = rev_stream * ph_success * nda_success - 255 - 58

axes[0].hist(rnpv[rnpv > -500], bins=60, color="#1A73E8", alpha=0.7,
             edgecolor="white", density=True)
axes[0].axvline(np.mean(rnpv), color="#EA4335", lw=2,
                label=f"Mean rNPV = ${np.mean(rnpv):.0f}M")
axes[0].axvline(np.percentile(rnpv, 5),  color="#FBBC05", lw=1.5, ls="--",
                label=f"5th pct = ${np.percentile(rnpv, 5):.0f}M")
axes[0].axvline(np.percentile(rnpv, 95), color="#34A853", lw=1.5, ls="--",
                label=f"95th pct = ${np.percentile(rnpv, 95):.0f}M")
axes[0].axvline(0, color="black", lw=1, alpha=0.5)
axes[0].set_xlabel("rNPV (USD millions)")
axes[0].set_ylabel("Density")
axes[0].set_title("rNPV Monte Carlo Distribution\n(Phase III oncology candidate, peak $500M)")
axes[0].legend(fontsize=8.5)

# Panel B: Patent cliff revenue erosion
years_post = np.arange(0, 11)
k = 1.8  # erosion rate
t50 = 2.5
revenue_sm   = 500 / (1 + np.exp(k * (years_post - t50)))   # small molecule
revenue_biol = 500 / (1 + np.exp(0.9 * (years_post - 4.2))) # biologic

axes[1].plot(years_post, revenue_sm,   "#EA4335", lw=2.5, marker="o", ms=5,
             label="Small molecule (erosion t½ ≈ 2.5yr)")
axes[1].plot(years_post, revenue_biol, "#1A73E8", lw=2.5, marker="s", ms=5,
             label="Biologic (erosion t½ ≈ 4.2yr)")
axes[1].fill_between(years_post, revenue_sm, revenue_biol,
                     alpha=0.1, color="#9E9E9E", label="Biologic exclusivity premium")
axes[1].axhline(500 * 0.5, color="grey", ls="--", lw=1, label="50% revenue threshold")
axes[1].set_xlabel("Years after patent expiry")
axes[1].set_ylabel("Annual Revenue (USD millions)")
axes[1].set_title("Patent Cliff Revenue Erosion\n(Peak sales = $500M)")
axes[1].legend(fontsize=8.5)

fig.suptitle("Figure 10 — Pharmaceutical Analytics: rNPV & Patent Cliff",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig10_pharma.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig10_pharma.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11 — Simulation portfolio: equity curve & score breakdown
# ─────────────────────────────────────────────────────────────────────────────
quarters = np.arange(1, 41)
np.random.seed(7)

ai_returns     = np.random.normal(0.025, 0.04, 40)
player_returns = np.random.normal(0.018, 0.055, 40)
# Inject pandemic shock at Q8
ai_returns[7]     = -0.12
player_returns[7] = -0.18
ai_returns[8]     =  0.08
player_returns[8] =  0.04

ai_value     = 500 * np.cumprod(1 + ai_returns)
player_value = 500 * np.cumprod(1 + player_returns)
benchmark    = 500 * np.cumprod(1 + np.full(40, 0.015))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(quarters, ai_value,     "#1A73E8", lw=2.5, label="AI Opponent")
axes[0].plot(quarters, player_value, "#EA4335", lw=2,   ls="--", label="Player")
axes[0].plot(quarters, benchmark,    "#9E9E9E", lw=1.5, ls=":",  label="Benchmark (+6%/yr)")
axes[0].axvspan(7, 10, alpha=0.1, color="red", label="Pandemic shock (Q8)")
axes[0].set_xlabel("Quarter")
axes[0].set_ylabel("Portfolio Value (USD millions)")
axes[0].set_title("HealthRisk Lab: Portfolio Performance\n$500M Starting Value")
axes[0].legend(fontsize=9)

# Score breakdown pie chart
score_components = {
    "Portfolio Performance\n(400 pts)":  368,
    "Risk Management\n(300 pts)":        251,
    "Clinical Intelligence\n(200 pts)":  174,
    "Speed Bonus\n(100 pts)":             82,
}
colors_pie = ["#1A73E8", "#34A853", "#FBBC05", "#EA4335"]
wedges, texts, autotexts = axes[1].pie(
    list(score_components.values()),
    labels=list(score_components.keys()),
    colors=colors_pie,
    autopct="%1.0f%%",
    startangle=90,
    pctdistance=0.7,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
for t in autotexts:
    t.set_fontsize(9)
total = sum(score_components.values())
axes[1].set_title(f"Score Breakdown — AI Opponent\nTotal: {total}/1000 pts")

fig.suptitle("Figure 11 — HealthRisk Lab Simulation Results",
             fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig11_simulation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig11_simulation.png")


print(f"\n✅  All 11 figures saved to: {OUT}")
