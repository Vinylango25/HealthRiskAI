"""
risk_stratifier.py
==================
Member risk stratification into care management tiers.

Pipeline:
  1. HCC-based risk scoring  – weighted HCC condition categories
  2. Clinical complexity      – chronic burden, utilization history
  3. K-means clustering       – unsupervised tier seeding
  4. Rule-based override      – guardrail thresholds per tier
  5. Care management flag     – members who require outreach

Tiers: Low | Medium | High | Very High

Outputs per member:
  - risk_score       : composite 0-100 score
  - risk_tier        : Low / Medium / High / Very High
  - care_mgmt_flag   : bool (True → proactive care management required)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed – K-means clustering unavailable.")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    HAS_MLFLOW = False


# ---------------------------------------------------------------------------
# HCC Condition Category weights
# ---------------------------------------------------------------------------

# Relative risk weights by HCC category (simplified representative set).
# In production these map to CMS-HCC model v28 coefficients.
HCC_WEIGHTS: dict[str, float] = {
    "diabetes_with_complications": 0.45,
    "diabetes_without_complications": 0.20,
    "chf": 0.75,
    "copd": 0.55,
    "ckd_stage_3_4": 0.50,
    "ckd_stage_5_esrd": 1.20,
    "cancer_active": 1.50,
    "cancer_historical": 0.40,
    "hiv_aids": 0.90,
    "major_depression": 0.35,
    "schizophrenia": 0.65,
    "substance_use_disorder": 0.30,
    "alzheimers": 0.80,
    "stroke": 0.60,
    "ami": 0.70,
    "pvd": 0.45,
    "amputation": 0.55,
    "septicemia": 1.10,
    "trauma": 0.40,
    "transplant": 1.30,
}

# Tier thresholds (risk score 0-100)
TIER_THRESHOLDS = {
    "Very High": 75.0,
    "High": 50.0,
    "Medium": 25.0,
    "Low": 0.0,
}

# Care management flag triggers
CARE_MGMT_RULES = {
    "min_risk_score": 50.0,          # score >= 50
    "high_hcc_conditions": 3,        # ≥ 3 active HCC conditions
    "recent_er_visits": 2,           # ≥ 2 ER visits in last 12m
    "hospital_admissions": 1,        # any inpatient admission
    "predicted_cost_percentile": 80, # top 20% predicted cost
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StratificationResult:
    """Per-member stratification output."""

    member_id: np.ndarray
    risk_score: np.ndarray
    risk_tier: np.ndarray          # dtype: object (string tier labels)
    care_mgmt_flag: np.ndarray     # dtype: bool
    cluster_id: Optional[np.ndarray] = None

    def to_dataframe(self) -> pd.DataFrame:
        d: dict = {
            "member_id": self.member_id,
            "risk_score": self.risk_score.round(2),
            "risk_tier": self.risk_tier,
            "care_mgmt_flag": self.care_mgmt_flag,
        }
        if self.cluster_id is not None:
            d["cluster_id"] = self.cluster_id
        return pd.DataFrame(d)


@dataclass
class StratificationEvaluation:
    """Evaluation metrics for the stratification model."""

    tier_distribution: dict[str, float]   # % in each tier
    mean_score_by_tier: dict[str, float]
    care_mgmt_rate: float
    silhouette_score: Optional[float]
    tier_cost_ratio: Optional[dict[str, float]]  # mean actual cost per tier
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clinical Complexity Scorer
# ---------------------------------------------------------------------------


class ClinicalComplexityScorer:
    """
    Scores a member's clinical complexity based on chronic burden,
    utilization, and demographics.

    Score components (0-100 scale):
      - HCC composite (0-40)
      - Utilization burden (0-30)
      - Age/demographic adjustment (0-15)
      - Medication complexity (0-15)
    """

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute clinical complexity scores.

        Expected columns (at minimum):
          hcc_score, chronic_count, er_visits_12m, inpatient_admits_12m,
          age, rx_drug_count
        """
        n = len(df)
        scores = np.zeros(n)

        # --- HCC component (0-40) ---
        hcc_raw = df.get("hcc_score", pd.Series(np.ones(n))).values.astype(float)
        # Normalise: HCC score > 3.0 considered extreme
        hcc_component = np.clip(hcc_raw / 3.0, 0, 1) * 40
        scores += hcc_component

        # --- Utilization component (0-30) ---
        er = df.get("er_visits_12m", pd.Series(np.zeros(n))).values.astype(float)
        ip = df.get("inpatient_admits_12m", pd.Series(np.zeros(n))).values.astype(float)
        util_raw = np.clip(er / 5.0, 0, 0.5) + np.clip(ip / 3.0, 0, 0.5)
        scores += util_raw * 30

        # --- Age component (0-15) ---
        age = df.get("age", pd.Series(np.full(n, 45.0))).values.astype(float)
        age_norm = np.clip((age - 18) / (85 - 18), 0, 1) * 15
        scores += age_norm

        # --- Medication complexity (0-15) ---
        rx = df.get("rx_drug_count", pd.Series(np.zeros(n))).values.astype(float)
        rx_component = np.clip(rx / 15.0, 0, 1) * 15
        scores += rx_component

        return np.clip(scores, 0, 100)


# ---------------------------------------------------------------------------
# Core Risk Stratifier
# ---------------------------------------------------------------------------


class RiskStratifier:
    """
    Member risk stratification model.

    Parameters
    ----------
    n_clusters : int
        Number of K-means clusters (default 4 maps to 4 tiers).
    hcc_weight : float
        Blending weight for HCC-based score vs clinical complexity (0-1).
    mlflow_experiment : str, optional
        MLflow experiment name.
    """

    CLUSTERING_FEATURES = [
        "hcc_score", "chronic_count", "age",
        "er_visits_12m", "inpatient_admits_12m", "rx_drug_count",
        "prior_pmpm",
    ]

    def __init__(
        self,
        n_clusters: int = 4,
        hcc_weight: float = 0.6,
        mlflow_experiment: Optional[str] = "healthrisk_stratification",
    ) -> None:
        self.n_clusters = n_clusters
        self.hcc_weight = hcc_weight
        self.mlflow_experiment = mlflow_experiment

        self._kmeans: Optional[KMeans] = None  # type: ignore[name-defined]
        self._scaler = StandardScaler()
        self._clinical_scorer = ClinicalComplexityScorer()
        self._cluster_to_tier: dict[int, str] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # HCC-based scoring
    # ------------------------------------------------------------------

    def _compute_hcc_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute HCC-based risk score from condition flags or composite HCC field.

        If individual HCC condition columns exist, sum their weights.
        Falls back to the scalar 'hcc_score' column scaled to 0-100.
        """
        n = len(df)
        condition_scores = np.zeros(n)

        # Check for individual condition flag columns
        for condition, weight in HCC_WEIGHTS.items():
            if condition in df.columns:
                condition_scores += df[condition].fillna(0).astype(float).values * weight

        if condition_scores.sum() > 0:
            # Scale sum to 0-100 (max theoretical weight ≈ 10)
            return np.clip(condition_scores / 10.0 * 100, 0, 100)
        elif "hcc_score" in df.columns:
            # CMS HCC score: community average ≈ 1.0; scale to 0-100
            return np.clip(df["hcc_score"].fillna(1.0).values * 30, 0, 100)
        else:
            return np.full(n, 20.0)  # default community average

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        feat_df = pd.DataFrame(index=df.index)
        for col in self.CLUSTERING_FEATURES:
            if col in df.columns:
                feat_df[col] = df[col].fillna(df[col].median() if col in df.columns else 0)
            else:
                feat_df[col] = 0.0
        return feat_df[self.CLUSTERING_FEATURES].values.astype(float)

    # ------------------------------------------------------------------
    # Composite risk score
    # ------------------------------------------------------------------

    def _composite_score(self, df: pd.DataFrame) -> np.ndarray:
        """Blend HCC score and clinical complexity into 0-100 composite."""
        hcc_s = self._compute_hcc_score(df)
        clinical_s = self._clinical_scorer.score(df)
        return self.hcc_weight * hcc_s + (1 - self.hcc_weight) * clinical_s

    # ------------------------------------------------------------------
    # Rule-based tier assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_tier(scores: np.ndarray) -> np.ndarray:
        """Map 0-100 composite scores to tier labels via threshold rules."""
        tiers = np.empty(len(scores), dtype=object)
        tiers[:] = "Low"
        tiers[scores >= TIER_THRESHOLDS["Medium"]] = "Medium"
        tiers[scores >= TIER_THRESHOLDS["High"]] = "High"
        tiers[scores >= TIER_THRESHOLDS["Very High"]] = "Very High"
        return tiers

    # ------------------------------------------------------------------
    # Care management flag
    # ------------------------------------------------------------------

    def _flag_care_management(self, df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
        """Apply rule-based care management flagging logic."""
        n = len(df)
        flag = np.zeros(n, dtype=bool)

        # Score threshold
        flag |= scores >= CARE_MGMT_RULES["min_risk_score"]

        # ER visits
        if "er_visits_12m" in df.columns:
            flag |= df["er_visits_12m"].fillna(0).values >= CARE_MGMT_RULES["recent_er_visits"]

        # Inpatient admissions
        if "inpatient_admits_12m" in df.columns:
            flag |= df["inpatient_admits_12m"].fillna(0).values >= CARE_MGMT_RULES["hospital_admissions"]

        # HCC condition count
        if "chronic_count" in df.columns:
            flag |= df["chronic_count"].fillna(0).values >= CARE_MGMT_RULES["high_hcc_conditions"]

        # Top predicted cost percentile (use score as proxy)
        pct_threshold = np.percentile(scores, CARE_MGMT_RULES["predicted_cost_percentile"])
        flag |= scores >= pct_threshold

        return flag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "RiskStratifier":
        """
        Fit the stratification model on historical member data.

        Parameters
        ----------
        df : pd.DataFrame
            Member population with clinical and utilization features.
        """
        logger.info("Fitting RiskStratifier on %d members.", len(df))

        X = self._prepare_features(df)
        X_scaled = self._scaler.fit_transform(X)

        if HAS_SKLEARN:
            self._kmeans = KMeans(
                n_clusters=self.n_clusters,
                init="k-means++",
                n_init=10,
                random_state=42,
            )
            cluster_ids = self._kmeans.fit_predict(X_scaled)

            # Map clusters to tiers by ascending centroid norm (proxy for risk)
            centroid_norms = np.linalg.norm(self._kmeans.cluster_centers_, axis=1)
            order = np.argsort(centroid_norms)
            tier_names = ["Low", "Medium", "High", "Very High"]
            self._cluster_to_tier = {int(order[i]): tier_names[i] for i in range(self.n_clusters)}
            logger.info("K-means cluster→tier mapping: %s", self._cluster_to_tier)
        else:
            logger.warning("K-means unavailable; rule-based only.")

        self._is_fitted = True

        if HAS_MLFLOW and self.mlflow_experiment:
            self._log_to_mlflow(df)

        return self

    def stratify(self, df: pd.DataFrame) -> StratificationResult:
        """
        Assign risk tiers to a member population.

        Parameters
        ----------
        df : pd.DataFrame
            Member data with clinical/utilization features.

        Returns
        -------
        StratificationResult
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before stratify().")

        composite_scores = self._composite_score(df)

        # Primary tier from composite score (rule-based)
        rule_tiers = self._score_to_tier(composite_scores)

        # Optionally blend with K-means tier
        cluster_ids: Optional[np.ndarray] = None
        if HAS_SKLEARN and self._kmeans is not None:
            X = self._prepare_features(df)
            X_scaled = self._scaler.transform(X)
            cluster_ids = self._kmeans.predict(X_scaled)
            km_tiers = np.array([self._cluster_to_tier.get(int(c), "Medium") for c in cluster_ids])

            # Use rule tier as primary; upgrade to km_tier if higher risk
            tier_order = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
            final_tiers = np.where(
                [tier_order[km_tiers[i]] > tier_order[rule_tiers[i]] for i in range(len(rule_tiers))],
                km_tiers,
                rule_tiers,
            )
        else:
            final_tiers = rule_tiers

        care_flags = self._flag_care_management(df, composite_scores)

        member_ids = (
            df["member_id"].values if "member_id" in df.columns else np.arange(len(df))
        )

        return StratificationResult(
            member_id=member_ids,
            risk_score=composite_scores,
            risk_tier=final_tiers,
            care_mgmt_flag=care_flags,
            cluster_id=cluster_ids,
        )

    def evaluate_stratification(
        self,
        df: pd.DataFrame,
        actual_cost_col: Optional[str] = "actual_pmpm",
    ) -> StratificationEvaluation:
        """
        Evaluate tier separation quality.

        Parameters
        ----------
        df : pd.DataFrame
            Member data (can include actual cost column for cost validation).
        actual_cost_col : str, optional
            Column name for actual PMPM cost.

        Returns
        -------
        StratificationEvaluation
        """
        result = self.stratify(df)
        tiers = result.risk_tier
        scores = result.risk_score
        n = len(df)

        # Tier distribution
        tier_dist = {
            t: float((tiers == t).sum() / n * 100) for t in ["Low", "Medium", "High", "Very High"]
        }
        mean_score_by_tier = {
            t: float(scores[tiers == t].mean()) if (tiers == t).any() else 0.0
            for t in ["Low", "Medium", "High", "Very High"]
        }
        care_rate = float(result.care_mgmt_flag.mean() * 100)

        # Silhouette score
        sil: Optional[float] = None
        if HAS_SKLEARN and result.cluster_id is not None and len(np.unique(result.cluster_id)) > 1:
            try:
                from sklearn.metrics import silhouette_score
                X_scaled = self._scaler.transform(self._prepare_features(df))
                sil = float(silhouette_score(X_scaled, result.cluster_id, sample_size=min(2000, n)))
            except Exception:
                pass

        # Cost validation per tier
        cost_by_tier: Optional[dict[str, float]] = None
        if actual_cost_col and actual_cost_col in df.columns:
            costs = df[actual_cost_col].values.astype(float)
            cost_by_tier = {
                t: float(costs[tiers == t].mean()) if (tiers == t).any() else 0.0
                for t in ["Low", "Medium", "High", "Very High"]
            }

        eval_result = StratificationEvaluation(
            tier_distribution=tier_dist,
            mean_score_by_tier=mean_score_by_tier,
            care_mgmt_rate=care_rate,
            silhouette_score=sil,
            tier_cost_ratio=cost_by_tier,
            details={"n_members": n},
        )

        logger.info(
            "Stratification — Tier distribution: %s | Care mgmt rate: %.1f%%",
            tier_dist, care_rate,
        )

        return eval_result

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------

    def _log_to_mlflow(self, df: pd.DataFrame) -> None:
        try:
            mlflow.set_experiment(self.mlflow_experiment)
            with mlflow.start_run(run_name="risk_stratifier_fit"):
                mlflow.log_param("n_clusters", self.n_clusters)
                mlflow.log_param("hcc_weight", self.hcc_weight)
                mlflow.log_param("n_members", len(df))
        except Exception as exc:
            logger.debug("MLflow logging skipped: %s", exc)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def _generate_synthetic_members(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 90, size=n)
    hcc = np.clip(rng.gamma(1.2, 0.6, size=n), 0.1, 6.0).round(3)
    chronic = rng.integers(0, 8, size=n)
    er = rng.integers(0, 10, size=n)
    ip = rng.integers(0, 5, size=n)
    rx = rng.integers(0, 20, size=n)
    prior_pmpm = np.clip(200 + 3 * ages + 120 * hcc + 30 * chronic + rng.normal(0, 50, n), 50, 8000)
    actual_pmpm = np.clip(prior_pmpm * rng.normal(1.0, 0.15, n), 50, 10000)

    return pd.DataFrame({
        "member_id": np.arange(n),
        "age": ages,
        "hcc_score": hcc,
        "chronic_count": chronic,
        "er_visits_12m": er,
        "inpatient_admits_12m": ip,
        "rx_drug_count": rx,
        "prior_pmpm": prior_pmpm.round(2),
        "actual_pmpm": actual_pmpm.round(2),
        # Example HCC condition flags
        "diabetes_with_complications": rng.binomial(1, 0.08, n).astype(float),
        "chf": rng.binomial(1, 0.05, n).astype(float),
        "ckd_stage_3_4": rng.binomial(1, 0.07, n).astype(float),
        "cancer_active": rng.binomial(1, 0.03, n).astype(float),
    })


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== RiskStratifier smoke test ===")

    data = _generate_synthetic_members(n=3000)
    train_df = data.sample(frac=0.8, random_state=42)
    test_df = data.drop(train_df.index)

    stratifier = RiskStratifier(n_clusters=4, hcc_weight=0.6, mlflow_experiment=None)
    stratifier.fit(train_df)

    result = stratifier.stratify(test_df)
    print("\nSample stratification results:")
    print(result.to_dataframe().head(15).to_string(index=False))

    eval_metrics = stratifier.evaluate_stratification(test_df, actual_cost_col="actual_pmpm")
    print("\nStratification Evaluation:")
    print(f"  Tier Distribution (%)    : {eval_metrics.tier_distribution}")
    print(f"  Mean Score by Tier       : {eval_metrics.mean_score_by_tier}")
    print(f"  Care Management Rate (%) : {eval_metrics.care_mgmt_rate:.1f}%")
    if eval_metrics.silhouette_score is not None:
        print(f"  Silhouette Score         : {eval_metrics.silhouette_score:.4f}")
    if eval_metrics.tier_cost_ratio:
        print(f"  Mean Cost by Tier ($)    : {eval_metrics.tier_cost_ratio}")

    # Sanity checks
    tiers_set = set(result.risk_tier)
    assert tiers_set.issubset({"Low", "Medium", "High", "Very High"}), f"Unexpected tiers: {tiers_set}"
    assert 0 <= eval_metrics.care_mgmt_rate <= 100, "Care mgmt rate out of range"
    # Very High should have higher mean score than Low
    if "Very High" in eval_metrics.mean_score_by_tier and "Low" in eval_metrics.mean_score_by_tier:
        assert eval_metrics.mean_score_by_tier["Very High"] >= eval_metrics.mean_score_by_tier["Low"], \
            "Very High tier should have higher mean score than Low"

    logger.info("=== Smoke test PASSED ===")
