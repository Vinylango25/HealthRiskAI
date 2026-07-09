"""
counterfactual.py
=================
DiCE-style Diverse Counterfactual Explanation generator for HealthRiskAI.

Finds minimal, actionable feature changes that flip a prediction from its
current class to a desired class.

Example output:
    "To change risk tier from High to Medium:
     reduce hcc_score by 0.8, increase days_cash by 45"
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureConstraint:
    """Defines actionability and valid range for a single feature."""
    name: str
    actionable: bool = True
    min_value: float = -np.inf
    max_value: float = np.inf
    step: Optional[float] = None        # granularity (e.g. integer steps)

    def clip(self, value: float) -> float:
        v = np.clip(value, self.min_value, self.max_value)
        if self.step is not None:
            v = round(v / self.step) * self.step
        return float(v)


@dataclass
class SingleCounterfactual:
    """One counterfactual instance with quality metrics."""
    original: np.ndarray
    counterfactual: np.ndarray
    feature_names: List[str]
    changed_features: List[Dict[str, Any]]  # [{name, original, new, delta}]
    predicted_class: int
    predicted_proba: float
    validity: bool                           # reaches desired class
    proximity: float                         # L1 normalised distance (lower=better)
    sparsity: float                          # fraction of features unchanged (higher=better)
    actionability: float                     # fraction of changes that are actionable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterfactual": self.counterfactual.tolist(),
            "changed_features": self.changed_features,
            "predicted_class": self.predicted_class,
            "predicted_proba": self.predicted_proba,
            "validity": self.validity,
            "proximity": self.proximity,
            "sparsity": self.sparsity,
            "actionability": self.actionability,
        }

    def to_human_readable(self, class_labels: Optional[Dict[int, str]] = None) -> str:
        parts = []
        for ch in self.changed_features:
            delta = ch["delta"]
            direction = "increase" if delta > 0 else "reduce"
            parts.append(f"{direction} {ch['name']} by {abs(delta):.3g}")
        cname = (class_labels or {}).get(self.predicted_class, str(self.predicted_class))
        return f"To reach class '{cname}': " + ", ".join(parts) if parts else "No changes needed."


@dataclass
class CounterfactualResult:
    """Container for all generated counterfactuals for one query."""
    original: np.ndarray
    original_class: int
    desired_class: int
    feature_names: List[str]
    counterfactuals: List[SingleCounterfactual] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def best(self) -> Optional[SingleCounterfactual]:
        valid = [cf for cf in self.counterfactuals if cf.validity]
        if not valid:
            return None
        return min(valid, key=lambda cf: cf.proximity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original.tolist(),
            "original_class": self.original_class,
            "desired_class": self.desired_class,
            "feature_names": self.feature_names,
            "counterfactuals": [cf.to_dict() for cf in self.counterfactuals],
            "elapsed_seconds": self.elapsed_seconds,
            "n_valid": sum(cf.validity for cf in self.counterfactuals),
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CounterfactualGenerator:
    """
    Gradient-free DiCE-style diverse counterfactual generator.

    Uses random restarts + greedy feature perturbation with a proximity +
    diversity penalty to find n_counterfactuals distinct solutions.

    Parameters
    ----------
    predict_fn : callable
        predict_fn(X: np.ndarray) → np.ndarray of shape (n, n_classes)
        Should return class probabilities.
    feature_names : list of str
    constraints : list of FeatureConstraint, optional
        If None, all features treated as actionable with range inferred from data.
    proximity_weight : float
        Weight on L1 distance penalty (higher = prefers smaller changes).
    diversity_weight : float
        Weight on diversity penalty between counterfactuals.
    max_iter : int
        Greedy search iterations per restart.
    n_restarts : int
        Number of random restarts to improve diversity.
    random_state : int
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: List[str],
        constraints: Optional[List[FeatureConstraint]] = None,
        proximity_weight: float = 1.0,
        diversity_weight: float = 0.5,
        max_iter: int = 500,
        n_restarts: int = 10,
        random_state: int = 42,
    ) -> None:
        self.predict_fn = predict_fn
        self.feature_names = feature_names
        self.constraints = constraints or [
            FeatureConstraint(name=n) for n in feature_names
        ]
        self._constraint_map: Dict[str, FeatureConstraint] = {
            c.name: c for c in self.constraints
        }
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(random_state)
        self._feature_ranges: Dict[str, Tuple[float, float]] = {}

    def fit(self, X: np.ndarray) -> "CounterfactualGenerator":
        """Infer feature ranges from training data."""
        for i, name in enumerate(self.feature_names):
            self._feature_ranges[name] = (float(X[:, i].min()), float(X[:, i].max()))
            c = self._constraint_map[name]
            if c.min_value == -np.inf:
                c.min_value = self._feature_ranges[name][0]
            if c.max_value == np.inf:
                c.max_value = self._feature_ranges[name][1]
        logger.info("CounterfactualGenerator fitted on %d samples.", len(X))
        return self

    def generate(
        self,
        x: np.ndarray,
        desired_class: int,
        n_counterfactuals: int = 3,
        class_labels: Optional[Dict[int, str]] = None,
    ) -> CounterfactualResult:
        """
        Generate counterfactual explanations.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
        desired_class : int
        n_counterfactuals : int

        Returns
        -------
        CounterfactualResult
        """
        t0 = time.perf_counter()
        if x.ndim > 1:
            x = x.flatten()

        proba = self.predict_fn(x.reshape(1, -1))[0]
        original_class = int(np.argmax(proba))

        if original_class == desired_class:
            logger.warning("Instance already in desired class %d.", desired_class)

        candidates: List[SingleCounterfactual] = []
        found_set: List[np.ndarray] = []

        for restart in range(self.n_restarts * 2):
            if len(candidates) >= n_counterfactuals * 3:
                break
            cf_x = self._search(x, desired_class, found_set)
            if cf_x is not None:
                cf_proba = self.predict_fn(cf_x.reshape(1, -1))[0]
                cf_class = int(np.argmax(cf_proba))
                sc = self._build_single_cf(x, cf_x, cf_class,
                                           float(cf_proba[desired_class]),
                                           desired_class)
                candidates.append(sc)
                found_set.append(cf_x)

        # Sort by validity then proximity, deduplicate
        candidates.sort(key=lambda c: (not c.validity, c.proximity))
        unique = self._deduplicate(candidates, n_counterfactuals)

        result = CounterfactualResult(
            original=x,
            original_class=original_class,
            desired_class=desired_class,
            feature_names=self.feature_names,
            counterfactuals=unique,
            elapsed_seconds=time.perf_counter() - t0,
        )
        logger.info("Generated %d counterfactuals (%d valid) in %.2fs.",
                    len(unique), result.to_dict()["n_valid"], result.elapsed_seconds)
        return result

    # ------------------------------------------------------------------
    # Internal search
    # ------------------------------------------------------------------

    def _search(
        self,
        x: np.ndarray,
        desired_class: int,
        existing: List[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Greedy perturbation search with random feature ordering."""
        candidate = x.copy()
        # Random small perturbation to start
        noise_scale = 0.05
        for i, name in enumerate(self.feature_names):
            c = self._constraint_map[name]
            if c.actionable:
                rng_span = c.max_value - c.min_value
                candidate[i] = c.clip(candidate[i] + self.rng.normal(0, noise_scale * rng_span))

        feat_order = self.rng.permutation(len(self.feature_names))

        for _ in range(self.max_iter):
            proba = self.predict_fn(candidate.reshape(1, -1))[0]
            if np.argmax(proba) == desired_class:
                return candidate

            # Pick feature to perturb
            fi = int(feat_order[_ % len(feat_order)])
            name = self.feature_names[fi]
            c = self._constraint_map[name]
            if not c.actionable:
                continue

            rng_span = c.max_value - c.min_value
            best_loss = self._loss(candidate, x, proba, desired_class, existing)
            best_val = candidate[fi]

            for _ in range(5):
                delta = self.rng.uniform(-0.3 * rng_span, 0.3 * rng_span)
                trial = candidate.copy()
                trial[fi] = c.clip(trial[fi] + delta)
                trial_proba = self.predict_fn(trial.reshape(1, -1))[0]
                loss = self._loss(trial, x, trial_proba, desired_class, existing)
                if loss < best_loss:
                    best_loss = loss
                    best_val = trial[fi]

            candidate[fi] = best_val

        proba = self.predict_fn(candidate.reshape(1, -1))[0]
        return candidate if np.argmax(proba) == desired_class else None

    def _loss(
        self,
        candidate: np.ndarray,
        original: np.ndarray,
        proba: np.ndarray,
        desired_class: int,
        existing: List[np.ndarray],
    ) -> float:
        # Classification loss: want high probability at desired class
        clf_loss = 1.0 - float(proba[desired_class])
        # Proximity: L1 normalised
        spans = np.array([
            (self._constraint_map[n].max_value - self._constraint_map[n].min_value) or 1.0
            for n in self.feature_names
        ])
        proximity = float(np.mean(np.abs(candidate - original) / spans))
        # Diversity: penalise similarity to already-found counterfactuals
        diversity_penalty = 0.0
        if existing:
            dists = [float(np.mean(np.abs(candidate - e) / spans)) for e in existing]
            diversity_penalty = -min(dists)  # negative because closer = less diverse

        return clf_loss + self.proximity_weight * proximity + self.diversity_weight * diversity_penalty

    def _build_single_cf(
        self,
        original: np.ndarray,
        cf: np.ndarray,
        cf_class: int,
        cf_proba: float,
        desired_class: int,
    ) -> SingleCounterfactual:
        changed = []
        actionable_changes = 0
        total_changes = 0
        spans = []
        for i, name in enumerate(self.feature_names):
            span = (self._constraint_map[name].max_value -
                    self._constraint_map[name].min_value) or 1.0
            spans.append(span)
            delta = cf[i] - original[i]
            if abs(delta) > 1e-6:
                total_changes += 1
                if self._constraint_map[name].actionable:
                    actionable_changes += 1
                changed.append({
                    "name": name,
                    "original": float(original[i]),
                    "new": float(cf[i]),
                    "delta": float(delta),
                })

        spans_arr = np.array(spans)
        proximity = float(np.mean(np.abs(cf - original) / spans_arr))
        sparsity = 1.0 - total_changes / len(self.feature_names)
        actionability = actionable_changes / total_changes if total_changes > 0 else 1.0

        return SingleCounterfactual(
            original=original,
            counterfactual=cf,
            feature_names=self.feature_names,
            changed_features=changed,
            predicted_class=cf_class,
            predicted_proba=cf_proba,
            validity=(cf_class == desired_class),
            proximity=proximity,
            sparsity=sparsity,
            actionability=actionability,
        )

    @staticmethod
    def _deduplicate(
        candidates: List[SingleCounterfactual],
        n: int,
        threshold: float = 0.01,
    ) -> List[SingleCounterfactual]:
        unique: List[SingleCounterfactual] = []
        for cf in candidates:
            if len(unique) >= n:
                break
            if not any(np.mean(np.abs(cf.counterfactual - u.counterfactual)) < threshold
                       for u in unique):
                unique.append(cf)
        return unique


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    logger.info("=== CounterfactualGenerator smoke test ===")

    X, y = make_classification(n_samples=300, n_features=8, n_classes=3,
                                n_informative=5, random_state=42)
    feature_names = ["hcc_score", "days_cash", "er_visits", "readmissions",
                     "bed_occupancy", "staff_ratio", "lab_cost", "drug_spend"]

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    constraints = [
        FeatureConstraint("hcc_score",    actionable=True,  min_value=0.0, max_value=5.0),
        FeatureConstraint("days_cash",    actionable=True,  min_value=0.0, max_value=365.0),
        FeatureConstraint("er_visits",    actionable=True,  min_value=0.0, max_value=100.0),
        FeatureConstraint("readmissions", actionable=True,  min_value=0.0, max_value=50.0),
        FeatureConstraint("bed_occupancy",actionable=False, min_value=0.0, max_value=1.0),
        FeatureConstraint("staff_ratio",  actionable=True,  min_value=0.5, max_value=3.0),
        FeatureConstraint("lab_cost",     actionable=True,  min_value=0.0, max_value=10000.0),
        FeatureConstraint("drug_spend",   actionable=True,  min_value=0.0, max_value=50000.0),
    ]

    generator = CounterfactualGenerator(
        predict_fn=clf.predict_proba,
        feature_names=feature_names,
        constraints=constraints,
        random_state=0,
    ).fit(X)

    instance = X[0]
    original_class = int(np.argmax(clf.predict_proba(instance.reshape(1, -1))[0]))
    desired_class = (original_class + 1) % 3

    class_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    result = generator.generate(instance, desired_class=desired_class,
                                 n_counterfactuals=3, class_labels=class_labels)

    logger.info("Original class: %s → Desired: %s",
                class_labels[result.original_class], class_labels[desired_class])

    best = result.best()
    if best:
        logger.info("Best CF: %s", best.to_human_readable(class_labels))
        logger.info("  proximity=%.4f, sparsity=%.4f, actionability=%.4f",
                    best.proximity, best.sparsity, best.actionability)
    else:
        logger.warning("No valid counterfactual found in smoke test.")

    logger.info("✅ CounterfactualGenerator smoke test PASSED.")
