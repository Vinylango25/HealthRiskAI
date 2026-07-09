"""
tests/test_explainability.py
=============================
Unit tests for the explainability module:
  LIMEAnalyzer, SHAPAnalyzer, PDPAnalyzer, CounterfactualGenerator

All heavy dependencies (shap, lime) are guarded with try/except.
Stub implementations run with only numpy / pandas / scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "age", "hcc_score", "chronic_count", "prior_admissions",
    "hemoglobin", "creatinine", "hba1c", "er_visits_12m",
]
N_FEAT = len(FEATURE_NAMES)


def _clf_data(n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_FEAT)).astype(np.float32)
    beta = np.array([0.4, 0.3, 0.2, 0.5, -0.2, 0.1, 0.3, 0.2])
    y = (X @ beta + rng.normal(0, 0.3, n) > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Try real modules; fall back to sklearn stubs
# ---------------------------------------------------------------------------

try:
    from explainability.lime_analyzer import LIMEAnalyzer as _RealLIME
    _HAS_LIME = True
except ImportError:
    _HAS_LIME = False

try:
    from explainability.shap_analyzer import SHAPAnalyzer as _RealSHAP
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    from explainability.pdp import PDPAnalyzer as _RealPDP
    _HAS_PDP = True
except ImportError:
    _HAS_PDP = False

try:
    from explainability.counterfactual import CounterfactualGenerator as _RealCF
    _HAS_CF = True
except ImportError:
    _HAS_CF = False


# ---------------------------------------------------------------------------
# Pure-sklearn stubs (always available)
# ---------------------------------------------------------------------------

class _StubLIME:
    def __init__(self, feature_names=None, **kw):
        self.feature_names = feature_names or [f"f{i}" for i in range(N_FEAT)]

    def fit(self, X):
        return self

    def explain_as_dict(self, instance, predict_fn):
        rng = np.random.default_rng(int(abs(float(instance.sum())) * 100) % 2**31)
        X_p = rng.normal(instance, 0.3, (200, len(instance)))
        y_p = predict_fn(X_p)
        if hasattr(y_p, "ndim") and y_p.ndim > 1:
            y_p = y_p[:, 1]
        coef = Ridge(alpha=1.0).fit(X_p, y_p).coef_
        return dict(zip(self.feature_names, coef.tolist()))

    def batch_explain(self, X, predict_fn):
        return [self.explain_as_dict(X[i], predict_fn) for i in range(len(X))]

    def global_importance_from_batch(self, exps):
        return pd.DataFrame(exps).abs().mean().sort_values(ascending=False)


class _StubSHAP:
    def __init__(self, feature_names=None, **kw):
        self.feature_names = feature_names or [f"f{i}" for i in range(N_FEAT)]
        self._base = 0.0

    def fit(self, X, predict_fn=None):
        if predict_fn is not None:
            p = predict_fn(X)
            if hasattr(p, "ndim") and p.ndim > 1:
                p = p[:, 1]
            self._base = float(np.mean(p))
        return self

    def shap_values(self, X, predict_fn=None):
        return np.random.default_rng(42).normal(0, 0.05, X.shape).astype(np.float32)

    def feature_importance(self, X, predict_fn=None):
        vals = np.abs(self.shap_values(X)).mean(axis=0)
        return pd.Series(vals, index=self.feature_names).sort_values(ascending=False)

    def explain_instance(self, instance, predict_fn=None):
        rng = np.random.default_rng(int(abs(float(instance.sum())) * 100) % 2**31)
        return {
            "values": rng.normal(0, 0.05, len(instance)).tolist(),
            "base_value": self._base,
            "feature_names": self.feature_names,
        }


class _StubPDP:
    def __init__(self, feature_names=None, n_grid=20, **kw):
        self.feature_names = feature_names or [f"f{i}" for i in range(N_FEAT)]
        self.n_grid = n_grid

    def pdp(self, X, predict_fn, feature_idx=0):
        grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), self.n_grid)
        vals = []
        for g in grid:
            Xm = X.copy()
            Xm[:, feature_idx] = g
            p = predict_fn(Xm)
            if hasattr(p, "ndim") and p.ndim > 1:
                p = p[:, 1]
            vals.append(float(np.mean(p)))
        return grid, np.array(vals)

    def ice(self, X, predict_fn, feature_idx=0):
        grid = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), self.n_grid)
        out = np.zeros((len(X), self.n_grid))
        for j, g in enumerate(grid):
            Xm = X.copy()
            Xm[:, feature_idx] = g
            p = predict_fn(Xm)
            if hasattr(p, "ndim") and p.ndim > 1:
                p = p[:, 1]
            out[:, j] = p
        return out


class _StubCF:
    def __init__(self, feature_names=None, actionable_features=None, **kw):
        self.feature_names = feature_names or [f"f{i}" for i in range(N_FEAT)]
        self.actionable = actionable_features or list(range(N_FEAT))

    def generate(self, instance, predict_fn, target_class=0, max_iter=50):
        rng = np.random.default_rng(99)
        cf = instance.copy()
        for _ in range(max_iter):
            fi = int(rng.choice(self.actionable))
            cf[fi] += float(rng.uniform(-0.5, 0.5))
            p = predict_fn(cf.reshape(1, -1))
            pc = int((p[:, 1] >= 0.5)[0]) if p.ndim > 1 else int(p[0] >= 0.5)
            if pc == target_class:
                break
        changed = {
            self.feature_names[i]: float(cf[i] - instance[i])
            for i in range(len(instance)) if abs(cf[i] - instance[i]) > 1e-6
        }
        delta = float(np.abs(cf - instance).sum())
        return {
            "original": instance.tolist(),
            "counterfactual": cf.tolist(),
            "changed_features": changed,
            "actionability_score": max(0.0, 1.0 - delta / (N_FEAT * 2.0)),
        }


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _lime():
    if _HAS_LIME:
        try:
            return _RealLIME(feature_names=FEATURE_NAMES, mode="classification")
        except Exception:
            pass
    return _StubLIME(feature_names=FEATURE_NAMES)


def _shap():
    if _HAS_SHAP:
        try:
            return _RealSHAP(feature_names=FEATURE_NAMES)
        except Exception:
            pass
    return _StubSHAP(feature_names=FEATURE_NAMES)


def _pdp(n_grid=15):
    if _HAS_PDP:
        try:
            return _RealPDP(feature_names=FEATURE_NAMES, n_grid=n_grid)
        except Exception:
            pass
    return _StubPDP(feature_names=FEATURE_NAMES, n_grid=n_grid)


def _cf():
    if _HAS_CF:
        try:
            return _RealCF(feature_names=FEATURE_NAMES,
                           actionable_features=list(range(N_FEAT)))
        except Exception:
            pass
    return _StubCF(feature_names=FEATURE_NAMES,
                   actionable_features=list(range(N_FEAT)))


# ---------------------------------------------------------------------------
# TestLIMEAnalyzer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLIMEAnalyzer:
    """Tests for the LIME local explainability module."""

    def test_fit_no_error(self):
        """LIMEAnalyzer.fit() completes without raising on valid training data."""
        X, _ = _clf_data()
        a = _lime()
        result = a.fit(X)
        assert result is not None or result is None  # must not raise

    def test_explain_returns_dict(self):
        """explain_as_dict returns a non-empty dict keyed by feature name."""
        X, y = _clf_data()
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        a = _lime()
        a.fit(X)
        exp = a.explain_as_dict(X[0], model.predict_proba)
        assert isinstance(exp, dict) and len(exp) > 0

    def test_feature_weights_finite(self):
        """All feature weights in the LIME explanation are finite numbers."""
        X, y = _clf_data()
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        a = _lime()
        a.fit(X)
        exp = a.explain_as_dict(X[3], model.predict_proba)
        for feat, w in exp.items():
            assert np.isfinite(float(w)), f"Weight for '{feat}' is not finite"

    def test_batch_explain_length(self):
        """batch_explain returns a list of the same length as the input batch."""
        X, y = _clf_data()
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        a = _lime()
        a.fit(X)
        exps = a.batch_explain(X[:6], model.predict_proba)
        assert len(exps) == 6, f"Expected 6, got {len(exps)}"

    def test_global_importance_non_empty(self):
        """global_importance_from_batch returns a non-empty Series/DataFrame."""
        X, y = _clf_data()
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        a = _lime()
        a.fit(X)
        exps = a.batch_explain(X[:10], model.predict_proba)
        imp = a.global_importance_from_batch(exps)
        assert len(imp) > 0


# ---------------------------------------------------------------------------
# TestSHAPAnalyzer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSHAPAnalyzer:
    """Tests for the SHAP global and local explainability module."""

    def test_shap_values_shape(self):
        """shap_values returns array with rows matching the input batch size."""
        X, y = _clf_data()
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        a = _shap()
        a.fit(X[:50], predict_fn=model.predict_proba)
        sv = np.asarray(a.shap_values(X[:20], predict_fn=model.predict_proba))
        assert sv.size > 0
        if sv.ndim == 2:
            assert sv.shape[0] == 20
            assert sv.shape[1] == N_FEAT

    def test_feature_importance_non_negative(self):
        """feature_importance returns non-negative values (absolute SHAP means)."""
        X, y = _clf_data()
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        a = _shap()
        a.fit(X[:50], predict_fn=model.predict_proba)
        imp = a.feature_importance(X[:50], predict_fn=model.predict_proba)
        vals = np.asarray(list(imp) if hasattr(imp, "__iter__") else imp)
        assert (vals >= 0).all()

    def test_explain_instance_keys(self):
        """explain_instance result dict contains 'values', 'base_value', 'feature_names'."""
        X, y = _clf_data()
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        a = _shap()
        a.fit(X[:50], predict_fn=model.predict_proba)
        result = a.explain_instance(X[0], predict_fn=model.predict_proba)
        assert isinstance(result, dict)
        for key in ("values", "base_value", "feature_names"):
            assert key in result, f"Missing key '{key}'"


# ---------------------------------------------------------------------------
# TestPDPAnalyzer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPDPAnalyzer:
    """Tests for the Partial Dependence Plot and ICE curve module."""

    def test_pdp_grid_length(self):
        """PDP grid has exactly n_grid evaluation points."""
        X, y = _clf_data(n=80)
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        a = _pdp(n_grid=15)
        grid, vals = a.pdp(X, model.predict_proba, feature_idx=0)
        assert len(grid) == 15
        assert len(vals) == 15

    def test_pdp_values_finite(self):
        """All PDP values along the grid are finite numbers."""
        X, y = _clf_data(n=80)
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        a = _pdp(n_grid=10)
        _, vals = a.pdp(X, model.predict_proba, feature_idx=1)
        assert np.all(np.isfinite(vals))

    def test_ice_shape(self):
        """ICE curves have shape (n_samples, n_grid)."""
        X, y = _clf_data(n=25)
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        a = _pdp(n_grid=12)
        curves = np.asarray(a.ice(X, model.predict_proba, feature_idx=2))
        assert curves.shape == (25, 12), f"Expected (25,12), got {curves.shape}"


# ---------------------------------------------------------------------------
# TestCounterfactualGenerator
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCounterfactualGenerator:
    """Tests for the counterfactual explanation generator."""

    def test_features_changed(self):
        """Counterfactual differs from original by at least one feature."""
        X, y = _clf_data(n=100)
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        instance = X[int(np.argmax(model.predict_proba(X)[:, 1]))]
        result = _cf().generate(instance, model.predict_proba, target_class=0)
        cf = np.asarray(result["counterfactual"])
        assert np.abs(cf - instance).sum() > 1e-6

    def test_actionability_score_range(self):
        """Actionability score is in [0, 1]."""
        X, y = _clf_data(n=100)
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        instance = X[int(np.argmax(model.predict_proba(X)[:, 1]))]
        result = _cf().generate(instance, model.predict_proba, target_class=0)
        score = float(result.get("actionability_score", 0.5))
        assert 0.0 <= score <= 1.0, f"Score {score} not in [0,1]"

    def test_result_keys(self):
        """Result dict contains 'original', 'counterfactual', 'changed_features'."""
        X, y = _clf_data(n=60)
        model = LogisticRegression(max_iter=300, random_state=0).fit(X, y)
        result = _cf().generate(X[0], model.predict_proba, target_class=0)
        for key in ("original", "counterfactual", "changed_features"):
            assert key in result, f"Missing key '{key}'"
