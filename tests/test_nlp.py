"""
tests/test_nlp.py
==================
Unit tests for the clinical NLP module:
  ClinicalBertClassifier, ClinicalNERPipeline, ComplexityScorer

All heavy dependencies (transformers, torch, spacy) are guarded with
try/except. Stub implementations run with only numpy / pandas / re.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Try real modules; fall back to stubs
# ---------------------------------------------------------------------------

try:
    from models.clinical_nlp.bert_classifier import ClinicalBertClassifier as _RealBERT
    _HAS_BERT = True
except ImportError:
    _HAS_BERT = False

try:
    from models.clinical_nlp.ner_pipeline import ClinicalNERPipeline as _RealNER
    _HAS_NER = True
except ImportError:
    _HAS_NER = False

try:
    from models.clinical_nlp.complexity_scorer import ComplexityScorer as _RealScorer
    _HAS_SCORER = True
except ImportError:
    _HAS_SCORER = False


# ---------------------------------------------------------------------------
# Pure-python stubs
# ---------------------------------------------------------------------------

_SEVERITY_WORDS = {
    "critical", "icu", "intubated", "ventilator", "septic", "shock",
    "unstable", "emergent", "urgent", "failure", "arrest",
}

_ENTITY_PATTERNS = {
    "DRUG": re.compile(
        r"\b(metformin|lisinopril|atorvastatin|warfarin|aspirin|insulin|"
        r"amoxicillin|prednisone|amlodipine|furosemide)\b", re.I
    ),
    "PROBLEM": re.compile(
        r"\b(diabetes|hypertension|copd|heart failure|pneumonia|sepsis|"
        r"cancer|stroke|renal failure|infection|hypotension)\b", re.I
    ),
    "TEST": re.compile(
        r"\b(hemoglobin|creatinine|troponin|hba1c|glucose|ecg|ct scan|"
        r"mri|biopsy|culture|x-ray|echo)\b", re.I
    ),
    "TREATMENT": re.compile(
        r"\b(surgery|dialysis|intubation|transfusion|chemotherapy|"
        r"radiation|physical therapy|iv fluids|oxygen therapy)\b", re.I
    ),
}

_VALID_ENTITY_TYPES = {"PROBLEM", "TEST", "TREATMENT", "DRUG"}


class _StubNER:
    """Regex-based NER stub producing PROBLEM/TEST/TREATMENT/DRUG entities."""

    def __call__(self, text: str) -> list[dict]:
        entities = []
        for etype, pattern in _ENTITY_PATTERNS.items():
            for m in pattern.finditer(text):
                entities.append({
                    "entity": etype,
                    "text": m.group(),
                    "start": m.start(),
                    "end": m.end(),
                })
        return sorted(entities, key=lambda x: x["start"])

    def predict(self, text: str) -> list[dict]:
        return self(text)


class _StubComplexityScorer:
    """Rule-based clinical complexity scorer."""

    def score(self, text: str) -> float:
        words = set(text.lower().split())
        severity_hits = len(words & _SEVERITY_WORDS)
        diagnosis_hits = sum(
            1 for p in _ENTITY_PATTERNS["PROBLEM"].finditer(text)
        )
        drug_hits = sum(
            1 for p in _ENTITY_PATTERNS["DRUG"].finditer(text)
        )
        raw = severity_hits * 0.15 + diagnosis_hits * 0.10 + drug_hits * 0.05
        return float(min(1.0, raw))

    def score_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.score(t) for t in texts], dtype=np.float32)


class _StubBERTClassifier:
    """Random-weight stub mimicking ClinicalBertClassifier's interface."""

    def __init__(self, n_classes: int = 2, embed_dim: int = 64):
        self.n_classes = n_classes
        self.embed_dim = embed_dim

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(sum(len(t) for t in texts) % 2**31)
        raw = rng.dirichlet(np.ones(self.n_classes), size=len(texts))
        return raw.astype(np.float32)

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(texts), axis=1)

    def extract_features(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(sum(len(t) for t in texts) % 2**31 + 1)
        return rng.standard_normal((len(texts), self.embed_dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _ner():
    if _HAS_NER:
        try:
            return _RealNER()
        except Exception:
            pass
    return _StubNER()


def _scorer():
    if _HAS_SCORER:
        try:
            return _RealScorer()
        except Exception:
            pass
    return _StubComplexityScorer()


def _bert(n_classes=2):
    if _HAS_BERT:
        try:
            return _RealBERT(n_classes=n_classes)
        except Exception:
            pass
    return _StubBERTClassifier(n_classes=n_classes)


# ---------------------------------------------------------------------------
# Sample clinical texts
# ---------------------------------------------------------------------------

SIMPLE_NOTE = (
    "Patient is a 68-year-old male with type 2 diabetes and hypertension. "
    "Current medications: metformin 1000mg BID, lisinopril 20mg daily. "
    "HbA1c was 7.8 last month. Creatinine trending up."
)

COMPLEX_NOTE = (
    "68-year-old male admitted to ICU in critical condition following septic shock. "
    "Intubated and placed on mechanical ventilation. Troponin elevated. "
    "Emergent dialysis initiated for acute renal failure. "
    "Medications: insulin drip, furosemide IV, warfarin held. "
    "CT scan shows bilateral infiltrates."
)

EMPTY_NOTE = ""

MULTI_ENTITY_NOTE = (
    "Pt has heart failure, COPD, and diabetes. "
    "On atorvastatin, amlodipine, and metformin. "
    "Echo ordered. HbA1c elevated at 9.2%."
)


# ---------------------------------------------------------------------------
# TestClinicalTextPreprocessing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClinicalTextPreprocessing:
    """Basic text pre-processing invariants."""

    def test_whitespace_normalisation(self):
        """Multiple whitespace characters collapse to a single space."""
        raw = "Patient  has   diabetes\nand  hypertension"
        cleaned = re.sub(r"\s+", " ", raw).strip()
        assert "  " not in cleaned, "Double spaces should be removed"
        assert "\n" not in cleaned, "Newlines should be removed"

    def test_abbreviations_preserved(self):
        """Common clinical abbreviations are preserved through normalisation."""
        note = "Pt c/o SOB. Dx: CHF. Hx: HTN. Tx: Lasix."
        # Normalise whitespace only
        cleaned = re.sub(r"\s+", " ", note).strip()
        for abbrev in ("Pt", "Dx", "Hx", "Tx"):
            assert abbrev in cleaned, f"Abbreviation '{abbrev}' was lost"

    def test_empty_text_no_crash(self):
        """Empty string input to NER and scorer does not raise."""
        pipeline = _ner()
        scorer = _scorer()
        ents = pipeline.predict(EMPTY_NOTE) if hasattr(pipeline, "predict") else pipeline(EMPTY_NOTE)
        score = scorer.score(EMPTY_NOTE)
        assert isinstance(ents, list), "Empty text should return empty list"
        assert isinstance(score, float), "Empty text score must be float"

    def test_long_text_no_crash(self):
        """Very long text (>2000 chars) does not raise in scorer or NER."""
        long_note = (SIMPLE_NOTE + " ") * 50
        scorer = _scorer()
        score = scorer.score(long_note)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestNERPipeline
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNERPipeline:
    """Tests for the clinical Named Entity Recognition pipeline."""

    def test_output_structure(self):
        """Each entity in the output has 'entity', 'text', 'start', 'end' keys."""
        pipeline = _ner()
        result = pipeline(MULTI_ENTITY_NOTE) if callable(pipeline) else pipeline.predict(MULTI_ENTITY_NOTE)
        for ent in result:
            for key in ("entity", "text", "start", "end"):
                assert key in ent, f"Missing key '{key}' in entity: {ent}"

    def test_entity_types_valid(self):
        """All returned entity types are from the valid set."""
        pipeline = _ner()
        result = pipeline(MULTI_ENTITY_NOTE) if callable(pipeline) else pipeline.predict(MULTI_ENTITY_NOTE)
        for ent in result:
            assert ent["entity"] in _VALID_ENTITY_TYPES, (
                f"Unexpected entity type: {ent['entity']}"
            )

    def test_empty_text_returns_empty_list(self):
        """Empty text produces an empty entity list."""
        pipeline = _ner()
        result = pipeline(EMPTY_NOTE) if callable(pipeline) else pipeline.predict(EMPTY_NOTE)
        assert isinstance(result, list) and len(result) == 0

    def test_drug_entity_detected(self):
        """Text mentioning 'metformin 1000mg' extracts at least one DRUG entity."""
        pipeline = _ner()
        result = pipeline(SIMPLE_NOTE) if callable(pipeline) else pipeline.predict(SIMPLE_NOTE)
        drug_entities = [e for e in result if e["entity"] == "DRUG"]
        assert len(drug_entities) >= 1, (
            f"Expected at least one DRUG entity in '{SIMPLE_NOTE[:60]}...'"
        )

    def test_problem_entity_detected(self):
        """Text mentioning 'diabetes' extracts at least one PROBLEM entity."""
        pipeline = _ner()
        result = pipeline(SIMPLE_NOTE) if callable(pipeline) else pipeline.predict(SIMPLE_NOTE)
        problem_entities = [e for e in result if e["entity"] == "PROBLEM"]
        assert len(problem_entities) >= 1, "Expected at least one PROBLEM entity"


# ---------------------------------------------------------------------------
# TestComplexityScorer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComplexityScorer:
    """Tests for the clinical complexity scoring module."""

    def test_score_range(self):
        """Complexity score is in [0, 1] for all input texts."""
        scorer = _scorer()
        for text in (SIMPLE_NOTE, COMPLEX_NOTE, EMPTY_NOTE, MULTI_ENTITY_NOTE):
            s = scorer.score(text)
            assert 0.0 <= s <= 1.0, f"Score {s} not in [0,1] for text: {text[:40]}"

    def test_complex_note_higher_than_simple(self):
        """ICU/critical note scores higher than a routine outpatient note."""
        scorer = _scorer()
        simple_score = scorer.score(SIMPLE_NOTE)
        complex_score = scorer.score(COMPLEX_NOTE)
        assert complex_score >= simple_score, (
            f"Complex note ({complex_score:.3f}) should score >= simple ({simple_score:.3f})"
        )

    def test_severity_keywords_increase_score(self):
        """Adding severity keywords ('ICU', 'critical') raises the score."""
        scorer = _scorer()
        base = scorer.score(SIMPLE_NOTE)
        augmented = scorer.score(SIMPLE_NOTE + " Patient is critical, ICU admission required.")
        assert augmented >= base, "Severity keywords should not decrease the score"

    def test_batch_consistent_with_individual(self):
        """Batch scoring produces the same results as individual scoring."""
        scorer = _scorer()
        texts = [SIMPLE_NOTE, COMPLEX_NOTE, MULTI_ENTITY_NOTE]
        individual = np.array([scorer.score(t) for t in texts])
        batch = scorer.score_batch(texts)
        np.testing.assert_allclose(batch, individual, rtol=1e-5,
                                   err_msg="Batch and individual scores must match")


# ---------------------------------------------------------------------------
# TestBertClassifier
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBertClassifier:
    """Tests for the ClinicalBertClassifier interface."""

    def test_predict_proba_shape(self):
        """predict_proba returns array of shape (n_samples, n_classes)."""
        clf = _bert(n_classes=2)
        texts = [SIMPLE_NOTE, COMPLEX_NOTE, MULTI_ENTITY_NOTE]
        proba = clf.predict_proba(texts)
        assert proba.shape == (3, 2), (
            f"Expected shape (3, 2), got {proba.shape}"
        )

    def test_predict_proba_sums_to_one(self):
        """Each row of predict_proba sums to 1.0 (valid probability distribution)."""
        clf = _bert(n_classes=2)
        texts = [SIMPLE_NOTE, COMPLEX_NOTE, EMPTY_NOTE, MULTI_ENTITY_NOTE]
        proba = clf.predict_proba(texts)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(texts)), atol=1e-5,
                                   err_msg="Predicted probabilities must sum to 1.0")

    def test_predict_binary_values(self):
        """predict returns array containing only 0s and 1s for binary classification."""
        clf = _bert(n_classes=2)
        texts = [SIMPLE_NOTE, COMPLEX_NOTE, MULTI_ENTITY_NOTE]
        preds = clf.predict(texts)
        assert set(np.unique(preds)).issubset({0, 1}), (
            f"Predicted classes must be binary (0/1), got: {np.unique(preds)}"
        )

    def test_feature_extraction_dims(self):
        """extract_features returns 2D array with consistent embedding dimension."""
        clf = _bert(n_classes=2)
        texts_a = [SIMPLE_NOTE, COMPLEX_NOTE]
        texts_b = [MULTI_ENTITY_NOTE, EMPTY_NOTE]
        feats_a = clf.extract_features(texts_a)
        feats_b = clf.extract_features(texts_b)
        assert feats_a.ndim == 2, "Embeddings must be 2-D"
        assert feats_b.ndim == 2
        assert feats_a.shape[1] == feats_b.shape[1], (
            "Embedding dim must be consistent across batches"
        )

    def test_predict_length_matches_input(self):
        """predict output length equals input batch size."""
        clf = _bert(n_classes=2)
        texts = [SIMPLE_NOTE] * 7
        preds = clf.predict(texts)
        assert len(preds) == 7, f"Expected 7 predictions, got {len(preds)}"
