"""
models/clinical_nlp/ner_pipeline.py
======================================
medspaCy NER pipeline for clinical notes.

Extracts:
  - Medications (drug names, dosages, routes, frequencies)
  - Conditions (diagnoses, symptoms, ICD-10 concepts)
  - Procedures (surgical, diagnostic, therapeutic)
  - Lab values (numeric results with units and reference ranges)
  - Negations and uncertainty (negated/uncertain entities marked)

Target: F1 > 0.70 on clinical NER benchmarks (i2b2, n2c2)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]


# ─── Entity dataclass ─────────────────────────────────────────────────────────


@dataclass
class ClinicalEntity:
    """Represents a single extracted clinical entity."""

    text: str
    entity_type: str          # MEDICATION | CONDITION | PROCEDURE | LAB_VALUE
    start: int
    end: int
    negated: bool = False
    uncertain: bool = False
    value: Optional[str] = None      # numeric value for lab results
    unit: Optional[str] = None       # unit for lab / dosage
    normalized: Optional[str] = None # UMLS/RxNorm concept (if available)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "negated": self.negated,
            "uncertain": self.uncertain,
            "value": self.value,
            "unit": self.unit,
            "normalized": self.normalized,
            "confidence": self.confidence,
        }


# ─── Regex patterns for rule-based extraction ────────────────────────────────

# Medication patterns
MED_PATTERNS = [
    r"\b(aspirin|metformin|lisinopril|atorvastatin|amlodipine|omeprazole|"
    r"metoprolol|losartan|albuterol|furosemide|warfarin|heparin|insulin|"
    r"vancomycin|ceftriaxone|piperacillin|amoxicillin|azithromycin|"
    r"prednisone|dexamethasone|morphine|oxycodone|hydrocodone|acetaminophen|"
    r"ibuprofen|naproxen|gabapentin|sertraline|escitalopram|quetiapine|"
    r"lorazepam|diazepam|haloperidol|ondansetron|pantoprazole|metronidazole|"
    r"ciprofloxacin|levofloxacin|clindamycin|gentamicin|amiodarone|"
    r"atenolol|carvedilol|spironolactone|digoxin|colchicine|allopurinol)\b",
]

# Lab value pattern: "sodium 135 mEq/L" or "WBC: 8.5 k/uL"
LAB_PATTERN = re.compile(
    r"\b(sodium|potassium|chloride|bicarbonate|bun|creatinine|glucose|"
    r"hemoglobin|hematocrit|wbc|platelets|inr|pt|ptt|"
    r"alt|ast|alkaline phosphatase|bilirubin|albumin|"
    r"troponin|bnp|nt-probnp|lactate|ph|pco2|po2|"
    r"tsh|t4|hba1c|ldl|hdl|triglycerides|cholesterol|"
    r"calcium|magnesium|phosphorus|uric acid)"
    r"\s*[:\s]\s*"
    r"(\d+\.?\d*)\s*"
    r"(mg\/dl|meq\/l|mmol\/l|g\/dl|k\/ul|u\/l|iu\/l|ng\/ml|"
    r"pg\/ml|mcg\/dl|miu\/ml|%|mm hg|mmhg)?",
    re.IGNORECASE,
)

# Negation cues
NEGATION_CUES = [
    "no ", "not ", "without ", "denies ", "denied ", "negative for ",
    "no evidence of ", "no history of ", "absent ", "rules out ",
    "ruled out ", "free of ", "none "
]

# Uncertainty cues
UNCERTAINTY_CUES = [
    "possible ", "possible ", "probable ", "likely ", "unlikely ",
    "may ", "might ", "could ", "suggest ", "consistent with ",
    "concern for ", "cannot rule out ", "question of "
]

# Condition patterns (ICD-10 categories)
CONDITION_PATTERNS = [
    r"\b(hypertension|diabetes mellitus|type 2 diabetes|heart failure|"
    r"atrial fibrillation|coronary artery disease|chronic kidney disease|"
    r"sepsis|pneumonia|urinary tract infection|deep vein thrombosis|"
    r"pulmonary embolism|myocardial infarction|stroke|copd|asthma|"
    r"anemia|hypothyroidism|hyperthyroidism|depression|anxiety|"
    r"chronic pain|obesity|hyponatremia|hyperkalemia|acute kidney injury|"
    r"liver cirrhosis|hepatitis|pancreatitis|appendicitis|"
    r"breast cancer|lung cancer|colon cancer|prostate cancer|"
    r"alzheimer|dementia|parkinson|multiple sclerosis|"
    r"rheumatoid arthritis|systemic lupus|fibromyalgia)\b",
]

# Procedure patterns
PROCEDURE_PATTERNS = [
    r"\b(intubation|mechanical ventilation|central line|foley catheter|"
    r"echocardiogram|electrocardiogram|ekg|ecg|ct scan|mri|x-ray|"
    r"ultrasound|colonoscopy|endoscopy|bronchoscopy|"
    r"surgery|appendectomy|cholecystectomy|colectomy|"
    r"angioplasty|stent|bypass|pacemaker|defibrillator|"
    r"dialysis|hemodialysis|peritoneal dialysis|"
    r"biopsy|lumbar puncture|thoracentesis|paracentesis|"
    r"blood transfusion|platelet transfusion|"
    r"physical therapy|occupational therapy|speech therapy)\b",
]


# ─── NER Pipeline ─────────────────────────────────────────────────────────────


class ClinicalNERPipeline:
    """
    Clinical NER pipeline with two modes:
      1. medspaCy (preferred): Uses medspaCy + scispaCy models with UMLS linking
      2. Rule-based fallback: Regex + custom rules when medspaCy unavailable

    Usage
    -----
    ner = ClinicalNERPipeline()
    entities = ner.extract(note_text)
    df = ner.extract_batch(list_of_notes)
    """

    def __init__(
        self,
        use_medspacy: bool = True,
        use_scispacy: bool = False,
        umls_linker: bool = False,
        negation_window: int = 5,  # tokens to look back for negation
    ) -> None:
        self.negation_window = negation_window
        self.nlp: Optional[Any] = None
        self._mode: str = "rule_based"

        if use_medspacy:
            self._try_load_medspacy(umls_linker)
        if self.nlp is None and use_scispacy:
            self._try_load_scispacy()

        # Compile regex patterns
        self._med_regex = re.compile(
            "|".join(MED_PATTERNS), re.IGNORECASE
        )
        self._cond_regex = re.compile(
            "|".join(CONDITION_PATTERNS), re.IGNORECASE
        )
        self._proc_regex = re.compile(
            "|".join(PROCEDURE_PATTERNS), re.IGNORECASE
        )
        logger.info("ClinicalNERPipeline initialised in mode: %s", self._mode)

    def _try_load_medspacy(self, umls_linker: bool) -> None:
        try:
            import medspacy
            self.nlp = medspacy.load()

            if umls_linker:
                try:
                    from scispacy.linking import EntityLinker
                    self.nlp.add_pipe(
                        "scispacy_linker",
                        config={"resolve_abbreviations": True, "linker_name": "umls"},
                    )
                except ImportError:
                    logger.warning("scispacy not available — skipping UMLS linking")

            self._mode = "medspacy"
            logger.info("medspaCy loaded successfully")
        except ImportError:
            logger.info("medspaCy not installed — using rule-based NER")

    def _try_load_scispacy(self) -> None:
        try:
            import spacy
            self.nlp = spacy.load("en_core_sci_md")
            self._mode = "scispacy"
            logger.info("scispaCy en_core_sci_md loaded")
        except Exception:
            logger.info("scispaCy model not available — using rule-based NER")

    # ── Core extraction ──────────────────────────────────────────────────────

    def extract(self, text: str) -> List[ClinicalEntity]:
        """Extract all clinical entities from a note string."""
        if self.nlp is not None and self._mode in ("medspacy", "scispacy"):
            return self._extract_spacy(text)
        return self._extract_rule_based(text)

    def _extract_spacy(self, text: str) -> List[ClinicalEntity]:
        """Use spaCy/medspaCy pipeline for NER."""
        doc = self.nlp(text)
        entities: List[ClinicalEntity] = []

        for ent in doc.ents:
            # Map spaCy label to our entity types
            label = self._map_spacy_label(ent.label_)
            if label is None:
                continue

            negated = False
            uncertain = False

            # Check medspaCy context attributes
            if hasattr(ent._, "is_negated"):
                negated = bool(ent._.is_negated)
            if hasattr(ent._, "is_uncertain"):
                uncertain = bool(ent._.is_uncertain)
            if not negated:
                negated = self._check_negation_rule(text, ent.start_char)
            if not uncertain:
                uncertain = self._check_uncertainty_rule(text, ent.start_char)

            entity = ClinicalEntity(
                text=ent.text,
                entity_type=label,
                start=ent.start_char,
                end=ent.end_char,
                negated=negated,
                uncertain=uncertain,
            )
            entities.append(entity)

        # Supplement with lab value regex
        entities += self._extract_lab_values(text)
        return entities

    def _map_spacy_label(self, label: str) -> Optional[str]:
        mapping = {
            "CHEMICAL": "MEDICATION",
            "DRUG": "MEDICATION",
            "TREATMENT": "MEDICATION",
            "DISEASE": "CONDITION",
            "DISORDER": "CONDITION",
            "SIGN_SYMPTOM": "CONDITION",
            "PROCEDURE": "PROCEDURE",
            "TEST": "PROCEDURE",
            "LAB": "LAB_VALUE",
        }
        return mapping.get(label.upper())

    def _extract_rule_based(self, text: str) -> List[ClinicalEntity]:
        """Regex + rule-based NER fallback."""
        entities: List[ClinicalEntity] = []

        for m in self._med_regex.finditer(text):
            neg = self._check_negation_rule(text, m.start())
            unc = self._check_uncertainty_rule(text, m.start())
            entities.append(ClinicalEntity(
                text=m.group(),
                entity_type="MEDICATION",
                start=m.start(),
                end=m.end(),
                negated=neg,
                uncertain=unc,
            ))

        for m in self._cond_regex.finditer(text):
            neg = self._check_negation_rule(text, m.start())
            unc = self._check_uncertainty_rule(text, m.start())
            entities.append(ClinicalEntity(
                text=m.group(),
                entity_type="CONDITION",
                start=m.start(),
                end=m.end(),
                negated=neg,
                uncertain=unc,
            ))

        for m in self._proc_regex.finditer(text):
            entities.append(ClinicalEntity(
                text=m.group(),
                entity_type="PROCEDURE",
                start=m.start(),
                end=m.end(),
                negated=self._check_negation_rule(text, m.start()),
            ))

        entities += self._extract_lab_values(text)
        return sorted(entities, key=lambda e: e.start)

    def _extract_lab_values(self, text: str) -> List[ClinicalEntity]:
        entities = []
        for m in LAB_PATTERN.finditer(text):
            groups = m.groups()
            lab_name = groups[0]
            value = groups[1] if len(groups) > 1 else None
            unit = groups[2] if len(groups) > 2 else None
            entities.append(ClinicalEntity(
                text=m.group(),
                entity_type="LAB_VALUE",
                start=m.start(),
                end=m.end(),
                value=value,
                unit=unit,
                normalized=lab_name.lower() if lab_name else None,
            ))
        return entities

    def _check_negation_rule(self, text: str, start: int) -> bool:
        """Check if entity at position is negated by a preceding cue."""
        window_start = max(0, start - 60)
        preceding = text[window_start:start].lower()
        return any(cue in preceding for cue in NEGATION_CUES)

    def _check_uncertainty_rule(self, text: str, start: int) -> bool:
        """Check if entity at position is uncertain."""
        window_start = max(0, start - 60)
        preceding = text[window_start:start].lower()
        return any(cue in preceding for cue in UNCERTAINTY_CUES)

    # ── Batch processing ─────────────────────────────────────────────────────

    def extract_batch(
        self,
        texts: List[str],
        note_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Extract entities from a list of clinical notes.
        Returns a long-format DataFrame with one row per entity.
        """
        rows = []
        for i, text in enumerate(texts):
            note_id = note_ids[i] if note_ids else str(i)
            entities = self.extract(text)
            for ent in entities:
                row = ent.to_dict()
                row["note_id"] = note_id
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=[
                "note_id", "text", "entity_type", "start", "end",
                "negated", "uncertain", "value", "unit", "normalized", "confidence"
            ])
        return pd.DataFrame(rows)

    def extract_structured(
        self, text: str
    ) -> Dict[str, Any]:
        """
        Return a structured dict with entity lists by type,
        suitable for feature engineering.
        """
        entities = self.extract(text)

        result: Dict[str, Any] = {
            "medications": [],
            "conditions": [],
            "procedures": [],
            "lab_values": [],
            "negated_conditions": [],
            "uncertain_conditions": [],
        }

        for ent in entities:
            if ent.entity_type == "MEDICATION" and not ent.negated:
                result["medications"].append(ent.text.lower())
            elif ent.entity_type == "CONDITION":
                if ent.negated:
                    result["negated_conditions"].append(ent.text.lower())
                elif ent.uncertain:
                    result["uncertain_conditions"].append(ent.text.lower())
                else:
                    result["conditions"].append(ent.text.lower())
            elif ent.entity_type == "PROCEDURE" and not ent.negated:
                result["procedures"].append(ent.text.lower())
            elif ent.entity_type == "LAB_VALUE":
                result["lab_values"].append({
                    "name": ent.normalized,
                    "value": ent.value,
                    "unit": ent.unit,
                })

        # Dedup
        for key in ["medications", "conditions", "procedures", "negated_conditions", "uncertain_conditions"]:
            result[key] = list(dict.fromkeys(result[key]))

        return result

    def summarise_note(self, text: str) -> pd.Series:
        """
        Summarise a single note as a feature vector:
          - medication count, condition count, procedure count
          - negation count, uncertainty count
          - key lab presence flags
        """
        structured = self.extract_structured(text)
        return pd.Series({
            "n_medications": len(structured["medications"]),
            "n_conditions": len(structured["conditions"]),
            "n_procedures": len(structured["procedures"]),
            "n_negated": len(structured["negated_conditions"]),
            "n_uncertain": len(structured["uncertain_conditions"]),
            "n_lab_values": len(structured["lab_values"]),
            "has_sepsis_flag": "sepsis" in str(structured["conditions"]),
            "has_cancer_flag": any(
                "cancer" in c or "carcinoma" in c for c in structured["conditions"]
            ),
            "has_heart_failure": "heart failure" in str(structured["conditions"]),
            "has_diabetes": "diabetes" in str(structured["conditions"]),
        })


# ─── Evaluation helpers ───────────────────────────────────────────────────────


def evaluate_ner(
    predicted: List[List[Tuple[int, int, str]]],  # [(start, end, type), ...]
    gold: List[List[Tuple[int, int, str]]],
    entity_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute P/R/F1 for NER predictions against gold standard.
    Uses exact span matching.
    """
    entity_types = entity_types or ["MEDICATION", "CONDITION", "PROCEDURE", "LAB_VALUE"]
    results: Dict[str, Any] = {}

    for etype in entity_types:
        tp = fp = fn = 0
        for pred_spans, gold_spans in zip(predicted, gold):
            pred_set = {(s, e) for s, e, t in pred_spans if t == etype}
            gold_set = {(s, e) for s, e, t in gold_spans if t == etype}
            tp += len(pred_set & gold_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        results[f"{etype.lower()}_precision"] = precision
        results[f"{etype.lower()}_recall"] = recall
        results[f"{etype.lower()}_f1"] = f1

    # Macro average F1
    f1_scores = [results[k] for k in results if k.endswith("_f1")]
    results["macro_f1"] = float(sum(f1_scores) / max(len(f1_scores), 1))
    return results


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_note = """
    DISCHARGE SUMMARY

    Patient is a 67-year-old male admitted for acute exacerbation of heart failure.
    He has a history of hypertension, type 2 diabetes mellitus, and coronary artery disease.

    Medications on admission:
    - Metoprolol 25mg twice daily
    - Furosemide 40mg daily
    - Lisinopril 10mg daily
    - Insulin glargine 20 units at bedtime
    - Aspirin 81mg daily
    - Atorvastatin 40mg at bedtime

    Labs: sodium 130 mEq/L, creatinine 1.8 mg/dL, BNP 1240 pg/mL, hemoglobin 10.2 g/dL.

    No evidence of pneumonia or pulmonary embolism.
    The patient denies chest pain. Possible urinary tract infection.

    Procedures: echocardiogram performed showing EF 30%.
    """

    ner = ClinicalNERPipeline(use_medspacy=False)  # rule-based for smoke test
    entities = ner.extract(sample_note)

    print(f"\nExtracted {len(entities)} entities:")
    for e in entities:
        flag = " [NEGATED]" if e.negated else ""
        flag += " [UNCERTAIN]" if e.uncertain else ""
        print(f"  [{e.entity_type:15s}] {e.text!r}{flag}")

    structured = ner.extract_structured(sample_note)
    print(f"\nMedications ({len(structured['medications'])}): {structured['medications']}")
    print(f"Conditions ({len(structured['conditions'])}): {structured['conditions']}")
    print(f"Negated conditions: {structured['negated_conditions']}")
    print(f"Lab values: {structured['lab_values'][:3]}")

    summary = ner.summarise_note(sample_note)
    print(f"\nNote summary:\n{summary}")

    assert len(structured["medications"]) >= 3, "Expected at least 3 medications"
    assert len(structured["conditions"]) >= 2, "Expected at least 2 conditions"
    assert len(structured["negated_conditions"]) >= 1, "Expected negated entities"
    logger.info("=== PASS ===")
