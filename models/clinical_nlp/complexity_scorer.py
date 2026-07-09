"""
models/clinical_nlp/complexity_scorer.py
==========================================
Clinical complexity scorer using BERT note embeddings + NER features.

Predicts:
  - Expected cost quintile (1-5) from discharge notes
  - Clinical complexity tier (low / medium / high / very_high)
  - 90-day readmission risk score

Features:
  - [CLS] embedding from ClinicalBERT (768-dim)
  - NER-derived features (med count, condition count, negation density)
  - Optional: structured EHR features (diagnoses count, LOS, age)

Model: Ridge regression / Light MLP on top of frozen BERT embeddings
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .ner_pipeline import ClinicalNERPipeline

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "reports" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLINICALBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


# ─── Embedding extractor ─────────────────────────────────────────────────────


class ClinicalBERTEmbedder:
    """
    Extract [CLS] embeddings from ClinicalBERT for downstream tasks.
    Uses mean pooling over last 4 layers for richer representation.
    """

    def __init__(
        self,
        model_name: str = CLINICALBERT_MODEL,
        max_length: int = 512,
        batch_size: int = 16,
        pooling: str = "cls",  # "cls" | "mean" | "mean_last4"
        layer_indices: Tuple[int, ...] = (-1, -2, -3, -4),
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pooling = pooling
        self.layer_indices = layer_indices
        self._loaded = False
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
            logger.info("Loading ClinicalBERT for embedding: %s", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name, output_hidden_states=True
            ).to(DEVICE)
            self.model.eval()
            self._loaded = True
            logger.info("ClinicalBERT embedder ready")
        except Exception as e:
            logger.warning("Could not load ClinicalBERT: %s. Using random embeddings.", e)
            self._loaded = True  # Mark as loaded (will use random)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Compute sentence embeddings for a list of texts.

        Returns
        -------
        embeddings : (N, 768) float32 array
        """
        self._load()

        if self.model is None:
            # Fallback: random embeddings of correct dim for testing
            logger.warning("Using random embeddings (ClinicalBERT not available)")
            return np.random.standard_normal((len(texts), 768)).astype(np.float32)

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            encoded = self.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            if self.pooling == "cls":
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif self.pooling == "mean":
                # Mean over non-padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).float()
                summed = (outputs.last_hidden_state * mask_expanded).sum(dim=1)
                counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
                emb = (summed / counts).cpu().numpy()
            elif self.pooling == "mean_last4":
                # Mean pool over last 4 hidden layers
                hidden_states = outputs.hidden_states
                selected = torch.stack(
                    [hidden_states[idx] for idx in self.layer_indices], dim=0
                )
                layer_mean = selected.mean(dim=0)  # (B, T, 768)
                mask_expanded = attention_mask.unsqueeze(-1).float()
                summed = (layer_mean * mask_expanded).sum(dim=1)
                counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
                emb = (summed / counts).cpu().numpy()
            else:
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(emb)

        return np.vstack(all_embeddings).astype(np.float32)


# ─── NER feature extractor ───────────────────────────────────────────────────


class NERFeatureExtractor:
    """Extract numerical features from NER output for ML models."""

    def __init__(self) -> None:
        self.ner = ClinicalNERPipeline(use_medspacy=False)

    def extract(self, text: str) -> np.ndarray:
        """Return 20-dim feature vector from note NER."""
        summary = self.ner.summarise_note(text)
        feats = np.array([
            float(summary.get("n_medications", 0)),
            float(summary.get("n_conditions", 0)),
            float(summary.get("n_procedures", 0)),
            float(summary.get("n_negated", 0)),
            float(summary.get("n_uncertain", 0)),
            float(summary.get("n_lab_values", 0)),
            float(summary.get("has_sepsis_flag", 0)),
            float(summary.get("has_cancer_flag", 0)),
            float(summary.get("has_heart_failure", 0)),
            float(summary.get("has_diabetes", 0)),
            # Text statistics
            float(len(text)),
            float(len(text.split())),
            float(text.count("\n")),
            float(len([s for s in text.split(".") if len(s.strip()) > 10])),
            # Keyword density
            float(text.lower().count("urgent") + text.lower().count("emergent")),
            float(text.lower().count("transfer") + text.lower().count("icu")),
            float(text.lower().count("admit") + text.lower().count("hospitali")),
            float(text.lower().count("discharge") + text.lower().count("follow-up")),
            float(text.lower().count("complication") + text.lower().count("adverse")),
            float(text.lower().count("procedure") + text.lower().count("surgery")),
        ], dtype=np.float32)
        return feats

    def extract_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.extract(t) for t in texts])


# ─── Complexity prediction MLP ────────────────────────────────────────────────


class ComplexityMLP(nn.Module):
    """
    Small MLP on top of BERT embeddings + NER features.
    Predicts cost quintile (5-class) or continuous cost.
    """

    def __init__(
        self,
        bert_dim: int = 768,
        ner_dim: int = 20,
        ehr_dim: int = 0,
        hidden_dim: int = 256,
        out_dim: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        in_dim = bert_dim + ner_dim + ehr_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── High-level scorer ───────────────────────────────────────────────────────


class ClinicalComplexityScorer:
    """
    Combines ClinicalBERT embeddings + NER features to predict:
      - cost_quintile (1-5)
      - complexity_tier (low/medium/high/very_high)
      - readmission_risk_score (0-1)

    Uses a lightweight Ridge regression for cost quintile prediction,
    or an MLP for richer representations.

    Usage
    -----
    scorer = ClinicalComplexityScorer()
    scorer.fit(train_texts, train_costs, train_readmission_labels)
    results = scorer.score(["Discharge note text..."])
    """

    COMPLEXITY_TIERS = {1: "low", 2: "low", 3: "medium", 4: "high", 5: "very_high"}

    def __init__(
        self,
        use_bert: bool = True,
        use_ner: bool = True,
        model_type: str = "ridge",  # "ridge" | "mlp"
        bert_pooling: str = "cls",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.use_bert = use_bert
        self.use_ner = use_ner
        self.model_type = model_type

        self.embedder = ClinicalBERTEmbedder(
            max_length=max_length,
            batch_size=batch_size,
            pooling=bert_pooling,
        ) if use_bert else None

        self.ner_extractor = NERFeatureExtractor() if use_ner else None

        self.cost_model: Optional[Any] = None
        self.readmission_model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.mlp: Optional[ComplexityMLP] = None
        self._is_fitted = False

    # ── Feature engineering ──────────────────────────────────────────────────

    def _get_features(self, texts: List[str]) -> np.ndarray:
        parts = []

        if self.use_bert and self.embedder:
            bert_feats = self.embedder.embed(texts)  # (N, 768)
            parts.append(bert_feats)

        if self.use_ner and self.ner_extractor:
            ner_feats = self.ner_extractor.extract_batch(texts)  # (N, 20)
            parts.append(ner_feats)

        if not parts:
            raise ValueError("At least one feature source must be enabled")

        return np.hstack(parts).astype(np.float32)

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        texts: List[str],
        costs: Optional[np.ndarray] = None,          # continuous costs for regression
        cost_quintiles: Optional[np.ndarray] = None,  # 1-5 quintile labels
        readmission_labels: Optional[np.ndarray] = None,  # binary
    ) -> Dict[str, float]:
        """
        Fit complexity scorer on labelled notes.

        At least one of costs/cost_quintiles must be provided.
        """
        logger.info("Extracting features for %d notes…", len(texts))
        X = self._get_features(texts)
        X_scaled = self.scaler.fit_transform(X)

        metrics: Dict[str, float] = {}

        # Cost quintile model
        if cost_quintiles is not None:
            y_quintile = np.asarray(cost_quintiles) - 1  # 0-indexed
            if self.model_type == "ridge":
                self.cost_model = Ridge(alpha=10.0)
                self.cost_model.fit(X_scaled, y_quintile)
                preds = self.cost_model.predict(X_scaled).clip(0, 4).round()
                acc = float((preds == y_quintile).mean())
                metrics["train_cost_accuracy"] = acc
                logger.info("Cost quintile model fitted. Train acc: %.4f", acc)
            else:
                self._fit_mlp(X_scaled, y_quintile, n_classes=5)
                metrics["mlp_fitted"] = 1.0
        elif costs is not None:
            # Bin costs into quintiles
            quintiles = pd.qcut(costs, q=5, labels=[0, 1, 2, 3, 4], duplicates="drop")
            y_quintile = quintiles.astype(int).values
            self.cost_model = Ridge(alpha=10.0)
            self.cost_model.fit(X_scaled, y_quintile)
            metrics["train_cost_r2"] = float(self.cost_model.score(X_scaled, y_quintile))

        # Readmission model
        if readmission_labels is not None:
            from sklearn.linear_model import LogisticRegression
            y_readmit = np.asarray(readmission_labels)
            self.readmission_model = LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=500
            )
            self.readmission_model.fit(X_scaled, y_readmit)
            from sklearn.metrics import roc_auc_score
            proba = self.readmission_model.predict_proba(X_scaled)[:, 1]
            auroc = float(roc_auc_score(y_readmit, proba))
            metrics["train_readmission_auroc"] = auroc
            logger.info("Readmission model fitted. Train AUROC: %.4f", auroc)

        self._is_fitted = True
        return metrics

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> None:
        """Train a small MLP on extracted features."""
        feat_dim = X.shape[1]
        self.mlp = ComplexityMLP(
            bert_dim=0, ner_dim=feat_dim, ehr_dim=0,
            hidden_dim=128, out_dim=n_classes, dropout=0.3
        ).to(DEVICE)

        X_t = torch.FloatTensor(X).to(DEVICE)
        y_t = torch.LongTensor(y).to(DEVICE)
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        self.mlp.train()
        for epoch in range(50):
            optimizer.zero_grad()
            loss = criterion(self.mlp(X_t), y_t)
            loss.backward()
            optimizer.step()

        self.mlp.eval()

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score(self, texts: List[str]) -> pd.DataFrame:
        """
        Score clinical notes.

        Returns DataFrame with:
          - cost_quintile (1-5)
          - complexity_tier (low/medium/high/very_high)
          - readmission_risk (0-1, if model fitted)
        """
        if not self._is_fitted:
            raise RuntimeError("Scorer not fitted. Call fit() first.")

        X = self._get_features(texts)
        X_scaled = self.scaler.transform(X)

        results = pd.DataFrame()

        if self.cost_model is not None:
            raw_preds = self.cost_model.predict(X_scaled)
            quintiles = np.clip(raw_preds.round().astype(int) + 1, 1, 5)
            results["cost_quintile"] = quintiles
            results["complexity_tier"] = [
                self.COMPLEXITY_TIERS[q] for q in quintiles
            ]

        if self.readmission_model is not None:
            results["readmission_risk"] = self.readmission_model.predict_proba(X_scaled)[:, 1]

        results["note_index"] = range(len(texts))
        return results

    def score_single(self, text: str) -> Dict[str, Any]:
        """Score a single note. Returns dict."""
        return self.score([text]).iloc[0].to_dict()

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "complexity_scorer.pkl"
        bundle = {
            "cost_model": self.cost_model,
            "readmission_model": self.readmission_model,
            "scaler": self.scaler,
            "use_bert": self.use_bert,
            "use_ner": self.use_ner,
        }
        joblib.dump(bundle, path)
        logger.info("Complexity scorer saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ClinicalComplexityScorer":
        bundle = joblib.load(path)
        instance = cls(use_bert=bundle["use_bert"], use_ner=bundle["use_ner"])
        instance.cost_model = bundle["cost_model"]
        instance.readmission_model = bundle["readmission_model"]
        instance.scaler = bundle["scaler"]
        instance._is_fitted = True
        return instance


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Clinical Complexity Scorer Smoke Test ===")

    rng = np.random.default_rng(42)

    sample_notes = [
        "Patient admitted for CHF exacerbation. Medications: furosemide, lisinopril. "
        "Labs: sodium 130, creatinine 2.1, BNP 1200. ICU stay for 3 days.",

        "Simple urinary tract infection. Started ciprofloxacin. Discharge home.",

        "Complex sepsis with multi-organ failure. Patient on mechanical ventilation, "
        "vasopressors, hemodialysis. History of diabetes, heart failure, CKD.",

        "Hypertension management. Metoprolol dose adjusted. No acute issues.",

        "Post-operative care following appendectomy. No complications.",
    ] * 20  # 100 notes

    costs = rng.lognormal(8, 1.5, len(sample_notes))
    readmission_labels = rng.binomial(1, 0.2, len(sample_notes))

    scorer = ClinicalComplexityScorer(
        use_bert=False,   # no BERT download needed for smoke test
        use_ner=True,
        model_type="ridge",
    )
    metrics = scorer.fit(sample_notes, costs=costs, readmission_labels=readmission_labels)
    logger.info("Training metrics: %s", metrics)

    results = scorer.score(sample_notes[:5])
    logger.info("\nScoring results:\n%s", results)

    assert "cost_quintile" in results.columns
    assert "complexity_tier" in results.columns
    assert "readmission_risk" in results.columns
    assert results["cost_quintile"].between(1, 5).all()

    path = scorer.save()
    loaded = ClinicalComplexityScorer.load(path)
    results2 = loaded.score(sample_notes[:5])
    pd.testing.assert_frame_equal(results[["cost_quintile"]], results2[["cost_quintile"]])

    logger.info("=== PASS ===")
