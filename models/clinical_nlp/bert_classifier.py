"""
models/clinical_nlp/bert_classifier.py
=========================================
ClinicalBERT fine-tuning for discharge note classification.

Tasks:
  - Binary: 30-day readmission from discharge summary
  - Multi-class: discharge disposition (home, SNF, LTACH, hospice, expired)

Uses: emilyalsentzer/Bio_ClinicalBERT (HuggingFace)
Target: AUROC > 0.75 for readmission classification
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "reports" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLINICALBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


# ─── Dataset ──────────────────────────────────────────────────────────────────


class ClinicalNoteDataset(Dataset):
    """
    Tokenised clinical notes for BERT fine-tuning.
    Truncates to max_length tokens (typically 512).
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        self.labels = labels
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.token_type_ids = encodings.get(
            "token_type_ids", torch.zeros_like(encodings["input_ids"])
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── Model ────────────────────────────────────────────────────────────────────


class ClinicalBERTClassifier(nn.Module):
    """
    ClinicalBERT + classification head.

    Architecture:
      [CLS] token representation → dropout → linear → logits
    
    Optionally freezes BERT layers for faster fine-tuning with small datasets.
    """

    def __init__(
        self,
        n_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert_layers: int = 10,  # freeze first N layers
        model_name: str = CLINICALBERT_MODEL,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        # Load pre-trained ClinicalBERT
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        # Freeze early layers for efficiency
        if freeze_bert_layers > 0:
            modules_to_freeze = [self.bert.embeddings]
            if hasattr(self.bert, "encoder"):
                encoder_layers = list(self.bert.encoder.layer)
                modules_to_freeze += encoder_layers[:freeze_bert_layers]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


# ─── High-level wrapper ───────────────────────────────────────────────────────


class ClinicalNoteClassifier:
    """
    High-level trainer / predictor for discharge note classification.

    Supports:
      - Binary readmission prediction
      - Multi-class disposition classification
      - MLflow experiment tracking
      - MIMIC-IV note format
    """

    def __init__(
        self,
        task: str = "readmission",  # "readmission" | "disposition"
        model_name: str = CLINICALBERT_MODEL,
        max_length: int = 512,
        batch_size: int = 16,
        epochs: int = 5,
        lr: float = 2e-5,
        warmup_frac: float = 0.1,
        dropout: float = 0.3,
        freeze_layers: int = 10,
        weight_decay: float = 0.01,
        patience: int = 3,
        mlflow_experiment: str = "clinical_bert",
    ) -> None:
        self.task = task
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.warmup_frac = warmup_frac
        self.dropout = dropout
        self.freeze_layers = freeze_layers
        self.weight_decay = weight_decay
        self.patience = patience
        self.mlflow_experiment = mlflow_experiment

        self.tokenizer: Optional[Any] = None
        self.model: Optional[ClinicalBERTClassifier] = None
        self.label_map: Dict[int, str] = {}
        self._is_fitted = False

    def _load_tokenizer(self) -> None:
        if self.tokenizer is None:
            logger.info("Loading tokenizer: %s", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _prepare_labels(self, labels: List[Any]) -> Tuple[List[int], Dict]:
        """Convert string labels to integers if needed."""
        if all(isinstance(l, (int, np.integer)) for l in labels):
            unique = sorted(set(int(l) for l in labels))
            label_map = {i: str(v) for i, v in enumerate(unique)}
            return [int(l) for l in labels], label_map
        unique = sorted(set(labels))
        str_to_int = {v: i for i, v in enumerate(unique)}
        int_to_str = {i: v for v, i in str_to_int.items()}
        return [str_to_int[l] for l in labels], int_to_str

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[Any],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[Any]] = None,
    ) -> Dict[str, List[float]]:
        """
        Fine-tune ClinicalBERT on discharge notes.

        Parameters
        ----------
        train_texts : list of discharge note strings
        train_labels : list of labels (int or str)
        """
        self._load_tokenizer()

        int_train, label_map = self._prepare_labels(train_labels)
        self.label_map = label_map
        n_classes = len(set(int_train))

        logger.info(
            "Fine-tuning ClinicalBERT: %d samples, %d classes, task=%s",
            len(train_texts), n_classes, self.task,
        )

        # Build model
        self.model = ClinicalBERTClassifier(
            n_classes=n_classes,
            dropout=self.dropout,
            freeze_bert_layers=self.freeze_layers,
            model_name=self.model_name,
        ).to(DEVICE)

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Trainable parameters: %d", n_trainable)

        # Datasets
        train_ds = ClinicalNoteDataset(
            train_texts, int_train, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if val_texts and val_labels:
            int_val, _ = self._prepare_labels(val_labels)
            val_ds = ClinicalNoteDataset(
                val_texts, int_val, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_frac)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Class weights for imbalanced binary
        counts = np.bincount(int_train, minlength=n_classes).astype(float)
        weights = torch.FloatTensor(counts.sum() / (n_classes * counts + 1e-8)).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)

        history: Dict[str, List[float]] = {"train_loss": [], "val_auroc": []}
        best_val = 0.0
        patience_counter = 0
        best_state = None

        mlflow.set_experiment(self.mlflow_experiment)
        with mlflow.start_run(run_name=f"clinical_bert_{self.task}"):
            mlflow.log_params({
                "task": self.task,
                "epochs": self.epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "n_classes": n_classes,
                "n_train": len(train_texts),
            })

            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss = 0.0

                for batch in train_loader:
                    ids = batch["input_ids"].to(DEVICE)
                    mask = batch["attention_mask"].to(DEVICE)
                    tt_ids = batch["token_type_ids"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)

                    optimizer.zero_grad()
                    logits = self.model(ids, mask, tt_ids)
                    loss = criterion(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)
                history["train_loss"].append(avg_loss)

                if val_loader:
                    val_metrics = self._evaluate_loader(val_loader, n_classes)
                    val_auroc = val_metrics.get("auroc", 0.5)
                    history["val_auroc"].append(val_auroc)
                    logger.info(
                        "Epoch %d | loss: %.4f | val AUROC: %.4f",
                        epoch + 1, avg_loss, val_auroc,
                    )
                    if val_auroc > best_val:
                        best_val = val_auroc
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break
                else:
                    logger.info("Epoch %d | loss: %.4f", epoch + 1, avg_loss)

            if best_state:
                self.model.load_state_dict(best_state)

            mlflow.log_metrics({"best_val_auroc": best_val})

        self._is_fitted = True
        return history

    def _evaluate_loader(self, loader: DataLoader, n_classes: int) -> Dict[str, float]:
        self.model.eval()
        all_proba, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                tt_ids = batch["token_type_ids"].to(DEVICE)
                labels = batch["labels"].cpu().numpy()
                logits = self.model(ids, mask, tt_ids)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                all_proba.append(proba)
                all_labels.extend(labels)

        all_proba = np.vstack(all_proba)
        all_labels = np.array(all_labels)

        metrics: Dict[str, float] = {}
        try:
            if n_classes == 2:
                metrics["auroc"] = float(roc_auc_score(all_labels, all_proba[:, 1]))
                metrics["auprc"] = float(average_precision_score(all_labels, all_proba[:, 1]))
            else:
                metrics["auroc"] = float(
                    roc_auc_score(all_labels, all_proba, multi_class="ovr", average="macro")
                )
        except ValueError as e:
            logger.warning("Metric computation failed: %s", e)
            metrics["auroc"] = 0.5
        return metrics

    def predict(self, texts: List[str]) -> pd.DataFrame:
        """
        Returns DataFrame with columns: predicted_class, probability_*, label
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        self._load_tokenizer()
        self.model.eval()

        ds = ClinicalNoteDataset(
            texts, [0] * len(texts), self.tokenizer, self.max_length
        )
        loader = DataLoader(ds, batch_size=self.batch_size)

        all_proba = []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                tt_ids = batch["token_type_ids"].to(DEVICE)
                logits = self.model(ids, mask, tt_ids)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                all_proba.append(proba)

        all_proba = np.vstack(all_proba)
        preds = np.argmax(all_proba, axis=1)

        result = pd.DataFrame({"predicted_class": preds})
        for i in range(all_proba.shape[1]):
            label = self.label_map.get(i, str(i))
            result[f"prob_{label}"] = all_proba[:, i]

        return result

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / f"clinical_bert_{self.task}"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.model:
            torch.save(self.model.state_dict(), path / "model.pt")
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(path / "tokenizer"))
        import json
        with open(path / "config.json", "w") as f:
            json.dump({
                "task": self.task,
                "label_map": {str(k): v for k, v in self.label_map.items()},
                "max_length": self.max_length,
                "n_classes": len(self.label_map),
            }, f)
        logger.info("ClinicalBERT classifier saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ClinicalNoteClassifier":
        import json
        path = Path(path)
        with open(path / "config.json") as f:
            cfg = json.load(f)
        instance = cls(task=cfg["task"])
        instance.label_map = {int(k): v for k, v in cfg["label_map"].items()}
        instance._load_tokenizer()
        n_classes = cfg["n_classes"]
        instance.model = ClinicalBERTClassifier(n_classes=n_classes)
        instance.model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        instance.model.to(DEVICE)
        instance._is_fitted = True
        return instance


# ─── Smoke test (no actual BERT download — uses random weights) ───────────────

def _mock_bert_smoke_test() -> None:
    """Test the classifier structure without downloading weights."""
    import torch

    logger.info("Running mock structure test (no BERT download)…")

    # Build classifier with small random model
    n_classes = 2
    hidden = 128

    class MockBERT(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(32, hidden // 2, batch_first=True, bidirectional=True)

        class Config:
            hidden_size = 128

        config = Config()
        embeddings = nn.Embedding(100, 32)

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            emb = self.embeddings(input_ids)
            out, _ = self.encoder(emb)

            class Output:
                pass

            o = Output()
            o.last_hidden_state = out
            return o

    clf = ClinicalBERTClassifier.__new__(ClinicalBERTClassifier)
    clf.bert = MockBERT()
    clf.n_classes = n_classes
    clf.classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden // 2, n_classes),
    )

    x = torch.randint(0, 100, (4, 32))
    mask = torch.ones(4, 32, dtype=torch.long)
    out = clf(x, mask)
    assert out.shape == (4, n_classes)
    logger.info("Mock structure test passed. Output shape: %s", out.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== ClinicalBERT Classifier Smoke Test ===")
    _mock_bert_smoke_test()
    logger.info("=== PASS (structure test) ===")
    logger.info("Note: Full fine-tuning requires HuggingFace model download + MIMIC-IV data.")
