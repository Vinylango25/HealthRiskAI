"""
models/graph_network/trainer.py
=================================
Training loop for the GAT model:
  - Node classification (30-day mortality, readmission)
  - Link prediction (drug-disease associations)
  - Negative sampling for link prediction
  - MLflow tracking
  - Early stopping
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .gat_model import GATModel, build_homogeneous_edge_index
from .graph_builder import make_synthetic_graph

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "reports" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Config ───────────────────────────────────────────────────────────────────


class GNNTrainerConfig:
    def __init__(
        self,
        task: str = "node_classification",
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        epochs: int = 200,
        patience: int = 20,
        batch_size_links: int = 512,
        neg_ratio: int = 3,
        mlflow_experiment: str = "gnn_graph",
    ) -> None:
        self.task = task
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size_links = batch_size_links
        self.neg_ratio = neg_ratio
        self.mlflow_experiment = mlflow_experiment


# ─── Data preparation ─────────────────────────────────────────────────────────


def prepare_node_classification_data(
    graph: Dict[str, Any],
    node_labels: np.ndarray,  # (N_patients,) binary labels
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, Any]:
    """
    Prepare a flattened homogeneous graph tensor for node classification.
    All node types are concatenated with type-indicator features appended.
    """
    node_types = ["patient", "disease", "drug", "procedure"]
    feats = []
    offsets: Dict[str, int] = {}
    type_sizes: Dict[str, int] = {}
    cur_offset = 0

    max_feat_dim = 0
    for ntype in node_types:
        if ntype in graph.get("node_features", {}):
            max_feat_dim = max(max_feat_dim, graph["node_features"][ntype].shape[1])

    for ntype in node_types:
        if ntype not in graph.get("node_features", {}):
            continue
        f = graph["node_features"][ntype]
        n = f.shape[0]
        # Pad to max_feat_dim
        if f.shape[1] < max_feat_dim:
            pad = np.zeros((n, max_feat_dim - f.shape[1]), dtype=np.float32)
            f = np.hstack([f, pad])
        # Add node type one-hot (4 types)
        type_one_hot = np.zeros((n, 4), dtype=np.float32)
        type_idx = node_types.index(ntype)
        type_one_hot[:, type_idx] = 1.0
        f = np.hstack([f, type_one_hot])
        feats.append(f)
        offsets[ntype] = cur_offset
        type_sizes[ntype] = n
        cur_offset += n

    x_all = torch.FloatTensor(np.vstack(feats))
    in_dim = x_all.shape[1]

    edge_index = build_homogeneous_edge_index(graph, offsets)

    # Patient nodes only
    n_patients = type_sizes.get("patient", len(node_labels))
    n_patients = min(n_patients, len(node_labels))
    patient_indices = torch.arange(offsets.get("patient", 0),
                                   offsets.get("patient", 0) + n_patients)
    y = torch.LongTensor(node_labels[:n_patients])

    # Train / val / test split (temporal — use index as proxy)
    n_train = int(n_patients * train_frac)
    n_val = int(n_patients * val_frac)
    train_mask = torch.zeros(n_patients, dtype=torch.bool)
    val_mask = torch.zeros(n_patients, dtype=torch.bool)
    test_mask = torch.zeros(n_patients, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train: n_train + n_val] = True
    test_mask[n_train + n_val:] = True

    return {
        "x": x_all.to(DEVICE),
        "edge_index": edge_index.to(DEVICE),
        "y": y.to(DEVICE),
        "patient_offset": offsets.get("patient", 0),
        "patient_indices": patient_indices,
        "train_mask": train_mask.to(DEVICE),
        "val_mask": val_mask.to(DEVICE),
        "test_mask": test_mask.to(DEVICE),
        "in_dim": in_dim,
        "n_patients": n_patients,
    }


def prepare_link_prediction_data(
    graph: Dict[str, Any],
    edge_type: str = "disease-treated_by-drug",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Dict[str, Any]:
    """
    Prepare positive/negative edge pairs for link prediction.
    Negative edges are sampled randomly from non-existing pairs.
    """
    if edge_type not in graph.get("edges", {}):
        raise ValueError(f"Edge type '{edge_type}' not in graph")

    pos_edges = graph["edges"][edge_type]  # (2, E)
    E = pos_edges.shape[1]

    # Split positive edges
    perm = np.random.permutation(E)
    n_val = int(E * val_frac)
    n_test = int(E * test_frac)
    n_train = E - n_val - n_test

    train_pos = pos_edges[:, perm[:n_train]]
    val_pos = pos_edges[:, perm[n_train: n_train + n_val]]
    test_pos = pos_edges[:, perm[n_train + n_val:]]

    return {
        "train_pos": torch.LongTensor(train_pos),
        "val_pos": torch.LongTensor(val_pos),
        "test_pos": torch.LongTensor(test_pos),
        "all_pos_set": set(map(tuple, pos_edges.T.tolist())),
    }


def sample_negative_edges(
    pos_edges: torch.Tensor,
    n_nodes_src: int,
    n_nodes_dst: int,
    all_pos_set: set,
    neg_ratio: int = 3,
) -> torch.Tensor:
    """Sample random negative edges (not in positive set)."""
    n_pos = pos_edges.shape[1]
    n_neg = n_pos * neg_ratio
    neg_src, neg_dst = [], []

    rng = np.random.default_rng()
    attempts = 0
    while len(neg_src) < n_neg and attempts < n_neg * 10:
        s = rng.integers(0, n_nodes_src)
        d = rng.integers(0, n_nodes_dst)
        if (s, d) not in all_pos_set:
            neg_src.append(s)
            neg_dst.append(d)
        attempts += 1

    return torch.LongTensor([neg_src, neg_dst])


# ─── Trainer class ────────────────────────────────────────────────────────────


class GNNTrainer:
    """
    Trains GATModel for node classification or link prediction.

    Usage
    -----
    trainer = GNNTrainer(config)
    trainer.fit(graph, node_labels=labels)
    metrics = trainer.evaluate(graph)
    """

    def __init__(self, config: Optional[GNNTrainerConfig] = None) -> None:
        self.config = config or GNNTrainerConfig()
        self.model: Optional[GATModel] = None
        self._is_fitted = False

    # ── Node classification ───────────────────────────────────────────────────

    def fit_node_classification(
        self,
        data: Dict[str, Any],
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[float]]:
        """Train GAT for node classification."""
        cfg = self.config
        in_dim = data["in_dim"]

        # Count classes
        n_classes = int(data["y"].max().item()) + 1

        self.model = GATModel(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=n_classes,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            task="node_classification",
        ).to(DEVICE)

        optimizer = Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=False)

        if class_weights is not None:
            cw = class_weights.to(DEVICE)
        else:
            # Auto class weights for imbalanced
            y_train = data["y"][data["train_mask"]].cpu().numpy()
            counts = np.bincount(y_train, minlength=n_classes).astype(float)
            cw = torch.FloatTensor(counts.sum() / (n_classes * counts + 1e-8)).to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=cw)

        history: Dict[str, List[float]] = {"train_loss": [], "val_auroc": []}
        best_val_auroc = 0.0
        patience_counter = 0
        best_state = None

        x = data["x"]
        edge_index = data["edge_index"]
        y = data["y"]
        pat_offset = data["patient_offset"]
        n_pat = data["n_patients"]
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]

        mlflow.set_experiment(cfg.mlflow_experiment)
        with mlflow.start_run(run_name="gnn_node_clf"):
            mlflow.log_params({
                "hidden_dim": cfg.hidden_dim,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "dropout": cfg.dropout,
                "epochs": cfg.epochs,
            })

            for epoch in range(cfg.epochs):
                self.model.train()
                optimizer.zero_grad()

                all_logits = self.model(x, edge_index)
                # Slice patient nodes
                pat_logits = all_logits[pat_offset: pat_offset + n_pat]
                loss = criterion(pat_logits[train_mask], y[train_mask])

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                history["train_loss"].append(loss.item())

                # Validation
                if epoch % 5 == 0 or epoch == cfg.epochs - 1:
                    val_auroc = self._eval_node_clf(
                        x, edge_index, pat_logits.detach(), y, val_mask
                    )
                    history["val_auroc"].append(val_auroc)
                    scheduler.step(1 - val_auroc)

                    if epoch % 20 == 0:
                        logger.info(
                            "Epoch %3d | loss: %.4f | val AUROC: %.4f",
                            epoch + 1, loss.item(), val_auroc,
                        )

                    if val_auroc > best_val_auroc:
                        best_val_auroc = val_auroc
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= cfg.patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

            if best_state:
                self.model.load_state_dict(best_state)

            mlflow.log_metrics({
                "best_val_auroc": best_val_auroc,
                "final_train_loss": history["train_loss"][-1],
            })

        self._is_fitted = True
        logger.info("Training complete. Best val AUROC: %.4f", best_val_auroc)
        return history

    def _eval_node_clf(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            proba = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
        y_true = y[mask].cpu().numpy()
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, proba))

    # ── Link prediction ───────────────────────────────────────────────────────

    def fit_link_prediction(
        self,
        data: Dict[str, Any],
        link_data: Dict[str, Any],
        n_nodes_src: int,
        n_nodes_dst: int,
    ) -> Dict[str, List[float]]:
        """Train GAT for link prediction."""
        cfg = self.config
        in_dim = data["in_dim"]

        self.model = GATModel(
            in_dim=in_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=1,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            task="link_prediction",
        ).to(DEVICE)

        optimizer = Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        history: Dict[str, List[float]] = {"train_loss": [], "val_auroc": []}
        best_auroc = 0.0
        best_state = None
        patience_counter = 0

        mlflow.set_experiment(cfg.mlflow_experiment)
        with mlflow.start_run(run_name="gnn_link_pred"):
            for epoch in range(cfg.epochs):
                self.model.train()

                # Sample negatives fresh each epoch
                neg_edges = sample_negative_edges(
                    link_data["train_pos"],
                    n_nodes_src, n_nodes_dst,
                    link_data["all_pos_set"],
                    cfg.neg_ratio,
                ).to(DEVICE)

                pos = link_data["train_pos"].to(DEVICE)
                all_pairs = torch.cat([pos, neg_edges], dim=1)
                labels = torch.cat([
                    torch.ones(pos.shape[1]),
                    torch.zeros(neg_edges.shape[1]),
                ]).to(DEVICE)

                optimizer.zero_grad()
                scores = self.model(data["x"], data["edge_index"], link_pairs=all_pairs)
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                history["train_loss"].append(loss.item())

                if epoch % 10 == 0:
                    val_auroc = self._eval_link(data, link_data, n_nodes_src, n_nodes_dst)
                    history["val_auroc"].append(val_auroc)
                    logger.info("Epoch %3d | loss: %.4f | val AUROC: %.4f",
                                epoch + 1, loss.item(), val_auroc)

                    if val_auroc > best_auroc:
                        best_auroc = val_auroc
                        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if patience_counter >= cfg.patience // 10:
                        break

            if best_state:
                self.model.load_state_dict(best_state)

        self._is_fitted = True
        return history

    def _eval_link(self, data, link_data, n_src, n_dst) -> float:
        self.model.eval()
        neg = sample_negative_edges(
            link_data["val_pos"], n_src, n_dst, link_data["all_pos_set"], 3
        ).to(DEVICE)
        pos = link_data["val_pos"].to(DEVICE)
        all_pairs = torch.cat([pos, neg], dim=1)
        y_true = np.concatenate([
            np.ones(pos.shape[1]), np.zeros(neg.shape[1])
        ])
        with torch.no_grad():
            scores = torch.sigmoid(
                self.model(data["x"], data["edge_index"], link_pairs=all_pairs)
            ).cpu().numpy()
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, scores))

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "gnn_model.pt"
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "config": self.config.__dict__,
        }, path)
        logger.info("GNN model saved to %s", path)
        return path

    def load(self, path: Path, in_dim: int, out_dim: int) -> None:
        state = torch.load(path, map_location="cpu")
        cfg_dict = state["config"]
        self.model = GATModel(
            in_dim=in_dim,
            hidden_dim=cfg_dict["hidden_dim"],
            out_dim=out_dim,
            n_layers=cfg_dict["n_layers"],
            n_heads=cfg_dict["n_heads"],
            dropout=cfg_dict["dropout"],
            task=cfg_dict["task"],
        )
        self.model.load_state_dict(state["model_state"])
        self.model.to(DEVICE)
        self._is_fitted = True


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== GNN Trainer Smoke Test ===")

    # Build synthetic graph
    graph = make_synthetic_graph(150, n_diseases=20, n_drugs=30)

    # Synthetic labels: 15% mortality rate for patients
    n_patients = graph["node_counts"]["patient"]
    rng = np.random.default_rng(42)
    labels = rng.binomial(1, 0.15, n_patients)

    data = prepare_node_classification_data(graph, labels)
    logger.info("Graph data prepared. in_dim=%d, n_patients=%d", data["in_dim"], n_patients)

    config = GNNTrainerConfig(
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        epochs=10,
        patience=5,
    )
    trainer = GNNTrainer(config)
    history = trainer.fit_node_classification(data)

    assert len(history["train_loss"]) > 0
    logger.info("Training history train loss: %.4f (final)", history["train_loss"][-1])
    logger.info("=== PASS ===")
