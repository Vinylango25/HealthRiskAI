"""
models/graph_network/gat_model.py
===================================
Graph Attention Network (GAT) for patient-disease-drug heterogeneous graph.

Architecture:
  - 3 GAT layers, 4 attention heads per layer
  - Dropout 0.3 on attention + feature dropout
  - Node classification head (mortality prediction)
  - Link prediction head (drug-disease association)
  - Supports heterogeneous graphs via type-specific linear projections

Target: AUROC > 0.78 for 30-day mortality node classification
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "reports" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── GAT Attention Layer ─────────────────────────────────────────────────────


class GATLayer(nn.Module):
    """
    Single GAT layer with multi-head attention.

    For each node i: h'_i = concat_k alpha_ik * W * h_k
    where alpha_ik = softmax(LeakyReLU(a^T [Wh_i || Wh_k]))
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.3,
        alpha: float = 0.2,  # LeakyReLU negative slope
        concat: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout_rate = dropout
        self.residual = residual

        # Per-head linear projections
        self.W = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        # Attention vector: a^T [Wh_i || Wh_k] → scalar
        self.attn_src = nn.Parameter(torch.empty(1, n_heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(1, n_heads, out_dim))

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dims differ
        if residual:
            self.res_proj = nn.Linear(in_dim, n_heads * out_dim if concat else out_dim, bias=False)
        else:
            self.res_proj = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attn_src, gain=1.414)
        nn.init.xavier_uniform_(self.attn_dst, gain=1.414)

    def forward(
        self,
        x: torch.Tensor,          # (N, in_dim)
        edge_index: torch.Tensor, # (2, E) — [src, dst]
    ) -> torch.Tensor:
        """
        Returns
        -------
        h_prime : (N, n_heads * out_dim) if concat else (N, out_dim)
        """
        N = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # Linear projection → (N, n_heads, out_dim)
        Wh = self.W(x).view(N, self.n_heads, self.out_dim)

        # Attention scores: e_ij = LeakyReLU(a_src * Wh_i + a_dst * Wh_j)
        e_src = (Wh * self.attn_src).sum(dim=-1)  # (N, n_heads)
        e_dst = (Wh * self.attn_dst).sum(dim=-1)

        e = self.leaky_relu(e_src[src_idx] + e_dst[dst_idx])  # (E, n_heads)

        # Softmax per destination node
        alpha = self._sparse_softmax(e, dst_idx, N)  # (E, n_heads)
        alpha = self.dropout(alpha)

        # Aggregate
        Wh_src = Wh[src_idx]  # (E, n_heads, out_dim)
        alpha_expanded = alpha.unsqueeze(-1)  # (E, n_heads, 1)
        msg = Wh_src * alpha_expanded  # (E, n_heads, out_dim)

        h_prime = torch.zeros(N, self.n_heads, self.out_dim, device=x.device)
        h_prime.index_add_(0, dst_idx, msg)

        # Residual
        if self.residual and self.res_proj is not None:
            res = self.res_proj(x)
            if self.concat:
                res = res.view(N, self.n_heads, self.out_dim)
                h_prime = h_prime + res
            else:
                h_prime = h_prime + res.unsqueeze(1)

        if self.concat:
            return h_prime.view(N, self.n_heads * self.out_dim)
        else:
            return h_prime.mean(dim=1)  # (N, out_dim)

    @staticmethod
    def _sparse_softmax(
        e: torch.Tensor, dst_idx: torch.Tensor, n_nodes: int
    ) -> torch.Tensor:
        """Per-node softmax over incoming edges."""
        # Shift for numerical stability
        max_val = torch.zeros(n_nodes, e.size(1), device=e.device)
        max_val.index_reduce_(0, dst_idx, e, reduce="amax", include_self=True)
        e_shifted = e - max_val[dst_idx]

        exp_e = torch.exp(e_shifted)
        sum_exp = torch.zeros(n_nodes, e.size(1), device=e.device)
        sum_exp.index_add_(0, dst_idx, exp_e)

        alpha = exp_e / (sum_exp[dst_idx] + 1e-10)
        return alpha


# ─── Heterogeneous linear projection ─────────────────────────────────────────


class HeteroLinear(nn.Module):
    """
    Type-specific linear projections for heterogeneous node types.
    Projects all node types to a common embedding dimension.
    """

    def __init__(
        self, in_dims: Dict[str, int], out_dim: int
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {ntype: nn.Linear(dim, out_dim) for ntype, dim in in_dims.items()}
        )

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            ntype: F.elu(self.projections[ntype](x))
            for ntype, x in x_dict.items()
            if ntype in self.projections
        }


# ─── Full GAT model ───────────────────────────────────────────────────────────


class GATModel(nn.Module):
    """
    3-layer GAT for node classification and link prediction.

    Supports heterogeneous graphs by flattening all nodes into a shared
    embedding space via type-specific projections before GAT layers.

    Usage (homogeneous):
        model = GATModel(in_dim=32, hidden_dim=64, out_dim=2, n_layers=3, n_heads=4)
        logits = model(x, edge_index)

    Usage (heterogeneous):
        model = GATModel.from_hetero(node_feat_dims={'patient': 8, 'disease': 4, 'drug': 4},
                                      hidden_dim=64, out_dim=2)
        logits = model.forward_hetero(x_dict, edge_index_dict, node_type_map)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 2,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.3,
        task: str = "node_classification",  # or "link_prediction"
    ) -> None:
        super().__init__()
        self.task = task
        self.dropout_rate = dropout

        # Build GAT layers
        # Layer 1: in_dim → hidden_dim * n_heads (concat)
        # Layer 2: hidden_dim * n_heads → hidden_dim * n_heads
        # Layer 3: hidden_dim * n_heads → hidden_dim (mean-pooling heads)
        self.gat_layers = nn.ModuleList()

        layer_in = in_dim
        for i in range(n_layers):
            is_last = i == n_layers - 1
            self.gat_layers.append(
                GATLayer(
                    in_dim=layer_in,
                    out_dim=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    concat=not is_last,
                    residual=True,
                )
            )
            layer_in = hidden_dim * n_heads if not is_last else hidden_dim

        # Task heads
        if task == "node_classification":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, out_dim),
            )
        elif task == "link_prediction":
            self.link_predictor = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        self.feature_dropout = nn.Dropout(dropout)

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Run GAT layers and return node embeddings."""
        h = self.feature_dropout(x)
        for i, layer in enumerate(self.gat_layers):
            h = layer(h, edge_index)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
                h = self.feature_dropout(h)
        return h  # (N, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        link_pairs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, in_dim) node features
        edge_index : (2, E) edge list
        link_pairs : (2, P) pairs for link prediction (optional)

        Returns
        -------
        For node classification: (N, out_dim) logits
        For link prediction: (P,) edge scores
        """
        embeddings = self.encode(x, edge_index)

        if self.task == "node_classification":
            return self.classifier(embeddings)
        elif self.task == "link_prediction":
            if link_pairs is None:
                raise ValueError("link_pairs required for link prediction task")
            src_emb = embeddings[link_pairs[0]]
            dst_emb = embeddings[link_pairs[1]]
            pair_emb = torch.cat([src_emb, dst_emb], dim=1)
            return self.link_predictor(pair_emb).squeeze(-1)

        return embeddings

    @classmethod
    def from_hetero(
        cls,
        node_feat_dims: Dict[str, int],
        hidden_dim: int = 64,
        out_dim: int = 2,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.3,
        task: str = "node_classification",
    ) -> "HeteroGATModel":
        """Create a heterogeneous GAT model."""
        return HeteroGATModel(
            node_feat_dims=node_feat_dims,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            task=task,
        )


class HeteroGATModel(nn.Module):
    """
    Heterogeneous GAT: handles multiple node types with type-specific projections.
    Flattens all nodes into a shared embedding space, runs GAT, then applies
    a node-type-specific classification head.
    """

    def __init__(
        self,
        node_feat_dims: Dict[str, int],
        hidden_dim: int = 64,
        out_dim: int = 2,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.3,
        task: str = "node_classification",
    ) -> None:
        super().__init__()
        self.node_types = list(node_feat_dims.keys())
        self.hidden_dim = hidden_dim

        # Project each node type to common embedding
        self.input_proj = HeteroLinear(node_feat_dims, hidden_dim)

        # Shared GAT backbone
        self.gat = GATModel(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            task="node_classification",  # use encode only
        )

        # Replace classifier with identity to get embeddings
        self.gat.classifier = nn.Identity()

        # Per-node-type classification heads
        self.classifiers = nn.ModuleDict(
            {
                ntype: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, out_dim),
                )
                for ntype in self.node_types
            }
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        node_type_offsets: Dict[str, Tuple[int, int]],
        target_node_type: str = "patient",
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_dict : {node_type: (N_type, feat_dim)}
        edge_index : (2, E) using global node indices
        node_type_offsets : {node_type: (start_idx, end_idx)}
        target_node_type : which node type to classify

        Returns
        -------
        logits : (N_target, out_dim)
        """
        # Project all node types
        proj_dict = self.input_proj(x_dict)

        # Concatenate in consistent order
        x_all = torch.cat([proj_dict[nt] for nt in self.node_types if nt in proj_dict])

        # Run GAT
        embeddings = self.gat.encode(x_all, edge_index)

        # Extract target node type embeddings
        start, end = node_type_offsets[target_node_type]
        target_emb = embeddings[start:end]

        return self.classifiers[target_node_type](target_emb)


# ─── Utility: build edge index from graph dict ───────────────────────────────


def build_homogeneous_edge_index(
    graph: Dict[str, Any],
    node_type_offsets: Dict[str, int],
) -> torch.Tensor:
    """
    Flatten heterogeneous edge lists into a single edge_index tensor,
    offsetting node indices by type.
    """
    all_edges: List[np.ndarray] = []

    for etype, edges in graph["edges"].items():
        # Parse edge type: "src_type-rel-dst_type"
        parts = etype.split("-")
        if len(parts) < 3:
            continue
        src_type = parts[0]
        dst_type = parts[-1]

        src_offset = node_type_offsets.get(src_type, 0)
        dst_offset = node_type_offsets.get(dst_type, 0)

        src_idx = edges[0] + src_offset
        dst_idx = edges[1] + dst_offset
        all_edges.append(np.stack([src_idx, dst_idx], axis=0))

    if not all_edges:
        return torch.zeros(2, 0, dtype=torch.long)

    edge_arr = np.concatenate(all_edges, axis=1)
    return torch.LongTensor(edge_arr)


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== GAT Model Smoke Test ===")

    # Simple homogeneous graph test
    N, E = 100, 300
    in_dim = 16
    rng = np.random.default_rng(42)

    x = torch.randn(N, in_dim)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    edge_index = torch.stack([src, dst])

    # Node classification
    model = GATModel(
        in_dim=in_dim,
        hidden_dim=32,
        out_dim=2,
        n_layers=3,
        n_heads=4,
        dropout=0.3,
        task="node_classification",
    )
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
    assert logits.shape == (N, 2), f"Expected ({N}, 2), got {logits.shape}"
    logger.info("Node classification output: %s", logits.shape)

    # Link prediction
    model_lp = GATModel(
        in_dim=in_dim, hidden_dim=32, out_dim=1, n_layers=2, n_heads=2,
        task="link_prediction",
    )
    P = 50
    link_pairs = torch.stack([
        torch.randint(0, N, (P,)), torch.randint(0, N, (P,))
    ])
    with torch.no_grad():
        scores = model_lp(x, edge_index, link_pairs=link_pairs)
    assert scores.shape == (P,), f"Expected ({P},), got {scores.shape}"
    logger.info("Link prediction output: %s", scores.shape)

    # Heterogeneous graph test
    node_feat_dims = {"patient": 8, "disease": 4, "drug": 4}
    hetero_model = GATModel.from_hetero(
        node_feat_dims, hidden_dim=32, out_dim=2, n_layers=2, n_heads=2
    )
    logger.info("HeteroGATModel created: %d parameters",
                sum(p.numel() for p in hetero_model.parameters()))

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("GATModel parameters: %d", n_params)
    logger.info("=== PASS ===")
