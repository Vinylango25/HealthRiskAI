"""
models/survival/dynamic_deephit.py
====================================
Dynamic-DeepHit: LSTM-based survival model for longitudinal patient trajectories.

Architecture:
  - LSTM encoder processes time-series of clinical measurements
  - Shared representation → cause-specific sub-networks
  - Multi-task loss: cause-specific hazards at multiple time horizons (3, 6, 12 months)
  - Handles competing risks (readmission vs mortality vs complication)

Reference: Lee et al. (2019) "Dynamic-DeepHit: A Deep Learning Approach for
           Dynamic Survival Analysis With Competing Risks Based on Longitudinal Data"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE / "reports" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Config ───────────────────────────────────────────────────────────────────


@dataclass
class DynamicDeepHitConfig:
    """Configuration for Dynamic-DeepHit model."""

    # LSTM encoder
    input_dim: int = 32           # number of longitudinal features per timestep
    hidden_dim: int = 128         # LSTM hidden size
    n_lstm_layers: int = 2        # stacked LSTM layers
    dropout_lstm: float = 0.3

    # Shared representation MLP
    shared_dim: int = 64
    shared_layers: int = 2

    # Cause-specific sub-networks
    n_causes: int = 3             # readmission, mortality, complication
    cause_dim: int = 32

    # Time grid
    time_horizons: List[int] = field(default_factory=lambda: [3, 6, 12])  # months
    n_time_bins: int = 36         # discretise time into 36 bins

    # Training
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    alpha: float = 0.2            # ranking loss weight
    sigma: float = 0.1            # ranking kernel width
    patience: int = 15            # early stopping
    max_seq_len: int = 24         # max longitudinal visits

    # MLflow
    mlflow_experiment: str = "dynamic_deephit"
    cause_names: List[str] = field(
        default_factory=lambda: ["readmission", "mortality", "complication"]
    )


# ─── Dataset ──────────────────────────────────────────────────────────────────


class LongitudinalSurvivalDataset(Dataset):
    """
    PyTorch Dataset for longitudinal survival data.

    Each sample:
      - X_seq: (seq_len, input_dim) — time-series of clinical measurements
      - X_static: (static_dim,) — baseline covariates
      - event: int — cause index (0 = censored, 1..n_causes)
      - time_bin: int — discretised event/censor time bin
      - mask: (seq_len,) — padding mask
    """

    def __init__(
        self,
        X_seq: np.ndarray,       # (N, max_seq_len, input_dim)
        X_static: np.ndarray,    # (N, static_dim)
        events: np.ndarray,      # (N,) — 0=censored, 1..k cause
        time_bins: np.ndarray,   # (N,) — discretised time
        seq_lengths: np.ndarray, # (N,) — actual sequence length
    ) -> None:
        self.X_seq = torch.FloatTensor(X_seq)
        self.X_static = torch.FloatTensor(X_static)
        self.events = torch.LongTensor(events)
        self.time_bins = torch.LongTensor(time_bins)
        self.seq_lengths = seq_lengths

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        return (
            self.X_seq[idx],
            self.X_static[idx],
            self.events[idx],
            self.time_bins[idx],
            self.seq_lengths[idx],
        )


# ─── Model architecture ───────────────────────────────────────────────────────


class LSTMEncoder(nn.Module):
    """LSTM encoder for longitudinal clinical sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)
        seq_lengths : (batch,) actual lengths for masking

        Returns
        -------
        last_hidden : (batch, hidden_dim) — state at last valid timestep
        """
        # Pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            seq_lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )
        output, (h_n, _) = self.lstm(packed)
        # Use last layer's final hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        return self.layer_norm(last_hidden)


class SharedRepresentation(nn.Module):
    """Shared MLP layers after LSTM encoder."""

    def __init__(self, in_dim: int, shared_dim: int, n_layers: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        cur_dim = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(cur_dim, shared_dim), nn.ReLU(), nn.Dropout(0.2)]
            cur_dim = shared_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CauseSpecificNet(nn.Module):
    """Sub-network for a single competing risk cause."""

    def __init__(self, shared_dim: int, cause_dim: int, n_time_bins: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shared_dim, cause_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(cause_dim, n_time_bins),
        )

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        """Returns (batch, n_time_bins) log-hazard values."""
        return self.net(shared)


class DynamicDeepHitNet(nn.Module):
    """
    Full Dynamic-DeepHit architecture.

    Forward pass returns:
      - cause_hazards: (batch, n_causes, n_time_bins) — softmax-normalised
        probability mass over time for each cause
    """

    def __init__(self, cfg: DynamicDeepHitConfig, static_dim: int = 16) -> None:
        super().__init__()
        self.cfg = cfg

        # LSTM encoder
        self.encoder = LSTMEncoder(
            cfg.input_dim, cfg.hidden_dim, cfg.n_lstm_layers, cfg.dropout_lstm
        )

        # Combine LSTM output + static features
        combined_dim = cfg.hidden_dim + static_dim
        self.shared = SharedRepresentation(combined_dim, cfg.shared_dim, cfg.shared_layers)

        # Cause-specific heads
        self.cause_nets = nn.ModuleList(
            [CauseSpecificNet(cfg.shared_dim, cfg.cause_dim, cfg.n_time_bins)
             for _ in range(cfg.n_causes)]
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
        -------
        pmf : (batch, n_causes, n_time_bins) — probability mass function
              each row sums to ≤ 1 (remainder = still-at-risk probability)
        """
        lstm_out = self.encoder(x_seq, seq_lengths)         # (B, hidden_dim)
        combined = torch.cat([lstm_out, x_static], dim=1)   # (B, hidden + static)
        shared = self.shared(combined)                       # (B, shared_dim)

        # Cause-specific logits → softmax over time bins per cause
        cause_logits = torch.stack(
            [head(shared) for head in self.cause_nets], dim=1
        )  # (B, n_causes, n_time_bins)

        # Joint softmax: all causes compete over all time bins
        flat = cause_logits.view(cause_logits.size(0), -1)   # (B, n_causes * T)
        pmf = F.softmax(flat, dim=-1).view_as(cause_logits)  # (B, n_causes, T)
        return pmf


# ─── Loss functions ───────────────────────────────────────────────────────────


def log_likelihood_loss(
    pmf: torch.Tensor,
    events: torch.LongTensor,
    time_bins: torch.LongTensor,
) -> torch.Tensor:
    """
    Negative log-likelihood for cause-specific survival.

    For uncensored subjects: -log P(T=t, cause=k)
    For censored subjects: -log S(t) = -log(1 - sum P(T<=t, all causes))
    """
    batch_size = pmf.size(0)
    n_causes = pmf.size(1)
    eps = 1e-8

    nll = torch.zeros(batch_size, device=pmf.device)

    # Cumulative hazard up to each time bin
    cum_pmf = pmf.cumsum(dim=2)  # (B, n_causes, T)

    for i in range(batch_size):
        t = time_bins[i].item()
        e = events[i].item()
        t_idx = min(int(t), pmf.size(2) - 1)

        if e > 0:
            # Observed event: cause index is e-1 (1-indexed)
            cause_idx = min(e - 1, n_causes - 1)
            prob = pmf[i, cause_idx, t_idx].clamp(min=eps)
            nll[i] = -torch.log(prob)
        else:
            # Censored: probability of surviving beyond t
            total_cum = cum_pmf[i, :, t_idx].sum().clamp(max=1 - eps)
            surv = (1.0 - total_cum).clamp(min=eps)
            nll[i] = -torch.log(surv)

    return nll.mean()


def ranking_loss(
    pmf: torch.Tensor,
    events: torch.LongTensor,
    time_bins: torch.LongTensor,
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    Ranking loss: penalise ordering violations between pairs.
    Encourages higher predicted risk for subjects with earlier events.
    """
    # For simplicity: sum predicted risk up to event time (per cause)
    batch_size = pmf.size(0)
    eps = 1e-8
    rank_loss = torch.tensor(0.0, device=pmf.device)
    n_pairs = 0

    # Cumulative risk for each subject (total across all causes)
    cum_risk = pmf.sum(dim=1).cumsum(dim=1)  # (B, T)

    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            e_i = events[i].item()
            t_i = time_bins[i].item()
            e_j = events[j].item()
            t_j = time_bins[j].item()

            # Only consider pairs where i had earlier event
            if e_i > 0 and t_i < t_j:
                t_idx = min(int(t_i), cum_risk.size(1) - 1)
                risk_i = cum_risk[i, t_idx]
                risk_j = cum_risk[j, t_idx]
                rank_loss += torch.exp(-(risk_i - risk_j) / sigma)
                n_pairs += 1

    return rank_loss / max(n_pairs, 1)


# ─── Trainer ─────────────────────────────────────────────────────────────────


class DynamicDeepHit:
    """
    High-level wrapper for training and inference.

    Usage
    -----
    model = DynamicDeepHit(cfg)
    model.fit(train_data, val_data)
    preds = model.predict(test_data, time_horizons=[3, 6, 12])
    """

    def __init__(self, cfg: Optional[DynamicDeepHitConfig] = None, static_dim: int = 16) -> None:
        self.cfg = cfg or DynamicDeepHitConfig()
        self.static_dim = static_dim
        self.net: Optional[DynamicDeepHitNet] = None
        self.scaler_seq = StandardScaler()
        self.scaler_static = StandardScaler()
        self._is_fitted = False
        self.time_bin_edges: Optional[np.ndarray] = None

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _discretise_time(self, durations: np.ndarray, fit: bool = False) -> np.ndarray:
        """Map continuous duration to bin index."""
        if fit:
            max_t = np.percentile(durations, 99)
            self.time_bin_edges = np.linspace(0, max_t, self.cfg.n_time_bins + 1)
        bins = np.digitize(durations, self.time_bin_edges[1:-1]).clip(
            0, self.cfg.n_time_bins - 1
        )
        return bins

    def _make_dataset(
        self,
        X_seq: np.ndarray,
        X_static: np.ndarray,
        events: np.ndarray,
        durations: np.ndarray,
        seq_lengths: np.ndarray,
        fit_scaler: bool = False,
    ) -> LongitudinalSurvivalDataset:
        # Normalise
        if fit_scaler:
            N, T, D = X_seq.shape
            self.scaler_seq.fit(X_seq.reshape(-1, D))
            self.scaler_static.fit(X_static)

        N, T, D = X_seq.shape
        X_seq_norm = self.scaler_seq.transform(X_seq.reshape(-1, D)).reshape(N, T, D)
        X_static_norm = self.scaler_static.transform(X_static)

        time_bins = self._discretise_time(durations, fit=fit_scaler)

        return LongitudinalSurvivalDataset(
            X_seq_norm, X_static_norm, events, time_bins, seq_lengths
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the Dynamic-DeepHit model.

        train_data keys: X_seq, X_static, events, durations, seq_lengths
        """
        mlflow.set_experiment(self.cfg.mlflow_experiment)

        static_dim = train_data["X_static"].shape[1]
        self.static_dim = static_dim
        self.net = DynamicDeepHitNet(self.cfg, static_dim).to(DEVICE)

        train_ds = self._make_dataset(
            train_data["X_seq"],
            train_data["X_static"],
            train_data["events"],
            train_data["durations"],
            train_data["seq_lengths"],
            fit_scaler=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        val_loader = None
        if val_data is not None:
            val_ds = self._make_dataset(
                val_data["X_seq"],
                val_data["X_static"],
                val_data["events"],
                val_data["durations"],
                val_data["seq_lengths"],
                fit_scaler=False,
            )
            val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False
        )

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = np.inf
        patience_counter = 0
        best_state = None

        with mlflow.start_run(run_name="dynamic_deephit_training"):
            mlflow.log_params(
                {
                    "epochs": self.cfg.epochs,
                    "hidden_dim": self.cfg.hidden_dim,
                    "n_lstm_layers": self.cfg.n_lstm_layers,
                    "n_causes": self.cfg.n_causes,
                    "alpha": self.cfg.alpha,
                }
            )

            for epoch in range(self.cfg.epochs):
                self.net.train()
                epoch_loss = 0.0
                n_batches = 0

                for x_seq, x_static, events, t_bins, seq_lens in train_loader:
                    x_seq = x_seq.to(DEVICE)
                    x_static = x_static.to(DEVICE)
                    events = events.to(DEVICE)
                    t_bins = t_bins.to(DEVICE)

                    optimizer.zero_grad()
                    pmf = self.net(x_seq, x_static, seq_lens)

                    nll = log_likelihood_loss(pmf, events, t_bins)
                    rank = ranking_loss(pmf, events, t_bins, self.cfg.sigma)
                    loss = nll + self.cfg.alpha * rank

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_train_loss = epoch_loss / max(n_batches, 1)
                history["train_loss"].append(avg_train_loss)

                # Validation
                if val_loader is not None:
                    val_loss = self._eval_loss(val_loader)
                    history["val_loss"].append(val_loss)
                    scheduler.step(val_loss)

                    if val_loss < best_val:
                        best_val = val_loss
                        best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if epoch % 10 == 0:
                        logger.info(
                            "Epoch %3d | train: %.4f | val: %.4f",
                            epoch + 1,
                            avg_train_loss,
                            val_loss,
                        )

                    if patience_counter >= self.cfg.patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break
                else:
                    if epoch % 10 == 0:
                        logger.info("Epoch %3d | train: %.4f", epoch + 1, avg_train_loss)

            # Restore best weights
            if best_state is not None:
                self.net.load_state_dict(best_state)

            mlflow.log_metrics(
                {
                    "final_train_loss": history["train_loss"][-1],
                    "best_val_loss": best_val if best_val < np.inf else -1,
                }
            )

        self._is_fitted = True
        return history

    def _eval_loss(self, loader: DataLoader) -> float:
        self.net.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for x_seq, x_static, events, t_bins, seq_lens in loader:
                x_seq, x_static = x_seq.to(DEVICE), x_static.to(DEVICE)
                events, t_bins = events.to(DEVICE), t_bins.to(DEVICE)
                pmf = self.net(x_seq, x_static, seq_lens)
                nll = log_likelihood_loss(pmf, events, t_bins)
                rank = ranking_loss(pmf, events, t_bins, self.cfg.sigma)
                loss = nll + self.cfg.alpha * rank
                total += loss.item()
                n += 1
        return total / max(n, 1)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        data: Dict[str, np.ndarray],
        time_horizons: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Predict cumulative incidence at given time horizons.

        Returns DataFrame with columns:
          - risk_{cause}_{horizon}m : P(event=cause by horizon months)
          - overall_risk_{horizon}m : P(any event by horizon months)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        time_horizons = time_horizons or self.cfg.time_horizons
        self.net.eval()

        N, T, D = data["X_seq"].shape
        X_seq_norm = self.scaler_seq.transform(data["X_seq"].reshape(-1, D)).reshape(N, T, D)
        X_static_norm = self.scaler_static.transform(data["X_static"])
        seq_lengths = data.get("seq_lengths", np.full(N, T))

        x_seq = torch.FloatTensor(X_seq_norm).to(DEVICE)
        x_static = torch.FloatTensor(X_static_norm).to(DEVICE)
        seq_lens = torch.LongTensor(seq_lengths)

        with torch.no_grad():
            pmf = self.net(x_seq, x_static, seq_lens).cpu().numpy()
        # pmf shape: (N, n_causes, n_time_bins)

        results: Dict[str, np.ndarray] = {}

        for horizon in time_horizons:
            # Find time bin corresponding to horizon months
            if self.time_bin_edges is not None:
                max_t = self.time_bin_edges[-1]
                horizon_bin = int(
                    horizon / max_t * self.cfg.n_time_bins
                )
                horizon_bin = min(horizon_bin, self.cfg.n_time_bins - 1)
            else:
                horizon_bin = min(horizon, self.cfg.n_time_bins - 1)

            # Cumulative incidence up to horizon
            cum_pmf = pmf[:, :, : horizon_bin + 1].sum(axis=2)  # (N, n_causes)
            overall = cum_pmf.sum(axis=1)  # (N,)

            results[f"overall_risk_{horizon}m"] = overall
            for c_idx, cause_name in enumerate(self.cfg.cause_names):
                results[f"risk_{cause_name}_{horizon}m"] = cum_pmf[:, c_idx]

        return pd.DataFrame(results)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "dynamic_deephit.pt"
        state = {
            "cfg": self.cfg,
            "static_dim": self.static_dim,
            "net_state": self.net.state_dict() if self.net else None,
            "scaler_seq": self.scaler_seq,
            "scaler_static": self.scaler_static,
            "time_bin_edges": self.time_bin_edges,
        }
        torch.save(state, path)
        logger.info("Dynamic-DeepHit saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "DynamicDeepHit":
        state = torch.load(path, map_location="cpu")
        instance = cls(cfg=state["cfg"], static_dim=state["static_dim"])
        if state["net_state"] is not None:
            instance.net = DynamicDeepHitNet(state["cfg"], state["static_dim"])
            instance.net.load_state_dict(state["net_state"])
            instance.net.to(DEVICE)
        instance.scaler_seq = state["scaler_seq"]
        instance.scaler_static = state["scaler_static"]
        instance.time_bin_edges = state["time_bin_edges"]
        instance._is_fitted = True
        logger.info("Dynamic-DeepHit loaded from %s", path)
        return instance


# ─── Synthetic smoke test ─────────────────────────────────────────────────────


def _make_synthetic_longitudinal(
    n: int = 200,
    max_seq_len: int = 12,
    input_dim: int = 8,
    static_dim: int = 6,
) -> Dict[str, np.ndarray]:
    """Generate small synthetic longitudinal survival dataset."""
    rng = np.random.default_rng(42)
    seq_lengths = rng.integers(3, max_seq_len + 1, n)
    X_seq = np.zeros((n, max_seq_len, input_dim))
    for i in range(n):
        l = seq_lengths[i]
        X_seq[i, :l, :] = rng.standard_normal((l, input_dim))

    X_static = rng.standard_normal((n, static_dim))
    events = rng.integers(0, 4, n)   # 0=censored, 1/2/3 causes
    durations = rng.uniform(1, 36, n)

    return {
        "X_seq": X_seq.astype(np.float32),
        "X_static": X_static.astype(np.float32),
        "events": events.astype(np.int64),
        "durations": durations.astype(np.float32),
        "seq_lengths": seq_lengths.astype(np.int64),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Dynamic-DeepHit Smoke Test ===")

    cfg = DynamicDeepHitConfig(
        input_dim=8,
        hidden_dim=32,
        n_lstm_layers=1,
        shared_dim=16,
        n_causes=3,
        cause_dim=16,
        epochs=5,
        batch_size=32,
        max_seq_len=12,
        n_time_bins=12,
    )

    train_data = _make_synthetic_longitudinal(160, max_seq_len=12, input_dim=8, static_dim=6)
    val_data = _make_synthetic_longitudinal(40, max_seq_len=12, input_dim=8, static_dim=6)

    model = DynamicDeepHit(cfg, static_dim=6)
    history = model.fit(train_data, val_data)
    logger.info("Train loss history: %s", history["train_loss"])

    preds = model.predict(val_data, time_horizons=[3, 6, 12])
    logger.info("Predictions shape: %s", preds.shape)
    logger.info("Prediction columns: %s", list(preds.columns))
    assert preds.shape == (40, 12)  # 3 horizons × (3 causes + 1 overall)
    assert (preds.values >= 0).all()

    path = model.save()
    loaded = DynamicDeepHit.load(path)
    preds2 = loaded.predict(val_data)
    np.testing.assert_allclose(preds.values, preds2.values, atol=1e-5)
    logger.info("=== PASS ===")
