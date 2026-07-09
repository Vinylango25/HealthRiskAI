"""
models/survival/deepsurv.py
============================
DeepSurv neural survival model (Faraggi-Simon network).

Implements the DeepSurv model using ``pycox.models.CoxPH`` with a
``torchtuples`` neural network backbone. Supports early stopping, MLflow
tracking of C-index every 10 epochs, and survival function prediction.

Reference:
    Katzman et al. (2018). DeepSurv: Personalised Treatment Recommender System
    Using a Cox Proportional Hazards Deep Neural Network.
    BMC Medical Research Methodology, 18, 24.
    https://doi.org/10.1186/s12874-018-0482-1

Author: HealthRisk AI Team
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger

# ── Optional deep-learning dependencies ────────────────────────────────────────
try:
    import torch
    import torchtuples as tt
    from pycox.evaluation import EvalSurv
    from pycox.models import CoxPH as PycoxCoxPH
    _PYCOX_AVAILABLE = True
except ImportError:
    _PYCOX_AVAILABLE = False
    logger.warning(
        "pycox / torchtuples not installed. "
        "Install with: pip install pycox torchtuples torch"
    )

try:
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from models.tabular.trainer import MLflowTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parents[2] / "configs" / "model_config.yaml"


def _load_config() -> dict:
    """Load model configuration YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}


# ---------------------------------------------------------------------------
# _MLP: internal torchtuples MLP builder
# ---------------------------------------------------------------------------

def _build_net(
    in_features: int,
    hidden_layers: list[int],
    dropout: float,
    activation: str = "selu",
    batch_norm: bool = True,
) -> "tt.practical.MLPVanilla":
    """
    Build a torchtuples MLPVanilla network.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_layers : list[int]
        Hidden layer sizes.
    dropout : float
        Dropout probability applied after each hidden layer.
    activation : str
        Activation function name ('selu', 'relu', 'elu').
    batch_norm : bool
        Whether to apply batch normalisation.

    Returns
    -------
    tt.practical.MLPVanilla
    """
    # torchtuples MLPVanilla signature:
    # (in_features, num_nodes, out_features, batch_norm, dropout, activation)
    activation_map = {
        "selu": torch.nn.SELU,
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "tanh": torch.nn.Tanh,
    }
    act_cls = activation_map.get(activation.lower(), torch.nn.SELU)

    return tt.practical.MLPVanilla(
        in_features=in_features,
        num_nodes=hidden_layers,
        out_features=1,
        batch_norm=batch_norm,
        dropout=dropout,
        activation=act_cls,
        output_bias=False,
    )


# ---------------------------------------------------------------------------
# DeepSurvModel
# ---------------------------------------------------------------------------


class DeepSurvModel:
    """
    Faraggi-Simon DeepSurv model for survival analysis.

    A neural network extension of the Cox Proportional Hazards model where
    the log-partial hazard is modelled by a multi-layer perceptron. Training
    minimises the negative partial log-likelihood (Cox loss).

    Parameters
    ----------
    hidden_layers : list[int]
        Number of nodes in each hidden layer. Default [64, 64].
    dropout : float
        Dropout rate applied after each hidden layer. Default 0.3.
    lr : float
        Adam learning rate. Default 0.001.
    batch_size : int
        Mini-batch size. Default 256.
    epochs : int
        Maximum number of training epochs. Default 100.

    Attributes
    ----------
    model : pycox.models.CoxPH or None
        Fitted DeepSurv model.
    scaler : sklearn.preprocessing.StandardScaler or None
        Fitted feature scaler.
    net : torchtuples MLPVanilla or None
        Underlying PyTorch network.
    """

    def __init__(
        self,
        hidden_layers: Optional[list[int]] = None,
        dropout: float = 0.3,
        lr: float = 0.001,
        batch_size: int = 256,
        epochs: int = 100,
    ) -> None:
        if not _PYCOX_AVAILABLE:
            raise ImportError(
                "pycox and torchtuples are required. "
                "Install with: pip install pycox torchtuples torch"
            )

        raw_config = _load_config()
        ds_cfg = raw_config.get("survival", {}).get("deepsurv", {})

        self.hidden_layers: list[int] = (
            hidden_layers
            or ds_cfg.get("hidden_layers", [64, 64])
        )
        self.dropout: float = dropout or float(ds_cfg.get("dropout", 0.3))
        self.lr: float = lr or float(ds_cfg.get("learning_rate", 0.001))
        self.batch_size: int = batch_size or int(ds_cfg.get("batch_size", 256))
        self.epochs: int = epochs or int(ds_cfg.get("num_epochs", 100))
        self.activation: str = ds_cfg.get("activation", "selu")
        self.batch_norm: bool = bool(ds_cfg.get("batch_norm", True))

        self.model: Optional[PycoxCoxPH] = None
        self.net: Optional[tt.practical.MLPVanilla] = None
        self.scaler: Optional[StandardScaler] = None
        self._baseline_hazards: Optional[pd.Series] = None
        self._feature_cols: list[str] = []

        mlflow_uri = raw_config.get("training", {}).get("mlflow_tracking_uri")
        self.tracker = MLflowTracker(
            experiment_name="healthrisk/survival_deepsurv",
            tracking_uri=mlflow_uri,
        )

        logger.info(
            f"DeepSurvModel initialised — hidden_layers={self.hidden_layers}, "
            f"dropout={self.dropout}, lr={self.lr}, "
            f"batch_size={self.batch_size}, epochs={self.epochs}"
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        feature_cols: list[str],
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Normalise features and format data for pycox training.

        pycox expects:
        - ``x``: float32 feature matrix, shape (n_samples, n_features)
        - ``(durations, events)``: 1D float32 and float32 arrays

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        duration_col : str
            Time-to-event column.
        event_col : str
            Event indicator column (1=event, 0=censored).
        feature_cols : list[str]
            Feature columns to use.

        Returns
        -------
        x : np.ndarray, shape (n_samples, n_features)
        (durations, events) : tuple[np.ndarray, np.ndarray]
        """
        available = [c for c in feature_cols if c in df.columns]
        missing = set(feature_cols) - set(available)
        if missing:
            logger.warning(
                f"prepare_data: {len(missing)} columns missing — "
                f"filling with 0: {sorted(missing)}"
            )
            df = df.copy()
            for col in missing:
                df[col] = 0.0

        # Fill NaN with median
        x_raw = df[feature_cols].copy()
        for col in feature_cols:
            if x_raw[col].isna().any():
                x_raw[col].fillna(x_raw[col].median(), inplace=True)

        # Normalise
        if self.scaler is None:
            self.scaler = StandardScaler()
            x = self.scaler.fit_transform(x_raw.values.astype(np.float32))
        else:
            x = self.scaler.transform(x_raw.values.astype(np.float32))

        durations = df[duration_col].values.astype(np.float32)
        events = df[event_col].values.astype(np.float32)

        logger.debug(
            f"prepare_data: x.shape={x.shape}, "
            f"events={int(events.sum())}/{len(events)}"
        )
        return x, (durations, events)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        feature_cols: list[str],
        run_name: str = "deepsurv_v1",
    ) -> dict:
        """
        Train the DeepSurv model with early stopping.

        Logs C-index on the validation set to MLflow every 10 epochs.
        Early stopping monitors validation C-index with a patience of 15
        epochs.

        Training uses a 90/10 random train/validation split.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.
        duration_col : str
            Time-to-event column.
        event_col : str
            Event indicator column.
        feature_cols : list[str]
            Feature columns.
        run_name : str
            MLflow run name.

        Returns
        -------
        dict with keys:
            ``best_c_index`` (float), ``train_history`` (list[dict]).
        """
        if not _PYCOX_AVAILABLE:
            raise ImportError("pycox and torchtuples are required.")

        self._feature_cols = feature_cols

        # 90/10 train/val split
        val_size = max(1, int(0.10 * len(df)))
        idx = np.arange(len(df))
        np.random.default_rng(42).shuffle(idx)
        val_idx, train_idx = idx[:val_size], idx[val_size:]

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        logger.info(
            f"Training DeepSurv: {len(df_train)} train, {len(df_val)} val samples."
        )

        # Reset scaler so prepare_data fits on train set
        self.scaler = None
        x_train, (dur_train, ev_train) = self.prepare_data(
            df_train, duration_col, event_col, feature_cols
        )
        x_val, (dur_val, ev_val) = self.prepare_data(
            df_val, duration_col, event_col, feature_cols
        )

        in_features = x_train.shape[1]
        self.net = _build_net(
            in_features=in_features,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
        )

        self.model = PycoxCoxPH(
            net=self.net,
            optimizer=tt.optim.Adam(lr=self.lr),
        )

        # Convert to pycox tensors
        x_train_t = x_train.astype(np.float32)
        x_val_t = x_val.astype(np.float32)

        # Manual training loop with early stopping and MLflow logging
        patience = 15
        best_c_index = 0.0
        best_weights: Optional[dict] = None
        no_improve_count = 0
        train_history: list[dict] = []

        self.tracker.start_run(run_name)
        self.tracker.log_params(
            {
                "hidden_layers": str(self.hidden_layers),
                "dropout": self.dropout,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "n_train": len(df_train),
                "n_val": len(df_val),
                "n_features": in_features,
            }
        )

        logger.info(
            f"Starting training for up to {self.epochs} epochs "
            f"(early stopping patience={patience})…"
        )

        for epoch in range(1, self.epochs + 1):
            # One epoch of training
            train_loss = self.model.fit(
                x_train_t,
                (dur_train, ev_train),
                batch_size=self.batch_size,
                epochs=1,          # train for exactly 1 epoch at a time
                verbose=False,
                val_data=(x_val_t, (dur_val, ev_val)),
            )

            # Compute C-index on validation set every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                try:
                    # Compute baseline hazards then survival function
                    self.model.compute_baseline_hazards()
                    surv_df = self.model.predict_surv_df(x_val_t)
                    ev_val_obj = EvalSurv(
                        surv_df,
                        dur_val,
                        ev_val,
                        censor_surv="km",
                    )
                    c_idx = float(ev_val_obj.concordance_td())
                except Exception as exc:
                    logger.warning(f"Epoch {epoch}: C-index computation failed: {exc}")
                    c_idx = 0.0

                epoch_log = {"epoch": epoch, "c_index": c_idx}
                train_history.append(epoch_log)

                self.tracker.log_metrics({"val_c_index": c_idx}, step=epoch)
                logger.info(f"Epoch {epoch:3d}/{self.epochs} — val C-index={c_idx:.4f}")

                # Early stopping check
                if c_idx > best_c_index:
                    best_c_index = c_idx
                    # Deep-copy weights
                    best_weights = {
                        k: v.clone() for k, v in self.net.state_dict().items()
                    }
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch}. "
                            f"Best C-index={best_c_index:.4f}"
                        )
                        break

        # Restore best weights
        if best_weights is not None:
            self.net.load_state_dict(best_weights)
            logger.info(f"Restored best weights (C-index={best_c_index:.4f}).")

        # Final baseline hazard computation on full training set
        self.model.compute_baseline_hazards()

        # Log final metrics
        self.tracker.log_metrics({"best_val_c_index": best_c_index})
        self.tracker.end_run()

        logger.info(
            f"DeepSurv training complete. Best val C-index={best_c_index:.4f}"
        )
        return {
            "best_c_index": best_c_index,
            "train_history": train_history,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_risk(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> np.ndarray:
        """
        Predict the log partial hazard (risk score) per patient.

        Higher values indicate higher risk (shorter expected survival).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        feature_cols : list[str]
            Feature columns (must match those used during training).

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Log partial hazard scores.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        x, _ = self.prepare_data(
            df,
            duration_col="_dummy_dur",
            event_col="_dummy_ev",
            feature_cols=feature_cols,
        ) if False else self._transform_features(df, feature_cols)

        risk_scores = self.model.predict(x)  # log partial hazard
        return np.asarray(risk_scores).ravel()

    def _transform_features(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> np.ndarray:
        """Apply fitted scaler to features (no target needed)."""
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call train() first.")

        available = [c for c in feature_cols if c in df.columns]
        x_raw = df[available].copy()
        for col in available:
            if x_raw[col].isna().any():
                x_raw[col].fillna(x_raw[col].median(), inplace=True)

        return self.scaler.transform(x_raw.values.astype(np.float32))

    def predict_survival_function(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        times: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """
        Predict survival probabilities at specified time points.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        feature_cols : list[str]
            Feature columns.
        times : list[float], optional
            Time points (days) at which to evaluate. Default [7, 14, 30, 60, 90].

        Returns
        -------
        pd.DataFrame
            Long-format table with columns:
            ``patient_id``, ``time``, ``survival_prob``.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if times is None:
            times = [7.0, 14.0, 30.0, 60.0, 90.0]

        x = self._transform_features(df, feature_cols)

        # pycox returns (time_grid × n_patients) survival DataFrame
        surv_df = self.model.predict_surv_df(x)

        patient_ids = (
            df["patient_id"].values
            if "patient_id" in df.columns
            else np.arange(len(df))
        )

        rows = []
        for col_idx, pid in enumerate(patient_ids):
            col = surv_df.columns[col_idx]
            for t in times:
                # Find nearest time in the survival grid
                nearest_idx = (surv_df.index - t).abs().argmin()
                sp = float(surv_df.iloc[nearest_idx, col_idx])
                rows.append({"patient_id": pid, "time": t, "survival_prob": sp})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the DeepSurv model to disk.

        Saves the PyTorch network weights separately (as a ``.pt`` file)
        alongside a pickle for the scaler and metadata.

        Parameters
        ----------
        path : str
            Base file path (without extension). Two files are created:
            ``<path>.pkl`` and ``<path>_weights.pt``.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        base = Path(path).with_suffix("")

        # Save PyTorch weights
        weights_path = str(base) + "_weights.pt"
        if self.net is not None:
            torch.save(self.net.state_dict(), weights_path)
            logger.info(f"Network weights saved to '{weights_path}'.")

        payload = {
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "activation": self.activation,
            "batch_norm": self.batch_norm,
            "scaler": self.scaler,
            "_feature_cols": self._feature_cols,
            "weights_path": weights_path,
        }

        pkl_path = str(base) + ".pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"DeepSurvModel metadata saved to '{pkl_path}'.")

    def load(self, path: str) -> None:
        """
        Load the DeepSurv model from disk.

        Parameters
        ----------
        path : str
            Base file path (without extension) as used in ``save``.
        """
        if not _PYCOX_AVAILABLE:
            raise ImportError("pycox and torchtuples are required.")

        base = Path(path).with_suffix("")
        pkl_path = str(base) + ".pkl"
        weights_path = str(base) + "_weights.pt"

        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        self.hidden_layers = payload["hidden_layers"]
        self.dropout = payload["dropout"]
        self.lr = payload["lr"]
        self.batch_size = payload["batch_size"]
        self.epochs = payload["epochs"]
        self.activation = payload.get("activation", "selu")
        self.batch_norm = payload.get("batch_norm", True)
        self.scaler = payload["scaler"]
        self._feature_cols = payload.get("_feature_cols", [])

        # Rebuild net and model
        if self.scaler is not None:
            in_features = self.scaler.n_features_in_
            self.net = _build_net(
                in_features=in_features,
                hidden_layers=self.hidden_layers,
                dropout=self.dropout,
                activation=self.activation,
                batch_norm=self.batch_norm,
            )
            self.model = PycoxCoxPH(
                net=self.net,
                optimizer=tt.optim.Adam(lr=self.lr),
            )

            # Load weights
            if Path(weights_path).exists():
                state_dict = torch.load(
                    weights_path, map_location=torch.device("cpu")
                )
                self.net.load_state_dict(state_dict)
                logger.info(f"Network weights loaded from '{weights_path}'.")
            else:
                logger.warning(
                    f"Weights file '{weights_path}' not found. "
                    "Model architecture restored but weights are random."
                )

        logger.info(f"DeepSurvModel loaded from '{pkl_path}'.")
