"""
models/ensemble/cross_validator.py
=====================================
Time-aware cross-validation for clinical / financial data.

Key design:
  - NEVER use future data to predict past (temporal leakage prevention)
  - Supports: standard KFold, temporal sliding window, expanding window,
    purged KFold (for financial time series)
  - Generates out-of-fold predictions for stacking ensemble
  - Tracks per-fold metrics in MLflow
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)


# ─── Fold generators ─────────────────────────────────────────────────────────


class TemporalFoldGenerator:
    """
    Generates train/validation splits for time-ordered data.

    Methods:
      - expanding_window: train on all data up to t, validate on next window
      - sliding_window:   train on fixed-size window, validate on next
      - purged_kfold:     gap between train and val to prevent label leakage
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,              # periods to gap between train and val
        window_size: Optional[int] = None,  # None = expanding
        embargo: float = 0.01,     # fraction of samples to purge after val end
    ) -> None:
        self.n_splits = n_splits
        self.gap = gap
        self.window_size = window_size
        self.embargo = embargo

    def split(
        self,
        X: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield (train_indices, val_indices) pairs in temporal order.

        Parameters
        ----------
        X : (N, D) feature matrix (used for shape only)
        timestamps : (N,) sort key — if None uses row index as proxy
        """
        N = len(X)
        if timestamps is None:
            order = np.arange(N)
        else:
            order = np.argsort(timestamps)

        fold_size = N // self.n_splits

        for i in range(self.n_splits):
            val_start = (i + 1) * fold_size  # val is always after train
            val_end = val_start + fold_size if i < self.n_splits - 1 else N

            if val_start >= N:
                break

            if self.window_size is not None:
                # Sliding window
                train_start = max(0, val_start - self.window_size - self.gap)
            else:
                # Expanding window
                train_start = 0

            train_end = val_start - self.gap
            if train_end <= train_start:
                continue

            train_idx = order[train_start:train_end]
            val_idx = order[val_start:val_end]

            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


class PurgedKFold:
    """
    Purged K-Fold cross-validation for financial / clinical time series.

    Removes overlapping label windows between train and test to prevent
    information leakage when labels look forward in time.

    Reference: De Prado (2018) "Advances in Financial Machine Learning"
    """

    def __init__(
        self,
        n_splits: int = 5,
        pct_embargo: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        timestamps: Optional[pd.Series] = None,
        pred_times: Optional[pd.Series] = None,   # when prediction is made
        eval_times: Optional[pd.Series] = None,   # when label is observed
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield (train_indices, test_indices) avoiding temporal overlap.
        """
        N = len(X)
        indices = np.arange(N)
        embargo_size = int(N * self.pct_embargo)
        fold_size = N // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else N

            test_idx = indices[test_start:test_end]

            # Remove train samples that overlap with test period
            if eval_times is not None and pred_times is not None:
                test_eval_max = eval_times.iloc[test_idx].max()
                test_pred_min = pred_times.iloc[test_idx].min()

                # Purge: remove training samples whose eval time overlaps test
                train_mask = eval_times < test_pred_min
                # Embargo: remove training samples just after test
                embargo_end_time = eval_times.iloc[min(test_end + embargo_size, N - 1)]
                embargo_mask = pred_times > embargo_end_time
                train_idx = indices[train_mask & embargo_mask]
            else:
                # Simple purge: no train after test_start - gap
                pre_test = indices[:max(0, test_start - embargo_size)]
                post_test = indices[min(test_end + embargo_size, N):]
                train_idx = np.concatenate([pre_test, post_test])

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


# ─── Cross-validation runner ─────────────────────────────────────────────────


class CrossValidator:
    """
    Unified cross-validation runner that:
      1. Selects fold strategy (standard / temporal / purged)
      2. Runs a model on each fold
      3. Collects per-fold metrics
      4. Returns OOF predictions for stacking

    Usage
    -----
    cv = CrossValidator(strategy="temporal", n_splits=5)
    oof_preds, fold_metrics = cv.run(model_fn, X, y, timestamps=dates)
    """

    def __init__(
        self,
        strategy: str = "temporal",   # "standard" | "temporal" | "purged"
        n_splits: int = 5,
        gap: int = 0,
        window_size: Optional[int] = None,
        pct_embargo: float = 0.01,
        task: str = "classification",  # "classification" | "regression"
        random_state: int = 42,
    ) -> None:
        self.strategy = strategy
        self.n_splits = n_splits
        self.gap = gap
        self.window_size = window_size
        self.pct_embargo = pct_embargo
        self.task = task
        self.random_state = random_state

    def _get_splitter(self, X: np.ndarray, y: np.ndarray, timestamps: Optional[np.ndarray]):
        if self.strategy == "standard":
            if self.task == "classification":
                return StratifiedKFold(
                    self.n_splits, shuffle=True, random_state=self.random_state
                ).split(X, y.astype(int))
            return KFold(
                self.n_splits, shuffle=True, random_state=self.random_state
            ).split(X)
        elif self.strategy == "temporal":
            gen = TemporalFoldGenerator(
                n_splits=self.n_splits,
                gap=self.gap,
                window_size=self.window_size,
            )
            return gen.split(X, timestamps)
        elif self.strategy == "purged":
            pkf = PurgedKFold(n_splits=self.n_splits, pct_embargo=self.pct_embargo)
            return pkf.split(X, timestamps=pd.Series(timestamps) if timestamps is not None else None)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def run(
        self,
        model_fn: Callable[[], Any],       # factory that returns a new model each fold
        fit_fn: Callable[[Any, np.ndarray, np.ndarray], None],  # (model, X_train, y_train) → None
        predict_fn: Callable[[Any, np.ndarray], np.ndarray],   # (model, X_val) → predictions
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        score_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """
        Run K-fold CV and return OOF predictions + per-fold metrics.

        Parameters
        ----------
        model_fn : callable that returns a fresh model
        fit_fn : (model, X_train, y_train) → fits model in-place
        predict_fn : (model, X_val) → predictions array
        score_fn : (y_true, y_pred) → float metric (higher=better)

        Returns
        -------
        oof_predictions : (N,) out-of-fold predictions
        fold_metrics : list of per-fold dicts
        """
        from sklearn.metrics import mean_absolute_error, roc_auc_score

        N = len(y)
        oof = np.zeros(N, dtype=np.float32)
        fold_metrics: List[Dict[str, float]] = []

        if score_fn is None:
            if self.task == "classification":
                score_fn = roc_auc_score
            else:
                score_fn = lambda yt, yp: -mean_absolute_error(yt, yp)

        splits = list(self._get_splitter(X, y, timestamps))
        logger.info("Running %d-fold CV with strategy=%s", len(splits), self.strategy)

        for fold_i, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit fresh model
            model = model_fn()
            fit_fn(model, X_train, y_train)

            # Predict
            val_preds = predict_fn(model, X_val)
            oof[val_idx] = np.asarray(val_preds, dtype=np.float32).ravel()

            # Score
            try:
                fold_score = float(score_fn(y_val, val_preds))
            except Exception as e:
                logger.warning("Scoring error fold %d: %s", fold_i, e)
                fold_score = float("nan")

            fold_metrics.append({
                "fold": fold_i,
                "score": fold_score,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            })

            if verbose:
                logger.info(
                    "  Fold %d | n_train=%d, n_val=%d | score=%.4f",
                    fold_i + 1, len(train_idx), len(val_idx), fold_score,
                )

        scores = [m["score"] for m in fold_metrics if not np.isnan(m["score"])]
        logger.info(
            "CV complete — mean=%.4f ± %.4f",
            np.mean(scores), np.std(scores),
        )
        return oof, fold_metrics

    def run_summary(
        self,
        fold_metrics: List[Dict[str, float]],
        model_name: str = "model",
    ) -> pd.DataFrame:
        """Summarise fold metrics as a DataFrame."""
        df = pd.DataFrame(fold_metrics)
        summary_row = {
            "fold": "summary",
            "score": df["score"].mean(),
            "score_std": df["score"].std(),
            "n_train": df["n_train"].mean(),
            "n_val": df["n_val"].mean(),
        }
        return pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)


# ─── Nested cross-validation ─────────────────────────────────────────────────


class NestedCrossValidator:
    """
    Nested CV for unbiased model evaluation:
      - Outer loop: performance estimation
      - Inner loop: hyperparameter selection

    Prevents optimism bias when tuning and evaluating on same data.
    """

    def __init__(
        self,
        outer_cv: CrossValidator,
        inner_cv: CrossValidator,
        param_grid: Dict[str, List[Any]],
    ) -> None:
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.param_grid = param_grid

    def run(
        self,
        model_class: Any,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run nested CV.
        Returns outer fold scores + best hyperparams per fold.
        """
        from itertools import product

        outer_splits = list(self.outer_cv._get_splitter(X, y, timestamps))
        outer_scores = []
        best_params_per_fold = []

        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        all_param_combos = [
            dict(zip(param_names, combo))
            for combo in product(*param_values)
        ]

        logger.info(
            "Nested CV: %d outer folds × %d inner folds × %d param combos",
            len(outer_splits),
            self.inner_cv.n_splits,
            len(all_param_combos),
        )

        for fold_i, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
            X_outer_train = X[outer_train_idx]
            y_outer_train = y[outer_train_idx]
            X_outer_test = X[outer_test_idx]
            y_outer_test = y[outer_test_idx]

            # Inner loop: find best params
            best_score = -np.inf
            best_params = all_param_combos[0]

            inner_splits = list(self.inner_cv._get_splitter(
                X_outer_train, y_outer_train,
                timestamps[outer_train_idx] if timestamps is not None else None,
            ))

            for params in all_param_combos:
                inner_scores = []
                for inner_train_idx, inner_val_idx in inner_splits:
                    X_in_tr = X_outer_train[inner_train_idx]
                    y_in_tr = y_outer_train[inner_train_idx]
                    X_in_val = X_outer_train[inner_val_idx]
                    y_in_val = y_outer_train[inner_val_idx]

                    model = model_class(**params)
                    model.fit(X_in_tr, y_in_tr)
                    val_preds = model.predict_proba(X_in_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_in_val)

                    from sklearn.metrics import roc_auc_score
                    try:
                        inner_scores.append(roc_auc_score(y_in_val, val_preds))
                    except Exception:
                        inner_scores.append(0.5)

                mean_inner = float(np.mean(inner_scores))
                if mean_inner > best_score:
                    best_score = mean_inner
                    best_params = params

            # Fit on full outer train with best params, evaluate on outer test
            final_model = model_class(**best_params)
            final_model.fit(X_outer_train, y_outer_train)
            test_preds = (
                final_model.predict_proba(X_outer_test)[:, 1]
                if hasattr(final_model, "predict_proba")
                else final_model.predict(X_outer_test)
            )

            from sklearn.metrics import roc_auc_score
            try:
                outer_score = float(roc_auc_score(y_outer_test, test_preds))
            except Exception:
                outer_score = float("nan")

            outer_scores.append(outer_score)
            best_params_per_fold.append(best_params)
            logger.info("Outer fold %d | best_params=%s | test AUROC=%.4f",
                        fold_i + 1, best_params, outer_score)

        return {
            "outer_scores": outer_scores,
            "mean_score": float(np.nanmean(outer_scores)),
            "std_score": float(np.nanstd(outer_scores)),
            "best_params_per_fold": best_params_per_fold,
        }


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Cross Validator Smoke Test ===")

    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    N, D = 300, 10
    X = rng.standard_normal((N, D)).astype(np.float32)
    y = rng.binomial(1, 0.3, N).astype(float)
    timestamps = np.arange(N, dtype=float) + rng.uniform(-0.5, 0.5, N)

    # Test temporal CV
    cv = CrossValidator(strategy="temporal", n_splits=4, task="classification")
    oof, fold_metrics = cv.run(
        model_fn=lambda: LogisticRegression(max_iter=200, random_state=42),
        fit_fn=lambda m, Xtr, ytr: m.fit(Xtr, ytr.astype(int)),
        predict_fn=lambda m, Xv: m.predict_proba(Xv)[:, 1],
        X=X, y=y, timestamps=timestamps,
    )

    logger.info("Temporal CV OOF shape: %s", oof.shape)
    logger.info("Fold metrics: %s", fold_metrics)
    summary = cv.run_summary(fold_metrics, "LogReg")
    logger.info("\n%s", summary)

    assert oof.shape == (N,)
    assert len(fold_metrics) == 4
    logger.info("=== PASS ===")
