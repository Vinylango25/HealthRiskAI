"""
data.processing.splitter — Train / validation / test splitting utilities.

Provides :class:`DataSplitter` with four splitting strategies:

- **random** — random shuffle split (default)
- **temporal** — time-ordered split (no future leakage)
- **stratified** — stratified by target class distribution
- **group** — group-aware split (no patient / subject leakage across folds)

All strategies return ``(df_train, df_val, df_test)`` triples and preserve
the original DataFrame index on each subset.

Dependencies
------------
scikit-learn is used *only* for the ``StratifiedShuffleSplit``,
``GroupShuffleSplit``, ``KFold``, ``StratifiedKFold``, and
``GroupKFold`` classes in the splitter internals.  All data manipulation
is done with numpy / pandas.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# sklearn is used only for CV splitters — not for any data transformation.
try:
    from sklearn.model_selection import (
        GroupKFold,
        GroupShuffleSplit,
        KFold,
        StratifiedKFold,
        StratifiedShuffleSplit,
        TimeSeriesSplit,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not installed — stratified/group splits and k-fold will "
        "fall back to random splitting."
    )

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
SplitTriple = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
FoldList = List[Tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# DataSplitter
# ---------------------------------------------------------------------------


class DataSplitter:
    """Train / validation / test splitting for clinical ML datasets.

    Parameters
    ----------
    test_size:
        Fraction of data reserved for test set.  Default ``0.2``.
    val_size:
        Fraction of **total** data reserved for validation set.
        Default ``0.1``.
    strategy:
        Default splitting strategy: ``'random'``, ``'temporal'``,
        ``'stratified'``, or ``'group'``.
    random_state:
        Integer seed for reproducibility.

    Examples
    --------
    >>> splitter = DataSplitter(test_size=0.2, val_size=0.1)
    >>> train, val, test = splitter.split(df, target_col="readmit_30d")
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        strategy: str = "random",
        random_state: int = 42,
    ) -> None:
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
        if not 0 < val_size < 1:
            raise ValueError(f"val_size must be in (0, 1), got {val_size}.")
        if test_size + val_size >= 1.0:
            raise ValueError(
                f"test_size + val_size = {test_size + val_size:.2f} ≥ 1.0; "
                "leave room for training data."
            )

        self.test_size = test_size
        self.val_size = val_size
        self.strategy = strategy
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_split(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, strategy: str
    ) -> None:
        total = len(train) + len(val) + len(test)
        logger.info(
            "split [%s]: total=%d | train=%d (%.1f%%) | val=%d (%.1f%%) | test=%d (%.1f%%)",
            strategy,
            total,
            len(train), 100 * len(train) / total,
            len(val), 100 * len(val) / total,
            len(test), 100 * len(test) / total,
        )

    def _random_indices(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (train_idx, val_idx, test_idx) arrays for *n* rows."""
        idx = self._rng.permutation(n)
        n_test = max(1, int(n * self.test_size))
        n_val = max(1, int(n * self.val_size))
        test_idx = idx[:n_test]
        val_idx = idx[n_test: n_test + n_val]
        train_idx = idx[n_test + n_val:]
        return train_idx, val_idx, test_idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        df: pd.DataFrame,
        target_col: str,
        timestamp_col: Optional[str] = None,
    ) -> SplitTriple:
        """Split *df* using the configured default strategy.

        Routes to :meth:`temporal_split` when *timestamp_col* is provided
        and ``strategy == 'temporal'``, otherwise uses the instance's
        ``strategy`` attribute.

        Parameters
        ----------
        df:
            DataFrame to split.
        target_col:
            Name of the target / label column.
        timestamp_col:
            Optional datetime column.  Used automatically when strategy is
            ``'temporal'``.

        Returns
        -------
        tuple[df_train, df_val, df_test]
        """
        if self.strategy == "temporal" and timestamp_col:
            return self.temporal_split(df, timestamp_col)
        elif self.strategy == "stratified":
            return self.stratified_split(df, target_col)
        else:
            return self._random_split(df)

    def _random_split(self, df: pd.DataFrame) -> SplitTriple:
        """Pure random split (no stratification)."""
        train_idx, val_idx, test_idx = self._random_indices(len(df))
        train = df.iloc[train_idx].copy()
        val = df.iloc[val_idx].copy()
        test = df.iloc[test_idx].copy()
        self._log_split(train, val, test, "random")
        return train, val, test

    def temporal_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        test_frac: Optional[float] = None,
        val_frac: Optional[float] = None,
    ) -> SplitTriple:
        """Time-ordered train / val / test split.

        Rows are sorted ascending by *timestamp_col*.  The **earliest**
        rows become train, the middle band is validation, and the
        **latest** rows become test — ensuring no future leakage.

        Parameters
        ----------
        df:
            DataFrame to split.
        timestamp_col:
            Column with datetime-like values.
        test_frac:
            Override :attr:`test_size` for this call.
        val_frac:
            Override :attr:`val_size` for this call.

        Returns
        -------
        tuple[df_train, df_val, df_test]

        Raises
        ------
        KeyError
            If *timestamp_col* is not in *df*.
        """
        if timestamp_col not in df.columns:
            raise KeyError(f"temporal_split: column '{timestamp_col}' not found.")

        t_frac = test_frac if test_frac is not None else self.test_size
        v_frac = val_frac if val_frac is not None else self.val_size

        df_sorted = df.sort_values(timestamp_col).copy()
        n = len(df_sorted)

        n_test = max(1, int(n * t_frac))
        n_val = max(1, int(n * v_frac))
        n_train = n - n_test - n_val

        if n_train <= 0:
            raise ValueError(
                f"temporal_split: not enough rows for train set "
                f"(n={n}, test_frac={t_frac}, val_frac={v_frac})."
            )

        train = df_sorted.iloc[:n_train]
        val = df_sorted.iloc[n_train: n_train + n_val]
        test = df_sorted.iloc[n_train + n_val:]

        # Log time boundaries
        ts_col = pd.to_datetime(df_sorted[timestamp_col], errors="coerce")
        logger.info(
            "temporal_split: train ends %s | val ends %s | test ends %s",
            ts_col.iloc[n_train - 1],
            ts_col.iloc[n_train + n_val - 1],
            ts_col.iloc[-1],
        )
        self._log_split(train, val, test, "temporal")
        return train.copy(), val.copy(), test.copy()

    def stratified_split(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> SplitTriple:
        """Stratified split preserving class proportions in all three sets.

        Requires scikit-learn.  Falls back to random split if sklearn is
        not available.

        Parameters
        ----------
        df:
            DataFrame to split.
        target_col:
            Binary or multi-class target column for stratification.

        Returns
        -------
        tuple[df_train, df_val, df_test]
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning("stratified_split: sklearn not available — falling back to random.")
            return self._random_split(df)

        if target_col not in df.columns:
            raise KeyError(f"stratified_split: column '{target_col}' not found.")

        y = df[target_col].values

        # First split off test set
        sss_test = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        train_val_idx, test_idx = next(sss_test.split(df, y))

        df_train_val = df.iloc[train_val_idx]
        y_train_val = y[train_val_idx]

        # Split remaining into train + val
        val_frac_of_remainder = self.val_size / (1.0 - self.test_size)
        val_frac_of_remainder = min(val_frac_of_remainder, 0.99)

        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_frac_of_remainder,
            random_state=self.random_state,
        )
        train_rel_idx, val_rel_idx = next(sss_val.split(df_train_val, y_train_val))

        train = df_train_val.iloc[train_rel_idx].copy()
        val = df_train_val.iloc[val_rel_idx].copy()
        test = df.iloc[test_idx].copy()

        self._log_split(train, val, test, "stratified")
        return train, val, test

    def group_split(
        self,
        df: pd.DataFrame,
        group_col: str,
        target_col: Optional[str] = None,
    ) -> SplitTriple:
        """Group-aware split ensuring no group (e.g. patient) spans sets.

        All rows belonging to a given group ID are placed entirely in one
        of train / val / test.  This prevents patient leakage in
        multi-encounter datasets.

        Parameters
        ----------
        df:
            DataFrame to split.
        group_col:
            Column holding the group identifier (e.g. ``'subject_id'``).
        target_col:
            Unused; kept for API symmetry.

        Returns
        -------
        tuple[df_train, df_val, df_test]
        """
        if group_col not in df.columns:
            raise KeyError(f"group_split: column '{group_col}' not found.")

        if _SKLEARN_AVAILABLE:
            groups = df[group_col].values

            # Test split
            gss_test = GroupShuffleSplit(
                n_splits=1,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

            df_train_val = df.iloc[train_val_idx]
            groups_tv = groups[train_val_idx]

            val_frac_of_remainder = self.val_size / (1.0 - self.test_size)
            gss_val = GroupShuffleSplit(
                n_splits=1,
                test_size=min(val_frac_of_remainder, 0.99),
                random_state=self.random_state,
            )
            train_rel_idx, val_rel_idx = next(
                gss_val.split(df_train_val, groups=groups_tv)
            )

            train = df_train_val.iloc[train_rel_idx].copy()
            val = df_train_val.iloc[val_rel_idx].copy()
            test = df.iloc[test_idx].copy()
        else:
            # Fallback: shuffle unique groups, then assign proportionally
            unique_groups = df[group_col].unique()
            self._rng.shuffle(unique_groups)
            n_g = len(unique_groups)
            n_test_g = max(1, int(n_g * self.test_size))
            n_val_g = max(1, int(n_g * self.val_size))

            test_groups = set(unique_groups[:n_test_g])
            val_groups = set(unique_groups[n_test_g: n_test_g + n_val_g])
            train_groups = set(unique_groups[n_test_g + n_val_g:])

            train = df[df[group_col].isin(train_groups)].copy()
            val = df[df[group_col].isin(val_groups)].copy()
            test = df[df[group_col].isin(test_groups)].copy()

        # Verify no overlap
        if _SKLEARN_AVAILABLE or True:
            train_groups_set = set(train[group_col].unique())
            val_groups_set = set(val[group_col].unique())
            test_groups_set = set(test[group_col].unique())
            overlap_tv = train_groups_set & val_groups_set
            overlap_tt = train_groups_set & test_groups_set
            if overlap_tv or overlap_tt:
                logger.error(
                    "group_split: group overlap detected! train∩val=%d, train∩test=%d",
                    len(overlap_tv), len(overlap_tt),
                )
            else:
                logger.info("group_split: no group overlap — leakage check passed.")

        self._log_split(train, val, test, "group")
        return train, val, test

    def get_fold_indices(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        strategy: str = "kfold",
        timestamp_col: Optional[str] = None,
        target_col: Optional[str] = None,
        group_col: Optional[str] = None,
    ) -> FoldList:
        """Generate cross-validation fold indices.

        Parameters
        ----------
        df:
            DataFrame to fold.
        n_splits:
            Number of CV folds.
        strategy:
            ``'kfold'``, ``'stratified_kfold'``, ``'group_kfold'``,
            or ``'timeseries'``.
        timestamp_col:
            Required for ``'timeseries'`` strategy.
        target_col:
            Required for ``'stratified_kfold'`` strategy.
        group_col:
            Required for ``'group_kfold'`` strategy.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of ``(train_indices, val_indices)`` integer arrays.

        Raises
        ------
        ImportError
            If sklearn is not installed.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "get_fold_indices requires scikit-learn.  "
                "Install it with: pip install scikit-learn"
            )

        X = np.arange(len(df))

        if strategy == "kfold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            folds = list(cv.split(X))

        elif strategy == "stratified_kfold":
            if target_col is None or target_col not in df.columns:
                raise ValueError("stratified_kfold requires a valid target_col.")
            y = df[target_col].values
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            folds = list(cv.split(X, y))

        elif strategy == "group_kfold":
            if group_col is None or group_col not in df.columns:
                raise ValueError("group_kfold requires a valid group_col.")
            groups = df[group_col].values
            cv = GroupKFold(n_splits=n_splits)
            folds = list(cv.split(X, groups=groups))

        elif strategy == "timeseries":
            if timestamp_col is None or timestamp_col not in df.columns:
                raise ValueError("timeseries strategy requires a valid timestamp_col.")
            df_sorted = df.sort_values(timestamp_col)
            X_sorted = np.arange(len(df_sorted))
            cv = TimeSeriesSplit(n_splits=n_splits)
            folds = list(cv.split(X_sorted))

        else:
            raise ValueError(
                f"Unknown fold strategy '{strategy}'.  "
                "Choose from: kfold, stratified_kfold, group_kfold, timeseries."
            )

        logger.info(
            "get_fold_indices [%s]: %d folds generated from %d rows.",
            strategy, len(folds), len(df),
        )
        return folds


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    rng = np.random.default_rng(99)
    n = 1000

    df = pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(200), 5),  # 200 patients, 5 encounters each
            "admit_date": pd.date_range("2015-01-01", periods=n, freq="4h"),
            "age": rng.uniform(18, 90, n),
            "readmit_30d": rng.integers(0, 2, n),
        }
    )

    splitter = DataSplitter(test_size=0.2, val_size=0.1, random_state=42)

    print("=== Random split ===")
    tr, va, te = splitter._random_split(df)
    print(f"  train={len(tr)}, val={len(va)}, test={len(te)}")

    print("\n=== Temporal split ===")
    tr, va, te = splitter.temporal_split(df, "admit_date")
    print(f"  train={len(tr)}, val={len(va)}, test={len(te)}")

    print("\n=== Stratified split ===")
    tr, va, te = splitter.stratified_split(df, "readmit_30d")
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        rate = part["readmit_30d"].mean()
        print(f"  {name}: n={len(part)}, readmit_rate={rate:.3f}")

    print("\n=== Group split (no patient leakage) ===")
    tr, va, te = splitter.group_split(df, "subject_id")
    print(f"  train={len(tr)}, val={len(va)}, test={len(te)}")
    all_ids = set(df["subject_id"])
    print(f"  unique groups covered: {len(set(tr['subject_id']) | set(va['subject_id']) | set(te['subject_id']))} / {len(all_ids)}")

    print("\n=== K-Fold indices ===")
    folds = splitter.get_fold_indices(df, n_splits=5, strategy="kfold")
    for i, (ti, vi) in enumerate(folds):
        print(f"  fold {i+1}: train={len(ti)}, val={len(vi)}")
