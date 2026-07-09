"""
Feature store for HealthRiskAI.
Provides versioned parquet-backed persistence for ML feature sets,
with metadata tracking, schema validation, and version management.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class FeatureStore:
    """
    Versioned local feature store backed by Parquet files.

    Directory layout:
        {store_dir}/
            {feature_set}/
                {version}.parquet
                {version}.json       ← metadata
                latest.parquet       ← symlink/copy of most-recent version
                latest.json          ← metadata of most-recent version
    """

    def __init__(self, store_dir: str = "data/features/store") -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FeatureStore initialised at {self.store_dir.resolve()}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _feature_dir(self, feature_set: str) -> Path:
        return self.store_dir / feature_set

    def _ensure_feature_dir(self, feature_set: str) -> Path:
        path = self._feature_dir(feature_set)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _parquet_path(self, feature_set: str, version: str) -> Path:
        return self._feature_dir(feature_set) / f"{version}.parquet"

    def _meta_path(self, feature_set: str, version: str) -> Path:
        return self._feature_dir(feature_set) / f"{version}.json"

    def _latest_parquet(self, feature_set: str) -> Path:
        return self._feature_dir(feature_set) / "latest.parquet"

    def _latest_meta(self, feature_set: str) -> Path:
        return self._feature_dir(feature_set) / "latest.json"

    def _write_metadata(
        self,
        feature_set: str,
        version: str,
        df: pd.DataFrame,
        path: str,
    ) -> None:
        meta: Dict[str, Any] = {
            "version":    version,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "rows":       int(len(df)),
            "columns":    int(df.shape[1]),
            "dtypes":     {col: str(dtype) for col, dtype in df.dtypes.items()},
            "source":     path,
        }
        meta_path = self._meta_path(feature_set, version)
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.debug(f"  Metadata written to {meta_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_features(
        self,
        df: pd.DataFrame,
        feature_set: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Persist a feature DataFrame as a versioned Parquet file.

        Parameters
        ----------
        df          : Feature DataFrame to save.
        feature_set : Logical name for this feature group (e.g. 'clinical').
        version     : Optional version string. Defaults to 'YYYYMMDD_HHMMSS'.

        Returns
        -------
        Absolute path string of the saved Parquet file.
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        feature_dir = self._ensure_feature_dir(feature_set)
        parquet_path = self._parquet_path(feature_set, version)
        latest_path  = self._latest_parquet(feature_set)

        logger.info(
            f"save_features: feature_set='{feature_set}', version='{version}', "
            f"shape={df.shape}"
        )

        df.to_parquet(parquet_path, index=True, engine="pyarrow", compression="snappy")
        logger.debug(f"  Saved parquet: {parquet_path}")

        # Write metadata
        self._write_metadata(feature_set, version, df, str(parquet_path))

        # Update 'latest'
        shutil.copy2(parquet_path, latest_path)
        shutil.copy2(
            self._meta_path(feature_set, version),
            self._latest_meta(feature_set),
        )
        logger.debug(f"  Updated latest: {latest_path}")

        logger.success(
            f"save_features: '{feature_set}' v{version} saved "
            f"({len(df):,} rows × {df.shape[1]} cols)"
        )
        return str(parquet_path)

    def load_features(
        self,
        feature_set: str,
        version: str = "latest",
    ) -> pd.DataFrame:
        """
        Load a feature DataFrame from the store.

        Parameters
        ----------
        feature_set : Name of the feature group.
        version     : Version string, or 'latest' (default).

        Returns
        -------
        Loaded DataFrame.

        Raises
        ------
        FileNotFoundError if the requested version does not exist.
        """
        if version == "latest":
            path = self._latest_parquet(feature_set)
        else:
            path = self._parquet_path(feature_set, version)

        if not path.exists():
            raise FileNotFoundError(
                f"Feature set '{feature_set}' version '{version}' not found at {path}"
            )

        logger.info(f"load_features: loading '{feature_set}' v={version} from {path}")
        df = pd.read_parquet(path, engine="pyarrow")
        logger.success(
            f"load_features: loaded {len(df):,} rows × {df.shape[1]} cols "
            f"from '{feature_set}' v={version}"
        )
        return df

    def list_feature_sets(self) -> List[str]:
        """
        List all available feature set names (subdirectory names in store_dir).

        Returns
        -------
        Sorted list of feature set name strings.
        """
        feature_sets = sorted(
            [d.name for d in self.store_dir.iterdir() if d.is_dir()]
        )
        logger.info(f"list_feature_sets: {len(feature_sets)} found: {feature_sets}")
        return feature_sets

    def list_versions(self, feature_set: str) -> List[str]:
        """
        List all saved versions for a feature set (excludes 'latest').

        Returns
        -------
        Sorted list of version strings (from .json filenames, excluding 'latest').
        """
        feature_dir = self._feature_dir(feature_set)
        if not feature_dir.exists():
            logger.warning(f"list_versions: feature set '{feature_set}' not found")
            return []

        versions = sorted(
            p.stem
            for p in feature_dir.glob("*.json")
            if p.stem != "latest"
        )
        logger.info(f"list_versions: '{feature_set}' has {len(versions)} version(s)")
        return versions

    def get_feature_schema(self, feature_set: str) -> dict:
        """
        Load the metadata (schema) for the latest version of a feature set.

        Returns
        -------
        Metadata dict with keys: version, created_at, rows, columns, dtypes, source.

        Raises
        ------
        FileNotFoundError if no metadata exists.
        """
        meta_path = self._latest_meta(feature_set)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No schema found for feature set '{feature_set}' at {meta_path}"
            )

        schema = json.loads(meta_path.read_text())
        logger.info(
            f"get_feature_schema: '{feature_set}' v={schema.get('version')} "
            f"({schema.get('rows')} rows × {schema.get('columns')} cols)"
        )
        return schema

    def validate_features(self, df: pd.DataFrame, feature_set: str) -> bool:
        """
        Validate a DataFrame against the stored schema for a feature set.

        Checks:
        - All expected columns are present
        - No column is entirely null

        Parameters
        ----------
        df          : DataFrame to validate.
        feature_set : Feature set whose schema to compare against.

        Returns
        -------
        True if all checks pass, False otherwise (with warnings logged).
        """
        logger.info(
            f"validate_features: validating {df.shape} against schema '{feature_set}'"
        )

        try:
            schema = self.get_feature_schema(feature_set)
        except FileNotFoundError:
            logger.warning(
                f"validate_features: no schema found for '{feature_set}' – cannot validate"
            )
            return False

        expected_cols: List[str] = list(schema.get("dtypes", {}).keys())
        df_cols = set(df.columns.tolist())
        all_ok  = True

        # Check for missing columns
        missing = [c for c in expected_cols if c not in df_cols]
        if missing:
            logger.warning(
                f"validate_features: {len(missing)} missing column(s): {missing}"
            )
            all_ok = False

        # Check for fully-null columns
        present_cols = [c for c in expected_cols if c in df_cols]
        for col in present_cols:
            if df[col].isna().all():
                logger.warning(
                    f"validate_features: column '{col}' is entirely null"
                )
                all_ok = False

        if all_ok:
            logger.success(
                f"validate_features: '{feature_set}' PASSED "
                f"({len(expected_cols)} columns checked)"
            )
        else:
            logger.warning(
                f"validate_features: '{feature_set}' FAILED – see warnings above"
            )

        return all_ok

    def delete_version(self, feature_set: str, version: str) -> None:
        """
        Delete a specific version of a feature set (both .parquet and .json).
        If the deleted version was 'latest', promotes the next-most-recent version.

        Parameters
        ----------
        feature_set : Name of the feature group.
        version     : Version string to delete (cannot be 'latest').

        Raises
        ------
        ValueError if version == 'latest' (use delete_version with the actual version name).
        FileNotFoundError if the version does not exist.
        """
        if version == "latest":
            raise ValueError(
                "Cannot delete 'latest' directly. Specify the actual version string."
            )

        parquet_path = self._parquet_path(feature_set, version)
        meta_path    = self._meta_path(feature_set, version)

        if not parquet_path.exists() and not meta_path.exists():
            raise FileNotFoundError(
                f"Version '{version}' of feature set '{feature_set}' not found"
            )

        # Check whether this version is currently 'latest'
        was_latest = False
        latest_meta_path = self._latest_meta(feature_set)
        if latest_meta_path.exists():
            try:
                latest_meta = json.loads(latest_meta_path.read_text())
                if latest_meta.get("version") == version:
                    was_latest = True
            except (json.JSONDecodeError, KeyError):
                pass

        # Delete files
        for path in (parquet_path, meta_path):
            if path.exists():
                path.unlink()
                logger.debug(f"  Deleted {path}")

        logger.info(f"delete_version: '{feature_set}' v={version} deleted")

        # If this was latest, promote the next-most-recent version
        if was_latest:
            remaining = self.list_versions(feature_set)
            latest_parquet = self._latest_parquet(feature_set)
            latest_meta_path = self._latest_meta(feature_set)

            if remaining:
                new_latest = remaining[-1]  # list_versions returns sorted → last is newest
                logger.info(
                    f"delete_version: '{version}' was latest – promoting '{new_latest}' to latest"
                )
                shutil.copy2(self._parquet_path(feature_set, new_latest), latest_parquet)
                shutil.copy2(self._meta_path(feature_set, new_latest), latest_meta_path)
            else:
                # No versions remain; remove latest files if they exist
                for path in (latest_parquet, latest_meta_path):
                    if path.exists():
                        path.unlink()
                        logger.debug(f"  Removed stale {path}")
                logger.warning(
                    f"delete_version: no versions remain for '{feature_set}'"
                )
