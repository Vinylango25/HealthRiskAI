"""
Master Data Acquisition Pipeline
==================================
Orchestrates all health data source clients — WHO GHO, ClinicalTrials.gov,
FDA FAERS, CDC WONDER, CMS Hospital Compare, and SEC EDGAR — into a single
runnable pipeline with validation and data-quality reporting.

Usage
-----
    # Run all sources:
    python -m data.acquisition.pipeline

    # Run specific sources:
    python -m data.acquisition.pipeline --sources cdc,cms

    # Validate existing outputs only:
    python -m data.acquisition.pipeline --validate

    # Generate data quality report only:
    python -m data.acquisition.pipeline --report

Environment variables
---------------------
All client-specific env vars are forwarded from the loaded .env file.
See individual client modules for their required variables.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Lazy client imports — deferred to avoid circular imports and so that
# missing optional dependencies don't break unrelated sources.
# ---------------------------------------------------------------------------

def _import_who():
    from data.acquisition.who_gho import WHOGHOClient
    return WHOGHOClient


def _import_clinicaltrials():
    from data.acquisition.clinicaltrials import ClinicalTrialsClient
    return ClinicalTrialsClient


def _import_fda():
    from data.acquisition.fda_faers import FDAFAERSClient
    return FDAFAERSClient


def _import_cdc():
    from data.acquisition.cdc_wonder import CDCWonderClient
    return CDCWonderClient


def _import_cms():
    from data.acquisition.cms_scraper import CMSDataClient
    return CMSDataClient


def _import_sec():
    from data.acquisition.sec_edgar import SECEdgarClient
    return SECEdgarClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ALL_SOURCES: list[str] = [
    "who_gho",
    "clinicaltrials",
    "fda_faers",
    "cdc",
    "cms",
    "sec_edgar",
]

logger = logger.bind(module="pipeline")


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------


class DataPipeline:
    """
    Master orchestrator for all HealthRiskAI data acquisition sources.

    Parameters
    ----------
    config_path:
        Path to YAML configuration file (``configs/data_sources.yaml``).
    env_file:
        Path to ``.env`` file with API keys and configuration overrides.
    """

    def __init__(
        self,
        config_path: str = "configs/data_sources.yaml",
        env_file: str = ".env",
    ) -> None:
        # --- Load environment variables ---
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info("Loaded env from {}", env_path)
        else:
            load_dotenv(override=True)
            logger.warning(".env file not found at {}; using system env", env_path)

        # --- Load YAML config ---
        cfg_path = Path(config_path)
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as fh:
                self.config: dict = yaml.safe_load(fh) or {}
            logger.info("Loaded config from {}", cfg_path)
        else:
            logger.warning("Config file {} not found; using empty config.", cfg_path)
            self.config = {}

        # --- Ensure output directories exist ---
        Path("reports").mkdir(parents=True, exist_ok=True)

        # --- Lazily initialised client instances (populated on first use) ---
        self._clients: dict = {}

        logger.info("DataPipeline initialised | sources available: {}", _ALL_SOURCES)

    # ------------------------------------------------------------------
    # Client accessors (lazy init so missing env vars only fail on use)
    # ------------------------------------------------------------------

    def _get_client(self, source_name: str):
        """Return (and cache) the client instance for the given source."""
        if source_name not in self._clients:
            self._clients[source_name] = self._build_client(source_name)
        return self._clients[source_name]

    def _build_client(self, source_name: str):
        if source_name == "who_gho":
            return _import_who()()
        if source_name == "clinicaltrials":
            return _import_clinicaltrials()()
        if source_name == "fda_faers":
            return _import_fda()()
        if source_name == "cdc":
            return _import_cdc()()
        if source_name == "cms":
            return _import_cms()()
        if source_name == "sec_edgar":
            ua = os.getenv(
                "SEC_EDGAR_USER_AGENT",
                self.config.get("sec_edgar", {}).get(
                    "user_agent", "HealthRiskAI/1.0 contact@healthrisk.ai"
                ),
            )
            return _import_sec()(user_agent=ua)
        raise ValueError(f"Unknown source: '{source_name}'")

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        sources: Optional[list[str]] = None,
    ) -> dict[str, dict]:
        """
        Run the full acquisition pipeline for the specified (or all) sources.

        Note: MIMIC-IV is excluded from the default run because it requires
        manual PhysioNet credentialed access.  Add ``'mimic'`` explicitly
        to ``sources`` if credentials are available.

        Parameters
        ----------
        sources:
            List of source names to run. ``None`` runs all default sources.

        Returns
        -------
        dict
            ``{source_name: {'status': 'success'|'failed', 'rows': int,
            'duration_seconds': float, 'error': str|None}}``
        """
        run_sources = sources if sources is not None else _ALL_SOURCES
        logger.info("run_full_pipeline | sources={}", run_sources)

        results: dict[str, dict] = {}
        pipeline_start = time.monotonic()

        for source in run_sources:
            logger.info("── Running source: {} ──", source)
            source_result = self.run_source(source)
            results[source] = source_result
            status = source_result["status"]
            rows = source_result.get("rows", 0)
            dur = source_result.get("duration_seconds", 0)
            logger.info(
                "Source {} | status={} rows={} duration={:.1f}s",
                source, status, rows, dur,
            )

        total_duration = time.monotonic() - pipeline_start
        self.save_run_metadata(results)

        logger.success(
            "run_full_pipeline complete | total_duration={:.1f}s results={}",
            total_duration,
            {k: v["status"] for k, v in results.items()},
        )
        return results

    def run_source(self, source_name: str) -> dict:
        """
        Fetch data for a single named source, persist it to parquet, and
        return a status dictionary.

        Parameters
        ----------
        source_name:
            One of ``'who_gho'``, ``'clinicaltrials'``, ``'fda_faers'``,
            ``'cdc'``, ``'cms'``, ``'sec_edgar'``.

        Returns
        -------
        dict
            ``{'status': str, 'rows': int, 'duration_seconds': float,
            'output_path': str, 'error': str|None}``
        """
        start = time.monotonic()
        output_dir = Path("data/raw") / source_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "latest.parquet"

        try:
            df = self._fetch_source(source_name)

            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                logger.warning("Source {} returned empty data.", source_name)
                return {
                    "status": "empty",
                    "rows": 0,
                    "duration_seconds": time.monotonic() - start,
                    "output_path": str(output_path),
                    "error": None,
                }

            df.to_parquet(output_path, index=False)
            rows = len(df)
            logger.success("Source {} saved {} rows to {}", source_name, rows, output_path)

            return {
                "status": "success",
                "rows": rows,
                "duration_seconds": time.monotonic() - start,
                "output_path": str(output_path),
                "error": None,
            }

        except Exception as exc:  # noqa: BLE001
            logger.error("Source {} FAILED: {}", source_name, exc)
            return {
                "status": "failed",
                "rows": 0,
                "duration_seconds": time.monotonic() - start,
                "output_path": str(output_path),
                "error": str(exc),
            }

    def _fetch_source(self, source_name: str) -> pd.DataFrame:
        """Dispatch to the correct client method(s) and return a DataFrame."""
        client = self._get_client(source_name)

        if source_name == "who_gho":
            frames: list[pd.DataFrame] = []
            for method in ("get_disease_burden", "get_risk_factors", "get_health_expenditure"):
                fn = getattr(client, method, None)
                if fn:
                    try:
                        df = fn()
                        if not df.empty:
                            df["_source_method"] = method
                            frames.append(df)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("WHO GHO {} failed: {}", method, exc)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if source_name == "clinicaltrials":
            company_names: list[str] = (
                self.config.get("clinicaltrials", {}).get("company_names", [])
            )
            fn = getattr(client, "build_pipeline_dataset", None)
            if fn:
                return fn(company_names=company_names) if company_names else fn()
            # Fallback: try search_trials
            fn2 = getattr(client, "search_trials", None)
            if fn2:
                return fn2()
            raise AttributeError("ClinicalTrialsClient has no build_pipeline_dataset method.")

        if source_name == "fda_faers":
            drug_list: list[str] = (
                self.config.get("fda_faers", {}).get("drug_list", [])
            )
            fn = getattr(client, "build_pharmacovigilance_report", None)
            if fn:
                return fn(drug_list=drug_list) if drug_list else fn()
            fn2 = getattr(client, "get_adverse_events", None)
            if fn2:
                return fn2()
            raise AttributeError("FDAFAERSClient has no build_pharmacovigilance_report method.")

        if source_name == "cdc":
            # download_all saves files; also return a combined DataFrame
            results = client.download_all()
            frames = []
            for name, path in results.items():
                if path != "FAILED":
                    try:
                        df = pd.read_parquet(path)
                        df["_dataset"] = name
                        frames.append(df)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not re-read {}: {}", path, exc)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if source_name == "cms":
            return client.merge_all_hospital_data()

        if source_name == "sec_edgar":
            return client.get_pharma_universe()

        raise ValueError(f"No fetch handler for source '{source_name}'.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_outputs(self) -> dict[str, dict]:
        """
        Check that each source's ``latest.parquet`` exists, has rows, and
        has no entirely-null columns.

        Returns
        -------
        dict
            ``{source: {'exists': bool, 'rows': int, 'columns': int,
            'all_null_columns': list[str], 'valid': bool}}``
        """
        report: dict[str, dict] = {}

        for source in _ALL_SOURCES:
            parquet_path = Path("data/raw") / source / "latest.parquet"
            entry: dict = {
                "path": str(parquet_path),
                "exists": parquet_path.exists(),
                "rows": 0,
                "columns": 0,
                "all_null_columns": [],
                "valid": False,
            }

            if not parquet_path.exists():
                logger.warning("Validation: {} missing at {}", source, parquet_path)
                report[source] = entry
                continue

            try:
                df = pd.read_parquet(parquet_path)
                entry["rows"] = len(df)
                entry["columns"] = len(df.columns)

                all_null = [c for c in df.columns if df[c].isna().all()]
                entry["all_null_columns"] = all_null

                entry["valid"] = (
                    len(df) > 0
                    and len(all_null) == 0
                )
                logger.info(
                    "Validation: {} | rows={} cols={} all_null={} valid={}",
                    source, entry["rows"], entry["columns"],
                    all_null, entry["valid"],
                )
            except Exception as exc:  # noqa: BLE001
                entry["error"] = str(exc)
                logger.error("Validation read error for {}: {}", source, exc)

            report[source] = entry

        return report

    # ------------------------------------------------------------------
    # Data quality report
    # ------------------------------------------------------------------

    def generate_data_quality_report(self) -> dict:
        """
        Generate a comprehensive data quality report for all source outputs.

        For each source's ``latest.parquet`` computes:
        * ``row_count``
        * ``column_count``
        * ``completeness_pct`` — mean non-null ratio × 100
        * ``date_range`` — min/max of any datetime columns
        * ``file_size_mb``

        The report is saved to ``reports/data_quality_{timestamp}.json``.

        Returns
        -------
        dict
            Full quality report dict.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report: dict = {
            "generated_at": timestamp,
            "sources": {},
        }

        for source in _ALL_SOURCES:
            parquet_path = Path("data/raw") / source / "latest.parquet"
            source_report: dict = {
                "path": str(parquet_path),
                "exists": parquet_path.exists(),
            }

            if not parquet_path.exists():
                source_report["error"] = "File not found"
                report["sources"][source] = source_report
                continue

            try:
                df = pd.read_parquet(parquet_path)
                file_size_mb = parquet_path.stat().st_size / (1024 ** 2)

                # Completeness
                if len(df) > 0:
                    null_ratio = df.isnull().mean()  # per-column
                    completeness_pct = float((1 - null_ratio).mean() * 100)
                else:
                    completeness_pct = 0.0

                # Date range across all datetime-like columns
                date_range: dict = {}
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        non_null = df[col].dropna()
                        if not non_null.empty:
                            date_range[col] = {
                                "min": str(non_null.min()),
                                "max": str(non_null.max()),
                            }
                    elif col in ("year", "yearstart", "yearend"):
                        non_null = pd.to_numeric(df[col], errors="coerce").dropna()
                        if not non_null.empty:
                            date_range[col] = {
                                "min": int(non_null.min()),
                                "max": int(non_null.max()),
                            }

                # Column-level null pct
                null_by_col = {
                    col: float(round(df[col].isnull().mean() * 100, 2))
                    for col in df.columns
                }

                source_report.update(
                    {
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "completeness_pct": round(completeness_pct, 2),
                        "date_range": date_range,
                        "file_size_mb": round(file_size_mb, 4),
                        "null_pct_by_column": null_by_col,
                        "columns": list(df.columns),
                    }
                )

            except Exception as exc:  # noqa: BLE001
                logger.error("Quality report error for {}: {}", source, exc)
                source_report["error"] = str(exc)

            report["sources"][source] = source_report

        # Persist report
        report_path = Path("reports") / f"data_quality_{timestamp}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

        logger.success("Data quality report saved to {}", report_path)
        return report

    # ------------------------------------------------------------------
    # Run metadata
    # ------------------------------------------------------------------

    def save_run_metadata(self, run_results: dict) -> None:
        """
        Append pipeline run metadata to ``reports/pipeline_runs.json``.

        Parameters
        ----------
        run_results:
            Dict returned by ``run_full_pipeline``.
        """
        runs_path = Path("reports/pipeline_runs.json")
        runs_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing records
        existing: list[dict] = []
        if runs_path.exists():
            try:
                with runs_path.open("r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, OSError):
                existing = []

        total_duration = sum(
            v.get("duration_seconds", 0)
            for v in run_results.values()
            if isinstance(v, dict)
        )

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sources_run": list(run_results.keys()),
            "total_duration_seconds": round(total_duration, 2),
            "statuses": {
                src: {
                    "status": info.get("status"),
                    "rows": info.get("rows", 0),
                    "duration_seconds": round(info.get("duration_seconds", 0), 2),
                    "error": info.get("error"),
                }
                for src, info in run_results.items()
                if isinstance(info, dict)
            },
        }

        existing.append(record)

        with runs_path.open("w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, default=str)

        logger.info("Run metadata saved to {} ({} total runs)", runs_path, len(existing))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description=(
            "HealthRiskAI data acquisition pipeline. "
            "Fetches, validates, and reports on all health data sources."
        ),
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "Comma-separated list of sources to run "
            f"(default: all). Available: {', '.join(_ALL_SOURCES)}"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Run output validation only (no data download).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate data quality report only (no data download).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_sources.yaml",
        help="Path to data sources YAML config (default: configs/data_sources.yaml).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file (default: .env).",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    pipeline = DataPipeline(config_path=args.config, env_file=args.env)

    if args.validate:
        validation = pipeline.validate_outputs()
        print(json.dumps(validation, indent=2, default=str))

    elif args.report:
        quality_report = pipeline.generate_data_quality_report()
        print(json.dumps(quality_report, indent=2, default=str))

    else:
        sources_list: Optional[list[str]] = None
        if args.sources:
            sources_list = [s.strip() for s in args.sources.split(",") if s.strip()]

        results = pipeline.run_full_pipeline(sources=sources_list)
        print(json.dumps(results, indent=2, default=str))
