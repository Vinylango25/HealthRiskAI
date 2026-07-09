"""
data/acquisition/cli.py
========================
CLI entry point for the HealthRiskAI data ingestion pipeline.

Registered as the ``healthrisk-ingest`` console script in pyproject.toml:

    healthrisk-ingest <subcommand> [options]

Subcommands
-----------
all             Run all acquisition sources via DataPipeline.
mimic           Fetch MIMIC-IV clinical data (requires PhysioNet credentials).
who             Fetch WHO Global Health Observatory indicators.
clinicaltrials  Fetch ClinicalTrials.gov study records.
faers           Fetch FDA FAERS adverse-event reports.
cdc             Fetch CDC WONDER mortality / surveillance data.
cms             Fetch CMS Hospital Compare & cost-report data.
sec             Fetch SEC EDGAR pharma filings.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="healthrisk-ingest",
        description=(
            "HealthRiskAI data ingestion CLI. "
            "Fetches data from clinical, regulatory, and financial sources."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Global flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--output-dir", "-o",
        default="data/raw/",
        metavar="DIR",
        help="Directory where fetched data files are written (default: data/raw/).",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/data_sources.yaml",
        metavar="FILE",
        help="Path to data_sources.yaml configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without fetching any data.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    # ── Subcommands ──────────────────────────────────────────────────────────
    sub = parser.add_subparsers(dest="source", metavar="SOURCE")
    sub.required = True

    sub.add_parser(
        "all",
        help="Run all available sources via the master DataPipeline.",
        description="Orchestrates every acquisition source in a single run.",
    )
    sub.add_parser(
        "mimic",
        help="Fetch MIMIC-IV clinical data from PhysioNet.",
        description="Downloads MIMIC-IV tables using the MIMICLoader client.",
    )
    sub.add_parser(
        "who",
        help="Fetch WHO Global Health Observatory indicators.",
        description="Queries the WHO GHO OData API for selected health indicators.",
    )
    sub.add_parser(
        "clinicaltrials",
        help="Fetch ClinicalTrials.gov study records.",
        description="Pulls study records from the ClinicalTrials.gov API v2.",
    )
    sub.add_parser(
        "faers",
        help="Fetch FDA FAERS adverse-event reports.",
        description="Downloads adverse-event reports from the openFDA FAERS endpoint.",
    )
    sub.add_parser(
        "cdc",
        help="Fetch CDC WONDER mortality and surveillance data.",
        description="Queries CDC WONDER / Socrata for mortality statistics.",
    )
    sub.add_parser(
        "cms",
        help="Fetch CMS Hospital Compare and cost-report data.",
        description="Scrapes CMS Hospital Compare datasets and cost reports.",
    )
    sub.add_parser(
        "sec",
        help="Fetch SEC EDGAR pharma/biotech filings.",
        description="Downloads 10-K, 10-Q, and 8-K filings from SEC EDGAR.",
    )

    return parser


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _run_all(args: argparse.Namespace) -> None:
    """Run every acquisition source through the master DataPipeline."""
    from data.acquisition.pipeline import DataPipeline  # noqa: PLC0415

    logger.info("Initialising DataPipeline | config=%s output_dir=%s", args.config, args.output_dir)
    if args.dry_run:
        logger.info("[dry-run] Would execute DataPipeline.run_all() — skipping.")
        return

    pipeline = DataPipeline(config_path=args.config)
    pipeline.run_all(output_dir=args.output_dir)
    logger.info("DataPipeline completed successfully.")


def _run_mimic(args: argparse.Namespace) -> None:
    from data.acquisition.mimic_loader import MIMICLoader  # noqa: PLC0415

    output = Path(args.output_dir) / "mimic"
    logger.info("Fetching MIMIC-IV data → %s", output)
    if args.dry_run:
        logger.info("[dry-run] Would call MIMICLoader().fetch(output_dir=%s)", output)
        return

    loader = MIMICLoader()
    loader.fetch(output_dir=str(output))
    logger.info("MIMIC-IV fetch complete.")


def _run_who(args: argparse.Namespace) -> None:
    from data.acquisition.who_gho import WHOGHOClient  # noqa: PLC0415

    output = Path(args.output_dir) / "who_gho"
    logger.info("Fetching WHO GHO indicators → %s", output)
    if args.dry_run:
        logger.info("[dry-run] Would call WHOGHOClient().fetch_all(output_dir=%s)", output)
        return

    client = WHOGHOClient()
    client.fetch_all(output_dir=str(output))
    logger.info("WHO GHO fetch complete.")


def _run_clinicaltrials(args: argparse.Namespace) -> None:
    from data.acquisition.clinicaltrials import ClinicalTrialsClient  # noqa: PLC0415

    output = Path(args.output_dir) / "clinicaltrials"
    logger.info("Fetching ClinicalTrials.gov data → %s", output)
    if args.dry_run:
        logger.info("[dry-run] Would call ClinicalTrialsClient().fetch(output_dir=%s)", output)
        return

    client = ClinicalTrialsClient()
    client.fetch(output_dir=str(output))
    logger.info("ClinicalTrials.gov fetch complete.")


def _run_faers(args: argparse.Namespace) -> None:
    from data.acquisition.fda_faers import FDAFAERSClient  # noqa: PLC0415

    output = Path(args.output_dir) / "faers"
    logger.info("Fetching FDA FAERS adverse-event reports → %s", output)
    if args.dry_run:
        logger.info("[dry-run] Would call FDAFAERSClient().fetch(output_dir=%s)", output)
        return

    client = FDAFAERSClient()
    client.fetch(output_dir=str(output))
    logger.info("FDA FAERS fetch complete.")


def _run_cdc(args: argparse.Namespace) -> None:
    from data.acquisition.cdc_wonder import CDCWonderClient  # noqa: PLC0415

    output = Path(args.output_dir) / "cdc"
    logger.info("Fetching CDC WONDER data → %s", output)
    if args.dry_run:
        logger.info("[dry-run] Would call CDCWonderClient().fetch(output_dir=%s)", output)
        return

    client = CDCWonderClient()
    client.fetch(output_dir=str(output))
    logger.info("CDC WONDER fetch complete.")


def _run_cms(args: argparse.Namespace) -> None:
    from data.acquisition.cms_scraper import CMSDataClient  # noqa: PLC0415

    output = Path(args.output_dir) / "cms"
    logger.info("Fetching CMS hospital data → %s", output)
    if args.dry_run:
        logger.info("[dry-run] Would call CMSDataClient().fetch(output_dir=%s)", output)
        return

    client = CMSDataClient()
    client.fetch(output_dir=str(output))
    logger.info("CMS fetch complete.")


def _run_sec(args: argparse.Namespace) -> None:
    from data.acquisition.sec_edgar import SECEdgarClient  # noqa: PLC0415

    output = Path(args.output_dir) / "sec_edgar"
    logger.info("Fetching SEC EDGAR filings → %s", output)
    if args.dry_run:
        logger.info("[dry-run] Would call SECEdgarClient().fetch(output_dir=%s)", output)
        return

    client = SECEdgarClient()
    client.fetch(output_dir=str(output))
    logger.info("SEC EDGAR fetch complete.")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH = {
    "all":            _run_all,
    "mimic":          _run_mimic,
    "who":            _run_who,
    "clinicaltrials": _run_clinicaltrials,
    "faers":          _run_faers,
    "cdc":            _run_cdc,
    "cms":            _run_cms,
    "sec":            _run_sec,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point registered as ``healthrisk-ingest`` in pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Ensure output directory exists (unless dry-run)
    if not args.dry_run:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    handler = _DISPATCH.get(args.source)
    if handler is None:
        logger.error("Unknown source: %s", args.source)
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ingestion failed for source '%s': %s", args.source, exc, exc_info=args.verbose)
        sys.exit(1)

    sys.exit(0)
