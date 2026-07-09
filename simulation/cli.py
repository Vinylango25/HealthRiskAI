"""
simulation/cli.py
=================
CLI entry point for the HealthRiskAI Monte Carlo portfolio simulation.

Registered as the ``healthrisk-simulate`` console script in pyproject.toml:

    healthrisk-simulate <subcommand> [options]

Subcommands
-----------
run         Run a full Monte Carlo simulation over the configured horizon.
scenario    Generate and display a specific stress scenario without a full run.
score       Score a completed simulation run from a saved state file.
report      Generate an HTML/CSV report from persisted simulation results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="healthrisk-simulate",
        description=(
            "HealthRiskAI portfolio simulation CLI. "
            "Runs Monte Carlo simulations, generates stress scenarios, and produces reports."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Global flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--sim-config", "-c",
        default="configs/simulation_config.yaml",
        metavar="FILE",
        help="Path to simulation_config.yaml (default: configs/simulation_config.yaml).",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=1000,
        metavar="N",
        help="Number of Monte Carlo scenarios to simulate (default: 1000).",
    )
    parser.add_argument(
        "--n-years",
        type=int,
        default=10,
        metavar="N",
        help="Simulation horizon in years (default: 10 → 40 quarters).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="reports/simulation/",
        metavar="DIR",
        help="Directory for saving simulation outputs (default: reports/simulation/).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    # ── Subcommands ──────────────────────────────────────────────────────────
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # run --------------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Run a full Monte Carlo simulation.",
        description=(
            "Initialises the SimulationEngine, iterates through all scenarios, "
            "and writes final state and per-scenario results to --output-dir."
        ),
    )
    p_run.add_argument(
        "--portfolio-value",
        type=float,
        default=500_000_000.0,
        metavar="USD",
        help="Starting portfolio value in USD (default: 500000000).",
    )
    p_run.add_argument(
        "--save-state",
        action="store_true",
        help="Persist the final GameState to a JSON file in --output-dir.",
    )

    # scenario ---------------------------------------------------------------
    p_scenario = sub.add_parser(
        "scenario",
        help="Generate and display a specific stress scenario.",
        description=(
            "Uses ScenarioGenerator to create a single scenario of the requested "
            "type and prints its parameters without running a full simulation."
        ),
    )
    p_scenario.add_argument(
        "--type",
        dest="scenario_type",
        choices=["pandemic", "drug_safety_crisis", "regulatory_change", "hospital_merger"],
        required=True,
        help="Type of stress scenario to generate.",
    )
    p_scenario.add_argument(
        "--severity",
        choices=["mild", "moderate", "severe", "catastrophic"],
        default="moderate",
        help="Scenario severity level (default: moderate).",
    )
    p_scenario.add_argument(
        "--quarter",
        type=int,
        default=1,
        metavar="Q",
        help="Simulated quarter in which the scenario occurs (default: 1).",
    )

    # score ------------------------------------------------------------------
    p_score = sub.add_parser(
        "score",
        help="Score a completed simulation run.",
        description=(
            "Loads a saved GameState JSON file and runs the ScoringEngine "
            "to produce a full 1000-point breakdown."
        ),
    )
    p_score.add_argument(
        "--state-file",
        required=True,
        metavar="FILE",
        help="Path to a persisted GameState JSON file produced by 'run --save-state'.",
    )

    # report -----------------------------------------------------------------
    p_report = sub.add_parser(
        "report",
        help="Generate HTML/CSV report from simulation results.",
        description=(
            "Reads simulation output files from --output-dir and renders "
            "an HTML summary report and CSV metrics table."
        ),
    )
    p_report.add_argument(
        "--format",
        dest="report_format",
        choices=["html", "csv", "both"],
        default="both",
        help="Output format for the report (default: both).",
    )
    p_report.add_argument(
        "--results-file",
        metavar="FILE",
        help="Path to a specific simulation results JSON file (optional; "
             "defaults to the latest run in --output-dir).",
    )

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_run(args: argparse.Namespace) -> None:
    """Run the full Monte Carlo simulation."""
    import random  # noqa: PLC0415

    from simulation.engine import SimulationEngine  # noqa: PLC0415
    from simulation.portfolio_manager import PortfolioManager  # noqa: PLC0415
    from simulation.scenario_generator import ScenarioGenerator  # noqa: PLC0415

    random.seed(args.seed)

    logger.info(
        "Starting simulation | scenarios=%d  years=%d  seed=%d  portfolio=$%.0f",
        args.n_scenarios, args.n_years, args.seed, args.portfolio_value,
    )

    engine = SimulationEngine(
        config_path=args.sim_config,
        n_years=args.n_years,
        seed=args.seed,
    )

    results = engine.run(
        n_scenarios=args.n_scenarios,
        starting_portfolio_value=args.portfolio_value,
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "simulation_results.json"
    with results_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Results written to %s", results_file)

    if args.save_state:
        state_file = output_path / "game_state.json"
        with state_file.open("w", encoding="utf-8") as fh:
            json.dump(engine.get_state(), fh, indent=2, default=str)
        logger.info("GameState persisted to %s", state_file)

    logger.info("Simulation complete.")


def _handle_scenario(args: argparse.Namespace) -> None:
    """Generate and display a single stress scenario."""
    import random  # noqa: PLC0415

    from simulation.scenario_generator import ScenarioGenerator, ScenarioType, Severity  # noqa: PLC0415

    random.seed(args.seed)

    logger.info(
        "Generating scenario | type=%s  severity=%s  quarter=%d",
        args.scenario_type, args.severity, args.quarter,
    )

    generator = ScenarioGenerator(config_path=args.sim_config, seed=args.seed)
    scenario = generator.generate(
        scenario_type=ScenarioType(args.scenario_type),
        severity=Severity(args.severity),
        quarter=args.quarter,
    )

    # Pretty-print scenario parameters to stdout
    print(json.dumps(scenario, indent=2, default=str))

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"scenario_{args.scenario_type}_{args.severity}.json"
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(scenario, fh, indent=2, default=str)
    logger.info("Scenario saved to %s", out_file)


def _handle_score(args: argparse.Namespace) -> None:
    """Score a completed simulation run from a saved state file."""
    from simulation.scoring import ScoringEngine  # noqa: PLC0415

    state_path = Path(args.state_file)
    if not state_path.exists():
        logger.error("State file not found: %s", state_path)
        sys.exit(1)

    logger.info("Loading GameState from %s", state_path)
    with state_path.open("r", encoding="utf-8") as fh:
        state = json.load(fh)

    scorer = ScoringEngine()
    breakdown = scorer.score(state)

    print(json.dumps(breakdown, indent=2, default=str))

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    score_file = output_path / "score_breakdown.json"
    with score_file.open("w", encoding="utf-8") as fh:
        json.dump(breakdown, fh, indent=2, default=str)
    logger.info("Score breakdown written to %s", score_file)


def _handle_report(args: argparse.Namespace) -> None:
    """Generate an HTML/CSV report from simulation results."""
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine results file to use
    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = output_path / "simulation_results.json"

    if not results_path.exists():
        logger.error(
            "Results file not found: %s. Run 'healthrisk-simulate run' first.",
            results_path,
        )
        sys.exit(1)

    logger.info("Loading simulation results from %s", results_path)
    with results_path.open("r", encoding="utf-8") as fh:
        results = json.load(fh)

    # ── CSV ─────────────────────────────────────────────────────────────────
    if args.report_format in ("csv", "both"):
        import csv  # noqa: PLC0415

        csv_path = output_path / "simulation_report.csv"
        # Flatten top-level keys into a single-row CSV; per-scenario rows if list
        rows = results if isinstance(results, list) else [results]
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            logger.info("CSV report written to %s", csv_path)

    # ── HTML ─────────────────────────────────────────────────────────────────
    if args.report_format in ("html", "both"):
        html_path = output_path / "simulation_report.html"
        rows = results if isinstance(results, list) else [results]
        header = "".join(f"<th>{k}</th>" for k in rows[0].keys()) if rows else ""
        body_rows = "".join(
            "<tr>" + "".join(f"<td>{v}</td>" for v in row.values()) + "</tr>"
            for row in rows
        )
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>HealthRiskAI Simulation Report</title>"
            "<style>body{font-family:sans-serif;padding:2rem}"
            "table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid #ccc;padding:.4rem .8rem;text-align:left}"
            "th{background:#1a3a5c;color:#fff}</style></head><body>"
            "<h1>HealthRiskAI – Monte Carlo Simulation Report</h1>"
            f"<table><thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody></table>"
            "</body></html>"
        )
        html_path.write_text(html, encoding="utf-8")
        logger.info("HTML report written to %s", html_path)

    logger.info("Report generation complete.")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH = {
    "run":      _handle_run,
    "scenario": _handle_scenario,
    "score":    _handle_score,
    "report":   _handle_report,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point registered as ``healthrisk-simulate`` in pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = _DISPATCH.get(args.command)
    if handler is None:
        logger.error("Unknown command: %s", args.command)
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        logger.warning("Simulation interrupted by user.")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.error("Command '%s' failed: %s", args.command, exc, exc_info=args.verbose)
        sys.exit(1)

    sys.exit(0)
