"""
models/cli.py
=============
CLI entry point for HealthRiskAI model training and evaluation.

Registered as the ``healthrisk-train`` console script in pyproject.toml:

    healthrisk-train <subcommand> [options]

Subcommands
-----------
tabular     Train tabular models (readmission, cost, hospital_default, claims).
survival    Train survival models (CoxPH, DeepSurv, DynamicDeepHit).
nlp         Fine-tune ClinicalBERT NLP models (classifier, NER, complexity).
gnn         Train the Graph Attention Network (GAT) model.
ensemble    Train the stacking meta-learner across all base models.
evaluate    Evaluate saved model artifacts and produce metric reports.
crossval    Run cross-validation for any model family.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="healthrisk-train",
        description=(
            "HealthRiskAI model training CLI. "
            "Trains, evaluates, and cross-validates all model families."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Global flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--model-config", "-c",
        default="configs/model_config.yaml",
        metavar="FILE",
        help="Path to model_config.yaml (default: configs/model_config.yaml).",
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="data/processed/",
        metavar="DIR",
        help="Directory containing processed feature files (default: data/processed/).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="reports/models/",
        metavar="DIR",
        help="Directory for saving model artifacts and reports (default: reports/models/).",
    )
    parser.add_argument(
        "--experiment", "-e",
        default="healthrisk-default",
        metavar="NAME",
        help="MLflow experiment name (default: healthrisk-default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and validate inputs without starting any training.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    # ── Subcommands ──────────────────────────────────────────────────────────
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # tabular ----------------------------------------------------------------
    p_tab = sub.add_parser(
        "tabular",
        help="Train tabular models.",
        description="Train readmission, cost, hospital_default, and claims models.",
    )
    p_tab.add_argument(
        "--model",
        choices=["readmission", "cost", "hospital_default", "claims", "all"],
        default="all",
        help="Which tabular model to train (default: all).",
    )

    # survival ---------------------------------------------------------------
    p_surv = sub.add_parser(
        "survival",
        help="Train survival analysis models.",
        description="Train CoxPH, DeepSurv, and DynamicDeepHit models.",
    )
    p_surv.add_argument(
        "--model",
        choices=["cox_ph", "deepsurv", "deephit", "all"],
        default="all",
        help="Which survival model to train (default: all).",
    )

    # nlp --------------------------------------------------------------------
    p_nlp = sub.add_parser(
        "nlp",
        help="Fine-tune clinical NLP models.",
        description="Fine-tune ClinicalBERT classifier, NER pipeline, and complexity scorer.",
    )
    p_nlp.add_argument(
        "--model",
        choices=["bert_classifier", "ner", "complexity", "all"],
        default="all",
        help="Which NLP model to train (default: all).",
    )

    # gnn --------------------------------------------------------------------
    sub.add_parser(
        "gnn",
        help="Train the Graph Attention Network model.",
        description="Builds a patient–hospital graph and trains the GAT model.",
    )

    # ensemble ---------------------------------------------------------------
    sub.add_parser(
        "ensemble",
        help="Train the stacking meta-learner.",
        description="Trains a Ridge meta-learner on top of all base-model OOF predictions.",
    )

    # evaluate ---------------------------------------------------------------
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate saved model artifacts.",
        description="Loads saved models, runs evaluation, and writes metric reports.",
    )
    p_eval.add_argument(
        "--family",
        choices=["tabular", "survival", "nlp", "gnn", "ensemble", "all"],
        default="all",
        help="Model family to evaluate (default: all).",
    )

    # crossval ---------------------------------------------------------------
    p_cv = sub.add_parser(
        "crossval",
        help="Run cross-validation.",
        description="Run time-aware cross-validation for a given model family.",
    )
    p_cv.add_argument(
        "--family",
        choices=["tabular", "survival", "nlp", "gnn", "ensemble"],
        required=True,
        help="Model family to cross-validate.",
    )
    p_cv.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5).",
    )

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_tabular(args: argparse.Namespace) -> None:
    """Train one or all tabular models."""
    import mlflow  # noqa: PLC0415
    from models.tabular.trainer import TimeAwareCrossValidator  # noqa: PLC0415

    model_map = {
        "readmission":      "models.tabular.readmission_model",
        "cost":             "models.tabular.cost_predictor",
        "hospital_default": "models.tabular.hospital_default_model",
        "claims":           "models.tabular.lightgbm_claims",
    }
    targets = list(model_map.keys()) if args.model == "all" else [args.model]

    if args.dry_run:
        logger.info("[dry-run] Would train tabular models: %s", targets)
        logger.info("[dry-run] config=%s  data_dir=%s  output_dir=%s", args.model_config, args.data_dir, args.output_dir)
        return

    mlflow.set_experiment(args.experiment)
    for target in targets:
        logger.info("Training tabular model: %s", target)
        import importlib  # noqa: PLC0415
        mod = importlib.import_module(model_map[target])
        mod.train(
            config_path=args.model_config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    logger.info("Tabular training complete.")


def _handle_survival(args: argparse.Namespace) -> None:
    """Train one or all survival models."""
    import mlflow  # noqa: PLC0415

    model_map = {
        "cox_ph":   "models.survival.cox_ph",
        "deepsurv": "models.survival.deepsurv",
        "deephit":  "models.survival.dynamic_deephit",
    }
    targets = list(model_map.keys()) if args.model == "all" else [args.model]

    if args.dry_run:
        logger.info("[dry-run] Would train survival models: %s", targets)
        return

    mlflow.set_experiment(args.experiment)
    for target in targets:
        logger.info("Training survival model: %s", target)
        import importlib  # noqa: PLC0415
        mod = importlib.import_module(model_map[target])
        mod.train(
            config_path=args.model_config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    logger.info("Survival model training complete.")


def _handle_nlp(args: argparse.Namespace) -> None:
    """Fine-tune clinical NLP models."""
    import mlflow  # noqa: PLC0415

    model_map = {
        "bert_classifier": "models.clinical_nlp.bert_classifier",
        "ner":             "models.clinical_nlp.ner_pipeline",
        "complexity":      "models.clinical_nlp.complexity_scorer",
    }
    targets = list(model_map.keys()) if args.model == "all" else [args.model]

    if args.dry_run:
        logger.info("[dry-run] Would fine-tune NLP models: %s", targets)
        return

    mlflow.set_experiment(args.experiment)
    for target in targets:
        logger.info("Fine-tuning NLP model: %s", target)
        import importlib  # noqa: PLC0415
        mod = importlib.import_module(model_map[target])
        mod.train(
            config_path=args.model_config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    logger.info("NLP fine-tuning complete.")


def _handle_gnn(args: argparse.Namespace) -> None:
    """Train the Graph Attention Network."""
    import mlflow  # noqa: PLC0415
    from models.graph_network.trainer import GNNTrainer  # noqa: PLC0415

    if args.dry_run:
        logger.info("[dry-run] Would build graph and train GAT model.")
        return

    mlflow.set_experiment(args.experiment)
    logger.info("Building patient–hospital graph...")
    from models.graph_network.graph_builder import GraphBuilder  # noqa: PLC0415
    builder = GraphBuilder(config_path=args.model_config)
    graph = builder.build(data_dir=args.data_dir)

    logger.info("Training GAT model...")
    trainer = GNNTrainer(config_path=args.model_config, output_dir=args.output_dir)
    trainer.train(graph)
    logger.info("GNN training complete.")


def _handle_ensemble(args: argparse.Namespace) -> None:
    """Train the stacking meta-learner."""
    import mlflow  # noqa: PLC0415
    from models.ensemble.meta_learner import StackingMetaLearner  # noqa: PLC0415

    if args.dry_run:
        logger.info("[dry-run] Would train stacking meta-learner.")
        return

    mlflow.set_experiment(args.experiment)
    logger.info("Training stacking meta-learner...")
    learner = StackingMetaLearner(
        config_path=args.model_config,
        output_dir=args.output_dir,
    )
    learner.fit(data_dir=args.data_dir)
    logger.info("Ensemble meta-learner training complete.")


def _handle_evaluate(args: argparse.Namespace) -> None:
    """Evaluate saved model artifacts."""
    from models.ensemble.evaluator import EnsembleEvaluator  # noqa: PLC0415
    from models.survival.evaluator import SurvivalEvaluator  # noqa: PLC0415

    if args.dry_run:
        logger.info("[dry-run] Would evaluate model family: %s", args.family)
        return

    logger.info("Evaluating model family: %s", args.family)
    if args.family in ("ensemble", "all"):
        ev = EnsembleEvaluator(model_dir=args.output_dir)
        ev.evaluate(data_dir=args.data_dir)
    if args.family in ("survival", "all"):
        ev = SurvivalEvaluator(model_dir=args.output_dir)
        ev.evaluate(data_dir=args.data_dir)
    logger.info("Evaluation complete. Reports written to %s", args.output_dir)


def _handle_crossval(args: argparse.Namespace) -> None:
    """Run cross-validation."""
    from models.ensemble.cross_validator import CrossValidator  # noqa: PLC0415

    if args.dry_run:
        logger.info("[dry-run] Would run %d-fold CV for family: %s", args.n_folds, args.family)
        return

    logger.info("Running %d-fold cross-validation | family=%s", args.n_folds, args.family)
    cv = CrossValidator(
        config_path=args.model_config,
        n_folds=args.n_folds,
        output_dir=args.output_dir,
    )
    cv.run(family=args.family, data_dir=args.data_dir)
    logger.info("Cross-validation complete.")


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH = {
    "tabular":  _handle_tabular,
    "survival": _handle_survival,
    "nlp":      _handle_nlp,
    "gnn":      _handle_gnn,
    "ensemble": _handle_ensemble,
    "evaluate": _handle_evaluate,
    "crossval": _handle_crossval,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point registered as ``healthrisk-train`` in pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if not args.dry_run:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    handler = _DISPATCH.get(args.command)
    if handler is None:
        logger.error("Unknown command: %s", args.command)
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.error("Command '%s' failed: %s", args.command, exc, exc_info=args.verbose)
        sys.exit(1)

    sys.exit(0)
