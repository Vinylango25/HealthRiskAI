"""
model_cards.py
==============
Auto-generate Model Cards for all HealthRiskAI components.

Covers all 10 HealthRiskAI models:
  1.  financial_risk_classifier
  2.  clinical_risk_scorer
  3.  readmission_predictor
  4.  er_utilization_forecaster
  5.  staffing_optimizer
  6.  revenue_cycle_analyzer
  7.  drug_spend_predictor
  8.  quality_metrics_ranker
  9.  patient_outcome_forecaster
  10. network_adequacy_scorer

Each model card documents: purpose, training data, performance, limitations,
fairness considerations, and intended use — following the Google Model Card
specification (Mitchell et al., 2019).
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Known models registry
# ---------------------------------------------------------------------------

HEALTHRISKAI_MODELS: Dict[str, Dict[str, str]] = {
    "financial_risk_classifier": {
        "task": "Multi-class classification",
        "description": "Classifies health system financial risk into Low / Medium / High / Critical tiers.",
        "primary_metric": "macro-F1",
    },
    "clinical_risk_scorer": {
        "task": "Regression / risk scoring",
        "description": "Produces a continuous clinical risk score (0–100) based on HCC codes and clinical utilisation.",
        "primary_metric": "MAE",
    },
    "readmission_predictor": {
        "task": "Binary classification",
        "description": "Predicts 30-day all-cause hospital readmission probability.",
        "primary_metric": "AUC-ROC",
    },
    "er_utilization_forecaster": {
        "task": "Time-series regression",
        "description": "Forecasts emergency department visit volume over a 30-day horizon.",
        "primary_metric": "MAPE",
    },
    "staffing_optimizer": {
        "task": "Multi-output regression",
        "description": "Estimates optimal nursing and support staff ratios given census and acuity data.",
        "primary_metric": "RMSE",
    },
    "revenue_cycle_analyzer": {
        "task": "Anomaly detection + classification",
        "description": "Identifies revenue-cycle anomalies and categorises denial root causes.",
        "primary_metric": "Precision@K",
    },
    "drug_spend_predictor": {
        "task": "Regression",
        "description": "Predicts per-member drug spend for the next quarter.",
        "primary_metric": "R²",
    },
    "quality_metrics_ranker": {
        "task": "Learning-to-rank",
        "description": "Ranks health system quality metrics by relative impact on composite quality score.",
        "primary_metric": "NDCG@10",
    },
    "patient_outcome_forecaster": {
        "task": "Survival analysis",
        "description": "Estimates time-to-adverse-event for high-risk patient cohorts.",
        "primary_metric": "C-index",
    },
    "network_adequacy_scorer": {
        "task": "Binary classification",
        "description": "Predicts whether a provider network meets CMS adequacy standards.",
        "primary_metric": "F1",
    },
}


# ---------------------------------------------------------------------------
# ModelCard dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """
    Structured model card following Google's Model Card specification.

    Parameters
    ----------
    model_name : str
    version : str
    task : str
    description : str
    training_data : dict
        Keys: source, n_samples, date_range, features, label_distribution
    performance_metrics : dict
        Keys: metric_name → value or {overall, subgroups}
    limitations : list of str
    fairness : dict
        Keys: demographic_parity, equal_opportunity, notes
    intended_use : dict
        Keys: primary_use_case, out_of_scope, users
    metadata : dict
        Arbitrary extra metadata (framework, author, licence, …)
    generated_at : str  ISO timestamp
    """

    model_name: str
    version: str
    task: str
    description: str
    training_data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    fairness: Dict[str, Any] = field(default_factory=dict)
    intended_use: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation (JSON-serialisable)."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Render the model card as a Markdown document."""
        md = []

        def h(level: int, text: str) -> str:
            return "#" * level + " " + text

        md.append(h(1, f"Model Card: {self.model_name}"))
        md.append(f"**Version:** {self.version}  |  **Generated:** {self.generated_at}\n")

        md.append(h(2, "Model Details"))
        md.append(f"- **Task:** {self.task}")
        md.append(f"- **Description:** {self.description}")
        for k, v in self.metadata.items():
            md.append(f"- **{k.replace('_', ' ').title()}:** {v}")
        md.append("")

        md.append(h(2, "Training Data"))
        td = self.training_data
        if td:
            md.append(f"- **Source:** {td.get('source', 'N/A')}")
            md.append(f"- **Samples:** {td.get('n_samples', 'N/A')}")
            md.append(f"- **Date Range:** {td.get('date_range', 'N/A')}")
            md.append(f"- **Features:** {td.get('features', 'N/A')}")
            if "label_distribution" in td:
                md.append(f"- **Label Distribution:** {td['label_distribution']}")
        else:
            md.append("_Not specified._")
        md.append("")

        md.append(h(2, "Performance Metrics"))
        if self.performance_metrics:
            for metric, value in self.performance_metrics.items():
                if isinstance(value, dict):
                    md.append(f"### {metric}")
                    for sub_k, sub_v in value.items():
                        md.append(f"  - **{sub_k}:** {sub_v}")
                else:
                    md.append(f"- **{metric}:** {value}")
        else:
            md.append("_Not yet evaluated._")
        md.append("")

        md.append(h(2, "Limitations"))
        if self.limitations:
            for lim in self.limitations:
                md.append(f"- {lim}")
        else:
            md.append("_None documented._")
        md.append("")

        md.append(h(2, "Fairness & Bias"))
        if self.fairness:
            for k, v in self.fairness.items():
                md.append(f"- **{k.replace('_', ' ').title()}:** {v}")
        else:
            md.append("_Fairness evaluation not yet completed._")
        md.append("")

        md.append(h(2, "Intended Use"))
        iu = self.intended_use
        if iu:
            md.append(f"- **Primary Use Case:** {iu.get('primary_use_case', 'N/A')}")
            md.append(f"- **Users:** {iu.get('users', 'N/A')}")
            oob = iu.get("out_of_scope", [])
            if oob:
                md.append("- **Out-of-Scope Uses:**")
                for o in (oob if isinstance(oob, list) else [oob]):
                    md.append(f"  - {o}")
        else:
            md.append("_Not specified._")
        md.append("")

        return "\n".join(md)

    def save(
        self,
        output_dir: str = ".",
        fmt: str = "markdown",
    ) -> Path:
        """
        Persist the model card to disk.

        Parameters
        ----------
        output_dir : str
            Directory to write into.
        fmt : str
            'markdown' → .md file, 'json' → .json file, 'both' → both.

        Returns
        -------
        Path  (primary output file)
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = self.model_name.replace(" ", "_").lower()
        primary: Optional[Path] = None

        if fmt in ("markdown", "both"):
            p = out / f"{slug}_v{self.version}.md"
            p.write_text(self.to_markdown(), encoding="utf-8")
            logger.info("Model card saved → %s", p)
            primary = p

        if fmt in ("json", "both"):
            p = out / f"{slug}_v{self.version}.json"
            p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
            logger.info("Model card saved → %s", p)
            if primary is None:
                primary = p

        return primary  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ModelCardGenerator
# ---------------------------------------------------------------------------

class ModelCardGenerator:
    """
    Auto-generate ModelCards for HealthRiskAI models.

    Parameters
    ----------
    default_version : str
    default_output_dir : str
    """

    def __init__(
        self,
        default_version: str = "1.0.0",
        default_output_dir: str = "./model_cards",
    ) -> None:
        self.default_version = default_version
        self.default_output_dir = default_output_dir

    def generate(
        self,
        model_name: str,
        eval_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
    ) -> ModelCard:
        """
        Generate a ModelCard for a given model.

        Parameters
        ----------
        model_name : str
            Must be one of the keys in HEALTHRISKAI_MODELS (or arbitrary for custom models).
        eval_results : dict
            Evaluation results dict. Expected keys (all optional):
              training_data, performance_metrics, limitations, fairness, intended_use
        metadata : dict, optional
            Arbitrary metadata (framework, author, licence, git_sha, …)
        version : str, optional
            Overrides default_version.

        Returns
        -------
        ModelCard
        """
        known = HEALTHRISKAI_MODELS.get(model_name, {})
        task = known.get("task", eval_results.get("task", "Unknown"))
        description = known.get("description", eval_results.get("description", ""))

        card = ModelCard(
            model_name=model_name,
            version=version or self.default_version,
            task=task,
            description=description,
            training_data=eval_results.get("training_data", {}),
            performance_metrics=eval_results.get("performance_metrics", {}),
            limitations=eval_results.get("limitations", self._default_limitations(model_name)),
            fairness=eval_results.get("fairness", self._default_fairness()),
            intended_use=eval_results.get(
                "intended_use",
                self._default_intended_use(model_name, known),
            ),
            metadata=metadata or {},
        )

        logger.info("Generated model card for '%s' v%s.", model_name, card.version)
        return card

    def generate_all(
        self,
        eval_results_map: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
        version: Optional[str] = None,
        save: bool = False,
        fmt: str = "markdown",
    ) -> Dict[str, ModelCard]:
        """
        Generate cards for all 10 HealthRiskAI models.

        Parameters
        ----------
        eval_results_map : dict, optional
            {model_name: eval_results}. Uses empty dicts for missing entries.
        metadata_map : dict, optional
            {model_name: metadata}
        save : bool
            If True, write each card to default_output_dir.
        fmt : str
            'markdown', 'json', or 'both'.

        Returns
        -------
        dict of {model_name: ModelCard}
        """
        results_map = eval_results_map or {}
        meta_map = metadata_map or {}
        cards: Dict[str, ModelCard] = {}

        for name in HEALTHRISKAI_MODELS:
            card = self.generate(
                model_name=name,
                eval_results=results_map.get(name, {}),
                metadata=meta_map.get(name),
                version=version,
            )
            if save:
                card.save(self.default_output_dir, fmt=fmt)
            cards[name] = card

        return cards

    # ------------------------------------------------------------------
    # Default content helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_limitations(model_name: str) -> List[str]:
        base = [
            "Model trained on historical data; performance may degrade with distribution shift.",
            "Requires complete feature vector; missing values imputed with median during inference.",
            "Not validated on patient populations outside the training geography.",
        ]
        extra: Dict[str, List[str]] = {
            "readmission_predictor": [
                "Does not account for social determinants of health.",
                "Performance lower for patients with rare diagnoses.",
            ],
            "er_utilization_forecaster": [
                "Forecast accuracy degrades beyond 30-day horizon.",
                "Does not model pandemic-level demand shocks.",
            ],
            "staffing_optimizer": [
                "Optimisation does not account for union contract constraints.",
            ],
        }
        return base + extra.get(model_name, [])

    @staticmethod
    def _default_fairness() -> Dict[str, Any]:
        return {
            "demographic_parity": "Not yet evaluated — evaluation planned for Q3.",
            "equal_opportunity": "Not yet evaluated — evaluation planned for Q3.",
            "protected_attributes": ["race", "gender", "age_group", "payer_type"],
            "notes": (
                "Models should not be used as sole basis for resource allocation decisions "
                "without human oversight."
            ),
        }

    @staticmethod
    def _default_intended_use(model_name: str, known: Dict[str, str]) -> Dict[str, Any]:
        return {
            "primary_use_case": known.get("description", f"Support tool for {model_name}."),
            "users": "Health system analysts, clinical operations teams, risk managers.",
            "out_of_scope": [
                "Direct clinical diagnosis or treatment decisions.",
                "Individual patient-level legal or financial determinations.",
                "Deployment outside US healthcare regulatory context without re-validation.",
            ],
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logger.info("=== ModelCardGenerator smoke test ===")

    generator = ModelCardGenerator(default_version="1.2.0")

    # Simulate eval results for one model
    eval_results = {
        "training_data": {
            "source": "HealthRiskAI internal EHR + claims dataset",
            "n_samples": 125_000,
            "date_range": "2019-01-01 to 2023-12-31",
            "features": 87,
            "label_distribution": {"Low": "42%", "Medium": "31%", "High": "18%", "Critical": "9%"},
        },
        "performance_metrics": {
            "macro_f1": 0.81,
            "accuracy": 0.84,
            "auc_roc": 0.93,
            "subgroup_f1": {
                "Medicare Advantage": 0.79,
                "Medicaid": 0.76,
                "Commercial": 0.85,
            },
        },
        "limitations": [
            "Model trained on historical data; performance may degrade with distribution shift.",
            "Critical tier has lowest recall (0.71) due to class imbalance.",
        ],
        "fairness": {
            "demographic_parity_gap": 0.04,
            "equal_opportunity_gap": 0.06,
            "notes": "Gap within acceptable range per internal policy (< 0.10).",
        },
    }

    metadata = {
        "framework": "XGBoost 1.7.6",
        "author": "HealthRiskAI MLOps Team",
        "licence": "Proprietary",
        "git_sha": "a1b2c3d4",
    }

    card = generator.generate(
        model_name="financial_risk_classifier",
        eval_results=eval_results,
        metadata=metadata,
    )

    md = card.to_markdown()
    logger.info("Markdown card preview (first 5 lines):\n%s", "\n".join(md.splitlines()[:5]))

    card_dict = card.to_dict()
    assert card_dict["model_name"] == "financial_risk_classifier"
    assert card_dict["version"] == "1.2.0"

    # Test save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = card.save(tmpdir, fmt="both")
        assert path.exists()
        logger.info("Card saved to temp dir: %s", tmpdir)

    # Generate all 10 model cards
    all_cards = generator.generate_all()
    assert len(all_cards) == 10
    logger.info("Generated cards for all 10 models: %s",
                list(all_cards.keys()))

    # Verify each card has required fields
    for name, c in all_cards.items():
        assert c.model_name == name
        assert c.task
        assert isinstance(c.limitations, list)
        assert isinstance(c.fairness, dict)

    logger.info("✅ ModelCardGenerator smoke test PASSED.")
