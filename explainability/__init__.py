"""
explainability — Model explainability and interpretability toolkit.

Modules:
    shap_analyzer       - SHAP TreeExplainer / DeepExplainer wrappers
    lime_analyzer       - LIME tabular explainer with batch & global importance
    pdp                 - PDP and ICE plot generation
    counterfactual      - Counterfactual explanation generator
    model_cards         - Model card generation
"""

from explainability.lime_analyzer import LIMEAnalyzer, create_lime_report

__all__ = [
    "LIMEAnalyzer",
    "create_lime_report",
]
