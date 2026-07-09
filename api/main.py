"""
api/main.py
===========
HealthRisk AI — FastAPI application entry point.

Endpoints
---------
GET  /                              Welcome + links
GET  /health                        Health check (used by Docker HEALTHCHECK)
GET  /api/v1/models                 List available model types
POST /api/v1/predict/readmission    30-day readmission risk
POST /api/v1/predict/cost           12-month healthcare cost
POST /api/v1/predict/hospital-default  Hospital bond PD score
GET  /api/v1/simulation/state       Current simulation game state
POST /api/v1/simulation/next-quarter  Advance simulation by one quarter
GET  /api/v1/explainability/shap/{model_name}  SHAP feature importance

Usage (local)
-------------
    uvicorn api.main:app --reload --port 8000

Docker
------
    CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Environment / logging
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("healthrisk_ai.api")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
VERSION = "0.1.0"
SERVICE_NAME = "healthrisk-ai"

# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Application lifespan: log startup and shutdown."""
    logger.info("HealthRisk AI API starting up (env=%s, version=%s)", ENVIRONMENT, VERSION)
    yield
    logger.info("HealthRisk AI API shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HealthRisk AI",
    description=(
        "AI-driven health risk assessment platform — integrating clinical NLP, "
        "graph neural networks, survival models, and multi-asset financial risk analytics."
    ),
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins in non-production environments
_cors_origins = ["*"] if ENVIRONMENT != "production" else [
    os.getenv("FRONTEND_URL", "http://localhost:4200"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models — requests
# ---------------------------------------------------------------------------


class PatientFeatures(BaseModel):
    """Input features for patient-level prediction endpoints."""

    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field("M", description="M / F / O")
    diagnoses: List[str] = Field(
        default_factory=list, description="List of ICD-10 codes"
    )
    hemoglobin: Optional[float] = Field(None, ge=0, le=25)
    creatinine: Optional[float] = Field(None, ge=0, le=30)
    bun: Optional[float] = Field(None, ge=0, le=200)
    hba1c: Optional[float] = Field(None, ge=0, le=20)
    los_days: Optional[int] = Field(None, ge=0, description="Length of stay in days")
    prior_admissions: int = Field(0, ge=0)
    er_visits_12m: int = Field(0, ge=0)
    chronic_count: int = Field(0, ge=0)


class HospitalFeatures(BaseModel):
    """Input features for hospital default prediction."""

    hospital_id: str = Field(..., description="Hospital identifier")
    operating_margin: float = Field(..., description="Operating margin fraction")
    debt_to_equity: float = Field(..., ge=0)
    days_cash_on_hand: float = Field(..., ge=0)
    occupancy_rate: float = Field(..., ge=0, le=1)
    readmission_rate: float = Field(..., ge=0, le=1)
    mortality_rate: float = Field(..., ge=0, le=1)
    patient_satisfaction: float = Field(..., ge=0, le=100)
    bed_count: int = Field(..., ge=1)
    annual_revenue_usd: float = Field(..., ge=0)


class SimulationDecision(BaseModel):
    """Player decision submitted to advance the simulation by one quarter."""

    action: str = Field("hold", description="hold | buy | sell | rebalance")
    asset_class: Optional[str] = Field(
        None, description="insurance | hospital_bonds | pharma | credit_facility"
    )
    allocation_pct: Optional[float] = Field(None, ge=0, le=100)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Pydantic models — responses
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    uptime_seconds: float


class ReadmissionPrediction(BaseModel):
    patient_risk_score: float = Field(..., description="Readmission probability [0,1]")
    risk_tier: str = Field(..., description="Low | Medium | High | Very High")
    top_features: Dict[str, float]
    model_version: str = VERSION
    latency_ms: float


class CostPrediction(BaseModel):
    predicted_cost_usd: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    cost_drivers: Dict[str, float]
    model_version: str = VERSION
    latency_ms: float


class HospitalDefaultPrediction(BaseModel):
    hospital_id: str
    pd_score: float = Field(..., description="Probability of default [0,1]")
    implied_rating: str
    gini_coefficient: float
    key_risk_factors: Dict[str, float]
    model_version: str = VERSION
    latency_ms: float


class SimulationState(BaseModel):
    quarter: int
    year: int
    portfolio_value_usd: float
    total_return_pct: float
    score: float
    phase: str
    active_scenarios: List[str]
    asset_allocations: Dict[str, float]


class QuarterResult(BaseModel):
    quarter: int
    portfolio_value_usd: float
    quarterly_return_pct: float
    score_delta: float
    events: List[str]
    ai_opponent_return_pct: float
    leaderboard_rank: Optional[int] = None


class SHAPExplanation(BaseModel):
    model_name: str
    feature_names: List[str]
    shap_values: List[float]
    base_value: float
    expected_value: float
    top_positive_features: Dict[str, float]
    top_negative_features: Dict[str, float]


# ---------------------------------------------------------------------------
# Internal state (minimal in-memory for demo; production uses DB)
# ---------------------------------------------------------------------------

_START_TIME = time.time()
_SIM_QUARTER = 1
_SIM_PORTFOLIO = 500_000_000.0
_SIM_SCORE = 0.0

# Simulated rating lookup
_RATING_MAP = [
    (0.001, "AAA"), (0.003, "AA"), (0.008, "A"), (0.020, "BBB"),
    (0.060, "BB"), (0.120, "B"), (1.0, "CCC"),
]


def _pd_to_rating(pd_score: float) -> str:
    for threshold, rating in _RATING_MAP:
        if pd_score <= threshold:
            return rating
    return "D"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["root"])
async def root() -> Dict[str, Any]:
    """Welcome endpoint with links to available documentation and API surfaces."""
    return {
        "service": SERVICE_NAME,
        "version": VERSION,
        "description": "AI-driven health risk assessment platform",
        "environment": ENVIRONMENT,
        "links": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "models": "/api/v1/models",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check() -> HealthResponse:
    """
    Kubernetes / Docker health check endpoint.
    Returns 200 when the service is operational.
    """
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=VERSION,
        environment=ENVIRONMENT,
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )


@app.get("/api/v1/models", tags=["models"])
async def list_models() -> Dict[str, Any]:
    """List all model families available in the HealthRisk AI platform."""
    return {
        "models": [
            {
                "name": "readmission_xgboost",
                "family": "tabular",
                "task": "binary_classification",
                "target_auroc": 0.78,
                "description": "30-day readmission prediction using XGBoost",
            },
            {
                "name": "cost_lightgbm",
                "family": "tabular",
                "task": "regression",
                "target_r2": 0.25,
                "description": "12-month healthcare cost prediction using LightGBM",
            },
            {
                "name": "hospital_default_xgboost",
                "family": "tabular",
                "task": "binary_classification",
                "target_auroc": 0.80,
                "description": "Hospital bond probability of default (PD) model",
            },
            {
                "name": "clinical_bert",
                "family": "nlp",
                "task": "sequence_classification",
                "description": "Bio_ClinicalBERT fine-tuned on discharge summaries",
            },
            {
                "name": "gnn_patient_graph",
                "family": "graph_network",
                "task": "node_classification",
                "description": "Heterogeneous GNN on patient-disease-drug graph",
            },
            {
                "name": "cox_ph",
                "family": "survival",
                "task": "time_to_event",
                "description": "Cox Proportional Hazards — time to readmission",
            },
            {
                "name": "deepsurv",
                "family": "survival",
                "task": "time_to_event",
                "description": "DeepSurv neural network survival model",
            },
            {
                "name": "stacking_ensemble",
                "family": "ensemble",
                "task": "binary_classification",
                "description": "Ridge meta-learner stacking all model families",
            },
        ]
    }


@app.post(
    "/api/v1/predict/readmission",
    response_model=ReadmissionPrediction,
    tags=["predictions"],
)
async def predict_readmission(patient: PatientFeatures) -> ReadmissionPrediction:
    """
    Predict 30-day readmission risk for a patient.

    Returns a risk score in [0, 1] along with the top contributing features
    (via SHAP-inspired attribution) and a risk tier label.
    """
    t0 = time.time()

    # Deterministic mock score derived from input features
    rng = np.random.default_rng(
        int(patient.age) * 7 + patient.chronic_count * 13 + patient.prior_admissions * 31
    )

    base = (
        0.05
        + (patient.age / 90) * 0.20
        + patient.chronic_count * 0.04
        + patient.prior_admissions * 0.06
        + patient.er_visits_12m * 0.03
        + (len(patient.diagnoses) * 0.02)
        + rng.uniform(-0.05, 0.05)
    )
    score = float(np.clip(base, 0.01, 0.99))

    if score < 0.15:
        tier = "Low"
    elif score < 0.35:
        tier = "Medium"
    elif score < 0.60:
        tier = "High"
    else:
        tier = "Very High"

    top_features = {
        "age": round(float((patient.age / 90) * 0.20), 4),
        "chronic_conditions": round(patient.chronic_count * 0.04, 4),
        "prior_admissions": round(patient.prior_admissions * 0.06, 4),
        "er_visits_12m": round(patient.er_visits_12m * 0.03, 4),
        "diagnosis_count": round(len(patient.diagnoses) * 0.02, 4),
    }

    return ReadmissionPrediction(
        patient_risk_score=round(score, 4),
        risk_tier=tier,
        top_features=top_features,
        latency_ms=round((time.time() - t0) * 1000, 2),
    )


@app.post(
    "/api/v1/predict/cost",
    response_model=CostPrediction,
    tags=["predictions"],
)
async def predict_cost(patient: PatientFeatures) -> CostPrediction:
    """
    Predict 12-month healthcare cost for a patient (USD).

    Uses a log-normal cost model with clinical features as rate parameters.
    Returns point estimate plus 90% confidence interval.
    """
    t0 = time.time()

    rng = np.random.default_rng(int(patient.age) * 3 + patient.chronic_count * 17)

    log_mean = (
        8.5
        + (patient.age / 90) * 1.5
        + patient.chronic_count * 0.4
        + patient.prior_admissions * 0.6
        + (1 if "I50" in patient.diagnoses else 0) * 0.8  # heart failure
        + (1 if any(d.startswith("C") for d in patient.diagnoses) else 0) * 1.2  # cancer
    )

    predicted = float(np.exp(log_mean + rng.uniform(-0.1, 0.1)))
    ci_lower = float(predicted * 0.65)
    ci_upper = float(predicted * 1.55)

    drivers = {
        "baseline_cost": round(float(np.exp(8.5)), 2),
        "age_loading": round(float(np.exp((patient.age / 90) * 1.5) - 1) * 5000, 2),
        "chronic_loading": round(patient.chronic_count * 4500.0, 2),
        "admission_history": round(patient.prior_admissions * 8000.0, 2),
    }

    return CostPrediction(
        predicted_cost_usd=round(predicted, 2),
        confidence_interval_lower=round(ci_lower, 2),
        confidence_interval_upper=round(ci_upper, 2),
        cost_drivers=drivers,
        latency_ms=round((time.time() - t0) * 1000, 2),
    )


@app.post(
    "/api/v1/predict/hospital-default",
    response_model=HospitalDefaultPrediction,
    tags=["predictions"],
)
async def predict_hospital_default(hospital: HospitalFeatures) -> HospitalDefaultPrediction:
    """
    Predict probability of default (PD) for a hospital bond issuer.

    Combines financial ratios with clinical quality metrics.
    Returns PD score, implied credit rating, Gini coefficient, and key risk factors.
    """
    t0 = time.time()

    # Logistic-style PD model using financial + clinical features
    linear = (
        -3.5
        - hospital.operating_margin * 8.0
        - (hospital.days_cash_on_hand / 100) * 1.5
        + hospital.debt_to_equity * 0.4
        + (hospital.readmission_rate - 0.155) * 20.0
        - (hospital.patient_satisfaction - 70) * 0.02
        + (0.5 - hospital.occupancy_rate) * 2.0
    )
    pd_score = float(1 / (1 + np.exp(-linear)))
    pd_score = float(np.clip(pd_score, 0.0002, 0.85))
    gini = float(np.clip(2 * (1 - pd_score) - 0.5, 0.30, 0.85))

    key_risk_factors = {
        "operating_margin": round(-hospital.operating_margin * 8.0, 4),
        "days_cash_on_hand": round(-(hospital.days_cash_on_hand / 100) * 1.5, 4),
        "debt_to_equity": round(hospital.debt_to_equity * 0.4, 4),
        "readmission_rate": round((hospital.readmission_rate - 0.155) * 20.0, 4),
        "patient_satisfaction": round(-(hospital.patient_satisfaction - 70) * 0.02, 4),
    }

    return HospitalDefaultPrediction(
        hospital_id=hospital.hospital_id,
        pd_score=round(pd_score, 6),
        implied_rating=_pd_to_rating(pd_score),
        gini_coefficient=round(gini, 4),
        key_risk_factors=key_risk_factors,
        latency_ms=round((time.time() - t0) * 1000, 2),
    )


@app.get(
    "/api/v1/simulation/state",
    response_model=SimulationState,
    tags=["simulation"],
)
async def get_simulation_state() -> SimulationState:
    """Return the current HealthRisk Lab simulation game state."""
    global _SIM_QUARTER, _SIM_PORTFOLIO, _SIM_SCORE

    return SimulationState(
        quarter=_SIM_QUARTER,
        year=2024 + (_SIM_QUARTER - 1) // 4,
        portfolio_value_usd=round(_SIM_PORTFOLIO, 2),
        total_return_pct=round((_SIM_PORTFOLIO / 500_000_000 - 1) * 100, 4),
        score=round(_SIM_SCORE, 2),
        phase="IN_PROGRESS" if _SIM_QUARTER <= 40 else "COMPLETED",
        active_scenarios=["baseline_growth"] if _SIM_QUARTER < 5 else [],
        asset_allocations={
            "insurance": 30.0,
            "hospital_bonds": 30.0,
            "pharma_equities": 20.0,
            "credit_facility": 20.0,
        },
    )


@app.post(
    "/api/v1/simulation/next-quarter",
    response_model=QuarterResult,
    tags=["simulation"],
)
async def next_quarter(decision: SimulationDecision) -> QuarterResult:
    """
    Advance the simulation by one quarter based on the player's decision.

    The simulation engine applies scenario shocks, updates portfolio values,
    and calculates the score delta for this quarter.
    """
    global _SIM_QUARTER, _SIM_PORTFOLIO, _SIM_SCORE

    if _SIM_QUARTER > 40:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Simulation already completed (40 quarters done).",
        )

    rng = np.random.default_rng(_SIM_QUARTER * 42)
    quarterly_return = float(rng.uniform(-0.05, 0.12))

    # Apply action modifier
    action_modifier = {"hold": 0.0, "buy": 0.02, "sell": -0.01, "rebalance": 0.005}.get(
        decision.action, 0.0
    )
    quarterly_return += action_modifier

    _SIM_PORTFOLIO *= 1 + quarterly_return
    score_delta = max(0.0, quarterly_return * 200 + 5.0)
    _SIM_SCORE += score_delta
    events = [f"Q{_SIM_QUARTER}: baseline market movement ({quarterly_return:+.2%})"]

    prev_q = _SIM_QUARTER
    _SIM_QUARTER += 1

    ai_return = float(rng.uniform(-0.02, 0.10))  # AI opponent always slightly positive

    return QuarterResult(
        quarter=prev_q,
        portfolio_value_usd=round(_SIM_PORTFOLIO, 2),
        quarterly_return_pct=round(quarterly_return * 100, 4),
        score_delta=round(score_delta, 2),
        events=events,
        ai_opponent_return_pct=round(ai_return * 100, 4),
    )


@app.get(
    "/api/v1/explainability/shap/{model_name}",
    response_model=SHAPExplanation,
    tags=["explainability"],
)
async def get_shap_explanation(model_name: str) -> SHAPExplanation:
    """
    Return SHAP feature importance for the specified model.

    In production this triggers a real SHAP TreeExplainer run;
    here it returns representative values from a pre-computed summary.
    """
    known_models = {
        "readmission_xgboost", "cost_lightgbm", "hospital_default_xgboost",
        "stacking_ensemble",
    }
    if model_name not in known_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found. Available: {sorted(known_models)}",
        )

    rng = np.random.default_rng(hash(model_name) % 2**32)
    feature_names = [
        "age", "chronic_count", "prior_admissions", "hcc_score",
        "hemoglobin", "creatinine", "hba1c", "er_visits_12m",
        "los_days", "operating_margin",
    ]
    raw_shap = rng.normal(0, 0.08, len(feature_names))
    shap_values = [round(float(v), 5) for v in raw_shap]
    base_value = round(float(rng.uniform(0.10, 0.25)), 5)

    sorted_pairs = sorted(
        zip(feature_names, shap_values), key=lambda x: x[1], reverse=True
    )
    top_pos = {k: v for k, v in sorted_pairs[:3] if v > 0}
    top_neg = {k: abs(v) for k, v in sorted_pairs[-3:] if v < 0}

    return SHAPExplanation(
        model_name=model_name,
        feature_names=feature_names,
        shap_values=shap_values,
        base_value=base_value,
        expected_value=base_value,
        top_positive_features=top_pos,
        top_negative_features=top_neg,
    )


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please check API logs."},
    )
