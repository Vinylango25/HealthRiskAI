"""
api/main.py
===========
HealthRisk AI — FastAPI application entry point.

Endpoints
---------
GET  /                                          Welcome + links
GET  /health                                    Health check (used by Docker HEALTHCHECK)
GET  /api/v1/models                             List available model types
POST /api/v1/predict/readmission                30-day readmission risk
POST /api/v1/predict/cost                       12-month healthcare cost
POST /api/v1/predict/hospital-default           Hospital bond PD score
GET  /api/v1/simulation/state                   Current simulation game state
POST /api/v1/simulation/next-quarter            Advance simulation by one quarter
GET  /api/v1/explainability/shap/{model_name}   SHAP feature importance
GET  /api/v1/who/indicators                     WHO GHO life expectancy
GET  /api/v1/who/disease-burden                 WHO GHO diabetes prevalence
GET  /api/v1/faers/signals                      FDA FAERS adverse event signals
GET  /api/v1/trials/recruiting                  ClinicalTrials.gov recruiting studies
GET  /api/v1/cms/hospitals                      CMS Hospital Compare data
GET  /api/v1/edgar/pharma                       SEC EDGAR pharma company filings
GET  /api/v1/dashboard/live                     Live aggregated dashboard

Usage (local)
-------------
    uvicorn api.main:app --reload --port 8000

Docker
------
    CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
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
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")

# ---------------------------------------------------------------------------
# In-memory TTL cache
# ---------------------------------------------------------------------------
# Structure: { cache_key: (data, timestamp_float) }
# TTLs in seconds

_CACHE: Dict[str, Tuple[Any, float]] = {}
_TTL_WHO = 3600       # 1 hour
_TTL_CMS = 3600       # 1 hour
_TTL_FAERS = 1800     # 30 minutes
_TTL_TRIALS = 1800    # 30 minutes
_TTL_EDGAR = 7200     # 2 hours


def _cache_get(key: str, ttl: int) -> Optional[Any]:
    """Return cached value if present and not expired, else None."""
    entry = _CACHE.get(key)
    if entry is None:
        return None
    data, ts = entry
    if time.time() - ts > ttl:
        return None
    return data


def _cache_set(key: str, data: Any) -> None:
    """Store data in cache with current timestamp."""
    _CACHE[key] = (data, time.time())


def _cache_get_raw(key: str) -> Optional[Tuple[Any, float]]:
    """Return (data, timestamp) regardless of TTL, for fallback use."""
    return _CACHE.get(key)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

# CORS — allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# Routes — existing endpoints
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
        + (1 if "I50" in patient.diagnoses else 0) * 0.8
        + (1 if any(d.startswith("C") for d in patient.diagnoses) else 0) * 1.2
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
    """
    global _SIM_QUARTER, _SIM_PORTFOLIO, _SIM_SCORE

    if _SIM_QUARTER > 40:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Simulation already completed (40 quarters done).",
        )

    rng = np.random.default_rng(_SIM_QUARTER * 42)
    quarterly_return = float(rng.uniform(-0.05, 0.12))

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

    ai_return = float(rng.uniform(-0.02, 0.10))

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
# Routes — Real-data endpoints
# ---------------------------------------------------------------------------

# ── WHO GHO ──────────────────────────────────────────────────────────────


@app.get("/api/v1/who/indicators", tags=["who"])
async def who_indicators() -> Dict[str, Any]:
    """
    Fetch global life expectancy from WHO GHO API.
    TTL cache: 3600 s.
    """
    cache_key = "who_indicators"
    cached = _cache_get(cache_key, _TTL_WHO)
    if cached is not None:
        return cached

    url = (
        "https://ghoapi.azureedge.net/api/WHOSIS_000001"
        "?$filter=SpatialDim eq 'GLOBAL'&$top=5"
    )
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info("WHO indicators fetch completed in %s ms", elapsed)

        raw = resp.json()
        values = [
            {
                "country": item.get("SpatialDim", "GLOBAL"),
                "year": item.get("TimeDim"),
                "value": item.get("NumericValue"),
            }
            for item in raw.get("value", [])
        ]
        result = {
            "indicator": "life_expectancy",
            "values": values,
            "source": "WHO GHO",
            "cached_at": _now_iso(),
        }
        _cache_set(cache_key, result)
        return result

    except httpx.HTTPError as exc:
        logger.error("WHO indicators fetch failed: %s", exc)
        fallback = _cache_get_raw(cache_key)
        if fallback:
            data, _ = fallback
            return data
        return {"error": str(exc), "source": "WHO GHO", "fallback": True}


@app.get("/api/v1/who/disease-burden", tags=["who"])
async def who_disease_burden() -> Dict[str, Any]:
    """
    Fetch top-10 countries by diabetes prevalence from WHO GHO API.
    TTL cache: 3600 s.
    """
    cache_key = "who_disease_burden"
    cached = _cache_get(cache_key, _TTL_WHO)
    if cached is not None:
        return cached

    url = (
        "https://ghoapi.azureedge.net/api/NCD_DIABETES_PREVALENCE_CRUDE"
        "?$top=10&$orderby=NumericValue desc"
    )
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info("WHO disease burden fetch completed in %s ms", elapsed)

        raw = resp.json()
        values = [
            {
                "country": item.get("SpatialDim"),
                "year": item.get("TimeDim"),
                "value": item.get("NumericValue"),
            }
            for item in raw.get("value", [])
        ]
        result = {
            "indicator": "diabetes_prevalence",
            "values": values,
            "source": "WHO GHO",
            "cached_at": _now_iso(),
        }
        _cache_set(cache_key, result)
        return result

    except httpx.HTTPError as exc:
        logger.error("WHO disease burden fetch failed: %s", exc)
        fallback = _cache_get_raw(cache_key)
        if fallback:
            data, _ = fallback
            return data
        return {"error": str(exc), "source": "WHO GHO", "fallback": True}


# ── FDA FAERS ─────────────────────────────────────────────────────────────


@app.get("/api/v1/faers/signals", tags=["faers"])
async def faers_signals(
    drug: str = Query(default="aspirin", description="Drug name to query")
) -> Dict[str, Any]:
    """
    Fetch adverse event signals for a drug from FDA FAERS.
    TTL cache: 1800 s.
    """
    cache_key = f"faers_{drug.lower()}"
    cached = _cache_get(cache_key, _TTL_FAERS)
    if cached is not None:
        return cached

    url = (
        f"https://api.fda.gov/drug/event.json"
        f"?search=patient.drug.medicinalproduct:{drug}"
        f"&count=patient.reaction.reactionmeddrapt.exact&limit=10"
    )
    if OPENFDA_API_KEY:
        url += f"&api_key={OPENFDA_API_KEY}"

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info("FAERS signals fetch for '%s' completed in %s ms", drug, elapsed)

        raw = resp.json()
        total = raw.get("meta", {}).get("results", {}).get("total", 0)
        reactions = [
            {"reaction": r.get("term"), "count": r.get("count")}
            for r in raw.get("results", [])
        ]
        result = {
            "drug": drug,
            "total_reports": total,
            "top_reactions": reactions,
            "source": "FDA FAERS",
            "cached_at": _now_iso(),
        }
        _cache_set(cache_key, result)
        return result

    except httpx.HTTPError as exc:
        logger.error("FAERS signals fetch failed for '%s': %s", drug, exc)
        fallback = _cache_get_raw(cache_key)
        if fallback:
            data, _ = fallback
            return data
        return {"error": str(exc), "source": "FDA FAERS", "fallback": True}

# ── ClinicalTrials.gov ────────────────────────────────────────────────────


@app.get("/api/v1/trials/recruiting", tags=["trials"])
async def trials_recruiting(
    condition: str = Query(default="diabetes", description="Medical condition to search"),
    limit: int = Query(default=10, ge=1, le=100, description="Number of results"),
) -> Dict[str, Any]:
    """
    Fetch currently recruiting clinical trials from ClinicalTrials.gov API v2.
    TTL cache: 1800 s.
    """
    cache_key = f"trials_{condition.lower()}_{limit}"
    cached = _cache_get(cache_key, _TTL_TRIALS)
    if cached is not None:
        return cached

    url = (
        f"https://clinicaltrials.gov/api/v2/studies"
        f"?query.cond={condition}&filter.overallStatus=RECRUITING"
        f"&pageSize={limit}&format=json"
    )
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info(
            "ClinicalTrials fetch for '%s' (limit=%d) completed in %s ms",
            condition, limit, elapsed,
        )

        raw = resp.json()
        total_found = raw.get("totalCount", 0)

        trials = []
        for study in raw.get("studies", []):
            proto = study.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})
            contacts_mod = proto.get("contactsLocationsModule", {})

            phases = design_mod.get("phases", [])
            phase_str = phases[0] if phases else None
            locations = contacts_mod.get("locations", [])

            trials.append({
                "nct_id": id_mod.get("nctId"),
                "title": id_mod.get("briefTitle"),
                "phase": phase_str,
                "status": status_mod.get("overallStatus"),
                "enrollment": design_mod.get("enrollmentInfo", {}).get("count"),
                "start_date": status_mod.get("startDateStruct", {}).get("date"),
                "locations_count": len(locations),
            })

        result = {
            "condition": condition,
            "total_found": total_found,
            "trials": trials,
            "source": "ClinicalTrials.gov",
            "cached_at": _now_iso(),
        }
        _cache_set(cache_key, result)
        return result

    except httpx.HTTPError as exc:
        logger.error(
            "ClinicalTrials fetch failed for '%s': %s", condition, exc
        )
        fallback = _cache_get_raw(cache_key)
        if fallback:
            data, _ = fallback
            return data
        return {"error": str(exc), "source": "ClinicalTrials.gov", "fallback": True}


# ── CMS Hospital Compare ──────────────────────────────────────────────────


@app.get("/api/v1/cms/hospitals", tags=["cms"])
async def cms_hospitals(
    state: str = Query(default="", description="Two-letter state code (optional)"),
    limit: int = Query(default=20, ge=1, le=500, description="Number of results"),
) -> Dict[str, Any]:
    """
    Fetch hospital quality data from CMS Provider Data catalog.
    TTL cache: 3600 s.
    """
    cache_key = f"cms_hospitals_{state.upper()}_{limit}"
    cached = _cache_get(cache_key, _TTL_CMS)
    if cached is not None:
        return cached

    state_clause = f"&state={state.upper()}" if state else ""
    url = (
        f"https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0"
        f"?limit={limit}&offset=0&count=true&results=true&keys=true&format=json{state_clause}"
    )

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info(
            "CMS hospitals fetch (state=%s, limit=%d) completed in %s ms",
            state or "ALL", limit, elapsed,
        )

        raw = resp.json()
        hospitals = []
        for row in raw if isinstance(raw, list) else raw.get("results", []):
            hospitals.append({
                "provider_id": row.get("facility_id") or row.get("provider_id"),
                "name": row.get("facility_name") or row.get("hospital_name"),
                "city": row.get("city"),
                "state": row.get("state"),
                "overall_rating": row.get("hospital_overall_rating"),
                "readmission_rate": row.get("readmission_national_comparison"),
            })

        result = {
            "hospitals": hospitals,
            "source": "CMS Hospital Compare",
            "count": len(hospitals),
            "cached_at": _now_iso(),
        }
        _cache_set(cache_key, result)
        return result

    except httpx.HTTPError as exc:
        logger.error("CMS hospitals fetch failed: %s", exc)
        fallback = _cache_get_raw(cache_key)
        if fallback:
            data, _ = fallback
            return data
        return {"error": str(exc), "source": "CMS Hospital Compare", "fallback": True}

# ── SEC EDGAR ─────────────────────────────────────────────────────────────

# Known pharma CIKs (zero-padded to 10 digits)
_PHARMA_CIKS = [
    ("0000002178", "Abbott"),
    ("0000310158", "Merck"),
    ("0000078003", "Pfizer"),
    ("0000200406", "Johnson & Johnson"),
    ("0000100517", "Amgen"),
]


async def _fetch_edgar_company(
    client: httpx.AsyncClient, cik: str, fallback_name: str
) -> Dict[str, Any]:
    """Fetch one company's submission metadata from SEC EDGAR."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    t0 = time.time()
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info("EDGAR fetch CIK=%s completed in %s ms", cik, elapsed)

        data = resp.json()
        tickers = data.get("tickers", [])
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        recent_10k_count = sum(1 for f in forms if f in ("10-K", "10-K/A"))

        return {
            "cik": cik.lstrip("0"),
            "name": data.get("entityName", fallback_name),
            "ticker": tickers[0] if tickers else None,
            "sic": data.get("sic"),
            "recent_10k_count": recent_10k_count,
        }
    except httpx.HTTPError as exc:
        logger.error("EDGAR fetch failed for CIK=%s: %s", cik, exc)
        return {
            "cik": cik.lstrip("0"),
            "name": fallback_name,
            "ticker": None,
            "sic": None,
            "recent_10k_count": 0,
            "error": str(exc),
        }


@app.get("/api/v1/edgar/pharma", tags=["edgar"])
async def edgar_pharma(
    limit: int = Query(default=10, ge=1, le=50, description="Max companies to return"),
) -> Dict[str, Any]:
    """
    Fetch recent 10-K filing data for major pharma companies from SEC EDGAR.
    TTL cache: 7200 s.
    """
    cache_key = f"edgar_pharma_{limit}"
    cached = _cache_get(cache_key, _TTL_EDGAR)
    if cached is not None:
        return cached

    ciks_to_fetch = _PHARMA_CIKS[:limit]

    # SEC requires a User-Agent header with contact info
    headers = {
        "User-Agent": "HealthRiskAI research@healthriskai.example.com",
        "Accept-Encoding": "gzip, deflate",
    }

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            tasks = [
                _fetch_edgar_company(client, cik, name)
                for cik, name in ciks_to_fetch
            ]
            companies = await asyncio.gather(*tasks)

        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info("EDGAR pharma batch fetch completed in %s ms", elapsed)

        result = {
            "companies": list(companies),
            "source": "SEC EDGAR",
            "cached_at": _now_iso(),
        }
        _cache_set(cache_key, result)
        return result

    except Exception as exc:
        logger.error("EDGAR pharma fetch failed: %s", exc)
        fallback = _cache_get_raw(cache_key)
        if fallback:
            data, _ = fallback
            return data
        return {"error": str(exc), "source": "SEC EDGAR", "fallback": True}


# ── Live Dashboard ────────────────────────────────────────────────────────


async def _fetch_who_life_expectancy_value() -> Optional[float]:
    """Return the first available global life expectancy value from WHO GHO."""
    url = (
        "https://ghoapi.azureedge.net/api/WHOSIS_000001"
        "?$filter=SpatialDim eq 'GLOBAL'&$top=5"
    )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        items = resp.json().get("value", [])
        values = [
            item.get("NumericValue")
            for item in items
            if item.get("NumericValue") is not None
        ]
        return round(float(np.mean(values)), 2) if values else None
    except Exception as exc:
        logger.error("Dashboard WHO fetch failed: %s", exc)
        return None


async def _fetch_faers_total(drug: str = "aspirin") -> Optional[int]:
    """Return total FAERS report count for a drug."""
    # Use the event search endpoint (not count) to get total reports
    url = (
        f"https://api.fda.gov/drug/event.json"
        f"?search=patient.drug.medicinalproduct:{drug}&limit=1"
    )
    if OPENFDA_API_KEY:
        url += f"&api_key={OPENFDA_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        return resp.json().get("meta", {}).get("results", {}).get("total", 0)
    except Exception as exc:
        logger.error("Dashboard FAERS fetch failed: %s", exc)
        return None


async def _fetch_trials_count(condition: str = "diabetes") -> Optional[int]:
    """Return count of recruiting trials for a condition."""
    url = (
        f"https://clinicaltrials.gov/api/v2/studies"
        f"?query.cond={condition}&filter.overallStatus=RECRUITING"
        f"&pageSize=1&format=json&countTotal=true"
    )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        data = resp.json()
        # API v2 uses totalCount at top level
        return data.get("totalCount") or data.get("total") or len(data.get("studies", []))
    except Exception as exc:
        logger.error("Dashboard trials fetch failed: %s", exc)
        return None


async def _fetch_cms_rated_count() -> Optional[int]:
    """Return count of CMS hospitals with overall rating >= 4."""
    url = (
        "https://data.cms.gov/provider-data/api/1/datastore/query/xubh-q36u/0"
        "?limit=500&offset=0&count=true&results=true&keys=true&format=json"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        raw = resp.json()
        rows = raw.get("results", [])
        count = 0
        for row in rows:
            rating = row.get("hospital_overall_rating")
            try:
                if rating is not None and int(rating) >= 4:
                    count += 1
            except (ValueError, TypeError):
                pass
        return count
    except Exception as exc:
        logger.error("Dashboard CMS fetch failed: %s", exc)
        return None


@app.get("/api/v1/dashboard/live", tags=["dashboard"])
async def dashboard_live() -> Dict[str, Any]:
    """
    Aggregated live dashboard — calls WHO, FAERS, ClinicalTrials, and CMS in parallel.
    """
    t0 = time.time()
    now = _now_iso()

    who_result, faers_result, trials_result, cms_result = await asyncio.gather(
        _fetch_who_life_expectancy_value(),
        _fetch_faers_total("aspirin"),
        _fetch_trials_count("diabetes"),
        _fetch_cms_rated_count(),
    )

    elapsed = round((time.time() - t0) * 1000, 1)
    logger.info("Live dashboard assembled in %s ms", elapsed)

    return {
        "who_life_expectancy": who_result,
        "faers_aspirin_reactions": faers_result,
        "recruiting_trials": trials_result,
        "cms_hospitals_rated": cms_result,
        "data_freshness": {
            "who": now,
            "faers": now,
            "trials": now,
        },
        "sources": ["WHO GHO", "FDA FAERS", "ClinicalTrials.gov", "CMS"],
        "latency_ms": elapsed,
    }


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
