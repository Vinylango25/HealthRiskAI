"""
HealthRisk AI — pytest configuration and shared fixtures.

All fixtures use scope="session" to avoid expensive re-computation across
the test suite.  Synthetic data is generated with Faker + numpy.random so
tests never depend on external services.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Seed everything for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Lazy Faker import (optional dev dependency)
# ---------------------------------------------------------------------------
try:
    from faker import Faker

    _faker = Faker()
    Faker.seed(RANDOM_SEED)
except ImportError:  # pragma: no cover
    _faker = None  # type: ignore[assignment]


def _random_date(start: date, end: date) -> date:
    """Return a random date between *start* and *end* (inclusive)."""
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta + 1)))


# ---------------------------------------------------------------------------
# ICD-10 codes and lab names used across fixtures
# ---------------------------------------------------------------------------
ICD10_CODES = [
    "I10",    # Essential (primary) hypertension
    "E11.9",  # Type 2 diabetes without complications
    "J44.1",  # COPD with acute exacerbation
    "I25.10", # Atherosclerotic heart disease
    "N18.3",  # Chronic kidney disease stage 3
    "F32.1",  # Major depressive disorder, moderate
    "C34.10", # Lung cancer, unspecified
    "I50.9",  # Heart failure, unspecified
    "G20",    # Parkinson disease
    "M79.3",  # Panniculitis (unspecified)
]

LAB_NAMES = [
    "hemoglobin",
    "creatinine",
    "bun",
    "sodium",
    "potassium",
    "glucose",
    "hba1c",
    "ldl_cholesterol",
    "hdl_cholesterol",
    "troponin",
]

GENDERS = ["M", "F", "O"]
PHASES = ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]
TRIAL_STATUSES = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED"]
DRUG_CLASSES = [
    "Antihypertensive",
    "Antidiabetic",
    "Anticoagulant",
    "Statin",
    "Beta-blocker",
    "ACE inhibitor",
    "SSRI",
    "NSAID",
    "Immunosuppressant",
    "Chemotherapy",
]


# ===========================================================================
# Fixture: sample_patient_df
# ===========================================================================
@pytest.fixture(scope="session")
def sample_patient_df() -> pd.DataFrame:
    """
    100 synthetic inpatient records mimicking MIMIC-IV structure.

    Columns
    -------
    subject_id          : int      unique patient identifier
    age                 : int      18–90
    gender              : str      M / F / O
    admission_date      : date
    discharge_date      : date
    los_days            : int      length of stay
    diagnoses           : list[str] list of ICD-10 codes (1–5 per patient)
    primary_diagnosis   : str      first ICD-10 code
    hemoglobin          : float
    creatinine          : float
    bun                 : float
    sodium              : float
    potassium           : float
    glucose             : float
    hba1c               : float
    ldl_cholesterol     : float
    hdl_cholesterol     : float
    troponin            : float
    total_cost          : float    USD
    insurance_paid      : float    USD
    mortality_30d       : int      0/1
    readmission_30d     : int      0/1
    icu_admission       : int      0/1
    """
    n = 100

    ages = rng.integers(18, 91, size=n).astype(int)
    genders = rng.choice(GENDERS, size=n, p=[0.49, 0.49, 0.02])
    admit_dates = [_random_date(date(2018, 1, 1), date(2023, 12, 31)) for _ in range(n)]
    los = rng.integers(1, 30, size=n).astype(int)
    discharge_dates = [admit_dates[i] + timedelta(days=int(los[i])) for i in range(n)]

    # Each patient gets 1–5 random diagnoses
    all_diagnoses = [
        list(rng.choice(ICD10_CODES, size=int(rng.integers(1, 6)), replace=False))
        for _ in range(n)
    ]
    primary_diagnoses = [d[0] for d in all_diagnoses]

    # Lab values (rough physiological ranges)
    hgb = rng.normal(13.5, 2.0, n).clip(6, 20)
    creatinine = rng.lognormal(0.1, 0.4, n).clip(0.4, 12)
    bun = rng.normal(18, 8, n).clip(5, 100)
    sodium = rng.normal(139, 3, n).clip(120, 160)
    potassium = rng.normal(4.2, 0.5, n).clip(2.5, 7.0)
    glucose = rng.lognormal(4.8, 0.3, n).clip(60, 600)
    hba1c = rng.normal(6.5, 1.5, n).clip(4.0, 14.0)
    ldl = rng.normal(115, 30, n).clip(40, 250)
    hdl = rng.normal(50, 15, n).clip(20, 120)
    troponin = rng.lognormal(-3, 1.2, n).clip(0.001, 50)

    # Costs (right-skewed Tweedie-like)
    base_cost = rng.lognormal(9.5, 0.8, n).clip(1_000, 500_000)
    insurance_paid = base_cost * rng.uniform(0.70, 0.95, n)

    # Binary outcomes (correlated with age and los)
    mortality_prob = 1 / (1 + np.exp(-(ages / 90 * 2 - 1.5 + los / 30 * 1.5)))
    mortality = rng.binomial(1, mortality_prob.clip(0.01, 0.30)).astype(int)
    readmit = rng.binomial(1, np.full(n, 0.18)).astype(int)
    icu = rng.binomial(1, (mortality_prob * 1.5).clip(0.05, 0.40)).astype(int)

    df = pd.DataFrame(
        {
            "subject_id": range(10001, 10001 + n),
            "age": ages,
            "gender": genders,
            "admission_date": admit_dates,
            "discharge_date": discharge_dates,
            "los_days": los,
            "diagnoses": all_diagnoses,
            "primary_diagnosis": primary_diagnoses,
            "hemoglobin": hgb.round(1),
            "creatinine": creatinine.round(2),
            "bun": bun.round(1),
            "sodium": sodium.round(1),
            "potassium": potassium.round(2),
            "glucose": glucose.round(1),
            "hba1c": hba1c.round(1),
            "ldl_cholesterol": ldl.round(1),
            "hdl_cholesterol": hdl.round(1),
            "troponin": troponin.round(4),
            "total_cost": base_cost.round(2),
            "insurance_paid": insurance_paid.round(2),
            "mortality_30d": mortality,
            "readmission_30d": readmit,
            "icu_admission": icu,
        }
    )
    return df


# ===========================================================================
# Fixture: sample_hospital_df
# ===========================================================================
@pytest.fixture(scope="session")
def sample_hospital_df() -> pd.DataFrame:
    """
    20 synthetic hospital records combining CMS-style financial and
    clinical quality metrics.

    Columns
    -------
    hospital_id         : str
    hospital_name       : str
    state               : str
    bed_count           : int
    teaching_status     : str      community / teaching / academic
    annual_revenue_usd  : float
    operating_margin    : float    fraction
    debt_to_equity      : float
    days_cash_on_hand   : float
    occupancy_rate      : float    0–1
    readmission_rate    : float    0–1
    mortality_rate      : float    0–1
    infection_rate      : float    0–1
    patient_satisfaction: float    0–100
    bond_rating         : str
    credit_spread_bps   : float
    default_probability : float    0–1
    """
    n = 20
    states = [
        "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
        "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
    ]
    teaching = rng.choice(
        ["community", "teaching", "academic"], size=n, p=[0.55, 0.30, 0.15]
    )
    bed_count = rng.integers(50, 900, size=n).astype(int)
    revenue = (bed_count * rng.uniform(500_000, 1_500_000, size=n)).round(0)
    op_margin = rng.normal(0.03, 0.04, n).clip(-0.10, 0.12)
    dte = rng.lognormal(0.5, 0.5, n).clip(0.2, 5.0)
    dcoh = rng.normal(85, 30, n).clip(10, 250)
    occupancy = rng.uniform(0.55, 0.92, n)
    readmit = rng.normal(0.155, 0.025, n).clip(0.08, 0.25)
    mortality = rng.normal(0.012, 0.005, n).clip(0.003, 0.030)
    infection = rng.normal(0.008, 0.003, n).clip(0.001, 0.020)
    satisfaction = rng.normal(72, 10, n).clip(40, 95)

    bond_ratings = rng.choice(
        ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"],
        size=n,
        p=[0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05],
    )
    rating_spread = {
        "AAA": 20, "AA": 35, "A": 60, "BBB": 120, "BB": 250, "B": 400, "CCC": 700
    }
    credit_spreads = np.array(
        [rating_spread[r] + rng.integers(-15, 15) for r in bond_ratings], dtype=float
    )
    pd_map = {
        "AAA": 0.0002, "AA": 0.0005, "A": 0.001, "BBB": 0.005,
        "BB": 0.02, "B": 0.06, "CCC": 0.20
    }
    default_probs = np.array([pd_map[r] for r in bond_ratings], dtype=float)

    df = pd.DataFrame(
        {
            "hospital_id": [f"HOSP{i:03d}" for i in range(1, n + 1)],
            "hospital_name": [f"Regional Medical Center {i}" for i in range(1, n + 1)],
            "state": states,
            "bed_count": bed_count,
            "teaching_status": teaching,
            "annual_revenue_usd": revenue,
            "operating_margin": op_margin.round(4),
            "debt_to_equity": dte.round(3),
            "days_cash_on_hand": dcoh.round(1),
            "occupancy_rate": occupancy.round(3),
            "readmission_rate": readmit.round(4),
            "mortality_rate": mortality.round(5),
            "infection_rate": infection.round(5),
            "patient_satisfaction": satisfaction.round(1),
            "bond_rating": bond_ratings,
            "credit_spread_bps": credit_spreads.round(1),
            "default_probability": default_probs,
        }
    )
    return df


# ===========================================================================
# Fixture: sample_trial_df
# ===========================================================================
@pytest.fixture(scope="session")
def sample_trial_df() -> pd.DataFrame:
    """
    50 synthetic clinical trial records mimicking ClinicalTrials.gov format.

    Columns
    -------
    nct_id              : str
    brief_title         : str
    condition           : str
    phase               : str
    enrollment_count    : int
    start_date          : date
    completion_date     : date
    status              : str
    primary_endpoint_met: int      0/1  (completed trials only)
    sponsor_class       : str
    drug_class          : str
    duration_months     : float
    """
    n = 50
    conditions = [
        "Cardiovascular Disease", "Type 2 Diabetes", "Non-Small Cell Lung Cancer",
        "Chronic Kidney Disease", "Heart Failure", "COPD", "Hypertension",
        "Atrial Fibrillation", "Alzheimer Disease", "Breast Cancer",
    ]
    sponsor_classes = ["INDUSTRY", "NIH", "OTHER_GOV", "INDIVIDUAL", "NETWORK"]

    phases = rng.choice(PHASES, size=n, p=[0.10, 0.25, 0.40, 0.25])
    statuses = rng.choice(TRIAL_STATUSES, size=n, p=[0.20, 0.15, 0.55, 0.10])
    enrollment = rng.integers(50, 5000, size=n).astype(int)
    start_dates = [_random_date(date(2015, 1, 1), date(2022, 12, 31)) for _ in range(n)]
    duration_months = rng.uniform(12, 60, n)
    completion_dates = [
        start_dates[i] + timedelta(days=int(duration_months[i] * 30.4))
        for i in range(n)
    ]

    endpoint_met = np.where(
        statuses == "COMPLETED",
        rng.binomial(1, 0.55, n),
        -1,  # -1 means not applicable
    )

    df = pd.DataFrame(
        {
            "nct_id": [f"NCT{rng.integers(10000000, 99999999)}" for _ in range(n)],
            "brief_title": [
                f"Phase {phases[i].replace('PHASE', '')} Study of "
                f"{rng.choice(DRUG_CLASSES)} in {rng.choice(conditions)}"
                for i in range(n)
            ],
            "condition": rng.choice(conditions, size=n),
            "phase": phases,
            "enrollment_count": enrollment,
            "start_date": start_dates,
            "completion_date": completion_dates,
            "status": statuses,
            "primary_endpoint_met": endpoint_met,
            "sponsor_class": rng.choice(sponsor_classes, size=n, p=[0.55, 0.20, 0.10, 0.05, 0.10]),
            "drug_class": rng.choice(DRUG_CLASSES, size=n),
            "duration_months": duration_months.round(1),
        }
    )
    return df


# ===========================================================================
# Fixture: sample_faers_df
# ===========================================================================
@pytest.fixture(scope="session")
def sample_faers_df() -> pd.DataFrame:
    """
    200 synthetic FDA FAERS adverse event records.

    Columns
    -------
    report_id           : str
    receive_date        : date
    patient_age         : float
    patient_sex         : str      1=Male, 2=Female
    drug_name           : str
    drug_class          : str
    reaction            : str
    outcome             : int      1=Death, 2=Life-threatening, 3=Hospitalisation, 6=Other
    serious             : int      0/1
    reporter_type       : str
    country             : str
    """
    n = 200
    reactions = [
        "Nausea", "Vomiting", "Diarrhoea", "Headache", "Dizziness",
        "Rash", "Anaphylaxis", "Hepatotoxicity", "Renal failure",
        "Cardiac arrest", "Stroke", "Hypotension", "QT prolongation",
        "Agranulocytosis", "Stevens-Johnson syndrome",
    ]
    reporter_types = ["physician", "pharmacist", "consumer", "other_health_professional"]
    countries = ["US", "DE", "GB", "FR", "JP", "CA", "AU", "IT", "BR", "IN"]
    drug_names = [
        "Atorvastatin", "Metformin", "Lisinopril", "Amlodipine", "Warfarin",
        "Rivaroxaban", "Metoprolol", "Omeprazole", "Levothyroxine", "Albuterol",
        "Pembrolizumab", "Adalimumab", "Infliximab", "Trastuzumab", "Rituximab",
    ]
    outcome_probs = [0.05, 0.10, 0.45, 0.40]  # death / life-threat / hosp / other

    receive_dates = [_random_date(date(2018, 1, 1), date(2023, 12, 31)) for _ in range(n)]
    ages = rng.normal(58, 16, n).clip(18, 95)
    sex = rng.choice([1, 2], size=n)
    drugs = rng.choice(drug_names, size=n)
    drug_classes_arr = rng.choice(DRUG_CLASSES, size=n)
    rxns = rng.choice(reactions, size=n)
    outcomes = rng.choice([1, 2, 3, 6], size=n, p=outcome_probs)
    serious = (outcomes <= 3).astype(int)

    df = pd.DataFrame(
        {
            "report_id": [f"FAERS{i:07d}" for i in range(1, n + 1)],
            "receive_date": receive_dates,
            "patient_age": ages.round(0).astype(int),
            "patient_sex": sex,
            "drug_name": drugs,
            "drug_class": drug_classes_arr,
            "reaction": rxns,
            "outcome": outcomes,
            "serious": serious,
            "reporter_type": rng.choice(reporter_types, size=n),
            "country": rng.choice(countries, size=n, p=[0.50, 0.08, 0.08, 0.07, 0.05,
                                                          0.05, 0.04, 0.04, 0.05, 0.04]),
        }
    )
    return df


# ===========================================================================
# Fixture: mock_api_responses
# ===========================================================================
@pytest.fixture(scope="session")
def mock_api_responses() -> dict[str, Any]:
    """
    Dictionary of mock API responses for each external data source.
    Suitable for use with pytest-mock or unittest.mock.patch.
    """
    return {
        "who_gho": {
            "status_code": 200,
            "json": {
                "@odata.context": "https://ghoapi.azureedge.net/api/$metadata#Indicator",
                "value": [
                    {
                        "IndicatorCode": "WHOSIS_000001",
                        "SpatialDim": "USA",
                        "TimeDim": 2022,
                        "NumericValue": 76.4,
                        "Low": 75.8,
                        "High": 77.1,
                    },
                    {
                        "IndicatorCode": "NCD_BMI_30A",
                        "SpatialDim": "GBR",
                        "TimeDim": 2022,
                        "NumericValue": 27.8,
                        "Low": 25.1,
                        "High": 30.5,
                    },
                ],
            },
        },
        "clinicaltrials": {
            "status_code": 200,
            "json": {
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {
                                "nctId": "NCT04567890",
                                "briefTitle": "Phase 3 Cardio Trial",
                            },
                            "statusModule": {"overallStatus": "RECRUITING"},
                            "designModule": {"phases": ["PHASE3"]},
                            "eligibilityModule": {"maximumAge": "75 Years"},
                        }
                    }
                ],
                "nextPageToken": None,
                "totalCount": 1,
            },
        },
        "fda_faers": {
            "status_code": 200,
            "json": {
                "meta": {"results": {"total": 150000, "skip": 0, "limit": 10}},
                "results": [
                    {
                        "safetyreportid": "12345678",
                        "receivedate": "20231015",
                        "serious": 1,
                        "patient": {
                            "patientonsetage": "65",
                            "patientsex": "1",
                            "drug": [
                                {
                                    "medicinalproduct": "ATORVASTATIN",
                                    "drugindication": "HYPERCHOLESTEROLAEMIA",
                                }
                            ],
                            "reaction": [{"reactionmeddrapt": "Rhabdomyolysis"}],
                        },
                    }
                ],
            },
        },
        "cms_hospital": {
            "status_code": 200,
            "json": {
                "results": [
                    {
                        "facility_id": "010001",
                        "facility_name": "EXAMPLE MEDICAL CENTER",
                        "state": "AL",
                        "mortality_group_measure_count": "7",
                        "count_of_facility_mort_measures_better": "1",
                        "count_of_facility_mort_measures_no_different": "5",
                        "count_of_facility_mort_measures_worse": "1",
                    }
                ]
            },
        },
        "sec_edgar": {
            "status_code": 200,
            "json": {
                "cik": "0000078003",
                "entityType": "operating",
                "sic": "2836",
                "name": "PFIZER INC",
                "facts": {
                    "us-gaap": {
                        "Revenues": {
                            "units": {
                                "USD": [
                                    {
                                        "end": "2022-12-31",
                                        "val": 100330000000,
                                        "form": "10-K",
                                        "filed": "2023-02-23",
                                        "accn": "0000078003-23-000004",
                                    }
                                ]
                            }
                        }
                    }
                },
            },
        },
        "physionet_mimic": {
            "status_code": 200,
            "json": {
                "message": "Authentication successful",
                "token": "mock_token_abc123",
            },
        },
    }


# ===========================================================================
# Fixture: temp_config_dir (provides path to configs/ for config-loading tests)
# ===========================================================================
@pytest.fixture(scope="session")
def project_root() -> str:
    """Return the project root directory path."""
    import os

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def config_dir(project_root: str) -> str:
    """Return the configs/ directory path."""
    import os

    return os.path.join(project_root, "configs")
