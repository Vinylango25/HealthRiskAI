"""
data.processing — Data cleaning, normalisation, and cohort-building pipelines.

Modules
-------
cleaner      - Missing value imputation, outlier clipping, dtype casting
normaliser   - Lab value standardisation and ICD code normalisation
cohort       - Patient cohort extraction, filtering, and survival prep
splitter     - Train / validation / test splitting (random, temporal, stratified, group)
validator    - Schema validation for patient and hospital DataFrames

Quick start
-----------
>>> from data.processing import DataCleaner, LabNormaliser, CohortBuilder
>>> cleaner = DataCleaner()
>>> df_clean = cleaner.clean_patient_df(raw_df)
"""

from data.processing.cleaner import DataCleaner
from data.processing.cohort import CohortBuilder
from data.processing.normaliser import ICDNormaliser, LabNormaliser
from data.processing.splitter import DataSplitter
from data.processing.validator import SchemaValidator, ValidationResult

__all__ = [
    "DataCleaner",
    "LabNormaliser",
    "ICDNormaliser",
    "CohortBuilder",
    "DataSplitter",
    "SchemaValidator",
    "ValidationResult",
]
