"""
data.acquisition — Data ingestion modules for all external health data sources.

Modules
-------
mimic_loader    - PhysioNet MIMIC-IV downloader (requires PhysioNet credentials)
who_gho         - WHO Global Health Observatory API client
clinicaltrials  - ClinicalTrials.gov API v2 client
fda_faers       - openFDA FAERS adverse event fetcher
cdc_wonder      - CDC WONDER / Socrata mortality & surveillance data fetcher
cms_scraper     - CMS Hospital Compare & cost report fetcher
sec_edgar       - SEC EDGAR pharma filing scraper
pipeline        - Master data acquisition orchestrator
"""

from data.acquisition.cdc_wonder import CDCWonderClient
from data.acquisition.cms_scraper import CMSDataClient
from data.acquisition.fda_faers import FDAFAERSClient
from data.acquisition.pipeline import DataPipeline
from data.acquisition.sec_edgar import SECEdgarClient
from data.acquisition.who_gho import WHOGHOClient

__all__ = [
    "CDCWonderClient",
    "CMSDataClient",
    "FDAFAERSClient",
    "DataPipeline",
    "SECEdgarClient",
    "WHOGHOClient",
]
