"""
Financial feature engineering for HealthRiskAI.
Covers hospital financial ratios, clinical quality metrics, credit scoring,
IBNR actuarial estimation, pharma pipeline signals, and ESG scoring.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class FinancialFeatureEngineer:
    """
    Computes financial and operational risk features for healthcare entities.
    All methods return new DataFrames; inputs are never mutated.
    """

    # ------------------------------------------------------------------
    # Hospital Financial Ratios
    # ------------------------------------------------------------------

    def compute_hospital_financial_ratios(self, hospital_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute standardised financial ratios for hospital providers.

        Parameters
        ----------
        hospital_df : DataFrame with columns:
            provider_id, net_patient_revenue, total_expenses,
            operating_expenses, operating_income, total_beds,
            total_discharges, unrestricted_cash, total_debt,
            net_assets, annual_debt_service, medicare_revenue,
            medicaid_revenue, commercial_revenue, selfpay_revenue

        Returns
        -------
        DataFrame with original columns plus computed ratio columns.
        """
        logger.info(
            f"compute_hospital_financial_ratios: {len(hospital_df):,} providers"
        )

        df = hospital_df.copy()

        REQUIRED = [
            "provider_id", "net_patient_revenue", "total_expenses",
            "operating_expenses", "operating_income", "total_beds",
            "total_discharges", "unrestricted_cash", "total_debt",
            "net_assets", "annual_debt_service", "medicare_revenue",
            "medicaid_revenue", "commercial_revenue", "selfpay_revenue",
        ]
        for col in REQUIRED:
            if col not in df.columns:
                logger.warning(f"  Missing column '{col}' – filling with NaN")
                df[col] = np.nan

        # Coerce all numeric columns
        for col in REQUIRED[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Operating margin ---
        df["operating_margin"] = (
            df["operating_income"] / df["net_patient_revenue"]
        ).clip(-1.0, 1.0)

        # --- Debt Service Coverage Ratio ---
        df["dscr"] = np.where(
            df["annual_debt_service"].fillna(0) == 0,
            0.0,
            df["operating_income"] / df["annual_debt_service"],
        )

        # --- Days Cash on Hand ---
        daily_opex = df["operating_expenses"] / 365.0
        df["days_cash_on_hand"] = df["unrestricted_cash"] / daily_opex.replace(0, np.nan)

        # --- Debt to Capitalisation ---
        total_cap = df["total_debt"] + df["net_assets"]
        df["debt_to_capitalization"] = (
            df["total_debt"] / total_cap.replace(0, np.nan)
        ).clip(0.0, 1.0)

        # --- Revenue per Adjusted Discharge ---
        df["revenue_per_adjusted_discharge"] = (
            df["net_patient_revenue"] / df["total_discharges"].replace(0, np.nan)
        )

        # --- Payer Mix ---
        npr = df["net_patient_revenue"].replace(0, np.nan)
        df["payer_mix_medicare_pct"]    = df["medicare_revenue"]   / npr
        df["payer_mix_medicaid_pct"]    = df["medicaid_revenue"]   / npr
        df["payer_mix_commercial_pct"]  = df["commercial_revenue"] / npr
        df["payer_mix_selfpay_pct"]     = df["selfpay_revenue"]    / npr
        df["government_payer_concentration"] = (
            df["payer_mix_medicare_pct"] + df["payer_mix_medicaid_pct"]
        )

        # --- Financial Stress Flag ---
        df["financial_stress_flag"] = (
            (df["operating_margin"] < 0)
            | (df["days_cash_on_hand"] < 30)
            | (df["dscr"] < 1.0)
        ).astype(int)

        logger.success(
            f"compute_hospital_financial_ratios: completed for {len(df):,} providers; "
            f"{df['financial_stress_flag'].sum()} stressed"
        )
        return df

    # ------------------------------------------------------------------
    # Clinical Quality Features
    # ------------------------------------------------------------------

    def compute_clinical_quality_features(self, quality_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive quality risk metrics from hospital quality reporting data.

        Parameters
        ----------
        quality_df : DataFrame with columns:
            provider_id, readmission_rate, hcahps_star_rating, hai_sir,
            psi_composite_score, ed_boarding_hours_avg,
            cmi_current, cmi_prior_year
            (plus net_patient_revenue for penalty calculation)

        Returns
        -------
        DataFrame with original columns plus quality risk features.
        """
        logger.info(
            f"compute_clinical_quality_features: {len(quality_df):,} providers"
        )

        QUALITY_COLS = [
            "provider_id", "readmission_rate", "hcahps_star_rating",
            "hai_sir", "psi_composite_score", "ed_boarding_hours_avg",
            "cmi_current", "cmi_prior_year",
        ]
        df = quality_df.copy()
        for col in QUALITY_COLS:
            if col not in df.columns:
                logger.warning(f"  Missing quality column '{col}' – filling with NaN")
                df[col] = np.nan

        for col in QUALITY_COLS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # --- Derived metrics ---
        df["readmission_rate_excess"]  = df["readmission_rate"] - 0.155
        df["hcahps_below_4_star"]      = (df["hcahps_star_rating"] < 4).astype(int)
        df["hai_elevated"]             = (df["hai_sir"] > 1.0).astype(int)
        df["psi_elevated"]             = (df["psi_composite_score"] > 1.0).astype(int)
        df["ed_boarding_critical"]     = (df["ed_boarding_hours_avg"] > 4.0).astype(int)
        df["cmi_trend"]                = df["cmi_current"] - df["cmi_prior_year"]

        # Penalty exposure: 3% of Medicare portion (48% of NPR)
        if "net_patient_revenue" not in df.columns:
            logger.warning("  'net_patient_revenue' not in quality_df – penalty set to NaN")
            df["net_patient_revenue"] = np.nan
        else:
            df["net_patient_revenue"] = pd.to_numeric(df["net_patient_revenue"], errors="coerce")

        df["readmission_penalty_exposure_usd"] = (
            df["readmission_rate_excess"].clip(lower=0)
            * 0.03
            * (df["net_patient_revenue"] * 0.48)
        )

        # Quality risk score (0–100 composite)
        readmit_component  = df["readmission_rate_excess"].fillna(0) * 30
        hcahps_component   = (1 - df["hcahps_star_rating"].fillna(3) / 5) * 25
        hai_component      = df["hai_sir"].fillna(1).clip(0, 3) * 20
        psi_component      = df["psi_composite_score"].fillna(1).clip(0, 3) * 15
        ed_component       = (df["ed_boarding_hours_avg"].fillna(0) / 10) * 10

        df["quality_risk_score"] = (
            readmit_component + hcahps_component + hai_component
            + psi_component + ed_component
        ).clip(0, 100)

        logger.success(
            f"compute_clinical_quality_features: mean quality risk score="
            f"{df['quality_risk_score'].mean():.1f}"
        )
        return df

    # ------------------------------------------------------------------
    # Hospital Credit Score
    # ------------------------------------------------------------------

    def compute_hospital_credit_score(
        self,
        financial_df: pd.DataFrame,
        quality_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute a composite credit score and rating tier for hospital providers.

        Parameters
        ----------
        financial_df : Output of compute_hospital_financial_ratios (must contain
                       operating_margin, dscr, days_cash_on_hand, provider_id).
        quality_df   : Optional output of compute_clinical_quality_features.

        Returns
        -------
        DataFrame: provider_id, credit_score, financial_score,
                   clinical_score, credit_tier, pd_1yr
        """
        logger.info(
            f"compute_hospital_credit_score: {len(financial_df):,} providers, "
            f"quality_df={'provided' if quality_df is not None else 'None'}"
        )

        fin = financial_df.copy()

        # Ensure required financial columns exist
        for col in ["operating_margin", "dscr", "days_cash_on_hand"]:
            if col not in fin.columns:
                logger.warning(f"  Financial column '{col}' missing – scoring as 0")
                fin[col] = np.nan

        # ---- Financial Score (60 pts) ----
        def _margin_score(m: float) -> int:
            if pd.isna(m) or m < 0:     return 0
            if m < 0.02:                return 10
            if m < 0.04:                return 16
            return 20

        def _dscr_score(d: float) -> int:
            if pd.isna(d) or d < 1.0:  return 0
            if d < 1.5:                return 8
            if d < 2.5:                return 14
            return 20

        def _cash_score(c: float) -> int:
            if pd.isna(c) or c < 30:   return 0
            if c < 60:                 return 8
            if c < 120:                return 14
            return 20

        fin["financial_score"] = (
            fin["operating_margin"].apply(_margin_score)
            + fin["dscr"].apply(_dscr_score)
            + fin["days_cash_on_hand"].apply(_cash_score)
        )

        # ---- Clinical Score (40 pts) ----
        if quality_df is not None:
            qual = quality_df.copy()
            for col in ["readmission_rate_excess", "hcahps_star_rating",
                        "cmi_trend", "hai_sir"]:
                if col not in qual.columns:
                    logger.warning(f"  Quality column '{col}' missing – scoring as 0")
                    qual[col] = np.nan

            def _readmit_score(excess: float) -> int:
                if pd.isna(excess) or excess < 0:    return 12
                if excess < 0.02:                    return 8
                if excess < 0.05:                    return 4
                return 0

            def _hcahps_score(stars: float) -> int:
                if pd.isna(stars):      return 0
                if stars >= 5:          return 12
                if stars >= 4:          return 9
                if stars >= 3:          return 5
                return 0

            def _cmi_score(trend: float) -> int:
                if pd.isna(trend):          return 5
                if trend > 0.02:            return 8
                if trend >= -0.02:          return 5
                if trend >= -0.1:           return 2
                return 0

            def _hai_score(sir: float) -> int:
                if pd.isna(sir) or sir < 0.7:  return 8
                if sir < 1.0:                  return 6
                if sir < 1.5:                  return 3
                return 0

            qual["clinical_score"] = (
                qual["readmission_rate_excess"].apply(_readmit_score)
                + qual["hcahps_star_rating"].apply(_hcahps_score)
                + qual["cmi_trend"].apply(_cmi_score)
                + qual["hai_sir"].apply(_hai_score)
            )
            merged = fin[["provider_id", "financial_score"]].merge(
                qual[["provider_id", "clinical_score"]],
                on="provider_id",
                how="left",
            )
            merged["clinical_score"] = merged["clinical_score"].fillna(0).astype(int)
        else:
            merged = fin[["provider_id", "financial_score"]].copy()
            merged["clinical_score"] = 0

        merged["credit_score"] = merged["financial_score"] + merged["clinical_score"]

        # Credit tier
        TIER_MAP: List[Tuple[int, str]] = [
            (90, "AAA"), (80, "AA"), (70, "A"), (60, "BBB"),
            (50, "BB"),  (40, "B"),  (0,  "CCC"),
        ]

        def _tier(score: int) -> str:
            for threshold, tier in TIER_MAP:
                if score >= threshold:
                    return tier
            return "CCC"

        PD_MAP: Dict[str, float] = {
            "AAA": 0.001, "AA": 0.003, "A": 0.007, "BBB": 0.018,
            "BB": 0.045,  "B": 0.095,  "CCC": 0.22,
        }

        merged["credit_tier"] = merged["credit_score"].apply(_tier)
        merged["pd_1yr"]      = merged["credit_tier"].map(PD_MAP)

        result = merged[["provider_id", "credit_score", "financial_score",
                          "clinical_score", "credit_tier", "pd_1yr"]]
        logger.success(
            f"compute_hospital_credit_score: tier distribution=\n"
            f"{result['credit_tier'].value_counts().to_dict()}"
        )
        return result

    # ------------------------------------------------------------------
    # IBNR Estimation
    # ------------------------------------------------------------------

    def estimate_ibnr(
        self,
        paid_claims_df: pd.DataFrame,
        eval_date: str,
        a_priori_loss_ratio: float = 0.85,
        tail_factor: float = 1.05,
        earned_premium_by_period: Optional[pd.Series] = None,
    ) -> dict:
        """
        Estimate Incurred But Not Reported (IBNR) reserves using
        Chain Ladder and Bornhuetter-Ferguson methods.

        Parameters
        ----------
        paid_claims_df : DataFrame with accident_period (YYYY-MM),
                         development_lag (int, months 1-24), paid_amount
        eval_date      : Evaluation date string 'YYYY-MM-DD'
        a_priori_loss_ratio : BF a priori expected loss ratio (default 0.85)
        tail_factor    : Tail development factor beyond last observed lag
        earned_premium_by_period : Optional Series indexed by accident_period

        Returns
        -------
        dict with keys: method, ibnr_total, ibnr_by_period, ultimate_claims,
                        development_factors, bf_ibnr_total, bf_by_period
        """
        logger.info(
            f"estimate_ibnr: {len(paid_claims_df):,} claim rows, eval_date={eval_date}"
        )

        df = paid_claims_df[["accident_period", "development_lag", "paid_amount"]].copy()
        df["paid_amount"]    = pd.to_numeric(df["paid_amount"],    errors="coerce").fillna(0)
        df["development_lag"]= pd.to_numeric(df["development_lag"],errors="coerce")

        # Build cumulative paid triangle
        # First aggregate incremental
        incremental = (
            df.groupby(["accident_period", "development_lag"])["paid_amount"]
            .sum()
            .unstack(fill_value=0)
        )
        incremental.sort_index(inplace=True)
        incremental.columns = sorted(incremental.columns)

        # Cumulate
        triangle = incremental.cumsum(axis=1)
        accident_periods = triangle.index.tolist()
        lags = sorted(triangle.columns.tolist())

        logger.debug(
            f"  Triangle shape: {triangle.shape}, "
            f"periods={accident_periods[:3]}...{accident_periods[-3:] if len(accident_periods) > 3 else ''}"
        )

        # ---- Chain Ladder age-to-age factors ----
        ata_factors: Dict[float, float] = {}
        for i in range(len(lags) - 1):
            lag_curr = lags[i]
            lag_next = lags[i + 1]
            # Only use rows where both columns have data
            mask = triangle[[lag_curr, lag_next]].notna().all(axis=1) & (triangle[lag_curr] > 0)
            if mask.sum() == 0:
                ata_factors[lag_curr] = 1.0
            else:
                ata_factors[lag_curr] = (
                    triangle.loc[mask, lag_next].sum()
                    / triangle.loc[mask, lag_curr].sum()
                )

        # Cumulative development factors (CDF) from each lag to ultimate
        max_lag = lags[-1]
        cdf: Dict[float, float] = {max_lag: tail_factor}
        for lag in reversed(lags[:-1]):
            next_lag = lags[lags.index(lag) + 1]
            cdf[lag] = ata_factors[lag] * cdf[next_lag]

        # Latest diagonal (most recent paid for each accident period)
        latest_diagonal = {}
        for period in accident_periods:
            row = triangle.loc[period]
            non_null = row.dropna()
            if len(non_null) > 0:
                latest_diagonal[period] = float(non_null.iloc[-1])
                latest_lag_for_period   = non_null.index[-1]
            else:
                latest_diagonal[period] = 0.0
                latest_lag_for_period   = lags[0]
            # Project ultimate using CDF from latest lag
            cdf_val = cdf.get(latest_lag_for_period, tail_factor)
            latest_diagonal[f"_cdf_{period}"] = cdf_val

        latest_diag_series = pd.Series(
            {p: latest_diagonal[p] for p in accident_periods}
        )
        cdf_series = pd.Series(
            {p: latest_diagonal[f"_cdf_{p}"] for p in accident_periods}
        )

        ultimate_cl = latest_diag_series * cdf_series
        ibnr_by_period = (ultimate_cl - latest_diag_series).clip(lower=0)
        ibnr_total = float(ibnr_by_period.sum())

        # ---- Bornhuetter-Ferguson ----
        if earned_premium_by_period is None:
            # Use a flat premium proxy: latest diagonal / a_priori_loss_ratio
            total_diag = latest_diag_series.sum()
            flat_prem  = total_diag / max(a_priori_loss_ratio, 1e-9) / len(accident_periods)
            earned_premium_by_period = pd.Series(flat_prem, index=accident_periods)

        expected_ultimate = a_priori_loss_ratio * earned_premium_by_period.reindex(accident_periods)
        # BF ultimate = expected_ult * (1 - 1/CDF) + actual_paid
        bf_ultimate = (
            expected_ultimate * (1 - 1 / cdf_series.replace(0, np.nan))
            + latest_diag_series
        ).fillna(latest_diag_series)

        bf_ibnr_by_period = (bf_ultimate - latest_diag_series).clip(lower=0)
        bf_ibnr_total     = float(bf_ibnr_by_period.sum())

        logger.success(
            f"estimate_ibnr: Chain Ladder IBNR={ibnr_total:,.0f}, "
            f"BF IBNR={bf_ibnr_total:,.0f}"
        )
        return {
            "method":               "chain_ladder",
            "ibnr_total":           ibnr_total,
            "ibnr_by_period":       ibnr_by_period,
            "ultimate_claims":      ultimate_cl,
            "development_factors":  ata_factors,
            "bf_ibnr_total":        bf_ibnr_total,
            "bf_by_period":         bf_ibnr_by_period,
        }

    # ------------------------------------------------------------------
    # Pharma Pipeline Signals
    # ------------------------------------------------------------------

    def compute_pharma_signals(
        self,
        trial_df: pd.DataFrame,
        company_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute pipeline risk / value signals for clinical trials.

        Parameters
        ----------
        trial_df : DataFrame with columns: trial_id, phase, condition,
                   enrollment_actual, enrollment_estimated, months_active,
                   expected_duration, peak_sales_estimate, years_to_launch,
                   primary_completion_date
        company_df : Optional company metadata to join on company_id

        Returns
        -------
        trial_df with additional signal columns.
        """
        logger.info(
            f"compute_pharma_signals: {len(trial_df):,} trials"
        )

        BASE_RATES: Dict[str, float] = {
            "PHASE1": 0.63,
            "PHASE2": 0.31,
            "PHASE3": 0.58,
            "PHASE4": 0.90,
        }
        ONCOLOGY_ADJ: Dict[str, float]    = {"PHASE2": 0.70, "PHASE3": 0.85}
        RARE_DISEASE_ADJ: Dict[str, float]= {"PHASE2": 1.15, "PHASE3": 1.10}

        today = datetime.now(tz=timezone.utc).date()

        df = trial_df.copy()

        # Normalise phase column
        if "phase" in df.columns:
            df["phase"] = df["phase"].astype(str).str.upper().str.replace(" ", "")

        # Ensure optional numeric columns with defaults
        for col, default in [
            ("enrollment_actual",    0.0),
            ("enrollment_estimated", 200.0),
            ("months_active",        12.0),
            ("expected_duration",    24.0),
            ("peak_sales_estimate",  500e6),
            ("years_to_launch",      5.0),
        ]:
            if col not in df.columns:
                logger.warning(f"  '{col}' missing – using default {default}")
                df[col] = default
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

        if "primary_completion_date" not in df.columns:
            df["primary_completion_date"] = pd.NaT
        df["primary_completion_date"] = pd.to_datetime(
            df["primary_completion_date"], errors="coerce"
        )

        if "condition" not in df.columns:
            df["condition"] = ""
        df["condition"] = df["condition"].astype(str).str.lower()

        # Indication flags
        df["_is_oncology"] = df["condition"].str.contains(
            "cancer|oncol|tumor|tumour", regex=True, na=False
        )
        df["_is_rare_disease"] = (
            df["condition"].str.contains("rare", na=False)
            | (df["enrollment_estimated"] < 200)
        )

        def _phase_success_prob(row: pd.Series) -> float:
            phase   = row.get("phase", "")
            base    = BASE_RATES.get(phase, 0.5)
            adj     = 1.0
            if row["_is_oncology"]:
                adj = ONCOLOGY_ADJ.get(phase, 1.0)
            elif row["_is_rare_disease"]:
                adj = RARE_DISEASE_ADJ.get(phase, 1.0)
            return float(base * adj)

        df["phase_success_prob"] = df.apply(_phase_success_prob, axis=1)

        # Enrollment velocity ratio
        expected_enrolled = (
            df["enrollment_estimated"]
            * df["months_active"]
            / df["expected_duration"].replace(0, 1)
        ).clip(lower=1)
        df["enrollment_velocity_ratio"] = (
            df["enrollment_actual"] / expected_enrolled
        ).clip(0, 3)

        df["velocity_signal"] = pd.cut(
            df["enrollment_velocity_ratio"],
            bins=[-np.inf, 0.85, 1.05, np.inf],
            labels=["NEGATIVE", "NEUTRAL", "POSITIVE"],
        ).astype(str)

        # rNPV contribution
        discount_factor = np.power(1.10, df["years_to_launch"].clip(lower=0))
        df["rnpv_contribution"] = (
            df["phase_success_prob"]
            * df["peak_sales_estimate"]
            * 0.20
            * 0.70
            / discount_factor
        )

        # Patent runway flag
        cutoff = pd.Timestamp(today + timedelta(days=365 * 5), tz=None)
        pcd = df["primary_completion_date"].dt.tz_localize(None) \
            if df["primary_completion_date"].dt.tz is not None \
            else df["primary_completion_date"]
        df["patent_runway_flag"] = (pcd > cutoff).astype(int)

        # Competitive pressure: count of other trials in same condition
        # with same or later phase (PHASE3 >= PHASE2, etc.)
        phase_order = {"PHASE1": 1, "PHASE2": 2, "PHASE3": 3, "PHASE4": 4}
        df["_phase_rank"] = df["phase"].map(phase_order).fillna(0)

        def _competitive_pressure(row: pd.Series) -> int:
            cond        = row["condition"]
            phase_rank  = row["_phase_rank"]
            peers = df[
                (df["condition"] == cond)
                & (df["_phase_rank"] >= phase_rank)
            ]
            # Exclude self
            return max(0, len(peers) - 1)

        df["competitive_pressure"] = df.apply(_competitive_pressure, axis=1)

        # Drop internal helper columns
        df.drop(columns=["_is_oncology", "_is_rare_disease", "_phase_rank"],
                errors="ignore", inplace=True)

        if company_df is not None and "company_id" in df.columns and "company_id" in company_df.columns:
            df = df.merge(company_df, on="company_id", how="left", suffixes=("", "_co"))
            logger.debug("  Merged company_df on company_id")

        logger.success(
            f"compute_pharma_signals: {len(df):,} trials processed"
        )
        return df

    # ------------------------------------------------------------------
    # ESG Scores
    # ------------------------------------------------------------------

    def compute_esg_scores(
        self,
        company_df: pd.DataFrame,
        faers_df: Optional[pd.DataFrame] = None,
        trial_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute ESG composite scores for pharmaceutical companies.

        Parameters
        ----------
        company_df : DataFrame with columns:
            company_name, total_revenue, top_drug_revenue, rd_expense,
            fda_warning_letters_3yr, clinical_trials_registered,
            clinical_trials_results_posted, manufacturing_sites,
            opioid_revenue_pct (0-1)
        faers_df : Optional FDA adverse event data (currently informational)
        trial_df : Optional trial data (currently informational)

        Returns
        -------
        company_df with ESG score columns appended.
        """
        logger.info(
            f"compute_esg_scores: {len(company_df):,} companies"
        )

        ESG_COLS = [
            "company_name", "total_revenue", "top_drug_revenue",
            "rd_expense", "fda_warning_letters_3yr",
            "clinical_trials_registered", "clinical_trials_results_posted",
            "manufacturing_sites", "opioid_revenue_pct",
        ]
        df = company_df.copy()
        for col in ESG_COLS:
            if col not in df.columns:
                logger.warning(f"  Missing ESG column '{col}' – filling with NaN/0")
                df[col] = np.nan if col not in ("opioid_revenue_pct",) else 0.0

        # Coerce
        for col in ESG_COLS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["opioid_revenue_pct"]           = df["opioid_revenue_pct"].fillna(0.0)
        df["clinical_trials_registered"]   = df["clinical_trials_registered"].fillna(0)
        df["clinical_trials_results_posted"]= df["clinical_trials_results_posted"].fillna(0)
        df["fda_warning_letters_3yr"]      = df["fda_warning_letters_3yr"].fillna(0)

        # Transparency ratio
        transparency_ratio = (
            df["clinical_trials_results_posted"]
            / df["clinical_trials_registered"].replace(0, 1)
        ).clip(0, 1)

        # ---- Social Score (0–40) ----
        df["drug_pricing_exposure_score"] = (
            (1 - df["top_drug_revenue"] / df["total_revenue"].replace(0, np.nan))
            .clip(0, 1)
            .fillna(0)
            * 15
        )
        df["opioid_risk_score"]      = (1 - df["opioid_revenue_pct"]) * 15
        df["trial_transparency_score"] = transparency_ratio * 10
        df["social_score"] = (
            df["drug_pricing_exposure_score"]
            + df["opioid_risk_score"]
            + df["trial_transparency_score"]
        )

        # ---- Environmental Score (0–30) ----
        df["manufacturing_compliance"] = (
            (1 - (df["fda_warning_letters_3yr"] / 3).clip(0, 1)) * 30
        )
        df["environmental_score"] = df["manufacturing_compliance"]

        # ---- Governance Score (0–30) ----
        df["trial_transparency_governance"] = transparency_ratio * 15
        df["fda_compliance_governance"]      = (
            (1 - (df["fda_warning_letters_3yr"] / 5).clip(0, 1)) * 15
        )
        df["governance_score"] = (
            df["trial_transparency_governance"] + df["fda_compliance_governance"]
        )

        # ---- Composite ----
        df["esg_composite"] = (
            df["social_score"] + df["environmental_score"] + df["governance_score"]
        ).clip(0, 100)

        df["esg_tier"] = pd.cut(
            df["esg_composite"],
            bins=[-np.inf, 50, 75, np.inf],
            labels=["Laggard", "Average", "Leader"],
        ).astype(str)

        logger.success(
            f"compute_esg_scores: mean ESG={df['esg_composite'].mean():.1f}, "
            f"tier distribution={df['esg_tier'].value_counts().to_dict()}"
        )
        return df
