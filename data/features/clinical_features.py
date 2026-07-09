"""
Clinical feature engineering for HealthRiskAI.
Produces ICD-10 hierarchy encodings, HCC risk scores, lab trajectories,
medication features, Charlson comorbidity index, and utilisation features.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


# ---------------------------------------------------------------------------
# ICD-10 chapter mapping
# ---------------------------------------------------------------------------
CHAPTER_NAMES: Dict[str, str] = {
    "A": "Infectious",
    "B": "Parasitic",
    "C": "Neoplasms",
    "D": "Blood",
    "E": "Endocrine",
    "F": "Mental",
    "G": "Nervous",
    "H": "Eye_Ear",
    "I": "Circulatory",
    "J": "Respiratory",
    "K": "Digestive",
    "L": "Skin",
    "M": "Musculoskeletal",
    "N": "Genitourinary",
    "O": "Pregnancy",
    "P": "Perinatal",
    "Q": "Congenital",
    "R": "Symptoms",
    "S": "Injury",
    "T": "Poisoning",
    "U": "Special",
    "V": "External",
    "W": "External",
    "X": "External",
    "Y": "External",
    "Z": "Factors",
}

# ---------------------------------------------------------------------------
# HCC mapping: icd_prefix -> (hcc_number, hcc_label, risk_weight)
# ---------------------------------------------------------------------------
HCC_MAPPING: Dict[str, Tuple[Optional[int], str, float]] = {
    "E10":  (19, "Diabetes_no_complication",     0.302),
    "E11":  (19, "Diabetes_no_complication",     0.302),
    "E102": (18, "Diabetes_acute_complication",  0.318),
    "E112": (18, "Diabetes_acute_complication",  0.318),
    "E104": (17, "Diabetes_chronic_complication", 0.368),
    "E114": (17, "Diabetes_chronic_complication", 0.368),
    "I50":  (85, "Heart_failure",                0.331),
    "I21":  (86, "Acute_MI",                     0.199),
    "I22":  (86, "Acute_MI",                     0.199),
    "I48":  (96, "Atrial_fibrillation",           0.168),
    "N184": (136, "CKD_stage4",                  0.289),
    "N185": (136, "CKD_stage5",                  0.289),
    "N186": (136, "ESRD",                        0.289),
    "N183": (138, "CKD_stage3",                  0.069),
    "J44":  (111, "COPD",                        0.335),
    "G30":  (51, "Dementia",                     0.346),
    "G31":  (51, "Dementia",                     0.346),
    "F20":  (57, "Schizophrenia",                0.421),
    "F31":  (58, "Bipolar",                      0.345),
    "C18":  (10, "Colorectal_cancer",            1.023),
    "C34":  (9,  "Lung_cancer",                  1.456),
    "C61":  (12, "Prostate_cancer",              0.578),
    "C50":  (11, "Breast_cancer",                0.634),
    "Z87":  (None, "History_flag",               0.0),
}

# HCC hierarchy groups: more-specific HCC overrides less-specific within group
# key = (less_specific_hcc, more_specific_hcc)
HCC_HIERARCHY: List[Tuple[int, int]] = [
    (19, 18),  # diabetes no complication -> acute complication
    (19, 17),  # diabetes no complication -> chronic complication
    (18, 17),  # acute complication -> chronic complication
]

# ---------------------------------------------------------------------------
# Lab item mappings: itemid -> (lab_name, (low, high), unit)
# ---------------------------------------------------------------------------
LAB_ITEM_MAP: Dict[int, Tuple[str, Tuple[float, float], str]] = {
    50852:  ("hba1c",         (4.0,  7.0),   "%"),
    50912:  ("creatinine",    (0.6,  1.2),   "mg/dL"),
    50820:  ("ph_arterial",   (7.35, 7.45),  ""),
    50811:  ("hemoglobin",    (12.0, 17.5),  "g/dL"),
    51006:  ("bun",           (7.0,  20.0),  "mg/dL"),
    50983:  ("sodium",        (136.0, 145.0),"mEq/L"),
    50971:  ("potassium",     (3.5,  5.1),   "mEq/L"),
    51222:  ("hemoglobin_cbc",(12.0, 17.5),  "g/dL"),
    51301:  ("wbc",           (4.5,  11.0),  "K/uL"),
    51265:  ("platelets",     (150.0, 400.0),"K/uL"),
    50885:  ("bilirubin",     (0.1,  1.2),   "mg/dL"),
    50878:  ("ast",           (10.0, 40.0),  "IU/L"),
    50861:  ("alt",           (7.0,  56.0),  "IU/L"),
}

# ---------------------------------------------------------------------------
# Medication ATC class mapping (partial drug name -> class_name)
# ---------------------------------------------------------------------------
ATC_DRUG_MAP: Dict[str, str] = {
    "metformin":    "antidiabetic_biguanide",
    "insulin":      "antidiabetic_insulin",
    "glipizide":    "antidiabetic_sulfonylurea",
    "sitagliptin":  "antidiabetic_dpp4",
    "lisinopril":   "antihypertensive_acei",
    "losartan":     "antihypertensive_arb",
    "amlodipine":   "antihypertensive_ccb",
    "metoprolol":   "antihypertensive_bb",
    "atorvastatin": "statin",
    "simvastatin":  "statin",
    "warfarin":     "anticoagulant",
    "heparin":      "anticoagulant",
    "aspirin":      "antiplatelet",
    "morphine":     "opioid",
    "oxycodone":    "opioid",
    "fentanyl":     "opioid",
    "hydrocodone":  "opioid",
    "sertraline":   "antidepressant_ssri",
    "fluoxetine":   "antidepressant_ssri",
    "quetiapine":   "antipsychotic",
    "haloperidol":  "antipsychotic",
    "amoxicillin":  "antibiotic",
    "vancomycin":   "antibiotic",
    "albuterol":    "bronchodilator",
    "omeprazole":   "ppi",
}

# OME conversion factors (morphine milligram equivalents per mg of drug)
OME_FACTORS: Dict[str, float] = {
    "morphine":    1.0,
    "oxycodone":   1.5,
    "hydrocodone": 1.0,
    "fentanyl":    100.0,
    "codeine":     0.15,
    "tramadol":    0.1,
}

# ---------------------------------------------------------------------------
# Charlson Comorbidity Index mapping
# ---------------------------------------------------------------------------
CHARLSON_CONDITIONS: List[Tuple[List[str], str, int]] = [
    (["I21", "I22", "I25"],                                                          "myocardial_infarction",     1),
    (["I50"],                                                                         "heart_failure",             1),
    (["I70", "I71", "I73", "I74"],                                                   "peripheral_vascular",       1),
    (["I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"],        "cerebrovascular",           1),
    (["F00", "F01", "F02", "F03", "G30"],                                            "dementia",                  1),
    (["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47"],                      "copd",                      1),
    (["M05", "M06", "M09", "M30", "M31", "M32", "M33", "M34", "M35"],               "connective_tissue",         1),
    (["K25", "K26", "K27", "K28"],                                                   "peptic_ulcer",              1),
    (["B18", "K70", "K73", "K74"],                                                   "mild_liver",                1),
    (["E10", "E11", "E12", "E13", "E14"],                                            "diabetes_no_complication",  1),
    (["E102", "E103", "E104", "E105", "E107", "E112", "E113", "E114", "E115", "E117"], "diabetes_with_complication", 2),
    (["G81", "G82", "G83"],                                                          "hemiplegia",                2),
    (["N18", "N19"],                                                                 "renal_disease",             2),
    (["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],                             "cancer_no_metastasis",      2),
    (["K721", "K726", "K766"],                                                       "severe_liver",              3),
    (["C77", "C78", "C79", "C80"],                                                   "metastatic_cancer",         6),
    (["B20", "B21", "B22", "B24"],                                                   "aids_hiv",                  6),
]


class ClinicalFeatureEngineer:
    """
    Transforms raw MIMIC-IV style clinical data into ML-ready feature sets.
    All methods return a new DataFrame; inputs are never mutated.
    """

    # ------------------------------------------------------------------
    # ICD-10 Hierarchy Encoding
    # ------------------------------------------------------------------

    def encode_icd10_hierarchy(self, diagnoses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode ICD-10 codes into hierarchical features per admission.

        Parameters
        ----------
        diagnoses_df : DataFrame with columns subject_id, hadm_id, icd_code

        Returns
        -------
        DataFrame with one row per (subject_id, hadm_id) containing:
          - n_codes_{chapter_name} for every known chapter
          - binary flags for top-50 most common 3-char blocks
          - n_total_diagnoses
          - n_unique_chapters
        """
        logger.info(
            f"encode_icd10_hierarchy: {len(diagnoses_df):,} diagnosis rows, "
            f"{diagnoses_df['hadm_id'].nunique():,} admissions"
        )

        df = diagnoses_df[["subject_id", "hadm_id", "icd_code"]].copy()
        df["icd_code"] = df["icd_code"].astype(str).str.strip().str.upper()

        # Derive hierarchy levels
        df["chapter_letter"] = df["icd_code"].str[0]
        df["chapter_name"]   = df["chapter_letter"].map(CHAPTER_NAMES).fillna("Unknown")
        df["block"]          = df["icd_code"].str[:3]

        # ----- Chapter-level counts -----
        chapter_counts = (
            df.groupby(["subject_id", "hadm_id", "chapter_name"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        # Rename chapter columns
        chapter_cols = [
            c for c in chapter_counts.columns if c not in ("subject_id", "hadm_id")
        ]
        chapter_counts = chapter_counts.rename(
            columns={c: f"n_codes_{c.lower()}" for c in chapter_cols}
        )

        # Ensure every known chapter column exists
        all_chapter_cols = [
            f"n_codes_{name.lower()}" for name in set(CHAPTER_NAMES.values())
        ]
        for col in all_chapter_cols:
            if col not in chapter_counts.columns:
                chapter_counts[col] = 0

        # ----- Aggregate stats -----
        agg = (
            df.groupby(["subject_id", "hadm_id"])
            .agg(
                n_total_diagnoses=("icd_code", "count"),
                n_unique_chapters=("chapter_letter", "nunique"),
            )
            .reset_index()
        )

        # ----- Top-50 block flags -----
        top_blocks: List[str] = (
            df["block"].value_counts().head(50).index.tolist()
        )
        logger.debug(f"Top-50 ICD blocks: {top_blocks[:5]} ...")

        block_flags = (
            df[df["block"].isin(top_blocks)]
            .groupby(["subject_id", "hadm_id", "block"])
            .size()
            .unstack(fill_value=0)
            .clip(upper=1)  # binary flag
            .reset_index()
        )
        block_flags = block_flags.rename(
            columns={b: f"block_{b}" for b in top_blocks if b in block_flags.columns}
        )

        # Merge everything
        result = agg.merge(chapter_counts, on=["subject_id", "hadm_id"], how="left")
        result = result.merge(block_flags, on=["subject_id", "hadm_id"], how="left")

        # Fill any remaining NaNs with 0
        block_flag_cols = [c for c in result.columns if c.startswith("block_")]
        result[block_flag_cols] = result[block_flag_cols].fillna(0).astype(int)

        logger.success(
            f"encode_icd10_hierarchy: produced {len(result):,} rows × {result.shape[1]} cols"
        )
        return result

    # ------------------------------------------------------------------
    # HCC Risk Score
    # ------------------------------------------------------------------

    def compute_hcc_risk_score(self, diagnoses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute CMS-HCC risk scores per admission.

        Parameters
        ----------
        diagnoses_df : DataFrame with columns subject_id, hadm_id, icd_code

        Returns
        -------
        DataFrame with one row per (subject_id, hadm_id) and HCC features.
        """
        logger.info(
            f"compute_hcc_risk_score: {len(diagnoses_df):,} rows"
        )

        df = diagnoses_df[["subject_id", "hadm_id", "icd_code"]].copy()
        df["icd_code"] = df["icd_code"].astype(str).str.strip().str.upper()

        # Sort prefixes by length descending so longer (more specific) match first
        sorted_prefixes = sorted(HCC_MAPPING.keys(), key=len, reverse=True)

        def _match_hcc(code: str) -> List[Tuple[Optional[int], str, float]]:
            """Return all HCC entries whose prefix matches `code`."""
            matches: List[Tuple[Optional[int], str, float]] = []
            for prefix in sorted_prefixes:
                if code.startswith(prefix):
                    matches.append(HCC_MAPPING[prefix])
            return matches

        # Explode to (subject_id, hadm_id, hcc_number, hcc_label, risk_weight)
        records = []
        for _, row in df.iterrows():
            for hcc_num, hcc_label, weight in _match_hcc(row["icd_code"]):
                records.append({
                    "subject_id": row["subject_id"],
                    "hadm_id":    row["hadm_id"],
                    "hcc_number": hcc_num,
                    "hcc_label":  hcc_label,
                    "risk_weight": weight,
                })

        if not records:
            logger.warning("compute_hcc_risk_score: no HCC matches found – returning empty frame")
            keys = diagnoses_df[["subject_id", "hadm_id"]].drop_duplicates()
            for col in ["hcc_count", "hcc_risk_score", "top_3_hcc_labels",
                        "has_diabetes", "has_heart_failure", "has_ckd",
                        "has_copd", "has_cancer", "has_dementia"]:
                keys[col] = 0
            keys["top_3_hcc_labels"] = ""
            return keys

        hcc_df = pd.DataFrame(records)

        # Apply HCC hierarchy: within each admission, if more-specific HCC present,
        # drop the less-specific one for the same category group
        def _apply_hierarchy(group: pd.DataFrame) -> pd.DataFrame:
            present_hcc = set(group["hcc_number"].dropna().unique())
            to_drop: set = set()
            for less_spec, more_spec in HCC_HIERARCHY:
                if less_spec in present_hcc and more_spec in present_hcc:
                    to_drop.add(less_spec)
            return group[~group["hcc_number"].isin(to_drop)]

        hcc_df = (
            hcc_df.groupby(["subject_id", "hadm_id"], group_keys=False)
            .apply(_apply_hierarchy)
        )

        # Deduplicate: one HCC number per admission (keep highest weight)
        hcc_df = (
            hcc_df.sort_values("risk_weight", ascending=False)
            .drop_duplicates(subset=["subject_id", "hadm_id", "hcc_number"])
        )

        def _top3_labels(labels: pd.Series) -> str:
            return "|".join(labels.head(3).tolist())

        # Aggregate per admission
        agg = (
            hcc_df.groupby(["subject_id", "hadm_id"])
            .agg(
                hcc_count    =("hcc_number", "nunique"),
                hcc_risk_score=("risk_weight", "sum"),
                top_3_hcc_labels=("hcc_label", _top3_labels),
            )
            .reset_index()
        )

        # Condition flags
        diabetes_labels   = {"Diabetes_no_complication", "Diabetes_acute_complication", "Diabetes_chronic_complication"}
        hf_labels         = {"Heart_failure"}
        ckd_labels        = {"CKD_stage3", "CKD_stage4", "CKD_stage5", "ESRD"}
        copd_labels       = {"COPD"}
        cancer_labels     = {"Colorectal_cancer", "Lung_cancer", "Prostate_cancer", "Breast_cancer"}
        dementia_labels   = {"Dementia"}

        def _flag(group: pd.DataFrame, label_set: set) -> int:
            return int(group["hcc_label"].isin(label_set).any())

        for flag_col, label_set in [
            ("has_diabetes",      diabetes_labels),
            ("has_heart_failure", hf_labels),
            ("has_ckd",           ckd_labels),
            ("has_copd",          copd_labels),
            ("has_cancer",        cancer_labels),
            ("has_dementia",      dementia_labels),
        ]:
            flags = (
                hcc_df.groupby(["subject_id", "hadm_id"])
                .apply(lambda g, ls=label_set: int(g["hcc_label"].isin(ls).any()))
                .reset_index(name=flag_col)
            )
            agg = agg.merge(flags, on=["subject_id", "hadm_id"], how="left")

        flag_cols = ["has_diabetes", "has_heart_failure", "has_ckd",
                     "has_copd", "has_cancer", "has_dementia"]
        agg[flag_cols] = agg[flag_cols].fillna(0).astype(int)

        logger.success(
            f"compute_hcc_risk_score: {len(agg):,} admissions, "
            f"mean risk score={agg['hcc_risk_score'].mean():.3f}"
        )
        return agg

    # ------------------------------------------------------------------
    # Lab Trajectories
    # ------------------------------------------------------------------

    def compute_lab_trajectories(self, lab_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-patient lab trend statistics and return a wide DataFrame.

        Parameters
        ----------
        lab_df : DataFrame with subject_id, hadm_id, itemid, charttime,
                 valuenum, valueuom

        Returns
        -------
        Wide DataFrame: one row per subject_id with columns
          lab_{lab_name}_{stat}
        """
        logger.info(
            f"compute_lab_trajectories: {len(lab_df):,} rows, "
            f"{lab_df['subject_id'].nunique():,} patients"
        )

        df = lab_df[["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]].copy()
        df = df[df["itemid"].isin(LAB_ITEM_MAP)]
        df["charttime"] = pd.to_datetime(df["charttime"], utc=True, errors="coerce")
        df = df.dropna(subset=["valuenum", "charttime"])

        today = datetime.now(tz=timezone.utc)

        # Attach lab name and normal range
        df["lab_name"]   = df["itemid"].map(lambda i: LAB_ITEM_MAP[i][0])
        df["normal_low"] = df["itemid"].map(lambda i: LAB_ITEM_MAP[i][1][0])
        df["normal_high"]= df["itemid"].map(lambda i: LAB_ITEM_MAP[i][1][1])

        all_wide: List[pd.DataFrame] = []

        for lab_name, lab_group in df.groupby("lab_name"):
            logger.debug(f"  Processing lab: {lab_name} ({len(lab_group):,} rows)")
            stat_rows: List[dict] = []

            for subject_id, pt_group in lab_group.groupby("subject_id"):
                vals        = pt_group["valuenum"].values.astype(float)
                times       = pt_group["charttime"]
                time_numeric= times.astype(np.int64).values.astype(float)  # nanoseconds

                n  = len(vals)
                mean_val = float(np.mean(vals)) if n > 0 else np.nan

                # Trend slope via linear regression
                if n >= 2:
                    slope, *_ = stats.linregress(time_numeric, vals)
                else:
                    slope = 0.0

                # Coefficient of variation
                cv = float(np.std(vals) / mean_val) if (mean_val != 0 and not np.isnan(mean_val)) else 0.0

                normal_low  = pt_group["normal_low"].iloc[0]
                normal_high = pt_group["normal_high"].iloc[0]

                above_pct = float(np.mean(vals > normal_high)) if n > 0 else 0.0
                below_pct = float(np.mean(vals < normal_low))  if n > 0 else 0.0

                last_time = times.max()
                days_since = (today - last_time).days if pd.notnull(last_time) else np.nan

                stat_rows.append({
                    "subject_id": subject_id,
                    f"lab_{lab_name}_latest_value":     float(vals[-1]) if n > 0 else np.nan,
                    f"lab_{lab_name}_min_value":        float(np.min(vals)),
                    f"lab_{lab_name}_max_value":        float(np.max(vals)),
                    f"lab_{lab_name}_mean_value":       mean_val,
                    f"lab_{lab_name}_n_measurements":   n,
                    f"lab_{lab_name}_trend_slope":      float(slope),
                    f"lab_{lab_name}_cv":               cv,
                    f"lab_{lab_name}_above_normal_pct": above_pct,
                    f"lab_{lab_name}_below_normal_pct": below_pct,
                    f"lab_{lab_name}_days_since_last":  days_since,
                })

            if stat_rows:
                all_wide.append(pd.DataFrame(stat_rows))

        if not all_wide:
            logger.warning("compute_lab_trajectories: no lab data matched known item IDs")
            return pd.DataFrame(columns=["subject_id"])

        # Merge all lab DataFrames on subject_id
        result = all_wide[0]
        for other in all_wide[1:]:
            result = result.merge(other, on="subject_id", how="outer")

        logger.success(
            f"compute_lab_trajectories: {len(result):,} patients × {result.shape[1]} features"
        )
        return result

    # ------------------------------------------------------------------
    # Medication Features
    # ------------------------------------------------------------------

    def compute_medication_features(self, rx_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute medication-based features per admission.

        Parameters
        ----------
        rx_df : DataFrame with subject_id, hadm_id, drug, ndc,
                starttime, stoptime, dose_val_rx, route

        Returns
        -------
        DataFrame with one row per (subject_id, hadm_id) and medication features.
        """
        logger.info(
            f"compute_medication_features: {len(rx_df):,} rows"
        )

        df = rx_df[["subject_id", "hadm_id", "drug", "starttime",
                    "stoptime", "dose_val_rx"]].copy()
        df["drug"]         = df["drug"].astype(str).str.lower().str.strip()
        df["starttime"]    = pd.to_datetime(df["starttime"], errors="coerce")
        df["stoptime"]     = pd.to_datetime(df["stoptime"],  errors="coerce")
        df["dose_val_rx"]  = pd.to_numeric(df["dose_val_rx"], errors="coerce").fillna(0.0)

        # Days supply per prescription row
        df["days_supply"] = (
            (df["stoptime"] - df["starttime"])
            .dt.total_seconds()
            .div(86_400)
            .clip(lower=0)
            .fillna(0.0)
        )

        # Map drug -> ATC class
        def _get_class(drug_name: str) -> Optional[str]:
            for partial, cls in ATC_DRUG_MAP.items():
                if partial in drug_name:
                    return cls
            return None

        df["drug_class"] = df["drug"].apply(_get_class)

        # OME factor
        def _ome_factor(drug_name: str) -> float:
            for opioid, factor in OME_FACTORS.items():
                if opioid in drug_name:
                    return factor
            return 0.0

        df["ome_factor"] = df["drug"].apply(_ome_factor)
        df["ome_dose"]   = df["ome_factor"] * df["dose_val_rx"]

        results = []
        for (subject_id, hadm_id), group in df.groupby(["subject_id", "hadm_id"]):
            drug_count   = group["drug"].nunique()
            n_drug_classes = group["drug_class"].nunique() - (1 if group["drug_class"].isna().any() else 0)
            n_drug_classes = max(0, int(n_drug_classes))

            classes_present = set(group["drug_class"].dropna().unique())

            # Observation period
            obs_start = group["starttime"].min()
            obs_end   = group["stoptime"].max()
            obs_days  = max(
                (obs_end - obs_start).total_seconds() / 86_400
                if pd.notnull(obs_start) and pd.notnull(obs_end)
                else 1.0,
                1.0,
            )

            total_days_supply = group["days_supply"].sum()
            mpr_proxy = min(total_days_supply / obs_days, 1.0)

            # Opioid MME daily
            opioid_rows = group[group["ome_factor"] > 0]
            if len(opioid_rows) > 0 and obs_days > 0:
                opioid_mme_daily = float(opioid_rows["ome_dose"].sum() / obs_days)
            else:
                opioid_mme_daily = 0.0

            results.append({
                "subject_id":            subject_id,
                "hadm_id":               hadm_id,
                "drug_count":            drug_count,
                "polypharmacy_flag":     int(drug_count >= 5),
                "high_polypharmacy_flag":int(drug_count >= 10),
                "n_drug_classes":        n_drug_classes,
                "has_antidiabetic":      int(any("antidiabetic" in c for c in classes_present)),
                "has_antihypertensive":  int(any("antihypertensive" in c for c in classes_present)),
                "has_statin":            int("statin" in classes_present),
                "has_anticoagulant":     int("anticoagulant" in classes_present),
                "has_opioid":            int("opioid" in classes_present),
                "opioid_mme_daily":      opioid_mme_daily,
                "mpr_proxy":             float(mpr_proxy),
            })

        result = pd.DataFrame(results)
        logger.success(
            f"compute_medication_features: {len(result):,} admissions"
        )
        return result

    # ------------------------------------------------------------------
    # Charlson Comorbidity Index
    # ------------------------------------------------------------------

    def compute_charlson_index(self, diagnoses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the Charlson Comorbidity Index (CCI) per admission.

        Parameters
        ----------
        diagnoses_df : DataFrame with columns subject_id, hadm_id, icd_code

        Returns
        -------
        DataFrame with charlson_score, 10-year mortality probability,
        and individual condition flags.
        """
        logger.info(
            f"compute_charlson_index: {len(diagnoses_df):,} rows"
        )

        df = diagnoses_df[["subject_id", "hadm_id", "icd_code"]].copy()
        df["icd_code"] = df["icd_code"].astype(str).str.strip().str.upper()

        keys = df[["subject_id", "hadm_id"]].drop_duplicates().reset_index(drop=True)

        # For each condition, flag admissions
        condition_frames: List[pd.DataFrame] = []
        for prefixes, condition_name, weight in CHARLSON_CONDITIONS:
            def _has_condition(codes: pd.Series, pfxs: List[str] = prefixes) -> bool:
                return codes.apply(
                    lambda c: any(c.startswith(p) for p in pfxs)
                ).any()

            flags = (
                df.groupby(["subject_id", "hadm_id"])["icd_code"]
                .apply(lambda codes, pfxs=prefixes: int(
                    codes.apply(lambda c: any(c.startswith(p) for p in pfxs)).any()
                ))
                .reset_index(name=f"cci_{condition_name}")
            )
            condition_frames.append((flags, weight, condition_name))

        # Merge all conditions onto keys
        result = keys.copy()
        for flags, weight, condition_name in condition_frames:
            result = result.merge(flags, on=["subject_id", "hadm_id"], how="left")
            result[f"cci_{condition_name}"] = result[f"cci_{condition_name}"].fillna(0).astype(int)

        # Charlson score = weighted sum of condition flags
        score_series = pd.Series(0.0, index=result.index)
        for _, weight, condition_name in condition_frames:
            score_series += result[f"cci_{condition_name}"] * weight

        result["charlson_score"] = score_series.astype(int)

        # 10-year mortality probability: 1 - 0.983^exp(0.9 * score)
        result["charlson_10yr_mortality_prob"] = (
            1 - np.power(0.983, np.exp(0.9 * result["charlson_score"]))
        ).round(4)

        logger.success(
            f"compute_charlson_index: {len(result):,} admissions, "
            f"mean score={result['charlson_score'].mean():.2f}"
        )
        return result

    # ------------------------------------------------------------------
    # Utilisation Features
    # ------------------------------------------------------------------

    def compute_utilisation_features(self, admissions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute historical utilisation features for each admission, looking
        back at prior admissions for the same patient.

        Parameters
        ----------
        admissions_df : DataFrame with subject_id, hadm_id, admittime,
                        dischtime, admission_type, edregtime

        Returns
        -------
        DataFrame with one row per (subject_id, hadm_id) and utilisation features.
        """
        logger.info(
            f"compute_utilisation_features: {len(admissions_df):,} admissions"
        )

        df = admissions_df[
            ["subject_id", "hadm_id", "admittime", "dischtime", "edregtime"]
        ].copy()
        df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
        df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")
        df["edregtime"] = pd.to_datetime(df["edregtime"], errors="coerce")

        df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

        results = []

        for subject_id, pt_df in df.groupby("subject_id"):
            pt_df = pt_df.sort_values("admittime").reset_index(drop=True)

            for idx, row in pt_df.iterrows():
                current_admit = row["admittime"]

                # All PRIOR admissions for this patient
                prior = pt_df[pt_df["admittime"] < current_admit]

                # Within 365 days
                prior_12m = prior[
                    (current_admit - prior["admittime"]).dt.days <= 365
                ]
                # Within 180 days
                prior_6m = prior[
                    (current_admit - prior["admittime"]).dt.days <= 180
                ]
                # Within 365 days AND ED visit
                prior_ed_12m = prior_12m[prior_12m["edregtime"].notna()]

                n_12m = len(prior_12m)
                n_6m  = len(prior_6m)
                n_ed  = len(prior_ed_12m)

                # Average LOS of prior admissions
                if len(prior) > 0:
                    los_days = (prior["dischtime"] - prior["admittime"]).dt.days
                    avg_los = float(los_days.dropna().mean()) if los_days.notna().any() else np.nan
                else:
                    avg_los = np.nan

                # Days since last admission
                if len(prior) > 0:
                    last_disch = prior["dischtime"].max()
                    days_since = (current_admit - last_disch).days if pd.notnull(last_disch) else np.nan
                else:
                    days_since = np.nan

                # Readmission acceleration: n_6m / max(n_12m - n_6m, 0.5)
                distant_half = n_12m - n_6m
                readmission_acceleration = float(n_6m) / max(float(distant_half), 0.5)

                results.append({
                    "subject_id":                row["subject_id"],
                    "hadm_id":                   row["hadm_id"],
                    "n_admissions_12m":           n_12m,
                    "n_admissions_6m":            n_6m,
                    "n_ed_visits_12m":            n_ed,
                    "avg_los_days":               avg_los,
                    "days_since_last_admission":  days_since,
                    "readmission_acceleration":   readmission_acceleration,
                })

        result = pd.DataFrame(results)
        logger.success(
            f"compute_utilisation_features: {len(result):,} rows produced"
        )
        return result
