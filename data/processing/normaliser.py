"""
data.processing.normaliser — Lab value and ICD code normalisation.

Two classes are provided:

:class:`LabNormaliser`
    Normalises numeric lab values to a ``[0, 1]`` range relative to
    clinically-standard reference ranges, with optional gender
    stratification.  Also produces ``'high'`` / ``'low'`` / ``'normal'``
    flags for each lab.

:class:`ICDNormaliser`
    Cleans and categorises ICD-9/10 codes into broad clinical domains
    (cardiovascular, respiratory, …) and maps them to ICD-10 chapters.

References
----------
- Clinical reference ranges: AACC, Mayo Clinic Laboratory Reference
- ICD-10 chapter structure: WHO ICD-10 2019 edition
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LabNormaliser — reference table
# ---------------------------------------------------------------------------

# Each entry:  lab_name -> {
#   'lo_f': female lower bound, 'hi_f': female upper bound,
#   'lo_m': male lower bound,   'hi_m': male upper bound,
#   'lo':   gender-neutral lower, 'hi': gender-neutral upper,
#   'unit': display unit,
# }
_LAB_REFERENCE: Dict[str, Dict] = {
    "hemoglobin": {
        "lo_f": 12.0, "hi_f": 16.0,
        "lo_m": 13.5, "hi_m": 17.5,
        "lo": 12.0, "hi": 17.5,
        "unit": "g/dL",
    },
    "creatinine": {
        "lo_f": 0.5, "hi_f": 1.1,
        "lo_m": 0.7, "hi_m": 1.3,
        "lo": 0.5, "hi": 1.3,
        "unit": "mg/dL",
    },
    "bun": {
        "lo": 7.0, "hi": 25.0,
        "lo_f": 7.0, "hi_f": 25.0,
        "lo_m": 7.0, "hi_m": 25.0,
        "unit": "mg/dL",
    },
    "sodium": {
        "lo": 136.0, "hi": 145.0,
        "lo_f": 136.0, "hi_f": 145.0,
        "lo_m": 136.0, "hi_m": 145.0,
        "unit": "mEq/L",
    },
    "potassium": {
        "lo": 3.5, "hi": 5.1,
        "lo_f": 3.5, "hi_f": 5.1,
        "lo_m": 3.5, "hi_m": 5.1,
        "unit": "mEq/L",
    },
    "glucose": {
        "lo": 70.0, "hi": 100.0,
        "lo_f": 70.0, "hi_f": 100.0,
        "lo_m": 70.0, "hi_m": 100.0,
        "unit": "mg/dL",
    },
    "hba1c": {
        "lo": 4.0, "hi": 5.7,
        "lo_f": 4.0, "hi_f": 5.7,
        "lo_m": 4.0, "hi_m": 5.7,
        "unit": "%",
    },
    "ldl": {
        "lo": 0.0, "hi": 100.0,
        "lo_f": 0.0, "hi_f": 100.0,
        "lo_m": 0.0, "hi_m": 100.0,
        "unit": "mg/dL",
    },
    "hdl": {
        "lo_f": 50.0, "hi_f": 80.0,
        "lo_m": 40.0, "hi_m": 60.0,
        "lo": 40.0, "hi": 80.0,
        "unit": "mg/dL",
    },
    "troponin": {
        "lo": 0.0, "hi": 0.04,
        "lo_f": 0.0, "hi_f": 0.04,
        "lo_m": 0.0, "hi_m": 0.04,
        "unit": "ng/mL",
    },
    "wbc": {
        "lo": 4.5, "hi": 11.0,
        "lo_f": 4.5, "hi_f": 11.0,
        "lo_m": 4.5, "hi_m": 11.0,
        "unit": "×10⁹/L",
    },
    "platelets": {
        "lo": 150.0, "hi": 400.0,
        "lo_f": 150.0, "hi_f": 400.0,
        "lo_m": 150.0, "hi_m": 400.0,
        "unit": "×10⁹/L",
    },
    "alt": {
        "lo_f": 7.0, "hi_f": 35.0,
        "lo_m": 7.0, "hi_m": 55.0,
        "lo": 7.0, "hi": 55.0,
        "unit": "U/L",
    },
    "ast": {
        "lo_f": 10.0, "hi_f": 34.0,
        "lo_m": 10.0, "hi_m": 40.0,
        "lo": 10.0, "hi": 40.0,
        "unit": "U/L",
    },
}

# Gender value aliases → canonical 'm' / 'f'
_GENDER_MAP: Dict[str, str] = {
    "m": "m", "male": "m", "1": "m",
    "f": "f", "female": "f", "0": "f",
}


class LabNormaliser:
    """Normalise numeric lab values to a ``[0, 1]`` range.

    The normalisation maps the clinical reference interval to ``[0, 1]``
    using a linear transform.  Values outside the reference range are
    clipped to ``[0, 1]`` **after** the transform so that truly extreme
    values do not silently appear within range.

    Parameters
    ----------
    reference:
        Override the built-in reference table.  Useful for site-specific
        adjustments or additional labs.
    clip:
        When *True* (default), the output is hard-clipped to ``[0, 1]``
        after transformation.

    Examples
    --------
    >>> ln = LabNormaliser()
    >>> df["hgb_norm"] = ln.normalise(df, "hemoglobin", gender_col="gender")
    """

    def __init__(
        self,
        reference: Optional[Dict[str, Dict]] = None,
        clip: bool = True,
    ) -> None:
        self._ref: Dict[str, Dict] = reference if reference is not None else _LAB_REFERENCE
        self.clip = clip
        self.recognised_labs: frozenset[str] = frozenset(self._ref.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_bounds(
        self,
        lab: str,
        gender_series: Optional[pd.Series],
        index: pd.Index,
    ) -> Tuple[pd.Series, pd.Series]:
        """Return (lo, hi) Series aligned to *index*."""
        ref = self._ref[lab]

        if gender_series is not None:
            # Canonicalise gender values
            g = (
                gender_series
                .str.lower()
                .map(_GENDER_MAP)
                .reindex(index)
            )
            lo = g.map({"m": ref["lo_m"], "f": ref["lo_f"]}).fillna(ref["lo"])
            hi = g.map({"m": ref["hi_m"], "f": ref["hi_f"]}).fillna(ref["hi"])
        else:
            lo = pd.Series(ref["lo"], index=index, dtype=float)
            hi = pd.Series(ref["hi"], index=index, dtype=float)

        return lo.astype(float), hi.astype(float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalise(
        self,
        df: pd.DataFrame,
        lab_col: str,
        gender_col: Optional[str] = None,
    ) -> pd.Series:
        """Normalise a single lab column to ``[0, 1]``.

        Parameters
        ----------
        df:
            DataFrame containing the lab column.
        lab_col:
            Name of the lab column in *df*.  Must match a key in the
            reference table (case-insensitive).
        gender_col:
            Optional column name holding patient gender
            (``'M'`` / ``'F'``, ``'male'`` / ``'female'``, ``0`` / ``1``).
            When provided, gender-specific reference ranges are used.

        Returns
        -------
        pd.Series
            Normalised values (``float64``, ``NaN`` preserved).

        Raises
        ------
        KeyError
            If *lab_col* is not found in the reference table.
        """
        lab_key = lab_col.lower()
        if lab_key not in self._ref:
            raise KeyError(
                f"Lab '{lab_col}' not in reference table.  "
                f"Known labs: {sorted(self._ref)}"
            )

        g_series: Optional[pd.Series] = (
            df[gender_col] if gender_col and gender_col in df.columns else None
        )
        lo, hi = self._resolve_bounds(lab_key, g_series, df.index)
        range_width = (hi - lo).replace(0, np.nan)  # avoid division by zero

        normalised = (df[lab_col].astype(float) - lo) / range_width
        if self.clip:
            normalised = normalised.clip(lower=0.0, upper=1.0)

        logger.debug("normalise: '%s' — %d values processed.", lab_col, len(normalised))
        return normalised.rename(f"{lab_col}_norm")

    def normalise_all_labs(
        self,
        df: pd.DataFrame,
        gender_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Normalise all recognised lab columns found in *df*.

        Columns whose lowercased name matches a key in the reference table
        are normalised and appended as ``{col}_norm`` columns.

        Parameters
        ----------
        df:
            DataFrame to normalise.
        gender_col:
            Optional gender column for gender-stratified normalisation.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with normalised columns appended.
        """
        df = df.copy()
        found: List[str] = []
        for col in df.columns:
            if col.lower() in self.recognised_labs:
                norm_series = self.normalise(df, col, gender_col=gender_col)
                df[norm_series.name] = norm_series
                found.append(col)

        logger.info(
            "normalise_all_labs: normalised %d lab columns: %s", len(found), found
        )
        return df

    def flag_abnormal(
        self,
        df: pd.DataFrame,
        lab_col: str,
        gender_col: Optional[str] = None,
    ) -> pd.Series:
        """Flag each lab result as ``'high'``, ``'low'``, or ``'normal'``.

        Parameters
        ----------
        df:
            DataFrame containing the lab column.
        lab_col:
            Column name in *df*.
        gender_col:
            Optional gender column for gender-stratified ranges.

        Returns
        -------
        pd.Series
            String series with values ``'high'``, ``'low'``, ``'normal'``,
            or ``NaN`` for missing input values.
        """
        lab_key = lab_col.lower()
        if lab_key not in self._ref:
            raise KeyError(f"Lab '{lab_col}' not in reference table.")

        g_series: Optional[pd.Series] = (
            df[gender_col] if gender_col and gender_col in df.columns else None
        )
        lo, hi = self._resolve_bounds(lab_key, g_series, df.index)

        values = df[lab_col].astype(float)
        flags = pd.Series("normal", index=df.index, dtype=object)
        flags[values < lo] = "low"
        flags[values > hi] = "high"
        flags[values.isna()] = np.nan

        logger.debug(
            "flag_abnormal '%s': %s",
            lab_col,
            flags.value_counts().to_dict(),
        )
        return flags.rename(f"{lab_col}_flag")


# ---------------------------------------------------------------------------
# ICD-10 chapter & category maps
# ---------------------------------------------------------------------------

# ICD-10 chapter ranges (first letter + numeric prefix ranges).
# Format: (start_code_prefix, end_code_prefix_exclusive) → chapter name
_ICD10_CHAPTERS: List[Tuple[str, str, str]] = [
    ("A00", "B99", "Infectious and Parasitic Diseases"),
    ("C00", "D48", "Neoplasms"),
    ("D50", "D89", "Blood Diseases"),
    ("E00", "E89", "Endocrine, Nutritional and Metabolic Diseases"),
    ("F01", "F99", "Mental and Behavioural Disorders"),
    ("G00", "G99", "Diseases of the Nervous System"),
    ("H00", "H59", "Diseases of the Eye and Adnexa"),
    ("H60", "H95", "Diseases of the Ear and Mastoid Process"),
    ("I00", "I99", "Cardiovascular Diseases"),
    ("J00", "J99", "Respiratory Diseases"),
    ("K00", "K95", "Digestive Diseases"),
    ("L00", "L99", "Skin Diseases"),
    ("M00", "M99", "Musculoskeletal Diseases"),
    ("N00", "N99", "Genitourinary Diseases"),
    ("O00", "O9A", "Pregnancy and Childbirth"),
    ("P00", "P96", "Perinatal Conditions"),
    ("Q00", "Q99", "Congenital Malformations"),
    ("R00", "R99", "Symptoms and Signs"),
    ("S00", "T88", "Injury and Poisoning"),
    ("V00", "Y99", "External Causes"),
    ("Z00", "Z99", "Factors Influencing Health Status"),
]

# Broad category mapping based on code prefix patterns (ICD-9 + ICD-10).
_CATEGORY_PATTERNS: List[Tuple[str, str]] = [
    # ICD-10
    (r"^I",         "cardiovascular"),
    (r"^J",         "respiratory"),
    (r"^C|^D[0-4]", "neoplasms"),
    (r"^E",         "endocrine_metabolic"),
    (r"^F",         "mental_health"),
    (r"^G",         "neurological"),
    (r"^K",         "gastrointestinal"),
    (r"^N",         "renal_urological"),
    (r"^M",         "musculoskeletal"),
    (r"^L",         "dermatological"),
    (r"^A|^B",      "infectious"),
    (r"^R",         "symptoms_signs"),
    (r"^S|^T",      "injury_poisoning"),
    (r"^Z",         "social_determinants"),
    # ICD-9 ranges (kept as prefix patterns)
    (r"^39[0-9]|^4[0-5]", "cardiovascular"),
    (r"^46[0-9]|^47[0-9]|^48[0-9]|^49[0-9]|^5[0-1]", "respiratory"),
    (r"^14[0-9]|^15[0-9]|^16[0-9]|^17[0-9]|^18[0-9]|^19[0-9]|^20[0-9]", "neoplasms"),
    (r"^25[0-9]|^24[0-9]", "endocrine_metabolic"),
    (r"^29[0-9]|^3[0-2]", "mental_health"),
    (r"^32[0-9]|^33[0-9]|^34[0-9]", "neurological"),
    (r"^5[2-7]", "gastrointestinal"),
    (r"^58[0-9]|^59[0-9]", "renal_urological"),
]
_COMPILED_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(pat), cat) for pat, cat in _CATEGORY_PATTERNS
]


class ICDNormaliser:
    """Clean and categorise ICD-9/ICD-10 diagnostic codes.

    Parameters
    ----------
    unknown_label:
        Label used for codes that do not match any known category.
        Defaults to ``'other'``.

    Examples
    --------
    >>> icd = ICDNormaliser()
    >>> icd.to_category("I21.3")
    'cardiovascular'
    >>> icd.code_to_chapter("J18.9")
    'Respiratory Diseases'
    """

    def __init__(self, unknown_label: str = "other") -> None:
        self.unknown_label = unknown_label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalise_codes(self, codes: List[str]) -> List[str]:
        """Clean and uppercase a list of ICD codes.

        Strips whitespace, removes non-alphanumeric characters (except
        the decimal point), and uppercases every code.

        Parameters
        ----------
        codes:
            Raw ICD code strings (may include None / NaN entries).

        Returns
        -------
        list[str]
            Cleaned codes.  ``None`` / ``NaN`` inputs become empty string.
        """
        result: List[str] = []
        for c in codes:
            if c is None or (isinstance(c, float) and np.isnan(c)):
                result.append("")
                continue
            cleaned = re.sub(r"[^A-Za-z0-9.]", "", str(c)).upper().strip()
            result.append(cleaned)
        return result

    def to_category(self, code: str) -> str:
        """Map a single ICD code to a broad clinical category.

        Parameters
        ----------
        code:
            An ICD-9 or ICD-10 code string (raw or pre-cleaned).

        Returns
        -------
        str
            One of: ``'cardiovascular'``, ``'respiratory'``,
            ``'neoplasms'``, ``'endocrine_metabolic'``, ``'mental_health'``,
            ``'neurological'``, ``'gastrointestinal'``, ``'renal_urological'``,
            ``'musculoskeletal'``, ``'dermatological'``, ``'infectious'``,
            ``'symptoms_signs'``, ``'injury_poisoning'``,
            ``'social_determinants'``, or :attr:`unknown_label`.
        """
        if not code:
            return self.unknown_label
        clean = self.normalise_codes([code])[0]
        for pattern, category in _COMPILED_PATTERNS:
            if pattern.match(clean):
                return category
        return self.unknown_label

    def code_to_chapter(self, code: str) -> str:
        """Return the ICD-10 chapter name for *code*.

        Performs a linear scan over the ICD-10 chapter table.  ICD-9
        codes (purely numeric) return ``'ICD-9 — chapter lookup not supported'``.

        Parameters
        ----------
        code:
            An ICD-10 code string.

        Returns
        -------
        str
            Chapter name, e.g. ``'Cardiovascular Diseases'``.
        """
        clean = self.normalise_codes([code])[0]
        if not clean:
            return "Unknown"

        # ICD-9 codes are purely numeric
        if clean.isdigit():
            return "ICD-9 — chapter lookup not supported"

        # Strip decimal for prefix comparison
        stripped = clean.replace(".", "").upper()

        for start, end, chapter in _ICD10_CHAPTERS:
            if start <= stripped[:3] <= end:
                return chapter

        return "Unknown Chapter"

    def categorise_series(self, series: pd.Series) -> pd.Series:
        """Vectorised :meth:`to_category` over a pandas Series.

        Parameters
        ----------
        series:
            Series of ICD codes.

        Returns
        -------
        pd.Series
            Categorical series with broad category labels.
        """
        return series.map(self.to_category).astype("category")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    # --- LabNormaliser smoke test ---
    rng = np.random.default_rng(0)
    n = 50
    lab_df = pd.DataFrame(
        {
            "hemoglobin": rng.uniform(8, 20, n),
            "glucose": rng.uniform(50, 300, n),
            "creatinine": rng.uniform(0.3, 4.0, n),
            "troponin": rng.uniform(0.0, 0.5, n),
            "gender": rng.choice(["M", "F"], n),
        }
    )

    ln = LabNormaliser()
    lab_df = ln.normalise_all_labs(lab_df, gender_col="gender")
    print("=== LabNormaliser ===")
    print(lab_df[["hemoglobin", "hemoglobin_norm", "glucose", "glucose_norm"]].head(8))

    flags = ln.flag_abnormal(lab_df, "glucose")
    print("\nGlucose flags:")
    print(flags.value_counts())

    # --- ICDNormaliser smoke test ---
    icd = ICDNormaliser()
    test_codes = ["I21.3", "J18.9", "C50.911", "E11.9", "F32.1", "V10.0", "250.00", "410"]
    print("\n=== ICDNormaliser ===")
    for code in test_codes:
        cat = icd.to_category(code)
        chapter = icd.code_to_chapter(code)
        print(f"  {code:<12} → category={cat:<25} chapter={chapter}")
