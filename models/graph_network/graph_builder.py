"""
models/graph_network/graph_builder.py
=======================================
Builds a heterogeneous patient-disease-drug graph from MIMIC-IV data.

Node types:
  - Patient  (admissions / patients tables)
  - Disease  (ICD-10 diagnoses)
  - Drug     (prescriptions / NDC codes)
  - Procedure (ICD procedure codes)

Edge types:
  - patient-has_diagnosis-disease
  - patient-prescribed-drug
  - patient-underwent-procedure
  - disease-treated_by-drug       (co-occurrence)
  - drug-interacts_with-drug      (co-prescription)

Output: PyTorch Geometric HeteroData object (or edge-list files for custom GNN).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data" / "raw"
GRAPH_DIR = BASE / "data" / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ─── Node / edge index helpers ────────────────────────────────────────────────


class NodeIndex:
    """Maps string identifiers to consecutive integer indices."""

    def __init__(self, prefix: str = "") -> None:
        self._map: Dict[str, int] = {}
        self._list: List[str] = []
        self.prefix = prefix

    def get_or_add(self, key: str) -> int:
        if key not in self._map:
            idx = len(self._list)
            self._map[key] = idx
            self._list.append(key)
        return self._map[key]

    def __len__(self) -> int:
        return len(self._list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"id": self._list, "idx": range(len(self._list))})

    def get(self, key: str) -> Optional[int]:
        return self._map.get(key)


# ─── Graph builder ────────────────────────────────────────────────────────────


class PatientDrugDiseaseGraphBuilder:
    """
    Constructs a heterogeneous biomedical graph from MIMIC-IV (or synthetic) data.

    Usage
    -----
    builder = PatientDrugDiseaseGraphBuilder()
    builder.load_mimic(admissions_df, diagnoses_df, prescriptions_df, procedures_df)
    graph_data = builder.build()
    builder.save(graph_data)
    """

    def __init__(
        self,
        max_patients: Optional[int] = None,
        min_disease_freq: int = 5,
        min_drug_freq: int = 5,
        co_occurrence_threshold: int = 3,
    ) -> None:
        self.max_patients = max_patients
        self.min_disease_freq = min_disease_freq
        self.min_drug_freq = min_drug_freq
        self.co_occurrence_threshold = co_occurrence_threshold

        # Node indices
        self.patient_idx = NodeIndex("P")
        self.disease_idx = NodeIndex("D")
        self.drug_idx = NodeIndex("R")
        self.procedure_idx = NodeIndex("PR")

        # Edge lists (src_idx, dst_idx)
        self.edges: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # Edge features
        self.edge_features: Dict[str, List[np.ndarray]] = defaultdict(list)

        # Node features
        self.node_features: Dict[str, List[np.ndarray]] = defaultdict(list)

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_mimic(
        self,
        admissions: pd.DataFrame,
        diagnoses: pd.DataFrame,
        prescriptions: pd.DataFrame,
        procedures: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Load MIMIC-IV tables into builder.

        Expected columns:
        admissions: subject_id, hadm_id, admittime, dischtime, hospital_expire_flag, los_days
        diagnoses: subject_id, hadm_id, icd_code, icd_version, seq_num
        prescriptions: subject_id, hadm_id, drug, ndc, dose_val_rx, starttime
        procedures: subject_id, hadm_id, icd_code, icd_version
        """
        logger.info("Loading MIMIC-IV data into graph builder…")

        if self.max_patients:
            patient_ids = admissions["subject_id"].unique()[: self.max_patients]
            admissions = admissions[admissions["subject_id"].isin(patient_ids)]
            diagnoses = diagnoses[diagnoses["subject_id"].isin(patient_ids)]
            prescriptions = prescriptions[prescriptions["subject_id"].isin(patient_ids)]
            if procedures is not None:
                procedures = procedures[procedures["subject_id"].isin(patient_ids)]

        # Filter rare diseases / drugs
        disease_freq = diagnoses["icd_code"].value_counts()
        valid_diseases = disease_freq[disease_freq >= self.min_disease_freq].index
        diagnoses = diagnoses[diagnoses["icd_code"].isin(valid_diseases)]

        drug_freq = prescriptions["drug"].value_counts()
        valid_drugs = drug_freq[drug_freq >= self.min_drug_freq].index
        prescriptions = prescriptions[prescriptions["drug"].isin(valid_drugs)]

        logger.info(
            "Nodes: %d patients, %d diseases, %d drugs",
            admissions["subject_id"].nunique(),
            len(valid_diseases),
            len(valid_drugs),
        )

        self._build_patient_features(admissions)
        self._build_patient_disease_edges(diagnoses)
        self._build_patient_drug_edges(prescriptions)
        if procedures is not None:
            self._build_patient_procedure_edges(procedures)
        self._build_disease_drug_edges(diagnoses, prescriptions)
        self._build_drug_interaction_edges(prescriptions)

    def _build_patient_features(self, admissions: pd.DataFrame) -> None:
        """Create patient node features from admission data."""
        for _, row in admissions.drop_duplicates("subject_id").iterrows():
            pid = str(row["subject_id"])
            self.patient_idx.get_or_add(pid)

            # Feature vector: [age_bucket, sex_code, icu_flag, los_bucket]
            feat = np.zeros(8, dtype=np.float32)
            if "anchor_age" in row:
                feat[0] = min(row["anchor_age"] / 100.0, 1.0)
            if "gender" in row:
                feat[1] = 1.0 if str(row.get("gender", "")).upper() == "M" else 0.0
            if "los_days" in admissions.columns:
                feat[2] = min(row.get("los_days", 0) / 30.0, 1.0)
            if "hospital_expire_flag" in admissions.columns:
                feat[3] = float(row.get("hospital_expire_flag", 0))
            self.node_features["patient"].append(feat)

    def _build_patient_disease_edges(self, diagnoses: pd.DataFrame) -> None:
        """Add patient → disease edges."""
        count = 0
        for _, row in diagnoses.iterrows():
            pid = str(row["subject_id"])
            did = f"ICD{row['icd_version']}_{row['icd_code']}"

            p_idx = self.patient_idx.get_or_add(pid)
            d_idx = self.disease_idx.get_or_add(did)
            self.edges["patient-has_diagnosis-disease"].append((p_idx, d_idx))

            # Edge feature: [seq_num (primary vs secondary), icd_version]
            seq = float(row.get("seq_num", 1))
            feat = np.array([min(seq / 20.0, 1.0), float(row.get("icd_version", 10)) / 10.0])
            self.edge_features["patient-has_diagnosis-disease"].append(feat)
            count += 1

        # Disease node features: [frequency_normalized, icd_chapter_code]
        for did in self.disease_idx._list:
            code = did.split("_")[-1] if "_" in did else did
            feat = np.zeros(4, dtype=np.float32)
            feat[0] = hash(code) % 100 / 100.0  # pseudo-chapter encoding
            self.node_features["disease"].append(feat)

        logger.info("Added %d patient-disease edges", count)

    def _build_patient_drug_edges(self, prescriptions: pd.DataFrame) -> None:
        """Add patient → drug edges."""
        count = 0
        for _, row in prescriptions.iterrows():
            pid = str(row["subject_id"])
            drug = str(row["drug"]).upper().strip()

            p_idx = self.patient_idx.get_or_add(pid)
            r_idx = self.drug_idx.get_or_add(drug)
            self.edges["patient-prescribed-drug"].append((p_idx, r_idx))

            # Edge feature: [dose_normalized, duration_days]
            dose = float(str(row.get("dose_val_rx", 0)).replace(",", "") or 0)
            feat = np.array([min(dose / 1000.0, 1.0)])
            self.edge_features["patient-prescribed-drug"].append(feat)
            count += 1

        # Drug node features
        for drug in self.drug_idx._list:
            feat = np.zeros(4, dtype=np.float32)
            feat[0] = hash(drug) % 100 / 100.0  # placeholder ATC encoding
            self.node_features["drug"].append(feat)

        logger.info("Added %d patient-drug edges", count)

    def _build_patient_procedure_edges(self, procedures: pd.DataFrame) -> None:
        """Add patient → procedure edges."""
        count = 0
        for _, row in procedures.iterrows():
            pid = str(row["subject_id"])
            proc = f"PROC{row.get('icd_version', 10)}_{row['icd_code']}"

            p_idx = self.patient_idx.get_or_add(pid)
            pr_idx = self.procedure_idx.get_or_add(proc)
            self.edges["patient-underwent-procedure"].append((p_idx, pr_idx))
            count += 1

        logger.info("Added %d patient-procedure edges", count)

    def _build_disease_drug_edges(
        self, diagnoses: pd.DataFrame, prescriptions: pd.DataFrame
    ) -> None:
        """Add disease–drug co-occurrence edges (same admission)."""
        # Join on hadm_id
        merged = diagnoses[["hadm_id", "icd_code", "icd_version"]].merge(
            prescriptions[["hadm_id", "drug"]], on="hadm_id"
        )

        co_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for _, row in merged.iterrows():
            did = f"ICD{row['icd_version']}_{row['icd_code']}"
            drug = str(row["drug"]).upper().strip()
            co_counts[(did, drug)] += 1

        count = 0
        for (did, drug), freq in co_counts.items():
            if freq >= self.co_occurrence_threshold:
                if self.disease_idx.get(did) is not None and self.drug_idx.get(drug) is not None:
                    d_idx = self.disease_idx.get(did)
                    r_idx = self.drug_idx.get(drug)
                    self.edges["disease-treated_by-drug"].append((d_idx, r_idx))
                    self.edge_features["disease-treated_by-drug"].append(
                        np.array([min(freq / 100.0, 1.0)])
                    )
                    count += 1

        logger.info("Added %d disease-drug co-occurrence edges", count)

    def _build_drug_interaction_edges(self, prescriptions: pd.DataFrame) -> None:
        """Add drug–drug co-prescription edges (same admission)."""
        drug_pairs: Dict[Tuple[str, str], int] = defaultdict(int)

        for hadm_id, group in prescriptions.groupby("hadm_id"):
            drugs = list(set(group["drug"].str.upper().str.strip().tolist()))
            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    pair = tuple(sorted([drugs[i], drugs[j]]))
                    drug_pairs[pair] += 1

        count = 0
        for (d1, d2), freq in drug_pairs.items():
            if freq >= self.co_occurrence_threshold:
                if self.drug_idx.get(d1) is not None and self.drug_idx.get(d2) is not None:
                    r1 = self.drug_idx.get(d1)
                    r2 = self.drug_idx.get(d2)
                    self.edges["drug-interacts_with-drug"].append((r1, r2))
                    # Undirected — add reverse too
                    self.edges["drug-interacts_with-drug"].append((r2, r1))
                    count += 1

        logger.info("Added %d drug interaction edges", count)

    # ── Build / export ────────────────────────────────────────────────────────

    def build(self) -> Dict[str, Any]:
        """
        Returns a graph data dict:
        {
          'node_counts': {'patient': N, 'disease': M, 'drug': K, ...},
          'node_features': {'patient': np.array, 'disease': ...},
          'edges': {'patient-has_diagnosis-disease': np.array(2, E), ...},
          'edge_features': {'patient-has_diagnosis-disease': np.array(E, F), ...},
          'node_indices': {'patient': NodeIndex, ...},
        }
        """
        graph: Dict[str, Any] = {}

        graph["node_counts"] = {
            "patient": len(self.patient_idx),
            "disease": len(self.disease_idx),
            "drug": len(self.drug_idx),
            "procedure": len(self.procedure_idx),
        }

        # Stack node features
        graph["node_features"] = {}
        for ntype, feat_list in self.node_features.items():
            if feat_list:
                graph["node_features"][ntype] = np.stack(feat_list).astype(np.float32)

        # Stack edge tensors
        graph["edges"] = {}
        for etype, edge_list in self.edges.items():
            if edge_list:
                arr = np.array(edge_list, dtype=np.int64).T  # (2, E)
                graph["edges"][etype] = arr

        graph["edge_features"] = {}
        for etype, feat_list in self.edge_features.items():
            if feat_list:
                graph["edge_features"][etype] = np.stack(feat_list).astype(np.float32)

        graph["node_indices"] = {
            "patient": self.patient_idx,
            "disease": self.disease_idx,
            "drug": self.drug_idx,
            "procedure": self.procedure_idx,
        }

        logger.info("Graph built: %s", {k: v for k, v in graph["node_counts"].items()})
        logger.info(
            "Edges: %s", {k: v.shape[1] for k, v in graph["edges"].items()}
        )
        return graph

    def to_pyg(self, graph: Dict[str, Any]) -> Any:
        """Convert to PyTorch Geometric HeteroData (requires torch_geometric)."""
        try:
            from torch_geometric.data import HeteroData
            import torch

            data = HeteroData()

            for ntype, feats in graph["node_features"].items():
                data[ntype].x = torch.FloatTensor(feats)

            for etype, edges in graph["edges"].items():
                parts = etype.split("-")
                if len(parts) == 3:
                    src_type, rel, dst_type = parts
                    data[src_type, rel, dst_type].edge_index = torch.LongTensor(edges)
                    if etype in graph["edge_features"]:
                        data[src_type, rel, dst_type].edge_attr = torch.FloatTensor(
                            graph["edge_features"][etype]
                        )

            return data
        except ImportError:
            logger.warning("torch_geometric not installed — returning raw dict")
            return graph

    def save(self, graph: Dict[str, Any], output_dir: Optional[Path] = None) -> Path:
        """Save graph arrays as numpy files."""
        output_dir = output_dir or GRAPH_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        for ntype, feats in graph["node_features"].items():
            np.save(output_dir / f"node_features_{ntype}.npy", feats)

        for etype, edges in graph["edges"].items():
            safe_name = etype.replace("-", "_")
            np.save(output_dir / f"edges_{safe_name}.npy", edges)

        for etype, feats in graph["edge_features"].items():
            safe_name = etype.replace("-", "_")
            np.save(output_dir / f"edge_features_{safe_name}.npy", feats)

        # Save node count summary
        summary = pd.DataFrame(
            [{"node_type": k, "count": v} for k, v in graph["node_counts"].items()]
        )
        summary.to_csv(output_dir / "graph_summary.csv", index=False)
        logger.info("Graph saved to %s", output_dir)
        return output_dir

    @classmethod
    def load(cls, output_dir: Path) -> Dict[str, Any]:
        """Load saved graph from directory."""
        graph: Dict[str, Any] = {"node_features": {}, "edges": {}, "edge_features": {}}

        for f in output_dir.glob("node_features_*.npy"):
            ntype = f.stem.replace("node_features_", "")
            graph["node_features"][ntype] = np.load(f)

        for f in output_dir.glob("edges_*.npy"):
            # Reconstruct edge type name
            etype = f.stem.replace("edges_", "").replace("_", "-", 2)
            graph["edges"][etype] = np.load(f)

        for f in output_dir.glob("edge_features_*.npy"):
            etype = f.stem.replace("edge_features_", "").replace("_", "-", 2)
            graph["edge_features"][etype] = np.load(f)

        logger.info("Graph loaded from %s", output_dir)
        return graph


# ─── Synthetic graph for testing ────────────────────────────────────────────


def make_synthetic_graph(
    n_patients: int = 100,
    n_diseases: int = 30,
    n_drugs: int = 50,
    avg_diag_per_patient: int = 4,
    avg_drug_per_patient: int = 3,
) -> Dict[str, Any]:
    """Generate a small synthetic graph for unit tests."""
    rng = np.random.default_rng(42)
    builder = PatientDrugDiseaseGraphBuilder(
        min_disease_freq=1, min_drug_freq=1, co_occurrence_threshold=1
    )

    # Synthetic admissions
    admissions = pd.DataFrame({
        "subject_id": range(n_patients),
        "hadm_id": range(n_patients),
        "anchor_age": rng.integers(18, 90, n_patients),
        "gender": rng.choice(["M", "F"], n_patients),
        "los_days": rng.exponential(5, n_patients),
        "hospital_expire_flag": rng.binomial(1, 0.08, n_patients),
    })

    diseases = [f"E{i:02d}" for i in range(n_diseases)]
    drugs = [f"DRUG_{i:02d}" for i in range(n_drugs)]

    rows_diag = []
    rows_presc = []
    for pid in range(n_patients):
        hadm = pid
        for _ in range(avg_diag_per_patient):
            rows_diag.append({
                "subject_id": pid,
                "hadm_id": hadm,
                "icd_code": rng.choice(diseases),
                "icd_version": 10,
                "seq_num": 1,
            })
        for _ in range(avg_drug_per_patient):
            rows_presc.append({
                "subject_id": pid,
                "hadm_id": hadm,
                "drug": rng.choice(drugs),
                "ndc": "000",
                "dose_val_rx": rng.uniform(50, 500),
                "starttime": "2020-01-01",
            })

    diagnoses = pd.DataFrame(rows_diag)
    prescriptions = pd.DataFrame(rows_presc)

    builder.load_mimic(admissions, diagnoses, prescriptions)
    return builder.build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Graph Builder Smoke Test ===")

    graph = make_synthetic_graph(200, n_diseases=40, n_drugs=60)
    logger.info("Node counts: %s", graph["node_counts"])
    logger.info("Edge types: %s", list(graph["edges"].keys()))

    builder = PatientDrugDiseaseGraphBuilder()
    path = builder.save(graph)
    loaded = PatientDrugDiseaseGraphBuilder.load(path)
    assert len(loaded["edges"]) > 0, "No edges loaded"
    logger.info("=== PASS ===")
