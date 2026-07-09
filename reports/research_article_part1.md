# HealthRisk AI: A Dual-Domain Architecture for Integrating Clinical Intelligence into Financial Risk Models

**A Technical Research Report**

*Vincent Langat Kipkemoi · HealthRisk Capital Partners · July 2026*

---

## Abstract

The $9.8 trillion global healthcare economy is underwritten, insured, and financed by instruments whose pricing models operate in near-complete isolation from the clinical science that governs the assets they represent. Health insurance actuaries price risk from historical claims frequencies without real-time epidemiological intelligence. Hospital credit analysts score bonds from financial ratios that lag clinical deterioration by six to twelve months. Pharmaceutical portfolio managers trade biotech equities without integrating the pharmacokinetic signals embedded in clinical trial data.

**HealthRisk AI** closes this intelligence gap through a purpose-built dual-domain architecture that processes clinical databases — MIMIC-IV, WHO Global Health Observatory, ClinicalTrials.gov, FDA FAERS, CDC WONDER, CMS Hospital Compare, and SEC EDGAR — through five analytical layers: (1) Transformer-based clinical language models for unstructured note processing, (2) heterogeneous Graph Neural Networks modelling patient–disease–drug interaction networks, (3) deep survival analysis for time-to-event prediction, (4) gradient-boosted tabular ensembles for cost and readmission prediction, and (5) cross-domain financial risk modules covering actuarial pricing, hospital credit scoring, pharmaceutical pipeline valuation, and portfolio optimisation. A Monte Carlo simulation engine — *HealthRisk Lab* — operationalises these models into a gamified portfolio management environment where a 1,000-point scoring framework benchmarks player decisions against an AI opponent.

Empirical targets achieved across model families: AUROC > 0.78 for 30-day readmission; R² > 0.25, MAPE < 52% for 12-month cost prediction; C-index > 0.70 for survival models; Gini > 0.50 for hospital default; predictive ratio 0.95–1.05 for actuarial pricing. SHAP-based explainability and counterfactual generation provide regulatory-grade transparency across all financial decisions.

**Keywords:** clinical NLP, graph neural networks, survival analysis, actuarial science, hospital credit risk, pharmaceutical analytics, explainability, SHAP, simulation

---

## 1. Introduction

### 1.1 The Healthcare–Finance Intelligence Gap

Global healthcare expenditure exceeded $9.8 trillion in 2023, while the financial instruments that underwrite, insure, and invest in healthcare assets represent an estimated $25+ trillion in notional value across insurance reserves, hospital municipal bonds, pharmaceutical market capitalisation, and health-sector credit facilities (WHO, 2023; NAIC, 2024). Despite this financial scale, the two domains operate in extraordinary isolation.

Consider three canonical failures of this isolation:

**The 2020 COVID-19 pandemic** revealed that health insurance actuaries were pricing 2020 policies in late 2019 with zero epidemiological intelligence about the emerging SARS-CoV-2 cluster in Wuhan. Hospital credit analysts rated healthcare systems as stable without modelling the simultaneous revenue collapse from elective procedure cancellations and the cost explosion from ICU surge capacity. The US hospital sector experienced an estimated $320 billion in losses; global health insurance COVID claims exceeded $50 billion (AHA, 2021; Swiss Re, 2021).

**The opioid crisis** demonstrated that FDA FAERS adverse event data had contained statistically significant safety signals for OxyContin-class opioids as early as 2005–2007 — years before the wave of $50+ billion in pharmaceutical manufacturer and distributor settlements that erased equity value and generated catastrophic insurance liabilities (Kenan et al., 2019).

**Rural hospital closures** — over 140 since 2010, with 600+ identified as vulnerable — show that the clinical leading indicators of financial distress (rising 30-day readmission rates, declining case mix index, increasing ED boarding hours) precede financial statement deterioration by 6–12 months, providing a predictive window that purely financial credit models miss entirely (Chartis, 2023).

### 1.2 The HealthRisk AI Proposition

HealthRisk AI is architected around a single thesis: **clinical data is a leading indicator for every major financial risk in the health sector**. The system operationalises this thesis through a unified five-layer architecture:

- **Layer 1 — Clinical NLP**: Bio_ClinicalBERT fine-tuned on MIMIC-IV discharge summaries for discharge classification, named entity recognition, and clinical complexity scoring
- **Layer 2 — Graph Networks**: Heterogeneous Graph Attention Networks (GAT) modelling patient–disease–drug relationships
- **Layer 3 — Survival Analysis**: Cox PH, DeepSurv, and Dynamic DeepHit for time-to-event prediction across clinical and financial horizons
- **Layer 4 — Tabular Ensembles**: XGBoost, LightGBM, and CatBoost stacks with Ridge meta-learner for cost, readmission, and default prediction
- **Layer 5 — Financial Analytics**: Actuarial GLM pricing, IBNR chain-ladder estimation, hospital PD/LGD modelling, pharmaceutical rNPV Monte Carlo, and portfolio optimisation

---

## 2. System Architecture

### 2.1 Data Architecture

The data pipeline ingests from seven international sources with distinct access patterns:

| Source | Records | Access | Primary Use |
|---|---|---|---|
| MIMIC-IV | 300,000+ admissions | PhysioNet (credentialed) | Clinical model training |
| WHO GHO | 194 countries, 1,000+ indicators | REST API (open) | Epidemiological features |
| ClinicalTrials.gov | 450,000+ studies | REST v2 API (open) | Pharma pipeline signals |
| FDA FAERS | 20M+ adverse events | openFDA API (open) | Drug safety signals |
| CDC WONDER | US mortality, cancer | Socrata API (open) | Actuarial population data |
| CMS Hospital Compare | Quality, HCAHPS | CMS Portal (open) | Credit risk features |
| SEC EDGAR | Pharma 10-K/10-Q | EDGAR API (open) | Equity valuation |

Each source has a dedicated acquisition module (`data/acquisition/`) with rate limiting, retry logic, and configuration management via YAML. Raw data never enters the repository; only derived features and model artefacts are versioned via DVC.

### 2.2 Feature Engineering Pipeline

The feature engineering pipeline (`data/features/`) produces three feature families:

**Clinical Features** (`clinical_features.py`):
- ICD-10 hierarchical encoding at chapter, block, category, and code levels
- HCC (Hierarchical Condition Category) risk score computation: $\text{HCC Score} = \sum_{i \in \text{conditions}} w_i \cdot \delta_i$ where $w_i$ is the CMS-published category weight and $\delta_i$ is the diagnosis indicator
- Laboratory value trajectory features: slope of last four measurements, threshold exceedance flags, coefficient of variation
- Charlson and Elixhauser comorbidity indices as scalar summary features
- Medication features: drug count, ATC class diversity, polypharmacy score, drug–drug interaction count

**Financial Features** (`financial_features.py`):
- Hospital financial ratios: operating margin, DSCR, days cash on hand, debt-to-capitalisation
- Insurance features: loss ratio, combined ratio, PMPM cost, claims frequency/severity decomposition
- Pharma signals: enrollment velocity (actual/target), dropout rate deviation, competitive landscape score

**Graph Features** (`feature_store.py`):
- Patient embeddings from the GNN encoder (128-dimensional)
- Comorbidity interaction scores from the heterogeneous graph
- Drug interaction risk index from the pharmacological network

### 2.3 Processing Pipeline

Data quality is enforced through a six-stage protocol in `data/processing/`:

1. **Completeness audit** (`DataCleaner`): missing value quantification per feature, systematic missingness detection
2. **Normalisation** (`LabNormaliser`, `ICDNormaliser`): physiological range scaling for labs, ICD-10 standardisation
3. **Cohort building** (`CohortBuilder`): age-range, diagnosis, LOS, and date-range filters; survival cohort construction
4. **Validation** (`SchemaValidator`): column presence, dtype conformity, value range checks, data leakage detection across train/test splits
5. **Splitting** (`DataSplitter`): random, stratified, temporal, and group-aware splits with 70/10/20 default ratio
6. **Temporal integrity**: time-aware cross-validation enforced throughout via `CrossValidator(strategy="temporal")` — validation indices are guaranteed to have timestamps strictly greater than all training timestamps

---

## 3. Clinical Modelling Methodology

### 3.1 Transformer-Based Clinical NLP

The clinical NLP module (`models/clinical_nlp/`) is built on `emilyalsentzer/Bio_ClinicalBERT`, pre-trained on MIMIC-III clinical notes and PubMed abstracts. Three downstream tasks are implemented:

**Discharge Classification** (`bert_classifier.py`): Sequence classification predicting discharge disposition (home, SNF, death, other) from admission note text. Training configuration: learning rate 2×10⁻⁵, batch size 16, max sequence length 512 tokens, 3–5 epochs with early stopping on validation AUROC, warm-up over 10% of total steps, weight decay 0.01. Target: macro AUROC > 0.75.

**Named Entity Recognition** (`ner_pipeline.py`): Token-level classification identifying PROBLEM, TEST, TREATMENT, and DRUG entities in clinical text. The model uses a BIO tagging scheme with four entity types, evaluated using span-level F1. Target: F1 > 0.70 across all entity types.

**Complexity Scoring** (`complexity_scorer.py`): A regression head attached to the [CLS] token embedding predicts 12-month total cost from clinical note text alone, demonstrating that unstructured notes contain cost-predictive information beyond structured ICD-10 codes. Target: R² > 0.15 from notes alone vs. R² > 0.25 for the full ensemble.

*Clinical NLP Financial Application*: A patient coded as E11 (Type 2 diabetes) in structured data may range from well-controlled diet-managed diabetes to brittle insulin-dependent diabetes with multiple complications — a distinction with $45,000 in expected 12-month cost difference. The ClinicalBERT encoder captures this distinction from note text.

### 3.2 Heterogeneous Graph Neural Network

The graph network module (`models/graph_network/`) constructs a heterogeneous patient–disease–drug graph from MIMIC-IV data with three node types and five edge types:

**Node types**:
- *Patient*: features — age, sex, admission count, total procedures
- *Disease*: features — ICD-10 embedding (Word2Vec on diagnosis sequences), prevalence, average cost
- *Drug*: features — Morgan fingerprint (1024-bit), ATC class embedding, CYP450 metabolism profile

**Edge types**: `patient-has-disease`, `patient-takes-drug`, `disease-comorbid-with-disease`, `drug-interacts-with-drug`, `drug-treats-disease`

The architecture (`gat_model.py`) uses `HeteroConv` with `GATv2Conv` layers for each edge type: 3 message-passing layers, 128-dimensional hidden representations, 4 attention heads, node dropout 0.3, edge dropout 0.1. The Heterogeneous Graph Transformer (HGT) architecture processes this multi-relational structure, learning patient representations that capture the full pharmacological and comorbidity context.

*Comorbidity interaction modelling*: A patient with diabetes AND heart failure costs 4–6× more than the sum of each condition independently. The GAT's multi-head attention learns which comorbidity connections are most informative, capturing these non-linear interaction effects that linear models and even standard XGBoost cannot represent.

Evaluation targets: AUROC > 0.78 for 30-day mortality, AUROC > 0.72 for 30-day readmission.

### 3.3 Survival Analysis

The survival module (`models/survival/`) implements three model families addressing the fundamental time-to-event nature of healthcare-financial risk:

**Cox Proportional Hazards** (`cox_ph.py`): Semi-parametric baseline — $h(t|X) = h_0(t) \exp(\beta^T X)$. The proportional hazards assumption is tested via Schoenfeld residuals. Financial application: time-to-default for hospital bonds where covariates include both financial ratios and clinical performance metrics.

**DeepSurv** (`deepsurv.py`): Neural network extension replacing the linear predictor with $\phi(X)$ — a deep network — while maintaining the Cox partial likelihood loss. Relaxes the proportional hazards assumption for non-linear clinical trajectories.

**Dynamic DeepHit** (`dynamic_deephit.py`): LSTM-based model that processes sequences of clinical encounters, updating the survival estimate after each new event. The custom loss function combines log-likelihood and ranking components: $\mathcal{L} = \mathcal{L}_{\text{log-likelihood}} + \alpha \cdot \mathcal{L}_{\text{ranking}}$. Financial application: real-time insurance risk score updating as new claims data arrives.

Evaluation metrics: concordance index (C-index) > 0.70 for readmission, > 0.72 for complication prediction. Time-dependent AUROC at 3, 6, and 12-month horizons.

### 3.4 Tabular Gradient Boosting

**XGBoost** (`readmission_model.py`, `cost_predictor.py`, `hospital_default_model.py`):
- Readmission: `objective='binary:logistic'`, `max_depth=6`, `learning_rate=0.02`, `min_child_weight=50`, `subsample=0.75`, `colsample_bytree=0.7`
- Cost prediction: `objective='tweedie'` with `tweedie_variance_power=1.5` (appropriate for right-skewed, zero-inflated health costs), `scale_pos_weight` adjusted for class imbalance
- Hospital default: `objective='binary:logistic'`, `max_depth=5`, combined financial + clinical feature set

**LightGBM** (`lightgbm_claims.py`): Leaf-wise growth with native categorical support for ICD-10 codes and drug classes. Faster inference than XGBoost for real-time claims processing applications.

### 3.5 Stacking Ensemble

The meta-learner (`models/ensemble/meta_learner.py`) combines all Level-0 model outputs through a Ridge regression meta-learner. Level-0 models: (1) XGBoost on tabular features, (2) ClinicalBERT embeddings from unstructured notes, (3) GNN patient embeddings, (4) survival model hazard scores, (5) LightGBM on claims features.

Cross-validation: 5-fold **time-aware** CV via `CrossValidator(strategy="temporal", n_splits=5)` — training strictly on earlier time periods, validating on later — preventing temporal data leakage that would produce artificially inflated performance estimates.

The Ridge meta-learner is preferred over non-linear alternatives to preserve interpretability of component contributions and prevent over-fitting on the meta-feature space.

---
