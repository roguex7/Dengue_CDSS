# 🦟 Dengue Clinical Decision Support System (CDSS)

> **An AI-powered clinical tool for dengue risk stratification, platelet trajectory forecasting, and WHO 2009 triage — deployed on Render via Docker.**

<img width="2614" height="1459" alt="CDSS_Infounnamed" src="https://github.com/user-attachments/assets/8ff0cee9-e808-4671-8d69-6339dbda2c80" />

---

## 🏆 Performance at a Glance

| Metric | Value |
|---|---|
| 🎯 Risk Classifier Accuracy | **99.8%** |
| 📊 AUC-ROC | **0.9996** |
| 🔬 Sensitivity | **99.77%** |
| 🛡️ Specificity | **100.00%** |
| 📈 Forecast R² | **0.9953** |
| 📉 Forecast MAE | **2,515 cells/µL** |
| 👥 Training Cohort | **2,455 patients** |

<img width="4176" height="895" alt="06_model_metrics_summary" src="https://github.com/user-attachments/assets/b6bf3ea5-bfa8-4d2c-9bce-53e043e39599" />

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [ML Models](#ml-models)
- [OCR Pipeline](#ocr-pipeline)
- [Clinical Logic](#clinical-logic)
- [Installation](#installation)
- [Deployment](#deployment)
- [File Structure](#file-structure)
- [Evaluation & Testing](#evaluation--testing)
- [WHO 2009 Classification](#who-2009-classification)

---

## Overview

The **Dengue CDSS** is a Streamlit-based clinical web application that assists physicians in:

1. **Risk Stratification** — a 300-tree Random Forest classifier scores a patient's probability of dengue with warning signs or severe dengue (WHO 2009 Group B/C) in real time, with 95% confidence intervals derived from tree-ensemble variance.
2. **Platelet Trajectory Forecasting** — a Gradient Boosting regressor predicts next-day platelet counts from longitudinal Day-1 → Day-2 data, enabling early detection of deterioration or recovery.
3. **WHO 2009 Triage** — a rule-based engine assigns every patient to Group A (outpatient), Group B (inpatient monitoring), or Group C (emergency) based on 11 WHO clinical criteria.
4. **OCR-Powered Lab Report Ingestion** — lab PDFs or photographs are automatically parsed with Tesseract OCR → regex pattern matching → clinical plausibility validation → form auto-fill.
5. **Priority Serial Alerts** — real-time dashboard for detecting critical inter-report changes across longitudinal lab panels.
6. **Signed PDF Export** — generates structured clinical reports with embedded charts for physician documentation.

The entire pipeline runs inside a **Docker container** with models baked at build time, achieving cold-start latency under 0.5 seconds on Render.

---

## Features

- **Dual Data Entry** — manual form (demographics, vitals, CBC, extended labs) or bulk OCR upload of PDF/JPG lab reports
- **RF Risk Classifier** — 21-feature Random Forest with 95% confidence intervals via tree ensemble percentile calculation
- **Gradient Boosting Regressor** — 17-feature platelet trajectory forecast (Day 1 → Day 2 → Day 3)
- **OOD / Mahalanobis Detection** — flags when patient data falls outside the training distribution (P5 check)
- **SHAP Feature Explainer** — per-patient feature contribution analysis (when `shap` is installed)
- **Holliday-Segar Fluid Rate Calculator** — weight-based IV fluid rate in mL/hr
- **Seasonal Risk Mapping** — monsoon/post-monsoon epidemiological risk modifier
- **WHO Group A/B/C Triage Badge** — colour-coded (green/amber/red) with clinical rationale
- **Serial Reports Panel** — longitudinal management of multiple lab entries per patient visit
- **Multi-Format Export** — Signed PDF report (fpdf2) with Matplotlib/Plotly visuals

<img width="1920" height="1047" alt="Data Pipeline Diagram (Large)" src="https://github.com/user-attachments/assets/5fe17fc8-8fc7-4082-b5dd-a36c72160857" />

---

## Architecture

The system operates in four logical phases:

```
┌──────────────────────────────────────────────────────────────┐
│ PHASE 1 — DATA ACQUISITION                                   │
│  Manual Form Entry   ↔   Lab Report PDF / JPG Upload         │
├──────────────────────────────────────────────────────────────┤
│ PHASE 2 — OCR EXTRACTION PIPELINE                            │
│  OpenCV Preprocessing → Tesseract OCR → Regex Match →        │
│  Clinical Plausibility Check (PHYS_BOUNDS) → Auto-fill       │
├──────────────────────────────────────────────────────────────┤
│ PHASE 3 — FEATURE ENGINEERING + ML INFERENCE (Parallel)     │
│  Derive MAP / Shock Index / Holliday-Segar / Season Risk →   │
│  OOD Mahalanobis Check → RF Probability + 95% CI →           │
│  GBM Platelet Forecast → WHO 2009 Rule-based Triage          │
├──────────────────────────────────────────────────────────────┤
│ PHASE 4 — OUTPUT GENERATION                                  │
│  Comprehensive Dashboard → Serial Alerts → SHAP → PDF Export │
└──────────────────────────────────────────────────────────────┘
```

<img width="1920" height="1047" alt="System Architecture Diagram (Large)" src="https://github.com/user-attachments/assets/1a897fb3-0e43-41e5-9b0d-09f41c4db734" />

<img width="2614" height="1459" alt="DataPipeline_Infounnamed" src="https://github.com/user-attachments/assets/7b3cb1bd-637d-402b-912e-7393654f4633" />

<img width="2614" height="1459" alt="SysARchitecture_Infounnamed" src="https://github.com/user-attachments/assets/bd1cf79f-61a4-4c91-9891-94be4f190aee" />

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| ML — Classifier | RandomForestClassifier (scikit-learn) |
| ML — Regressor | GradientBoostingRegressor (scikit-learn) |
| Model Serialisation | joblib (compress=3) |
| Model Explainability | SHAP (TreeExplainer) |
| OCR Engine | Tesseract-OCR 5.x via pytesseract |
| Image Preprocessing | OpenCV-headless, Pillow |
| PDF Parsing | pdf2image, pypdf |
| PDF Report Generation | fpdf2 |
| Data Visualisation | Plotly, Matplotlib |
| Data Manipulation | Pandas, NumPy |
| Containerisation | Docker (python:3.11 base) |
| Cloud Deployment | Render (Docker runtime) |
| System Dependencies | tesseract-ocr, poppler-utils, libgl1 |

---

## Dataset

| Property | Value |
|---|---|
| File | `dengue_data_cleaned_debug.csv` |
| Total Records | **2,455 patients** |
| High Risk (Group B/C) | 2,197 — **89.5%** |
| Low Risk (Group A) | 258 — **10.5%** |
| Sex Split | 50.1% Female / 49.9% Male |
| Median Age — Low Risk | 26 years |
| Median Age — High Risk | 31 years |
| Seasonal Peak | **June** (Monsoon: Jun–Sep) |
| Physiological Bounds | Enforced via `PHYS_BOUNDS` dict before any computation |

**Core CSV columns:** `Platelet (cells/cu.mm)`, `Haemoglobin (gm/Dl)`, `Red Blood Cell Count (millions/cu.mm)`, `Hematocrit (Packed Cell Volume) (%)`, `Age`, `Sex`, `Date of Test & Time of Test`, `Symptoms`

Columns absent from the CSV are generated with clinically realistic synthetic distributions (WHO reference ranges + Indian dengue cohort studies) by `_build_synthetic_columns()`.

<img width="1959" height="2022" alt="01_risk_distribution" src="https://github.com/user-attachments/assets/b710a14b-e780-4922-81cc-f76be6262036" />

<img width="1979" height="2022" alt="04_gender_distribution" src="https://github.com/user-attachments/assets/6cc28ed4-fb46-49be-a4ae-4533ac8d746e" />

<img width="3030" height="1854" alt="07_platelet_distribution_by_risk" src="https://github.com/user-attachments/assets/14e598d2-b392-4f74-ad3d-1ae566b64508" />

<img width="2760" height="1874" alt="10_age_distribution_by_risk" src="https://github.com/user-attachments/assets/1fb405ea-689d-4785-a9b5-d0fab22d6097" />

<img width="3681" height="1910" alt="11_seasonal_risk_monthly" src="https://github.com/user-attachments/assets/2719664a-a926-4339-8325-88ebaeb4e606" />

---

## ML Models

### 1. Risk Classifier — Random Forest

```python
RandomForestClassifier(
    n_estimators    = 300,
    max_depth       = 12,
    min_samples_leaf = 4,
    max_features    = 'sqrt',
    class_weight    = 'balanced',   # Handles 90/10 class imbalance
    n_jobs          = -1,
    random_state    = 42,
)
```

**21 Classification Features (clf_features):**

| Category | Features |
|---|---|
| Core CBC | `Platelet (cells/cu.mm)`, `Haemoglobin (gm/Dl)`, `Red Blood Cell Count (millions/cu.mm)`, `Hematocrit (Packed Cell Volume) (%)` |
| Demographics | `Age`, `Sex_Code` |
| Haemodynamics | `Shock_Index`, `Pulse_Pressure` |
| Symptoms | `Has_Fever`, `Has_Headache`, `Has_Pain`, `Has_Vomit`, `Has_Bleeding` |
| Extended Clinical | `WBC`, `AST`, `INR`, `SpO2`, `GCS`, `Has_Pleural_Effusion`, `Ascites_Grade` |
| Temporal | `Season_Risk` |

**Top Feature Importances (MDI):**
1. Platelet Count — **0.282**
2. Shock Index — **0.267**
3. Pleural Effusion — **0.145**
4. INR — **0.124**
5. AST — **0.053**

<img width="3030" height="2607" alt="09_feature_importance_full" src="https://github.com/user-attachments/assets/2c6875f7-3250-4bca-bd9b-4b57c0e5456c" />

<img width="1503" height="1682" alt="02_confusion_matrix" src="https://github.com/user-attachments/assets/3abdce39-37a4-4986-8fef-798d1d9a6bd9" />

<img width="2206" height="2004" alt="14_roc_curve" src="https://github.com/user-attachments/assets/1887f437-6f5a-4522-8b54-31447b126d44" />

<img width="3392" height="3233" alt="08_cbc_correlation_heatmap" src="https://github.com/user-attachments/assets/b45be14f-5bdb-49a3-9d76-9623d70133c5" />


### 2. Platelet Forecast — Gradient Boosting Regressor

```python
GradientBoostingRegressor(
    n_estimators    = 300,
    learning_rate   = 0.08,
    max_depth       = 5,
    subsample       = 0.85,
    min_samples_leaf = 4,
    random_state    = 42,
)
```

**17 Regression Features (reg_features):**
`Day1_Platelets`, `Day2_Platelets`, `Delta_D1_D2`, `Haemoglobin`, `RBC`, `Hematocrit`, `Age`, `Sex_Code`, `Shock_Index`, `Pulse_Pressure`, `Has_Fever`, `Has_Vomit`, `Has_Pain`, `Has_Bleeding`, `AST`, `INR`, `Season_Risk`

**Target:** `Day3_Platelets` — synthetically generated from observed Day 2 platelet counts using recovery momentum modelling (recovery p=0.58, declining p=0.42, volatility uniform 0.08–0.28, Gaussian noise σ=1,800).

<img width="2441" height="2325" alt="03_forecast_actual_vs_predicted" src="https://github.com/user-attachments/assets/fbb10385-f64c-470e-984d-e4fb31240c0a" />

<img width="3090" height="1972" alt="13_platelet_trajectory" src="https://github.com/user-attachments/assets/d370277b-e7d9-4dd3-8dbd-f0fafd983282" />

---

## OCR Pipeline

```
Lab PDF / Image Upload
        │
        ▼
OpenCV Preprocessing
  ├── Grayscale conversion
  ├── Adaptive thresholding
  └── Deskewing / denoising
        │
        ▼
Tesseract OCR (language: eng)
  └── Binary path auto-resolved:
      Docker /usr/bin → Homebrew → Windows → PATH fallback
        │
        ▼
Regex Pattern Matching
  └── Extracts: patient name, date, platelet, Hb, RBC, Hct, WBC
        │
        ▼
Clinical Plausibility Check (PHYS_BOUNDS)
  └── Rejects physiologically impossible values
        │
        ▼
Auto-fill Form + Confidence Scoring
  └── User reviews and accepts / corrects extracted values
```

**Supported input formats:** PDF (multi-page via pdf2image + poppler), JPG, PNG, scanned photographs

---

## Clinical Logic

### WHO 2009 Multi-Criterion Labelling

A patient is classified **High Risk (Label = 1)** if **any single criterion** is met:

| Domain | Criterion | Threshold |
|---|---|---|
| CBC | Platelet Count | < 100,000 cells/µL |
| CBC | Hematocrit | > 50% |
| CBC | Haemoglobin | < 7 g/dL |
| Haemodynamics | Shock Index (HR ÷ SBP) | > 0.9 |
| Haemodynamics | Pulse Pressure | ≤ 20 mmHg |
| Liver | AST | ≥ 500 IU/L |
| Coagulation | INR | ≥ 1.5 |
| Imaging | Pleural Effusion | Present |
| Imaging | Ascites Grade | ≥ 2 |
| Vitals | SpO2 | < 93% |
| Vitals | GCS | < 13 |

<img width="3030" height="2345" alt="12_who_criteria_frequency" src="https://github.com/user-attachments/assets/ac444509-00eb-4399-8905-e35ab3cf6e02" />

<img width="3030" height="1854" alt="15_shock_index_by_risk" src="https://github.com/user-attachments/assets/0630e965-e190-4da3-84c8-cbac2eed8492" />

<img width="2991" height="2135" alt="05_hct_platelet_relationship" src="https://github.com/user-attachments/assets/9b8fc18b-5360-4ca9-9602-96b35e05c8f0" />

### Seasonal Risk Scores

| Months | Season | Risk Score |
|---|---|---|
| Dec, Jan, Feb | Off-season | 0 |
| Mar, Apr, May | Pre-Monsoon | 1 |
| Oct, Nov | Post-Monsoon | 2 |
| Jun, Jul, Aug, Sep | Monsoon Peak | **3** |

### Derived Haemodynamic Indices

```
Shock Index    = Heart Rate ÷ Systolic BP          (clipped 0.2–3.5)
Pulse Pressure = Systolic BP − Diastolic BP
MAP            = Diastolic BP + (Pulse Pressure ÷ 3)
```

---

## Installation

### Local Setup

```bash
# Clone
git clone https://github.com/<your-org>/dengue-cdss.git
cd dengue-cdss

# System dependencies (Ubuntu/Debian)
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils libgl1 libglib2.0-0

# Python dependencies
pip install -r requirements.txt

# Train models (first run — ~30 seconds, then cached as .pkl)
python train_model.py

# Launch
streamlit run app.py
```

### Verify OCR Dependencies

```bash
python verify_ocr.py
# Expect: ✅ pytesseract, ✅ OpenCV, ✅ Pillow, ✅ pdf2image
```

### Run Tests

```bash
python test_suite.py       # 12 automated ML pipeline tests
python evaluate_model.py   # Full evaluation with cross-validation
```

---

## Deployment

### Docker (Local or Any Host)

```bash
docker build -t dengue-cdss .
docker run -p 8501:8501 dengue-cdss
# Open: http://localhost:8501
```

### Render (render.yaml)

Deployment is zero-config — push to GitHub, Render detects the `render.yaml` and builds the Docker image. Models are pre-baked during `docker build`, so there is **no training delay at runtime**.

```yaml
services:
  - type: web
    name: denguecdss
    runtime: docker
    dockerfilePath: ./Dockerfile
```

**Docker Build Strategy (cold-start optimisation):**

| Approach | Cold-start Time |
|---|---|
| Train at startup (old) | 45–90 seconds |
| Bake PKL into image (current) | **< 0.5 seconds** |

Docker layer caching ensures retraining only occurs when `dengue_data_cleaned_debug.csv` or `train_model.py` change.

---

## File Structure

```
dengue-cdss/
├── app.py                          # Main Streamlit app (5,144 lines)
│                                   #  UI, OCR, session state, analysis engine
├── train_model.py                  # Model training — source of truth for features
│                                   #  PHYS_BOUNDS, synthetic columns, WHO labelling
├── evaluate_model.py               # Full evaluation (hold-out + 5-fold CV + SHAP)
├── generate_graphs.py              # Dashboard visualisation generation
├── test_suite.py                   # 12 automated ML pipeline tests
├── verify_ocr.py                   # OCR dependency health check
├── dengue_data_cleaned_debug.csv   # Patient dataset (2,455 records)
├── Dockerfile                      # Bakes models at build time
├── render.yaml                     # Render.com deployment config
├── requirements.txt                # Python packages
├── packages.txt                    # OS packages (tesseract, poppler)
└── models/                         # Auto-generated at build time
    ├── classifier.pkl              # Serialised RF classifier
    ├── regressor.pkl               # Serialised GBM regressor
    └── features.pkl                # clf_features + reg_features lists
```

---

## Evaluation & Testing

### 12-Test Suite Summary

```bash
python test_suite.py
```

| # | Test Name | What It Verifies |
|---|---|---|
| T1 | Training smoke test | Models not None, feature lists not empty |
| T2 | Feature contract | clf_features ⊆ app.py risk_input keys exactly |
| T3 | Classifier inference shape | predict_proba returns (1, 2), prob ∈ [0, 1] |
| T4 | Regressor inference shape | predict returns (1,), forecast ≥ 0 |
| T5 | Clinical sanity | Severe patient probability > mild patient |
| T6 | All-zero edge case | No crash, valid probability output |
| T7 | Extreme values | Probability within [0, 1] at extreme CBC values |
| T8 | Partial input | App safety loop zero-fill pattern works |
| T9 | Tree CI contract | 95% CI bounds in [0, 1], width valid |
| T10 | Regressor direction | Recovering trajectory → higher forecast than declining |
| T11 | Seasonal modifier | Peak season risk ≥ off-season risk |
| T12 | No duplicates | No duplicate feature names in either list |

### Cross-Validation Results (5-Fold Stratified)

| Metric | Mean ± Std |
|---|---|
| AUC | 0.9996 ± ~0.0001 |
| Accuracy | ~0.998 ± ~0.001 |
| F1 | ~0.999 ± ~0.001 |

---

## WHO 2009 Classification

| Group | Risk Level | Clinical Action |
|---|---|---|
| **A** 🟢 | Low — No warning signs | Oral hydration, outpatient monitoring |
| **B** 🟡 | Moderate — Warning signs | Inpatient admission, IV fluids, serial CBC |
| **C** 🔴 | Severe — Organ impairment / shock | ICU, emergency fluid resuscitation |

The CDSS assigns every patient to exactly one group at inference time, displays a colour-coded badge, and generates a priority serial alert if the risk category has worsened since the previous report.

<img width="3603" height="4237" alt="Work_Flow_NotebookLM Mind Map" src="https://github.com/user-attachments/assets/c169bcf5-1d06-4671-bda0-3a1301b89f70" />

<img width="1920" height="1047" alt="Workflow Diagram (Large)" src="https://github.com/user-attachments/assets/50bd6f6c-f43a-4589-8de2-34950ce925c9" />

<img width="2752" height="1492" alt="WorkFlowInfo" src="https://github.com/user-attachments/assets/76d407f1-ed16-436f-9d16-2ebb85092580" />

---

## Contributing

1. Fork the repository
2. Run `python test_suite.py` — all 12 tests must pass
3. If adding features to `clf_features`, update `APP_RISK_INPUT_KEYS` in `test_suite.py` and the `risk_input` dict in `app.py`
4. Open a pull request with evaluation results from `evaluate_model.py`

---

## License

This project is for research and educational purposes. Clinical deployment requires validation by qualified medical professionals. AI outputs are decision-support tools, not diagnostic replacements.

---

## Citation

- WHO. *Dengue: Guidelines for Diagnosis, Treatment, Prevention and Control.* WHO Press, 2009.
- Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR, 2011.
- Dataset: Anonymised dengue patient cohort — 2,455 records (Regional Indian dengue studies).
