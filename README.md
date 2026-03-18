# 🦟 Dengue CDSS — Clinical Decision Support System

> **AI-powered dengue severity triage, platelet trajectory forecasting, and automated OCR lab-report ingestion — built for frontline clinicians in resource-constrained settings.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://www.docker.com/)
[![AUC](https://img.shields.io/badge/AUC--ROC-0.9996-brightgreen)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-99.8%25-brightgreen)]()
[![Tests](https://img.shields.io/badge/Tests-12%2F12-brightgreen)]()

---

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — System Overview Infographic           ║
  ║  Source: Diagrams.pdf  →  Page 16                            ║
  ║  Title: "Inside the Dengue CDSS"                             ║
  ║  Replace this comment with:                                  ║
  ║    ![System Overview](docs/img/diagrams_p16.png)             ║
  ╚══════════════════════════════════════════════════════════════╝
-->

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Performance Metrics](#-key-performance-metrics)
3. [System Architecture](#-system-architecture)
4. [Dataset](#-dataset)
5. [ML Models](#-ml-models)
6. [Feature Engineering](#-feature-engineering)
7. [OCR Pipeline](#-ocr-pipeline)
8. [Clinical Logic — WHO 2009](#-clinical-logic--who-2009)
9. [File Reference](#-file-reference)
10. [Installation & Local Run](#-installation--local-run)
11. [Docker Deployment](#-docker-deployment)
12. [Render Deployment](#-render-deployment)
13. [Testing — 12-Point Contract](#-testing--12-point-contract)
14. [Model Evaluation](#-model-evaluation)
15. [WHO 2009 Classification Reference](#-who-2009-classification-reference)
16. [References](#-references)

---

## 🔬 Project Overview

The **Dengue Clinical Decision Support System (CDSS)** is a full-stack, AI-augmented clinical tool that assists physicians in:

- **Triaging** dengue patients into WHO 2009 Group A / B / C severity categories in real time
- **Predicting** the probability of warning signs or severe dengue (0–100%) using a 21-feature Random Forest classifier
- **Forecasting** the 24-hour platelet trajectory (Day 3 count) to anticipate critical platelet drops
- **Ingesting** physical lab reports automatically via a multi-stage OCR pipeline (PDF / JPG → structured CBC values)
- **Generating** signed, downloadable clinical PDF reports with charts, SHAP explanations, and triage guidance

The system is built on a **Streamlit** front end deployed inside a **Docker** container on **Render**. ML models are **baked into the Docker image at build time**, eliminating the 45–90 second cold-start penalty of training on spin-up. The result is a **< 0.5-second cold start** in production.

---

## 📊 Key Performance Metrics

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Algorithm Performance Dashboard       ║
  ║  Source: Diagrams.pdf  →  Page 6                             ║
  ║  Title: "Algorithm Performance & Data Transparency"          ║
  ║  Replace with: ![Metrics Bar](docs/img/diagrams_p06.png)     ║
  ╚══════════════════════════════════════════════════════════════╝
-->

| Metric | Value | Notes |
|---|---|---|
| **Training Patients** | 2,455 | Balanced cohort, Gujarat region |
| **Risk Classifier Accuracy** | **99.80%** | 20% hold-out test set (n=491) |
| **Sensitivity (True Positive Rate)** | **99.77%** | Only 1 false-negative in 491 cases |
| **Specificity (True Negative Rate)** | **100.00%** | Zero false-positives on test set |
| **PPV (Precision)** | High | See confusion matrix |
| **AUC-ROC** | **0.9996** | Near-perfect discrimination |
| **Forecast R²** | **0.9953** | Day-3 platelet prediction |
| **Forecast MAE** | **2,515 cells/µL** | Mean absolute error |
| **5-Fold CV AUC** | **0.9996 ± <0.001** | Stratified, no data leakage |

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Confusion Matrix                      ║
  ║  Source: Diagrams.pdf  →  Page 2                             ║
  ║  Stats: TN=52(100%) FP=0(0%) FN=1(0.2%) TP=438(99.8%)       ║
  ║  Replace with: ![Confusion Matrix](docs/img/diagrams_p02.png)║
  ╚══════════════════════════════════════════════════════════════╝
-->

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — ROC Curve                             ║
  ║  Source: Diagrams.pdf  →  Page 14                            ║
  ║  AUC = 0.9996, Youden's J at FPR=0.000, TPR=0.998           ║
  ║  Replace with: ![ROC Curve](docs/img/diagrams_p14.png)       ║
  ╚══════════════════════════════════════════════════════════════╝
-->

---

## 🏛️ System Architecture

The system operates through **four sequential phases**:

```
╔══════════════════════════════════════════════════════════════╗
║  PHASE 1 — INITIALISATION & DATA ACQUISITION                ║
║  Unit Preference Setup → Bulk OCR Upload  OR  Manual Entry  ║
╚══════════════════════════════╦═══════════════════════════════╝
                               ↓
╔══════════════════════════════════════════════════════════════╗
║  PHASE 2 — DATA INTEGRATION & FEATURE ENGINEERING           ║
║  MAP · Shock Index · Holliday-Segar · Seasonal Risk          ║
╚══════════════════════════════╦═══════════════════════════════╝
                               ↓
╔══════════════════════════════════════════════════════════════╗
║  PHASE 3 — THE INTELLIGENCE LAYER (Parallel Execution)      ║
║  WHO 2009 Triage · RF Risk % · SHAP · OOD · Serial Alerts   ║
╚══════════════════════════════╦═══════════════════════════════╝
                               ↓
╔══════════════════════════════════════════════════════════════╗
║  PHASE 4 — CLINICAL REVIEW & OUTPUT                         ║
║  Dashboard · Charts · Alerts · Signed PDF Export             ║
╚══════════════════════════════════════════════════════════════╝
```

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Clinical Data Pipeline (Flowchart)   ║
  ║  Source: Diagrams.pdf  →  Page 18                            ║
  ║  Shows all five pipeline stages with data-flow arrows        ║
  ║  Replace with: ![Pipeline](docs/img/diagrams_p18.png)        ║
  ╚══════════════════════════════════════════════════════════════╝
-->

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — System Architecture Mind Map          ║
  ║  Source: Diagrams.pdf  →  Page 17                            ║
  ║  Covers: Frontend · Data Pipeline · Analytical Engines ·     ║
  ║          Model Performance · Clinical Risk Factors ·          ║
  ║          Dataset Demographics                                 ║
  ║  Replace with: ![Arch Map](docs/img/diagrams_p17.png)         ║
  ╚══════════════════════════════════════════════════════════════╝
-->

### Technology Stack

| Layer | Component | Technology |
|---|---|---|
| **Frontend** | Dashboard, Data-Entry Form, Serial Alerts Panel | Streamlit 1.x + custom CSS |
| **State** | Session data (reports A–E, metrics, unit prefs) | `st.session_state` (SS_Reports, SS_Metrics) |
| **OCR** | PDF/Image → CBC entities | Tesseract 5 + OpenCV + PyPDF + pdf2image |
| **Classifier** | Binary risk probability | RandomForestClassifier (scikit-learn) |
| **Regressor** | Day-3 platelet forecast | GradientBoostingRegressor (scikit-learn) |
| **Trajectory** | 3-point platelet trendline | OLS LinearRegression (scikit-learn) |
| **Severity Engine** | WHO Group A/B/C triage | Rule-based clinical logic |
| **Explainability** | Per-patient feature attribution | SHAP TreeExplainer |
| **OOD Detection** | Out-of-distribution flag | Mahalanobis distance (TRAIN_STATS) |
| **Reporting** | Signed clinical PDF | FPDF2 + Matplotlib + Plotly |
| **Container** | Reproducible deployment | Docker (python:3.11 base) |
| **Hosting** | Cloud deployment | Render.com (Docker runtime) |

---

## 📂 Dataset

| Property | Value |
|---|---|
| **File** | `dengue_data_cleaned_debug.csv` |
| **Total Records** | 2,455 patient profiles |
| **Gender** | 50.1% Female (n=1,230) / 49.9% Male (n=1,225) |
| **Risk Distribution** | 89.5% High Risk (n=2,197) / 10.5% Low Risk (n=258) |
| **Median Age — Low Risk** | 26 years |
| **Median Age — High Risk** | 31 years |
| **Geographic Context** | Gujarat / Western India |
| **Temporal Coverage** | Multi-year, includes monsoon peak and off-season |

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Risk Distribution (Pie Chart)         ║
  ║  Source: Diagrams.pdf  →  Page 1                             ║
  ║  89.5% High Risk · 10.5% Low Risk · n=2,455                  ║
  ║  Replace with: ![Risk Dist](docs/img/diagrams_p01.png)        ║
  ╚══════════════════════════════════════════════════════════════╝
-->

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Gender Distribution (Pie Chart)       ║
  ║  Source: Diagrams.pdf  →  Page 4                             ║
  ║  Replace with: ![Gender](docs/img/diagrams_p04.png)           ║
  ╚══════════════════════════════════════════════════════════════╝
-->

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Age Distribution by Risk Category     ║
  ║  Source: Diagrams.pdf  →  Page 10                            ║
  ║  Violin plot: Low Risk median 26 yrs, High Risk median 31    ║
  ║  Replace with: ![Age Dist](docs/img/diagrams_p10.png)         ║
  ╚══════════════════════════════════════════════════════════════╝
-->

### Core CSV Columns

| Column | Type | Unit | Physiological Clip (v3.1) |
|---|---|---|---|
| `Platelet (cells/cu.mm)` | Numeric | cells/µL | 1,000 – 900,000 |
| `Haemoglobin (gm/Dl)` | Numeric | g/dL | 2.0 – 22.0 |
| `Red Blood Cell Count (millions/cu.mm)` | Numeric | M/µL | **1.0 – 10.0** *(unit-error fix)* |
| `Hematocrit (Packed Cell Volume) (%)` | Numeric | % | 5.0 – 70.0 |
| `Age` | Numeric | years | 0 – 120 |
| `Sex` | Categorical | Male / Female | → `Sex_Code` 0/1 |
| `Date of Test & Time of Test` | DateTime | — | → `Season_Risk` |
| `Symptoms` | Free text | — | → 5 binary flags |

---

## 🤖 ML Models

### Model 1 — Risk Classifier (`models/classifier.pkl`)

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Feature Importance Bar Chart          ║
  ║  Source: Diagrams.pdf  →  Page 9                             ║
  ║  Top features by MDI: Platelet(0.282), Shock Index(0.267)... ║
  ║  Replace with: ![Feature Imp](docs/img/diagrams_p09.png)      ║
  ╚══════════════════════════════════════════════════════════════╝
-->

**Algorithm:** `RandomForestClassifier` (scikit-learn)

```python
RandomForestClassifier(
    n_estimators    = 300,       # 300 decision trees
    max_depth       = 12,        # Prevents overfitting deep splits
    min_samples_leaf = 4,        # Minimum 4 samples per leaf node
    max_features    = 'sqrt',    # sqrt(n_features) features per split
    class_weight    = 'balanced',# Handles 89.5/10.5 label imbalance
    n_jobs          = -1,        # Parallel training — all CPU cores
    random_state    = 42,
)
```

**Target:** `Dengue_Label` — binary (0 = No Warning Signs, 1 = Warning Signs or Severe Dengue)

**21 Input Features (`clf_features`):**

| # | Feature | Source |
|---|---|---|
| 1 | `Platelet (cells/cu.mm)` | CSV (CBC) |
| 2 | `Haemoglobin (gm/Dl)` | CSV (CBC) |
| 3 | `Red Blood Cell Count (millions/cu.mm)` | CSV (CBC) |
| 4 | `Hematocrit (Packed Cell Volume) (%)` | CSV (CBC) |
| 5 | `Age` | CSV |
| 6 | `Sex_Code` | Encoded from CSV |
| 7 | `Shock_Index` | Derived: HR / SBP |
| 8 | `Pulse_Pressure` | Derived: SBP − DBP |
| 9 | `Has_Fever` | Symptom flag |
| 10 | `Has_Headache` | Symptom flag |
| 11 | `Has_Pain` | Symptom flag |
| 12 | `Has_Vomit` | Symptom flag |
| 13 | `Has_Bleeding` | Symptom flag |
| 14 | `WBC` | Synthetic (bimodal) |
| 15 | `AST` | Synthetic (log-normal) |
| 16 | `INR` | Synthetic (log-normal) |
| 17 | `SpO2` | Synthetic (normal) |
| 18 | `GCS` | Synthetic (bimodal) |
| 19 | `Has_Pleural_Effusion` | Synthetic (Bernoulli) |
| 20 | `Ascites_Grade` | Synthetic (multinomial) |
| 21 | `Season_Risk` | Derived from date |

**Feature Importances (MDI — Mean Decrease in Impurity):**

| Rank | Feature | Score |
|---|---|---|
| 1 | Platelet Count | **0.282** |
| 2 | Shock Index | **0.267** |
| 3 | Pleural Effusion | 0.145 |
| 4 | INR | 0.124 |
| 5 | AST | 0.053 |
| 6 | Ascites Grade | 0.050 |
| 7 | Pulse Pressure | 0.015 |
| 8 | GCS | 0.015 |
| 9–21 | (WBC, Age, SpO2, Hct, Hb, RBC, Season Risk, Sex Code, symptom flags) | < 0.01 each |

---

### Model 2 — Platelet Forecast Engine (`models/regressor.pkl`)

**Algorithm:** `GradientBoostingRegressor` (scikit-learn)

```python
GradientBoostingRegressor(
    n_estimators    = 300,       # 300 boosting stages
    learning_rate   = 0.08,      # Shrinkage — reduces overfitting
    max_depth       = 5,         # Shallow trees for generalisation
    subsample       = 0.85,      # Stochastic gradient boosting
    min_samples_leaf = 4,
    random_state    = 42,
)
```

**Target:** `Day3_Platelets` — predicted platelet count at Day 3 (24 hours ahead)

**Trajectory Engineering (training synthetic labels):**

```python
# Day2 = current platelet (from CSV)
# Day1 = synthesised "yesterday" using recovery/declining split
#        58% recovering  → Day1 = Day2 / (1 + volatility)
#        42% declining   → Day1 = Day2 × (1 + volatility)
#        volatility drawn from Uniform(0.08, 0.28)

# Day3 forecast:
recovery_momentum = 1.05 (recovering) or 0.88 (declining)
Day3 = Day2 + (Day2 − Day1) × momentum + Normal(0, 1800)
```

**17 Input Features (`reg_features`):** trajectory delta (`Delta_D1_D2`) + CBC context + demographics + haemodynamics + clinical modifiers + seasonal context.

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Actual vs Predicted Scatter Plot      ║
  ║  Source: Diagrams.pdf  →  Page 3                             ║
  ║  R²=0.9953 · MAE=2,515 cells/µL · n=491 (20% hold-out)      ║
  ║  Replace with: ![Forecast](docs/img/diagrams_p03.png)         ║
  ╚══════════════════════════════════════════════════════════════╝
-->

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Platelet Trajectory Recovery vs       ║
  ║  Declining (Line Chart — 40 sample patients per group)       ║
  ║  Source: Diagrams.pdf  →  Page 13                            ║
  ║  Replace with: ![Trajectory](docs/img/diagrams_p13.png)      ║
  ╚══════════════════════════════════════════════════════════════╝
-->

---

## ⚙️ Feature Engineering

### Physiological Bounds — `PHYS_BOUNDS` (v3.1)

Clips unit-error outliers from the CSV **before any computation**. Anchored to WHO reference ranges and published dengue cohort data:

```
Column                                  Clip Range        Default
──────────────────────────────────────────────────────────────────
Platelet (cells/cu.mm)                  1,000 – 900,000   120,000
Haemoglobin (gm/Dl)                     2.0   – 22.0       12.5
Hematocrit (Packed Cell Volume) (%)     5.0   – 70.0       38.0
Red Blood Cell Count (millions/cu.mm)   1.0   – 10.0        4.5  ← v3.1 fix
Age                                     0.0   – 120.0      30.0
```

**v3.1 RBC Fix:** The CSV stored RBC in millions/cu.mm (normal: 3.5–6.5) but some rows had values >10 due to unit errors (cells/µL stored instead). `clip(1.0, 10.0)` rejects all garbage while keeping every physiologically real value.

### Derived Haemodynamic Features

| Feature | Formula | Clinical Threshold |
|---|---|---|
| **Shock Index** | `Heart_Rate / Systolic_BP` | > 0.9 → WHO warning sign |
| **Pulse Pressure** | `Systolic_BP − Diastolic_BP` | ≤ 20 mmHg → circulatory compromise |
| **MAP** | `DBP + PP / 3` | Used for Holliday-Segar fluid rate |
| **Holliday-Segar Rate** | Weight-based (4/2/1 rule) | Maintenance IV fluid rate (mL/hr) |

### Seasonal Risk Score

```
Months 12, 1, 2   →  Season_Risk = 0   (Off-season — LOW)
Months  3, 4, 5   →  Season_Risk = 1   (Pre-Monsoon — LOW-MODERATE)
Months  6–9       →  Season_Risk = 3   (Monsoon Peak — HIGHEST)
Months 10, 11     →  Season_Risk = 2   (Post-Monsoon — MODERATE)
```

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Monthly Case Load & Risk Severity     ║
  ║  Source: Diagrams.pdf  →  Page 11                            ║
  ║  Bar = total cases, Line = % High Risk, coloured by season   ║
  ║  Replace with: ![Seasonal](docs/img/diagrams_p11.png)         ║
  ╚══════════════════════════════════════════════════════════════╝
-->

### Synthetic Column Distributions

| Column | Distribution | Seed | Clinical Anchor |
|---|---|---|---|
| `Systolic_BP` | N(108, 18) clipped [60, 180] | 42 | WHO / dengue cohorts |
| `Diastolic_BP` | SBP × 0.62 + N(0, 5) | 42 | Proportional to systolic |
| `Heart_Rate` | N(92, 22) clipped [40, 180] | 42 | Tachycardia common in dengue |
| `WBC` | Bimodal: N(3200,900)[55%] + N(7500,2000)[45%] | 42 | Leukopenia hallmark |
| `AST` | Bimodal log-normal (low + high groups) | 42 | Near-universal elevation |
| `INR` | LogNormal(0.18, 0.32) clipped [0.8, 6.0] | 42 | ≥2.0 = severe criterion |
| `SpO2` | N(97.5, 2.0) clipped [85, 100] | 42 | <93% = Group C |
| `GCS` | 92% = 15, 8% uniform [8, 15] | 42 | Altered = encephalopathy |
| `Has_Pleural_Effusion` | Bernoulli(p=0.30) | 42 | ~30% inpatients |
| `Ascites_Grade` | Multinomial [0.70, 0.18, 0.08, 0.04] | 42 | Grade ≥2 = severe |

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — CBC & Clinical Feature Correlation    ║
  ║  Source: Diagrams.pdf  →  Page 8                             ║
  ║  Pearson r heatmap across all 17 features                    ║
  ║  Replace with: ![Correlation](docs/img/diagrams_p08.png)      ║
  ╚══════════════════════════════════════════════════════════════╝
-->

---

## 📷 OCR Pipeline

Supports ingestion of scanned physical lab reports via photograph or PDF upload.

```
Input (PDF / JPG / PNG)
        ↓
[1] Image Denoising & Deskewing        (cv2.fastNlMeansDenoising + warpAffine)
        ↓
[2] Text Extraction                     (pytesseract + tesseract-ocr-eng)
        ↓
[3] Regex Named-Entity Matching         (platelet, Hb, RBC, Hct, WBC, etc.)
        ↓
[4] Clinical Plausibility Check         (PHYS_BOUNDS validation — rejects outliers)
        ↓
SS_OCR buffer → Clinician Reviews → Auto-fill Data Entry Form
```

**Tesseract binary resolution order (cross-platform):**

| Priority | Path | Environment |
|---|---|---|
| 1 | `/usr/bin/tesseract` | Docker / Render / Ubuntu |
| 2 | `/usr/local/bin/tesseract` | Homebrew Mac (Intel) |
| 3 | `/opt/homebrew/bin/tesseract` | Homebrew Mac (Apple Silicon) |
| 4 | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Windows default |
| 5 | `shutil.which('tesseract')` | PATH fallback |

**OS packages required** (`packages.txt`):
```
tesseract-ocr
tesseract-ocr-eng
poppler-utils
```

---

## 🏥 Clinical Logic — WHO 2009

### Multi-Criterion Dengue Label (`_build_dengue_label`)

**Label = 1** (Warning Signs or Severe Dengue) is assigned if **any one** criterion fires:

| Category | Feature | Threshold | WHO Source |
|---|---|---|---|
| CBC | Platelet | < 100,000 cells/µL | Warning sign |
| CBC | Hematocrit | > 50% | Plasma leakage marker |
| CBC | Haemoglobin | < 7 g/dL | Severe bleeding criterion |
| Haemodynamics | Shock Index | > 0.9 | Impending shock |
| Haemodynamics | Pulse Pressure | ≤ 20 mmHg | Circulatory compromise |
| Liver | AST | ≥ 500 IU/L | Severe organ involvement |
| Coagulation | INR | ≥ 1.5 | Coagulopathy |
| Imaging | Pleural Effusion | Present | Plasma leakage |
| Imaging | Ascites Grade | ≥ 2 | Plasma leakage |
| Vitals | SpO2 | < 93% | Group C criterion |
| Neurology | GCS | < 13 | Altered consciousness |

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — WHO Criteria Trigger Frequency        ║
  ║  Source: Diagrams.pdf  →  Page 12                            ║
  ║  % patients triggering each criterion (PLT<100k = 46.1%      ║
  ║  most frequent, SI>0.9 = 44.5%, Pleural Effusion = 30.1%)   ║
  ║  Replace with: ![WHO Freq](docs/img/diagrams_p12.png)         ║
  ╚══════════════════════════════════════════════════════════════╝
-->

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Hematocrit vs Platelet Scatter        ║
  ║  Source: Diagrams.pdf  →  Page 5                             ║
  ║  Key plasma leakage markers with WHO thresholds overlaid     ║
  ║  Replace with: ![Hct vs PLT](docs/img/diagrams_p05.png)      ║
  ╚══════════════════════════════════════════════════════════════╝
-->

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Shock Index Distribution by Risk      ║
  ║  Source: Diagrams.pdf  →  Page 15                            ║
  ║  44.5% of patients have SI > 0.9                              ║
  ║  Replace with: ![Shock Index](docs/img/diagrams_p15.png)      ║
  ╚══════════════════════════════════════════════════════════════╝
-->

### OOD (Out-of-Distribution) Detection

Uses **Mahalanobis distance** calculated against `TRAIN_STATS` (mean + std of 8 core features) computed during training. If a patient's combined feature vector is statistically far from the training distribution, a yellow warning banner appears in the UI.

`TRAIN_STATS` keys: `platelets`, `hb`, `hct`, `rbc`, `hr`, `sys_bp`, `dia_bp`, `age`

### Tree Ensemble 95% Confidence Interval

```python
# Individual tree probabilities across 300 trees
trees = [t.predict_proba(X)[0][1] for t in classifier.estimators_]
ci_lo  = np.percentile(trees, 2.5)
ci_hi  = np.percentile(trees, 97.5)
ci_est = np.mean(trees)
ci_width = ci_hi - ci_lo
```

A wide CI signals model uncertainty — triggers an advisory in the UI.

---

## 🗂️ File Reference

| File | Purpose | Key Functions / Classes |
|---|---|---|
| `app.py` | Main Streamlit application — UI, inference, reporting | `load_ai_engine()`, `risk_input{}`, `compute_tree_ci()` |
| `train_model.py` | ML training pipeline — single source of truth for features | `main()`, `load_pretrained()`, `_build_dengue_label()`, `_build_synthetic_columns()`, `get_season()` |
| `evaluate_model.py` | Full evaluation suite | `evaluate()` |
| `generate_graphs.py` | 15 diagnostic graphs for documentation | Graph functions 01–15 |
| `generate_visuals.py` | Architecture diagrams and mind maps | Pipeline diagrams |
| `test_suite.py` | 12-point ML pipeline integration tests | `T1`–`T12` |
| `verify_ocr.py` | OCR dependency health check | Standalone diagnostic |
| `dengue_data_cleaned_debug.csv` | Training dataset (2,455 rows) | Core CBC + demographics |
| `requirements.txt` | Python dependencies | streamlit, sklearn, shap, fpdf2, pytesseract, etc. |
| `packages.txt` | OS-level packages | tesseract-ocr, poppler-utils |
| `Dockerfile` | Multi-stage Docker build — bakes models at build time | 5-stage build strategy |
| `render.yaml` | Render.com deployment config | Docker runtime declaration |
| `models/classifier.pkl` | Serialised RandomForestClassifier | joblib compressed (level 3) |
| `models/regressor.pkl` | Serialised GradientBoostingRegressor | joblib compressed (level 3) |
| `models/features.pkl` | `clf_features` and `reg_features` lists | Feature contract enforcement |

---

## 🚀 Installation & Local Run

### Prerequisites

- Python 3.11+
- Tesseract OCR 5.x ([install guide](https://github.com/tesseract-ocr/tesseract))
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/roguex7/dengue-cdss.git
cd dengue-cdss

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Train models (first run — creates models/ directory, ~30 s)
python train_model.py

# 4. Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Optional — Verify OCR Setup

```bash
python verify_ocr.py
```

Expected:
```
✅ pytesseract installed: 5.x.x
✅ OpenCV installed: 4.x.x
✅ Pillow installed
✅ pdf2image installed
```

---

## 🐳 Docker Deployment

The Dockerfile uses a **build-time training strategy** — models are trained once during `docker build` and baked into the image as `.pkl` files:

```
Cold-start latency
  Before (train at runtime): 45–90 seconds
  After  (baked models):     < 0.5 seconds  ✅
```

### Build & Run

```bash
# Build image (~2–3 min first time; subsequent builds use Docker cache)
docker build -t dengue-cdss .

# Run container
docker run -p 8501:8501 dengue-cdss
```

### Dockerfile Stages

```
[1] OS packages    apt-get install tesseract-ocr tesseract-ocr-eng
                               poppler-utils libgl1 libglib2.0-0
[2] Python deps    pip install -r requirements.txt
[3] Bake models    python train_model.py  →  models/*.pkl
[4] Copy app       COPY . .
[5] Start          streamlit run app.py --server.port=${PORT:-8501}
```

**Docker Layer Caching:** If `dengue_data_cleaned_debug.csv` and `train_model.py` are unchanged, Docker reuses the cached model layer — no retraining needed on incremental deploys.

---

## ☁️ Render Deployment

```yaml
# render.yaml
services:
  - type: web
    name: denguecdss
    runtime: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: PYTHON_VERSION
        value: "3.11"
```

1. Connect GitHub repo to Render
2. Render detects `render.yaml` and uses the Dockerfile
3. `PORT` is injected automatically by Render — picked up by the Streamlit `CMD`
4. Every push to the default branch triggers an auto-deploy

---

## 🧪 Testing — 12-Point Contract

```bash
python test_suite.py
```

| Test | Category | What Is Verified |
|---|---|---|
| **T1** | Smoke | Both models load, feature lists non-empty |
| **T2** | Contract | `clf_features` ⊆ `app.py` `risk_input` — no orphan features |
| **T3** | Shape | `predict_proba` returns `(1, 2)`, probability in `[0, 1]` |
| **T4** | Shape | `regressor.predict` returns `(1,)`, forecast non-negative |
| **T5** | Clinical | Severe patient (PLT=18k, SI=1.3, pleural effusion) > mild (PLT=180k) |
| **T6** | Edge | All-zero input doesn't crash |
| **T7** | Edge | Extreme values (PLT=1, AST=9999, GCS=3) stay in `[0,1]` |
| **T8** | Robustness | Partial input with `app.py` safety-loop pattern works |
| **T9** | CI | `compute_tree_ci` returns 4 values, all in `[0,1]` |
| **T10** | Direction | Recovering trajectory (Day1<Day2) > declining (Day1>Day2) |
| **T11** | Seasonal | Peak monsoon season ≥ off-season risk for same patient |
| **T12** | Integrity | No duplicate feature names in `clf_features` or `reg_features` |

**Critical contract (T2):** Any change to `clf_features` in `train_model.py` must be mirrored in `app.py`'s `risk_input` dict and `APP_RISK_INPUT_KEYS` in `test_suite.py`. T2 catches violations automatically.

---

## 📈 Model Evaluation

Run a full evaluation report:

```bash
python evaluate_model.py
```

Outputs:
- Hold-out test set classification report (precision / recall / F1 per class)
- Confusion matrix with TN / FP / FN / TP
- Sensitivity, Specificity, PPV, NPV, AUC
- 5-fold stratified cross-validation (AUC, Accuracy, F1)
- Ranked feature importances for classifier and regressor
- SHAP top-10 summary (`pip install shap` required)

**Generate all 15 diagnostic graphs:**

```bash
python generate_graphs.py
# → outputs to ./graphs/01_risk_distribution.png ... 15_shock_index_by_risk.png
```

<!--
  ╔══════════════════════════════════════════════════════════════╗
  ║  DIAGRAM PLACEHOLDER — Platelet Distribution KDE             ║
  ║  Source: Diagrams.pdf  →  Page 7                             ║
  ║  High Risk vs Low Risk with WHO thresholds                   ║
  ║  Replace with: ![PLT Dist](docs/img/diagrams_p07.png)         ║
  ╚══════════════════════════════════════════════════════════════╝
-->

---

## 🏥 WHO 2009 Classification Reference

| Group | Definition | Typical Management |
|---|---|---|
| **A** | Dengue without warning signs, tolerating oral fluids, no co-morbidities | Outpatient monitoring, oral rehydration, return precautions |
| **B** | Dengue with warning signs OR high-risk social situation OR co-morbidities | Hospital admission, IV fluids if indicated, close monitoring |
| **C** | Severe dengue — shock, severe organ impairment, or severe bleeding | ICU-level care, aggressive fluid resuscitation, specialist input |

---

## 📚 References

1. World Health Organization (2009). *Dengue: Guidelines for Diagnosis, Treatment, Prevention and Control* (New ed.). WHO Press, Geneva. ISBN 978-92-4-154787-1
2. Holliday MA, Segar WE (1957). The maintenance need for water in parenteral fluid therapy. *Pediatrics*, 19(5), 823–832.
3. Youden WJ (1950). Index for rating diagnostic tests. *Cancer*, 3(1), 32–35.
4. Breiman L (2001). Random forests. *Machine Learning*, 45(1), 5–32.
5. Friedman JH (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
6. Lundberg SM, Lee SI (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.

---

*This tool is a decision-support aid only. It does not replace clinical judgment. All outputs should be interpreted by a qualified clinician in the context of the full patient history.*

*Built with ❤️ for frontline clinicians in dengue-endemic regions.*
