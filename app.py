import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import io
import tempfile
import os
import re
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import train_model

# ── Optional OCR dependencies ──────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    import cv2

    # ── Explicit Tesseract binary path resolution ──────────────
    # Streamlit Cloud installs Tesseract via packages.txt but pytesseract
    # sometimes cannot auto-detect the binary from PATH.
    # We probe known install locations in priority order and set the path
    # explicitly — this is more reliable than relying on PATH alone.
    import shutil, os
    # Probe every known install location — works on Render (Docker),
    # Streamlit Cloud, Ubuntu servers, Mac (Homebrew), and Windows.
    # The first existing path wins and is set explicitly so pytesseract
    # never has to rely on subprocess PATH resolution.
    _TESS_CANDIDATES = [
        '/usr/bin/tesseract',             # Docker (python:slim) / Render / Ubuntu
        '/usr/local/bin/tesseract',       # Homebrew Mac Intel / some Linux builds
        '/opt/homebrew/bin/tesseract',    # Homebrew Mac Apple Silicon
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # Windows default
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    _tess_found = shutil.which('tesseract')  # Try PATH first
    for _candidate in _TESS_CANDIDATES:
        if os.path.isfile(_candidate):
            _tess_found = _candidate
            break
    if _tess_found:
        pytesseract.pytesseract.tesseract_cmd = _tess_found

    # Final verification — raises TesseractNotFoundError if binary still missing
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ══════════════════════════════════════════════════════════
#  1.  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Dengue CDSS",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
section[data-testid="stSidebar"] { background-color: #161b24 !important; }
.nav-container {
    background: linear-gradient(135deg, #1a1f2e 0%, #161b24 100%);
    border-bottom: 2px solid #2d3748; padding: 12px 0; margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
div.stButton > button:first-child {
    width: 100%; border-radius: 8px; font-weight: 700; height: 3em;
    background: linear-gradient(135deg, #c0392b 0%, #922b21 100%);
    color: white; border: none; letter-spacing: 0.08em; font-size: 0.95rem;
    transition: all 0.3s ease; box-shadow: 0 4px 12px rgba(192,57,43,.4);
}
div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #922b21 0%, #6e1f17 100%);
    box-shadow: 0 6px 18px rgba(192,57,43,.6); transform: translateY(-1px);
}
.section-header {
    font-size: .75rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .05em; color: #8b92a8;
    margin-top: 1rem; margin-bottom: .5rem;
    border-bottom: 1px solid #2d3748; padding-bottom: .25rem;
}
.preview-banner {
    background: linear-gradient(90deg,#1a2035,#1e2d40);
    border: 1px solid #2d4a6e; border-left: 4px solid #3498db;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 1rem;
}
.preview-banner h4 { color:#5dade2; margin:0 0 4px 0; font-size:.85rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
.preview-banner p  { color:#7f8c9b; margin:0; font-size:.75rem; }
.workflow-banner {
    background: linear-gradient(90deg, #1a2540, #12192b);
    border: 1px solid #2d4a6e; border-left: 4px solid #2ecc71;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 1rem;
}
.workflow-banner h4 { color: #2ecc71; margin: 0 0 6px 0; font-size: .82rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }
.workflow-banner p  { color: #7f8c9b; margin: 0; font-size: .78rem; line-height: 1.6; }
.intro-card {
    background: linear-gradient(135deg, #12192b 0%, #1a2540 100%);
    border: 1px solid #2d4a6e; border-radius: 12px; padding: 28px 32px; margin-bottom: 1.5rem;
}
.intro-card h2   { color: #5dade2; margin: 0 0 8px 0; font-size: 1.4rem; }
.intro-card p    { color: #a0aec0; margin: 0; font-size: .95rem; line-height: 1.6; }
.intro-card ul   { color: #a0aec0; margin: 10px 0 0 0; padding-left: 1.2rem; font-size: .9rem; line-height: 1.8; }
.feature-card {
    background: linear-gradient(135deg, #161b24 0%, #1a1f2e 100%);
    border: 1px solid #2d3748; border-radius: 12px; padding: 0;
    min-height: 190px; display: flex; flex-direction: column;
    align-items: stretch; text-align: center; box-sizing: border-box;
    overflow: hidden; transition: all 0.25s ease;
}
.feature-card:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(0,0,0,0.45); border-color: #4a5568; }
.fc-accent { height: 4px; width: 100%; flex-shrink: 0; }
.fc-accent-blue  { background: linear-gradient(90deg, #3498db, #5dade2); }
.fc-accent-red   { background: linear-gradient(90deg, #e74c3c, #f1948a); }
.fc-accent-green { background: linear-gradient(90deg, #2ecc71, #82e0aa); }
.fc-accent-amber { background: linear-gradient(90deg, #f39c12, #f8c471); }
.feature-card h3 { color: #e2e8f0; font-size: 0.95rem; margin: 16px 16px 8px 16px; font-weight: 700; text-align: center; line-height: 1.3; }
.feature-card p  { color: #718096; font-size: .80rem; margin: 0 16px 16px 16px; line-height: 1.65; text-align: center; flex: 1; }
.model-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2540 100%);
    border: 1px solid #2d4a6e; border-radius: 14px; padding: 28px 32px; margin: 24px 0;
}
.model-card h3 { color: #5dade2; margin: 0 0 20px 0; font-size: 1.2rem; font-weight: 700; }
.metric-pill-row { display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin-bottom: 16px; }
.metric-pill { background: rgba(52,152,219,0.12); border: 1px solid #3498db; border-radius: 8px; padding: 12px 8px; text-align: center; }
.metric-pill .val { color: #5dade2; font-size: 1.4rem; font-weight: 800; display: block; margin-bottom: 6px; }
.metric-pill .lbl { color: #8b92a8; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; line-height: 1.3; display: block; }
.metric-pill-good { border-color: #2ecc71; background: rgba(46,204,113,0.08); }
.metric-pill-good .val { color: #2ecc71; }
.metric-pill-warn { border-color: #f39c12; background: rgba(243,156,18,0.08); }
.metric-pill-warn .val { color: #f39c12; }
.alert-critical {
    background: rgba(231,76,60,0.12); border: 1px solid #e74c3c;
    border-left: 4px solid #e74c3c; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0; color: #f1948a; font-size: 0.88rem;
}
.alert-warning {
    background: rgba(243,156,18,0.12); border: 1px solid #f39c12;
    border-left: 4px solid #f39c12; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0; color: #f8c471; font-size: 0.88rem;
}
.alert-ok {
    background: rgba(46,204,113,0.10); border: 1px solid #2ecc71;
    border-left: 4px solid #2ecc71; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0; color: #82e0aa; font-size: 0.88rem;
}
.severity-panel {
    background: linear-gradient(135deg, #12192b 0%, #1a2540 100%);
    border: 1px solid #2d4a6e; border-radius: 12px; padding: 20px 24px; margin: 12px 0;
}
.severity-panel h4 { color: #5dade2; margin: 0 0 14px 0; font-size: 1rem; font-weight: 700; }
.trajectory-panel {
    background: linear-gradient(135deg, #1a2035 0%, #12192b 100%);
    border: 1px solid #2d4a6e; border-radius: 10px; padding: 16px 20px; margin: 10px 0;
}
.trajectory-panel h5 { color: #e2e8f0; margin: 0 0 10px 0; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
.countdown-badge {
    display: inline-block; padding: 6px 14px;
    background: rgba(231,76,60,0.15); border: 1px solid #e74c3c;
    border-radius: 6px; color: #f1948a; font-size: 0.8rem; font-weight: 700;
    margin: 4px 0;
}
.countdown-safe {
    background: rgba(46,204,113,0.12); border-color: #2ecc71; color: #82e0aa;
}
.ood-panel {
    background: rgba(243,156,18,0.08); border: 1px solid #f39c12;
    border-left: 4px solid #f39c12; border-radius: 8px;
    padding: 10px 14px; margin: 8px 0; font-size: 0.82rem; color: #f8c471;
}
.ci-panel {
    background: rgba(52,152,219,0.08); border: 1px solid #3498db;
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
    font-size: 0.82rem; color: #5dade2;
}
.delta-positive { color: #e74c3c; font-weight: 700; }
.delta-negative { color: #2ecc71; font-weight: 700; }
.delta-neutral  { color: #8b92a8; }
.who-badge-A { display: inline-block; padding: 8px 20px; background: rgba(46,204,113,0.15); border: 2px solid #2ecc71; border-radius: 20px; color: #2ecc71; font-weight: 700; font-size: 0.9rem; }
.who-badge-B { display: inline-block; padding: 8px 20px; background: rgba(243,156,18,0.15); border: 2px solid #f39c12; border-radius: 20px; color: #f39c12; font-weight: 700; font-size: 0.9rem; }
.who-badge-C { display: inline-block; padding: 8px 20px; background: rgba(231,76,60,0.15); border: 2px solid #e74c3c; border-radius: 20px; color: #e74c3c; font-weight: 700; font-size: 0.9rem; }
.ocr-panel {
    background: linear-gradient(135deg, #12192b 0%, #1a2035 100%);
    border: 1px solid #3498db; border-left: 4px solid #3498db;
    border-radius: 10px; padding: 16px 18px; margin: 10px 0 14px 0;
}
.ocr-panel h5 { color: #5dade2; margin: 0 0 8px 0; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
.ocr-panel p { color: #8b92a8; margin: 0; font-size: 0.78rem; line-height: 1.6; }
.multi-ocr-panel {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2540 100%);
    border: 1px solid #3498db; border-left: 4px solid #5dade2;
    border-radius: 10px; padding: 14px 16px; margin: 8px 0 12px 0;
}
.multi-ocr-panel h5 { color: #5dade2; margin: 0 0 6px 0; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
.multi-ocr-panel p { color: #8b92a8; margin: 0; font-size: 0.76rem; line-height: 1.55; }
.module-disabled {
    background: rgba(22, 27, 36, 0.45); border: 1px dashed #2d3748;
    border-radius: 10px; padding: 14px 18px; margin: 8px 0;
    text-align: center; user-select: none;
}
.module-disabled p { color: #4a5568; font-size: 0.78rem; margin: 0; font-style: italic; line-height: 1.5; }
.discharge-info-panel {
    background: linear-gradient(135deg, #1a2035 0%, #12192b 100%);
    border: 1px solid #6c5ce7; border-left: 4px solid #6c5ce7;
    border-radius: 10px; padding: 14px 18px; margin: 10px 0 14px 0;
}
.discharge-info-panel h5 { color: #a29bfe; margin: 0 0 8px 0; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
.discharge-info-panel p { color: #8b92a8; margin: 0; font-size: 0.78rem; line-height: 1.6; }
.conf-high  { color: #2ecc71; font-weight: 700; }
.conf-med   { color: #f39c12; font-weight: 700; }
.conf-low   { color: #e74c3c; font-weight: 700; }
.auto-fill-badge {
    display: inline-block; padding: 3px 8px;
    background: rgba(46,204,113,0.15); border: 1px solid #2ecc71;
    border-radius: 4px; color: #2ecc71; font-size: 0.68rem;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
    margin-left: 6px; vertical-align: middle;
}
.urine-info-card {
    background: linear-gradient(135deg, #1e2d40 0%, #1a2035 100%);
    border: 1px solid #3498db; border-left: 4px solid #3498db;
    border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;
}
.urine-info-card h5 { color: #5dade2; margin: 0 0 8px 0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.urine-info-card p  { color: #8b92a8; margin: 0; font-size: 0.8rem; line-height: 1.6; }
.urine-rate-display {
    background: linear-gradient(135deg, rgba(52,152,219,0.12) 0%, rgba(52,152,219,0.05) 100%);
    border: 2px solid #3498db; border-radius: 8px;
    padding: 14px 16px; margin-top: 12px; text-align: center;
}
.urine-rate-display .rate-value { color: #5dade2; font-size: 1.2rem; font-weight: 700; margin: 0 0 4px 0; }
.urine-rate-display .rate-label { color: #8b92a8; font-size: 0.75rem; margin: 0; text-transform: uppercase; letter-spacing: 0.05em; }
.weight-source-badge {
    display: inline-block; padding: 6px 12px;
    background: rgba(52,152,219,0.15); border: 1px solid #3498db;
    border-radius: 6px; color: #5dade2; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-top: 28px; vertical-align: middle; line-height: 1.4;
}
.clear-buttons-container {
    background: linear-gradient(135deg, #1a2540 0%, #12192b 100%);
    border: 1px solid #2d4a6e; border-radius: 10px;
    padding: 16px; margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.clear-buttons-header { color: #5dade2; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; text-align: center; }
.footer { text-align: center; padding: 40px 0 20px 0; color: #718096; font-size: 0.85rem; border-top: 1px solid #2d3748; margin-top: 60px; }
.footer a { color: #5dade2; text-decoration: none; font-weight: 500; }
div[data-testid="stMetricValue"] { font-size: 1.5rem; }
div[data-testid="stNumberInput"] > label { font-size: .875rem; font-weight: 500; margin-bottom: .25rem; }
div[data-testid="stNumberInput"]         { margin-bottom: .75rem; }
div[data-testid="stSelectbox"]  > label { font-size: .875rem; font-weight: 500; }
.sidebar-collapse-btn-st > div > button {
    background: linear-gradient(135deg,#1a2540,#12192b) !important;
    border: 1px solid #2d4a6e !important;
    color: #5dade2 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    height: 2.6em !important;
    letter-spacing: 0.04em !important;
    box-shadow: none !important;
}
.sidebar-collapse-btn-st > div > button:hover {
    border-color: #3498db !important;
    color: #7ec8e3 !important;
    transform: none !important;
}
.show-sidebar-bar {
    display: flex; align-items: center; justify-content: flex-start;
    gap: 12px; margin-bottom: 18px;
    background: linear-gradient(90deg,#1a2035,#12192b);
    border: 1px solid #2d4a6e; border-radius: 10px; padding: 10px 16px;
}
.show-sidebar-bar p { color: #8b92a8; margin: 0; font-size: 0.82rem; }
.season-badge {
    display: inline-block; padding: 5px 12px;
    border-radius: 6px; font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    vertical-align: middle; line-height: 1.4;
}
.season-badge-high   { background: rgba(231,76,60,0.15);  border: 1px solid #e74c3c; color: #f1948a; }
.season-badge-mod    { background: rgba(243,156,18,0.15); border: 1px solid #f39c12; color: #f8c471; }
.season-badge-pre    { background: rgba(243,156,18,0.10); border: 1px solid #d4ac0d; color: #f4d03f; }
.season-badge-low    { background: rgba(46,204,113,0.12); border: 1px solid #2ecc71; color: #82e0aa; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  2.  MODEL LOADER
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_ai_engine():
    """
    Load AI engine — permanent fast-start strategy:
      1. Try loading pre-trained .pkl files baked into the Docker image  (<0.5s)
      2. Fall back to training from scratch only if pkl files are missing
         (first local run, or non-Docker environment)
    This eliminates 45-90s cold-start training on every Render spin-up.
    """
    try:
        # ── Fast path: pre-trained models (Docker / Render production) ──────
        result = train_model.load_pretrained()
        if result is not None:
            return result[0], result[1], result[2], result[3]
        # ── Fallback: train from scratch (local dev / first run) ─────────────
        r = train_model.main()
        return r[0], r[1], r[2], r[3]
    except Exception:
        return None, None, None, None
    
# ══════════════════════════════════════════════════════════
#  3.  CORE HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════
def get_season_score(d):
    m = d.month
    if m in [12, 1, 2]:    return 0
    elif m in [3, 4, 5]:   return 1
    elif m in [6, 7, 8, 9]: return 3
    return 2

def get_season_meta(score: int) -> dict:
    """
    Maps Season_Risk score → badge label, CSS class, and human-readable context.
    Scores: 0=Low (Dec-Feb), 1=Pre (Mar-May), 2=Elevated (Oct-Nov), 3=Peak (Jun-Sep)
    """
    return {
        3: {
            'label':   'HIGH — Peak Season',
            'cls':     'season-badge season-badge-high',
            'context': 'Jun–Sep monsoon: highest dengue transmission risk',
            'tip':     'Seasonal modifier active — model assigns maximum seasonal weight',
        },
        2: {
            'label':   'MODERATE — Post-Monsoon',
            'cls':     'season-badge season-badge-mod',
            'context': 'Oct–Nov: declining but elevated transmission',
            'tip':     'Seasonal modifier active — moderate seasonal weight applied',
        },
        1: {
            'label':   'LOW-MOD — Pre-Season',
            'cls':     'season-badge season-badge-pre',
            'context': 'Mar–May: low but rising transmission',
            'tip':     'Minimal seasonal modifier applied',
        },
        0: {
            'label':   'LOW — Off-Season',
            'cls':     'season-badge season-badge-low',
            'context': 'Dec–Feb: lowest dengue transmission',
            'tip':     'No seasonal uplift — base risk applies',
        },
    }.get(score, {'label': 'UNKNOWN', 'cls': 'season-badge season-badge-low',
                  'context': '', 'tip': ''})

def calculate_map(s, d):
    """Mean Arterial Pressure = DBP + (SBP - DBP) / 3"""
    return d + (s - d) / 3

def calculate_bmi(w, h):
    """BMI = weight(kg) / height(m)^2"""
    return 0 if (h == 0 or w == 0) else w / (h / 100) ** 2

def calculate_fluid(w):
    """Holliday-Segar maintenance fluid rate (mL/hr)"""
    if w == 0:  return 0
    if w <= 10: return w * 4
    if w <= 20: return 40 + (w - 10) * 2
    return 60 + (w - 20) * 1

def calculate_urine_rate(volume_ml, time_hrs, weight_kg):
    """Urine output rate in mL/kg/hr"""
    if weight_kg <= 0 or time_hrs <= 0:
        return 0.0
    return round(volume_ml / (weight_kg * time_hrs), 2)

def interpret_urine_rate(rate):
    if rate <= 0:        return "Not Recorded",    "No urine data recorded"
    elif rate < 0.5:     return "Oliguria (Low)",  "Consider fluid bolus or investigate renal function"
    elif rate < 1.0:     return "Borderline Low",  "Monitor closely, may indicate early hypovolemia"
    elif rate <= 2.0:    return "Normal",           "Adequate urine output"
    elif rate <= 4.0:    return "High Normal",      "Good hydration status"
    else:                return "Polyuria (High)",  "Consider reducing fluid rate if sustained"

def urine_risk_impact(urine_rate):
    if urine_rate <= 0:     return 0.0
    elif urine_rate < 0.5:  return 0.28
    elif urine_rate < 1.0:  return 0.14
    elif urine_rate <= 2.0: return -0.06
    elif urine_rate <= 4.0: return -0.03
    else:                   return 0.09

# ══════════════════════════════════════════════════════════
#  3b.  EXTENDED CLINICAL HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════

def calculate_egfr_ckdepi(creatinine_mg_dl: float, age: int, sex: str) -> float:
    """CKD-EPI 2021 creatinine equation (race-free, JASN 2021)"""
    if creatinine_mg_dl <= 0 or age <= 0:
        return 0.0
    kappa      = 0.7  if sex == 'Female' else 0.9
    alpha      = -0.241 if sex == 'Female' else -0.302
    sex_factor = 1.012 if sex == 'Female' else 1.0
    cr_kappa   = creatinine_mg_dl / kappa
    if creatinine_mg_dl < kappa:
        egfr = 142 * (cr_kappa ** alpha) * (0.9938 ** age) * sex_factor
    else:
        egfr = 142 * (cr_kappa ** -1.200) * (0.9938 ** age) * sex_factor
    return round(egfr, 1)

def detect_aki(creatinine_current: float, creatinine_baseline: float, days_elapsed: float) -> tuple:
    """Returns (is_aki, stage_label, detail)"""
    if creatinine_baseline <= 0 or creatinine_current <= 0:
        return False, "No baseline", ""
    if days_elapsed <= 2:
        if creatinine_current >= creatinine_baseline + 0.3:
            return True, "AKI Stage 1", f"Rise ≥0.3 mg/dL in 48h: {creatinine_baseline:.2f}→{creatinine_current:.2f}"
    elif days_elapsed <= 7:
        if creatinine_current >= creatinine_baseline * 1.5:
            return True, "AKI Stage 1", f"Rise ≥1.5x baseline in 7d: {creatinine_baseline:.2f}→{creatinine_current:.2f}"
        if creatinine_current >= creatinine_baseline * 2.0:
            return True, "AKI Stage 2", f"Rise ≥2.0x baseline: {creatinine_baseline:.2f}→{creatinine_current:.2f}"
    return False, "No AKI", f"Creatinine stable: {creatinine_current:.2f} mg/dL"

def get_dengue_phase(illness_day, is_afebrile: bool = False) -> dict:
    """Returns phase metadata for trajectory priors and clinical alerts"""
    if illness_day is None or illness_day <= 0:
        return {'phase': 'Unknown', 'risk_window': False,
                'expected_plt_trend': 'unknown', 'color': '#8b92a8',
                'action': 'Enter symptom onset date to enable phase tracking'}
    if illness_day <= 3:
        return {'phase': 'Febrile', 'risk_window': False,
                'expected_plt_trend': 'mild_decline', 'color': '#f39c12',
                'action': 'Monitor closely — critical phase imminent on Day 4-6'}
    elif illness_day <= 7:
        return {'phase': 'Critical', 'risk_window': True,
                'expected_plt_trend': 'rapid_decline', 'color': '#e74c3c',
                'action': 'CRITICAL PHASE: Maximum monitoring — hospitalisation mandatory'}
    elif illness_day <= 10:
        if is_afebrile:
            return {'phase': 'Recovery', 'risk_window': False,
                    'expected_plt_trend': 'rising', 'color': '#2ecc71',
                    'action': 'Recovery phase: Watch for fluid overload as plasma reabsorbs'}
        else:
            return {'phase': 'Prolonged Critical', 'risk_window': True,
                    'expected_plt_trend': 'uncertain', 'color': '#e74c3c',
                    'action': 'Fever beyond Day 7 — consider secondary infection or complications'}
    else:
        return {'phase': 'Convalescent', 'risk_window': False,
                'expected_plt_trend': 'normalising', 'color': '#2ecc71',
                'action': 'Prepare discharge — follow-up CBC in 48 hours'}

def compute_plasma_leakage_score(report: dict, hct_baseline: float = 0.0,
                                  imaging: dict = None) -> tuple:
    """
    Multi-marker plasma leakage probability (0.0-1.0).
    Returns (score, contributing_factors_list, risk_label)
    """
    score   = 0.0
    factors = []

    hct_rise = report.get('hct', 0) - hct_baseline if hct_baseline > 0 else 0
    if hct_rise >= 10:
        score += 0.50; factors.append(f"Hct rise +{hct_rise:.1f}% (≥10% = definitive)")
    elif hct_rise >= 5:
        score += 0.25; factors.append(f"Hct rise +{hct_rise:.1f}% (5-10% = probable)")

    d_dimer = report.get('d_dimer', 0)
    if d_dimer >= 2000:
        score += 0.30; factors.append(f"D-dimer {d_dimer:,} ng/mL (≥2000 = high risk)")
    elif d_dimer >= 1000:
        score += 0.15; factors.append(f"D-dimer {d_dimer:,} ng/mL (≥1000 = elevated)")

    albumin = report.get('albumin', 0)
    if albumin > 0:
        if albumin < 3.0:
            score += 0.20; factors.append(f"Albumin {albumin:.1f} g/dL (<3.0 = significant hypoalbuminaemia)")
        elif albumin < 3.5:
            score += 0.10; factors.append(f"Albumin {albumin:.1f} g/dL (<3.5 = mild)")

    pp = report.get('sys_bp', 120) - report.get('dia_bp', 80)
    if pp <= 20:
        score += 0.25; factors.append(f"Narrow pulse pressure {pp} mmHg (≤20 = impending shock)")
    elif pp <= 30:
        score += 0.10; factors.append(f"Reduced pulse pressure {pp} mmHg")

    if imaging:
        if imaging.get('pleural_effusion', False):
            score += 0.50; factors.append("Pleural effusion confirmed on imaging (definitive)")
        gb_wall = imaging.get('gallbladder_wall_mm', 0)
        if gb_wall >= 5:
            score += 0.40; factors.append(f"GB wall {gb_wall}mm (≥5mm = highly specific)")
        elif gb_wall >= 3:
            score += 0.20; factors.append(f"GB wall {gb_wall}mm (≥3mm = abnormal)")
        ascites = imaging.get('ascites_grade', 0)
        if ascites >= 2:
            score += 0.45; factors.append(f"Ascites Grade {ascites} (≥2 = WHO severe criterion)")
        elif ascites == 1:
            score += 0.20; factors.append("Ascites Grade 1 (trace)")

    score = min(score, 1.0)
    if   score >= 0.85: label = "DEFINITIVE LEAKAGE — Emergency resuscitation"
    elif score >= 0.70: label = "CONFIRMED LEAKAGE — Group C criteria"
    elif score >= 0.50: label = "PROBABLE LEAKAGE — Group B admission"
    elif score >= 0.30: label = "POSSIBLE LEAKAGE — Enhanced monitoring"
    else:               label = "Low probability"
    return round(score, 2), factors, label

def compute_bleeding_risk_score(report: dict, is_secondary_dengue: bool = False) -> tuple:
    """
    Bleeding Risk Score (BRS) for platelet transfusion decision support.
    Returns (score, interpretation, recommendation)
    """
    score = 0
    details = []
    plt = report.get('platelets', 999999)
    if plt < 20000:
        score += 3; details.append(f"PLT <20,000: +3 (critical nadir)")
    elif plt < 50000:
        score += 1; details.append(f"PLT 20-50k: +1")

    inr = report.get('inr', 0)
    if inr >= 2.0:
        score += 2; details.append(f"INR {inr:.1f} ≥2.0: +2")
    elif inr >= 1.5:
        score += 2; details.append(f"INR {inr:.1f} ≥1.5: +2")

    aptt = report.get('aptt', 0)
    if aptt > 0 and aptt > 52.5:  # 1.5x ULN (ULN=35s)
        score += 1; details.append(f"aPTT {aptt:.0f}s >1.5x ULN: +1")

    who_signs = report.get('who_signs', [])
    syms      = report.get('symptoms', [])
    if "Mucosal Bleeding" in who_signs or "Bleeding" in syms:
        score += 3; details.append("Active mucosal bleeding: +3")

    if is_secondary_dengue:
        score += 1; details.append("Secondary dengue infection: +1")

    if   score >= 6: rec = "Transfusion RECOMMENDED per local protocol — haematology consult"
    elif score >= 3: rec = "Moderate risk — consider transfusion if invasive procedure planned"
    else:            rec = "Low risk — clinical observation"
    return score, details, rec

def compute_fors(sorted_reports: list) -> tuple:
    """
    Fluid Overload Risk Score (FORS) — post-resuscitation paradox detector.
    Returns (score, factors, is_high_risk)
    """
    if not sorted_reports:
        return 0, [], False
    latest = sorted_reports[-1]
    score  = 0
    factors = []
    
    if len(sorted_reports) >= 2:
        hct_vals  = [r['hct'] for r in sorted_reports]
        hct_peak  = max(hct_vals)
        hct_curr  = latest['hct']
        hct_fall  = hct_peak - hct_curr
        if hct_fall >= 10:
            score += 2; factors.append(f"Hct fell {hct_fall:.1f}% from peak {hct_peak:.1f}% — plasma reabsorption")
        elif hct_fall >= 5:
            score += 1; factors.append(f"Hct falling {hct_fall:.1f}% — early reabsorption signal")

    uo = latest.get('urine_output', 0)
    if uo > 4.0:
        score += 2; factors.append(f"Polyuria {uo:.2f} mL/kg/hr — reabsorption phase")
    elif uo > 3.0:
        score += 1; factors.append(f"High urine output {uo:.2f} mL/kg/hr")

    spo2 = latest.get('spo2', 0)
    if 0 < spo2 < 93:
        score += 3; factors.append(f"SpO2 {spo2}% — hypoxaemia, possible pulmonary oedema")
    elif 0 < spo2 < 95:
        score += 1; factors.append(f"SpO2 {spo2}% — borderline, monitor closely")

    rr = latest.get('rr', 0)
    if rr > 24:
        score += 1; factors.append(f"Tachypnoea RR {rr}/min — respiratory stress")

    return score, factors, score >= 4

def interpret_ast_alt(ast: float, alt: float) -> tuple:
    """Returns (ratio, pattern_label, severity_label, color)"""
    if ast <= 0:
        return 0.0, "Not recorded", "N/A", "#8b92a8"
    ratio = ast / alt if alt > 0 else 0.0
    if   ast >= 1000: sev = "CRITICAL — WHO Severe Dengue criterion";  color = "#e74c3c"
    elif ast >= 500:  sev = "Severe hepatitis";                         color = "#e74c3c"
    elif ast >= 200:  sev = "Moderate hepatitis";                       color = "#f39c12"
    elif ast >= 80:   sev = "Mild elevation";                           color = "#f39c12"
    else:             sev = "Normal range";                             color = "#2ecc71"
    pattern = f"AST:ALT ratio {ratio:.1f} — {'Dengue pattern (>2.0)' if ratio > 2.0 else 'Atypical for dengue' if ratio < 1.0 else 'Borderline'}"
    return ratio, pattern, sev, color

# ══════════════════════════════════════════════════════════
#  4.  WHO CLASSIFICATION ENGINE
# ══════════════════════════════════════════════════════════
def classify_who_dengue(platelets, who_signs, symptoms, shock_index, hct,
                         hct_prev=None, map_val=None, ast=0, inr=0.0,
                         pleural_effusion=False, ascites_grade=0, spo2=0, gcs=15):
    """WHO 2009 classification with extended organ criteria"""
    severe_criteria = []

    # ── Haemodynamic criteria ─────────────────────────
    if shock_index >= 1.0:
        severe_criteria.append(f"Hemodynamic compromise: Shock Index {shock_index:.2f} (≥1.0)")
    if map_val is not None and map_val < 65:
        severe_criteria.append(f"Low MAP: {map_val:.1f} mmHg (<65 mmHg)")
    if hct_prev is not None and hct > 0 and (hct - hct_prev) >= 10:
        severe_criteria.append(f"Rapid Hct rise: +{hct - hct_prev:.1f}% — Plasma Leakage")

    # ── Bleeding criteria ─────────────────────────────
    if "Mucosal Bleeding" in who_signs and 0 < platelets < 50000:
        severe_criteria.append("Mucosal bleeding + severe thrombocytopenia")
    if 0 < platelets < 20000:
        severe_criteria.append(f"Critical thrombocytopenia: {platelets:,} cells/uL")

    # ── Organ impairment criteria ─────────────────────
    if ast >= 1000:
        severe_criteria.append(f"Severe hepatitis: AST {ast:,} IU/L (≥1,000 = WHO criterion)")
    if inr >= 1.5:
        severe_criteria.append(f"Coagulopathy: INR {inr:.1f} (≥1.5 = severe coagulopathy)")
    if pleural_effusion:
        severe_criteria.append("Pleural effusion confirmed — definitive plasma leakage")
    if ascites_grade >= 2:
        severe_criteria.append(f"Ascites Grade {ascites_grade} — intraperitoneal plasma leakage")
    if 0 < spo2 < 92:
        severe_criteria.append(f"Hypoxaemia: SpO2 {spo2}% (<92% = respiratory compromise)")
    if gcs < 13:
        severe_criteria.append(f"Altered consciousness: GCS {gcs} (<13 = dengue encephalopathy)")

    if severe_criteria:
        return ("Severe Dengue", "C", severe_criteria,
                "Emergency treatment required — ICU-level care",
                ["Immediate IV fluid resuscitation (10-20 mL/kg bolus over 15-30 min)",
                 "Continuous vital sign monitoring every 15-30 minutes",
                 "Urgent CBC, LFT, coagulation profile, ABG",
                 "Consider ICU transfer and specialist referral",
                 "Blood products if severe bleeding: platelet concentrate if <20,000 or active bleed",
                 "Strict fluid balance — watch for overload after resuscitation",
                 "Hepatology/nephrology consult if organ impairment criteria met"])

    # ── Warning signs ─────────────────────────────────
    warning_signs = []
    if "Abdominal Pain"         in who_signs: warning_signs.append("Abdominal pain or tenderness")
    if "Persistent Vomiting"    in who_signs or "Vomiting" in symptoms:
                                              warning_signs.append("Persistent vomiting")
    if "Mucosal Bleeding"       in who_signs: warning_signs.append("Mucosal bleeding")
    if "Lethargy/Restlessness"  in who_signs: warning_signs.append("Lethargy or restlessness")
    if "Liver Enlargement >2cm" in who_signs: warning_signs.append("Liver enlargement >2 cm")
    if hct_prev is not None and hct > 0 and (hct - hct_prev) >= 5:
        warning_signs.append(f"Rising hematocrit (+{hct - hct_prev:.1f}%)")
    if 0 < platelets < 100000:
        warning_signs.append(f"Thrombocytopenia: {platelets:,} cells/uL")
    if shock_index >= 0.9:
        warning_signs.append(f"Elevated Shock Index: {shock_index:.2f}")
    if 0 < ast < 1000 and ast >= 200:
        warning_signs.append(f"Hepatitis: AST {ast:,} IU/L (>200 = moderate elevation)")
    if 0 < inr < 1.5 and inr >= 1.2:
        warning_signs.append(f"Borderline coagulopathy: INR {inr:.1f}")
    if 0 < spo2 < 95:
        warning_signs.append(f"Low SpO2: {spo2}% — possible early pleural effusion")
    if ascites_grade == 1:
        warning_signs.append("Trace ascites on imaging — early plasma leakage")

    if warning_signs:
        return ("Dengue With Warning Signs", "B", warning_signs,
                "In-hospital management required — close monitoring",
                ["Hospital admission mandatory",
                 "IV fluid therapy: 5-7 mL/kg/hr, reassess every 1-2 hours",
                 "CBC every 6-8 hours; LFT and coagulation profile if not done",
                 "Strict fluid balance and urine output monitoring",
                 "Watch for transition to Severe Dengue — repeat assessment every 4-6h",
                 "No aspirin, NSAIDs, or anticoagulants"])

    return ("Dengue Without Warning Signs", "A", [],
            "Outpatient management may be appropriate with close follow-up",
            ["Encourage oral fluid intake (ORS, coconut water, juices)",
             "Paracetamol for fever (15 mg/kg/dose, max 60 mg/kg/day)",
             "Avoid aspirin and NSAIDs",
             "Return immediately if warning signs develop",
             "Daily CBC until platelet trend reversal confirmed",
             "Educate family on warning signs requiring ER visit"])

# ══════════════════════════════════════════════════════════
#  5.  SERIAL ALERT SYSTEM
# ══════════════════════════════════════════════════════════
def check_serial_alerts(sorted_reports):
    alerts = []
    if len(sorted_reports) < 2:
        return alerts
    baseline = sorted_reports[0]
    for i in range(1, len(sorted_reports)):
        prev      = sorted_reports[i - 1]
        curr      = sorted_reports[i]
        delta_hrs  = max((curr['datetime'] - prev['datetime']).total_seconds() / 3600, 0.1)
        delta_days = delta_hrs / 24
        
        # Guard: skip platelet velocity calc if either report lacks a recorded count
        if prev['platelets'] > 0 and curr['platelets'] > 0:
            plt_drop     = prev['platelets'] - curr['platelets']
            plt_drop_24  = plt_drop / delta_days
            if plt_drop_24 > 40000:
                alerts.append(('CRITICAL',
                    f"Platelet crash: down {int(plt_drop_24):,}/24h ({prev['Label']} to {curr['Label']})",
                    "Immediate CBC repeat. Prepare for platelet transfusion if <20,000."))
            elif plt_drop_24 > 20000:
                alerts.append(('WARNING',
                    f"Rapid platelet drop: down {int(plt_drop_24):,}/24h ({prev['Label']} to {curr['Label']})",
                    "Increase monitoring frequency to every 6 hours."))
        
        hct_rise = curr['hct'] - prev['hct']
        if hct_rise >= 10:
            alerts.append(('CRITICAL', f"Plasma Leakage Alert: Hct up {hct_rise:.1f}% ({prev['hct']:.1f}% to {curr['hct']:.1f}%)", "WHO severe dengue criteria met. IV fluid resuscitation indicated."))
        elif hct_rise >= 5:
            alerts.append(('WARNING', f"Rising Hematocrit: +{hct_rise:.1f}% — Early plasma leakage possible", "Intensify fluid monitoring. Check for pleural effusion/ascites."))
        si_rise = curr['shock_index'] - prev['shock_index']
        if curr['shock_index'] >= 0.9 and si_rise > 0.1:
            alerts.append(('WARNING', f"Worsening Shock Index: {prev['shock_index']:.2f} to {curr['shock_index']:.2f}", "Assess perfusion status. Consider fluid challenge."))
        
        # Guard against 0-platelet false alerts 
        prev_u = prev.get('urine_output', 0)
        curr_u = curr.get('urine_output', 0)
        if prev_u > 0 and curr_u > 0 and curr_u < prev_u * 0.5:
            alerts.append(('WARNING',
                f"Urine Output Drop: {prev_u:.2f} to {curr_u:.2f} mL/kg/hr (50% decline)",
                "Evaluate renal perfusion. Check for fluid deficit vs. AKI."))
        
        # ── AST Alerts (NEW) ─────────────────────────
        prev_ast = prev.get('ast', 0)
        curr_ast = curr.get('ast', 0)
        if curr_ast >= 1000:
            alerts.append(('CRITICAL', f"WHO Severe Dengue criterion MET: AST {curr_ast:,} IU/L (≥1,000)",
                            "Hepatic failure risk. Immediate hepatology consult. Check coagulation profile."))
        elif curr_ast >= 500:
            alerts.append(('WARNING', f"Severe hepatitis: AST {curr_ast:,} IU/L — approaching WHO criterion",
                            "LFT trending critical. Reassess every 6 hours. Watch for coagulopathy."))
        elif prev_ast > 0 and curr_ast > 0 and curr_ast > prev_ast * 2:
            alerts.append(('WARNING', f"AST doubling: {prev_ast:,} → {curr_ast:,} IU/L ({prev['Label']} to {curr['Label']})",
                            "Rapid hepatic deterioration. Escalate LFT monitoring to every 8 hours."))

        # ── Coagulopathy Alert (NEW) ──────────────────
        curr_inr = curr.get('inr', 0)
        if curr_inr >= 1.5 and curr.get('platelets', 999999) < 50000:
            alerts.append(('CRITICAL', f"Coagulopathy pattern: INR {curr_inr:.1f} + PLT {curr.get('platelets',0):,} — DIC risk",
                            "Coagulation profile urgent. Haematology consult. Bleeding risk HIGH."))
        elif curr_inr >= 2.0:
            alerts.append(('CRITICAL', f"Critical INR {curr_inr:.1f} (≥2.0) — all invasive procedures contraindicated",
                            "Coagulation factor consumption. Avoid all invasive procedures until INR <1.5."))

        # ── AKI Alert (NEW) ──────────────────────────
        prev_cr = prev.get('creatinine', 0)
        curr_cr = curr.get('creatinine', 0)
        if prev_cr > 0 and curr_cr > 0:
            cr_rise = curr_cr - prev_cr
            if delta_hrs <= 48 and cr_rise >= 0.3:
                alerts.append(('WARNING', f"AKI Stage 1 (KDIGO): Creatinine +{cr_rise:.2f} mg/dL in {delta_hrs:.0f}h",
                                "Renal function compromised. Reduce nephrotoxic agents. Fluid balance review."))

        # ── SpO2 Alert (NEW) ─────────────────────────
        curr_spo2 = curr.get('spo2', 0)
        prev_spo2 = prev.get('spo2', 0)
        if 0 < curr_spo2 < 93:
            alerts.append(('CRITICAL', f"Hypoxaemia: SpO2 {curr_spo2}% (<93%) — respiratory compromise",
                            "Obtain CXR immediately. Assess for pleural effusion. Consider oxygen supplementation."))
        elif prev_spo2 > 0 and curr_spo2 > 0 and (prev_spo2 - curr_spo2) >= 3:
            alerts.append(('WARNING', f"SpO2 declining: {prev_spo2}% → {curr_spo2}% — early respiratory compromise",
                            "Monitor SpO2 hourly. CXR if trend continues."))

        # ── Critical Phase Entry Alert (NEW) ─────────
        illness_day = st.session_state.get('patient_illness_day', 0)
        if 4 <= illness_day <= 6:
            prev_temp = prev.get('temperature', 0)
            curr_temp = curr.get('temperature', 0)
            hct_rise_phase = curr.get('hct', 0) - prev.get('hct', 0)
            if prev_temp > 38.0 and 0 < curr_temp < 37.5 and hct_rise_phase >= 3:
                alerts.append(('CRITICAL',
                    f"CRITICAL PHASE ENTRY: Defervescence on Day {illness_day} + rising Hct (+{hct_rise_phase:.1f}%)",
                    "Paradoxical apparent improvement — MAXIMUM monitoring. Plasma leakage imminent."))
        
    latest = sorted_reports[-1]
    if latest['shock_index'] >= 0.9 and 0 < latest['platelets'] < 100000:
        alerts.append(('CRITICAL', f"DSS Risk Pattern: SI={latest['shock_index']:.2f} + PLT={latest['platelets']:,}", "Dengue Shock Syndrome precursor pattern. Immediate clinical review."))
    hct_overall  = sorted_reports[-1]['hct'] - baseline['hct']
    urine_latest = sorted_reports[-1].get('urine_output', 0)
    if hct_overall < -10 and urine_latest > 3.0 and len(sorted_reports) >= 2:
        alerts.append(('WARNING', f"Fluid Overload Risk: Hct fell {abs(hct_overall):.1f}% + Urine {urine_latest:.2f} mL/kg/hr", "Falling Hct + polyuria post-resuscitation — reduce fluid rate and reassess."))
    
    # ── Recovery Phase Positive Signal ──────────────────────────────────────
    if len(sorted_rep) >= 2:
        prev_r  = sorted_rep[-2]
        # Rising platelets — any upward trend from <50K is a meaningful positive signal
        plt_rising = (latest['platelets'] > 0 and prev_r['platelets'] > 0 and
                      latest['platelets'] > prev_r['platelets'])
        wbc_recovering = latest.get('wbc', 0) > prev_r.get('wbc', 0) > 3000
        if plt_rising and 20000 < latest['platelets'] < 100000 and wbc_recovering:
            alerts.append(('INFO',
                f"Recovery signals detected: PLT rising ({prev_r['platelets']:,} → {latest['platelets']:,}) + WBC recovering",
                "Consider discharge planning if haemodynamically stable and afebrile ≥48h."))

    # ── Critical nadir warning — still declining at dangerous level ─────────
    if 0 < latest['platelets'] < 50000:
        if len(sorted_rep) >= 2 and sorted_rep[-1]['platelets'] < sorted_rep[-2]['platelets']:
            alerts.append(('CRITICAL',
                f"Approaching Critical Nadir: PLT {latest['platelets']:,} and still declining",
                "Alert blood bank. Consider prophylactic platelet transfusion per local protocol."))
    
    return alerts

def get_top_alert(alerts):
    """
    Alert suppression: return (top_critical_or_none, remaining_alerts).
    Only 1 CRITICAL shown prominently; all others collapsed.
    Information-theoretic rationale: a signal that fires on every case carries
    near-zero information content. Surfacing only the highest-severity novel
    alert maximises signal-to-noise for the clinician.
    """
    criticals = [a for a in alerts if a[0] == 'CRITICAL']
    warnings  = [a for a in alerts if a[0] == 'WARNING']
    top       = criticals[0] if criticals else None
    rest      = criticals[1:] + warnings
    return top, rest

# ══════════════════════════════════════════════════════════
#  5b.  MODULE INTELLIGENCE — ACTIVATION STATUS ENGINE
#
#  Computes which analysis modules are active/inactive
#  given the current set of saved reports and patient data.
#  Used by both the Preview panel and the Analysis header.
# ══════════════════════════════════════════════════════════
def compute_module_status(sorted_reports: list, age: int, sex: str,
                           enable_metrics: bool = False) -> list:
    """
    Returns a list of dicts:
      { 'module', 'status', 'reason', 'color', 'icon' }
    status: 'active' | 'partial' | 'inactive'
    """
    n          = len(sorted_reports)
    latest     = sorted_reports[-1] if n > 0 else {}
    has_plt    = any(r.get('platelets', 0) > 0 for r in sorted_reports)
    n_plt      = sum(1 for r in sorted_reports if r.get('platelets', 0) > 0)
    has_2plus  = n >= 2
    has_2plt   = n_plt >= 2
    has_lft    = latest.get('ast', 0) > 0 or latest.get('alt', 0) > 0
    has_coag   = latest.get('inr', 0) > 0 or latest.get('d_dimer', 0) > 0
    has_renal  = latest.get('creatinine', 0) > 0
    has_ser    = any(latest.get(k, 'Not Done') not in ('Not Done', '')
                     for k in ('ns1', 'igm', 'igg'))
    has_img    = (latest.get('pleural_effusion', False) or
                  latest.get('gallbladder_wall_mm', 0) > 0 or
                  latest.get('ascites_grade', 0) > 0)

    modules = [
        {
            'module': 'WHO 2009 Classification',
            'status': 'active',
            'reason': 'Active on any saved report — uses BP, HR, Hct, WHO signs',
            'color':  '#2ecc71', 'icon': '✔',
        },
        {
            'module': 'Continuous Severity Score',
            'status': 'active',
            'reason': 'Active on any saved report — composite of 4 physiological domains',
            'color':  '#2ecc71', 'icon': '✔',
        },
        {
            'module': 'Hemodynamics & Treatment',
            'status': 'active' if enable_metrics else 'partial',
            'reason': ('Active — fluid rate calculated from body weight'
                       if enable_metrics
                       else 'Partial — MAP/SI/PP shown; fluid rate needs weight in Body Metrics'),
            'color':  '#2ecc71' if enable_metrics else '#f39c12', 'icon': '✔' if enable_metrics else '◑',
        },
        {
            'module': 'Serial Alert System',
            'status': 'active' if has_2plus else 'inactive',
            'reason': ('Active — monitoring changes across all saved reports'
                       if has_2plus
                       else 'Needs ≥ 2 saved reports to detect inter-report changes'),
            'color':  '#2ecc71' if has_2plus else '#e74c3c', 'icon': '✔' if has_2plus else '✗',
        },
        {
            'module': 'Trajectory Engine (PLT/Hct/SI)',
            'status': 'active' if has_2plus else 'inactive',
            'reason': ('Active — OLS regression computing velocity, acceleration, countdowns'
                       if has_2plus
                       else 'Needs ≥ 2 reports to compute velocity and time-to-threshold'),
            'color':  '#2ecc71' if has_2plus else '#e74c3c', 'icon': '✔' if has_2plus else '✗',
        },
        {
            'module': 'Personal Baseline Delta Analysis',
            'status': 'active' if has_2plus else 'inactive',
            'reason': ('Active — tracking % change from patient\'s own first report'
                       if has_2plus
                       else 'Needs ≥ 2 reports — compares current vs patient\'s own baseline'),
            'color':  '#2ecc71' if has_2plus else '#e74c3c', 'icon': '✔' if has_2plus else '✗',
        },
        {
            'module': 'AI Risk Model (Random Forest)',
            'status': ('active' if has_plt else 'inactive'),
            'reason': (f'Active — using platelet count from Report '
                       f'{next((r["Label"] for r in reversed(sorted_reports) if r.get("platelets",0)>0), "?")}'
                       if has_plt
                       else 'Needs Platelet Count in at least 1 report — #1 feature (MDI 0.282)'),
            'color':  '#2ecc71' if has_plt else '#e74c3c', 'icon': '✔' if has_plt else '✗',
        },
        {
            'module': 'Platelet Trajectory & 24h Forecast',
            'status': ('active' if has_2plt else
                       ('partial' if has_plt else 'inactive')),
            'reason': (f'Active — linear regression on {n_plt} reports with platelet data'
                       if has_2plt
                       else (f'Partial — {n_plt} report with platelets found; needs ≥ 2 to fit regression'
                             if has_plt
                             else 'Needs Platelet Count in ≥ 2 reports for regression and forecast')),
            'color':  ('#2ecc71' if has_2plt else '#f39c12' if has_plt else '#e74c3c'),
            'icon':   ('✔' if has_2plt else '◑' if has_plt else '✗'),
        },
        {
            'module': 'Extended Organ Panels (LFT/Coag/Renal)',
            'status': ('active' if (has_lft or has_coag or has_renal) else 'inactive'),
            'reason': (('Active panels: ' +
                        ', '.join(filter(None, [
                            'Hepatic' if has_lft else '',
                            'Coagulation' if has_coag else '',
                            'Renal' if has_renal else '',
                        ])))
                       if (has_lft or has_coag or has_renal)
                       else 'Enter AST/ALT, INR/D-dimer, or Creatinine to activate organ panels'),
            'color':  '#2ecc71' if (has_lft or has_coag or has_renal) else '#8b92a8',
            'icon':   '✔' if (has_lft or has_coag or has_renal) else '○',
        },
        {
            'module': 'Plasma Leakage Score',
            'status': ('active' if (has_img or has_coag or latest.get('albumin', 0) > 0) else 'inactive'),
            'reason': (('Active — scoring from: ' +
                        ', '.join(filter(None, [
                            'Imaging' if has_img else '',
                            'D-dimer' if latest.get('d_dimer',0)>0 else '',
                            'Albumin' if latest.get('albumin',0)>0 else '',
                            'Pulse pressure' if latest.get('pp', latest.get('sys_bp',120)-latest.get('dia_bp',80)) <= 30 else '',
                        ])) or 'Hct rise vs baseline')
                       if (has_img or has_coag or latest.get('albumin', 0) > 0)
                       else 'Enter any of: Hct baseline, D-dimer, Albumin, Imaging findings'),
            'color':  '#2ecc71' if (has_img or has_coag) else '#8b92a8',
            'icon':   '✔' if (has_img or has_coag) else '○',
        },
        {
            'module': 'Serology Integration',
            'status': 'active' if has_ser else 'inactive',
            'reason': ('Active — NS1/IgM/IgG results integrated into WHO classification and BRS'
                       if has_ser
                       else 'Enter NS1, IgM, or IgG result in the Serology panel (sidebar)'),
            'color':  '#2ecc71' if has_ser else '#8b92a8', 'icon': '✔' if has_ser else '○',
        },
        {
            'module': 'Discharge Readiness Checklist',
            'status': 'active',
            'reason': 'Always active — 5 of 7 criteria assessed automatically from saved data',
            'color':  '#2ecc71', 'icon': '✔',
        },
    ]
    return modules

# ══════════════════════════════════════════════════════════
#  6.  DISCHARGE READINESS CHECKLIST
# ══════════════════════════════════════════════════════════
def check_discharge_readiness(latest_report, prev_platelet=None, fever_free_hours=None, tolerating_orals=None):
    criteria = []
    if fever_free_hours is not None:
        passed = fever_free_hours >= 48
        criteria.append(("Afebrile >=48 hours", passed,
            f"{fever_free_hours:.0f} hours fever-free" if passed else f"Only {fever_free_hours:.0f} hours fever-free (need 48)"))
    else:
        criteria.append(("Afebrile >=48 hours", None, "Not recorded — ask patient"))
    if prev_platelet is not None and prev_platelet > 0:
        plt_improving = latest_report['platelets'] > prev_platelet
        criteria.append(("Platelet trend improving", plt_improving,
            f"{prev_platelet:,} to {latest_report['platelets']:,} ({'Improving' if plt_improving else 'Still declining'})"))
    else:
        criteria.append(("Platelet trend improving", None, "Need >=2 serial counts to assess"))
    plt_safe = latest_report['platelets'] >= 50000
    criteria.append(("Platelets >=50,000 cells/uL", plt_safe,
        f"Current: {latest_report['platelets']:,} ({'OK' if plt_safe else 'Below threshold'})"))
    hemo_stable = (latest_report['shock_index'] < 0.9 and latest_report['map'] >= 65 and
                   (latest_report['sys_bp'] - latest_report['dia_bp']) > 20)
    criteria.append(("Haemodynamics stable", hemo_stable,
        f"SI={latest_report['shock_index']:.2f}, MAP={latest_report['map']:.1f}, PP={latest_report['sys_bp']-latest_report['dia_bp']} mmHg"))
    uo = latest_report.get('urine_output', 0)
    if uo > 0:
        uo_ok = 0.5 <= uo <= 4.0
        criteria.append(("Adequate urine output (0.5-4.0 mL/kg/hr)", uo_ok,
            f"{uo:.2f} mL/kg/hr — {'Adequate' if uo_ok else 'Outside target range'}"))
    else:
        criteria.append(("Adequate urine output", None, "Not recorded"))
    no_warning = len(latest_report.get('who_signs', [])) == 0
    criteria.append(("No active WHO warning signs", no_warning,
        "None present" if no_warning else f"Still present: {', '.join(latest_report.get('who_signs', []))}"))
    if tolerating_orals is not None:
        criteria.append(("Tolerating oral fluids", tolerating_orals,
            "Yes" if tolerating_orals else "No — oral tolerance required before discharge"))
    else:
        criteria.append(("Tolerating oral fluids", None, "Not recorded — assess clinically"))
    known   = [c for c in criteria if c[1] is not None]
    all_pass = all(c[1] for c in known) and len(known) >= 4
    return criteria, all_pass

# ══════════════════════════════════════════════════════════
#  7.  ADVANCED ANALYTICAL ENGINES
#
#  These modules treat dengue as a dynamical system — a
#  time-evolving process with phase transitions — rather
#  than a static classification problem.
# ══════════════════════════════════════════════════════════

# ── 7a. Trajectory Engine ─────────────────────────────────
# Computes OLS linear + quadratic fits over serial CBCs.
# Outputs velocity (dy/dt), acceleration (d²y/dt²), R²,
# and time-to-threshold for each critical parameter.
#
# Linear model:  y = a + b*t           velocity = b
# Quadratic:     y = a + b*t + c*t²    acceleration = 2c
# Time-to-threshold: t* = (theta - y_current) / velocity
#   (linear approximation; undefined if velocity == 0)
# ──────────────────────────────────────────────────────────
THRESHOLDS = {
    'platelets': [
        (100000, "PLT <100k (warning)"),
        (50000,  "PLT <50k (severe thrombocytopenia)"),
        (20000,  "PLT <20k (critical nadir)"),
    ],
    'shock_index': [
        (0.9, "SI 0.9 (elevated)"),
        (1.0, "SI 1.0 (shock)"),
    ],
    'hct': [
        (None, None),   # rising hct — handled separately
    ],
}

def compute_trajectory(sorted_reports, param='platelets'):
    """
    Returns dict with keys:
      velocity, acceleration, r2_linear, r2_quad,
      countdowns  — list of (threshold_label, hours_remaining) or None
    Requires >= 2 reports.
    """
    if len(sorted_reports) < 2:
        return None
    t0 = sorted_reports[0]['datetime']
    days = np.array(
        [(r['datetime'] - t0).total_seconds() / 86400 for r in sorted_reports])
    vals = np.array([r[param] for r in sorted_reports], dtype=float)

    # Linear fit
    A_lin = np.vstack([np.ones_like(days), days]).T
    coef_lin, *_ = np.linalg.lstsq(A_lin, vals, rcond=None)
    velocity = coef_lin[1]   # units/day
    y_lin    = A_lin @ coef_lin
    ss_res   = np.sum((vals - y_lin) ** 2)
    ss_tot   = np.sum((vals - vals.mean()) ** 2)
    r2_lin   = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Quadratic fit (needs >= 3 points for meaningful curvature)
    acceleration = 0.0
    r2_quad      = r2_lin
    if len(sorted_reports) >= 3:
        A_quad = np.vstack([np.ones_like(days), days, days**2]).T
        coef_q, *_ = np.linalg.lstsq(A_quad, vals, rcond=None)
        acceleration = 2 * coef_q[2]   # 2c
        y_quad = A_quad @ coef_q
        ss_q   = np.sum((vals - y_quad) ** 2)
        r2_quad = 1 - ss_q / ss_tot if ss_tot > 0 else 0.0

    current_val  = float(vals[-1])
    current_day  = float(days[-1])

    # Time-to-threshold countdowns
    countdowns = []
    if param in THRESHOLDS:
        for theta, label in THRESHOLDS[param]:
            if theta is None or velocity == 0:
                continue
            delta = theta - current_val
            # Only compute countdown if heading toward threshold
            if (delta < 0 and velocity < 0) or (delta > 0 and velocity > 0):
                days_remaining  = delta / velocity
                hours_remaining = days_remaining * 24
                if 0 < hours_remaining < 168:   # cap at 7 days
                    countdowns.append((label, hours_remaining))
            elif delta > 0 and velocity <= 0:
                countdowns.append((label, None))   # already past / stabilised

    return {
        'velocity':     velocity,
        'acceleration': acceleration,
        'r2_linear':    r2_lin,
        'r2_quad':      r2_quad,
        'countdowns':   countdowns,
        'current_val':  current_val,
        'days_array':   days,
        'vals_array':   vals,
    }

# ── 7b. Continuous Severity Score ─────────────────────────
# 0–100 composite score capturing four physiological domains:
#   Haematological  (40%)   — PLT + Hct deviation
#   Haemodynamic    (30%)   — Shock Index + MAP
#   Trend velocity  (20%)   — PLT velocity + Hct velocity
#   Urine output    (10%)   — proximity to oliguria
#
# Direction vector derived from change in score between
# consecutive reports (or single-report heuristic).
# ──────────────────────────────────────────────────────────
def compute_severity_score(sorted_reports):
    """
    Returns (score_0_100, direction_label, component_dict)
    direction_label: 'Improving' | 'Stable' | 'Deteriorating' | 'Rapidly Deteriorating'
    """
    if not sorted_reports:
        return 0, "Unknown", {}

    latest = sorted_reports[-1]

    # Haematological component (0-40)
    plt = latest['platelets']
    if   plt >= 150000: plt_s = 0
    elif plt >= 100000: plt_s = 10
    elif plt >= 50000:  plt_s = 25
    elif plt >= 20000:  plt_s = 35
    else:               plt_s = 40

    hct_dev = latest['hct'] - 40.0   # deviation from normal midpoint
    hct_s   = min(max(hct_dev * 1.2, 0), 10)   # cap at 10 pts

    haem_score = plt_s + hct_s    # max 50, cap to 40
    haem_score = min(haem_score, 40)

    # Haemodynamic component (0-30)
    si   = latest['shock_index']
    map_ = latest['map']
    if   si >= 1.2:          si_s = 20
    elif si >= 1.0:          si_s = 15
    elif si >= 0.9:          si_s = 10
    else:                    si_s = 0
    if   map_ < 60:          map_s = 10
    elif map_ < 65:          map_s = 7
    elif map_ < 70:          map_s = 3
    else:                    map_s = 0
    hemo_score = min(si_s + map_s, 30)

    # Trend component (0-20) — only if >=2 reports
    trend_score = 0
    if len(sorted_reports) >= 2:
        traj = compute_trajectory(sorted_reports, 'platelets')
        if traj:
            vel = traj['velocity']
            if   vel < -50000: trend_score += 12
            elif vel < -20000: trend_score += 8
            elif vel < -5000:  trend_score += 4
        hct_traj = compute_trajectory(sorted_reports, 'hct')
        if hct_traj:
            hct_vel = hct_traj['velocity']
            if   hct_vel > 5:  trend_score += 8
            elif hct_vel > 2:  trend_score += 4
    trend_score = min(trend_score, 20)

    # Urine component (0-10)
    uo = latest.get('urine_output', 0)
    if   uo <= 0:           uo_s = 0    # not recorded, neutral
    elif uo < 0.5:          uo_s = 10
    elif uo < 1.0:          uo_s = 6
    elif uo <= 2.0:         uo_s = 0
    elif uo <= 4.0:         uo_s = 1
    else:                   uo_s = 4
    uo_score = uo_s

    total = haem_score + hemo_score + trend_score + uo_score

    # Direction vector
    direction = "Stable"
    if len(sorted_reports) >= 2:
        prev_report_list = sorted_reports[:-1]
        prev_score, *_ = compute_severity_score(prev_report_list)
        delta = total - prev_score
        if   delta <= -5:  direction = "Improving"
        elif delta <= 2:   direction = "Stable"
        elif delta <= 8:   direction = "Deteriorating"
        else:              direction = "Rapidly Deteriorating"
    else:
        if   total >= 70: direction = "Rapidly Deteriorating"
        elif total >= 45: direction = "Deteriorating"
        elif total >= 20: direction = "Stable"
        else:             direction = "Improving"

    components = {
        'Haematological': haem_score,
        'Haemodynamic':   hemo_score,
        'Trend Velocity': trend_score,
        'Urine Output':   uo_score,
    }
    return total, direction, components

# ── 7c. Personal Baseline Delta Analysis ──────────────────
# All parameters evaluated against the patient's own first
# report, not population reference ranges.
# delta_pct = (current - baseline) / baseline * 100
# ──────────────────────────────────────────────────────────
def compute_personal_deltas(sorted_reports):
    """
    Returns list of dicts per report (after baseline):
      {param, baseline_val, current_val, delta_abs, delta_pct, flag}
    flag: 'critical' | 'warning' | 'normal'
    """
    if len(sorted_reports) < 2:
        return []
    baseline = sorted_reports[0]
    results  = []
    params = [
        ('platelets', 'Platelets',   -20, -10, None),    # % thresholds: crit_neg, warn_neg, warn_pos
        ('hct',       'Hematocrit',   20,  10,   5),
        ('hb',        'Haemoglobin', -20, -10, None),
        ('shock_index','Shock Index',  20,  10, None),
    ]
    for key, label, crit_neg, warn_neg, warn_pos in params:
        bv = baseline.get(key, 0)
        if bv == 0:
            continue
        cv = sorted_reports[-1].get(key, 0)
        d_abs = cv - bv
        d_pct = (d_abs / bv) * 100 if bv != 0 else 0
        flag = 'normal'
        if crit_neg and d_pct <= crit_neg:     flag = 'critical'
        elif warn_neg and d_pct <= warn_neg:   flag = 'warning'
        elif warn_pos and d_pct >= warn_pos:   flag = 'warning'
        if key == 'shock_index' and d_pct >= 20: flag = 'warning'
        results.append({
            'param':        label,
            'baseline_val': bv,
            'current_val':  cv,
            'delta_abs':    d_abs,
            'delta_pct':    d_pct,
            'flag':         flag,
        })
    return results

# ── 7d. Out-of-Distribution Detection ─────────────────────
# Diagonal Mahalanobis distance: D = sqrt(mean(z_i^2))
# where z_i = (x_i - mu_i) / sigma_i
# Training distribution statistics (approximate, from n=2,455):
# ──────────────────────────────────────────────────────────
TRAIN_STATS = {
    'platelets': (114536.0, 76626.6),
    'hb':        (13.4,  1.8),
    'hct':       (39.9,  5.4),
    'rbc':       (4.9,   0.7),
    'hr':        (92.8,  21.9),
    'sys_bp':    (107.2, 18.0),
    'dia_bp':    (66.6,  12.2),
    'age':       (34.5,  17.9),
}

def compute_ood_score(report, age):
    """
    Returns (D, z_scores_dict, is_ood_bool)
    D > 2.5 flags out-of-distribution.
    """
    inputs = {
        'platelets': report['platelets'],
        'hb':        report['hb'],
        'hct':       report['hct'],
        'rbc':       report['rbc'],
        'hr':        report['hr'],
        'sys_bp':    report['sys_bp'],
        'dia_bp':    report['dia_bp'],
        'age':       age,
    }
    z_scores = {}
    sq_sum   = 0.0
    for k, v in inputs.items():
        if k in TRAIN_STATS:
            mu, sigma = TRAIN_STATS[k]
            z = (v - mu) / sigma if sigma > 0 else 0.0
            z_scores[k] = z
            sq_sum += z ** 2
    D = np.sqrt(sq_sum / len(z_scores)) if z_scores else 0.0
    return D, z_scores, D > 2.5


# ── 7e. Random Forest Confidence Interval ─────────────────
# 95% CI from the ensemble of individual tree predictions.
# Each tree casts a vote; percentile(2.5) and percentile(97.5)
# of the vote distribution form the interval.
# Wide CI (>40 pp) indicates high tree disagreement —
# the prediction is unreliable and clinical judgment dominates.
# ──────────────────────────────────────────────────────────
def compute_tree_ci(classifier, df_input):
    """
    Returns (point_estimate, ci_lower, ci_upper, ci_width)
    """
    try:
        tree_probs = np.array([
            t.predict_proba(df_input.values if hasattr(df_input, "values") else df_input)[0][1]
            for t in classifier.estimators_
        ])
        lo  = float(np.percentile(tree_probs, 2.5))
        hi  = float(np.percentile(tree_probs, 97.5))
        est = float(np.mean(tree_probs))
        return est, lo, hi, hi - lo
    except Exception:
        pt = classifier.predict_proba(df_input.values if hasattr(df_input, "values") else df_input)[0][1]
        return float(pt), max(0, float(pt) - 0.15), min(1, float(pt) + 0.15), 0.30

# ══════════════════════════════════════════════════════════
#  8.  OCR PARSING ENGINE
# ══════════════════════════════════════════════════════════
def parse_indian_global_number(s):
    s = s.strip().replace(' ', '')
    s = re.sub(r'[a-zA-Z°%]+$', '', s)
    if re.match(r'^\d{1,2},\d{2},\d{3}$', s):
        return float(s.replace(',', ''))
    parts = s.split(',')
    if len(parts) > 1:
        if re.match(r'^\d{3}$', parts[-1]):
            return float(s.replace(',', ''))
        elif re.match(r'^\d{1,2}$', parts[-1]):
            return float('.'.join([parts[0], parts[-1]]))
    return float(s.replace(',', ''))


def _plausibility_check(field, value):
    checks = {
        'platelets':  (1000,    1000000, "Expected 1,000-1,000,000 cells/uL"),
        'hb':         (2.0,     25.0,    "Expected 2-25 g/dL"),
        'hct':        (5.0,     70.0,    "Expected 5-70%"),
        'rbc':        (0.5,     10.0,    "Expected 0.5-10.0 M/uL"),
        'sys':        (50,      250,     "Expected SBP 50-250 mmHg"),
        'dia':        (30,      150,     "Expected DBP 30-150 mmHg"),
        'hr':         (30,      220,     "Expected HR 30-220 bpm"),
        'age':        (0,       120,     "Expected age 0-120 years"),
        'weight':     (1.0,     300.0,   "Expected weight 1-300 kg"),
        'height':     (30.0,    250.0,   "Expected height 30-250 cm"),
        'bmi':        (10.0,    70.0,    "Expected BMI 10-70"),
        'urine_vol':  (0.0,     5000.0,  "Expected urine volume 0-5000 mL"),
        'urine_time': (0.1,     24.0,    "Expected urine collection time 0.1-24 hrs"),
        'time_hour':  (0,       23,      "Expected hour 0-23"),
        'time_minute':(0,       59,      "Expected minute 0-59"),
    }
    if field not in checks:
        return True, 0.0, ""
    lo, hi, note = checks[field]
    if lo <= value <= hi:
        return True, 0.0, ""
    return False, -0.5, f"Outside plausible range ({note}): got {value}"

def _regex_extract(text, patterns, field_name):
    for pattern, conf in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            raw = matches[0].strip() if isinstance(matches[0], str) else matches[0][-1].strip()
            try:
                value = parse_indian_global_number(str(raw))
                plausible, adj, note = _plausibility_check(field_name, value)
                final_conf = max(0.0, conf + adj)
                return value, raw, final_conf, note
            except (ValueError, TypeError):
                continue
    return None, None, 0.0, ""

def parse_time_from_text(text):
    text_n = re.sub(r'\s+', ' ', text)
    explicit_patterns = [
        (r'(?:Sample\s*Time|Collection\s*Time|Report\s*Time|Time\s*of\s*Collection|Time)\s*[:/]?\s*(\d{1,2})[:\.](\d{2})\s*(AM|PM|am|pm)?', 0.98),
    ]
    for pattern, conf in explicit_patterns:
        m = re.search(pattern, text_n, re.IGNORECASE)
        if m:
            hr, mn = int(m.group(1)), int(m.group(2))
            ampm = m.group(3).upper().strip() if m.group(3) else None
            if ampm == 'PM' and hr != 12: hr += 12
            elif ampm == 'AM' and hr == 12: hr = 0
            if 0 <= hr <= 23 and 0 <= mn <= 59:
                return hr, mn, m.group(0).strip(), conf
    ampm_pattern = re.findall(r'\b(\d{1,2})[:\.](\d{2})\s*(AM|PM|am|pm)\b', text_n)
    if ampm_pattern:
        hr, mn, ampm = int(ampm_pattern[0][0]), int(ampm_pattern[0][1]), ampm_pattern[0][2].upper()
        if ampm == 'PM' and hr != 12: hr += 12
        elif ampm == 'AM' and hr == 12: hr = 0
        if 0 <= hr <= 23 and 0 <= mn <= 59:
            return hr, mn, f"{ampm_pattern[0][0]}:{ampm_pattern[0][1]} {ampm}", 0.92
    hr24_pattern = re.findall(r'(?<![/\-\d])(\d{2})[:](\d{2})(?![/\-\d])', text_n)
    for hr_s, mn_s in hr24_pattern:
        hr, mn = int(hr_s), int(mn_s)
        if 0 <= hr <= 23 and 0 <= mn <= 59:
            return hr, mn, f"{hr_s}:{mn_s}", 0.80
    return None

def parse_lab_report_text(text):
    results = {}
    text_n = re.sub(r'\s+', ' ', text)
    text_n = re.sub(r'[|]', ' ', text_n)

    name_patterns = [
        r'(?:Patient\s*Name|Name\s*of\s*Patient|Pt\.?\s*Name|Patient)[:\s]+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})',
        r'(?:Name)[:\s]+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})',
        r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})',
    ]
    for pat in name_patterns:
        nm = re.search(pat, text_n, re.IGNORECASE)
        if nm:
            candidate = nm.group(1).strip().title()
            blocklist = {'Blood', 'Report', 'Test', 'Result', 'Laboratory', 'Pathology',
                         'Clinical', 'Medical', 'Centre', 'Hospital', 'Health'}
            if candidate.split()[0] not in blocklist and len(candidate) >= 4:
                results['patient_name'] = (candidate, 0.88, candidate, "")
                break

    time_result = parse_time_from_text(text_n)
    if time_result:
        hr24, mn, raw_time, conf_time = time_result
        ampm_str = "AM" if hr24 < 12 else "PM"
        results['time_hour']   = (hr24, conf_time, raw_time, "24hr stored")
        results['time_minute'] = (mn,   conf_time, raw_time, "")
        results['time_ampm']   = (ampm_str, conf_time, raw_time, "")

    plt_patterns = [
        (r'[Pp]latelet[s]?\s*(?:count|Count|COUNT)?\s*[:/]?\s*([\d,\.]+)', 1.0),
        (r'\bPLT\s*[:/]?\s*([\d,\.]+)', 0.85),
        (r'\bTHROMBO(?:CYTE)?S?\s*[:/]?\s*([\d,\.]+)', 0.85),
        (r'[Pp][Cc]\s*[:/]?\s*([\d,\.]+)', 0.65),
    ]
    v, raw, conf, note = _regex_extract(text_n, plt_patterns, 'platelets')
    if v: results['platelets'] = (int(v), conf, raw, note)

    hb_patterns = [
        (r'[Hh]a?e?mo(?:globin)?\s*[:/]?\s*([\d,\.]+)', 1.0),
        (r'\bHGB\s*[:/]?\s*([\d,\.]+)', 0.85),
        (r'\bHb\b\s*[:/]?\s*([\d,\.]+)', 0.80),
    ]
    v, raw, conf, note = _regex_extract(text_n, hb_patterns, 'hb')
    if v: results['hb'] = (round(v, 1), conf, raw, note)

    hct_patterns = [
        (r'[Hh]ematocrit\s*[:/]?\s*([\d,\.]+)', 1.0),
        (r'[Hh]aematocrit\s*[:/]?\s*([\d,\.]+)', 1.0),
        (r'\bPCV\s*[:/]?\s*([\d,\.]+)', 0.90),
        (r'\bHCT\s*[:/]?\s*([\d,\.]+)', 0.85),
    ]
    v, raw, conf, note = _regex_extract(text_n, hct_patterns, 'hct')
    if v: results['hct'] = (round(v, 1), conf, raw, note)

    rbc_patterns = [
        (r'[Rr]ed\s*[Bb]lood\s*[Cc]ell[s]?\s*(?:[Cc]ount)?\s*[:/]?\s*([\d,\.]+)', 1.0),
        (r'[Ee]rythrocyte[s]?\s*[:/]?\s*([\d,\.]+)', 0.90),
        (r'\bRBC\s*[:/]?\s*([\d,\.]+)', 0.85),
    ]
    v, raw, conf, note = _regex_extract(text_n, rbc_patterns, 'rbc')
    if v: results['rbc'] = (round(v, 2), conf, raw, note)

    bp_combo = re.findall(
        r'(?:[Bb][Pp]|[Bb]lood\s*[Pp]ressure)\s*[:/]?\s*(\d{2,3})\s*/\s*(\d{2,3})', text_n)
    if bp_combo:
        sys_v, dia_v = int(bp_combo[0][0]), int(bp_combo[0][1])
        results['sys'] = (sys_v, 0.95, f"{sys_v}", "")
        results['dia'] = (dia_v, 0.95, f"{dia_v}", "")
    else:
        sys_pats = [(r'[Ss]ystolic\s*[:/]?\s*([\d,\.]+)', 1.0), (r'[Ss][Bb][Pp]\s*[:/]?\s*([\d,\.]+)', 0.90)]
        dia_pats = [(r'[Dd]iastolic\s*[:/]?\s*([\d,\.]+)', 1.0), (r'[Dd][Bb][Pp]\s*[:/]?\s*([\d,\.]+)', 0.90)]
        v, raw, conf, note = _regex_extract(text_n, sys_pats, 'sys')
        if v: results['sys'] = (int(v), conf, raw, note)
        v, raw, conf, note = _regex_extract(text_n, dia_pats, 'dia')
        if v: results['dia'] = (int(v), conf, raw, note)

    hr_patterns = [
        (r'(?:[Hh]eart\s*[Rr]ate|[Pp]ulse\s*[Rr]ate)\s*[:/]?\s*([\d,\.]+)', 1.0),
        (r'\bHR\b\s*[:/]?\s*([\d,\.]+)', 0.85),
        (r'\b[Pp]ulse\b\s*[:/]?\s*([\d,\.]+)', 0.80),
    ]
    v, raw, conf, note = _regex_extract(text_n, hr_patterns, 'hr')
    if v: results['hr'] = (int(v), conf, raw, note)

    age_patterns = [
        (r'[Aa]ge\s*[:/]?\s*(\d{1,3})\s*(?:years?|yrs?|Y)?', 1.0),
        (r'(\d{1,3})\s*(?:years?\s*old|yr\s*old)', 0.85),
    ]
    v, raw, conf, note = _regex_extract(text_n, age_patterns, 'age')
    if v: results['age'] = (int(v), conf, raw, note)

    sex_match = re.search(
        r'(?:Sex|Gender|Patient\s*Sex)\s*[:/]?\s*(Male|Female|M\b|F\b)', text_n, re.IGNORECASE)
    if not sex_match:
        sex_match = re.search(r'\b(Male|Female)\b', text_n, re.IGNORECASE)
    if sex_match:
        raw_sex = sex_match.group(1).strip().upper()
        sex_val = "Male" if raw_sex in ("MALE", "M") else "Female"
        results['sex'] = (sex_val, 0.90, sex_match.group(1), "")

    # ── Para (weight, height, bmi, urine, date, notes, return)
    # ── Weight — detect unit from text ───────────────────────────────────────
    
    _detected_w_unit = 'kg'   # default
    weight_kg_patterns = [
        (r'[Ww]eight\s*[:/]?\s*([\d,\.]+)\s*(?:kg|Kg|KG)\b',   1.0, 'kg'),
        (r'[Ww]t\.?\s*[:/]?\s*([\d,\.]+)\s*(?:kg|Kg|KG)\b',    0.90, 'kg'),
        (r'[Bb]ody\s*[Ww]eight\s*[:/]?\s*([\d,\.]+)\s*(?:kg|Kg|KG)\b', 0.90, 'kg'),
    ]
    weight_lbs_patterns = [
        (r'[Ww]eight\s*[:/]?\s*([\d,\.]+)\s*(?:lbs?|LBS?|pounds?)\b', 1.0, 'lbs'),
        (r'[Ww]t\.?\s*[:/]?\s*([\d,\.]+)\s*(?:lbs?|LBS?)\b',          0.90, 'lbs'),
    ]
    weight_generic = [
        (r'[Bb]ody\s*[Ww]eight\s*[:/]?\s*([\d,\.]+)',  0.75, 'kg'),
        (r'[Ww]eight\s*[:/]?\s*([\d,\.]+)',             0.65, 'kg'),
    ]
    _found_weight = False
    for pat_list in [weight_kg_patterns, weight_lbs_patterns, weight_generic]:
        for pat, conf, unit in pat_list:
            m = re.search(pat, text_n, re.IGNORECASE)
            if m:
                try:
                    raw_w = m.group(1).strip()
                    w_val = parse_indian_global_number(raw_w)
                    w_kg  = w_val * 0.453592 if unit == 'lbs' else w_val
                    plaus, adj, note = _plausibility_check('weight', w_kg)
                    if plaus:
                        results['weight'] = (round(w_kg, 1), max(0.0, conf + adj), raw_w,
                                             f"Detected as {unit}" + (f"; {note}" if note else ""))
                        results['_detected_weight_unit'] = (unit, 1.0, unit, "")
                        _detected_w_unit = unit
                        _found_weight = True
                        break
                except (ValueError, TypeError):
                    continue
        if _found_weight:
            break

    # ── Height — detect cm / ft+in ────────────────────────────────────────
    _detected_h_unit = 'cm'
    # Try ft/in first (e.g. "5'10\"", "5 ft 10 in", "Height: 5.10 ft")
    ft_in_m = re.search(
        r"[Hh](?:eight|t)\.?\s*[:/]?\s*(\d)\s*['\u2032ft]\s*(\d{1,2})\s*(?:\"|\u2033|in)?",
        text_n)
    if not ft_in_m:
        ft_in_m = re.search(
            r"(\d)\s*(?:feet?|ft)\s*(\d{1,2})\s*(?:inches?|in)\b",
            text_n, re.IGNORECASE)
    if ft_in_m:
        ft_n, in_n = int(ft_in_m.group(1)), int(ft_in_m.group(2))
        h_cm = ft_n * 30.48 + in_n * 2.54
        plaus, adj, note = _plausibility_check('height', h_cm)
        if plaus:
            results['height'] = (round(h_cm, 1), max(0.0, 0.95 + adj),
                                 ft_in_m.group(0), "Converted from ft/in")
            results['_detected_height_unit'] = ('ft/in', 1.0, 'ft/in', "")
            _detected_h_unit = 'ft/in'
    else:
        height_cm_patterns = [
            (r'[Hh]eight\s*[:/]?\s*([\d,\.]+)\s*(?:cm|CM)\b', 1.0),
            (r'[Hh]t\.?\s*[:/]?\s*([\d,\.]+)\s*(?:cm|CM)\b',  0.90),
            (r'[Hh]eight\s*[:/]?\s*([\d,\.]+)',                0.70),
        ]
        v, raw, conf, note = _regex_extract(text_n, height_cm_patterns, 'height')
        if v:
            results['height'] = (round(v, 1), conf, raw, note)
            results['_detected_height_unit'] = ('cm', conf, raw, "")

    # ── BMI ────────────────────────────────────────────────────────────────
    bmi_patterns = [
        (r'BMI\s*[:/]?\s*([\d,\.]+)', 1.0),
        (r'[Bb]ody\s*[Mm]ass\s*[Ii]ndex\s*[:/]?\s*([\d,\.]+)', 1.0),
    ]
    v, raw, conf, note = _regex_extract(text_n, bmi_patterns, 'bmi')
    if v:
        results['bmi'] = (round(v, 1), conf, raw, note)

    # ── Temperature — detect °C or °F ─────────────────────────────────────
    temp_f_patterns = [
        r'(?:[Tt]emp(?:erature)?)\s*[:/]?\s*([\d\.]+)\s*°?\s*F\b',
        r'(1(?:0[0-9]|1[0-9])(?:\.\d)?)\s*°?\s*F\b',   # 100-119°F range
    ]
    temp_c_patterns = [
        r'(?:[Tt]emp(?:erature)?)\s*[:/]?\s*([\d\.]+)\s*°?\s*C\b',
        r'\b(3[5-9]|4[0-3])(?:\.\d)?\s*°?\s*C\b',       # 35-43°C range
    ]
    _found_temp = False
    for pat in temp_f_patterns:
        m = re.search(pat, text_n, re.IGNORECASE)
        if m:
            try:
                tf = float(m.group(1))
                tc = round((tf - 32) * 5/9, 1)
                if 35.0 <= tc <= 43.0:
                    results['temperature']         = (tc,  0.95, m.group(0), f"Converted from {tf}°F")
                    results['_detected_temp_unit'] = ('°F', 1.0, '°F', "")
                    _found_temp = True
                    break
            except (ValueError, TypeError):
                continue
    if not _found_temp:
        for pat in temp_c_patterns:
            m = re.search(pat, text_n, re.IGNORECASE)
            if m:
                try:
                    tc = float(m.group(1))
                    if 35.0 <= tc <= 43.0:
                        results['temperature']         = (tc,  0.90, m.group(0), "")
                        results['_detected_temp_unit'] = ('°C', 1.0, '°C', "")
                        _found_temp = True
                        break
                except (ValueError, TypeError):
                    continue

    # ── SpO2 ────────────────────────────────────────────────────────────────
    spo2_m = re.search(r'(?:SpO2|Spo2|sp\.?o2|oxygen\s*sat(?:uration)?)\s*[:/]?\s*(\d{2,3})\s*%?',
                       text_n, re.IGNORECASE)
    if spo2_m:
        try:
            sv = int(spo2_m.group(1))
            if 70 <= sv <= 100:
                results['spo2'] = (sv, 0.92, spo2_m.group(0), "")
        except (ValueError, TypeError):
            pass

    # ── Urine Volume ──────────────────────────────────────────────────────
    urine_vol_patterns = [
        (r'[Uu]rine\s*(?:[Oo]utput|[Vv]olume|[Cc]ollected?)\s*[:/]?\s*([\d,\.]+)\s*(?:mL|ml|ML)', 1.0),
        (r'[Uu]rine\s*(?:[Oo]utput|[Vv]ol)\s*[:/]?\s*([\d,\.]+)', 0.80),
        (r'[Uu][Oo]\s*[:/]?\s*([\d,\.]+)\s*(?:mL|ml|ML)', 0.85),
    ]
    v, raw, conf, note = _regex_extract(text_n, urine_vol_patterns, 'urine_vol')
    if v:
        results['urine_vol'] = (round(v, 1), conf, raw, note)

    urine_time_patterns = [
        (r'(?:[Oo]ver|[Cc]ollection\s*[Pp]eriod|[Cc]ollected\s*[Oo]ver)\s*([\d,\.]+)\s*(?:hours?|hrs?|h)\b', 1.0),
        (r'([\d,\.]+)\s*[Hh](?:our|r)?\s*[Cc]ollection', 0.85),
    ]
    v, raw, conf, note = _regex_extract(text_n, urine_time_patterns, 'urine_time')
    if v:
        results['urine_time'] = (round(v, 1), conf, raw, note)

    # ── Date ────────────────────────────────────────────────────────────────
    date_patterns = [
        r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})',
        r'(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})',
    ]
    for dp in date_patterns:
        dm = re.findall(dp, text_n)
        if dm:
            try:
                g = dm[0]
                if len(g[0]) == 4:
                    d = datetime.date(int(g[0]), int(g[1]), int(g[2]))
                else:
                    yr = int(g[2]) if len(g[2]) == 4 else 2000 + int(g[2])
                    d = datetime.date(yr, int(g[1]), int(g[0]))
                if d <= datetime.date.today():
                    results['date'] = (d, 0.85, str(d), "")
                    break
            except ValueError:
                continue

    # ── Clinical Notes ────────────────────────────────────────────────────
    notes_patterns = [
        r'(?:Impression|Clinical\s*Notes?|Remarks?|Comments?|Conclusion|Final\s*Report)[:\s]+([^\n]{10,200})',
        r'(?:Doctor[\'s]*\s*Notes?|Physician\s*Notes?)[:\s]+([^\n]{10,200})',
    ]
    for pat in notes_patterns:
        nm = re.search(pat, text_n, re.IGNORECASE)
        if nm:
            note_text = nm.group(1).strip()
            if len(note_text) >= 10:
                results['clinical_notes'] = (note_text, 0.80, note_text[:60] + "...", "")
                break

    return results

# ══════════════════════════════════════════════════════════
#  9.  IMAGE/PDF OCR
# ══════════════════════════════════════════════════════════
def preprocess_image_for_ocr(pil_image):
    img_arr = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    coords = np.column_stack(np.where(gray < 200))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        if abs(angle) > 0.5:
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
    scaled = cv2.resize(thresh, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(scaled)


def extract_text_from_image(pil_image):
    processed = preprocess_image_for_ocr(pil_image)
    try:
        return pytesseract.image_to_string(processed, config=r'--oem 3 --psm 6')
    except Exception as tess_err:
        err_msg = str(tess_err).lower()
        if 'tesseract' in err_msg or 'path' in err_msg or 'not found' in err_msg:
            return ''
        raise


def ocr_from_pdf_bytes(pdf_bytes):
    if PYPDF_AVAILABLE:
        try:
            reader     = PdfReader(io.BytesIO(pdf_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                if i >= 5: break
                pages_text.append(page.extract_text() or "")
            combined = "\n".join(pages_text).strip()
            if len(combined) > 80:
                return combined
        except Exception:
            pass
    if not PDF2IMAGE_AVAILABLE:
        return ""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=3)
        return "\n".join(extract_text_from_image(p) for p in pages)
    except Exception:
        return ""


def confidence_color(conf):
    if conf >= 0.85:   return "conf-high", "High"
    elif conf >= 0.65: return "conf-med",  "Medium"
    else:              return "conf-low",   "Low"

# ══════════════════════════════════════════════════════════
#  10.  APPLY OCR TO SESSION STATE
# ══════════════════════════════════════════════════════════
def apply_ocr_to_session(rep_key, extracted):
    data           = st.session_state.reports[rep_key].copy()
    enable_urine   = False
    enable_metrics = False
    demo_updates   = {}

    field_map = {
        'platelets': (int,   'platelets'),
        'hb':        (float, 'hb'),
        'hct':       (float, 'hct'),
        'rbc':       (float, 'rbc'),
        'sys':       (int,   'sys'),
        'dia':       (int,   'dia'),
        'hr':        (int,   'hr'),
    }
    for field, (cast, dest) in field_map.items():
        if field in extracted:
            data[dest] = cast(extracted[field][0])

    if 'date' in extracted:
        data['date'] = extracted['date'][0]

    if 'time_hour' in extracted:
        hr24 = int(extracted['time_hour'][0])
        mn   = int(extracted.get('time_minute', (0,))[0])
        hr12 = hr24 % 12 or 12
        ampm_str = "AM" if hr24 < 12 else "PM"
        data['time_hour']   = hr12
        data['time_minute'] = mn
        data['time_ampm']   = ampm_str

    if 'age'          in extracted: demo_updates['age']           = int(extracted['age'][0])
    if 'sex'          in extracted: demo_updates['sex']           = extracted['sex'][0]
    if 'patient_name' in extracted: demo_updates['name']          = str(extracted['patient_name'][0])
    if 'clinical_notes' in extracted: demo_updates['clinical_notes'] = str(extracted['clinical_notes'][0])

    if 'weight' in extracted:
        w                  = float(extracted['weight'][0])
        data['ocr_weight'] = w
        st.session_state.metrics_from_ocr['weight']  = w
        st.session_state.metrics_from_ocr['enabled'] = True
        st.session_state['weight_val'] = w   # ← direct write, not pop
        enable_metrics = True
    if 'height' in extracted:
        h                  = float(extracted['height'][0])
        data['ocr_height'] = h
        st.session_state.metrics_from_ocr['height']  = h
        st.session_state.metrics_from_ocr['enabled'] = True
        st.session_state['height_cm'] = h    # ← direct write, not pop
        enable_metrics = True

    if 'urine_vol' in extracted:
        data['urine_vol'] = float(extracted['urine_vol'][0])
        enable_urine = True
    if 'urine_time' in extracted:
        data['urine_time'] = float(extracted['urine_time'][0])
        enable_urine = True
    if enable_urine and data.get('ocr_weight', 0) > 0:
        data['urine_weight'] = data['ocr_weight']
        
    # ── Propagate detected units to global preferences ─────────────────────
    # ── Sync age/sex backing fields without causing widget key conflict ───
    # Only update the backing session state vars, NOT the widget keys directly.
    # The widgets read from these on next render.
    if 'age' in demo_updates:
        st.session_state['patient_age'] = int(demo_updates['age'])
        # Force widget key to match if it hasn't been rendered yet
        if 'patient_age_widget' in st.session_state:
            st.session_state['patient_age_widget'] = int(demo_updates['age'])
    if 'sex' in demo_updates:
        st.session_state['patient_sex'] = demo_updates['sex']
        if 'patient_sex_widget' in st.session_state:
            st.session_state['patient_sex_widget'] = demo_updates['sex']

    st.session_state.reports[rep_key] = data
    st.session_state[f"form_ver_{rep_key}"] = st.session_state.get(f"form_ver_{rep_key}", 0) + 1

    if '_detected_height_unit' in extracted:
        du = extracted['_detected_height_unit'][0]
        st.session_state['unit_height']       = du
        st.session_state['unit_height_radio'] = du
    if '_detected_temp_unit' in extracted:
        du = extracted['_detected_temp_unit'][0]
        st.session_state['unit_temp']         = du
        st.session_state['unit_temp_radio']   = du
    if 'temperature' in extracted:
        data['temperature'] = float(extracted['temperature'][0])
    if 'spo2' in extracted:
        data['spo2'] = int(extracted['spo2'][0])

    st.session_state.reports[rep_key] = data
    st.session_state[f"form_ver_{rep_key}"] = st.session_state.get(f"form_ver_{rep_key}", 0) + 1

    return enable_urine, enable_metrics, demo_updates

# ══════════════════════════════════════════════════════════
#  11.  SESSION STATE & CLEAR FUNCTIONS
# ══════════════════════════════════════════════════════════
REPORT_KEYS = ['A', 'B', 'C', 'D', 'E']

def _blank_report():
    return {
        # ── Existing CBC ──────────────────────────────
        'date': datetime.date.today(), 'time_hour': 9, 'time_minute': 0, 'time_ampm': "AM",
        'platelets': 0, 'hb': 13.0, 'rbc': 4.5, 'hct': 40.0,
        'sys': 120, 'dia': 80, 'hr': 72,
        'urine_output': 0.0, 'urine_vol': 0.0, 'urine_time': 1.0, 'urine_weight': 0.0,
        'who': [], 'symptoms': [],
        'ocr_weight': 0.0, 'ocr_height': 0.0, 'ocr_bmi': 0.0,
        # ── CBC Differential (NEW) ────────────────────
        'wbc': 0, 'neutrophil_pct': 0.0, 'lymphocyte_pct': 0.0, 'mpv': 0.0,
        # ── Vital Signs Expansion (NEW) ───────────────
        'temperature': 0.0,     # °C
        'spo2': 0,              # %
        'rr': 0,                # breaths/min
        'gcs': 15,              # 3-15
        'crt': 0,               # 0=<2s, 1=2-3s, 2=>3s
        # ── LFT Panel (NEW) ──────────────────────────
        'ast': 0, 'alt': 0, 'albumin': 0.0, 'bilirubin_total': 0.0, 'bilirubin_direct': 0.0,
        # ── Coagulation (NEW) ─────────────────────────
        'pt': 0.0, 'inr': 0.0, 'aptt': 0.0, 'd_dimer': 0,
        # ── Renal / Electrolytes (NEW) ────────────────
        'creatinine': 0.0, 'bun': 0.0, 'sodium': 0.0, 'potassium': 0.0, 'bicarbonate': 0.0,
        # ── Serology (NEW) ────────────────────────────
        'ns1': 'Not Done',      # Positive / Negative / Not Done
        'igm': 'Not Done',      # Reactive / Non-Reactive / Not Done
        'igg': 'Not Done',      # Reactive / Non-Reactive / Not Done
        # ── Imaging (NEW) ────────────────────────────
        'pleural_effusion': False,
        'gallbladder_wall_mm': 0.0,
        'ascites_grade': 0,     # 0-3
    }

def _clear_report_widgets(rep_key):
    keys_to_clear = [
        f"plt_{rep_key}", f"hb_{rep_key}", f"rbc_{rep_key}", f"hct_{rep_key}",
        f"sys_{rep_key}", f"dia_{rep_key}", f"hr_{rep_key}",
        f"u_vol_{rep_key}", f"u_time_{rep_key}", f"u_wt_{rep_key}",
        f"hour_{rep_key}", f"min_{rep_key}", f"ampm_{rep_key}", f"date_{rep_key}",
    ]
    for wk in keys_to_clear:
        st.session_state.pop(wk, None)

def clear_current_report(report_key):
    st.session_state.reports[report_key] = _blank_report()
    _clear_report_widgets(report_key)
    _init_report_widget_defaults(report_key)
    st.session_state[f"form_ver_{report_key}"] = 0
    st.session_state.analysis_run = False

    # Permanently remove OCR data for this specific report
    st.session_state.ocr_pending.pop(report_key, None)
    st.session_state.bulk_ocr_accepted.discard(report_key)
    # Reset metrics_from_ocr only if no other report has OCR data
    remaining = [k for k in st.session_state.bulk_ocr_accepted]
    if not remaining:
        st.session_state.metrics_from_ocr = {}
        st.session_state['weight_val'] = 0.0
        st.session_state['height_cm']  = 0.0

def clear_all_reports():
    for char in REPORT_KEYS:
        st.session_state.reports[char] = _blank_report()
        _clear_report_widgets(char)
        _init_report_widget_defaults(char)
        st.session_state[f"form_ver_{char}"] = 0
    st.session_state['weight_val'] = 0.0
    st.session_state['height_cm']  = 0.0
    st.session_state['ft_val']     = 0
    st.session_state['in_val']     = 0
    st.session_state.analysis_run              = False
    st.session_state.clinician_notes           = ""
    st.session_state.discharge_fever_free      = 0.0
    st.session_state.discharge_tolerating_orals = False
    st.session_state.discharge_enabled         = False
    st.session_state.ocr_pending               = {}
    st.session_state.bulk_ocr_accepted         = set()
    st.session_state.patient_age               = 25
    st.session_state.patient_sex               = "Male"
    st.session_state.patient_name              = ""
    st.session_state['patient_age_widget']     = 25
    st.session_state['patient_sex_widget']     = "Male"
    st.session_state['patient_name_widget']    = ""
    st.session_state.metrics_from_ocr          = {}

def _report_has_meaningful_data(r):
    """
    True if report has ANY clinically meaningful data beyond factory defaults.
    Includes OCR-extracted demographic/vitals data so OCR-accepted reports
    always appear in the preview table even without CBC values.
    """
    return (
        # ── Primary clinical markers ──────────────────────────────────
        r.get('platelets', 0) > 0 or
        r.get('wbc', 0) > 0 or
        r.get('ast', 0) > 0 or
        r.get('alt', 0) > 0 or
        r.get('creatinine', 0) > 0 or
        r.get('inr', 0) > 0 or
        r.get('spo2', 0) > 0 or
        r.get('d_dimer', 0) > 0 or
        (r.get('temperature', 0.0) >= 30.0) or
        r.get('pleural_effusion', False) or
        r.get('ascites_grade', 0) > 0 or
        # ── Serology (non-default values) ─────────────────────────────
        r.get('ns1', 'Not Done') not in ('Not Done', '') or
        r.get('igm', 'Not Done') not in ('Not Done', '') or
        r.get('igg', 'Not Done') not in ('Not Done', '') or
        # ── Clinical signs ────────────────────────────────────────────
        len(r.get('who', [])) > 0 or
        len(r.get('symptoms', [])) > 0 or
        # ── OCR-populated fields — ensures OCR-accepted reports always ──
        # appear in preview even when CBC wasn't successfully extracted
        r.get('ocr_weight', 0.0) > 0 or
        r.get('urine_vol', 0.0) > 0 or
        r.get('hb', 13.0) != 13.0 or        # non-default Hb
        r.get('sys', 120) != 120 or          # non-default SBP
        r.get('hr', 72) != 72               # non-default HR
    )
    
def _init_report_widget_defaults(rep_key):
    r = st.session_state.reports[rep_key]
    defaults = {
        f"plt_{rep_key}":   r['platelets'],
        f"hb_{rep_key}":    r['hb'],
        f"rbc_{rep_key}":   r['rbc'],
        f"hct_{rep_key}":   r['hct'],
        f"sys_{rep_key}":   r['sys'],
        f"dia_{rep_key}":   r['dia'],
        f"hr_{rep_key}":    r['hr'],
        f"u_vol_{rep_key}": r.get('urine_vol', 0.0),
        f"u_time_{rep_key}":r.get('urine_time', 1.0),
        f"u_wt_{rep_key}":  r.get('urine_weight', 0.0),
        f"hour_{rep_key}":  r['time_hour'],
        f"min_{rep_key}":   r['time_minute'],
        f"ampm_{rep_key}":  r['time_ampm'],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ══════════════════════════════════════════════════════════
#  12.  SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════
if 'reports' not in st.session_state:
    st.session_state.reports = {c: _blank_report() for c in REPORT_KEYS}

for _c in REPORT_KEYS:
    _r = st.session_state.reports[_c]
    for _k, _v in [('urine_vol', 0.0), ('urine_time', 1.0), ('urine_weight', 0.0),
                   ('symptoms', []), ('ocr_weight', 0.0), ('ocr_height', 0.0), ('ocr_bmi', 0.0)]:
        if _k not in _r:
            _r[_k] = _v
    _init_report_widget_defaults(_c)

_SS_DEFAULTS = {
    'analysis_run':               False,
    'active_page':                "Home",
    'active_report':              "A",
    'clinician_notes':            "",
    'discharge_fever_free':       0.0,
    'discharge_tolerating_orals': False,
    'discharge_enabled':          False,
    'ocr_pending':                {},
    'bulk_ocr_accepted':          set(),
    'patient_onset_date':         None,
    'patient_illness_day':        0,
    'patient_dengue_phase':       'Unknown',
    'is_secondary_dengue':        False,
    'ns1_result':                 'Not Done',
    'patient_age':                25,
    'patient_sex':                "Male",
    'patient_name':               "",
    'metrics_from_ocr':           {},
    'patient_age_widget':         25,
    'patient_sex_widget':         "Male",
    'patient_name_widget':        "",
    'weight_val':                 0.0,
    'height_cm':                  0.0,
    'ft_val':                     0,
    'in_val':                     0,
    'sidebar_collapsed':          False,
    'form_ver_A': 0, 'form_ver_B': 0, 'form_ver_C': 0,
    'form_ver_D': 0, 'form_ver_E': 0,
    'unit_weight': 'kg',
    'unit_height': 'cm',
    'unit_temp':   '°C',
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.get('sidebar_collapsed', False):
    st.markdown("""
    <style>
    section[data-testid="stSidebar"]                { display: none !important; }
    section[data-testid="stSidebar"] + div          { margin-left: 0 !important; }
    .block-container                                { max-width: 100% !important; padding-left: 2rem !important; padding-right: 2rem !important; }
    </style>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  13.  CHART HELPERS
# ══════════════════════════════════════════════════════════
def _dark_ax(ax, fig):
    """White background for PDF clarity; all text forced to dark/readable colours."""
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    ax.tick_params(colors='#2c3e50', labelsize=8.5)
    ax.xaxis.label.set_color('#2c3e50')
    ax.yaxis.label.set_color('#2c3e50')
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#7f8c8d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, color='#bdc3c7')

FEATURE_DISPLAY = {
    'Sex_Code': 'Sex', 'Season_Risk': 'Season Risk',
    'Has_Fever': 'Fever', 'Has_Headache': 'Headache',
    'Has_Pain': 'Joint Pain', 'Has_Vomit': 'Vomiting', 'Has_Bleeding': 'Bleeding',
    'Platelet (cells/cu.mm)': 'Platelet Count',
    'Haemoglobin (gm/Dl)': 'Haemoglobin',
    'Red Blood Cell Count (millions/cu.mm)': 'RBC Count',
    'Hematocrit (Packed Cell Volume) (%)': 'Hematocrit',
}
def _display_names(names): return [FEATURE_DISPLAY.get(n, n) for n in names]

plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300, 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'font.family': 'DejaVu Sans', 'axes.linewidth': 1.2, 'patch.linewidth': 1.2,
})

# ══════════════════════════════════════════════════════════
#  14.  RISK CHART
# ══════════════════════════════════════════════════════════
def build_risk_chart(classifier, df_input, urine_rate=0.0,
                     latest_report=None, show_ml_panel: bool = True):
    """
    Two-panel risk chart.
      Left  (optional): SHAP / feature importances — requires valid ML input (show_ml_panel=True)
      Right (always):   Clinical risk factor summary for all entered parameters

    Parameters
    ----------
    show_ml_panel : bool
        False when called without a valid platelet count — skips SHAP computation entirely
        and renders only the clinical factors panel with an explanatory title.

    Returns
    -------
    (screen_buf, pdf_buf) — BytesIO PNG streams
    """
    import io as _io

    # ── Panel 1: ML feature importances / SHAP ────────────────────────────
    pairs  = []
    vals   = []
    labels = []

    if show_ml_panel:
        try:
            if SHAP_AVAILABLE:
                explainer = shap.TreeExplainer(classifier)
                shap_vals = explainer.shap_values(df_input)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0] if shap_vals.ndim == 1 else shap_vals[0]
                feat_names = df_input.columns.tolist()
                pairs = sorted(zip(sv, feat_names), key=lambda x: abs(x[0]), reverse=True)
            else:
                importances = classifier.feature_importances_
                feat_names  = df_input.columns.tolist()
                pairs = sorted(zip(importances, feat_names), key=lambda x: abs(x[0]), reverse=True)
        except Exception:
            importances = classifier.feature_importances_
            feat_names  = df_input.columns.tolist()
            pairs = sorted(zip(importances, feat_names), key=lambda x: abs(x[0]), reverse=True)

        top_n  = min(10, len(pairs))
        vals   = [p[0] for p in pairs[:top_n]]
        labels = [p[1].replace('_', ' ').title() for p in pairs[:top_n]]

    top_n   = min(10, len(pairs))
    vals    = [p[0] for p in pairs[:top_n]]
    labels  = [p[1].replace('_', ' ').title() for p in pairs[:top_n]]
    colors  = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals]

    # ── Panel 2: All clinical risk factors ──────────────────────────────
    clinical_factors = []
    if latest_report:
        r = latest_report
        plt_v = r.get('platelets', 0)
        si_v  = r.get('shock_index', 0)
        map_v = r.get('map', 0)
        pp_v  = r.get('sys_bp', 120) - r.get('dia_bp', 80)
        hct_v = r.get('hct', 0)
        inr_v = r.get('inr', 0)
        ast_v = r.get('ast', 0)
        spo2_v= r.get('spo2', 0)
        gcs_v = r.get('gcs', 15)
        cr_v  = r.get('creatinine', 0)
        uo_v  = r.get('urine_output', 0)
        ddim_v= r.get('d_dimer', 0)
        alb_v = r.get('albumin', 0)
        wbc_v = r.get('wbc', 0)
        temp_v= r.get('temperature', 0)
        # Each tuple: (label, value_str, risk_weight 0-1, risk_direction: 1=high=bad, -1=high=good)
        if plt_v > 0:
            w = 1.0 if plt_v < 20000 else (0.7 if plt_v < 50000 else (0.4 if plt_v < 100000 else 0.1))
            clinical_factors.append(("Platelets", f"{plt_v:,}/uL", w, -1))
        if si_v > 0:
            w = 1.0 if si_v >= 1.2 else (0.7 if si_v >= 1.0 else (0.4 if si_v >= 0.9 else 0.1))
            clinical_factors.append(("Shock Index", f"{si_v:.2f}", w, 1))
        if map_v > 0:
            w = 0.9 if map_v < 60 else (0.5 if map_v < 65 else 0.0)
            clinical_factors.append(("MAP", f"{map_v:.1f} mmHg", w, -1))
        if pp_v > 0:
            w = 0.9 if pp_v <= 20 else (0.4 if pp_v <= 30 else 0.0)
            clinical_factors.append(("Pulse Pressure", f"{pp_v} mmHg", w, -1))
        if hct_v > 0:
            hct_rise = max(0, hct_v - 40)
            w = 0.8 if hct_rise >= 10 else (0.4 if hct_rise >= 5 else 0.0)
            clinical_factors.append(("Hct (vs normal)", f"{hct_v:.1f}%", w, 1))
        if ast_v > 0:
            w = 1.0 if ast_v >= 1000 else (0.7 if ast_v >= 500 else (0.4 if ast_v >= 80 else 0.1))
            clinical_factors.append(("AST", f"{ast_v:,} IU/L", w, 1))
        if inr_v > 0:
            w = 0.9 if inr_v >= 2.0 else (0.6 if inr_v >= 1.5 else 0.1)
            clinical_factors.append(("INR", f"{inr_v:.2f}", w, 1))
        if spo2_v > 0:
            w = 1.0 if spo2_v < 90 else (0.7 if spo2_v < 93 else (0.3 if spo2_v < 95 else 0.0))
            clinical_factors.append(("SpO2", f"{spo2_v}%", w, -1))
        if gcs_v < 15:
            w = 1.0 if gcs_v < 10 else (0.7 if gcs_v < 13 else 0.3)
            clinical_factors.append(("GCS", f"{gcs_v}", w, -1))
        if cr_v > 0:
            w = 0.8 if cr_v > 2.0 else (0.5 if cr_v > 1.2 else 0.1)
            clinical_factors.append(("Creatinine", f"{cr_v:.2f} mg/dL", w, 1))
        if uo_v > 0:
            w = 0.9 if uo_v < 0.5 else (0.5 if uo_v < 1.0 else 0.0)
            clinical_factors.append(("Urine Output", f"{uo_v:.2f} mL/kg/hr", w, -1))
        if ddim_v > 0:
            w = 0.8 if ddim_v >= 2000 else (0.4 if ddim_v >= 1000 else 0.1)
            clinical_factors.append(("D-dimer", f"{ddim_v:,} ng/mL", w, 1))
        if alb_v > 0:
            w = 0.7 if alb_v < 3.0 else (0.3 if alb_v < 3.5 else 0.0)
            clinical_factors.append(("Albumin", f"{alb_v:.1f} g/dL", w, -1))
        if wbc_v > 0:
            w = 0.5 if wbc_v < 4000 else 0.1
            clinical_factors.append(("WBC", f"{wbc_v:,}/uL", w, -1))
        if temp_v >= 37.5:
            w = 0.6 if temp_v >= 39 else (0.3 if temp_v >= 38 else 0.1)
            clinical_factors.append(("Temperature", f"{temp_v:.1f}°C", w, 1))
        # WHO warning signs
        who_signs = r.get('who_signs', [])
        if who_signs:
            w = min(1.0, len(who_signs) * 0.35)
            clinical_factors.append(("WHO Warning Signs", f"{len(who_signs)} signs", w, 1))
        # Serology flags
        if r.get('ns1', 'Not Done') == 'Positive':
            clinical_factors.append(("NS1 Antigen", "Positive", 0.6, 1))
        if r.get('igg', 'Not Done') == 'Reactive':
            clinical_factors.append(("Secondary Dengue (IgG)", "Reactive", 0.9, 1))
        # Imaging
        if r.get('pleural_effusion', False):
            clinical_factors.append(("Pleural Effusion", "Present", 1.0, 1))
        if r.get('ascites_grade', 0) >= 2:
            clinical_factors.append(("Ascites", f"Grade {r.get('ascites_grade')}", 0.8, 1))
        if r.get('gallbladder_wall_mm', 0) >= 5:
            clinical_factors.append(("GB Wall", f"{r.get('gallbladder_wall_mm'):.1f} mm", 0.7, 1))

        # Sort by risk weight descending
        clinical_factors.sort(key=lambda x: x[2], reverse=True)

    # ── Build figure ──────────────────────────────────────────────────────
    # ── Build figure ──────────────────────────────────────────────────────
    _has_ml = show_ml_panel and bool(vals)
    _has_cf = bool(clinical_factors)

    if not _has_ml and not _has_cf:
        # Nothing to plot — return a minimal placeholder
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No clinical data available to chart",
                ha='center', va='center', color='#8b92a8', fontsize=10)
        ax.axis('off')
        _dark_ax(ax, fig)
    else:
        n_panels = int(_has_ml) + int(_has_cf)
        fig, axes = plt.subplots(1, n_panels, figsize=(14 if n_panels == 2 else 8, 5))
        if n_panels == 1:
            axes = [axes]

        ax_idx = 0

        if _has_ml:
            ax = axes[ax_idx]; ax_idx += 1
            _bar_colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals]
            ax.barh(labels[::-1], [abs(v) for v in vals[::-1]],
                    color=_bar_colors[::-1], alpha=0.85, height=0.65)
            ax.set_xlabel("SHAP Value / Feature Importance", color='#2c3e50', fontsize=9)
            ax.set_title("AI Risk Factors\n(ML Model)", color='#2c3e50',
                         fontsize=10, fontweight='700')
            ax.tick_params(colors='#2c3e50', labelsize=8.5)
            _dark_ax(ax, fig)

        if _has_cf:
            ax2 = axes[ax_idx]
            cf_labels = [f"{f[0]}\n{f[1]}" for f in clinical_factors[:12]][::-1]
            cf_vals   = [f[2] for f in clinical_factors[:12]][::-1]
            cf_colors = ['#e74c3c' if v >= 0.7 else ('#f39c12' if v >= 0.3 else '#2ecc71')
                         for v in cf_vals]
            ax2.barh(cf_labels, cf_vals, color=cf_colors, alpha=0.85, height=0.65)
            ax2.set_xlim(0, 1.1)
            ax2.set_xlabel("Risk Weight (0=Low, 1=Critical)", color='#2c3e50', fontsize=9)

            # Title adapts based on whether the ML panel is present
            _cf_title = ("All Clinical Risk Factors\n(This Patient)" if _has_ml
                         else "Clinical Risk Factor Assessment\n(Platelet count not recorded — ML score unavailable)")
            ax2.set_title(_cf_title, color='#2c3e50', fontsize=10, fontweight='700')
            ax2.tick_params(colors='#2c3e50', labelsize=8.0)
            for i, v in enumerate(cf_vals):
                ax2.text(v + 0.02, i, f"{v:.0%}", va='center', color='#2c3e50', fontsize=7.5)
            _dark_ax(ax2, fig)

    fig.tight_layout(pad=1.5)

    screen_buf = _io.BytesIO()
    fig.savefig(screen_buf, format='png', bbox_inches='tight', dpi=130)
    screen_buf.seek(0)

    pdf_buf = _io.BytesIO()
    fig.savefig(pdf_buf, format='png', bbox_inches='tight', dpi=200)
    pdf_buf.seek(0)

    plt.close(fig)
    return screen_buf, pdf_buf

# ══════════════════════════════════════════════════════════
#  15.  PDF GENERATOR
# ══════════════════════════════════════════════════════════
if PDF_AVAILABLE:
    def _s(text):
        replacements = {
            '\u2014': '-', '\u2013': '-', '\u2019': "'", '\u2018': "'",
            '\u201c': '"', '\u201d': '"', '\u2022': '*', '\u2026': '...',
            '\u2265': '>=', '\u2264': '<=', '\u00d7': 'x', '\u00b0': ' deg',
            '\u2192': '->', '\u2190': '<-', '\u00b1': '+/-',
        }
        s = str(text)
        for uni, asc in replacements.items():
            s = s.replace(uni, asc)
        return s.encode('latin-1', errors='replace').decode('latin-1')

    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, _s('Dengue CDSS - Clinical Trajectory Report'), 0, 1, 'C')
            self.set_font('Arial', 'I', 9)
            self.cell(0, 5, _s(
                f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}  |  '
                f'WHO 2009 Classification Engine  |  For Clinical Decision Support Only'), 0, 1, 'C')
            self.ln(3)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, _s(
                f'Page {self.page_no()} | Dengue CDSS | '
                f'Sensitivity 99.77% | Specificity 100% | '
                f'AUC=0.9996 | Forecast R\u00b2=0.9953 | MAE=2,515 | n=2,455'), 0, 0, 'C')

    def create_pdf(patient_data, all_valid_reports, clinical_data, plot_stream, risk_stream,
                   notes, who_data=None, alerts=None, discharge=None, severity_data=None):
        pdf = PDFReport(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Patient Demographics", 0, 1)
        pdf.set_font("Arial", size=11)
        name_str = f" | Patient: {patient_data['name']}" if patient_data.get('name') else ""
        base = _s(f"Age: {patient_data['age']} years | Sex: {patient_data['sex']}{name_str}")
        if patient_data['weight'] > 0 and patient_data['bmi'] > 0:
            pdf.cell(0, 8, _s(base + f" | Weight: {patient_data['weight']:.1f} kg | BMI: {patient_data['bmi']:.1f}"), 0, 1)
        else:
            pdf.cell(0, 8, _s(base + " | Body Metrics: Not Recorded"), 0, 1)
        pdf.ln(3)

        if severity_data:
            score, direction, components = severity_data
            pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Continuous Severity Score", 0, 1)
            pdf.set_font("Arial", 'B', 11)
            sev_color = (231,76,60) if score >= 70 else ((243,156,18) if score >= 40 else (46,204,113))
            pdf.set_text_color(*sev_color)
            pdf.cell(0, 8, _s(f"Severity: {score}/100 — {direction}"), 0, 1)
            pdf.set_text_color(0,0,0); pdf.set_font("Arial", size=9)
            for comp, val in components.items():
                pdf.cell(0, 5, _s(f"  {comp}: {val} pts"), 0, 1)
            pdf.ln(2)

        if who_data:
            pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "WHO 2009 Dengue Classification", 0, 1)
            pdf.set_font("Arial", 'B', 11)
            grp_color = {'A': (46, 204, 113), 'B': (243, 156, 18), 'C': (231, 76, 60)}.get(who_data[1], (200, 200, 200))
            pdf.set_text_color(*grp_color)
            pdf.cell(0, 8, _s(f"{who_data[0]} - Group {who_data[1]}: {who_data[3]}"), 0, 1)
            pdf.set_text_color(0, 0, 0)
            if who_data[2]:
                pdf.set_font("Arial", size=9)
                for c in who_data[2]: pdf.cell(0, 6, _s(f"  * {c}"), 0, 1)
            pdf.ln(2)
        if alerts:
            pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "Clinical Alerts", 0, 1)
            pdf.set_font("Arial", size=9)
            for sev, msg, rec in alerts:
                pdf.set_text_color(231, 76, 60) if sev == 'CRITICAL' else pdf.set_text_color(243, 156, 18)
                pdf.cell(0, 6, _s(f"[{sev}] {msg}"), 0, 1)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 5, _s(f"  -> {rec}"), 0, 1)
                pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        # ══════════════════════════════════════════════════════
        # Refined PDF Gen. 2098–2129 BLOCK
        # ══════════════════════════════════════════════════════
        n  = len(all_valid_reports)
        cw = min(46, 230 / n) if n > 0 else 30

        def _pdf_section_header(title):
            pdf.ln(3)
            pdf.set_font("Arial", 'B', 12)
            pdf.set_fill_color(30, 40, 60)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 9, _s(title), 0, 1, 'L', 1)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", 'B', 9)
            pdf.set_fill_color(220, 220, 220)

        def _pdf_col_headers():
            pdf.cell(46, 8, "Parameter", 1, 0, 'C', 1)
            for i, r in enumerate(all_valid_reports):
                pdf.cell(cw, 8, _s(r.get('Label', f"Report {chr(65+i)}")), 1, 0, 'C', 1)
            pdf.ln()
            pdf.set_font("Arial", size=9)

        def add_row(lbl, key, fmt="{}"):
            pdf.cell(46, 7, _s(lbl), 1)
            for r in all_valid_reports:
                v = r.get(key, "-")
                if   key == 'datetime':  v = v.strftime("%m-%d %I:%M %p")
                elif key == 'map':       v = f"{calculate_map(r['sys_bp'], r['dia_bp']):.1f}"
                elif key == 'bp':        v = f"{r['sys_bp']}/{r['dia_bp']}"
                elif key == 'pp':        v = f"{r['sys_bp'] - r['dia_bp']}"
                elif key == 'urine':     v = f"{r.get('urine_output',0):.2f}" if r.get('urine_output',0) > 0 else "-"
                elif key == 'platelets': v = f"{int(v):,}" if v != "-" and v != 0 else "-"
                elif key == 'temp':
                    tv = r.get('temperature', 0.0)
                    v  = f"{tv:.1f}C" if tv >= 30.0 else "-"
                elif key == 'crt':
                    v = ['N/A', '<2s', '2-3s', '>3s'][min(int(r.get('crt', 0)), 3)]
                elif key == 'who_signs': v = str(len(r.get('who_signs', []))) or "0"
                elif key == 'symptoms':  v = ", ".join(r.get('symptoms', [])) or "None"
                elif key == 'pe':        v = "Yes" if r.get('pleural_effusion', False) else "No"
                elif key == 'ascites':   v = f"Gr{r.get('ascites_grade',0)}" if r.get('ascites_grade',0) > 0 else "-"
                elif key == 'gb_wall':   v = f"{r.get('gallbladder_wall_mm',0):.1f}" if r.get('gallbladder_wall_mm',0) > 0 else "-"
                else:
                    try:
                        raw = r.get(key, 0)
                        v = fmt.format(raw) if raw and raw != 0 and raw != 0.0 else "-"
                    except Exception:
                        v = "-"
                # colour-code critical values
                if key == 'platelets':
                    try:
                        pv = r.get('platelets', 999999)
                        if   pv < 20000:  pdf.set_text_color(231, 76,  60)
                        elif pv < 50000:  pdf.set_text_color(231, 76,  60)
                        elif pv < 100000: pdf.set_text_color(243, 156, 18)
                        else:             pdf.set_text_color(46,  204, 113)
                    except Exception:
                        pdf.set_text_color(0, 0, 0)
                elif key == 'shock_index':
                    try:
                        si = float(r.get('shock_index', 0))
                        if   si >= 1.2: pdf.set_text_color(231, 76,  60)
                        elif si >= 0.9: pdf.set_text_color(243, 156, 18)
                        else:           pdf.set_text_color(46,  204, 113)
                    except Exception:
                        pdf.set_text_color(0, 0, 0)
                elif key == 'inr':
                    try:
                        iv = float(r.get('inr', 0))
                        if   iv >= 2.0: pdf.set_text_color(231, 76,  60)
                        elif iv >= 1.5: pdf.set_text_color(243, 156, 18)
                        else:           pdf.set_text_color(0, 0, 0)
                    except Exception:
                        pdf.set_text_color(0, 0, 0)
                elif key == 'spo2':
                    try:
                        sv = int(r.get('spo2', 100))
                        if   sv > 0 and sv < 93: pdf.set_text_color(231, 76,  60)
                        elif sv > 0 and sv < 95: pdf.set_text_color(243, 156, 18)
                        else:                    pdf.set_text_color(0, 0, 0)
                    except Exception:
                        pdf.set_text_color(0, 0, 0)
                else:
                    pdf.set_text_color(0, 0, 0)

                pdf.cell(cw, 7, _s(str(v)), 1, 0, 'C')
                pdf.set_text_color(0, 0, 0)
            pdf.ln()

        # ── Helper: only print a section if at least one report has that data ──
        def _any(key, threshold=0):
            return any(
                (r.get(key, 0) or 0) > threshold
                for r in all_valid_reports
            )

        # ─────────────────────────────────────────────
        # SECTION 1: Core CBC
        # ─────────────────────────────────────────────
        _pdf_section_header("Longitudinal Vitals Matrix — Core CBC & Haemodynamics")
        _pdf_col_headers()
        add_row("Date/Time",         "datetime")
        add_row("Platelets (c/uL)",  "platelets")
        add_row("Hematocrit (%)",    "hct",   "{:.1f}")
        add_row("Hemoglobin (g/dL)", "hb",    "{:.1f}")
        add_row("RBC (M/uL)",        "rbc",   "{:.2f}")
        add_row("BP (mmHg)",         "bp")
        add_row("Heart Rate (bpm)",  "hr")
        add_row("MAP (mmHg)",        "map")
        add_row("Shock Index",       "shock_index", "{:.2f}")
        add_row("Pulse Pressure",    "pp")
        add_row("Urine (mL/kg/hr)",  "urine")

        # ─────────────────────────────────────────────
        # SECTION 2: CBC Differential (if present)
        # ─────────────────────────────────────────────
        if _any('wbc'):
            _pdf_section_header("CBC Differential")
            _pdf_col_headers()
            add_row("WBC (cells/uL)",    "wbc",           "{:,}")
            if _any('neutrophil_pct'):
                add_row("Neutrophil %",  "neutrophil_pct", "{:.1f}")
            if _any('lymphocyte_pct'):
                add_row("Lymphocyte %",  "lymphocyte_pct", "{:.1f}")
            if _any('mpv'):
                add_row("MPV (fL)",      "mpv",            "{:.1f}")

        # ─────────────────────────────────────────────
        # SECTION 3: Extended Vitals (if present)
        # ─────────────────────────────────────────────
        _has_ext = (_any('spo2') or _any('rr') or
                    any(r.get('temperature', 0) >= 30.0 for r in all_valid_reports) or
                    any(r.get('gcs', 15) < 15 for r in all_valid_reports))
        if _has_ext:
            _pdf_section_header("Extended Vital Signs")
            _pdf_col_headers()
            add_row("Temperature",       "temp")
            if _any('spo2'):
                add_row("SpO2 (%)",      "spo2")
            if _any('rr'):
                add_row("Resp Rate (/min)", "rr")
            add_row("GCS (3-15)",        "gcs")
            add_row("Cap Refill Time",   "crt")

        # ─────────────────────────────────────────────
        # SECTION 4: Liver Function (if present)
        # ─────────────────────────────────────────────
        if _any('ast') or _any('alt') or _any('albumin') or _any('bilirubin_total'):
            _pdf_section_header("Liver Function Tests")
            _pdf_col_headers()
            if _any('ast'):
                add_row("AST / SGOT (IU/L)",     "ast",             "{:,}")
            if _any('alt'):
                add_row("ALT / SGPT (IU/L)",     "alt",             "{:,}")
            if _any('albumin'):
                add_row("Albumin (g/dL)",         "albumin",         "{:.1f}")
            if _any('bilirubin_total'):
                add_row("Total Bili (mg/dL)",     "bilirubin_total", "{:.1f}")
            if _any('bilirubin_direct'):
                add_row("Direct Bili (mg/dL)",    "bilirubin_direct","{:.1f}")

        # ─────────────────────────────────────────────
        # SECTION 5: Coagulation (if present)
        # ─────────────────────────────────────────────
        if _any('inr') or _any('pt') or _any('aptt') or _any('d_dimer'):
            _pdf_section_header("Coagulation Panel")
            _pdf_col_headers()
            if _any('pt'):
                add_row("PT (seconds)",           "pt",     "{:.1f}")
            if _any('inr'):
                add_row("INR",                    "inr",    "{:.2f}")
            if _any('aptt'):
                add_row("aPTT (seconds)",         "aptt",   "{:.1f}")
            if _any('d_dimer'):
                add_row("D-dimer (ng/mL FEU)",   "d_dimer", "{:,}")

        # ─────────────────────────────────────────────
        # SECTION 6: Renal & Electrolytes (if present)
        # ─────────────────────────────────────────────
        if _any('creatinine') or _any('bun') or _any('sodium') or _any('potassium'):
            _pdf_section_header("Renal Function & Electrolytes")
            _pdf_col_headers()
            if _any('creatinine'):
                add_row("Creatinine (mg/dL)",    "creatinine",  "{:.2f}")
            if _any('bun'):
                add_row("BUN (mg/dL)",           "bun",         "{:.1f}")
            if _any('sodium'):
                add_row("Sodium (mEq/L)",        "sodium",      "{:.0f}")
            if _any('potassium'):
                add_row("Potassium (mEq/L)",     "potassium",   "{:.1f}")
            if _any('bicarbonate'):
                add_row("HCO3 (mEq/L)",          "bicarbonate", "{:.1f}")

        # ─────────────────────────────────────────────
        # SECTION 7: Serology (if present)
        # ─────────────────────────────────────────────
        _has_serology = any(
            r.get(k, 'Not Done') not in ('Not Done', '')
            for r in all_valid_reports
            for k in ('ns1', 'igm', 'igg')
        )
        if _has_serology:
            _pdf_section_header("Serology")
            _pdf_col_headers()
            add_row("NS1 Antigen",           "ns1")
            add_row("IgM Anti-Dengue",       "igm")
            add_row("IgG Anti-Dengue",       "igg")

        # ─────────────────────────────────────────────
        # SECTION 8: Imaging (if present)
        # ─────────────────────────────────────────────
        _has_imaging = (
            any(r.get('pleural_effusion', False) for r in all_valid_reports) or
            _any('gallbladder_wall_mm') or
            _any('ascites_grade')
        )
        if _has_imaging:
            _pdf_section_header("Imaging Findings")
            _pdf_col_headers()
            add_row("Pleural Effusion",      "pe")
            if _any('gallbladder_wall_mm'):
                add_row("GB Wall (mm)",      "gb_wall")
            if _any('ascites_grade'):
                add_row("Ascites Grade",     "ascites")

        # ─────────────────────────────────────────────
        # SECTION 9: WHO / Symptoms — always shown
        # ─────────────────────────────────────────────
        _pdf_section_header("Clinical Signs & Symptoms")
        _pdf_col_headers()
        add_row("WHO Warning Signs (n)", "who_signs")
        add_row("Symptoms",              "symptoms")

        pdf.ln(4)
        
        pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "AI-Powered Clinical Assessment", 0, 1)
        pdf.set_font("Arial", size=11)
        rc = (231, 76, 60) if clinical_data['risk_prob'] > .5 else (46, 204, 113)
        pdf.set_text_color(*rc)
        pdf.cell(0, 8, _s(f"Risk Classification: {'HIGH RISK' if clinical_data['risk_prob'] > .5 else 'LOW RISK'} ({clinical_data['risk_prob']*100:.1f}%)"), 0, 1)
        pdf.set_text_color(0, 0, 0)
        if clinical_data.get('ci_lower') is not None:
            pdf.cell(0, 7, _s(f"95% Confidence Interval: {clinical_data['ci_lower']*100:.1f}% – {clinical_data['ci_upper']*100:.1f}%"), 0, 1)
        pdf.cell(0, 8, _s(f"24-Hour Platelet Forecast: {clinical_data['forecast_val']:,} cells/uL"), 0, 1)
        pdf.set_font("Arial", 'I', 9)
        pdf.cell(0, 6, _s("Model: Sensitivity 99.77% | Specificity 100% | AUC=0.9996 | Forecast R2=0.9953 | MAE=2,515 | n=2,455"), 0, 1)
        pdf.set_text_color(0, 0, 0)
        if clinical_data['fluid_rate'] > 0:
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 10, _s(f"Recommended IV Maintenance: {clinical_data['fluid_rate']} mL/hr (Holliday-Segar)"), 0, 1)
        if discharge:
            pdf.ln(2); pdf.set_font("Arial", 'B', 11)
            criteria, all_pass = discharge
            pdf.cell(0, 8, _s(f"Discharge Readiness: {'CRITERIA MET' if all_pass else 'NOT YET READY'}"), 0, 1)
            pdf.set_font("Arial", size=9)
            for c_name, c_pass, c_detail in criteria:
                symbol = '+' if c_pass else ('?' if c_pass is None else 'x')
                pdf.cell(0, 5, _s(f"  {symbol} {c_name}: {c_detail}"), 0, 1)
        if notes:
            pdf.ln(2); pdf.set_font("Arial", 'B', 11); pdf.cell(0, 8, "Clinician Notes:", 0, 1)
            pdf.set_font("Arial", 'I', 10); pdf.multi_cell(0, 5, _s(notes))
        pdf.ln(2); pdf.set_font("Arial", 'I', 9)
        pdf.multi_cell(0, 5, _s(
            "DISCLAIMER: This report is generated by a Clinical Decision Support System. "
            "All findings must be interpreted by a qualified clinician. "
            "This tool does not replace professional medical judgment."))

        def embed(stream, w=250):
            if not stream: return
            stream.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                f.write(stream.read())
            tmp = f.name
            pdf.image(tmp, x=10, y=None, w=w)
            try: os.unlink(tmp)
            except: pass

        if risk_stream:
            pdf.add_page(); pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Risk Factors Analysis", 0, 1); embed(risk_stream)
        if plot_stream:
            pdf.add_page(); pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Platelet Trajectory Analysis", 0, 1); embed(plot_stream)

        raw = pdf.output(dest='S')
        return bytes(raw) if isinstance(raw, (bytes, bytearray)) else raw.encode('latin-1')

# ══════════════════════════════════════════════════════════
#  16.  NAVIGATION
# ══════════════════════════════════════════════════════════
def render_custom_navigation():
    current_page = st.session_state.active_page
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        n1, n2 = st.columns(2)
        with n1:
            if st.button("Home", key="nav_home", width='stretch'):
                st.session_state.active_page = "Home"; st.rerun()
        with n2:
            if st.button("CDSS", key="nav_cdss", width='stretch'):
                st.session_state.active_page = "CDSS"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    return current_page

page = render_custom_navigation()

# ══════════════════════════════════════════════════════════
#  17.  HOME PAGE
# ══════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 24px 0;'>
        <h1 style='color:#e2e8f0; font-size:2.4rem; margin:0 0 10px 0; font-weight:800; letter-spacing:-0.01em;'>
            Dengue Clinical Decision Support System
        </h1>
        <p style='color:#718096; font-size:1.05rem; max-width:640px; margin:0 auto; line-height:1.7;'>
            An AI-powered platform for dengue risk stratification, WHO 2009 classification,
            trajectory-based severity scoring, serial alert monitoring, platelet forecast,
            and evidence-based clinical guidance.
        </p>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#2d3748; margin-bottom:32px;'>", unsafe_allow_html=True)

    # Workflow instructions
    st.markdown("""<div class="workflow-banner">
        <h4>Standard Workflow</h4>
        <p>
        <b>Step 1:</b> Upload lab reports via OCR (sidebar — supports JPG/PNG/PDF, multiple parameters auto-extracted) or enter values manually under Report A, B, C...
        &nbsp;&nbsp;<b>Units:</b> Set preferred units (kg/lbs, cm/ft, °C/°F) once in the <b>Unit Preferences</b> panel — applies globally.
        &nbsp;&nbsp;<b>Step 2:</b> Review auto-extracted values and click <b>Save Report Data</b> for each report.
        &nbsp;&nbsp;<b>Step 3:</b> After saving all reports, click <b>RUN ANALYSIS</b> to generate the full clinical dashboard.
        &nbsp;&nbsp;<b>Re-verification:</b> You may edit any report and save again before re-running the analysis.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 16px;">
    <div class="feature-card"><div class="fc-accent fc-accent-blue"></div><h3>AI Risk Assessment</h3><p>Random Forest (n=2,455) predicts severe dengue with 100% sensitivity and calibrated 95% CI from tree ensemble. Works with partial data — activates on platelet entry.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-blue"></div><h3>WHO 2009 Classification</h3><p>Full Group A/B/C triage extended with organ impairment: AST ≥1,000, INR ≥1.5, pleural effusion, ascites Grade ≥2, SpO₂ &lt;92%, GCS &lt;13. All guards against zero/missing values.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-red"></div><h3>Extended Organ Panel</h3><p>Hepatic (AST/ALT ratio, Albumin), Coagulation (PT/INR/aPTT/D-dimer + Bleeding Risk Score), Renal (Creatinine/eGFR CKD-EPI 2021/AKI KDIGO). Serial alert integration for all panels.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-amber"></div><h3>Plasma Leakage Score</h3><p>Multi-marker composite (0–100%) from Hct rise, D-dimer, albumin, pulse pressure, pleural effusion, gallbladder wall thickness, and ascites grade.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-green"></div><h3>Trajectory Engine</h3><p>Linear + quadratic OLS regression over serial CBCs. Outputs velocity (dy/dt), acceleration (d²y/dt²), R², and time-to-critical-threshold countdowns for PLT, Hct, and Shock Index.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-blue"></div><h3>Illness Phase Tracking</h3><p>Day-of-illness engine maps Febrile → Critical → Recovery with phase-specific action prompts. Defervescence detection on Day 4-6 triggers critical phase entry alert.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-amber"></div><h3>Serial Alert System</h3><p>Detects platelet crash velocity, AKI (KDIGO), AST doubling, coagulopathy (INR+PLT pattern), SpO₂ decline, and critical phase entry. Priority suppression prevents alert fatigue.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-green"></div><h3>Serology Integration</h3><p>NS1 antigen + IgM/IgG panel auto-detects primary vs. secondary dengue. Secondary pattern (50× risk amplifier) integrates with WHO classification and Bleeding Risk Score. Full assessment even without CBC.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-red"></div><h3>Bleeding Risk Score</h3><p>Composite BRS from platelet nadir, INR, aPTT, active bleeding signs, and secondary dengue flag — guides platelet transfusion threshold decisions per local protocol.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-blue"></div><h3>Fluid Overload Risk (FORS)</h3><p>Post-resuscitation paradox detector: falling Hct + polyuria + SpO₂ drop triggers early warning to reduce IV rate before pulmonary oedema develops.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-amber"></div><h3>Discharge Readiness</h3><p>WHO 7-criteria checklist: afebrile ≥48h, platelet trend and absolute threshold, haemodynamics, urine output, oral tolerance, and active warning sign clearance.</p></div>
    <div class="feature-card"><div class="fc-accent fc-accent-green"></div><h3>OCR Auto-Extraction</h3><p>Bulk upload lab reports (JPG/PNG/PDF). Extracts platelets, Hb, Hct, RBC, BP, HR, temperature, weight, height, urine output, timestamp, demographics and clinical notes — with per-field confidence scoring and plausibility validation.</p></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model Performance Card (sourced from evaluation graphs) ──────────
    st.markdown("""
    <div class="model-card">
      <h3>Algorithm Performance &amp; Data Transparency</h3>
      <div class="metric-pill-row">
        <div class="metric-pill">
          <span class="val">2,455</span>
          <span class="lbl">Training Patients</span>
        </div>
        <div class="metric-pill metric-pill-good">
          <span class="val">99.80%</span>
          <span class="lbl">Risk Accuracy</span>
        </div>
        <div class="metric-pill metric-pill-good">
          <span class="val">99.77%</span>
          <span class="lbl">Sensitivity</span>
        </div>
        <div class="metric-pill metric-pill-good">
          <span class="val">100.00%</span>
          <span class="lbl">Specificity</span>
        </div>
        <div class="metric-pill metric-pill-good">
          <span class="val">0.9996</span>
          <span class="lbl">AUC-ROC</span>
        </div>
        <div class="metric-pill metric-pill-good">
          <span class="val">0.9953</span>
          <span class="lbl">Forecast R²</span>
        </div>
        <div class="metric-pill metric-pill-warn">
          <span class="val">2,515</span>
          <span class="lbl">Forecast MAE (cells/µL)</span>
        </div>
      </div>

      <!-- Top Feature Importances (MDI from RandomForestClassifier) -->
      <div style="margin-top:14px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
        <div style="background:rgba(231,76,60,0.10);border:1px solid #e74c3c;border-radius:8px;padding:12px;text-align:center;">
          <div style="color:#f1948a;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;"></div>
          <div style="color:#e2e8f0;font-size:0.9rem;font-weight:700;">Platelet Count</div>
          <div style="color:#8b92a8;font-size:0.72rem;margin-top:3px;">MDI = 0.282 &nbsp;|&nbsp; 28.2%</div>
        </div>
        <div style="background:rgba(231,76,60,0.10);border:1px solid #e74c3c;border-radius:8px;padding:12px;text-align:center;">
          <div style="color:#f1948a;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;"></div>
          <div style="color:#e2e8f0;font-size:0.9rem;font-weight:700;">Shock Index</div>
          <div style="color:#8b92a8;font-size:0.72rem;margin-top:3px;">MDI = 0.267 &nbsp;|&nbsp; 26.7%</div>
        </div>
        <div style="background:rgba(231,76,60,0.10);border:1px solid #e74c3c;border-radius:8px;padding:12px;text-align:center;">
          <div style="color:#f1948a;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;"></div>
          <div style="color:#e2e8f0;font-size:0.9rem;font-weight:700;">Pleural Effusion</div>
          <div style="color:#8b92a8;font-size:0.72rem;margin-top:3px;">MDI = 0.145 &nbsp;|&nbsp; 14.5%</div>
        </div>
        <div style="background:rgba(52,152,219,0.10);border:1px solid #3498db;border-radius:8px;padding:12px;text-align:center;">
          <div style="color:#5dade2;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;"></div>
          <div style="color:#e2e8f0;font-size:0.9rem;font-weight:700;">INR</div>
          <div style="color:#8b92a8;font-size:0.72rem;margin-top:3px;">MDI = 0.124 &nbsp;|&nbsp; 12.4%</div>
        </div>
      </div>

      <!-- Secondary features row -->
      <div style="margin-top:10px;display:grid;grid-template-columns:repeat(6,1fr);gap:8px;">
        <div style="background:rgba(52,152,219,0.07);border:1px solid #2d4a6e;border-radius:6px;padding:8px;text-align:center;">
          <div style="color:#5dade2;font-size:0.65rem;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;"></div>
          <div style="color:#cbd5e0;font-size:0.82rem;font-weight:600;">AST</div>
          <div style="color:#8b92a8;font-size:0.66rem;">0.053</div>
        </div>
        <div style="background:rgba(52,152,219,0.07);border:1px solid #2d4a6e;border-radius:6px;padding:8px;text-align:center;">
          <div style="color:#5dade2;font-size:0.65rem;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;"></div>
          <div style="color:#cbd5e0;font-size:0.82rem;font-weight:600;">Ascites Grade</div>
          <div style="color:#8b92a8;font-size:0.66rem;">0.050</div>
        </div>
        <div style="background:rgba(52,152,219,0.07);border:1px solid #2d4a6e;border-radius:6px;padding:8px;text-align:center;">
          <div style="color:#5dade2;font-size:0.65rem;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;"></div>
          <div style="color:#cbd5e0;font-size:0.82rem;font-weight:600;">Pulse Pressure</div>
          <div style="color:#8b92a8;font-size:0.66rem;">0.015</div>
        </div>
        <div style="background:rgba(52,152,219,0.07);border:1px solid #2d4a6e;border-radius:6px;padding:8px;text-align:center;">
          <div style="color:#5dade2;font-size:0.65rem;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;"></div>
          <div style="color:#cbd5e0;font-size:0.82rem;font-weight:600;">GCS</div>
          <div style="color:#8b92a8;font-size:0.66rem;">0.015</div>
        </div>
        <div style="background:rgba(52,152,219,0.07);border:1px solid #2d4a6e;border-radius:6px;padding:8px;text-align:center;">
          <div style="color:#5dade2;font-size:0.65rem;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;"></div>
          <div style="color:#cbd5e0;font-size:0.82rem;font-weight:600;">WBC</div>
          <div style="color:#8b92a8;font-size:0.66rem;">0.010</div>
        </div>
        <div style="background:rgba(52,152,219,0.07);border:1px solid #2d4a6e;border-radius:6px;padding:8px;text-align:center;">
          <div style="color:#5dade2;font-size:0.65rem;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;"></div>
          <div style="color:#cbd5e0;font-size:0.82rem;font-weight:600;">Age / SpO₂ / Hct</div>
          <div style="color:#8b92a8;font-size:0.66rem;">≤ 0.008</div>
        </div>
      </div>

      <!-- Dataset Summary -->
      <div style="margin-top:12px;background:rgba(0,0,0,0.25);border:1px solid #2d3748;border-radius:6px;padding:10px 16px;display:flex;flex-wrap:wrap;gap:18px;align-items:center;">
        <span style="color:#8b92a8;font-size:0.75rem;">
          <b style="color:#e74c3c;">89.5% High-Risk</b> (n=2,197) &nbsp;·&nbsp;
          <b style="color:#2ecc71;">10.5% Low-Risk</b> (n=258) &nbsp;·&nbsp;
          Gender balanced: 50.1% F / 49.9% M &nbsp;·&nbsp;
          21-feature RandomForestClassifier (300 trees · max_depth=12) &nbsp;·&nbsp;
          5-Fold CV AUC: 0.9996 &nbsp;·&nbsp;
          Confusion matrix: TP=438, TN=52, FP=0, FN=1
        </span>
      </div>

      <!-- WHO Criteria frequency bar mini chart -->
      <div style="margin-top:12px;">
        <div style="color:#8b92a8;font-size:0.72rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">WHO 2009 Criteria Trigger Frequency (% patients meeting each criterion)</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;">
          <div style="background:rgba(231,76,60,0.08);border:1px solid #c0392b;border-radius:6px;padding:8px 10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <span style="color:#f1948a;font-size:0.72rem;font-weight:600;">Platelet &lt;100k</span>
              <span style="color:#e74c3c;font-size:0.78rem;font-weight:700;">46.1%</span>
            </div>
            <div style="height:4px;background:#1a1f2e;border-radius:2px;"><div style="width:46.1%;height:4px;background:#e74c3c;border-radius:2px;"></div></div>
          </div>
          <div style="background:rgba(231,76,60,0.08);border:1px solid #c0392b;border-radius:6px;padding:8px 10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <span style="color:#f1948a;font-size:0.72rem;font-weight:600;">Shock Index &gt;0.9</span>
              <span style="color:#e74c3c;font-size:0.78rem;font-weight:700;">44.5%</span>
            </div>
            <div style="height:4px;background:#1a1f2e;border-radius:2px;"><div style="width:44.5%;height:4px;background:#e74c3c;border-radius:2px;"></div></div>
          </div>
          <div style="background:rgba(231,76,60,0.08);border:1px solid #c0392b;border-radius:6px;padding:8px 10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <span style="color:#f1948a;font-size:0.72rem;font-weight:600;">Pleural Effusion</span>
              <span style="color:#e74c3c;font-size:0.78rem;font-weight:700;">30.1%</span>
            </div>
            <div style="height:4px;background:#1a1f2e;border-radius:2px;"><div style="width:30.1%;height:4px;background:#e74c3c;border-radius:2px;"></div></div>
          </div>
          <div style="background:rgba(52,152,219,0.08);border:1px solid #2d4a6e;border-radius:6px;padding:8px 10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <span style="color:#5dade2;font-size:0.72rem;font-weight:600;">INR ≥ 1.5</span>
              <span style="color:#3498db;font-size:0.78rem;font-weight:700;">25.5%</span>
            </div>
            <div style="height:4px;background:#1a1f2e;border-radius:2px;"><div style="width:25.5%;height:4px;background:#3498db;border-radius:2px;"></div></div>
          </div>
          <div style="background:rgba(52,152,219,0.08);border:1px solid #2d4a6e;border-radius:6px;padding:8px 10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <span style="color:#5dade2;font-size:0.72rem;font-weight:600;">AST ≥ 500 IU/L</span>
              <span style="color:#3498db;font-size:0.78rem;font-weight:700;">13.5%</span>
            </div>
            <div style="height:4px;background:#1a1f2e;border-radius:2px;"><div style="width:13.5%;height:4px;background:#3498db;border-radius:2px;"></div></div>
          </div>
          <div style="background:rgba(52,152,219,0.08);border:1px solid #2d4a6e;border-radius:6px;padding:8px 10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <span style="color:#5dade2;font-size:0.72rem;font-weight:600;">Ascites ≥ Grade 2</span>
              <span style="color:#3498db;font-size:0.78rem;font-weight:700;">12.6%</span>
            </div>
            <div style="height:4px;background:#1a1f2e;border-radius:2px;"><div style="width:12.6%;height:4px;background:#3498db;border-radius:2px;"></div></div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### About Dengue Fever")
    ac1, ac2 = st.columns([3, 2])
    with ac1:
        st.markdown("""
        <div class='intro-card'>
            <h2>Clinical Overview</h2>
            <p>Dengue fever is a mosquito-borne viral infection affecting 400 million people annually.
            Early identification of severe dengue — characterised by plasma leakage, severe bleeding,
            or organ impairment — is the primary determinant of mortality reduction.</p>
            <ul>
                <li><b>Febrile Phase (Days 1-3):</b> High fever, myalgia, headache, mild leukopenia</li>
                <li><b>Critical Phase (Days 4-6):</b> Rapid platelet drop, plasma leakage, risk of shock</li>
                <li><b>Recovery Phase (Days 7-10):</b> Platelet recovery, reabsorption of leaked fluids</li>
            </ul>
            <p style="margin-top:12px; font-size:0.85rem; color:#718096;">
            The CDSS models dengue as a dynamical system — each patient is tracked as a
            physiological trajectory, not a static snapshot — enabling time-to-threshold
            forecasting rather than simple risk labelling.
            </p>
        </div>""", unsafe_allow_html=True)
    with ac2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #161b24 0%, #1a1f2e 100%);
                    border:1px solid #2d3748; border-left:4px solid #e74c3c;
                    border-radius:10px; padding:20px;'>
            <h4 style='color:#e74c3c; margin:0 0 12px 0; font-size:0.9rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;'>WHO Warning Signs</h4>
            <ul style='color:#a0aec0; font-size:.88rem; line-height:2; margin:0; padding-left:1.2rem;'>
                <li>Abdominal pain or tenderness</li>
                <li>Persistent vomiting</li>
                <li>Clinical fluid accumulation</li>
                <li>Mucosal bleeding</li>
                <li>Lethargy or restlessness</li>
                <li>Liver enlargement &gt;2 cm</li>
                <li>Rapid increase in hematocrit</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align:center; color:#e2e8f0; font-size:1.3rem; font-weight:700; margin-bottom:4px;'>"
        "Resources &amp; Documentation</h3>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#718096; font-size:0.88rem; margin-bottom:24px;'>"
        "Comprehensive technical documentation and developer contact.</p>",
        unsafe_allow_html=True)

    doc_col, dev_col = st.columns(2, gap="medium")

    with doc_col:
        st.markdown(
            "<div style='background:linear-gradient(145deg,#0f1929,#1a2540);border:1px solid #1e3a5f;border-radius:14px;padding:28px 22px;text-align:center;position:relative;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.45);'>"
            "<div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#2980b9,#5dade2);border-radius:14px 14px 0 0;'></div>"
            "<div style='width:52px;height:52px;border-radius:13px;background:rgba(52,152,219,0.12);border:1px solid rgba(52,152,219,0.3);display:inline-flex;align-items:center;justify-content:center;font-size:1.5rem;margin-bottom:14px;'>&#128216;</div>"
            "<h4 style='color:#e2e8f0;margin:0 0 8px 0;font-size:1rem;font-weight:700;letter-spacing:0.02em;'>Documentation</h4>"
            "<p style='color:#718096;font-size:0.8rem;line-height:1.6;margin:0 0 20px 0;'>Complete technical guide covering architecture, ML model, analytical engines, and clinical parameters.</p>"
            "<a href='https://www.notion.so/Technical-Documentation-3041a46e5c678075a15dfff7db0cb0cb' target='_blank' "
            "style='display:inline-block;padding:10px 22px;background:linear-gradient(135deg,#2980b9,#3498db);color:white;font-weight:600;font-size:0.85rem;border-radius:8px;text-decoration:none;letter-spacing:0.03em;box-shadow:0 3px 10px rgba(52,152,219,0.4);'>"
            "Read Full Docs &#8594;</a>"
            "</div>",
            unsafe_allow_html=True)

    with dev_col:
        st.markdown(
            "<div style='background:linear-gradient(145deg,#0c1a2e,#152035);border:1px solid #1a3558;border-radius:14px;padding:28px 22px;text-align:center;position:relative;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.45);'>"
            "<div style='position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#005f8e,#0095e0);border-radius:14px 14px 0 0;'></div>"
            "<div style='width:52px;height:52px;border-radius:13px;background:rgba(0,119,181,0.12);border:1px solid rgba(0,119,181,0.3);display:inline-flex;align-items:center;justify-content:center;font-size:1.5rem;margin-bottom:14px;'>&#128188;</div>"
            "<h4 style='color:#e2e8f0;margin:0 0 8px 0;font-size:1rem;font-weight:700;letter-spacing:0.02em;'>Developer</h4>"
            "<p style='color:#718096;font-size:0.8rem;line-height:1.6;margin:0 0 20px 0;'>Connect for collaboration, feedback, feature requests, and clinical deployment discussions.</p>"
            "<a href='https://www.linkedin.com/in/annantgautam' target='_blank' "
            "style='display:inline-block;padding:10px 22px;background:linear-gradient(135deg,#005f8e,#0077b5);color:white;font-weight:600;font-size:0.85rem;border-radius:8px;text-decoration:none;letter-spacing:0.03em;box-shadow:0 3px 10px rgba(0,119,181,0.4);'>"
            "Connect on LinkedIn &#8594;</a>"
            "</div>",
            unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    _, cta_col, _ = st.columns([3, 2, 3])
    with cta_col:
        if st.button("Open CDSS", width='stretch', type="primary"):
            st.session_state.active_page = "CDSS"; st.rerun()

    st.markdown("""
    <div class='footer'>
        <p>Built with Streamlit &middot; scikit-learn &middot; SHAP &middot; Tesseract OCR &middot; WHO 2009 Dengue Guidelines<br>
        &copy; 2025 Dengue CDSS &nbsp;|&nbsp;
        <a href='https://www.notion.so/Technical-Documentation-3041a46e5c678075a15dfff7db0cb0cb' target='_blank'>Documentation</a> &nbsp;|&nbsp;
        <a href='https://www.linkedin.com/in/annantgautam' target='_blank'>LinkedIn</a>
        </p>
        <p style='margin-top:6px; font-size:0.75rem; color:#4a5568;'>
          n=2,455 &nbsp;&middot;&nbsp; RandomForestClassifier (300 trees) + RandomForestRegressor &nbsp;&middot;&nbsp;
          Sensitivity 99.77% &nbsp;&middot;&nbsp; Specificity 100% &nbsp;&middot;&nbsp; AUC=0.9996 &nbsp;&middot;&nbsp;
          Forecast R&sup2;=0.9953 &nbsp;&middot;&nbsp; MAE=2,515 cells/&micro;L
        </p>
        <p style='margin-top:4px; font-size:0.72rem; color:#4a5568;'>
          For Clinical Decision Support Only &bull; Not a Substitute for Professional Medical Judgment
        </p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  18.  CDSS PAGE
# ══════════════════════════════════════════════════════════
else:
    classifier, regressor, clf_features, reg_features = load_ai_engine()

    with st.sidebar:
        st.title("Dengue CDSS")
        st.caption("WHO 2009 · RFC 99.80% Acc · AUC 0.9996 · Forecast R²=0.9953")

        # ── Workflow Instructions ─────────────────────────
        st.markdown("""<div class="workflow-banner">
            <h4>Workflow</h4>
            <p>
            1. Upload or enter report values below.<br>
            2. Click <b>Save Report Data</b> to confirm each report.<br>
            3. After saving all reports, click <b>RUN ANALYSIS</b>.<br>
            <span style="color:#5dade2;">Re-save any report to update, then re-run analysis.</span>
            </p>
        </div>""", unsafe_allow_html=True)

        # ── Data Management ───────────────────────────────
        st.markdown("""<div class="clear-buttons-container"><div class="clear-buttons-header">Data Management</div></div>""", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Clear Current", width='stretch', key="clear_current_btn"):
                clear_current_report(st.session_state.get('active_report', 'A'))
                st.success(f"Report {st.session_state.get('active_report','A')} cleared!")
                st.rerun()
        with btn_col2:
            if st.button("Clear All", width='stretch', type="secondary", key="clear_all_btn"):
                clear_all_reports()
                st.success("All reports cleared!")
                st.rerun()

        # ── Bulk OCR Upload ─────────────────────────────── (MOVED: now BEFORE report selector)
        st.markdown('<div class="section-header">Lab Report Auto-Extraction</div>', unsafe_allow_html=True)
        st.markdown("""<div class="multi-ocr-panel">
            <h5>Bulk Upload — Lab Reports</h5>
            <p>Upload multiple lab reports (JPG, PNG, PDF). First file → Report A, second → Report B, and so on.<br>
            Extracts <b>platelets, Hb, Hct, RBC, BP, HR, vitals, timestamp, demographics,
            weight, height, urine output</b> and clinical notes — all with per-field confidence scoring
            and plausibility checks. Review extracted values and click <b>Accept</b> before saving.</p>
        </div>""", unsafe_allow_html=True)

        ocr_active = TESSERACT_AVAILABLE or PYPDF_AVAILABLE
        if not ocr_active:
            st.info(
                """📋 **OCR engine not available on this server.**  

"""
                """Image/PDF auto-extraction requires **Tesseract OCR** to be installed.  
"""
                """**To fix this:**  
"""
                """- **Streamlit Cloud:** add a  file to your repo containing   
"""
                """- **Linux/Mac local:** run  (Ubuntu) or  (Mac)  
"""
                """- **Windows local:** download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add it to PATH  

"""
                """In the meantime, please **enter your lab values manually** using the Report A/B/C tabs below."""
            )
        if ocr_active:
            uploaded_files = st.file_uploader(
                "Upload Reports (up to 5)",
                type=["jpg", "jpeg", "png", "pdf"],
                accept_multiple_files=True,
                key="bulk_ocr_uploader",
                help="First file -> Report A, second -> Report B...")

            if uploaded_files:
                if len(uploaded_files) > 5:
                    st.warning("Maximum 5 files. Only the first 5 will be processed.")
                    uploaded_files = uploaded_files[:5]

                for idx, uf in enumerate(uploaded_files):
                    rep_key_upload = REPORT_KEYS[idx]
                    if rep_key_upload in st.session_state.bulk_ocr_accepted:
                        continue
                    try:
                        ext      = uf.name.split('.')[-1].lower()
                        raw_text = ""
                        if ext == 'pdf':
                            raw_text = ocr_from_pdf_bytes(uf.read())
                        elif TESSERACT_AVAILABLE:
                            raw_text = extract_text_from_image(Image.open(uf))
                        else:
                            st.warning(
                                f"Cannot extract text from image **{uf.name}** — "
                                "Tesseract OCR binary is not installed on this server. "
                                "Please enter the values manually in the report tabs below."
                            )
                            continue
                        if raw_text.strip():
                            extracted = parse_lab_report_text(raw_text)
                            if extracted:
                                st.session_state.ocr_pending[rep_key_upload] = {
                                    'extracted': extracted,
                                    'filename':  uf.name,
                                }
                    except Exception as e:
                        st.warning(f"File {idx+1} ({uf.name}): extraction error — {e}")

                if st.session_state.ocr_pending:
                    FIELD_LABELS = {
                        'platelets':     'Platelets (cells/uL)',
                        'hb':            'Haemoglobin (g/dL)',
                        'hct':           'Hematocrit (%)',
                        'rbc':           'RBC (M/uL)',
                        'sys':           'Systolic BP (mmHg)',
                        'dia':           'Diastolic BP (mmHg)',
                        'hr':            'Heart Rate (bpm)',
                        'date':          'Report Date',
                        'time_hour':     'Time (24hr hour)',
                        'time_minute':   'Time (minute)',
                        'time_ampm':     'AM/PM',
                        'age':           'Age (years)',
                        'sex':           'Sex',
                        'patient_name':  'Patient Name',
                        'weight':        'Weight (kg)',
                        'height':        'Height (cm)',
                        'bmi':           'BMI',
                        'urine_vol':     'Urine Volume (mL)',
                        'urine_time':    'Urine Time (hrs)',
                        'clinical_notes':'Clinical Notes',
                    }
                    for rk in REPORT_KEYS:
                        if rk not in st.session_state.ocr_pending:
                            continue
                        if rk in st.session_state.bulk_ocr_accepted:
                            st.markdown(f"**Report {rk}** — <span style='color:#2ecc71'>Accepted</span>",
                                        unsafe_allow_html=True)
                            continue

                        pending   = st.session_state.ocr_pending[rk]
                        extracted = pending['extracted']
                        fname     = pending['filename']
                        st.markdown(f"**Report {rk}** — `{fname}`")
                        all_high_conf = True
                        for field, (val, conf, raw, note) in extracted.items():
                            cls, conf_lbl = confidence_color(conf)
                            lbl  = FIELD_LABELS.get(field, field)
                            note_str  = f" — {note}" if note else ""
                            is_extra  = field in ('age', 'sex', 'patient_name', 'weight', 'height',
                                                   'bmi', 'urine_vol', 'urine_time',
                                                   'time_hour', 'time_minute', 'clinical_notes')
                            badge = '<span class="auto-fill-badge">AUTO-FILL</span>' if is_extra else ""
                            st.markdown(
                                f"<small><b>{lbl}:</b> `{val}` "
                                f"<span class='{cls}'>{conf_lbl}</span>{badge}"
                                f"<span style='color:#8b92a8'>{note_str}</span></small>",
                                unsafe_allow_html=True)
                            if conf < 0.85:
                                all_high_conf = False

                        if not all_high_conf:
                            st.warning(f"Report {rk}: some fields are medium/low confidence — review before accepting.")

                        if st.button(f"Accept Report {rk}", key=f"bulk_accept_{rk}", width='stretch'):
                            _eu, _em, demo_updates = apply_ocr_to_session(rk, extracted)
                            # apply_ocr_to_session now handles age/sex sync internally.
                            # Only update non-widget session state keys here to avoid conflict.
                            if 'name' in demo_updates:
                                st.session_state.patient_name           = demo_updates['name']
                                st.session_state['patient_name_widget'] = demo_updates['name']
                            if 'clinical_notes' in demo_updates and not st.session_state.clinician_notes:
                                st.session_state.clinician_notes = demo_updates['clinical_notes']
                            st.session_state.bulk_ocr_accepted.add(rk)
                            st.success(f"Report {rk} accepted. Review values, then Save Report Data.")
                            st.rerun()
                            
                            if 'clinical_notes' in demo_updates and not st.session_state.clinician_notes:
                                st.session_state.clinician_notes          = demo_updates['clinical_notes']
                            st.session_state.bulk_ocr_accepted.add(rk)
                            st.success(f"Report {rk} accepted. Save the report to confirm.")
                            st.rerun()
                        st.divider()
                    pending_keys = [k for k in REPORT_KEYS
                                    if k in st.session_state.ocr_pending
                                    and k not in st.session_state.bulk_ocr_accepted]
                    if len(pending_keys) > 1:
                        if st.button("Accept ALL Reports", key="bulk_accept_all", width='stretch', type="primary"):
                            first_demo = {}
                            for rk in pending_keys:
                                _extracted = st.session_state.ocr_pending[rk]['extracted']
                                _eu, _em, demo = apply_ocr_to_session(rk, _extracted)
                                if not first_demo and demo:
                                    first_demo = demo
                                st.session_state.bulk_ocr_accepted.add(rk)
                            if 'age'  in first_demo:
                                st.session_state.patient_age           = first_demo['age']
                                st.session_state['patient_age_widget'] = first_demo['age']
                            if 'sex'  in first_demo:
                                st.session_state.patient_sex           = first_demo['sex']
                                st.session_state['patient_sex_widget'] = first_demo['sex']
                            if 'name' in first_demo:
                                st.session_state.patient_name          = first_demo['name']
                                st.session_state['patient_name_widget']= first_demo['name']
                            if 'clinical_notes' in first_demo and not st.session_state.clinician_notes:
                                st.session_state.clinician_notes       = first_demo['clinical_notes']
                            st.success("All reports accepted. Save each report below, then Run Analysis.")
                            st.rerun()
        else:
            st.markdown("""<div class="module-disabled">
                <p>OCR dependencies not available.<br>
                Install: pip install pypdf pytesseract pillow opencv-python</p>
            </div>""", unsafe_allow_html=True)
            
        # Demographics + Body Metrics moved into form below
        final_w_kg = 0.0; final_h_cm = 0.0; enable_metrics = False
        
        # ── Clinical Reports ──────────────────────────── (MOVED: now directly after OCR upload)
        st.markdown("### Clinical Reports")
        rep_key = st.radio(
            "Select Report", REPORT_KEYS, horizontal=True, key="report_selector",
            format_func=lambda x: f"  {x}  ")
        st.session_state.active_report = rep_key

        if st.session_state.reports[rep_key]['platelets'] == 0 and rep_key != 'A':
            prev_key = chr(ord(rep_key) - 1)
            if st.session_state.reports[prev_key]['platelets'] > 0:
                for k in ['hb', 'hct', 'rbc', 'sys', 'dia', 'hr']:
                    st.session_state.reports[rep_key][k] = st.session_state.reports[prev_key][k]

        # ── Global Unit Preferences — always-visible (no expander) ───
        st.markdown('<div class="section-header">Unit Preferences</div>', unsafe_allow_html=True)
        st.caption("Set preferred units here. Applies globally to Body Metrics and all temperature fields.")
        up_c1, up_c2, up_c3 = st.columns(3)
        with up_c1:
            st.markdown("<div style='color:#8b92a8; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; margin-bottom:6px;'>Weight</div>", unsafe_allow_html=True)
            u_wt = st.radio("Weight Unit", ["kg", "lbs"],
                index=0 if st.session_state.get('unit_weight', 'kg') == "kg" else 1,
                key="unit_weight_radio", label_visibility="collapsed")
            st.session_state.unit_weight = u_wt
        with up_c2:
            st.markdown("<div style='color:#8b92a8; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; margin-bottom:6px;'>Height</div>", unsafe_allow_html=True)
            u_ht = st.radio("Height Unit", ["cm", "ft/in"],
                index=0 if st.session_state.get('unit_height', 'cm') == "cm" else 1,
                key="unit_height_radio", label_visibility="collapsed")
            st.session_state.unit_height = u_ht
        with up_c3:
            st.markdown("<div style='color:#8b92a8; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; margin-bottom:6px;'>Temperature</div>", unsafe_allow_html=True)
            u_tp = st.radio("Temp Unit", ["°C", "°F"],
                index=0 if st.session_state.get('unit_temp', '°C') == "°C" else 1,
                key="unit_temp_radio", label_visibility="collapsed")
            st.session_state.unit_temp = u_tp
        st.markdown(
            f"<div style='background:rgba(52,152,219,0.08); border:1px solid #2d4a6e; "
            f"border-radius:6px; padding:8px 12px; margin-top:10px; font-size:0.76rem; color:#5dade2;'>"
            f"Active: <b>{st.session_state.get('unit_weight','kg')}</b> · "
            f"<b>{st.session_state.get('unit_height','cm')}</b> · "
            f"<b>{st.session_state.get('unit_temp','°C')}</b></div>",
            unsafe_allow_html=True)
        st.divider() 
       
        st.markdown(f"#### Report {rep_key} Data Entry")
        data = st.session_state.reports[rep_key]
        if 'symptoms' not in data:
            data['symptoms'] = []

        st.markdown('<div class="section-header">Date & Time</div>', unsafe_allow_html=True)
        st.caption("Select the date and time of this report/collection.")

        # Four equal columns: Date | Hour | Min | AM/PM — one widget per column, no stacking
        _dt_date_col, _dt_hr_col, _dt_min_col, _dt_ampm_col = st.columns([3, 1, 1, 1])

        with _dt_date_col:
            st.markdown(
                "<div style='color:#8b92a8; font-size:0.72rem; font-weight:600; "
                "text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px;'>Date</div>",
                unsafe_allow_html=True)
            form_date = st.date_input(
                "Report Date",
                value=data.get('date', datetime.date.today()),
                key=f"date_{rep_key}",
                label_visibility="collapsed")

        with _dt_hr_col:
            st.markdown(
                "<div style='color:#8b92a8; font-size:0.72rem; font-weight:600; "
                "text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px;'>Hour</div>",
                unsafe_allow_html=True)
            _hour_val = int(data.get('time_hour', 9))
            _hour_val = _hour_val if 1 <= _hour_val <= 12 else 9
            _pre_hour = st.number_input(
                "Hour", 1, 12,
                value=_hour_val,
                key=f"pre_hour_{rep_key}",
                label_visibility="collapsed")

        with _dt_min_col:
            st.markdown(
                "<div style='color:#8b92a8; font-size:0.72rem; font-weight:600; "
                "text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px;'>Min</div>",
                unsafe_allow_html=True)
            _pre_min = st.number_input(
                "Min", 0, 59,
                value=int(data.get('time_minute', 0)),
                key=f"pre_min_{rep_key}",
                label_visibility="collapsed")

        with _dt_ampm_col:
            st.markdown(
                "<div style='color:#8b92a8; font-size:0.72rem; font-weight:600; "
                "text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px;'>AM/PM</div>",
                unsafe_allow_html=True)
            _ampm_idx = 0 if data.get('time_ampm', 'AM') == "AM" else 1
            _pre_ampm = st.selectbox(
                "AM/PM", ["AM", "PM"],
                index=_ampm_idx,
                key=f"pre_ampm_{rep_key}",
                label_visibility="collapsed")

        st.caption(f"Report time: {_pre_hour:02d}:{_pre_min:02d} {_pre_ampm}")
        # ── Seasonal Dengue Risk Indicator ───────────────────────────────────────
        # Uses the same get_season_score/get_season_meta engine as the analysis tab.
        # Runs on the currently-selected report date, not today's date, so if a
        # clinician back-enters a historical report the badge reflects that date.
        try:
            _season_score = get_season_score(form_date)
            _season_meta  = get_season_meta(_season_score)
            # ↓ Extract BEFORE the f-string — no backslashes inside {} allowed in Python < 3.12
            _s_cls     = _season_meta['cls']
            _s_label   = _season_meta['label']
            _s_context = _season_meta['context']
            _s_tip     = _season_meta['tip']
            _s_month   = form_date.strftime('%B')
            st.markdown(
                f"<div style='margin-top:10px; padding:10px 14px; "
                f"background:rgba(0,0,0,0.15); border:1px solid #2d3748; "
                f"border-radius:8px;'>"
                f"<div style='color:#8b92a8; font-size:0.68rem; font-weight:600; "
                f"text-transform:uppercase; letter-spacing:.05em; margin-bottom:4px;'>"
                f"Seasonal Dengue Risk — {_s_month}</div>"
                f"<span class='{_s_cls}'>{_s_label}</span>"
                f"<div style='color:#718096; font-size:0.73rem; margin-top:5px;'>{_s_context}</div>"
                f"<div style='color:#4a5568; font-size:0.68rem; margin-top:3px; font-style:italic;'>"
                f"{_s_tip}</div>"
                f"</div>",
                unsafe_allow_html=True)
        except Exception:
            pass  # silently skip if date not yet valid
        
        
        # ── Patient Demographics ──────────────────────────
        st.markdown('<div class="section-header">Patient Demographics</div>', unsafe_allow_html=True)
        ocr_filled_demo = (rep_key in st.session_state.bulk_ocr_accepted and
                           (st.session_state.patient_age != 25
                            or st.session_state.patient_sex != "Male"
                            or st.session_state.patient_name != ""))
        if ocr_filled_demo:
            st.markdown('<span class="auto-fill-badge">Auto-filled from report</span>', unsafe_allow_html=True)

        sb_patient_name = st.text_input(
            "Patient Name",
            placeholder="Auto-filled from report or enter manually",
            key="patient_name_widget")                    # ← NO value= param
        st.session_state.patient_name = sb_patient_name  # keep backing field in sync

        _dc1, _dc2 = st.columns(2)
        with _dc1:
            # Do NOT pass index= when key is managed by session state API —
            # that causes "widget created with default but also set via Session State" error.
            # Session state is already seeded in _SS_DEFAULTS and apply_ocr_to_session.
            if 'patient_sex_widget' not in st.session_state:
                st.session_state['patient_sex_widget'] = st.session_state.get('patient_sex', 'Male')
            sb_sex = st.selectbox("Sex", ["Male", "Female"], key="patient_sex_widget")
            st.session_state.patient_sex = sb_sex
        with _dc2:
            if 'patient_age_widget' not in st.session_state:
                st.session_state['patient_age_widget'] = int(st.session_state.get('patient_age', 25))
            sb_age = st.number_input("Age (years)", 0, 120, key="patient_age_widget")
            st.session_state.patient_age = int(sb_age)

        # ── Body Metrics ──────────────────────────────────
        st.markdown('<div class="section-header">Body Metrics  <span style="color:#5dade2;font-weight:400;font-size:0.78rem;">(weight · height · BMI)</span></div>', unsafe_allow_html=True)
        ocr_metrics = st.session_state.metrics_from_ocr
        if ocr_metrics.get('enabled'):
            st.markdown('<span class="auto-fill-badge">Auto-filled from report</span>', unsafe_allow_html=True)
        st.caption("Fill weight and height to activate BMI, fluid rate, and urine weight sync.")
        _w_unit = st.session_state.get('unit_weight', 'kg')
        _h_unit = st.session_state.get('unit_height', 'cm')
        bm_c1, bm_c2 = st.columns(2)
        with bm_c1:
            st.markdown(f"<div style='color:#8b92a8;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;'>Weight ({_w_unit})</div>", unsafe_allow_html=True)
            
            # ── seed session state from OCR ONCE, before widget renders ──
            if ocr_metrics.get('weight') and st.session_state.get('weight_val', 0.0) == 0.0:
                st.session_state['weight_val'] = float(ocr_metrics['weight'])
            w_val = st.number_input(f"Weight ({_w_unit})", 0.0, 500.0, step=0.1,
                key="weight_val", label_visibility="collapsed")
        with bm_c2:
            if _h_unit == "ft/in":
                st.markdown("<div style='color:#8b92a8;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;'>Height (ft / in)</div>", unsafe_allow_html=True)
                _fc1, _fc2 = st.columns(2)
                with _fc1: ft   = st.number_input("Feet",   0, 8,  key="ft_val",  label_visibility="collapsed")
                with _fc2: inch = st.number_input("Inches", 0, 11, key="in_val",  label_visibility="collapsed")
                final_h_cm = ft * 30.48 + inch * 2.54
            else:
                st.markdown("<div style='color:#8b92a8;font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;'>Height (cm)</div>", unsafe_allow_html=True)
                # ── seed from OCR ONCE before widget renders ──
                if ocr_metrics.get('height') and st.session_state.get('height_cm', 0.0) == 0.0:
                    st.session_state['height_cm'] = float(ocr_metrics['height'])
                h_val = st.number_input("Centimeters", 0.0, 250.0, step=0.1,
                    key="height_cm", label_visibility="collapsed")
                final_h_cm = h_val
                
        final_w_kg     = w_val if _w_unit == "kg" else w_val * 0.453592
        enable_metrics = (final_w_kg > 0 and final_h_cm > 0)
        if enable_metrics:
            bmi_val = calculate_bmi(final_w_kg, final_h_cm)
            st.success(f"BMI: **{bmi_val:.1f}** — Body metrics active")
        elif final_w_kg > 0:
            st.info("Enter height to complete BMI.")
        else:
            st.caption("Enter weight and height to activate body metrics.")
        st.divider()
        
        # ── Illness Phase Tracking ────────────────────
        st.markdown('<div class="section-header">Illness Phase Tracking</div>', unsafe_allow_html=True)
        with st.expander("Symptom Onset Date  (enables phase engine)", expanded=True):
            st.caption("Enter the date symptoms first appeared. Enables phase-aware trajectory analysis.")
            onset_date_input = st.date_input(
                "Date of symptom onset",
                value=st.session_state.get('patient_onset_date', datetime.date.today()),
                key="onset_date_widget",
                max_value=datetime.date.today())
            st.session_state.patient_onset_date = onset_date_input

            if onset_date_input:
                illness_day = (datetime.date.today() - onset_date_input).days + 1
                st.session_state.patient_illness_day = illness_day
                phase_info = get_dengue_phase(illness_day)
                st.session_state.patient_dengue_phase = phase_info['phase']
                st.markdown(
                    f"<div style='background:rgba(0,0,0,0.2); border:1px solid {phase_info['color']}; "
                    f"border-left:4px solid {phase_info['color']}; border-radius:6px; padding:10px 14px; margin-top:8px;'>"
                    f"<div style='color:{phase_info['color']}; font-weight:700; font-size:0.88rem;'>"
                    f"Day {illness_day} of Illness — {phase_info['phase']}</div>"
                    f"<div style='color:#8b92a8; font-size:0.75rem; margin-top:4px;'>{phase_info['action']}</div>"
                    f"</div>", unsafe_allow_html=True)

        # ── Serology ──────────────────────────────────
        st.markdown('<div class="section-header">Serology</div>', unsafe_allow_html=True)
        with st.expander("Dengue Serology / NS1", expanded=True):
            st.caption("NS1 antigen enables early confirmation (Day 1-5). IgM/IgG determines primary vs. secondary infection.")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                ns1_val = st.selectbox("NS1 Antigen", ["Not Done", "Positive", "Negative"],
                                       key="ns1_widget",
                                       index=["Not Done","Positive","Negative"].index(
                                           st.session_state.get('ns1_result','Not Done')))
                st.session_state.ns1_result = ns1_val
            with sc2:
                igm_val = st.selectbox("IgM Anti-Dengue", ["Not Done", "Reactive", "Non-Reactive"], key="igm_widget")
            with sc3:
                igg_val = st.selectbox("IgG Anti-Dengue", ["Not Done", "Reactive", "Non-Reactive"], key="igg_widget")

            secondary_flag = (igg_val == "Reactive" and igm_val in ("Reactive","Not Done"))
            if secondary_flag:
                st.markdown(
                    "<div style='background:rgba(231,76,60,0.1); border:1px solid #e74c3c; "
                    "border-radius:6px; padding:8px 12px; margin-top:6px;'>"
                    "<span style='color:#f1948a; font-size:0.78rem; font-weight:700;'>"
                    "⚠ Secondary dengue pattern detected — 50× higher severe dengue risk</span></div>",
                    unsafe_allow_html=True)
            st.session_state.is_secondary_dengue = secondary_flag

        # rep_key set above
        if rep_key in st.session_state.bulk_ocr_accepted:
            st.markdown(
                f'Report {rep_key} pre-filled from OCR — '
                f'<span class="auto-fill-badge">Review values, then Save</span>',
                unsafe_allow_html=True)

        _form_ver = st.session_state.get(f"form_ver_{rep_key}", 0)
        _form_key = f"report_form_{rep_key}_v{_form_ver}"

        # ── Temperature unit — driven by global Unit Preferences (no per-report widget needed) ──
        # Sync the per-report key so the form's temperature field reads the correct unit.
        _global_temp = st.session_state.get('unit_temp', '°C')
        if st.session_state.get(f"temp_unit_{rep_key}") != _global_temp:
            st.session_state[f"temp_unit_{rep_key}"] = _global_temp
        with st.form(key=_form_key, border=False):

            # Date & Time already captured above the form — carry values in
            form_hour   = _pre_hour
            form_minute = _pre_min
            form_ampm   = _pre_ampm

            st.markdown('<div class="section-header">Laboratory Values</div>', unsafe_allow_html=True)
            form_platelets = st.number_input("Platelet Count (cells/uL)", 0, 1000000, value=int(data['platelets']), step=1000)
            lc1, lc2, lc3 = st.columns(3)
            with lc1: form_hb  = st.number_input("Hb (g/dL)",  2.0, 25.0, value=float(data['hb']),  step=0.1)
            with lc2: form_rbc = st.number_input("RBC (M/uL)", 0.1, 10.0, value=float(data['rbc']), step=0.1)
            with lc3: form_hct = st.number_input("Hct (%)",    5.0, 70.0, value=float(data['hct']), step=0.1)
            if form_hb > 2 and form_hct > 5:
                expected_hct = form_hb * 3
                if abs(form_hct - expected_hct) > 8:
                    st.caption(f"Hb x3 = {expected_hct:.1f}% vs Hct {form_hct:.1f}% — verify values")

            st.markdown('<div class="section-header">Hemodynamics</div>', unsafe_allow_html=True)
            hc1, hc2, hc3 = st.columns(3)
            with hc1: form_sys = st.number_input("SBP (mmHg)", 50, 250, value=int(data['sys']))
            with hc2: form_dia = st.number_input("DBP (mmHg)", 30, 150, value=int(data['dia']))
            with hc3: form_hr  = st.number_input("HR (bpm)",   40, 200, value=int(data['hr']))
            if form_sys > form_dia:
                live_map = calculate_map(form_sys, form_dia)
                live_si  = form_hr / form_sys if form_sys > 0 else 0
                live_pp  = form_sys - form_dia
                st.caption(f"MAP: {live_map:.1f} mmHg  |  Shock Index: {live_si:.2f}  |  Pulse Pressure: {live_pp} mmHg")
            elif form_sys <= form_dia:
                st.error("Systolic BP must be greater than Diastolic BP")

            st.markdown('<div class="section-header">Fluid Balance</div>', unsafe_allow_html=True)

            ocr_has_urine = (rep_key in st.session_state.bulk_ocr_accepted and
                             (data.get('urine_vol', 0) > 0 or data.get('urine_time', 1.0) != 1.0))
            urine_expander_open = ocr_has_urine or data.get('urine_output', 0.0) > 0
            if ocr_has_urine:
                st.markdown('<span class="auto-fill-badge">Auto-filled from report</span>', unsafe_allow_html=True)

            with st.expander("Urine Output Calculator", expanded=True):
                st.markdown("""<div class="urine-info-card"><h5>How It Works</h5>
                    <p>Enter volume collected and collection period. Weight syncs from Body Metrics when available.
                    Rate = Volume / (Weight x Time). Target: 1.0-2.0 mL/kg/hr. Set Volume to 0 to skip.</p>
                    </div>""", unsafe_allow_html=True)

                weight_source     = "manual"
                form_urine_weight = float(data.get('urine_weight', 0.0))
                if enable_metrics and final_w_kg > 0:
                    form_urine_weight = final_w_kg
                    weight_source     = "profile"
                elif data.get('ocr_weight', 0) > 0:
                    form_urine_weight = data['ocr_weight']
                    weight_source     = "ocr"

                uw_col1, uw_col2 = st.columns([4, 1])
                with uw_col1:
                    form_urine_weight_input = st.number_input(
                        "Patient Weight (kg)", 0.0, 500.0,
                        value=float(form_urine_weight),
                        step=0.5,
                        disabled=(weight_source in ("profile", "ocr")))
                with uw_col2:
                    if weight_source == "profile":
                        st.markdown('<div class="weight-source-badge">Profile</div>', unsafe_allow_html=True)
                    elif weight_source == "ocr":
                        st.markdown('<div class="weight-source-badge" style="border-color:#2ecc71;color:#2ecc71;background:rgba(46,204,113,0.1);">OCR</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="weight-source-badge" style="background:rgba(243,156,18,0.15);border-color:#f39c12;color:#f39c12;">Manual</div>', unsafe_allow_html=True)
                if weight_source == "manual":
                    form_urine_weight = form_urine_weight_input

                u_col1, u_col2 = st.columns(2)
                with u_col1:
                    form_urine_vol = st.number_input("Volume (mL)", 0.0, 5000.0, value=float(data.get('urine_vol', 0.0)), step=10.0)
                with u_col2:
                    form_urine_time = st.number_input("Time (hrs)", 0.1, 24.0, value=float(data.get('urine_time', 1.0)), step=0.5)

                preview_rate = calculate_urine_rate(form_urine_vol, form_urine_time, form_urine_weight)
                if preview_rate > 0:
                    status, recommendation = interpret_urine_rate(preview_rate)
                    st.markdown(f"""<div class="urine-rate-display">
                        <p class="rate-value">{preview_rate:.2f} mL/kg/hr</p>
                        <p class="rate-label">{status}</p></div>""", unsafe_allow_html=True)
                    if preview_rate < 0.5:   st.warning(recommendation)
                    elif preview_rate < 1.0: st.info(recommendation)
                    elif preview_rate > 4.0: st.info(recommendation)
                    else:                    st.success(recommendation)
                elif form_urine_weight > 0:
                    st.info("Enter Volume (mL) above to calculate urine output rate.")
                else:
                    st.caption("Enter weight and volume to activate the urine output calculator.")

            st.markdown('<div class="section-header">WHO Warning Signs</div>', unsafe_allow_html=True)
            who_opts = ["Abdominal Pain", "Persistent Vomiting", "Mucosal Bleeding",
                        "Lethargy/Restlessness", "Liver Enlargement >2cm"]
            form_who = st.multiselect("Select applicable signs", who_opts, default=data.get('who', []))

            st.markdown('<div class="section-header">General Symptoms</div>', unsafe_allow_html=True)
            symptom_opts = ["Fever", "Headache", "Joint Pain", "Vomiting", "Bleeding"]
            form_symptoms = st.multiselect("Select applicable symptoms", symptom_opts, default=data.get('symptoms', []))
            
            # ── CBC Differential ─────────────────────────
            st.markdown('<div class="section-header">CBC Differential</div>', unsafe_allow_html=True)
            with st.expander("WBC + Differential", expanded=True):
                st.caption("Leukopenia <4,000 and neutropenia <40% are hallmark dengue signs.")
                dc1, dc2, dc3, dc4 = st.columns(4)
                with dc1: form_wbc  = st.number_input("WBC (cells/uL)", 0, 100000, value=int(data.get('wbc',0)), step=100)
                with dc2: form_neu  = st.number_input("Neutrophil %",   0.0, 100.0, value=float(data.get('neutrophil_pct',0.0)), step=0.5)
                with dc3: form_lym  = st.number_input("Lymphocyte %",   0.0, 100.0, value=float(data.get('lymphocyte_pct',0.0)), step=0.5)
                with dc4: form_mpv  = st.number_input("MPV (fL)",       0.0, 20.0,  value=float(data.get('mpv',0.0)), step=0.1)
                if form_wbc > 0 and form_wbc < 4000:
                    st.caption(f"⚠ Leukopenia: WBC {form_wbc:,} — hallmark dengue sign")
                if form_neu > 0 and form_neu < 40:
                    st.caption(f"⚠ Neutropenia: {form_neu:.1f}% — supports dengue diagnosis")
                    
            # ── Vital Signs Expansion ─────────────────────────────
            st.markdown('<div class="section-header">Extended Vitals</div>', unsafe_allow_html=True)
            with st.expander("Temperature · SpO2 · RR · GCS · CRT", expanded=True):
                # Unit is selected above the form — read from session_state (always current)
                _temp_unit = st.session_state.get(f"temp_unit_{rep_key}", "°C")
                st.caption(f"ℹ Temperature unit: **{_temp_unit}** (toggle above). Values stored internally as °C.")
                
                ev1, ev2, ev3 = st.columns(3)
                with ev1:
                    _stored_c = float(data.get('temperature', 0.0))
                    if 0 < _stored_c < 30.0:
                        _stored_c = 0.0  # sanitise implausible stored value
                    if _temp_unit == "°F":
                        _disp_val = round(_stored_c * 9/5 + 32, 1) if _stored_c >= 30.0 else 98.6
                        _t_lbl    = "Temperature (°F)"
                    else:
                        _disp_val = _stored_c if _stored_c >= 30.0 else 37.0
                        _t_lbl    = "Temperature (°C)"
                    # Wide range (0-200) covers both °C (30-43) and °F (86-109.4)
                    # without requiring a rerun between unit selection and value entry
                    form_temp_raw = st.number_input(
                        _t_lbl, 0.0, 200.0,
                        value=float(_disp_val),
                        step=0.1,
                        key=f"temp_raw_{rep_key}_{_temp_unit}")
                    form_temp = round((form_temp_raw - 32) * 5/9, 2) if _temp_unit == "°F" else form_temp_raw
                    form_spo2 = st.number_input("SpO2 (%)", 0, 100, value=int(data.get('spo2', 0)))
                with ev2:
                    form_rr  = st.number_input("Resp Rate (/min)", 0, 60,  value=int(data.get('rr', 0)))
                    form_gcs = st.number_input("GCS (3-15)",        3, 15,  value=int(data.get('gcs', 15)))
                with ev3:
                    form_crt = st.selectbox(
                        "Cap. Refill Time",
                        ["Not assessed", "<2 sec (Normal)", "2-3 sec (Borderline)", ">3 sec (Prolonged)"],
                        index=int(data.get('crt', 0)),
                        key=f"crt_{rep_key}")
                if form_spo2 > 0 and form_spo2 < 93:
                    st.error(f"SpO2 {form_spo2}% — hypoxaemia: obtain CXR, assess for pleural effusion")
                elif form_spo2 > 0 and form_spo2 < 95:
                    st.warning(f"SpO2 {form_spo2}% — borderline: monitor closely")
                if form_gcs < 13:
                    st.error(f"GCS {form_gcs} — dengue encephalopathy criteria: immediate review")

            # ── Liver Function Tests ──────────────────────
            st.markdown('<div class="section-header">Liver Function Tests</div>', unsafe_allow_html=True)
            with st.expander("AST · ALT · Albumin · Bilirubin", expanded=True):
                st.caption("AST >1,000 is a standalone WHO Severe Dengue criterion. AST:ALT >2.0 is dengue-specific.")
                lft1, lft2 = st.columns(2)
                with lft1:
                    form_ast    = st.number_input("AST / SGOT (IU/L)", 0, 10000, value=int(data.get('ast', 0)), step=1)
                    form_alt    = st.number_input("ALT / SGPT (IU/L)", 0, 10000, value=int(data.get('alt', 0)), step=1)
                    form_alb    = st.number_input("Albumin (g/dL)",    0.0, 6.0,  value=float(data.get('albumin', 0.0)), step=0.1)
                with lft2:
                    form_tbili  = st.number_input("Total Bilirubin (mg/dL)",  0.0, 30.0, value=float(data.get('bilirubin_total', 0.0)), step=0.1)
                    form_dbili  = st.number_input("Direct Bilirubin (mg/dL)", 0.0, 20.0, value=float(data.get('bilirubin_direct', 0.0)), step=0.1)
                if form_ast > 0 and form_alt > 0:
                    _ratio, _pat, _sev, _col = interpret_ast_alt(form_ast, form_alt)
                    st.markdown(f"<span style='color:{_col}; font-size:0.8rem; font-weight:700;'>"
                                f"AST:ALT = {_ratio:.1f} — {_sev}</span>", unsafe_allow_html=True)
                    st.caption(_pat)
                if form_ast >= 1000:
                    st.error("AST ≥1,000 IU/L — WHO Severe Dengue criterion MET: hepatic failure risk")
                elif form_ast >= 500:
                    st.warning("AST ≥500 IU/L — Severe hepatitis approaching WHO criterion")

            # ── Coagulation Panel ─────────────────────────
            st.markdown('<div class="section-header">Coagulation Panel</div>', unsafe_allow_html=True)
            with st.expander("PT/INR · aPTT · D-dimer", expanded=True):
                st.caption("INR >1.5 + PLT <50k = CRITICAL coagulopathy pattern. D-dimer correlates with plasma leakage.")
                cg1, cg2 = st.columns(2)
                with cg1: form_pt     = st.number_input("PT (seconds)",       0.0, 60.0,  value=float(data.get('pt', 0.0)),  step=0.1)
                with cg2: form_inr    = st.number_input("INR",                0.0, 10.0,  value=float(data.get('inr', 0.0)), step=0.01)
                cg3, cg4 = st.columns(2)
                with cg3: form_aptt   = st.number_input("aPTT (seconds)",     0.0, 120.0, value=float(data.get('aptt', 0.0)),step=0.5)
                with cg4: form_ddimer = st.number_input("D-dimer (ng/mL FEU)",0,   20000, value=int(data.get('d_dimer', 0)), step=50)
                if form_inr >= 1.5 and int(data.get('platelets',999999)) < 50000:
                    st.error(f"CRITICAL: INR {form_inr:.1f} + PLT <50k — coagulopathy pattern, bleeding risk HIGH")
                elif form_inr >= 2.0:
                    st.error(f"INR {form_inr:.1f} ≥2.0 — contraindicates invasive procedures")

            # ── Renal Function ────────────────────────────
            st.markdown('<div class="section-header">Renal Function</div>', unsafe_allow_html=True)
            with st.expander("Creatinine · BUN · Electrolytes", expanded=True):
                st.caption("AKI = creatinine rise >0.3 mg/dL in 48h. BUN:Cr >20:1 suggests prerenal cause.")
                rf1, rf2 = st.columns(2)
                with rf1:
                    form_creat = st.number_input("Creatinine (mg/dL)",  0.0, 15.0, value=float(data.get('creatinine',0.0)), step=0.01)
                    form_bun   = st.number_input("BUN (mg/dL)",         0.0, 150.0,value=float(data.get('bun',0.0)),        step=0.5)
                    form_na    = st.number_input("Sodium (mEq/L)",      0.0, 170.0,value=float(data.get('sodium',0.0)),     step=0.5)
                with rf2:
                    form_k     = st.number_input("Potassium (mEq/L)",   0.0, 8.0,  value=float(data.get('potassium',0.0)), step=0.1)
                    form_hco3  = st.number_input("Bicarbonate (mEq/L)", 0.0, 40.0, value=float(data.get('bicarbonate',0.0)),step=0.5)
                if form_creat > 0:
                    egfr_live = calculate_egfr_ckdepi(form_creat,
                                                       int(st.session_state.get('patient_age',25)),
                                                       st.session_state.get('patient_sex','Male'))
                    st.caption(f"eGFR (CKD-EPI 2021): {egfr_live} mL/min/1.73m²")
                    if egfr_live < 60:
                        st.warning(f"eGFR {egfr_live} — renal function impaired: nephrology assessment recommended")

            # ── Imaging Findings ─────────────────────────
            st.markdown('<div class="section-header">Imaging Findings</div>', unsafe_allow_html=True)
            with st.expander("Pleural Effusion · Ultrasound Findings", expanded=True):
                st.caption("Confirmed pleural effusion = definitive plasma leakage — auto-triggers WHO Group C criteria.")
                img1, img2, img3 = st.columns(3)
                with img1:
                    form_pleural = st.checkbox("Pleural effusion confirmed", value=data.get('pleural_effusion', False))
                    if form_pleural:
                        st.markdown("<span style='color:#e74c3c; font-size:0.75rem; font-weight:700;'>WHO Severe Dengue criterion triggered</span>", unsafe_allow_html=True)
                with img2:
                    form_gb_wall = st.number_input("GB wall thickness (mm)", 0.0, 20.0, value=float(data.get('gallbladder_wall_mm', 0.0)), step=0.5)
                    if form_gb_wall >= 5:
                        st.caption("≥5mm: highly specific for dengue — plasma leakage precursor")
                with img3:
                    form_ascites = st.selectbox("Ascites Grade", [0, 1, 2, 3],
                                                index=int(data.get('ascites_grade', 0)),
                                                format_func=lambda x: f"Grade {x} ({'None' if x==0 else 'Trace' if x==1 else 'Moderate' if x==2 else 'Severe'})")
                    if form_ascites >= 2:
                        st.markdown("<span style='color:#e74c3c; font-size:0.75rem; font-weight:700;'>WHO Severe Dengue criterion triggered</span>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button(
                "Save Report Data",
                width='stretch',
                type="primary")

        if submitted:
            urine_enabled = form_urine_vol > 0 and form_urine_weight > 0
            _save_w = form_urine_weight if urine_enabled else 0.0
            _save_v = form_urine_vol    if urine_enabled else 0.0
            _save_t = form_urine_time   if urine_enabled else 1.0
            final_urine_output = calculate_urine_rate(_save_v, _save_t, _save_w) if urine_enabled else 0.0
            st.session_state.reports[rep_key] = {
                
                'ns1': st.session_state.get('ns1_result', 'Not Done'),
                'igm': st.session_state.get('igm_widget', 'Not Done'),
                'igg': st.session_state.get('igg_widget', 'Not Done'),
                
                'date':         form_date,
                'time_hour':    int(form_hour),
                'time_minute':  int(form_minute),
                'time_ampm':    form_ampm,
                'platelets':    int(form_platelets),
                'hb':           float(form_hb),
                'rbc':          float(form_rbc),
                'hct':          float(form_hct),
                'sys':          int(form_sys),
                'dia':          int(form_dia),
                'hr':           int(form_hr),
                'urine_output': final_urine_output,
                'urine_vol':    _save_v,
                'urine_time':   _save_t,
                'urine_weight': _save_w,
                'who':          form_who,
                'symptoms':     form_symptoms,
                'ocr_weight':   data.get('ocr_weight', 0.0),
                'ocr_height':   data.get('ocr_height', 0.0),
                'ocr_bmi':      data.get('ocr_bmi', 0.0),
                # ── CBC Differential ──────────────────────────
                'wbc':             int(form_wbc)   if 'form_wbc'   in dir() else int(data.get('wbc', 0)),
                'neutrophil_pct':  float(form_neu) if 'form_neu'   in dir() else float(data.get('neutrophil_pct', 0.0)),
                'lymphocyte_pct':  float(form_lym) if 'form_lym'   in dir() else float(data.get('lymphocyte_pct', 0.0)),
                'mpv':             float(form_mpv) if 'form_mpv'   in dir() else float(data.get('mpv', 0.0)),
                # ── Extended Vitals ───────────────────────────
                'temperature':     float(form_temp) if 'form_temp' in dir() else float(data.get('temperature', 0.0)),
                'spo2':            int(form_spo2)   if 'form_spo2' in dir() else int(data.get('spo2', 0)),
                'rr':              int(form_rr)     if 'form_rr'   in dir() else int(data.get('rr', 0)),
                'gcs':             int(form_gcs)    if 'form_gcs'  in dir() else int(data.get('gcs', 15)),
                'crt':             ['Not assessed','<2 sec (Normal)','2-3 sec (Borderline)','>3 sec (Prolonged)'].index(form_crt) if 'form_crt' in dir() else 0,
                # ── LFT ──────────────────────────────────────
                'ast':             int(form_ast)    if 'form_ast'   in dir() else int(data.get('ast', 0)),
                'alt':             int(form_alt)    if 'form_alt'   in dir() else int(data.get('alt', 0)),
                'albumin':         float(form_alb)  if 'form_alb'   in dir() else float(data.get('albumin', 0.0)),
                'bilirubin_total': float(form_tbili)if 'form_tbili' in dir() else float(data.get('bilirubin_total', 0.0)),
                'bilirubin_direct':float(form_dbili)if 'form_dbili' in dir() else float(data.get('bilirubin_direct', 0.0)),
                # ── Coagulation ───────────────────────────────
                'pt':              float(form_pt)   if 'form_pt'    in dir() else float(data.get('pt', 0.0)),
                'inr':             float(form_inr)  if 'form_inr'   in dir() else float(data.get('inr', 0.0)),
                'aptt':            float(form_aptt) if 'form_aptt'  in dir() else float(data.get('aptt', 0.0)),
                'd_dimer':         int(form_ddimer) if 'form_ddimer' in dir() else int(data.get('d_dimer', 0)),
                # ── Renal ─────────────────────────────────────
                'creatinine':      float(form_creat)if 'form_creat' in dir() else float(data.get('creatinine', 0.0)),
                'bun':             float(form_bun)  if 'form_bun'   in dir() else float(data.get('bun', 0.0)),
                'sodium':          float(form_na)   if 'form_na'    in dir() else float(data.get('sodium', 0.0)),
                'potassium':       float(form_k)    if 'form_k'     in dir() else float(data.get('potassium', 0.0)),
                'bicarbonate':     float(form_hco3) if 'form_hco3'  in dir() else float(data.get('bicarbonate', 0.0)),
                # ── Imaging ───────────────────────────────────
                'pleural_effusion':    form_pleural  if 'form_pleural'  in dir() else data.get('pleural_effusion', False),
                'gallbladder_wall_mm': float(form_gb_wall) if 'form_gb_wall' in dir() else float(data.get('gallbladder_wall_mm', 0.0)),
                'ascites_grade':       int(form_ascites)   if 'form_ascites'  in dir() else int(data.get('ascites_grade', 0)),
            }
            # Demographics now live outside the form — commit from sidebar widgets on save
            st.session_state.patient_name = st.session_state.get('patient_name_widget', '')
            st.session_state.patient_sex  = st.session_state.get('patient_sex_widget', 'Male')
            st.session_state.patient_age  = int(st.session_state.get('patient_age_widget', 25))
            st.success(
                f"Report {rep_key} saved — "
                f"PLT: {int(form_platelets):,}  |  "
                f"BP: {int(form_sys)}/{int(form_dia)}  |  "
                f"HR: {int(form_hr)}  |  "
                f"Hct: {float(form_hct):.1f}%  |  "
                f"Click RUN ANALYSIS to update the dashboard.")

        st.divider()

        # ── Discharge Assessment ──────────────────────────
        st.markdown('<div class="section-header">Discharge Assessment</div>', unsafe_allow_html=True)
        st.markdown("""<div class="discharge-info-panel">
            <h5>How to Use</h5>
            <p>Toggle ON when evaluating for discharge (typically Day 7-9).<br><br>
            Two criteria require manual input:<br>
            <b>1. Fever-free hours:</b> Hours since temperature normalised (target >=48 hrs)<br>
            <b>2. Tolerating orals:</b> Patient keeps down food/fluids without vomiting<br><br>
            All 7 WHO 2009 discharge criteria are evaluated in the Analysis panel.</p>
        </div>""", unsafe_allow_html=True)

        discharge_enabled = st.toggle(
            "Assess Discharge Readiness",
            value=st.session_state.discharge_enabled,
            key="discharge_toggle")
        if discharge_enabled != st.session_state.discharge_enabled:
            st.session_state.discharge_enabled = discharge_enabled
            st.rerun()

        if discharge_enabled:
            st.session_state.discharge_fever_free = st.number_input(
                "Hours afebrile (fever-free)", 0.0, 200.0,
                st.session_state.discharge_fever_free, step=1.0)
            st.session_state.discharge_tolerating_orals = st.checkbox(
                "Patient tolerating oral fluids",
                value=st.session_state.discharge_tolerating_orals)
        else:
            st.markdown("""<div class="module-disabled">
                <p>Toggle on when evaluating patient for discharge.</p>
            </div>""", unsafe_allow_html=True)
# ── Report Navigation (bottom convenience) ───────────────────
        st.markdown('<div class="section-header">Navigate Reports</div>', unsafe_allow_html=True)
        _curr_idx = REPORT_KEYS.index(rep_key)

        def _go_prev():
            _idx = REPORT_KEYS.index(st.session_state.get('report_selector', REPORT_KEYS[0]))
            if _idx > 0:
                st.session_state['report_selector'] = REPORT_KEYS[_idx - 1]

        def _go_next():
            _idx = REPORT_KEYS.index(st.session_state.get('report_selector', REPORT_KEYS[0]))
            if _idx < len(REPORT_KEYS) - 1:
                st.session_state['report_selector'] = REPORT_KEYS[_idx + 1]

        _nav1, _nav2 = st.columns(2)
        with _nav1:
            st.button("◀ Prev", width='stretch', key="prev_report_btn",
                        disabled=(_curr_idx == 0), on_click=_go_prev)   
        with _nav2:
            st.button("Next ▶", width='stretch', key="next_report_btn",
                        disabled=(_curr_idx == len(REPORT_KEYS) - 1), on_click=_go_next)
        
        st.markdown("**Clinician Notes**")
        st.session_state.clinician_notes = st.text_area(
            "Observations", st.session_state.clinician_notes,
            height=80, label_visibility="collapsed")

        if st.button("RUN ANALYSIS", type="primary", width='stretch'):
            st.session_state.analysis_run = True
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-collapse-btn-st">', unsafe_allow_html=True)
        if st.button("Collapse Sidebar", width='stretch', key="collapse_sidebar_btn"):
            st.session_state.sidebar_collapsed = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════════════
    #  MAIN CONTENT AREA
    # ══════════════════════════════════════════════════════
    age  = st.session_state.patient_age
    sex  = st.session_state.patient_sex
    name = st.session_state.patient_name
    any_valid = any(_report_has_meaningful_data(r) for r in st.session_state.reports.values())
    
    if st.session_state.get('sidebar_collapsed', False):
        rcol1, rcol2 = st.columns([0.18, 0.82])
        with rcol1:
            if st.button("Show Sidebar", key="restore_sidebar_btn"):
                st.session_state.sidebar_collapsed = False
                st.rerun()
        with rcol2:
            st.markdown(
                "<div class='show-sidebar-bar'>"
                "<p>Sidebar hidden — click <b>Show Sidebar</b> to return to data entry</p>"
                "</div>",
                unsafe_allow_html=True)

    if not any_valid:
        st.markdown("""<div class='intro-card'><h2>Welcome to the Dengue CDSS</h2>
            <p>AI-powered dengue severity stratification — WHO 2009 framework — Sensitivity 100%, Specificity 99.2%</p>
            <ul>
                <li><b>Step 1 — Enter data:</b> Upload CBC reports or enter values manually for Reports A through E</li>
                <li><b>Step 2 — Save:</b> Click "Save Report Data" after entering each report's values</li>
                <li><b>Step 3 — Analyse:</b> Click RUN ANALYSIS to generate the full clinical dashboard</li>
                <li><b>Re-verify:</b> Edit any report and save again, then re-run analysis to refresh results</li>
            </ul></div>""", unsafe_allow_html=True)

        qc1, qc2, qc3, qc4 = st.columns(4)
        ref_cards = [
            ("#3498db", "Platelet Thresholds",  ">150k Normal\n100-150k Mild\n50-100k Moderate\n<50k Severe"),
            ("#f39c12", "Shock Index Guide",     "<0.6 Normal\n0.6-0.9 Borderline\n>0.9 Elevated\n>1.2 Severe shock"),
            ("#2ecc71", "Urine Output Guide",    "<0.5 Oliguria\n0.5-1.0 Borderline\n1.0-2.0 Normal\n>4.0 Polyuria"),
            ("#9b59b6", "WHO Group Triage",      "Group A Outpatient\nGroup B Ward\nGroup C ICU\nBased on warning signs"),
        ]
        for col, (color, title, content) in zip([qc1, qc2, qc3, qc4], ref_cards):
            with col:
                lines_html = ''.join(
                    f"<p style='margin:2px 0; color:#a0aec0; font-size:0.78rem;'>{l}</p>"
                    for l in content.split('\n'))
                st.markdown(
                    f"<div style='background:#161b24; border:1px solid #2d4a6e; "
                    f"border-left:4px solid {color}; border-radius:8px; padding:16px;'>"
                    f"<h4 style='color:{color}; margin:0 0 8px 0; font-size:.85rem;'>{title}</h4>"
                    f"{lines_html}</div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        st.info("Start by uploading reports in the sidebar or selecting Report A and entering values manually.")
        
    elif not st.session_state.analysis_run:
        preview_rows = []
        for char in REPORT_KEYS:
            r = st.session_state.reports[char]
            if _report_has_meaningful_data(r):
                h24 = int(r.get('time_hour', 9))
                _ampm = r.get('time_ampm', 'AM')
                if _ampm == "PM" and h24 != 12:
                    h24 += 12
                elif _ampm == "AM" and h24 == 12:
                    h24 = 0
                h24 = max(0, min(h24, 23))   # clamp: prevents datetime.time ValueError
                try:
                    dt = datetime.datetime.combine(
                        r.get('date', datetime.date.today()),
                        datetime.time(h24, int(r.get('time_minute', 0))))
                except Exception:
                    dt = datetime.datetime.now()
                mv = calculate_map(r.get('sys', 120), r.get('dia', 80))
                si = r.get('hr', 72) / r.get('sys', 120) if r.get('sys', 0) > 0 else 0
                preview_rows.append({
                    "Report":    f"Report {char}",
                    "Date/Time": dt.strftime("%d-%b %I:%M %p"),
                    "Platelets": f"{r.get('platelets', 0):,}",
                    "Hb":        f"{r.get('hb', 0):.1f}",
                    "Hct":       f"{r.get('hct', 0):.1f}%",
                    "BP":        f"{r.get('sys', 0)}/{r.get('dia', 0)}",
                    "HR":        str(r.get('hr', 0)),
                    "MAP":       f"{mv:.1f}",
                    "SI":        f"{si:.2f}",
                    "Urine":     f"{r.get('urine_output', 0):.2f}" if r.get('urine_output', 0) > 0 else "--",
                    "WHO Signs": str(len(r.get('who', []))),
                })
        
        # ── Preview: compute module status from current saved data ──────────
        _prev_sorted = []
        for char in REPORT_KEYS:
            r = st.session_state.reports[char]
            if _report_has_meaningful_data(r):
                h24 = r['time_hour']
                if r['time_ampm'] == "PM" and h24 != 12: h24 += 12
                elif r['time_ampm'] == "AM" and h24 == 12: h24 = 0
                try:
                    _dt = datetime.datetime.combine(r['date'], datetime.time(max(0,min(h24,23)), r['time_minute']))
                except Exception:
                    _dt = datetime.datetime.now()
                _prev_sorted.append({**r, 'datetime': _dt, 'Label': f"Report {char}",
                                     'sys_bp': r['sys'], 'dia_bp': r['dia'],
                                     'shock_index': r['hr']/r['sys'] if r['sys']>0 else 0,
                                     'who_signs': r.get('who',[]), 'pp': r['sys']-r['dia']})
        _prev_sorted.sort(key=lambda x: x['datetime'])

        _mod_status = compute_module_status(
            _prev_sorted, st.session_state.patient_age,
            st.session_state.patient_sex, enable_metrics)

        _active_n  = sum(1 for m in _mod_status if m['status'] == 'active')
        _partial_n = sum(1 for m in _mod_status if m['status'] == 'partial')
        _total_n   = len(_mod_status)
        _partial_span = (f"&nbsp;·&nbsp;<span style='color:#f39c12; font-weight:700;'>{_partial_n} partial</span>" if _partial_n else "")
        _inactive_n = _total_n - _active_n - _partial_n
        st.markdown(
            f"<div class='preview-banner'>"
            f"<h4>Report Entry Preview</h4>"
            f"<p>Verify all data below. When ready, click <b>RUN ANALYSIS</b> in the sidebar."
            f"&nbsp;&nbsp;"
            f"<span style='color:#2ecc71; font-weight:700;'>{_active_n} modules active</span>"
            + _partial_span +
            f"&nbsp;·&nbsp;<span style='color:#8b92a8;'>{_inactive_n} inactive</span>"
            f"</p></div>",
            unsafe_allow_html=True)

        if preview_rows:
            df_prev = pd.DataFrame(preview_rows).set_index("Report")
            st.dataframe(df_prev, width='stretch')
            st.caption(f"{len(preview_rows)} report(s) saved. Click RUN ANALYSIS in the sidebar when ready.")

        # ── Module Activation Preview (collapsible) ──────────────────────────
        with st.expander(
                f"Module Activation Preview — {_active_n}/{_total_n} active  "
                f"({'click to expand' if _active_n < _total_n else 'all systems active'})",
                expanded=(_active_n < _total_n)):
            st.caption(
                "Shows which analysis modules will run when you click RUN ANALYSIS, "
                "and exactly what data each inactive module needs to activate.")
            _col_a, _col_b = st.columns(2)
            for i, m in enumerate(_mod_status):
                _target_col = _col_a if i % 2 == 0 else _col_b
                with _target_col:
                    _bg = ('rgba(46,204,113,0.07)' if m['status'] == 'active'
                           else 'rgba(243,156,18,0.07)' if m['status'] == 'partial'
                           else 'rgba(0,0,0,0.15)')
                    _border = m['color']
                    st.markdown(
                        f"<div style='background:{_bg}; border:1px solid {_border}; "
                        f"border-left:3px solid {_border}; border-radius:6px; "
                        f"padding:8px 12px; margin-bottom:6px;'>"
                        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                        f"<span style='color:#e2e8f0; font-size:0.8rem; font-weight:600;'>{m['module']}</span>"
                        f"<span style='color:{_border}; font-size:0.75rem; font-weight:700; "
                        f"text-transform:uppercase;'>{m['icon']} {m['status'].upper()}</span>"
                        f"</div>"
                        f"<div style='color:#8b92a8; font-size:0.72rem; margin-top:3px;'>{m['reason']}</div>"
                        f"</div>",
                        unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    #  FULL ANALYSIS DASHBOARD
    # ════════════════════════════════════════════════════════
    if st.session_state.analysis_run:
        valid_reports = []
        current_bmi   = calculate_bmi(final_w_kg, final_h_cm) if enable_metrics else 0

        for char, r in st.session_state.reports.items():
            if _report_has_meaningful_data(r):
                h24 = r['time_hour']
                if r['time_ampm'] == "PM" and h24 != 12: h24 += 12
                elif r['time_ampm'] == "AM" and h24 == 12: h24 = 0
                dt    = datetime.datetime.combine(r['date'], datetime.time(h24, r['time_minute']))
                map_v = calculate_map(r['sys'], r['dia'])
                si    = r['hr'] / r['sys'] if r['sys'] > 0 else 0
                pp    = r['sys'] - r['dia']
                valid_reports.append({
                    'Label':       f"Report {char}",
                    'datetime':    dt,
                    'platelets':   r['platelets'],
                    'sys_bp':      r['sys'],
                    'dia_bp':      r['dia'],
                    'hr':          r['hr'],
                    'map':         map_v,
                    'shock_index': si,
                    'hct':         r['hct'],
                    'hb':          r['hb'],
                    'rbc':         r['rbc'],
                    'who_signs':   r['who'],
                    'symptoms':    r.get('symptoms', []),
                    'pp':          pp,
                    'urine_output':r.get('urine_output', 0.0),
                    'age':         age,
                    'sex':         sex,
                    'bmi':         current_bmi,
                    'weight':      final_w_kg,
                    # ── Extended fields (NEW) ──────────────
                    'wbc':              r.get('wbc', 0),
                    'neutrophil_pct':   r.get('neutrophil_pct', 0.0),
                    'ast':              r.get('ast', 0),
                    'alt':              r.get('alt', 0),
                    'albumin':          r.get('albumin', 0.0),
                    'inr':              r.get('inr', 0.0),
                    'aptt':             r.get('aptt', 0.0),
                    'd_dimer':          r.get('d_dimer', 0),
                    'creatinine':       r.get('creatinine', 0.0),
                    'bun':              r.get('bun', 0.0),
                    'sodium':           r.get('sodium', 0.0),
                    'potassium':        r.get('potassium', 0.0),
                    'temperature':      r.get('temperature', 0.0),
                    'spo2':             r.get('spo2', 0),
                    'rr':               r.get('rr', 0),
                    'gcs':              r.get('gcs', 15),
                    'pleural_effusion': r.get('pleural_effusion', False),
                    'gallbladder_wall_mm': r.get('gallbladder_wall_mm', 0.0),
                    'ascites_grade':    r.get('ascites_grade', 0),
                })

        if valid_reports:
            sorted_rep = sorted(valid_reports, key=lambda x: x['datetime'])
            latest     = sorted_rep[-1]
            fluid_rate = calculate_fluid(final_w_kg) if (enable_metrics and final_w_kg > 0) else 0

            # ── Module-level guards: defined ONCE here, used in ALL downstream sections ──
            # This prevents NameError when any section references these before their
            # section-local redefinition (e.g. section K uses has_bleeding from section G)
            has_bleeding = 1 if (
                "Bleeding"         in latest.get('symptoms',  []) or
                "Mucosal Bleeding" in latest.get('who_signs', [])
            ) else 0
            risk_buf_pdf   = None
            adj_risk_prob  = 0.0
            adj_ci_lower   = 0.0
            adj_ci_upper   = 0.0
            pred_val       = 0
            velocity       = 0
            plot_buf_pdf   = None
            
            if name:
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,#1a2035,#12192b);"
                    f"border:1px solid #2d4a6e; border-left:4px solid #5dade2;"
                    f"border-radius:8px; padding:10px 16px; margin-bottom:16px;'>"
                    f"<span style='color:#5dade2; font-weight:700; font-size:0.95rem;'>"
                    f"Patient: {name}</span>"
                    f"<span style='color:#8b92a8; font-size:0.85rem; margin-left:16px;'>"
                    f"Age: {age}y | Sex: {sex}</span></div>",
                    unsafe_allow_html=True)
                # ── PHASE BANNER (NEW) ────────────────────────
            illness_day_dash = st.session_state.get('patient_illness_day', 0)
            if illness_day_dash > 0:
                phase_info_dash = get_dengue_phase(illness_day_dash)
                st.markdown(
                    f"<div style='background:linear-gradient(90deg,rgba(0,0,0,0.3),rgba(0,0,0,0.1)); "
                    f"border:1px solid {phase_info_dash['color']}; border-left:6px solid {phase_info_dash['color']}; "
                    f"border-radius:10px; padding:14px 20px; margin-bottom:16px; display:flex; align-items:center; gap:16px;'>"
                    f"<div style='font-size:2rem; font-weight:900; color:{phase_info_dash['color']};'>Day {illness_day_dash}</div>"
                    f"<div><div style='color:{phase_info_dash['color']}; font-weight:800; font-size:1rem; text-transform:uppercase; letter-spacing:0.06em;'>"
                    f"{phase_info_dash['phase']}</div>"
                    f"<div style='color:#a0aec0; font-size:0.82rem; margin-top:2px;'>{phase_info_dash['action']}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True)

            # ── A. Continuous Severity Score ──────────────
            st.subheader("Continuous Severity Score")
            sev_score, sev_direction, sev_components = compute_severity_score(sorted_rep)

            sev_col1, sev_col2 = st.columns([2, 3])
            with sev_col1:
                if sev_score >= 70:    sev_color = "#e74c3c"; sev_label = "CRITICAL"
                elif sev_score >= 45:  sev_color = "#f39c12"; sev_label = "ELEVATED"
                elif sev_score >= 20:  sev_color = "#f39c12"; sev_label = "MODERATE"
                else:                  sev_color = "#2ecc71"; sev_label = "LOW"
                dir_color = {"Improving": "#2ecc71", "Stable": "#3498db",
                             "Deteriorating": "#f39c12", "Rapidly Deteriorating": "#e74c3c"}.get(sev_direction, "#8b92a8")
                st.markdown(f"""<div class="severity-panel">
                    <h4>Severity Score</h4>
                    <div style='font-size:3rem; font-weight:900; color:{sev_color}; margin:0 0 4px 0;'>{sev_score}<span style='font-size:1.2rem; color:#8b92a8;'>/100</span></div>
                    <div style='font-size:0.85rem; color:{sev_color}; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;'>{sev_label}</div>
                    <div style='font-size:0.9rem; color:{dir_color}; margin-top:8px; font-weight:600;'>Direction: {sev_direction}</div>
                </div>""", unsafe_allow_html=True)
            with sev_col2:
                st.markdown("**Score Components**")
                comp_max = {'Haematological': 40, 'Haemodynamic': 30, 'Trend Velocity': 20, 'Urine Output': 10}
                for comp, val in sev_components.items():
                    max_val = comp_max.get(comp, 10)
                    pct = val / max_val if max_val > 0 else 0
                    bar_color = "#e74c3c" if pct > 0.7 else ("#f39c12" if pct > 0.4 else "#2ecc71")
                    bar_width = int(pct * 100)
                    st.markdown(
                        f"<div style='margin:6px 0;'>"
                        f"<div style='display:flex; justify-content:space-between; margin-bottom:3px;'>"
                        f"<span style='color:#a0aec0; font-size:0.8rem;'>{comp}</span>"
                        f"<span style='color:{bar_color}; font-size:0.8rem; font-weight:700;'>{val}/{max_val}</span>"
                        f"</div>"
                        f"<div style='background:#2d3748; border-radius:4px; height:6px;'>"
                        f"<div style='background:{bar_color}; width:{bar_width}%; height:6px; border-radius:4px;'></div>"
                        f"</div></div>",
                        unsafe_allow_html=True)
                st.caption("Haematological 40% | Haemodynamic 30% | Trend Velocity 20% | Urine Output 10%")
            st.divider()

            # ── B. Serial Alerts (with suppression) ───────
            serial_alerts = check_serial_alerts(sorted_rep)
            if serial_alerts:
                st.subheader("Serial Alert System")
                top_alert, remaining_alerts = get_top_alert(serial_alerts)
                if top_alert:
                    sev, msg, rec = top_alert
                    st.markdown(
                        f'<div class="alert-critical"><strong>[PRIORITY ALERT] {msg}</strong><br>'
                        f'<span style="font-size:0.82rem; opacity:0.85;">Recommended action: {rec}</span></div>',
                        unsafe_allow_html=True)
                if remaining_alerts:
                    with st.expander(f"Additional alerts ({len(remaining_alerts)})"):
                        for sev, msg, rec in remaining_alerts:
                            css_class = "alert-critical" if sev == "CRITICAL" else "alert-warning"
                            st.markdown(
                                f'<div class="{css_class}">{msg}<br>'
                                f'<span style="font-size:0.82rem; opacity:0.85;">{rec}</span></div>',
                                unsafe_allow_html=True)
                st.divider()
            elif len(sorted_rep) >= 2:
                st.markdown(
                    '<div class="alert-ok"><strong>Serial Alert System:</strong> '
                    'No critical patterns detected across reports.</div>',
                    unsafe_allow_html=True)
                st.divider()

            # ── C. WHO Classification ─────────────────────
            st.subheader("WHO 2009 Dengue Classification")
            hct_prev = sorted_rep[-2]['hct'] if len(sorted_rep) >= 2 else None
            who_class, who_group, who_criteria, who_desc, who_actions = classify_who_dengue(
                platelets=latest['platelets'],
                who_signs=latest['who_signs'],
                symptoms=latest['symptoms'],
                shock_index=latest['shock_index'],
                hct=latest['hct'],
                hct_prev=hct_prev,
                map_val=latest['map'],
                # ── Extended organ params (NEW) ────────
                ast=latest.get('ast', 0),
                inr=latest.get('inr', 0.0),
                pleural_effusion=latest.get('pleural_effusion', False),
                ascites_grade=latest.get('ascites_grade', 0),
                spo2=latest.get('spo2', 0),
                gcs=latest.get('gcs', 15))
            wc1, wc2 = st.columns([2, 1])
            with wc1:
                st.markdown(f'<div class="who-badge-{who_group}">WHO Classification: {who_class}</div>', unsafe_allow_html=True)
                st.markdown(f"<p style='color:#a0aec0; margin-top:8px; font-size:0.9rem;'><b>Management Group {who_group}:</b> {who_desc}</p>", unsafe_allow_html=True)
                if who_criteria:
                    st.markdown("**Criteria met:**")
                    for c in who_criteria: st.markdown(f"- {c}")
            with wc2:
                st.markdown("**Recommended Actions:**")
                for action in who_actions: st.markdown(f"- {action}")
            st.divider()

            # ── C2. Extended Organ Assessment (NEW) ──────
            has_lft  = latest.get('ast', 0) > 0 or latest.get('alt', 0) > 0 or latest.get('albumin', 0) > 0
            has_coag = latest.get('inr', 0) > 0 or latest.get('d_dimer', 0) > 0
            has_renal= latest.get('creatinine', 0) > 0

            if has_lft or has_coag or has_renal:
                st.subheader("Extended Organ Assessment")

                # ── Hepatic Panel ─────────────────────────
                if has_lft:
                    st.markdown("**Hepatic Function**")
                    hep1, hep2, hep3, hep4 = st.columns(4)
                    ast_v = latest.get('ast', 0)
                    alt_v = latest.get('alt', 0)
                    alb_v = latest.get('albumin', 0.0)
                    with hep1:
                        ast_color = "#e74c3c" if ast_v >= 1000 else ("#f39c12" if ast_v >= 200 else "#2ecc71")
                        st.metric("AST", f"{ast_v:,} IU/L" if ast_v > 0 else "N/R",
                                  delta="⚠ WHO criterion" if ast_v >= 1000 else ("↑ Elevated" if ast_v >= 80 else "Normal"),
                                  delta_color="inverse" if ast_v >= 80 else "normal")
                    with hep2:
                        st.metric("ALT", f"{alt_v:,} IU/L" if alt_v > 0 else "N/R",
                                  delta="↑ Elevated" if alt_v >= 80 else "Normal",
                                  delta_color="inverse" if alt_v >= 80 else "normal")
                    with hep3:
                        if ast_v > 0 and alt_v > 0:
                            ratio_v = ast_v / alt_v
                            ratio_color = "#2ecc71" if ratio_v > 2.0 else "#f39c12"
                            st.metric("AST:ALT Ratio", f"{ratio_v:.1f}",
                                      delta="Dengue pattern" if ratio_v > 2.0 else "Atypical",
                                      delta_color="normal" if ratio_v > 2.0 else "inverse")
                    with hep4:
                        st.metric("Albumin", f"{alb_v:.1f} g/dL" if alb_v > 0 else "N/R",
                                  delta="Low" if 0 < alb_v < 3.5 else "Normal",
                                  delta_color="inverse" if 0 < alb_v < 3.5 else "normal")
                    if ast_v >= 1000:
                        st.markdown('<div class="alert-critical"><b>WHO Severe Dengue — Hepatic criterion: AST ≥1,000 IU/L</b><br>'
                                    '<span style="font-size:0.82rem;">Hepatic failure risk. Hepatology consult. Monitor PT/INR daily.</span></div>',
                                    unsafe_allow_html=True)

                # ── Coagulation Panel ─────────────────────
                if has_coag:
                    st.markdown("**Coagulation Status**")
                    inr_v    = latest.get('inr', 0.0)
                    aptt_v   = latest.get('aptt', 0.0)
                    ddimer_v = latest.get('d_dimer', 0)
                    brs_score, brs_details, brs_rec = compute_bleeding_risk_score(
                        latest, is_secondary_dengue=st.session_state.get('is_secondary_dengue', False))
                    cg1, cg2, cg3, cg4 = st.columns(4)
                    with cg1:
                        st.metric("INR", f"{inr_v:.1f}" if inr_v > 0 else "N/R",
                                  delta="Critical" if inr_v >= 2.0 else ("Elevated" if inr_v >= 1.5 else "Normal"),
                                  delta_color="inverse" if inr_v >= 1.5 else "normal")
                    with cg2:
                        st.metric("aPTT", f"{aptt_v:.0f}s" if aptt_v > 0 else "N/R",
                                  delta="Prolonged" if aptt_v > 35 else "Normal",
                                  delta_color="inverse" if aptt_v > 35 else "normal")
                    with cg3:
                        st.metric("D-dimer", f"{ddimer_v:,}" if ddimer_v > 0 else "N/R",
                                  delta="High" if ddimer_v >= 1000 else "Normal",
                                  delta_color="inverse" if ddimer_v >= 1000 else "normal")
                    with cg4:
                        brs_color = "#e74c3c" if brs_score >= 6 else ("#f39c12" if brs_score >= 3 else "#2ecc71")
                        st.markdown(f"<div style='text-align:center; padding:8px;'>"
                                    f"<div style='color:#a0aec0; font-size:0.75rem; text-transform:uppercase;'>Bleeding Risk Score</div>"
                                    f"<div style='color:{brs_color}; font-size:1.8rem; font-weight:800;'>{brs_score}</div>"
                                    f"<div style='color:{brs_color}; font-size:0.72rem;'>{'HIGH' if brs_score>=6 else 'MODERATE' if brs_score>=3 else 'LOW'}</div>"
                                    f"</div>", unsafe_allow_html=True)
                    if brs_score >= 3:
                        st.markdown(f'<div class="{"alert-critical" if brs_score>=6 else "alert-warning"}">'
                                    f'<b>Bleeding Risk Score {brs_score}:</b> {brs_rec}</div>',
                                    unsafe_allow_html=True)

                # ── Renal Panel ───────────────────────────
                if has_renal:
                    st.markdown("**Renal Function**")
                    cr_v   = latest.get('creatinine', 0.0)
                    bun_v  = latest.get('bun', 0.0)
                    na_v   = latest.get('sodium', 0.0)
                    k_v    = latest.get('potassium', 0.0)
                    egfr_v = calculate_egfr_ckdepi(cr_v, age, sex) if cr_v > 0 else 0
                    ren1, ren2, ren3, ren4 = st.columns(4)
                    with ren1:
                        st.metric("Creatinine", f"{cr_v:.2f} mg/dL" if cr_v > 0 else "N/R",
                                  delta="Elevated" if cr_v > 1.2 else "Normal",
                                  delta_color="inverse" if cr_v > 1.2 else "normal")
                    with ren2:
                        st.metric("eGFR (CKD-EPI)", f"{egfr_v}" if egfr_v > 0 else "N/R",
                                  delta="Impaired" if 0 < egfr_v < 60 else "Normal",
                                  delta_color="inverse" if 0 < egfr_v < 60 else "normal")
                    with ren3:
                        bun_cr = bun_v / cr_v if cr_v > 0 else 0
                        st.metric("BUN:Cr Ratio", f"{bun_cr:.1f}" if bun_cr > 0 else "N/R",
                                  delta="Prerenal" if bun_cr > 20 else "Normal",
                                  delta_color="inverse" if bun_cr > 20 else "normal")
                    with ren4:
                        if na_v > 0:
                            st.metric("Sodium", f"{na_v:.0f} mEq/L",
                                      delta="Low" if na_v < 136 else ("High" if na_v > 145 else "Normal"),
                                      delta_color="inverse" if na_v < 136 or na_v > 145 else "normal")
                    # ── Serial AKI detection ──────────────
                    if len(sorted_rep) >= 2:
                        baseline_cr = sorted_rep[0].get('creatinine', 0)
                        days_el     = (latest['datetime'] - sorted_rep[0]['datetime']).total_seconds() / 86400
                        is_aki, aki_stage, aki_detail = detect_aki(cr_v, baseline_cr, days_el)
                        if is_aki:
                            st.markdown(f'<div class="alert-warning"><b>AKI Detected — {aki_stage}</b>: {aki_detail}<br>'
                                        f'<span style="font-size:0.82rem;">Reduce nephrotoxic agents. Strict fluid balance. Nephrology consult if eGFR <30.</span></div>',
                                        unsafe_allow_html=True)

                st.divider()

            # ── C3. Plasma Leakage Composite Score (NEW) ──
            hct_baseline_pl = sorted_rep[0].get('hct', 0) if sorted_rep else 0
            imaging_data    = {
                'pleural_effusion':    latest.get('pleural_effusion', False),
                'gallbladder_wall_mm': latest.get('gallbladder_wall_mm', 0.0),
                'ascites_grade':       latest.get('ascites_grade', 0),
            }
            has_imaging = any(imaging_data.values())
            pl_score, pl_factors, pl_label = compute_plasma_leakage_score(
                latest, hct_baseline=hct_baseline_pl,
                imaging=imaging_data if has_imaging else None)

            if pl_score > 0 or has_imaging:
                st.subheader("Plasma Leakage Assessment")
                pl_color = "#e74c3c" if pl_score >= 0.70 else ("#f39c12" if pl_score >= 0.30 else "#2ecc71")
                plc1, plc2 = st.columns([1, 3])
                with plc1:
                    st.markdown(
                        f"<div style='background:rgba(0,0,0,0.2); border:2px solid {pl_color}; "
                        f"border-radius:12px; padding:20px; text-align:center;'>"
                        f"<div style='color:#a0aec0; font-size:0.72rem; text-transform:uppercase; margin-bottom:6px;'>Leakage Score</div>"
                        f"<div style='color:{pl_color}; font-size:2.8rem; font-weight:900;'>{pl_score:.0%}</div>"
                        f"<div style='color:{pl_color}; font-size:0.72rem; margin-top:4px; font-weight:700;'>{pl_label.split('—')[0].strip()}</div>"
                        f"</div>", unsafe_allow_html=True)
                with plc2:
                    st.markdown(f"**{pl_label}**")
                    for f in pl_factors:
                        marker_color = "#e74c3c" if "definitive" in f.lower() or "confirmed" in f.lower() else "#f39c12"
                        st.markdown(f"<p style='color:{marker_color}; font-size:0.82rem; margin:2px 0;'>• {f}</p>",
                                    unsafe_allow_html=True)
                    if not pl_factors:
                        st.markdown("<p style='color:#718096; font-size:0.82rem;'>No leakage markers detected. Enter LFT/imaging data for comprehensive assessment.</p>",
                                    unsafe_allow_html=True)
                st.divider()

            # ── C4. Fluid Overload Risk (NEW — recovery phase) ──
            fors_score, fors_factors, fors_high = compute_fors(sorted_rep)
            if fors_score >= 2 or (illness_day_dash > 7 and len(sorted_rep) >= 2):
                st.subheader("Fluid Overload Risk  (Post-Resuscitation)")
                fors_color = "#e74c3c" if fors_high else ("#f39c12" if fors_score >= 2 else "#2ecc71")
                if fors_high:
                    st.markdown(
                        f'<div class="alert-critical"><b>Fluid Overload Risk Score {fors_score} — HIGH RISK</b><br>'
                        f'{"<br>".join(f"• {f}" for f in fors_factors)}<br>'
                        f'<span style="font-size:0.82rem; opacity:0.85;">Action: Reduce IV rate, consider furosemide, strict daily weight, escalate monitoring</span></div>',
                        unsafe_allow_html=True)
                elif fors_score >= 2:
                    st.markdown(
                        f'<div class="alert-warning"><b>Fluid Overload Risk Score {fors_score} — Monitor</b><br>'
                        f'{"<br>".join(f"• {f}" for f in fors_factors)}</div>',
                        unsafe_allow_html=True)
                st.divider()

            # ── D. Hemodynamics ───────────────────────────
            st.subheader("Hemodynamics & Treatment")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                if fluid_rate > 0:
                    if who_group == 'C':   st.metric("Emergency Bolus", "20 mL/kg/15min")
                    elif who_group == 'B': st.metric("IV Rate (Group B)", "5-7 mL/kg/hr")
                    else:                  st.metric("IV Maintenance", f"{int(fluid_rate)} mL/hr")
                else:
                    st.metric("IV Rate", "N/A")
            with c2:
                md = "Low" if latest['map'] < 65 else ("Normal" if latest['map'] < 90 else "High")
                st.metric("MAP", f"{latest['map']:.1f} mmHg", delta=md, delta_color="inverse" if latest['map'] < 65 else "normal")
            with c3:
                st.metric("Shock Index", f"{latest['shock_index']:.2f}", delta="High" if latest['shock_index'] > .9 else "Normal", delta_color="inverse" if latest['shock_index'] > .9 else "normal")
            with c4:
                pp_val = latest['sys_bp'] - latest['dia_bp']
                st.metric("Pulse Pressure", f"{pp_val} mmHg", delta="Narrow" if pp_val <= 20 else "Normal", delta_color="inverse" if pp_val <= 20 else "normal")
            with c5:
                urine_status, _ = interpret_urine_rate(latest['urine_output'])
                st.metric("Urine Output", f"{latest['urine_output']:.2f} mL/kg/hr" if latest['urine_output'] > 0 else "N/A", delta=f"{urine_status}")
            st.divider()

            # ── E. Trajectory Engine ──────────────────────
            if len(sorted_rep) >= 2:
                st.subheader("Trajectory Engine")
                st.caption("Velocity (units/day), acceleration (units/day2), and time-to-threshold countdowns based on current trajectory")

                traj_plt = compute_trajectory(sorted_rep, 'platelets')
                traj_hct = compute_trajectory(sorted_rep, 'hct')
                traj_si  = compute_trajectory(sorted_rep, 'shock_index')

                te1, te2, te3 = st.columns(3)
                with te1:
                    st.markdown("""<div class="trajectory-panel"><h5>Platelet Trajectory</h5>""", unsafe_allow_html=True)
                    if traj_plt:
                        vel = traj_plt['velocity']
                        acc = traj_plt['acceleration']
                        vel_color = "#e74c3c" if vel < -20000 else ("#f39c12" if vel < -5000 else "#2ecc71")
                        st.markdown(f"<div style='color:{vel_color}; font-size:0.9rem; font-weight:700;'>Velocity: {int(vel):+,}/day</div>", unsafe_allow_html=True)
                        if len(sorted_rep) >= 3:
                            acc_color = "#e74c3c" if acc < -10000 else ("#f39c12" if acc < 0 else "#2ecc71")
                            st.markdown(f"<div style='color:{acc_color}; font-size:0.8rem;'>Acceleration: {int(acc):+,}/day2</div>", unsafe_allow_html=True)
                        for label, hrs in (traj_plt.get('countdowns') or []):
                            if hrs is not None:
                                st.markdown(f'<div class="countdown-badge">{label}: {hrs:.1f}h</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="countdown-badge countdown-safe">{label}: threshold passed/stable</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with te2:
                    st.markdown("""<div class="trajectory-panel"><h5>Hematocrit Trajectory</h5>""", unsafe_allow_html=True)
                    if traj_hct:
                        vel = traj_hct['velocity']
                        vel_color = "#e74c3c" if vel > 2 else ("#f39c12" if vel > 0.5 else "#2ecc71")
                        st.markdown(f"<div style='color:{vel_color}; font-size:0.9rem; font-weight:700;'>Velocity: {vel:+.2f}%/day</div>", unsafe_allow_html=True)
                        if vel > 2:
                            st.markdown('<div class="countdown-badge">Rising Hct — monitor for plasma leakage</div>', unsafe_allow_html=True)
                        elif vel < -3:
                            st.markdown('<div class="countdown-badge countdown-safe">Falling Hct — possible fluid overload post-resuscitation</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with te3:
                    st.markdown("""<div class="trajectory-panel"><h5>Shock Index Trajectory</h5>""", unsafe_allow_html=True)
                    if traj_si:
                        vel = traj_si['velocity']
                        vel_color = "#e74c3c" if vel > 0.05 else ("#f39c12" if vel > 0 else "#2ecc71")
                        st.markdown(f"<div style='color:{vel_color}; font-size:0.9rem; font-weight:700;'>Velocity: {vel:+.3f}/day</div>", unsafe_allow_html=True)
                        for label, hrs in (traj_si.get('countdowns') or []):
                            if hrs is not None:
                                st.markdown(f'<div class="countdown-badge">{label}: {hrs:.1f}h</div>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                st.divider()

            # ── F. Personal Baseline Deltas ───────────────
            if len(sorted_rep) >= 2:
                st.subheader("Personal Baseline Delta Analysis")
                st.caption(f"All values compared against patient's own first report ({sorted_rep[0]['Label']}), not population reference ranges")
                deltas = compute_personal_deltas(sorted_rep)
                if deltas:
                    delta_cols = st.columns(len(deltas))
                    for col, d in zip(delta_cols, deltas):
                        with col:
                            flag_color = {"critical": "#e74c3c", "warning": "#f39c12", "normal": "#2ecc71"}.get(d['flag'], "#8b92a8")
                            # Pre-format values — can't use conditional inside f-string format spec
                            _cur = f"{d['current_val']:.1f}" if isinstance(d['current_val'], float) else f"{d['current_val']:,}"
                            _bas = f"{d['baseline_val']:.1f}" if isinstance(d['baseline_val'], float) else f"{d['baseline_val']:,}"
                            st.markdown(
                                f"<div style='background:#161b24; border:1px solid #2d3748; border-left:4px solid {flag_color}; border-radius:8px; padding:14px; text-align:center;'>"
                                f"<div style='color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>{d['param']}</div>"
                                f"<div style='color:#e2e8f0; font-size:0.95rem; font-weight:700;'>{_cur}</div>"
                                f"<div style='color:{flag_color}; font-size:0.82rem; font-weight:700; margin-top:4px;'>{d['delta_pct']:+.1f}% from baseline</div>"
                                f"<div style='color:#718096; font-size:0.72rem; margin-top:2px;'>Baseline: {_bas}</div>"
                                f"</div>",
                                unsafe_allow_html=True)
                st.divider()

            # ── G. AI Risk Assessment ─────────────────────────────────
            risk_buf_pdf  = None
            adj_risk_prob = 0.0
            adj_ci_lower  = 0.0
            adj_ci_upper  = 0.0

            # ── Serology Context Panel — runs regardless of CBC availability ──
            _ns1 = latest.get('ns1', 'Not Done')
            _igm = latest.get('igm', 'Not Done')
            _igg = latest.get('igg', 'Not Done')
            _has_serology = any(v not in ('Not Done', '') for v in [_ns1, _igm, _igg])

            if _has_serology:
                st.subheader("Serology")
                ser1, ser2, ser3, ser4 = st.columns(4)
                with ser1:
                    ns1_color = "#e74c3c" if _ns1 == "Positive" else ("#2ecc71" if _ns1 == "Negative" else "#8b92a8")
                    st.markdown(
                        f"<div style='text-align:center; padding:10px; background:#161b24; "
                        f"border:1px solid {ns1_color}; border-radius:8px;'>"
                        f"<div style='color:#a0aec0; font-size:0.72rem; text-transform:uppercase;'>NS1 Antigen</div>"
                        f"<div style='color:{ns1_color}; font-size:1.1rem; font-weight:700; margin-top:4px;'>{_ns1}</div>"
                        f"</div>", unsafe_allow_html=True)
                with ser2:
                    igm_color = "#f39c12" if _igm == "Reactive" else ("#2ecc71" if _igm == "Non-Reactive" else "#8b92a8")
                    st.markdown(
                        f"<div style='text-align:center; padding:10px; background:#161b24; "
                        f"border:1px solid {igm_color}; border-radius:8px;'>"
                        f"<div style='color:#a0aec0; font-size:0.72rem; text-transform:uppercase;'>IgM Anti-Dengue</div>"
                        f"<div style='color:{igm_color}; font-size:1.1rem; font-weight:700; margin-top:4px;'>{_igm}</div>"
                        f"</div>", unsafe_allow_html=True)
                with ser3:
                    igg_color = "#e74c3c" if _igg == "Reactive" else ("#2ecc71" if _igg == "Non-Reactive" else "#8b92a8")
                    st.markdown(
                        f"<div style='text-align:center; padding:10px; background:#161b24; "
                        f"border:1px solid {igg_color}; border-radius:8px;'>"
                        f"<div style='color:#a0aec0; font-size:0.72rem; text-transform:uppercase;'>IgG Anti-Dengue</div>"
                        f"<div style='color:{igg_color}; font-size:1.1rem; font-weight:700; margin-top:4px;'>{_igg}</div>"
                        f"</div>", unsafe_allow_html=True)
                with ser4:
                    _secondary  = st.session_state.get('is_secondary_dengue', False)
                    inf_type    = (
                        "Secondary Dengue" if _secondary
                        else ("Primary Dengue" if _igm == "Reactive" and _igg != "Reactive"
                              else "Indeterminate"))
                    inf_color   = "#e74c3c" if _secondary else ("#f39c12" if inf_type == "Primary Dengue" else "#8b92a8")
                    st.markdown(
                        f"<div style='text-align:center; padding:10px; background:#161b24; "
                        f"border:1px solid {inf_color}; border-radius:8px;'>"
                        f"<div style='color:#a0aec0; font-size:0.72rem; text-transform:uppercase;'>Infection Type</div>"
                        f"<div style='color:{inf_color}; font-size:1.1rem; font-weight:700; margin-top:4px;'>{inf_type}</div>"
                        f"</div>", unsafe_allow_html=True)

                if _ns1 == "Positive":
                    st.markdown(
                        '<div class="alert-warning"><b>NS1 Positive</b> — Active dengue viraemia confirmed '
                        '(typically Day 1-5). Monitor closely for clinical progression. '
                        'No steroids or NSAIDs.</div>',
                        unsafe_allow_html=True)
                if _secondary:
                    st.markdown(
                        '<div class="alert-critical"><b>Secondary Dengue Pattern</b> — IgG Reactive indicates '
                        'prior dengue infection. Risk of severe dengue is ~50× higher than primary infection. '
                        'Enhanced monitoring mandatory.</div>',
                        unsafe_allow_html=True)
                st.divider()

            # ── AI Risk Model ─────────────────────────────────────────
            st.subheader("AI Risk Assessment")

            # ── always-defined guard — used in treatment recs regardless of PLT availability
            has_bleeding = 1 if (
                "Bleeding"        in latest.get('symptoms',  []) or
                "Mucosal Bleeding" in latest.get('who_signs', [])
            ) else 0
            
            # ── Safe defaults — PDF export references these regardless of ML availability ──
            adj_risk_prob = 0.0
            adj_ci_lower  = 0.0
            adj_ci_upper  = 0.0
            pred_val      = 0.0       # platelet trajectory forecast
            risk_buf_pdf  = None     # PDF buffer for download (None if ML unavailable)
            
            # ── Use most recent report with a valid platelet count ─────────
            _risk_src = next((r for r in reversed(sorted_rep) if r['platelets'] > 0), None)

            if _risk_src is None:
                # ── ML unavailable — no platelet count in any saved report ─────────
                st.markdown(
                    "<div style='background:rgba(243,156,18,0.08); border:1px solid #f39c12; "
                    "border-left:4px solid #f39c12; border-radius:8px; padding:12px 16px; "
                    "margin-bottom:12px;'>"
                    "<b style='color:#f39c12;'>⚠ ML Risk Score Unavailable</b> — "
                    "<span style='color:#a0aec0; font-size:0.85rem;'>"
                    "Random Forest model requires a platelet count. "
                    "Add a CBC with platelet count to enable AI prediction, "
                    "24-hour trajectory forecast, and confidence intervals.</span><br>"
                    "<span style='color:#718096; font-size:0.78rem; margin-top:4px; display:block;'>"
                    "Clinical risk factor chart below remains active based on all other entered data.</span>"
                    "</div>",
                    unsafe_allow_html=True)

                # Build a dummy df for the function signature — ML panel will be skipped entirely
                _dummy_input = {col: [0] for col in clf_features}
                _dummy_df    = pd.DataFrame(_dummy_input)

                # Check whether there is enough clinical data to render the factors panel
                _has_any_clinical = any([
                    latest.get('ast', 0) > 0,    latest.get('alt', 0) > 0,
                    latest.get('inr', 0) > 0,    latest.get('spo2', 0) > 0,
                    latest.get('creatinine', 0) > 0,
                    latest.get('d_dimer', 0) > 0, latest.get('albumin', 0) > 0,
                    latest.get('pleural_effusion', False),
                    latest.get('ascites_grade', 0) > 0,
                    latest.get('gallbladder_wall_mm', 0) >= 5,
                    len(latest.get('who_signs', [])) > 0,
                    latest.get('shock_index', 0) >= 0.9,
                    latest.get('temperature', 0) >= 37.5,
                    latest.get('wbc', 0) > 0,
                    latest.get('map', 0) > 0 and latest.get('map', 0) < 65,
                ])

                if _has_any_clinical:
                    try:
                        risk_buf_screen, risk_buf_pdf = build_risk_chart(
                            classifier, _dummy_df,
                            urine_rate=latest.get('urine_output', 0),
                            latest_report=latest,
                            show_ml_panel=False)
                        st.image(risk_buf_screen, width='stretch')
                    except Exception as _rce:
                        st.warning(f"Clinical risk chart could not render: {_rce}")
                        risk_buf_pdf = None
                else:
                    risk_buf_pdf = None
                    st.markdown(
                        "<div style='background:rgba(0,0,0,0.15); border:1px solid #2d3748; "
                        "border-radius:6px; padding:10px 14px; color:#718096; font-size:0.82rem;'>"
                        "Add any of the following to activate the clinical chart: "
                        "<b>AST · ALT · INR · SpO2 · Creatinine · D-dimer · Albumin · "
                        "Imaging findings · WHO warning signs · Temperature · WBC</b>"
                        "</div>",
                        unsafe_allow_html=True)
                    
            else:
                # ── Seasonal Intelligence Badge ────────────────────────────
                _s_score = get_season_score(latest['datetime'])
                _s_meta  = get_season_meta(_s_score)
                st.markdown(
                    f"<div style='display:flex; align-items:center; gap:12px; "
                    f"flex-wrap:wrap; margin-bottom:8px;'>"
                    f"<span style='color:#8b92a8; font-size:0.82rem;'>"
                    f"Random Forest · n=2,455 · Sensitivity 100% · Specificity 99.2% · NPV 100%"
                    f"</span>"
                    f"<span class='{_s_meta['cls']}'>{_s_meta['label']}</span>"
                    f"</div>"
                    f"<div style='background:rgba(0,0,0,0.2); border-left:3px solid #2d3748; "
                    f"border-radius:4px; padding:4px 10px; margin-bottom:10px; "
                    f"font-size:0.74rem; color:#8b92a8;'>"
                    f"Seasonal Intelligence: {_s_meta['context']} — {_s_meta['tip']}"
                    f"</div>",
                    unsafe_allow_html=True)
                
                # ════════════════════════════════════════════════════════════════════════
                #  Changes:
                #    + Shock_Index and Pulse_Pressure added  (always available from latest)
                #    + WBC, AST, INR, SpO2, GCS, Has_Pleural_Effusion, Ascites_Grade added
                #      (0 when not entered — model trained to handle gracefully)
                # ════════════════════════════════════════════════════════════════════════
                syms         = latest.get('symptoms', [])
                who          = latest.get('who_signs', [])
                has_bleeding = 1 if ("Bleeding" in syms or "Mucosal Bleeding" in who) else 0

                risk_input = {
                    # ── Core CBC — from most recent report with valid platelets ─
                    'Platelet (cells/cu.mm)':                _risk_src['platelets'],
                    'Haemoglobin (gm/Dl)':                   _risk_src['hb'],
                    'Red Blood Cell Count (millions/cu.mm)':  _risk_src['rbc'],
                    'Hematocrit (Packed Cell Volume) (%)':    _risk_src['hct'],
                    # ── Demographics ──────────────────────────────────────
                    'Age':              age,
                    'Sex_Code':         1 if sex == "Male" else 0,
                    # ── Haemodynamics (derived — always available) ────────
                    'Shock_Index':      latest['shock_index'],
                    'Pulse_Pressure':   latest['pp'],
                    # ── Symptoms / WHO signs ──────────────────────────────
                    'Has_Fever':        1 if "Fever"      in syms else 0,
                    'Has_Headache':     1 if "Headache"   in syms else 0,
                    'Has_Pain':         1 if "Joint Pain" in syms else 0,
                    'Has_Vomit':        1 if ("Vomiting" in syms or
                                             "Persistent Vomiting" in who) else 0,
                    'Has_Bleeding':     has_bleeding,
                    # ── Extended clinical (0 when not entered) ────────────
                    'WBC':              latest.get('wbc', 0),
                    'AST':              latest.get('ast', 0),
                    'INR':              latest.get('inr', 0.0),
                    'SpO2':             latest.get('spo2', 0),
                    'GCS':              latest.get('gcs', 15),
                    'Has_Pleural_Effusion': 1 if latest.get('pleural_effusion', False) else 0,
                    'Ascites_Grade':    latest.get('ascites_grade', 0),
                    # ── Seasonal context ──────────────────────────────────
                    'Season_Risk':      get_season_score(latest['datetime']),
                }
                
                df_risk = pd.DataFrame([risk_input])
                for col_name in clf_features:
                    if col_name not in df_risk.columns:
                        df_risk[col_name] = 0
                df_risk = df_risk[clf_features]

                # ── Point estimate + 95% CI ───────────────────────────
                risk_prob_est, ci_lower, ci_upper, ci_width = compute_tree_ci(classifier, df_risk)
                urine_adj     = urine_risk_impact(latest['urine_output'])
                adj_risk_prob = float(np.clip(risk_prob_est + urine_adj * 0.4, 0.0, 1.0))
                adj_ci_lower  = float(np.clip(ci_lower      + urine_adj * 0.4, 0.0, 1.0))
                adj_ci_upper  = float(np.clip(ci_upper      + urine_adj * 0.4, 0.0, 1.0))

                # ── OOD detection ─────────────────────────────────────
                ood_score, ood_z_scores, is_ood = compute_ood_score(latest, age)

                # ── Risk label + donut ────────────────────────────────
                ai_col1, ai_col2 = st.columns([3, 1])
                with ai_col1:
                    if adj_risk_prob > .7:
                        st.error(f"### CRITICAL RISK ({adj_risk_prob*100:.1f}%)")
                    elif adj_risk_prob > .5:
                        st.warning(f"### HIGH RISK ({adj_risk_prob*100:.1f}%)")
                    else:
                        st.success(f"### LOW RISK ({adj_risk_prob*100:.1f}%)")

                    ci_label = (
                        "Wide — prediction unreliable" if ci_width > 0.4
                        else ("Moderate" if ci_width > 0.25 else "Narrow"))
                    st.markdown(
                        f'<div class="ci-panel">'
                        f'95% Confidence Interval: <b>{adj_ci_lower*100:.1f}% – {adj_ci_upper*100:.1f}%</b>'
                        f'&nbsp;&nbsp;|&nbsp;&nbsp;Width: {ci_width*100:.1f}pp ({ci_label})'
                        f'</div>',
                        unsafe_allow_html=True)

                    if latest['urine_output'] > 0:
                        adj_pct = (adj_risk_prob - risk_prob_est) * 100
                        st.caption(
                            f"ML base: {risk_prob_est*100:.1f}% | "
                            f"Urine adj: {'+' if adj_pct >= 0 else ''}{adj_pct:.1f}% | "
                            f"Final: {adj_risk_prob*100:.1f}%")

                    if is_ood:
                        top_z = sorted(ood_z_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                        z_str = ", ".join(f"{k} (z={v:+.1f})" for k, v in top_z)
                        st.markdown(
                            f'<div class="ood-panel">'
                            f'<b>Out-of-Distribution Warning</b> — '
                            f'Mahalanobis D={ood_score:.2f} (threshold 2.5)<br>'
                            f"This patient's profile is outside the training distribution. "
                            f'Prediction confidence is reduced. Clinical judgment should dominate.<br>'
                            f'<span style="font-size:0.78rem;">Outlier features: {z_str}</span>'
                            f'</div>',
                            unsafe_allow_html=True)

                with ai_col2:
                    fg, ag = plt.subplots(figsize=(2.5, 2.5))
                    ag.pie(
                        [adj_risk_prob, 1 - adj_risk_prob],
                        colors=['#e74c3c' if adj_risk_prob > .5 else '#2ecc71', '#ecf0f1'],
                        startangle=90, counterclock=False)
                    ag.add_artist(plt.Circle((0, 0), .7, color='#161b24'))
                    st.pyplot(fg)
                    plt.close(fg)

                # ── Risk Factors Analysis ─────────────────────────────
                st.markdown("---")
                st.markdown(
                    "**Risk Factors Analysis**&nbsp;&nbsp;"
                    f"<span style='color:#8b92a8; font-size:0.82rem; font-weight:400;'>"
                    f"{'SHAP values' if SHAP_AVAILABLE else 'Feature importances'} — "
                    f"contribution of each variable to this patient's risk score</span>",
                    unsafe_allow_html=True)

                try:
                    # Merge: current hemodynamics from latest + CBC from _risk_src
                    _risk_report_merged = {**latest,
                                           'platelets': _risk_src['platelets'],
                                           'hb':        _risk_src['hb'],
                                           'rbc':       _risk_src['rbc'],
                                           'hct':       _risk_src['hct']}
                    risk_buf_screen, risk_buf_pdf = build_risk_chart(
                        classifier, df_risk, urine_rate=latest['urine_output'],
                        latest_report=_risk_report_merged)
                    st.image(risk_buf_screen, width='stretch')
                
                except Exception as _rce:
                    import traceback
                    st.warning(f"Risk chart could not render: {_rce}")
                    st.code(traceback.format_exc(), language="python")  # dev visibility
                    risk_buf_pdf = None

            st.divider()
            
            # ── H. Vitals Matrix — ALL parameters ─────────
            st.subheader("Clinical Vitals Matrix")
            mdata = {}
            for r in sorted_rep:
                crt_labels = ['N/A', '<2s', '2-3s', '>3s']
                crt_v = crt_labels[min(int(r.get('crt', 0)), 3)]
                temp_v = r.get('temperature', 0.0)
                temp_str = f"{temp_v:.1f}°C" if temp_v >= 30.0 else "-"

                mdata[r['Label']] = [
                    # ── Demographics ──────────────────────
                    r['datetime'].strftime("%d-%b %I:%M %p"),
                    f"{r.get('age', '-')}y",
                    r.get('sex', '-'),
                    f"{r.get('weight', 0):.1f} kg" if r.get('weight', 0) > 0 else "-",
                    f"{r.get('bmi', 0):.1f}"        if r.get('bmi', 0) > 0    else "-",
                    # ── CBC Core ──────────────────────────
                    f"{r['platelets']:,}",
                    f"{r['hct']:.1f}%",
                    f"{r['hb']:.1f}",
                    f"{r['rbc']:.2f}",
                    # ── CBC Differential ──────────────────
                    f"{r.get('wbc', 0):,}"             if r.get('wbc', 0) > 0              else "-",
                    f"{r.get('neutrophil_pct', 0):.1f}%" if r.get('neutrophil_pct', 0) > 0  else "-",
                    f"{r.get('lymphocyte_pct', 0):.1f}%" if r.get('lymphocyte_pct', 0) > 0  else "-",
                    f"{r.get('mpv', 0):.1f} fL"         if r.get('mpv', 0) > 0              else "-",
                    # ── Haemodynamics ─────────────────────
                    f"{r['sys_bp']}/{r['dia_bp']}",
                    f"{r['hr']}",
                    f"{r['map']:.1f}",
                    f"{r['shock_index']:.2f}",
                    f"{r['sys_bp'] - r['dia_bp']} mmHg",
                    # ── Extended Vitals ───────────────────
                    temp_str,
                    f"{r.get('spo2', 0)}%"   if r.get('spo2', 0) > 0  else "-",
                    f"{r.get('rr', 0)}/min"  if r.get('rr', 0) > 0    else "-",
                    f"{r.get('gcs', 15)}",
                    crt_v,
                    # ── Fluid Balance ─────────────────────
                    f"{r.get('urine_output', 0):.2f}" if r.get('urine_output', 0) > 0 else "-",
                    # ── LFT ──────────────────────────────
                    f"{r.get('ast', 0):,}"       if r.get('ast', 0) > 0       else "-",
                    f"{r.get('alt', 0):,}"       if r.get('alt', 0) > 0       else "-",
                    f"{r.get('albumin', 0):.1f}" if r.get('albumin', 0) > 0   else "-",
                    f"{r.get('bilirubin_total', 0):.1f}"  if r.get('bilirubin_total', 0) > 0  else "-",
                    f"{r.get('bilirubin_direct', 0):.1f}" if r.get('bilirubin_direct', 0) > 0 else "-",
                    # ── Coagulation ───────────────────────
                    f"{r.get('pt', 0):.1f}s"   if r.get('pt', 0) > 0    else "-",
                    f"{r.get('inr', 0):.2f}"   if r.get('inr', 0) > 0   else "-",
                    f"{r.get('aptt', 0):.1f}s" if r.get('aptt', 0) > 0  else "-",
                    f"{r.get('d_dimer', 0):,}"  if r.get('d_dimer', 0) > 0 else "-",
                    # ── Renal / Electrolytes ──────────────
                    f"{r.get('creatinine', 0):.2f}"  if r.get('creatinine', 0) > 0  else "-",
                    f"{r.get('bun', 0):.1f}"         if r.get('bun', 0) > 0         else "-",
                    f"{r.get('sodium', 0):.0f}"      if r.get('sodium', 0) > 0      else "-",
                    f"{r.get('potassium', 0):.1f}"   if r.get('potassium', 0) > 0   else "-",
                    f"{r.get('bicarbonate', 0):.1f}" if r.get('bicarbonate', 0) > 0 else "-",
                    # ── Serology ─────────────────────────
                    r.get('ns1', 'Not Done'),
                    r.get('igm', 'Not Done'),
                    r.get('igg', 'Not Done'),
                    # ── Imaging ──────────────────────────
                    "Yes" if r.get('pleural_effusion', False) else "No",
                    f"{r.get('gallbladder_wall_mm', 0):.1f} mm" if r.get('gallbladder_wall_mm', 0) > 0 else "-",
                    f"Grade {r.get('ascites_grade', 0)}"        if r.get('ascites_grade', 0) > 0       else "-",
                    # ── WHO / Symptoms ────────────────────
                    ", ".join(r.get('who_signs', [])) if r.get('who_signs') else "None",
                    ", ".join(r.get('symptoms', []))  if r.get('symptoms')  else "None",
                ]

            _vitals_index = [
                # Demographics
                "Date/Time", "Age", "Sex", "Weight", "BMI",
                # CBC Core
                "Platelets (cells/uL)", "Hct (%)", "Hb (g/dL)", "RBC (M/uL)",
                # CBC Differential
                "WBC (cells/uL)", "Neutrophil %", "Lymphocyte %", "MPV (fL)",
                # Haemodynamics
                "BP (mmHg)", "HR (bpm)", "MAP (mmHg)", "Shock Index", "Pulse Pressure",
                # Extended Vitals
                "Temperature", "SpO2", "Resp Rate", "GCS", "Cap Refill",
                # Fluid
                "Urine Output (mL/kg/hr)",
                # LFT
                "AST (IU/L)", "ALT (IU/L)", "Albumin (g/dL)", "Total Bili (mg/dL)", "Direct Bili (mg/dL)",
                # Coagulation
                "PT (s)", "INR", "aPTT (s)", "D-dimer (ng/mL)",
                # Renal
                "Creatinine (mg/dL)", "BUN (mg/dL)", "Sodium (mEq/L)", "Potassium (mEq/L)", "HCO3 (mEq/L)",
                # Serology
                "NS1 Antigen", "IgM Anti-Dengue", "IgG Anti-Dengue",
                # Imaging
                "Pleural Effusion", "GB Wall Thickness", "Ascites Grade",
                # Clinical
                "WHO Warning Signs", "Symptoms",
            ]

            df_mat = pd.DataFrame(mdata, index=_vitals_index)
            # Only show rows with at least one non-default value across all reports
            _non_empty = df_mat.apply(
                lambda row: not all(v in ('-', 'Not Done', 'None', '0', '0%', '0/min', '15', 'N/A', 'No', 'Grade 0')
                                    for v in row), axis=1)
            st.dataframe(df_mat[_non_empty], width='stretch')
            st.caption("Rows with no data across all reports are hidden. '-' = not recorded.")
            st.divider()

            # ── I. Platelet Trajectory ────────────────────
            st.subheader("Platelet Trajectory & Forecast")
            plot_buf_pdf = None
            pred_val     = 0
            velocity     = 0
            
            # Only use reports that actually have a platelet count recorded
            _plt_reps = [r for r in sorted_rep if r['platelets'] > 0]
            if len(_plt_reps) >= 2:
                t0   = _plt_reps[0]['datetime']
                days = np.array(
                    [(d['datetime'] - t0).total_seconds() / 86400 for d in _plt_reps]).reshape(-1, 1)
                plts = np.array([d['platelets'] for d in _plt_reps]).reshape(-1, 1)
                lr       = LinearRegression().fit(days, plts)
                velocity = lr.coef_[0][0]
                pred_val = max(0, int(_plt_reps[-1]['platelets'] + velocity))

                # ── Define fd and pred_y here — used by BOTH chart blocks ──
                fd     = np.array([days[-1][0], days[-1][0] + 1]).reshape(-1, 1)
                pred_y = lr.predict(fd)

                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    st.metric("Overall Velocity", f"{int(velocity):,}/day", delta_color="inverse" if velocity < -20000 else "normal")
                with fc2:
                    st.metric("24h Platelet Forecast", f"{pred_val:,} cells/uL", delta=f"{int(pred_val - latest['platelets'])}")
                with fc3:
                    st.metric("Trend", "Declining" if velocity < 0 else "Rising")

                from sklearn.metrics import r2_score
                r2 = r2_score(plts, lr.predict(days))
                st.caption(f"Linear fit R2 on this patient: {r2:.4f} | Validation R2=0.9945 (n=2,455)")

                # ── Plotly interactive chart ───────────────────────────
                _y_max  = max(int(max(plts.flatten()) * 1.15), 160000)
                _x_max  = float(max(fd.flatten())) + 0.1

                fig = go.Figure()

                # Threshold zones (background)
                fig.add_shape(type="rect",
                              x0=float(min(days.flatten())), x1=_x_max,
                              y0=0, y1=50000,
                              fillcolor="rgba(231,76,60,0.15)",
                              layer="below", line_width=0)
                fig.add_shape(type="rect",
                              x0=float(min(days.flatten())), x1=_x_max,
                              y0=50000, y1=100000,
                              fillcolor="rgba(243,156,18,0.12)",
                              layer="below", line_width=0)

                # Explicit threshold reference lines (match PDF)
                fig.add_hline(y=100000,
                              line=dict(color="rgba(243,156,18,0.70)", width=1.5, dash="dash"),
                              annotation_text="100k  warning threshold",
                              annotation_position="right",
                              annotation_font=dict(color="rgba(243,156,18,0.85)", size=11))
                fig.add_hline(y=50000,
                              line=dict(color="rgba(231,76,60,0.70)", width=1.5, dash="dash"),
                              annotation_text="50k  severe threshold",
                              annotation_position="right",
                              annotation_font=dict(color="rgba(231,76,60,0.85)", size=11))

                # Observed trajectory
                fig.add_trace(go.Scatter(
                    x=days.flatten(), y=plts.flatten(),
                    mode='lines+markers', name='Observed',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=9, symbol='circle')))

                # 24-hour forecast
                fig.add_trace(go.Scatter(
                    x=fd.flatten(), y=pred_y.flatten(),
                    mode='lines', name='24h Forecast',
                    line=dict(color='#e74c3c', width=2.5, dash='dot')))

                fig.update_layout(
                    title=dict(text="Platelet Trajectory", font=dict(size=15)),
                    xaxis_title="Days since first report",
                    yaxis_title="Platelet Count (cells/uL)",
                    yaxis=dict(range=[0, _y_max], tickformat=","),
                    template="plotly_dark",
                    height=440,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(r=160))   # right margin for threshold annotations
                st.plotly_chart(fig, width='stretch')

                # ── Matplotlib static chart (PDF export) ──────────────
                _y_max_mpl = max(int(max(plts.flatten()) * 1.15), 160000)
                fig_s, ax  = plt.subplots(figsize=(11, 4.5))
                ax.fill_between(
                    [float(min(days.flatten())), float(max(fd.flatten()))],
                    0, 50000,
                    color='#e74c3c', alpha=0.08, zorder=0)
                ax.fill_between(
                    [float(min(days.flatten())), float(max(fd.flatten()))],
                    50000, 100000,
                    color='#f39c12', alpha=0.07, zorder=0)
                ax.plot(days, plts, 'o-',
                        color='#3498db', linewidth=2.5, markersize=8,
                        label='Observed', zorder=3)
                ax.plot(fd, pred_y, '--',
                        color='#e74c3c', linewidth=2.5,
                        label='24h Forecast', zorder=3)
                ax.axhline(100000, color='#f39c12', linestyle='--',
                           linewidth=1.2, alpha=0.75,
                           label='100k warning threshold')
                ax.axhline(50000,  color='#e74c3c', linestyle='--',
                           linewidth=1.2, alpha=0.75,
                           label='50k severe threshold')
                ax.set_ylim(0, _y_max_mpl)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
                ax.set_title("Platelet Trajectory", fontsize=13, fontweight='700')
                ax.set_xlabel("Days since first report")
                ax.set_ylabel("Platelet Count (cells/uL)")
                ax.legend(loc='upper right', fontsize=9)
                _dark_ax(ax, fig_s)
                plot_buf_pdf = io.BytesIO()
                fig_s.savefig(plot_buf_pdf, format='png',
                              bbox_inches='tight', dpi=200)
                plot_buf_pdf.seek(0)
                plt.close(fig_s)
            else:
                if len(sorted_rep) >= 2:
                    st.info("Enter at least 2 reports **with Platelet Count > 0** to generate trajectory and forecast.")
                else:
                    st.info("Enter at least 2 reports to generate trajectory analysis and serial alerts.")

            st.divider()
            
            # ── J. Discharge Readiness ────────────────────
            st.subheader("Discharge Readiness Checklist")
            st.caption("Based on WHO 2009 dengue discharge criteria")
            _is_discharge_on = st.session_state.get('discharge_enabled', False)
            prev_plt          = sorted_rep[-2]['platelets'] if len(sorted_rep) >= 2 else None
            discharge_result  = check_discharge_readiness(
                latest_report=latest,
                prev_platelet=prev_plt,
                fever_free_hours=(
                    st.session_state.discharge_fever_free
                    if _is_discharge_on and st.session_state.discharge_fever_free > 0 else None),
                tolerating_orals=(
                    st.session_state.discharge_tolerating_orals
                    if _is_discharge_on and st.session_state.discharge_tolerating_orals else None))
            discharge_criteria, discharge_ready = discharge_result
            if discharge_ready:
                st.success("WHO Discharge Criteria Met — Patient may be eligible for discharge")
            else:
                st.error("WHO Discharge Criteria Not Yet Met — Continue inpatient monitoring")
            dc1, dc2 = st.columns(2)
            half = len(discharge_criteria) // 2
            for col, cset in [(dc1, discharge_criteria[:half + 1]), (dc2, discharge_criteria[half + 1:])]:
                with col:
                    for c_name, c_pass, c_detail in cset:
                        marker = "+" if c_pass == True else ("?" if c_pass is None else "-")
                        color  = "#2ecc71" if c_pass == True else ("#8b92a8" if c_pass is None else "#e74c3c")
                        st.markdown(
                            f"<p style='color:{color}; font-size:0.85rem; margin:4px 0;'>"
                            f"[{marker}] <b>{c_name}</b><br>"
                            f"<span style='font-size:0.78rem; opacity:0.8; padding-left:20px;'>{c_detail}</span></p>",
                            unsafe_allow_html=True)
            if not _is_discharge_on:
                st.caption("Toggle Discharge Assessment in the sidebar to enter fever-free hours and oral tolerance status.")
            st.divider()

            # ── K. Clinical Recommendations ───────────────
            st.subheader("Clinical Recommendations")
            re1, re2 = st.columns(2)
            with re1:
                st.markdown(f"**Monitoring — WHO Group {who_group}:**")
                if who_group == 'C':
                    st.markdown("- ICU-level monitoring\n- Vitals every 15-30 min\n- CBC every 4-6 hours")
                elif who_group == 'B':
                    st.markdown("- Ward monitoring\n- Vitals every 1-2 hours\n- CBC every 6-8 hours")
                else:
                    st.markdown("- Outpatient or observation\n- Vitals every 4-6 hours if admitted\n- Daily CBC")
            with re2:
                st.markdown("**Treatment Considerations:**")
                if latest['shock_index'] > .9:
                    st.markdown("- Elevated shock index — assess fluid status urgently")
                if latest['map'] < 65:
                    st.markdown("- Low MAP — consider fluid bolus per Group C protocol")
                if latest['pp'] <= 20:
                    st.markdown("- Narrow Pulse Pressure <=20 mmHg — impending shock alert")
                if 0 < latest['platelets'] < 50000:
                    st.markdown("- Severe thrombocytopenia — prepare blood bank, no NSAIDs")
                if latest['platelets'] < 20000:
                    st.markdown("- Critical platelet nadir — consider prophylactic platelet transfusion")
                if latest['urine_output'] > 0:
                    if latest['urine_output'] < 0.5:
                        st.markdown("- Oliguria — renal function workup, consider fluid challenge")
                    elif latest['urine_output'] < 1.0:
                        st.markdown("- Borderline urine output — monitor hourly")
                if len(latest['who_signs']) > 0:
                    st.markdown(f"- {len(latest['who_signs'])} WHO warning sign(s) active — requires admission")
                if has_bleeding:
                    st.markdown("- Bleeding noted — coagulation panel, avoid anticoagulants")
                if len(sorted_rep) >= 2:
                    hct_trend = sorted_rep[-1]['hct'] - sorted_rep[0]['hct']
                    if hct_trend < -5 and latest.get('urine_output', 0) > 2.0:
                        st.markdown("- Fluid overload risk — reassess IV rate, watch for respiratory compromise")
            st.divider()

            # ── L. PDF Export ─────────────────────────────
            if PDF_AVAILABLE:
                st.subheader("Export Clinical Report")
                pdat = {
                    'age':    age,
                    'sex':    sex,
                    'name':   name,
                    'weight': final_w_kg if enable_metrics else 0,
                    'bmi':    current_bmi if enable_metrics else 0,
                }
                cdat = {
                    'risk_prob':    adj_risk_prob,
                    'forecast_val': pred_val,
                    'fluid_rate':   int(fluid_rate),
                    'ci_lower':     adj_ci_lower,
                    'ci_upper':     adj_ci_upper,
                }
                try:
                    pdf_bytes = create_pdf(
                        pdat, sorted_rep, cdat, plot_buf_pdf, risk_buf_pdf,
                        st.session_state.clinician_notes,
                        who_data=(who_class, who_group, who_criteria, who_desc),
                        alerts=serial_alerts if serial_alerts else None,
                        discharge=discharge_result if discharge_result else None,
                        severity_data=(sev_score, sev_direction, sev_components))
                    filename = (f"Dengue_CDSS_{name.replace(' ','_') + '_' if name else ''}"
                                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
                    st.download_button(
                        "Download Full Clinical Report (PDF)",
                        pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        width='stretch')
                    st.caption(
                        "Includes: WHO classification · Serial alerts · Severity score · Vitals matrix · "
                        "Risk chart with CI · Trajectory countdowns · Discharge checklist")
                except Exception as e:
                    st.error(f"PDF generation error: {e}")
            else:
                st.warning("PDF export unavailable. Install: pip install fpdf")
        else:
            st.info(
                "No saved reports found. Enter clinical data in the sidebar under Report A, "
                "then click **Save Report Data**, followed by **RUN ANALYSIS**. "
                "CBC with platelet count enables the full ML risk model; "
                "all other modules (WHO classification, vitals, imaging, serology) "
                "are active with or without platelet data.")