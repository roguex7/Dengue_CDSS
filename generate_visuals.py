"""
generate_visuals.py  —  Dengue CDSS  |  Architecture & Pipeline Visualizations
═══════════════════════════════════════════════════════════════════════════════
Generates three professional publication-grade figures:
  1. System Architecture   — component hierarchy and data flow
  2. ML Data Pipeline      — raw CSV → training → inference → output
  3. Clinical Workflow     — patient journey through the CDSS

Run standalone:
    python generate_visuals.py

Outputs (saved to ./visuals/):
  system_architecture.png
  data_pipeline.png
  clinical_workflow.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

OUTPUT_DIR = "./visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Global style ────────────────────────────────────────────────────────────
BG        = "#0e1117"
BG2       = "#161b24"
BG3       = "#1a2540"
BLUE      = "#3498db"
BLUE_LT   = "#5dade2"
RED       = "#e74c3c"
RED_LT    = "#f1948a"
GREEN     = "#2ecc71"
GREEN_LT  = "#82e0aa"
AMBER     = "#f39c12"
PURPLE    = "#9b59b6"
GREY      = "#8b92a8"
WHITE     = "#e2e8f0"
BORDER    = "#2d4a6e"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "text.color":        WHITE,
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})


def _box(ax, x, y, w, h, label, sublabel="", color=BLUE, alpha=0.18,
         fontsize=9, sublabel_size=7.5, radius=0.015):
    """Draw a rounded rectangle node with label + optional sublabel."""
    fc = (*[c/255 for c in bytes.fromhex(color.lstrip('#'))], alpha)
    ec = color
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0.005,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3)
    ax.add_patch(box)
    ty = y + (h * 0.12 if sublabel else 0)
    ax.text(x, ty, label, ha='center', va='center',
            fontsize=fontsize, fontweight='700', color=WHITE, zorder=4)
    if sublabel:
        ax.text(x, y - h * 0.22, sublabel, ha='center', va='center',
                fontsize=sublabel_size, color=GREY, zorder=4)


def _arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.2,
           style="-|>", label="", label_color=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.01, my, label, fontsize=6.5,
                color=label_color or color, ha='left', va='center', zorder=5)


def _section_bar(ax, x, y, w, h, color, label, fontsize=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                  boxstyle="round,pad=0.003",
                  facecolor=(*[c/255 for c in bytes.fromhex(color.lstrip('#'))], 0.22),
                  edgecolor=color, linewidth=1.0, zorder=1))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight='700', color=color, zorder=2)


# ════════════════════════════════════════════════════════════════════════════
#  1.  SYSTEM ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════
def plot_system_architecture():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    fig.suptitle("Dengue CDSS — System Architecture",
                 fontsize=15, fontweight='800', color=WHITE, y=0.97)
    ax.text(0.5, 0.935, "Component hierarchy · Data flow · Module dependencies",
            ha='center', fontsize=9, color=GREY)

    # ── Layer backgrounds ─────────────────────────────────────────────────
    layers = [
        (0.01, 0.86, 0.98, 0.10, BLUE,   "INPUT LAYER"),
        (0.01, 0.63, 0.98, 0.22, GREEN,  "ML ENGINE LAYER"),
        (0.01, 0.31, 0.98, 0.31, AMBER,  "CLINICAL RULES LAYER"),
        (0.01, 0.08, 0.98, 0.22, PURPLE, "OUTPUT LAYER"),
    ]
    for lx, ly, lw, lh, lc, ll in layers:
        _section_bar(ax, lx, ly, lw, lh, lc, ll, fontsize=7.5)

    # ── INPUT LAYER ───────────────────────────────────────────────────────
    inputs = [
        (0.10, 0.91, "CBC / Lab\nReport", "Platelets · Hb · Hct · RBC", BLUE),
        (0.26, 0.91, "Vital Signs", "BP · HR · Temp · SpO2 · RR · GCS", BLUE),
        (0.42, 0.91, "Extended\nPanels", "LFT · Coag · Renal · Electrolytes", BLUE),
        (0.58, 0.91, "Imaging\nFindings", "Pleural · Ascites · GB Wall", BLUE),
        (0.74, 0.91, "Serology", "NS1 · IgM · IgG", BLUE),
        (0.90, 0.91, "OCR Upload\n(JPG/PNG/PDF)", "Auto-extract · Confidence score", BLUE_LT),
    ]
    for x, y, lbl, sub, c in inputs:
        _box(ax, x, y, 0.13, 0.07, lbl, sub, c, fontsize=8)

    # ── ML ENGINE LAYER ───────────────────────────────────────────────────
    _box(ax, 0.22, 0.79, 0.28, 0.065,
         "RandomForestClassifier", "300 trees · 21 features · max_depth=12",
         GREEN, fontsize=9)
    _box(ax, 0.22, 0.715, 0.28, 0.055,
         "Risk Probability", "AUC=0.9996 · Sens=99.77% · Spec=100%",
         GREEN, fontsize=8.5)
    _box(ax, 0.60, 0.79, 0.28, 0.065,
         "GradientBoostingRegressor", "300 estimators · lr=0.08 · depth=5",
         GREEN, fontsize=9)
    _box(ax, 0.60, 0.715, 0.28, 0.055,
         "24h Platelet Forecast", "R²=0.9953 · MAE=2,515 cells/µL",
         GREEN, fontsize=8.5)
    _box(ax, 0.41, 0.75, 0.12, 0.075,
         "95% CI\nEngine", "Tree ensemble\npercentile CI", BLUE_LT, fontsize=7.5)
    _box(ax, 0.41, 0.665, 0.12, 0.055,
         "OOD Detector", "Mahalanobis D", AMBER, fontsize=7.5)
    _box(ax, 0.92, 0.75, 0.12, 0.055,
         "Seasonal\nIntelligence", "Score 0–3", BLUE_LT, fontsize=7.5)

    # ML arrows
    for x in [0.10, 0.26, 0.42, 0.58, 0.74]:
        _arrow(ax, x, 0.875, x, 0.83, GREEN)
    _arrow(ax, 0.22, 0.755, 0.22, 0.74, GREEN)
    _arrow(ax, 0.60, 0.755, 0.60, 0.74, GREEN)

    # ── CLINICAL RULES LAYER ──────────────────────────────────────────────
    clinical = [
        (0.09, 0.53, "WHO 2009\nClassification", "Group A / B / C\n+ Organ criteria", RED),
        (0.24, 0.53, "Trajectory\nEngine", "Velocity · Accel\nCountdowns", AMBER),
        (0.39, 0.53, "Serial Alert\nSystem", "7 alert types\nPriority suppression", RED),
        (0.54, 0.53, "Plasma Leakage\nScore", "Multi-marker\n0–100%", AMBER),
        (0.69, 0.53, "Severity\nScore", "0–100 composite\n4 domains", RED),
        (0.84, 0.53, "FORS\nEngine", "Fluid overload\npost-resus", PURPLE),
    ]
    for x, y, lbl, sub, c in clinical:
        _box(ax, x, y, 0.13, 0.075, lbl, sub, c, fontsize=8)

    clinical2 = [
        (0.16, 0.43, "Bleeding Risk\nScore", "6-point BRS\nTransfusion guide", RED),
        (0.32, 0.43, "AKI Detector", "KDIGO criteria\nCKD-EPI eGFR", AMBER),
        (0.48, 0.43, "Serology\nEngine", "Primary/Secondary\nInfection type", BLUE_LT),
        (0.64, 0.43, "Discharge\nChecklist", "7 WHO criteria\nReadiness score", GREEN),
        (0.80, 0.43, "Personal Delta\nAnalysis", "Patient-own\nbaseline compare", BLUE_LT),
    ]
    for x, y, lbl, sub, c in clinical2:
        _box(ax, x, y, 0.13, 0.075, lbl, sub, c, fontsize=8)

    for x in [0.09, 0.24, 0.39, 0.54, 0.69, 0.84]:
        _arrow(ax, x, 0.685, x, 0.57, AMBER)

    # ── OUTPUT LAYER ──────────────────────────────────────────────────────
    outputs = [
        (0.12, 0.23, "Clinical\nDashboard", "Streamlit UI\nReal-time", BLUE),
        (0.28, 0.23, "Vitals\nMatrix", "All parameters\nSerial table", BLUE),
        (0.44, 0.23, "Plotly\nTrajectory", "Interactive chart\n+ forecast", GREEN),
        (0.60, 0.23, "Risk Factor\nAnalysis", "SHAP / MDI\nDual panel", AMBER),
        (0.76, 0.23, "PDF Report\nExport", "fpdf · A4 landscape\nFull clinical", RED),
        (0.92, 0.23, "Alert\nBanner", "Priority suppressed\nTop CRITICAL", RED),
    ]
    for x, y, lbl, sub, c in outputs:
        _box(ax, x, y, 0.13, 0.08, lbl, sub, c, fontsize=8)

    for x in [0.12, 0.28, 0.44, 0.60, 0.76, 0.92]:
        _arrow(ax, x, 0.39, x, 0.27, PURPLE)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        (BLUE,   "Input / UI"),
        (GREEN,  "ML Engine"),
        (AMBER,  "Clinical Rules"),
        (RED,    "Alert / Severity"),
        (PURPLE, "Output"),
    ]
    for i, (c, lbl) in enumerate(legend_items):
        ax.add_patch(plt.Rectangle((0.02 + i*0.19, 0.02), 0.016, 0.03,
                     facecolor=c, alpha=0.7, transform=ax.transAxes))
        ax.text(0.042 + i*0.19, 0.035, lbl, fontsize=7.5, color=WHITE,
                transform=ax.transAxes, va='center')

    fig.savefig(f"{OUTPUT_DIR}/system_architecture.png")
    plt.close(fig)
    print("  ✅  system_architecture.png saved")


# ════════════════════════════════════════════════════════════════════════════
#  2.  ML DATA PIPELINE
# ════════════════════════════════════════════════════════════════════════════
def plot_data_pipeline():
    fig, ax = plt.subplots(figsize=(20, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    fig.suptitle("Dengue CDSS — ML Data Pipeline",
                 fontsize=15, fontweight='800', color=WHITE, y=0.97)
    ax.text(0.5, 0.935,
            "Raw CSV → Feature Engineering → Model Training → Inference → Clinical Output",
            ha='center', fontsize=9, color=GREY)

    # ── Stage backgrounds ─────────────────────────────────────────────────
    stages = [
        (0.01, 0.08, 0.145, 0.82, BLUE,   "DATA\nINGESTION"),
        (0.165, 0.08, 0.165, 0.82, GREEN,  "FEATURE\nENGINEERING"),
        (0.34, 0.08, 0.165, 0.82, AMBER,  "MODEL\nTRAINING"),
        (0.515, 0.08, 0.165, 0.82, PURPLE, "VALIDATION"),
        (0.69, 0.08, 0.145, 0.82, RED,    "INFERENCE"),
        (0.845, 0.08, 0.145, 0.82, BLUE_LT,"CLINICAL\nOUTPUT"),
    ]
    for sx, sy, sw, sh, sc, sl in stages:
        _section_bar(ax, sx, sy, sw, sh, sc, sl, fontsize=8)

    # ── Stage nodes ───────────────────────────────────────────────────────
    W = 0.115; H = 0.085

    # Ingestion
    _box(ax, 0.083, 0.82, W, H, "dengue_data_\ncleaned_debug.csv", "n=2,455 rows", BLUE)
    _box(ax, 0.083, 0.70, W, H, "PHYS_BOUNDS\nClipping", "CBC unit-error fix\nRBC: 1–10 M/µL", BLUE)
    _box(ax, 0.083, 0.58, W, H, "Column\nNormaliser", "Age · Sex\nAuto-rename", BLUE)
    _box(ax, 0.083, 0.46, W, H, "Datetime\nParse", "Date of Test\ndayfirst=True", BLUE)

    # Feature Eng
    _box(ax, 0.248, 0.82, W, H, "Season_Risk\nEncoder", "Month → 0/1/2/3\nMonsoon=3", GREEN)
    _box(ax, 0.248, 0.70, W, H, "Symptom\nFlags", "5 binary features\nfever/bleed/vomit…", GREEN)
    _box(ax, 0.248, 0.58, W, H, "Synthetic\nColumn Gen", "WBC · AST · INR\nSpO2 · GCS · Imaging", GREEN)
    _box(ax, 0.248, 0.46, W, H, "Derived\nHaemodynamics", "Shock Index\nPulse Pressure · MAP", GREEN)
    _box(ax, 0.248, 0.34, W, H, "WHO Label\nBuilder", "11-criterion\nmulti-label OR", AMBER)

    # Training
    _box(ax, 0.423, 0.82, W, H, "clf_features\n21 columns", "Contract with\napp.py risk_input", AMBER)
    _box(ax, 0.423, 0.70, W, H, "Train/Test Split", "80/20 · stratified\nrandom_state=42", AMBER)
    _box(ax, 0.423, 0.57, W, H, "RandomForest\nClassifier", "300 trees\nmax_depth=12\nbalanced", GREEN)
    _box(ax, 0.423, 0.44, W, H, "Gradient\nBoosting Reg.", "300 estimators\nlr=0.08 · depth=5", GREEN)
    _box(ax, 0.423, 0.31, W, H, "Platelet\nTrajectory Syn.", "Day1/Day2/Delta\nrecovery momentum", BLUE_LT)

    # Validation
    _box(ax, 0.598, 0.83, W, H, "Hold-out\nTest Set", "n=491 (20%)\nStratified", PURPLE)
    _box(ax, 0.598, 0.71, W, H, "Confusion\nMatrix", "TP=438  FP=0\nFN=1   TN=52", PURPLE)
    _box(ax, 0.598, 0.59, W, H, "Classifier\nMetrics", "AUC=0.9996\nSens=99.77%\nSpec=100%", GREEN)
    _box(ax, 0.598, 0.46, W, H, "5-Fold\nCross-Val", "StratifiedKFold\nn_splits=5", PURPLE)
    _box(ax, 0.598, 0.34, W, H, "Regressor\nMetrics", "R²=0.9953\nMAE=2,515", GREEN)
    _box(ax, 0.598, 0.22, W, H, "TRAIN_STATS\nSanity Check", "_STAT_SANITY\nOOD calibration", AMBER)

    # Inference
    _box(ax, 0.763, 0.82, W, H, "User Input\n(app.py form)", "Manual / OCR\nSidebar entry", RED)
    _box(ax, 0.763, 0.70, W, H, "Feature\nContract Check", "Missing → 0\nOrder enforced", RED)
    _box(ax, 0.763, 0.58, W, H, "clf.predict\n_proba()", "Point estimate\n+ tree CI", RED)
    _box(ax, 0.763, 0.46, W, H, "Urine Output\nAdjustment", "±40% modifier\nrenal signal", AMBER)
    _box(ax, 0.763, 0.34, W, H, "OOD\nDetector", "Mahalanobis D\nthreshold=2.5", AMBER)

    # Output
    _box(ax, 0.918, 0.82, W, H, "Risk Label\n+ 95% CI", "HIGH / LOW\nCI width flag", RED)
    _box(ax, 0.918, 0.70, W, H, "WHO\nClassification", "Group A/B/C\n+ organ criteria", RED)
    _box(ax, 0.918, 0.58, W, H, "Serial\nAlerts", "7 types\nPriority-suppressed", RED)
    _box(ax, 0.918, 0.46, W, H, "Trajectory\nCountdowns", "Time-to-threshold\nPLT · Hct · SI", BLUE_LT)
    _box(ax, 0.918, 0.34, W, H, "PDF Report\nExport", "fpdf A4\nFull clinical", BLUE_LT)

    # ── Arrows between stages (horizontal) ───────────────────────────────
    arrow_ys = [0.82, 0.70, 0.58, 0.46, 0.34]
    x_pairs  = [(0.14, 0.19), (0.31, 0.36), (0.485, 0.535), (0.66, 0.71), (0.83, 0.86)]
    for y in arrow_ys:
        for x0, x1 in x_pairs:
            if y >= 0.34:
                _arrow(ax, x0, y, x1, y, GREY, lw=1.0)

    # Vertical arrows within stages
    for x, ys in [
        (0.083, [0.775, 0.655, 0.535, 0.415]),
        (0.248, [0.775, 0.655, 0.535, 0.415, 0.295]),
        (0.423, [0.775, 0.655, 0.525, 0.395, 0.275]),
        (0.598, [0.785, 0.665, 0.545, 0.415, 0.295, 0.175]),
        (0.763, [0.775, 0.655, 0.535, 0.415, 0.295]),
        (0.918, [0.775, 0.655, 0.535, 0.415]),
    ]:
        for y in ys:
            _arrow(ax, x, y, x, y - 0.07, GREY, lw=0.9)

    # ── Stats callout ─────────────────────────────────────────────────────
    ax.text(0.5, 0.025,
            "n=2,455 patients  ·  21 clf features  ·  17 reg features  ·  "
            "80/20 stratified split  ·  5-Fold CV  ·  "
            "Label balance: 89.5% high-risk / 10.5% low-risk",
            ha='center', fontsize=8, color=GREY)

    fig.savefig(f"{OUTPUT_DIR}/data_pipeline.png")
    plt.close(fig)
    print("  ✅  data_pipeline.png saved")


# ════════════════════════════════════════════════════════════════════════════
#  3.  CLINICAL WORKFLOW
# ════════════════════════════════════════════════════════════════════════════
def plot_clinical_workflow():
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    fig.suptitle("Dengue CDSS — Clinical Decision Workflow",
                 fontsize=15, fontweight='800', color=WHITE, y=0.98)
    ax.text(0.5, 0.955,
            "Patient admission → Serial monitoring → WHO triage → Discharge",
            ha='center', fontsize=9, color=GREY)

    W = 0.13; H = 0.072

    # ── Column headers ────────────────────────────────────────────────────
    headers = [
        (0.10, "ADMISSION\n& DATA ENTRY", BLUE),
        (0.30, "AI ASSESSMENT", GREEN),
        (0.50, "WHO TRIAGE", AMBER),
        (0.70, "MONITORING\n& ALERTS", RED),
        (0.90, "OUTCOME\n& DISCHARGE", PURPLE),
    ]
    for hx, hl, hc in headers:
        _box(ax, hx, 0.91, W+0.01, 0.055, hl, "", hc, alpha=0.30, fontsize=8.5)

    # ── Column background lines ───────────────────────────────────────────
    for hx in [0.10, 0.30, 0.50, 0.70, 0.90]:
        ax.axvline(hx, 0.06, 0.87, color=BORDER, lw=0.5, alpha=0.4)

    # ── Node definitions: (col_x, row_y, label, sublabel, color) ─────────
    nodes = [
        # Admission
        (0.10, 0.83, "Patient\nPresents", "Dengue suspected", BLUE),
        (0.10, 0.73, "Serology\nInput", "NS1 · IgM · IgG\nPrimary/Secondary", BLUE_LT),
        (0.10, 0.63, "CBC + Vitals\nEntry", "Platelets · Hb · Hct\nBP · HR · Temp", BLUE),
        (0.10, 0.53, "Extended\nPanels", "LFT · Coag · Renal\nImaging findings", BLUE_LT),
        (0.10, 0.43, "OCR Upload\n(Optional)", "Auto-extract\nConfidence score", BLUE_LT),
        (0.10, 0.33, "Illness Day\nTracking", "Onset date →\nPhase engine", BLUE),

        # AI Assessment
        (0.30, 0.83, "Risk\nProbability", "RF 21-feat\nAUC=0.9996", GREEN),
        (0.30, 0.73, "95% CI\nCalibration", "Tree ensemble\nWide CI → flag", GREEN),
        (0.30, 0.63, "OOD\nDetection", "Mahalanobis D\nThreshold=2.5", AMBER),
        (0.30, 0.53, "Severity\nScore", "0–100 composite\n4 domains", GREEN),
        (0.30, 0.43, "24h Platelet\nForecast", "GBR · R²=0.9953\nMAE=2,515", GREEN),
        (0.30, 0.33, "Plasma\nLeakage %", "Multi-marker\n0–100%", AMBER),

        # WHO Triage
        (0.50, 0.83, "Group A\n(No Warning)", "Outpatient\nOral fluids", GREEN),
        (0.50, 0.73, "Group B\n(Warning Signs)", "Admission\n5–7 mL/kg/hr IV", AMBER),
        (0.50, 0.63, "Group C\n(Severe)", "ICU\n10–20 mL/kg bolus", RED),
        (0.50, 0.50, "Organ\nImpairment?", "AST≥1000 · INR≥1.5\nEffusion · Ascites≥2", RED),
        (0.50, 0.38, "Dengue\nPhase", "Febrile → Critical\n→ Recovery", AMBER),
        (0.50, 0.27, "Bleeding\nRisk Score", "BRS 0–10\nTransfusion guide", RED),

        # Monitoring
        (0.70, 0.83, "Serial CBC\nAlerts", "Platelet velocity\n>40k/24h", RED),
        (0.70, 0.73, "Haemodynamic\nAlerts", "SI rise · MAP fall\nPulse pressure", RED),
        (0.70, 0.63, "AKI\nDetection", "Creatinine +0.3\nKDIGO KDIGO", AMBER),
        (0.70, 0.53, "Trajectory\nCountdowns", "Time-to-PLT<50k\nTime-to-SI≥0.9", AMBER),
        (0.70, 0.43, "FORS\nEngine", "Post-resus\nFluid overload", PURPLE),
        (0.70, 0.33, "Coagulopathy\nAlerts", "INR+PLT pattern\nDIC risk", RED),

        # Outcome
        (0.90, 0.83, "Discharge\nChecklist", "7 WHO criteria\nAll must pass", PURPLE),
        (0.90, 0.73, "Afebrile\n≥48 hrs", "Fever-free\nhours input", GREEN),
        (0.90, 0.63, "PLT ≥50k\n& Rising", "Trend + absolute\nthreshold", GREEN),
        (0.90, 0.53, "Haemo-\nStable", "SI<0.9 · MAP≥65\nPP>20 mmHg", GREEN),
        (0.90, 0.43, "PDF Report\nGenerated", "fpdf · A4\nFull clinical", BLUE),
        (0.90, 0.33, "Follow-up\nCBC 48h", "Discharge\ninstruction", GREEN),
    ]

    for nx, ny, nl, ns, nc in nodes:
        _box(ax, nx, ny, W, H, nl, ns, nc, fontsize=8)

    # ── Vertical arrows within columns ────────────────────────────────────
    col_xs = [0.10, 0.30, 0.50, 0.70, 0.90]
    col_ys = {
        0.10: [0.83, 0.73, 0.63, 0.53, 0.43, 0.33],
        0.30: [0.83, 0.73, 0.63, 0.53, 0.43, 0.33],
        0.50: [0.83, 0.73, 0.63, 0.50, 0.38, 0.27],
        0.70: [0.83, 0.73, 0.63, 0.53, 0.43, 0.33],
        0.90: [0.83, 0.73, 0.63, 0.53, 0.43, 0.33],
    }
    for cx, ys in col_ys.items():
        for i in range(len(ys)-1):
            _arrow(ax, cx, ys[i] - H/2,
                   cx, ys[i+1] + H/2 + 0.002, GREY, lw=0.9)

    # ── Key horizontal flows ──────────────────────────────────────────────
    cross_arrows = [
        (0.10+W/2, 0.83, 0.30-W/2, 0.83, GREEN,  "CBC + vitals"),
        (0.30+W/2, 0.83, 0.50-W/2, 0.78, AMBER,  "risk prob"),
        (0.50+W/2, 0.73, 0.70-W/2, 0.73, RED,    "Group B/C"),
        (0.70+W/2, 0.53, 0.90-W/2, 0.53, PURPLE, "stable?"),
        (0.30+W/2, 0.43, 0.50-W/2, 0.38, AMBER,  "trend"),
        (0.50+W/2, 0.38, 0.70-W/2, 0.43, RED,    "critical phase"),
    ]
    for x0, y0, x1, y1, c, lbl in cross_arrows:
        _arrow(ax, x0, y0, x1, y1, c, lw=1.3, label=lbl, label_color=c)

    # ── Phase band on right ───────────────────────────────────────────────
    phase_bands = [
        (0.72, 0.78, "Day 1–3\nFebrile", AMBER),
        (0.68, 0.72, "Day 4–6\nCritical", RED),
        (0.66, 0.61, "Day 7–10\nRecovery", GREEN),
    ]
    for py, py2, pl, pc in phase_bands:
        pass  # included in monitoring nodes above

    # ── Bottom annotation ─────────────────────────────────────────────────
    ax.text(0.5, 0.03,
            "All modules operate on the latest report entry  ·  "
            "Serial analysis requires ≥2 saved reports  ·  "
            "WHO classification + serology assessment active without CBC  ·  "
            "PDF export includes all active modules",
            ha='center', fontsize=8, color=GREY)

    # ── Risk level legend ─────────────────────────────────────────────────
    legend = [
        (GREEN,   "Low Risk / Positive Signal"),
        (AMBER,   "Monitor / Moderate"),
        (RED,     "Critical / Severe"),
        (PURPLE,  "Discharge / Output"),
        (BLUE_LT, "Optional / Extended"),
    ]
    for i, (c, lbl) in enumerate(legend):
        lx = 0.01 + i * 0.195
        ax.add_patch(plt.Rectangle((lx, 0.005), 0.015, 0.018,
                     facecolor=c, alpha=0.8))
        ax.text(lx + 0.018, 0.014, lbl, fontsize=7.5, color=WHITE, va='center')

    fig.savefig(f"{OUTPUT_DIR}/clinical_workflow.png")
    plt.close(fig)
    print("  ✅  clinical_workflow.png saved")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  Dengue CDSS — Architecture & Pipeline Visualizer")
    print("=" * 65)
    print(f"\n  Output directory: {os.path.abspath(OUTPUT_DIR)}\n")
    plot_system_architecture()
    plot_data_pipeline()
    plot_clinical_workflow()
    print("\n  All visualizations complete.")
    print("=" * 65)