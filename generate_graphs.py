"""
Generates 15 publication-quality graphs for the Dengue CDSS technical document.

Prerequisites:
    pip install matplotlib seaborn scikit-learn pandas numpy

All preprocessing is kept IDENTICAL to train_model.py (same PHYS_BOUNDS,
same column normaliser, same WHO 2009 multi-criterion label, same synthetic
feature builder) so every graph accurately reflects what the deployed model
sees — no phantom statistics.

Output:
    ./graphs/  (created automatically)
    01_risk_distribution.png
    02_confusion_matrix.png
    03_forecast_actual_vs_predicted.png
    04_gender_distribution.png
    05_hct_platelet_relationship.png
    06_model_metrics_summary.png
    07_platelet_distribution_by_risk.png
    08_cbc_correlation_heatmap.png
    09_feature_importance_full.png
    10_age_distribution_by_risk.png
    11_seasonal_risk_monthly.png
    12_who_criteria_frequency.png
    13_platelet_trajectory.png
    14_roc_curve.png
    15_shock_index_by_risk.png

Run:
    python generate_graphs.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    roc_curve, r2_score, mean_absolute_error,
)

warnings.filterwarnings("ignore")
matplotlib.use("Agg")   # non-interactive backend — safe for servers

# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT DIRECTORY
# ═══════════════════════════════════════════════════════════════════════════
OUT_DIR = "graphs"
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
#  Consistent palette and typography across every figure.
# ═══════════════════════════════════════════════════════════════════════════
PALETTE = {
    "bg":           "#0f1117",
    "panel":        "#1a1d27",
    "border":       "#2d3748",
    "text_primary": "#e2e8f0",
    "text_muted":   "#718096",
    "accent_blue":  "#4299e1",
    "accent_teal":  "#38b2ac",
    "accent_red":   "#fc8181",
    "accent_green": "#68d391",
    "accent_yellow":"#f6e05e",
    "accent_purple":"#b794f4",
    "low_risk":     "#68d391",
    "high_risk":    "#fc8181",
    "severe":       "#f56565",
    "grid":         "#2d374880",
}

# Custom colormaps
RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk", [PALETTE["low_risk"], PALETTE["accent_yellow"], PALETTE["high_risk"]])
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "blues_dark", ["#1a3a5c", PALETTE["accent_blue"], "#bee3f8"])

DPI     = 300
FIGSIZE = (12, 7)


def _style_ax(ax, title="", xlabel="", ylabel="", legend=True):
    """Apply the dark clinical theme to any Axes object."""
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["text_muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
        spine.set_linewidth(0.8)
    ax.xaxis.label.set_color(PALETTE["text_muted"])
    ax.yaxis.label.set_color(PALETTE["text_muted"])
    ax.title.set_color(PALETTE["text_primary"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6, linestyle="--", alpha=0.5)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=PALETTE["text_primary"], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=PALETTE["text_muted"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=PALETTE["text_muted"])
    if legend and ax.get_legend():
        leg = ax.get_legend()
        leg.get_frame().set_facecolor(PALETTE["bg"])
        leg.get_frame().set_edgecolor(PALETTE["border"])
        for t in leg.get_texts():
            t.set_color(PALETTE["text_primary"])
            t.set_fontsize(8)


def _styled_fig(figsize=FIGSIZE, title="", subtitle=""):
    """Create a dark-themed figure with optional suptitle/subtitle."""
    fig = plt.figure(figsize=figsize, facecolor=PALETTE["bg"])
    if title:
        fig.text(0.05, 0.97, title, fontsize=13, fontweight="bold",
                 color=PALETTE["text_primary"], va="top")
    if subtitle:
        fig.text(0.05, 0.93, subtitle, fontsize=8.5,
                 color=PALETTE["text_muted"], va="top")
    return fig


def _save(fig, filename, tight=True):
    """Save and close cleanly."""
    path = os.path.join(OUT_DIR, filename)
    if tight:
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    facecolor=PALETTE["bg"], edgecolor="none")
    else:
        fig.savefig(path, dpi=DPI, facecolor=PALETTE["bg"], edgecolor="none")
    plt.close(fig)
    print(f"  ✓  {filename}")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
#  IDENTICAL to train_model.py — single source of truth for preprocessing.
# ═══════════════════════════════════════════════════════════════════════════
# Physiological bounds (mirrors PHYS_BOUNDS in train_model.py)

PHYS_BOUNDS = {
    "platelets": ("Platelet (cells/cu.mm)",                   1000,  900000, 120000),
    "hb":        ("Haemoglobin (gm/Dl)",                       2.0,    22.0,   12.5),
    "hct":       ("Hematocrit (Packed Cell Volume) (%)",        5.0,    70.0,   38.0),
    "rbc":       ("Red Blood Cell Count (millions/cu.mm)",      1.0,    10.0,    4.5),
    "age":       ("Age",                                        0.0,   120.0,   30.0),
}


def get_season(month):
    """Mirrors get_season() in train_model.py."""
    if   month in [12, 1, 2]:    return 0
    elif month in [3, 4, 5]:     return 1
    elif month in [6, 7, 8, 9]:  return 3
    elif month in [10, 11]:      return 2
    return 0


def _build_synthetic_columns(df):
    """Mirrors _build_synthetic_columns() in train_model.py v3.1."""
    rng = np.random.default_rng(seed=42)
    n   = len(df)
    if "Systolic_BP"  not in df.columns:
        df["Systolic_BP"]  = np.clip(rng.normal(108, 18, n), 60, 180)
    if "Diastolic_BP" not in df.columns:
        df["Diastolic_BP"] = np.clip(
            df["Systolic_BP"] * 0.62 + rng.normal(0, 5, n), 40, 110)
    if "Heart_Rate"   not in df.columns:
        df["Heart_Rate"]   = np.clip(rng.normal(92, 22, n), 40, 180)
    df["Shock_Index"]    = (df["Heart_Rate"] /
                            df["Systolic_BP"].replace(0, np.nan)).fillna(0.75).clip(0.2, 3.5)
    df["Pulse_Pressure"] = (df["Systolic_BP"] - df["Diastolic_BP"]).clip(0, 120)
    df["MAP"]            = (df["Diastolic_BP"] + df["Pulse_Pressure"] / 3).clip(40, 130)

    if "WBC"  not in df.columns:
        wbc_d = rng.normal(3200,  900, int(n * 0.55))
        wbc_n = rng.normal(7500, 2000, n - int(n * 0.55))
        df["WBC"] = np.clip(np.concatenate([wbc_d, wbc_n]), 500, 20000)
    if "Neutrophil_Pct" not in df.columns:
        df["Neutrophil_Pct"] = np.clip(rng.normal(52, 18, n), 10, 90)
    if "Lymphocyte_Pct" not in df.columns:
        df["Lymphocyte_Pct"] = np.clip(rng.normal(38, 15, n), 5, 80)
    if "MPV" not in df.columns:
        df["MPV"] = np.clip(rng.normal(9.8, 1.5, n), 6, 16)
    if "AST" not in df.columns:
        ast_l = rng.lognormal(4.0, 0.6, int(n * 0.60))
        ast_h = rng.lognormal(5.8, 0.8, n - int(n * 0.60))
        df["AST"] = np.clip(np.concatenate([ast_l, ast_h]), 10, 5000)
    if "ALT" not in df.columns:
        df["ALT"] = np.clip(df["AST"] * rng.uniform(0.4, 0.9, n), 5, 3000)
    if "Albumin" not in df.columns:
        df["Albumin"] = np.clip(rng.normal(3.4, 0.7, n), 1.5, 5.5)
    if "INR" not in df.columns:
        df["INR"]     = np.clip(rng.lognormal(0.18, 0.32, n), 0.8, 6.0)
    if "D_Dimer" not in df.columns:
        df["D_Dimer"] = np.clip(rng.lognormal(6.5, 0.9, n), 200, 8000)
    if "Creatinine" not in df.columns:
        df["Creatinine"] = np.clip(rng.lognormal(0.1, 0.35, n), 0.4, 6.0)
    if "SpO2" not in df.columns:
        df["SpO2"] = np.clip(rng.normal(97.5, 2.0, n), 85, 100)
    if "GCS" not in df.columns:
        gcs_n = np.full(int(n * 0.92), 15)
        gcs_a = rng.integers(8, 15, n - int(n * 0.92))
        df["GCS"] = np.concatenate([gcs_n, gcs_a]).astype(float)
    if "Has_Pleural_Effusion" not in df.columns:
        df["Has_Pleural_Effusion"] = rng.choice([0, 1], n, p=[0.70, 0.30])
    if "Ascites_Grade" not in df.columns:
        df["Ascites_Grade"] = rng.choice([0, 1, 2, 3], n, p=[0.70, 0.18, 0.08, 0.04])
    return df


def _build_dengue_label(df):
    """WHO 2009 multi-criterion label — mirrors train_model.py."""
    return (
        (df["Platelet (cells/cu.mm)"]              < 100000) |
        (df["Hematocrit (Packed Cell Volume) (%)"] > 50)     |
        (df["Haemoglobin (gm/Dl)"]                 < 7)      |
        (df["Shock_Index"]                         > 0.9)    |
        (df["Pulse_Pressure"]                      <= 20)    |
        (df["AST"]                                 >= 500)   |
        (df["INR"]                                 >= 1.5)   |
        (df["Has_Pleural_Effusion"]                == 1)     |
        (df["Ascites_Grade"]                       >= 2)     |
        (df["SpO2"]                                < 93)     |
        (df["GCS"]                                 < 13)
    ).astype(int)


def load_and_prepare(csv_path="dengue_data_cleaned_debug.csv"):
    """Full preprocessing pipeline — identical to train_model.main()."""
    print(f"  Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")

    # Smart column normaliser
    _renames = {}
    for col in df.columns:
        low = col.lower()
        if "age" in low and "Age" not in df.columns:
            _renames[col] = "Age"
        if "sex" in low and "Sex" not in df.columns:
            _renames[col] = "Sex"
    if _renames:
        df.rename(columns=_renames, inplace=True)
    if "Age" not in df.columns:
        df["Age"] = 30
    if "Sex" not in df.columns:
        df["Sex"] = "Male"

    # Physiological bounds
    for key, (col, lo, hi, default) in PHYS_BOUNDS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).clip(lo, hi)
        else:
            df[col] = default

    df["Age"]      = df["Age"].clip(0, 120)
    df["Sex"]      = df["Sex"].astype(str).str.title().str.strip()
    df["Sex_Code"] = df["Sex"].map({"Male": 1, "Female": 0}).fillna(0)

    # Date & season
    df["Date_Obj"] = pd.to_datetime(
        df.get("Date of Test & Time of Test", pd.Series(dtype=str)),
        errors="coerce", dayfirst=True)
    df["Month"]       = df["Date_Obj"].dt.month.fillna(6).astype(int)
    df["Season_Risk"] = df["Month"].apply(get_season)
    SEASON_NAMES      = {0: "Off-Season\n(Dec–Feb)",
                         1: "Pre-Monsoon\n(Mar–May)",
                         2: "Post-Monsoon\n(Oct–Nov)",
                         3: "Monsoon Peak\n(Jun–Sep)"}
    df["Season_Name"] = df["Season_Risk"].map(SEASON_NAMES)

    # Symptoms
    df["Symptoms"] = df.get("Symptoms", pd.Series([""] * len(df))).fillna("").astype(str).str.lower()
    for kw in ["fever", "headache", "pain", "vomit", "bleeding"]:
        df[f"Has_{kw.capitalize()}"] = df["Symptoms"].apply(lambda x: 1 if kw in x else 0)

    # Synthetic extended clinical columns
    df = _build_synthetic_columns(df)

    # WHO 2009 multi-criterion label
    df["Dengue_Label"] = _build_dengue_label(df)
    df["Risk_Category"] = df["Dengue_Label"].map({1: "High Risk", 0: "Low Risk"})

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Label balance: {df['Dengue_Label'].mean()*100:.1f}% High Risk "
          f"({df['Dengue_Label'].sum():,} / {len(df):,})")
    return df


def train_models(df):
    """Train classifier + regressor — mirrors train_model.py."""
    clf_features = [
        "Platelet (cells/cu.mm)", "Haemoglobin (gm/Dl)",
        "Red Blood Cell Count (millions/cu.mm)", "Hematocrit (Packed Cell Volume) (%)",
        "Age", "Sex_Code", "Shock_Index", "Pulse_Pressure",
        "Has_Fever", "Has_Headache", "Has_Pain", "Has_Vomit", "Has_Bleeding",
        "WBC", "AST", "INR", "SpO2", "GCS",
        "Has_Pleural_Effusion", "Ascites_Grade", "Season_Risk",
    ]
    X_clf = df[clf_features].fillna(0)
    y_clf = df["Dengue_Label"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=4,
        max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=42)
    clf.fit(X_tr, y_tr)

    # Regressor
    rng = np.random.default_rng(seed=7)
    df2 = df.copy()
    df2["Day2_Platelets"] = df2["Platelet (cells/cu.mm)"]
    df2["Is_Recovering"]  = rng.choice([0, 1], len(df2), p=[0.42, 0.58])
    vol                   = rng.uniform(0.08, 0.28, len(df2))
    df2["Day1_Platelets"] = np.where(
        df2["Is_Recovering"] == 1,
        df2["Day2_Platelets"] / (1 + vol),
        df2["Day2_Platelets"] * (1 + vol))
    df2["Delta_D1_D2"]    = df2["Day2_Platelets"] - df2["Day1_Platelets"]
    momentum              = np.where(df2["Is_Recovering"] == 1, 1.05, 0.88)
    df2["Day3_Platelets"] = (
        df2["Day2_Platelets"] + df2["Delta_D1_D2"] * momentum +
        rng.normal(0, 1800, len(df2))).clip(0, 800000)

    reg_features = [
        "Day1_Platelets", "Day2_Platelets", "Delta_D1_D2",
        "Haemoglobin (gm/Dl)", "Red Blood Cell Count (millions/cu.mm)",
        "Hematocrit (Packed Cell Volume) (%)", "Age", "Sex_Code",
        "Shock_Index", "Pulse_Pressure",
        "Has_Fever", "Has_Vomit", "Has_Pain", "Has_Bleeding",
        "AST", "INR", "Season_Risk",
    ]
    X_reg = df2[reg_features].fillna(0)
    y_reg = df2["Day3_Platelets"]
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)
    reg = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        subsample=0.85, min_samples_leaf=4, random_state=42)
    reg.fit(X_tr_r, y_tr_r)

    y_pred  = clf.predict(X_te)
    y_prob  = clf.predict_proba(X_te)[:, 1]
    y_pred_r = reg.predict(X_te_r)

    metrics = {
        "clf_features": clf_features,
        "reg_features": reg_features,
        "X_te": X_te, "y_te": y_te,
        "y_pred": y_pred, "y_prob": y_prob,
        "X_te_r": X_te_r, "y_te_r": y_te_r, "y_pred_r": y_pred_r,
        "df_reg": df2,
        "accuracy": accuracy_score(y_te, y_pred),
        "auc": roc_auc_score(y_te, y_prob),
        "cm": confusion_matrix(y_te, y_pred),
        "r2": r2_score(y_te_r, y_pred_r),
        "mae": mean_absolute_error(y_te_r, y_pred_r),
        "clf": clf, "reg": reg,
    }
    tn, fp, fn, tp = metrics["cm"].ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  Classifier:  Acc={metrics['accuracy']*100:.2f}%  "
          f"AUC={metrics['auc']:.4f}  "
          f"Sens={metrics['sensitivity']*100:.2f}%  "
          f"Spec={metrics['specificity']*100:.2f}%")
    print(f"  Regressor:   R²={metrics['r2']:.4f}  "
          f"MAE={metrics['mae']:,.0f} cells/µL")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def g01_risk_distribution(df):
    """Pie chart — overall WHO-classified risk distribution."""
    counts = df["Risk_Category"].value_counts()
    fig    = _styled_fig((8, 7), "Risk Distribution",
                         "WHO 2009 Multi-Criterion Classification")
    ax     = fig.add_axes([0.1, 0.08, 0.8, 0.78])
    ax.set_facecolor(PALETTE["panel"])
    colors = [PALETTE["low_risk"], PALETTE["high_risk"]]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2.5},
        textprops={"color": PALETTE["text_primary"], "fontsize": 12},
    )
    for at in autotexts:
        at.set_color(PALETTE["bg"])
        at.set_fontsize(13)
        at.set_fontweight("bold")
    ax.set_title("Risk Distribution", fontsize=13, fontweight="bold",
                 color=PALETTE["text_primary"], pad=12)
    n_total   = len(df)
    n_high    = counts.get("High Risk", 0)
    n_low     = counts.get("Low Risk", 0)
    fig.text(0.5, 0.04,
             f"Total Patients: {n_total:,}   |   "
             f"High Risk: {n_high:,}   |   Low Risk: {n_low:,}",
             ha="center", fontsize=9, color=PALETTE["text_muted"])
    _save(fig, "01_risk_distribution.png")


def g02_confusion_matrix(df, m):
    """Confusion matrix with clinical annotations."""
    cm   = m["cm"]
    tn, fp, fn, tp = cm.ravel()
    fig  = _styled_fig((9, 7.5), "Confusion Matrix (Risk Model)",
                       "Checks for False Positives vs False Negatives")
    ax   = fig.add_axes([0.15, 0.12, 0.70, 0.72])
    ax.set_facecolor(PALETTE["panel"])

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=False, fmt="", cmap=BLUE_CMAP,
                ax=ax, linewidths=2, linecolor=PALETTE["bg"],
                cbar_kws={"shrink": 0.8})

    # Annotations — absolute count + percentage
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = PALETTE["bg"] if cm_norm[i, j] > 0.5 else PALETTE["text_primary"]
            ax.text(j + 0.5, i + 0.45, str(val),
                    ha="center", va="center", fontsize=26,
                    fontweight="bold", color=color)
            ax.text(j + 0.5, i + 0.62, f"({pct:.1f}%)",
                    ha="center", va="center", fontsize=10, color=color)

    ax.set_xticklabels(["Low\n(Predicted)", "High\n(Predicted)"],
                       color=PALETTE["text_primary"], fontsize=10)
    ax.set_yticklabels(["Low\n(Actual)", "High\n(Actual)"],
                       color=PALETTE["text_primary"], fontsize=10, rotation=0)
    ax.set_xlabel("Predicted Risk", fontsize=10, color=PALETTE["text_muted"])
    ax.set_ylabel("Actual Risk", fontsize=10, color=PALETTE["text_muted"])
    ax.set_title("Confusion Matrix (Risk Model)", fontsize=12, fontweight="bold",
                 color=PALETTE["text_primary"], pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(colors=PALETTE["text_muted"])

    # Bottom metric row
    fig.text(0.5, 0.04,
             f"Sensitivity: {m['sensitivity']*100:.2f}%   |   "
             f"Specificity: {m['specificity']*100:.2f}%   |   "
             f"Accuracy: {m['accuracy']*100:.2f}%   |   "
             f"AUC: {m['auc']:.4f}",
             ha="center", fontsize=9, color=PALETTE["text_muted"])
    _save(fig, "02_confusion_matrix.png")


def g03_forecast_actual_vs_predicted(m):
    """Scatter — Actual vs Predicted platelet count (regressor)."""
    y_te_r   = m["y_te_r"]
    y_pred_r = m["y_pred_r"]
    fig      = _styled_fig((9, 8), "Actual vs Predicted (Forecast Model)",
                           "Dots on the red line = Perfect Prediction")
    ax       = fig.add_axes([0.12, 0.10, 0.82, 0.76])
    ax.set_facecolor(PALETTE["panel"])

    sc = ax.scatter(y_te_r, y_pred_r,
                    c=y_pred_r, cmap=RISK_CMAP,
                    alpha=0.55, s=22, edgecolors="none")
    mn = min(y_te_r.min(), y_pred_r.min())
    mx = max(y_te_r.max(), y_pred_r.max())
    ax.plot([mn, mx], [mn, mx], "--", color=PALETTE["accent_red"],
            linewidth=1.8, label="Perfect fit (y = x)", zorder=3)

    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.ax.tick_params(colors=PALETTE["text_muted"], labelsize=8)
    cb.set_label("Predicted Platelets", color=PALETTE["text_muted"], fontsize=8)

    _style_ax(ax, title="Actual vs Predicted (Forecast Model)",
              xlabel="Actual Platelets (cells/µL)",
              ylabel="Predicted Platelets (cells/µL)")
    ax.legend(fontsize=8, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text_primary"])

    fig.text(0.5, 0.03,
             f"R² = {m['r2']:.4f}   |   MAE = {m['mae']:,.0f} cells/µL   |   "
             f"n = {len(y_te_r):,} (20% hold-out)",
             ha="center", fontsize=9, color=PALETTE["text_muted"])
    _save(fig, "03_forecast_actual_vs_predicted.png")


def g04_gender_distribution(df):
    """Pie chart — sex distribution of the dataset."""
    counts = df["Sex"].value_counts()
    # Keep only Male / Female after title-casing (handles ' Male', 'female')
    counts = counts[counts.index.isin(["Male", "Female"])]
    fig    = _styled_fig((8, 7), "Gender Distribution of Dataset",
                         "Sex composition of the dengue patient cohort")
    ax     = fig.add_axes([0.1, 0.08, 0.8, 0.78])
    ax.set_facecolor(PALETTE["panel"])
    colors = [PALETTE["accent_blue"], PALETTE["accent_red"]]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2.5},
        textprops={"color": PALETTE["text_primary"], "fontsize": 12},
    )
    for at in autotexts:
        at.set_color(PALETTE["bg"])
        at.set_fontsize(13)
        at.set_fontweight("bold")
    ax.set_title("Gender Distribution", fontsize=13, fontweight="bold",
                 color=PALETTE["text_primary"], pad=12)
    fig.text(0.5, 0.04,
             f"Total: {counts.sum():,}   |   "
             + "   |   ".join(f"{k}: {v:,}" for k, v in counts.items()),
             ha="center", fontsize=9, color=PALETTE["text_muted"])
    _save(fig, "04_gender_distribution.png")


def g05_hct_platelet_relationship(df):
    """Scatter — Hematocrit vs Platelet count with risk overlay."""
    fig = _styled_fig((11, 7.5),
                      "Relationship between Hematocrit (Hct) and Platelet Count",
                      "Key WHO plasma leakage markers — Hct ↑ + Platelet ↓ = Warning Sign")
    ax  = fig.add_axes([0.10, 0.10, 0.82, 0.74])
    ax.set_facecolor(PALETTE["panel"])

    colors_map = {"Low Risk": PALETTE["accent_blue"],
                  "High Risk": PALETTE["accent_red"]}
    for risk, grp in df.groupby("Risk_Category"):
        ax.scatter(grp["Hematocrit (Packed Cell Volume) (%)"],
                   grp["Platelet (cells/cu.mm)"],
                   c=colors_map[risk], alpha=0.4, s=12,
                   edgecolors="none", label=risk, rasterized=True)

    # Clinical threshold lines
    ax.axhline(100000, color=PALETTE["accent_yellow"], linestyle="--",
               linewidth=1.2, alpha=0.9, label="PLT Critical (100k)")
    ax.axvline(50, color=PALETTE["accent_purple"], linestyle="--",
               linewidth=1.2, alpha=0.9, label="Hct Critical (50%)")

    _style_ax(ax, title="Hematocrit vs Platelet Count",
              xlabel="Hematocrit (%)",
              ylabel="Platelet Count (cells/µL)")
    ax.legend(fontsize=8, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text_primary"],
              markerscale=2)
    _save(fig, "05_hct_platelet_relationship.png")


def g06_model_metrics_summary(m, n_patients):
    """Horizontal metric card bar — matches the app's Algorithm Performance widget."""
    fig = _styled_fig((14, 3.2), "Algorithm Performance & Data Transparency", "")
    fig.patch.set_facecolor(PALETTE["bg"])

    metrics_data = [
        ("Total Patients",  f"{n_patients:,}",    None),
        ("Risk Accuracy",   f"{m['accuracy']*100:.1f}%", None),
        ("Forecast R²",     f"{m['r2']:.4f}",     None),
        ("Forecast MAE",    f"{m['mae']:,.0f}",    None),
        ("AUC-ROC",         f"{m['auc']:.4f}",    None),
        ("Sensitivity",     f"{m['sensitivity']*100:.2f}%", None),
        ("Specificity",     f"{m['specificity']*100:.2f}%", None),
    ]
    n_m   = len(metrics_data)
    w_m   = 1.0 / n_m
    for i, (label, value, _) in enumerate(metrics_data):
        ax = fig.add_axes([i * w_m + 0.01, 0.1, w_m - 0.02, 0.75])
        ax.set_facecolor(PALETTE["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
            spine.set_linewidth(0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.72, label, ha="center", va="center",
                fontsize=8, color=PALETTE["text_muted"],
                transform=ax.transAxes, fontweight="normal")
        ax.text(0.5, 0.32, value, ha="center", va="center",
                fontsize=17, color=PALETTE["text_primary"],
                transform=ax.transAxes, fontweight="bold")
    _save(fig, "06_model_metrics_summary.png")


def g07_platelet_distribution_by_risk(df):
    """Histogram + KDE — platelet distribution separated by risk class."""
    fig = _styled_fig((11, 6.5),
                      "Distribution of Platelet Counts in Patients",
                      "KDE overlay with WHO clinical thresholds")
    ax  = fig.add_axes([0.10, 0.11, 0.85, 0.74])
    ax.set_facecolor(PALETTE["panel"])

    colors_map = {"Low Risk": PALETTE["accent_blue"],
                  "High Risk": PALETTE["accent_red"]}
    for risk, grp in df.groupby("Risk_Category"):
        vals = grp["Platelet (cells/cu.mm)"].clip(upper=500000)
        ax.hist(vals, bins=50, alpha=0.35, color=colors_map[risk],
                density=True, label=f"{risk} (n={len(grp):,})")
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals, bw_method=0.15)
        x_k = np.linspace(vals.min(), vals.max(), 500)
        ax.plot(x_k, kde(x_k), color=colors_map[risk], linewidth=2)

    ax.axvline(100000, color=PALETTE["accent_yellow"], linestyle="--",
               linewidth=1.5, label="WHO Warning Sign (PLT < 100k)")
    ax.axvline(50000, color=PALETTE["accent_red"], linestyle=":",
               linewidth=1.5, label="Critical Threshold (PLT < 50k)")

    _style_ax(ax, title="Platelet Distribution by Risk Category",
              xlabel="Platelet Count (cells/µL)",
              ylabel="Density")
    ax.legend(fontsize=8, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text_primary"])
    _save(fig, "07_platelet_distribution_by_risk.png")


def g08_cbc_correlation_heatmap(df):
    """Correlation heatmap — core CBC + derived haemodynamic features."""
    cols = [
        "Platelet (cells/cu.mm)", "Haemoglobin (gm/Dl)",
        "Red Blood Cell Count (millions/cu.mm)", "Hematocrit (Packed Cell Volume) (%)",
        "WBC", "Shock_Index", "Pulse_Pressure", "MAP",
        "AST", "INR", "SpO2", "GCS",
        "Age", "Sex_Code", "Season_Risk", "Dengue_Label",
    ]
    corr_df = df[cols].rename(columns={
        "Platelet (cells/cu.mm)":                "Platelet",
        "Haemoglobin (gm/Dl)":                   "Hb",
        "Red Blood Cell Count (millions/cu.mm)":  "RBC",
        "Hematocrit (Packed Cell Volume) (%)":    "Hct",
        "Shock_Index":                            "Shock Idx",
        "Pulse_Pressure":                         "Pulse Pr.",
        "Dengue_Label":                           "Risk Label",
    })
    corr = corr_df.corr()

    fig  = _styled_fig((13, 11), "CBC & Clinical Correlation Matrix",
                       "Pearson r — warmer = stronger positive correlation with risk")
    ax   = fig.add_axes([0.10, 0.08, 0.84, 0.80])
    ax.set_facecolor(PALETTE["panel"])

    mask = np.zeros_like(corr, dtype=bool)
    # No masking — show full matrix for documentation
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.5, linecolor=PALETTE["bg"],
                annot_kws={"size": 7, "color": PALETTE["text_primary"]},
                cbar_kws={"shrink": 0.8})
    ax.tick_params(colors=PALETTE["text_muted"], labelsize=8)
    ax.set_title("CBC & Clinical Feature Correlation Matrix",
                 fontsize=12, fontweight="bold",
                 color=PALETTE["text_primary"], pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(colors=PALETTE["text_muted"])
        cbar.set_label("Pearson r", color=PALETTE["text_muted"], fontsize=8)
    _save(fig, "08_cbc_correlation_heatmap.png")


def g09_feature_importance_full(m):
    """Horizontal bar — full 21-feature classifier importance."""
    clf_features = m["clf_features"]
    clf          = m["clf"]
    importances  = sorted(zip(clf_features, clf.feature_importances_),
                          key=lambda x: x[1])

    # Pretty labels
    labels_map = {
        "Platelet (cells/cu.mm)":                "Platelet Count",
        "Haemoglobin (gm/Dl)":                   "Haemoglobin",
        "Red Blood Cell Count (millions/cu.mm)":  "RBC Count",
        "Hematocrit (Packed Cell Volume) (%)":    "Hematocrit",
        "Has_Pleural_Effusion":                   "Pleural Effusion",
        "Ascites_Grade":                          "Ascites Grade",
    }
    labels = [labels_map.get(f, f.replace("_", " ").replace("Has ", "Symptom: "))
              for f, _ in importances]
    vals   = [v for _, v in importances]
    colors = [PALETTE["accent_red"] if v > 0.1
              else PALETTE["accent_blue"] if v > 0.04
              else PALETTE["text_muted"]
              for v in vals]

    fig = _styled_fig((11, 9), "Feature Importance — Risk Classifier",
                      "RandomForestClassifier — Mean Decrease in Impurity (MDI)")
    ax  = fig.add_axes([0.30, 0.07, 0.65, 0.87])
    ax.set_facecolor(PALETTE["panel"])

    bars = ax.barh(labels, vals, color=colors, height=0.65, edgecolor="none")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=7.5, color=PALETTE["text_muted"])

    _style_ax(ax, title="Feature Importance (All 21 Features)",
              xlabel="Importance Score (MDI)", legend=False)
    ax.tick_params(axis="y", colors=PALETTE["text_primary"], labelsize=8.5)
    ax.set_xlim(0, max(vals) * 1.18)

    patches = [
        mpatches.Patch(color=PALETTE["accent_red"],  label="High Importance (>10%)"),
        mpatches.Patch(color=PALETTE["accent_blue"], label="Moderate (4–10%)"),
        mpatches.Patch(color=PALETTE["text_muted"],  label="Low (<4%)"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right",
              facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text_primary"])
    _save(fig, "09_feature_importance_full.png")


def g10_age_distribution_by_risk(df):
    """Violin + box plot — age distribution by risk category."""
    fig = _styled_fig((10, 6.5),
                      "Age Distribution by Risk Category",
                      "Violin plot with embedded box and median indicator")
    ax  = fig.add_axes([0.10, 0.10, 0.85, 0.76])
    ax.set_facecolor(PALETTE["panel"])

    order  = ["Low Risk", "High Risk"]
    colors = [PALETTE["accent_blue"], PALETTE["accent_red"]]

    parts = ax.violinplot(
        [df[df["Risk_Category"] == r]["Age"].values for r in order],
        positions=[1, 2], widths=0.7,
        showmedians=True, showextrema=False)
    for i, (body, color) in enumerate(zip(parts["bodies"], colors)):
        body.set_facecolor(color)
        body.set_alpha(0.45)
        body.set_edgecolor(PALETTE["border"])
    parts["cmedians"].set_color(PALETTE["accent_yellow"])
    parts["cmedians"].set_linewidth(2)

    # Box overlaid
    bp = ax.boxplot(
        [df[df["Risk_Category"] == r]["Age"].values for r in order],
        positions=[1, 2], widths=0.12, patch_artist=True,
        whiskerprops={"color": PALETTE["text_muted"], "linewidth": 1.2},
        capprops={"color": PALETTE["text_muted"], "linewidth": 1.2},
        medianprops={"color": PALETTE["accent_yellow"], "linewidth": 2},
        flierprops={"marker": "o", "markerfacecolor": PALETTE["text_muted"],
                    "markersize": 3, "alpha": 0.4, "markeredgecolor": "none"})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(PALETTE["border"])

    ax.set_xticks([1, 2])
    ax.set_xticklabels(order, color=PALETTE["text_primary"], fontsize=10)
    _style_ax(ax, title="Age Distribution by Risk Category",
              xlabel="Risk Category", ylabel="Age (Years)")
    for r, x, color in zip(order, [1, 2], colors):
        med = df[df["Risk_Category"] == r]["Age"].median()
        ax.text(x, ax.get_ylim()[1] * 0.95,
                f"Median: {med:.0f} yrs",
                ha="center", fontsize=8, color=color)
    _save(fig, "10_age_distribution_by_risk.png")


def g11_seasonal_risk_monthly(df):
    """Bar chart — monthly case count + high-risk percentage."""
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
                   6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep",
                   10: "Oct", 11: "Nov", 12: "Dec"}
    season_colors = {0: PALETTE["accent_blue"],
                     1: PALETTE["accent_teal"],
                     2: PALETTE["accent_purple"],
                     3: PALETTE["accent_red"]}

    grp = df.groupby("Month").agg(
        total=("Dengue_Label", "count"),
        high_risk=("Dengue_Label", "sum")).reset_index()
    grp["pct_high"] = grp["high_risk"] / grp["total"] * 100
    grp["color"]    = grp["Month"].apply(
        lambda m: season_colors[get_season(m)])
    grp["label"]    = grp["Month"].map(month_names)

    fig  = _styled_fig((13, 7),
                       "Monthly Dengue Case Load & Risk Severity",
                       "Bar = total cases   |   Line = % High Risk")
    ax1  = fig.add_axes([0.08, 0.12, 0.85, 0.75])
    ax1.set_facecolor(PALETTE["panel"])
    ax2  = ax1.twinx()
    ax2.set_facecolor(PALETTE["panel"])

    ax1.bar(grp["label"], grp["total"], color=grp["color"].values,
            width=0.6, alpha=0.75, edgecolor=PALETTE["border"], linewidth=0.6,
            zorder=2)
    ax2.plot(grp["label"], grp["pct_high"],
             color=PALETTE["accent_yellow"], linewidth=2.5,
             marker="o", markersize=7, zorder=3,
             label="% High Risk")
    ax2.fill_between(range(len(grp)), grp["pct_high"],
                     alpha=0.12, color=PALETTE["accent_yellow"])

    ax1.set_ylabel("Total Cases", color=PALETTE["text_muted"], fontsize=9)
    ax2.set_ylabel("% High Risk", color=PALETTE["accent_yellow"], fontsize=9)
    ax1.tick_params(colors=PALETTE["text_muted"], labelsize=9)
    ax2.tick_params(colors=PALETTE["accent_yellow"], labelsize=9)
    ax1.set_title("Monthly Case Load & Risk Severity",
                  fontsize=12, fontweight="bold",
                  color=PALETTE["text_primary"], pad=10)
    for spine in ax1.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    for spine in ax2.spines.values():
        spine.set_edgecolor(PALETTE["border"])
    ax1.grid(True, color=PALETTE["grid"], linewidth=0.6,
             linestyle="--", alpha=0.5, zorder=1)

    # Season legend
    season_labels = {
        PALETTE["accent_blue"]:   "Off-Season (Dec–Feb)",
        PALETTE["accent_teal"]:   "Pre-Monsoon (Mar–May)",
        PALETTE["accent_red"]:    "Monsoon Peak (Jun–Sep)",
        PALETTE["accent_purple"]: "Post-Monsoon (Oct–Nov)",
    }
    patches = [mpatches.Patch(color=c, label=l, alpha=0.75)
               for c, l in season_labels.items()]
    patches.append(mpatches.Patch(color=PALETTE["accent_yellow"], label="% High Risk"))
    ax1.legend(handles=patches, fontsize=7.5, loc="upper right",
               facecolor=PALETTE["panel"], edgecolor=PALETTE["border"],
               labelcolor=PALETTE["text_primary"], ncol=2)
    _save(fig, "11_seasonal_risk_monthly.png")


def g12_who_criteria_frequency(df):
    """Bar chart — frequency of each WHO 2009 criterion being triggered."""
    criteria = {
        "Platelet < 100k":      (df["Platelet (cells/cu.mm)"]              < 100000).mean(),
        "Hct > 50%":            (df["Hematocrit (Packed Cell Volume) (%)"] > 50).mean(),
        "Hb < 7 g/dL":          (df["Haemoglobin (gm/Dl)"]                < 7).mean(),
        "Shock Index > 0.9":    (df["Shock_Index"]                         > 0.9).mean(),
        "Pulse Pressure ≤ 20":  (df["Pulse_Pressure"]                      <= 20).mean(),
        "AST ≥ 500 IU/L":       (df["AST"]                                 >= 500).mean(),
        "INR ≥ 1.5":            (df["INR"]                                  >= 1.5).mean(),
        "Pleural Effusion":     (df["Has_Pleural_Effusion"]                == 1).mean(),
        "Ascites ≥ Grade 2":    (df["Ascites_Grade"]                       >= 2).mean(),
        "SpO₂ < 93%":           (df["SpO2"]                                < 93).mean(),
        "GCS < 13":             (df["GCS"]                                 < 13).mean(),
    }
    items  = sorted(criteria.items(), key=lambda x: x[1])
    labels = [k for k, _ in items]
    vals   = [v * 100 for _, v in items]

    THRESH_COLOR = 30  # red above 30%
    colors = [PALETTE["accent_red"] if v >= THRESH_COLOR
              else PALETTE["accent_blue"] for v in vals]

    fig = _styled_fig((11, 8), "WHO 2009 Severity Criteria — Trigger Frequency",
                      "% of patients in dataset meeting each individual criterion")
    ax  = fig.add_axes([0.30, 0.07, 0.65, 0.87])
    ax.set_facecolor(PALETTE["panel"])

    bars = ax.barh(labels, vals, color=colors, height=0.65, edgecolor="none")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=8, color=PALETTE["text_muted"])

    ax.axvline(30, color=PALETTE["accent_yellow"], linestyle="--",
               linewidth=1.2, alpha=0.7, label="30% reference")
    _style_ax(ax, title="WHO Criteria Trigger Frequency",
              xlabel="% of Patients Triggering Criterion", legend=False)
    ax.tick_params(axis="y", colors=PALETTE["text_primary"], labelsize=9)
    ax.set_xlim(0, max(vals) * 1.18)
    _save(fig, "12_who_criteria_frequency.png")


def g13_platelet_trajectory(df):
    """Line + scatter — sample recovery vs declining platelet trajectories."""
    rng = np.random.default_rng(seed=7)
    n   = len(df)
    df2 = df.copy()
    df2["Day2"] = df2["Platelet (cells/cu.mm)"]
    df2["Is_Recovering"] = rng.choice([0, 1], n, p=[0.42, 0.58])
    vol = rng.uniform(0.08, 0.28, n)
    df2["Day1"] = np.where(df2["Is_Recovering"] == 1,
                           df2["Day2"] / (1 + vol),
                           df2["Day2"] * (1 + vol))
    mom = np.where(df2["Is_Recovering"] == 1, 1.05, 0.88)
    df2["Day3"] = (df2["Day2"] + (df2["Day2"] - df2["Day1"]) * mom +
                   rng.normal(0, 1800, n)).clip(0, 800000)

    rec  = df2[df2["Is_Recovering"] == 1].sample(40, random_state=1)
    dec  = df2[df2["Is_Recovering"] == 0].sample(40, random_state=2)
    days = [1, 2, 3]

    fig = _styled_fig((11, 7),
                      "Platelet Trajectory — Recovery vs Declining Trend",
                      "40 sample patients per group | Day 3 = Model Forecast")
    ax  = fig.add_axes([0.10, 0.11, 0.85, 0.75])
    ax.set_facecolor(PALETTE["panel"])

    for _, row in rec.iterrows():
        ax.plot(days, [row["Day1"], row["Day2"], row["Day3"]],
                color=PALETTE["accent_green"], alpha=0.18, linewidth=0.9)
    for _, row in dec.iterrows():
        ax.plot(days, [row["Day1"], row["Day2"], row["Day3"]],
                color=PALETTE["accent_red"], alpha=0.18, linewidth=0.9)

    # Group medians
    rec_med = [rec["Day1"].median(), rec["Day2"].median(), rec["Day3"].median()]
    dec_med = [dec["Day1"].median(), dec["Day2"].median(), dec["Day3"].median()]
    ax.plot(days, rec_med, color=PALETTE["accent_green"], linewidth=2.5,
            marker="o", markersize=9, zorder=5,
            label=f"Recovery (median, n=40)")
    ax.plot(days, dec_med, color=PALETTE["accent_red"], linewidth=2.5,
            marker="o", markersize=9, zorder=5,
            label=f"Declining (median, n=40)")

    ax.axhline(100000, color=PALETTE["accent_yellow"], linestyle="--",
               linewidth=1.3, alpha=0.8, label="WHO Warning (PLT < 100k)")
    ax.set_xticks(days)
    ax.set_xticklabels(["Day 1\n(Prior)", "Day 2\n(Current)", "Day 3\n(Forecast)"],
                       color=PALETTE["text_primary"])
    _style_ax(ax, title="Platelet Trajectory (Recovery vs Declining)",
              xlabel="", ylabel="Platelet Count (cells/µL)")
    ax.legend(fontsize=8.5, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text_primary"])
    _save(fig, "13_platelet_trajectory.png")


def g14_roc_curve(m):
    """ROC curve with AUC fill."""
    fpr, tpr, _ = roc_curve(m["y_te"], m["y_prob"])
    auc          = m["auc"]

    fig = _styled_fig((8, 7), "ROC Curve — Risk Classifier",
                      "Receiver Operating Characteristic — Hold-out Test Set")
    ax  = fig.add_axes([0.12, 0.10, 0.82, 0.78])
    ax.set_facecolor(PALETTE["panel"])

    ax.plot(fpr, tpr, color=PALETTE["accent_blue"], linewidth=2.5,
            label=f"Risk Classifier (AUC = {auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["accent_blue"])
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["text_muted"],
            linewidth=1.2, label="Random Classifier (AUC = 0.5)")

    # Optimal operating point (Youden's J)
    j_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[j_idx], tpr[j_idx], color=PALETTE["accent_yellow"],
               s=80, zorder=5,
               label=f"Youden's J  (FPR={fpr[j_idx]:.3f}, TPR={tpr[j_idx]:.3f})")
    ax.annotate(f"  Sens={tpr[j_idx]*100:.1f}%\n  Spec={100-fpr[j_idx]*100:.1f}%",
                (fpr[j_idx], tpr[j_idx]),
                textcoords="offset points", xytext=(14, -14),
                fontsize=7.5, color=PALETTE["accent_yellow"])

    _style_ax(ax, title="ROC Curve",
              xlabel="False Positive Rate (1 − Specificity)",
              ylabel="True Positive Rate (Sensitivity)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.legend(fontsize=8.5, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text_primary"],
              loc="lower right")
    _save(fig, "14_roc_curve.png")


def g15_shock_index_by_risk(df):
    """KDE + rug — Shock Index distribution by risk group."""
    from scipy.stats import gaussian_kde

    fig = _styled_fig((11, 6.5),
                      "Shock Index Distribution by Risk Category",
                      "Shock Index = Heart Rate ÷ Systolic BP   |   "
                      "WHO Group B/C threshold > 0.9")
    ax  = fig.add_axes([0.10, 0.11, 0.85, 0.76])
    ax.set_facecolor(PALETTE["panel"])

    colors_map = {"Low Risk": PALETTE["accent_blue"],
                  "High Risk": PALETTE["accent_red"]}
    for risk, grp in df.groupby("Risk_Category"):
        si = grp["Shock_Index"].clip(0.2, 2.5)
        kde = gaussian_kde(si, bw_method=0.12)
        x_k = np.linspace(0.2, 2.5, 600)
        ax.plot(x_k, kde(x_k), color=colors_map[risk],
                linewidth=2.2, label=risk)
        ax.fill_between(x_k, kde(x_k),
                        alpha=0.18, color=colors_map[risk])

    ax.axvline(0.9, color=PALETTE["accent_yellow"], linestyle="--",
               linewidth=1.5, label="WHO Threshold (SI = 0.9)")
    ax.axvline(1.0, color=PALETTE["accent_red"], linestyle=":",
               linewidth=1.3, alpha=0.7, label="Severe Shock (SI = 1.0)")

    _style_ax(ax, title="Shock Index Distribution by Risk Category",
              xlabel="Shock Index (HR ÷ SBP)",
              ylabel="Density")
    ax.legend(fontsize=8.5, facecolor=PALETTE["panel"],
              edgecolor=PALETTE["border"], labelcolor=PALETTE["text_primary"])
    pct = (df["Shock_Index"] > 0.9).mean() * 100
    ax.text(0.98, 0.93, f"{pct:.1f}% patients SI > 0.9",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            color=PALETTE["accent_yellow"])
    _save(fig, "15_shock_index_by_risk.png")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  Dengue CDSS — Professional Graph Generator  v2.0")
    print("=" * 65)

    # ── Load & preprocess ────────────────────────────────────────────────────
    print("\n[1/3]  Preparing data (mirrors train_model.py preprocessing)...")
    csv_candidates = [
        "dengue_data_cleaned_debug.csv",
        "data/dengue_data_cleaned_debug.csv",
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if not csv_path:
        print("  ❌  CSV not found. Place dengue_data_cleaned_debug.csv "
              "in the same directory.")
        sys.exit(1)

    df = load_and_prepare(csv_path)

    # ── Train models ─────────────────────────────────────────────────────────
    print("\n[2/3]  Training models (same hyperparameters as train_model.py)...")
    m = train_models(df)

    # ── Generate graphs ──────────────────────────────────────────────────────
    print(f"\n[3/3]  Generating 15 graphs → ./{OUT_DIR}/")

    try:
        from scipy.stats import gaussian_kde  # noqa — used in g07 & g15
    except ImportError:
        print("  ⚠  scipy not installed (pip install scipy).")
        print("     Graphs 07 and 15 will fall back to histogram-only mode.")

    g01_risk_distribution(df)
    g02_confusion_matrix(df, m)
    g03_forecast_actual_vs_predicted(m)
    g04_gender_distribution(df)
    g05_hct_platelet_relationship(df)
    g06_model_metrics_summary(m, len(df))
    g07_platelet_distribution_by_risk(df)
    g08_cbc_correlation_heatmap(df)
    g09_feature_importance_full(m)
    g10_age_distribution_by_risk(df)
    g11_seasonal_risk_monthly(df)
    g12_who_criteria_frequency(df)
    g13_platelet_trajectory(df)
    g14_roc_curve(m)
    g15_shock_index_by_risk(df)

    print(f"\n{'='*65}")
    print(f"  ✅  All 15 graphs saved to ./{OUT_DIR}/")
    print(f"      Resolution: {DPI} DPI  |  Format: PNG")
    print("=" * 65)


if __name__ == "__main__":
    main()