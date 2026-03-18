"""
Trains two models that power app.py:
  • classifier  — RandomForestClassifier   → risk probability (0–1)
  • regressor   — GradientBoostingRegressor → 24-h platelet forecast

Feature contract
────────────────
clf_features and reg_features returned here are the SINGLE SOURCE OF TRUTH.
app.py reads them back and uses them for inference — do NOT redefine them there.

Any column absent from the CSV is generated synthetically using clinically
realistic distributions so the model handles partially-filled forms gracefully.

v3.1 changes
────────────
• PHYS_BOUNDS dict — clips all core CBC columns to physiological ranges
  before any computation, fixing the rbc=(101, 3471) unit-error bug.
• _STAT_SANITY — validates every computed TRAIN_STATS value before printing;
  substitutes a safe physiological default and warns if out of range.
• Clip report — prints how many rows were out-of-range per column.

Returns: classifier, regressor, clf_features, reg_features
         (4 values — matches app.py load_ai_engine r[0..3])
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, r2_score, confusion_matrix,
                              mean_absolute_error, roc_auc_score)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Model artefact paths ──────────────────────────────────────────────────────
#  Serialised models live in models/ sub-directory — consistently located
#  whether running locally, in Docker, or on Render.
MODELS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CLF_PATH      = os.path.join(MODELS_DIR, "classifier.pkl")
REG_PATH      = os.path.join(MODELS_DIR, "regressor.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.pkl")


# ═══════════════════════════════════════════════════════════════════════════
#  PHYSIOLOGICAL BOUNDS
#  Clips dirty CSV data to plausible clinical ranges before any computation.
#  Source: WHO reference ranges + published dengue cohort data.
#
#  Format: key → (csv_column_name, clip_lo, clip_hi, fillna_default)
#
#  ── RBC FIX ───────────────────────────────────────────────────────
#  The CSV stores RBC in millions/cu.mm (normal: 3.5 – 6.5).
#  Values > 10 are unit errors (cells/uL stored instead of M/uL).
#  clip(1.0, 10.0) rejects all garbage while keeping every real value.
# ═══════════════════════════════════════════════════════════════════════════
PHYS_BOUNDS = {
    'platelets': ('Platelet (cells/cu.mm)',                   1000,    900000, 120000),
    'hb':        ('Haemoglobin (gm/Dl)',                       2.0,      22.0,   12.5),
    'hct':       ('Hematocrit (Packed Cell Volume) (%)',        5.0,      70.0,   38.0),
    'rbc':       ('Red Blood Cell Count (millions/cu.mm)',      1.0,      10.0,    4.5),  # ← v3.1 fix
    'age':       ('Age',                                        0.0,     120.0,   30.0),
}


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN_STATS SANITY BOUNDS
#  Defines what a plausible MEAN and STD should look like after cleaning.
#  If a computed stat falls outside these, it's replaced with a safe default
#  so the OOD detector in app.py is never miscalibrated by dirty data.
#
#  Format: key → (mu_min, mu_max, sigma_min, sigma_max, safe_mu, safe_sigma)
# ═══════════════════════════════════════════════════════════════════════════
_STAT_SANITY = {
    'platelets': ( 50000, 200000,  20000, 120000,  115000, 60000),
    'hb':        (   8.0,   16.0,    1.0,    4.0,    12.5,   2.5),
    'hct':       (  25.0,   55.0,    2.0,   10.0,    38.0,   5.5),
    'rbc':       (   3.0,    6.5,    0.3,    1.5,     4.5,   0.8),  # ← key guard
    'hr':        (  55.0,  130.0,    8.0,   35.0,    92.0,  22.0),
    'sys_bp':    (  80.0,  145.0,    8.0,   30.0,   107.0,  18.0),
    'dia_bp':    (  50.0,  100.0,    5.0,   20.0,    66.0,  12.0),
    'age':       (   5.0,   70.0,    5.0,   25.0,    30.0,  15.0),
}


def get_season(month):
    """Maps calendar month → Season_Risk score (mirrors app.py get_season_score)."""
    if   month in [12, 1, 2]:     return 0   # Off-season   (Low)
    elif month in [3,  4, 5]:     return 1   # Pre-monsoon  (Low-Mod)
    elif month in [6,  7, 8, 9]:  return 3   # Monsoon      (HIGH — Peak)
    elif month in [10, 11]:       return 2   # Post-monsoon (Moderate)
    return 0


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC COLUMN GENERATOR
#  Generates extended clinical columns missing from the CSV.
#  Distributions anchored to WHO 2009 + regional Indian dengue studies.
# ═══════════════════════════════════════════════════════════════════════════
def _build_synthetic_columns(df):
    rng = np.random.default_rng(seed=42)
    n   = len(df)

    # ── Haemodynamics ──────────────────────────────────────────────────────
    if 'Systolic_BP' not in df.columns:
        df['Systolic_BP']  = np.clip(rng.normal(108, 18, n), 60, 180)
    if 'Diastolic_BP' not in df.columns:
        df['Diastolic_BP'] = np.clip(
            df['Systolic_BP'] * 0.62 + rng.normal(0, 5, n), 40, 110)
    if 'Heart_Rate' not in df.columns:
        df['Heart_Rate']   = np.clip(rng.normal(92, 22, n), 40, 180)

    # Derived — always recomputed from whatever BP/HR source is available
    df['Shock_Index']    = (df['Heart_Rate'] /
                            df['Systolic_BP'].replace(0, np.nan)).fillna(0.75)
    df['Shock_Index']    = df['Shock_Index'].clip(0.2, 3.5)
    df['Pulse_Pressure'] = (df['Systolic_BP'] - df['Diastolic_BP']).clip(0, 120)
    df['MAP']            = (df['Diastolic_BP'] +
                            df['Pulse_Pressure'] / 3).clip(40, 130)

    # ── CBC Differential ───────────────────────────────────────────────────
    if 'WBC' not in df.columns:
        # Leukopenia <4000 is hallmark dengue; bimodal with normal population
        wbc_dengue = rng.normal(3200,  900, int(n * 0.55))
        wbc_normal = rng.normal(7500, 2000, n - int(n * 0.55))
        df['WBC']  = np.clip(np.concatenate([wbc_dengue, wbc_normal]), 500, 20000)
    if 'Neutrophil_Pct' not in df.columns:
        df['Neutrophil_Pct'] = np.clip(rng.normal(52, 18, n), 10, 90)
    if 'Lymphocyte_Pct' not in df.columns:
        df['Lymphocyte_Pct'] = np.clip(rng.normal(38, 15, n), 5,  80)
    if 'MPV' not in df.columns:
        # Elevated MPV (>10 fL) common in dengue thrombocytopenia
        df['MPV'] = np.clip(rng.normal(9.8, 1.5, n), 6, 16)

    # ── Liver Function Tests ────────────────────────────────────────────────
    if 'AST' not in df.columns:
        # AST near-universal in dengue; WHO severe ≥ 1000 IU/L
        ast_low  = rng.lognormal(4.0, 0.6, int(n * 0.60))      # ~55–200
        ast_high = rng.lognormal(5.8, 0.8, n - int(n * 0.60))  # 200–3000+
        df['AST'] = np.clip(np.concatenate([ast_low, ast_high]), 10, 5000)
    if 'ALT' not in df.columns:
        df['ALT'] = np.clip(df['AST'] * rng.uniform(0.4, 0.9, n), 5, 3000)
    if 'Albumin' not in df.columns:
        df['Albumin'] = np.clip(rng.normal(3.4, 0.7, n), 1.5, 5.5)

    # ── Coagulation ────────────────────────────────────────────────────────
    if 'INR' not in df.columns:
        # ≥ 2.0 = WHO severe criterion
        df['INR']    = np.clip(rng.lognormal(0.18, 0.32, n), 0.8, 6.0)
    if 'D_Dimer' not in df.columns:
        df['D_Dimer'] = np.clip(rng.lognormal(6.5, 0.9, n), 200, 8000)

    # ── Renal ──────────────────────────────────────────────────────────────
    if 'Creatinine' not in df.columns:
        df['Creatinine'] = np.clip(rng.lognormal(0.1, 0.35, n), 0.4, 6.0)

    # ── Extended Vitals ────────────────────────────────────────────────────
    if 'SpO2' not in df.columns:
        # < 93% = WHO Group C criterion
        df['SpO2'] = np.clip(rng.normal(97.5, 2.0, n), 85, 100)
    if 'GCS' not in df.columns:
        # Depressed GCS = encephalopathy (WHO severe)
        gcs_normal  = np.full(int(n * 0.92), 15)
        gcs_altered = rng.integers(8, 15, n - int(n * 0.92))
        df['GCS']   = np.concatenate([gcs_normal, gcs_altered]).astype(float)

    # ── Imaging ────────────────────────────────────────────────────────────
    if 'Has_Pleural_Effusion' not in df.columns:
        # ~30% of dengue inpatients have pleural effusion
        df['Has_Pleural_Effusion'] = rng.choice([0, 1], n, p=[0.70, 0.30])
    if 'Ascites_Grade' not in df.columns:
        df['Ascites_Grade'] = rng.choice([0, 1, 2, 3], n, p=[0.70, 0.18, 0.08, 0.04])

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  DENGUE LABEL BUILDER  —  WHO 2009 multi-criterion labelling
# ═══════════════════════════════════════════════════════════════════════════
def _build_dengue_label(df):
    """
    Label = 1  →  dengue with warning signs / severe dengue
    Label = 0  →  dengue without warning signs

    Any ONE criterion triggers label = 1:
      CBC      PLT < 100,000  |  Hct > 50  |  Hb < 7
      Haemody  Shock Index > 0.9  |  Pulse Pressure ≤ 20
      LFT      AST ≥ 500
      Coag     INR ≥ 1.5
      Imaging  Pleural effusion = 1  |  Ascites ≥ 2
      Vitals   SpO2 < 93  |  GCS < 13
    """
    label = (
        (df['Platelet (cells/cu.mm)']              < 100000) |
        (df['Hematocrit (Packed Cell Volume) (%)'] > 50)     |
        (df['Haemoglobin (gm/Dl)']                 < 7)      |
        (df['Shock_Index']                         > 0.9)    |
        (df['Pulse_Pressure']                      <= 20)    |
        (df['AST']                                 >= 500)   |
        (df['INR']                                 >= 1.5)   |
        (df['Has_Pleural_Effusion']                == 1)     |
        (df['Ascites_Grade']                       >= 2)     |
        (df['SpO2']                                < 93)     |
        (df['GCS']                                 < 13)
    ).astype(int)
    return label


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN_STATS PRINTER  —  with physiological sanity validation
#  Validates every computed stat against _STAT_SANITY before printing.
#  Bad stats are replaced with safe defaults and flagged clearly.
# ═══════════════════════════════════════════════════════════════════════════
def _print_train_stats(df):
    stat_cols = {
        'platelets': 'Platelet (cells/cu.mm)',
        'hb':        'Haemoglobin (gm/Dl)',
        'hct':       'Hematocrit (Packed Cell Volume) (%)',
        'rbc':       'Red Blood Cell Count (millions/cu.mm)',
        'hr':        'Heart_Rate',
        'sys_bp':    'Systolic_BP',
        'dia_bp':    'Diastolic_BP',
        'age':       'Age',
    }

    print("\n[5/6]  Computed TRAIN_STATS for app.py OOD detector:")
    print("  TRAIN_STATS = {")

    any_warning = False
    for k, col in stat_cols.items():
        if col not in df.columns:
            continue

        mu    = float(df[col].mean())
        sigma = float(df[col].std())

        bounds = _STAT_SANITY.get(k)
        if bounds:
            mu_min, mu_max, s_min, s_max, safe_mu, safe_sigma = bounds
            mu_bad    = not (mu_min <= mu    <= mu_max)
            sigma_bad = not (s_min  <= sigma <= s_max)

            if mu_bad or sigma_bad:
                any_warning = True
                print(f"      # ⚠  '{k}' raw: mu={mu:.1f}, sigma={sigma:.1f}"
                      f" — outside physiological range — using safe default")
                mu, sigma = safe_mu, safe_sigma

        print(f"      '{k}': ({mu:.1f}, {sigma:.1f}),")

    print("  }")

    if any_warning:
        print("\n  ⚠  One or more stats replaced with safe defaults.")
        print("     This usually means the CSV has unit-error outliers.")
        print("     The clip in PHYS_BOUNDS should have caught them — check")
        print("     the clipped-rows report above in [2/6].")
    else:
        print("\n  ✅  All TRAIN_STATS passed physiological sanity checks.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  Dengue CDSS — ML Training Pipeline  v3.1")
    print("=" * 65)

    # ── [1/6] Load CSV ──────────────────────────────────────────────────────
    print("\n[1/6]  Loading data...")
    try:
        df = pd.read_csv('dengue_data_cleaned_debug.csv')
        print(f"       Loaded {len(df):,} rows × {len(df.columns)} columns")
    except FileNotFoundError:
        print("  ❌  'dengue_data_cleaned_debug.csv' not found.")
        return None, None, None, None

    # ── Smart column normaliser ──────────────────────────────────────────────
    _renames = {}
    for col in df.columns:
        low = col.lower()
        if 'age' in low and 'Age' not in df.columns:
            _renames[col] = 'Age'
        if 'sex' in low and 'Sex' not in df.columns:
            _renames[col] = 'Sex'
    if _renames:
        df.rename(columns=_renames, inplace=True)
    if 'Age' not in df.columns:
        df['Age'] = 30
    if 'Sex' not in df.columns:
        df['Sex'] = 'Male'

    # ── [2/6] Clean + physiological bounds ──────────────────────────────────
    print("[2/6]  Cleaning & engineering features...")

    # Apply PHYS_BOUNDS to core CBC columns — catches all unit errors in CSV
    for key, (col, lo, hi, default) in PHYS_BOUNDS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
            out_of_range = int(((df[col] < lo) | (df[col] > hi)).sum())
            df[col] = df[col].clip(lo, hi)
            if out_of_range > 0:
                print(f"       ⚠  {col}: {out_of_range} rows clipped "
                      f"to [{lo}, {hi}]")
        else:
            df[col] = default

    df['Age']      = df['Age'].clip(0, 120)
    df['Sex']      = df['Sex'].astype(str).str.title().str.strip()
    df['Sex_Code'] = df['Sex'].map({'Male': 1, 'Female': 0}).fillna(0)

    # Season
    df['Date_Obj']    = pd.to_datetime(
        df.get('Date of Test & Time of Test', pd.Series(dtype=str)),
        errors='coerce', dayfirst=True)
    df['Season_Risk'] = (df['Date_Obj'].dt.month
                         .fillna(6).astype(int)
                         .apply(get_season))

    # Symptom flags — 'bleeding' included permanently
    df['Symptoms'] = (df.get('Symptoms', pd.Series([''] * len(df)))
                      .fillna('').astype(str).str.lower())
    for kw in ['fever', 'headache', 'pain', 'vomit', 'bleeding']:
        df[f'Has_{kw.capitalize()}'] = (
            df['Symptoms'].apply(lambda x: 1 if kw in x else 0))

    # Generate all synthetic extended columns
    df = _build_synthetic_columns(df)

    # WHO 2009 multi-criterion label
    df['Dengue_Label'] = _build_dengue_label(df)

    pos_rate = df['Dengue_Label'].mean()
    print(f"       Label balance: {pos_rate*100:.1f}% positive "
          f"({df['Dengue_Label'].sum():,} / {len(df):,})")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ════════════════════════════════════════════════════════════════════════
    #  CLF FEATURE SET
    #  ⚠️  CONTRACT with app.py risk_input dict.
    #     Every key here must exist in app.py's risk_input (or be 0-filled
    #     by the safety loop at lines 4074–4076).
    # ════════════════════════════════════════════════════════════════════════
    clf_features = [
        # Core CBC  (always present from app.py form)
        'Platelet (cells/cu.mm)',
        'Haemoglobin (gm/Dl)',
        'Red Blood Cell Count (millions/cu.mm)',
        'Hematocrit (Packed Cell Volume) (%)',
        # Demographics
        'Age',
        'Sex_Code',
        # Haemodynamics  (derived in app.py from sys/dia/hr — always available)
        'Shock_Index',
        'Pulse_Pressure',
        # Symptoms / WHO signs
        'Has_Fever',
        'Has_Headache',
        'Has_Pain',
        'Has_Vomit',
        'Has_Bleeding',
        # Extended clinical  (0-filled when not entered — model handles gracefully)
        'WBC',
        'AST',
        'INR',
        'SpO2',
        'GCS',
        'Has_Pleural_Effusion',
        'Ascites_Grade',
        # Seasonal context
        'Season_Risk',
    ]

    # ── [3/6] Train classifier ───────────────────────────────────────────────
    print("\n[3/6]  Training Risk Classifier (Random Forest)...")
    X_clf = df[clf_features].fillna(0)
    y_clf = df['Dengue_Label']

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',   # handles label imbalance
        n_jobs=-1,
        random_state=42,
    )
    classifier.fit(X_tr_c, y_tr_c)

    y_pred_c    = classifier.predict(X_te_c)
    y_prob_c    = classifier.predict_proba(X_te_c)[:, 1]
    cm          = confusion_matrix(y_te_c, y_pred_c)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc         = roc_auc_score(y_te_c, y_prob_c)

    print(f"       Accuracy    : {accuracy_score(y_te_c, y_pred_c)*100:.2f}%")
    print(f"       Sensitivity : {sensitivity*100:.2f}%")
    print(f"       Specificity : {specificity*100:.2f}%")
    print(f"       AUC (test)  : {auc:.4f}")
    print(f"       Confusion   : TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    importances = sorted(
        zip(clf_features, classifier.feature_importances_),
        key=lambda x: x[1], reverse=True)[:5]
    print("       Top-5 features:")
    for fname, fimp in importances:
        print(f"         {fname:<45}  {fimp:.4f}")

    # ── [4/6] Train regressor ────────────────────────────────────────────────
    print("\n[4/6]  Training Forecast Regressor (Gradient Boosting)...")

    rng2 = np.random.default_rng(seed=7)
    df['Day2_Platelets'] = df['Platelet (cells/cu.mm)']
    df['Is_Recovering']  = rng2.choice([0, 1], len(df), p=[0.42, 0.58])
    volatility           = rng2.uniform(0.08, 0.28, len(df))
    df['Day1_Platelets'] = np.where(
        df['Is_Recovering'] == 1,
        df['Day2_Platelets'] / (1 + volatility),
        df['Day2_Platelets'] * (1 + volatility))
    df['Delta_D1_D2']    = df['Day2_Platelets'] - df['Day1_Platelets']

    recovery_momentum    = np.where(df['Is_Recovering'] == 1, 1.05, 0.88)
    df['Day3_Platelets'] = (
        df['Day2_Platelets'] +
        df['Delta_D1_D2'] * recovery_momentum +
        rng2.normal(0, 1800, len(df))
    ).clip(0, 800000)

    reg_features = [
        # Trajectory  (always available at inference from sorted reports)
        'Day1_Platelets',
        'Day2_Platelets',
        'Delta_D1_D2',
        # Core CBC context
        'Haemoglobin (gm/Dl)',
        'Red Blood Cell Count (millions/cu.mm)',
        'Hematocrit (Packed Cell Volume) (%)',
        # Demographics
        'Age',
        'Sex_Code',
        # Haemodynamics
        'Shock_Index',
        'Pulse_Pressure',
        # Clinical modifiers
        'Has_Fever',
        'Has_Vomit',
        'Has_Pain',
        'Has_Bleeding',
        'AST',
        'INR',
        # Seasonal context
        'Season_Risk',
    ]

    X_reg = df[reg_features].fillna(0)
    y_reg = df['Day3_Platelets']

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    regressor = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.85,
        min_samples_leaf=4,
        random_state=42,
    )
    regressor.fit(X_tr_r, y_tr_r)

    y_pred_r = regressor.predict(X_te_r)
    print(f"       R²  (test)  : {r2_score(y_te_r, y_pred_r):.4f}")
    print(f"       MAE (test)  : {mean_absolute_error(y_te_r, y_pred_r):,.0f} cells/uL")

    # ── [5/6] Print validated TRAIN_STATS ───────────────────────────────────
    _print_train_stats(df)

    # ── [6/6] Serialise models to disk ──────────────────────────────────────
    print("\n[6/6]  Saving trained models to disk...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(classifier, CLF_PATH,      compress=3)
    joblib.dump(regressor,  REG_PATH,      compress=3)
    joblib.dump({'clf_features': clf_features,
                 'reg_features': reg_features}, FEATURES_PATH, compress=3)
    print(f"       classifier  → {CLF_PATH}")
    print(f"       regressor   → {REG_PATH}")
    print(f"       features    → {FEATURES_PATH}")
    print("\n  ✅  Training complete — models baked and ready.\n")
    print("=" * 65)
    return classifier, regressor, clf_features, reg_features


def load_pretrained():
    """
    Load pre-trained models from disk.
    Returns (classifier, regressor, clf_features, reg_features) or None if missing.
    Called by app.py load_ai_engine() on every cold start — sub-second load time.
    """
    if not all(os.path.exists(p) for p in [CLF_PATH, REG_PATH, FEATURES_PATH]):
        return None
    clf      = joblib.load(CLF_PATH)
    reg      = joblib.load(REG_PATH)
    features = joblib.load(FEATURES_PATH)
    return clf, reg, features['clf_features'], features['reg_features']


if __name__ == "__main__":
    main()