"""
evaluate_model.py  —  Dengue CDSS  |  Model Evaluation & Validation
═══════════════════════════════════════════════════════════════════════════════
Runs a full evaluation of the trained classifier and regressor.
Outputs:
  • Classification report  (precision / recall / F1 per class)
  • ROC-AUC + confusion matrix
  • Stratified K-fold cross-validation
  • Regressor  R² / MAE / RMSE
  • Feature importance table
  • Optional: SHAP summary (if shap is installed)

Run standalone:
    python evaluate_model.py

Or import and call evaluate() from another script.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    r2_score, mean_absolute_error, mean_squared_error,
)
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import train_model


# ═══════════════════════════════════════════════════════════════════════════
def _section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print('═'*65)


def evaluate():
    _section("Dengue CDSS — Model Evaluation  v3.0")

    # ── Re-train on full dataset to get evaluation splits ───────────────────
    print("  Loading and training models via train_model.main()...")
    result = train_model.main()
    if result[0] is None:
        print("  ❌  Training failed — cannot evaluate.")
        return

    classifier, regressor, clf_features, reg_features = result

    # ── Reload and prepare data (mirrors train_model exactly) ───────────────
    try:
        df = pd.read_csv('dengue_data_cleaned_debug.csv')
    except FileNotFoundError:
        print("  ❌  CSV not found.")
        return

    # ── Apply same preprocessing as train_model ─────────────────────────────
    df['Age']      = pd.to_numeric(df.get('Age', 30), errors='coerce').fillna(30).clip(0, 120)
    df['Sex_Code'] = df.get('Sex', pd.Series(['Male']*len(df))).map(
                         {'Male': 1, 'Female': 0}).fillna(0)
    df['Date_Obj'] = pd.to_datetime(
        df.get('Date of Test & Time of Test', pd.Series(dtype=str)),
        errors='coerce', dayfirst=True)
    df['Season_Risk'] = df['Date_Obj'].dt.month.fillna(6).astype(int).apply(
                            train_model.get_season)
    df['Symptoms'] = df.get('Symptoms', pd.Series(['']*len(df))).fillna('').astype(str).str.lower()
    for kw in ['fever', 'headache', 'pain', 'vomit', 'bleeding']:
        df[f'Has_{kw.capitalize()}'] = df['Symptoms'].apply(lambda x: 1 if kw in x else 0)
    for col in ['Platelet (cells/cu.mm)', 'Haemoglobin (gm/Dl)',
                'Red Blood Cell Count (millions/cu.mm)',
                'Hematocrit (Packed Cell Volume) (%)']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0

    df = train_model._build_synthetic_columns(df)
    df['Dengue_Label'] = train_model._build_dengue_label(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df[clf_features].fillna(0)
    y = df['Dengue_Label']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ════════════════════════════════════════════════════════════════════════
    _section("CLASSIFIER — Hold-out Test Set")
    y_pred = classifier.predict(X_te)
    y_prob = classifier.predict_proba(X_te)[:, 1]

    cm           = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0  # precision
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0
    auc          = roc_auc_score(y_te, y_prob)

    print(f"  Accuracy     : {accuracy_score(y_te, y_pred)*100:.2f}%")
    print(f"  Sensitivity  : {sensitivity*100:.2f}%  (Recall for positives)")
    print(f"  Specificity  : {specificity*100:.2f}%")
    print(f"  PPV/Precision: {ppv*100:.2f}%")
    print(f"  NPV          : {npv*100:.2f}%")
    print(f"  ROC-AUC      : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted 0   Predicted 1")
    print(f"  Actual 0    :    {tn:>6}         {fp:>6}")
    print(f"  Actual 1    :    {fn:>6}         {tp:>6}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred,
          target_names=['No Warning Signs (0)', 'Warning Signs / Severe (1)'],
          digits=4))

    # ════════════════════════════════════════════════════════════════════════
    _section("CLASSIFIER — 5-Fold Stratified Cross-Validation")
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(classifier, X, y, cv=skf,
                             scoring='roc_auc', n_jobs=-1)
    cv_acc = cross_val_score(classifier, X, y, cv=skf,
                             scoring='accuracy', n_jobs=-1)
    cv_f1  = cross_val_score(classifier, X, y, cv=skf,
                             scoring='f1', n_jobs=-1)
    print(f"  AUC   : {cv_auc.mean():.4f}  ±  {cv_auc.std():.4f}  "
          f"(folds: {', '.join(f'{v:.4f}' for v in cv_auc)})")
    print(f"  Acc   : {cv_acc.mean():.4f}  ±  {cv_acc.std():.4f}")
    print(f"  F1    : {cv_f1.mean():.4f}  ±  {cv_f1.std():.4f}")

    # ════════════════════════════════════════════════════════════════════════
    _section("FEATURE IMPORTANCES — Classifier (all features)")
    imp = sorted(zip(clf_features, classifier.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    print(f"  {'Feature':<45}  Importance")
    print(f"  {'-'*55}")
    for fname, fimp in imp:
        bar = '█' * int(fimp * 80)
        print(f"  {fname:<45}  {fimp:.4f}  {bar}")

    # ════════════════════════════════════════════════════════════════════════
    _section("REGRESSOR — Platelet Forecast Engine")
    # Re-build forecast features the same way as train_model
    rng = np.random.default_rng(seed=7)
    df['Day2_Platelets'] = df['Platelet (cells/cu.mm)']
    df['Is_Recovering']  = rng.choice([0, 1], len(df), p=[0.42, 0.58])
    volatility           = rng.uniform(0.08, 0.28, len(df))
    df['Day1_Platelets'] = np.where(
        df['Is_Recovering'] == 1,
        df['Day2_Platelets'] / (1 + volatility),
        df['Day2_Platelets'] * (1 + volatility))
    df['Delta_D1_D2']    = df['Day2_Platelets'] - df['Day1_Platelets']
    recovery_momentum    = np.where(df['Is_Recovering'] == 1, 1.05, 0.88)
    df['Day3_Platelets'] = (
        df['Day2_Platelets'] + df['Delta_D1_D2'] * recovery_momentum +
        rng.normal(0, 1800, len(df))).clip(0, 800000)

    X_reg = df[reg_features].fillna(0)
    y_reg = df['Day3_Platelets']
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    y_pred_r = regressor.predict(X_te_r)
    r2   = r2_score(y_te_r, y_pred_r)
    mae  = mean_absolute_error(y_te_r, y_pred_r)
    rmse = mean_squared_error(y_te_r, y_pred_r) ** 0.5

    print(f"  R²   (test)  : {r2:.4f}")
    print(f"  MAE  (test)  : {mae:,.0f} cells/uL")
    print(f"  RMSE (test)  : {rmse:,.0f} cells/uL")
    print(f"\n  Regressor Feature Importances:")
    reg_imp = sorted(zip(reg_features, regressor.feature_importances_),
                     key=lambda x: x[1], reverse=True)
    print(f"  {'Feature':<45}  Importance")
    print(f"  {'-'*55}")
    for fname, fimp in reg_imp:
        bar = '█' * int(fimp * 60)
        print(f"  {fname:<45}  {fimp:.4f}  {bar}")

    # ════════════════════════════════════════════════════════════════════════
    if SHAP_AVAILABLE:
        _section("SHAP — Feature Contribution Analysis")
        try:
            explainer = shap.TreeExplainer(classifier)
            shap_vals  = explainer.shap_values(X_te.iloc[:200])
            print("  SHAP summary (mean |SHAP| per feature, positive class):")
            sv = np.abs(shap_vals[1] if isinstance(shap_vals, list) else shap_vals).mean(0)
            for fname, sv_val in sorted(zip(clf_features, sv), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    {fname:<45}  {sv_val:.4f}")
        except Exception as e:
            print(f"  SHAP unavailable: {e}")
    else:
        print("\n  ℹ  SHAP not installed. Run: pip install shap")

    print("\n  ✅  Evaluation complete.")


if __name__ == "__main__":
    evaluate()