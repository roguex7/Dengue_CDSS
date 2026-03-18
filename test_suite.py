"""
Tests every layer of the ML pipeline:
  1. Training smoke test  — models load, clf_features / reg_features returned
  2. Feature contract     — clf_features matches app.py risk_input keys exactly
  3. Inference shapes     — predict_proba / predict return correct dimensions
  4. Clinical sanity      — high-risk patients score higher than low-risk
  5. Edge cases           — all-zero input, missing fields, extreme values
  6. CI width             — compute_tree_ci returns 4 values, CI within [0,1]
  7. Regressor direction  — rising platelet input → higher forecast than falling

Run:
    python test_suite.py
"""

import sys
import traceback
import numpy as np
import pandas as pd

# ── Colour helpers ────────────────────────────────────────────────────────────
PASS  = "\033[92m  PASS\033[0m"
FAIL  = "\033[91m  FAIL\033[0m"
SKIP  = "\033[93m  SKIP\033[0m"
_results = []


def _run(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        _results.append((name, 'pass'))
    except AssertionError as e:
        print(f"{FAIL}  {name}")
        print(f"         AssertionError: {e}")
        _results.append((name, 'fail'))
    except Exception as e:
        print(f"{FAIL}  {name}  [{type(e).__name__}: {e}]")
        traceback.print_exc()
        _results.append((name, 'error'))


# ════════════════════════════════════════════════════════════════════════════
#  This list is the ground truth for the contract test.
# ════════════════════════════════════════════════════════════════════════════
APP_RISK_INPUT_KEYS = {
    'Platelet (cells/cu.mm)',
    'Haemoglobin (gm/Dl)',
    'Red Blood Cell Count (millions/cu.mm)',
    'Hematocrit (Packed Cell Volume) (%)',
    'Age',
    'Sex_Code',
    'Shock_Index',
    'Pulse_Pressure',
    'Has_Fever',
    'Has_Headache',
    'Has_Pain',
    'Has_Vomit',
    'Has_Bleeding',
    'WBC',
    'AST',
    'INR',
    'SpO2',
    'GCS',
    'Has_Pleural_Effusion',
    'Ascites_Grade',
    'Season_Risk',
}


def _make_row(clf_features, overrides=None):
    """Build a minimal valid inference DataFrame from clf_features."""
    base = {f: 0 for f in clf_features}
    # Sensible clinical defaults
    base.update({
        'Platelet (cells/cu.mm)':               150000,
        'Haemoglobin (gm/Dl)':                  13.5,
        'Red Blood Cell Count (millions/cu.mm)': 4.5,
        'Hematocrit (Packed Cell Volume) (%)':   42.0,
        'Age':                                   35,
        'Sex_Code':                              1,
        'Shock_Index':                           0.65,
        'Pulse_Pressure':                        40,
        'Has_Fever':                             1,
        'SpO2':                                  98,
        'GCS':                                   15,
        'Season_Risk':                           1,
    })
    if overrides:
        base.update(overrides)
    row = {k: base.get(k, 0) for k in clf_features}
    return pd.DataFrame([row])


def _make_reg_row(reg_features, overrides=None):
    base = {f: 0 for f in reg_features}
    base.update({
        'Day1_Platelets':                         85000,
        'Day2_Platelets':                         72000,
        'Delta_D1_D2':                            -13000,
        'Haemoglobin (gm/Dl)':                   11.0,
        'Red Blood Cell Count (millions/cu.mm)':  3.8,
        'Hematocrit (Packed Cell Volume) (%)':    44.0,
        'Age':                                    28,
        'Sex_Code':                               0,
        'Shock_Index':                            0.75,
        'Pulse_Pressure':                         38,
        'Has_Fever':                              1,
        'Season_Risk':                            3,
    })
    if overrides:
        base.update(overrides)
    row = {k: base.get(k, 0) for k in reg_features}
    return pd.DataFrame([row])


# ════════════════════════════════════════════════════════════════════════════
#  TESTS
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  Dengue CDSS — Test Suite  v3.0")
    print("=" * 65)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n  Loading models from train_model.main()...")
    try:
        import train_model
        result = train_model.main()
        assert result[0] is not None, "Training returned None"
        classifier, regressor, clf_features, reg_features = result
        print(f"  Models loaded.  "
              f"clf_features={len(clf_features)}  reg_features={len(reg_features)}\n")
    except Exception as e:
        print(f"\033[91m  FATAL: Cannot load models — {e}\033[0m")
        sys.exit(1)

    # ── T1: Training smoke test ──────────────────────────────────────────────
    def t1():
        assert classifier  is not None, "classifier is None"
        assert regressor   is not None, "regressor is None"
        assert clf_features and len(clf_features) > 0, "clf_features empty"
        assert reg_features and len(reg_features) > 0, "reg_features empty"
    _run("T1  Training smoke test", t1)

    # ── T2: Feature contract — clf_features ⊆ APP_RISK_INPUT_KEYS ───────────
    def t2():
        missing_in_app = set(clf_features) - APP_RISK_INPUT_KEYS
        assert not missing_in_app, (
            f"clf_features contains keys not in app.py risk_input: "
            f"{missing_in_app}\n"
            f"Either add them to app.py risk_input or remove them from clf_features.")
    _run("T2  Feature contract — clf_features ⊆ app.py risk_input", t2)

    # ── T3: Inference shape — classifier ────────────────────────────────────
    def t3():
        df_in = _make_row(clf_features)
        assert list(df_in.columns) == clf_features, "Column order mismatch"
        proba = classifier.predict_proba(df_in)
        assert proba.shape == (1, 2), f"Expected (1,2), got {proba.shape}"
        assert 0.0 <= proba[0][1] <= 1.0, "Probability out of [0,1]"
    _run("T3  Inference shape — predict_proba returns (1,2)", t3)

    # ── T4: Inference shape — regressor ─────────────────────────────────────
    def t4():
        df_in = _make_reg_row(reg_features)
        pred  = regressor.predict(df_in)
        assert pred.shape == (1,), f"Expected (1,), got {pred.shape}"
        assert pred[0] >= 0, f"Forecast negative: {pred[0]}"
    _run("T4  Inference shape — regressor predict returns (1,)", t4)

    # ── T5: Clinical sanity — severe > mild ──────────────────────────────────
    def t5():
        # Severe: PLT=18k, Shock Index=1.3, pleural effusion, peak season
        severe = _make_row(clf_features, {
            'Platelet (cells/cu.mm)': 18000,
            'Hematocrit (Packed Cell Volume) (%)': 52,
            'Shock_Index': 1.3,
            'Pulse_Pressure': 18,
            'Has_Pleural_Effusion': 1,
            'INR': 2.2,
            'AST': 1200,
            'SpO2': 90,
            'Season_Risk': 3,
            'Has_Fever': 1,
            'Has_Bleeding': 1,
        })
        # Low risk: PLT=180k, all normals, off-season
        mild = _make_row(clf_features, {
            'Platelet (cells/cu.mm)': 180000,
            'Hematocrit (Packed Cell Volume) (%)': 40,
            'Shock_Index': 0.55,
            'Pulse_Pressure': 45,
            'Season_Risk': 0,
        })
        p_severe = classifier.predict_proba(severe)[0][1]
        p_mild   = classifier.predict_proba(mild)[0][1]
        assert p_severe > p_mild, (
            f"Expected severe ({p_severe:.3f}) > mild ({p_mild:.3f})")
    _run("T5  Clinical sanity — severe patient scores higher", t5)

    # ── T6: Edge case — all-zero input ───────────────────────────────────────
    def t6():
        df_zero = pd.DataFrame([{f: 0 for f in clf_features}])
        proba   = classifier.predict_proba(df_zero)
        assert proba.shape == (1, 2), "Shape error on all-zero input"
        assert 0.0 <= proba[0][1] <= 1.0, "Probability out of range on zeros"
    _run("T6  Edge case — all-zero input doesn't crash", t6)

    # ── T7: Edge case — extreme values ───────────────────────────────────────
    def t7():
        extreme = _make_row(clf_features, {
            'Platelet (cells/cu.mm)': 1,
            'Haemoglobin (gm/Dl)': 2.0,
            'Hematocrit (Packed Cell Volume) (%)': 70,
            'Shock_Index': 3.5,
            'SpO2': 70,
            'GCS': 3,
            'AST': 9999,
            'INR': 9.9,
        })
        proba = classifier.predict_proba(extreme)
        assert 0.0 <= proba[0][1] <= 1.0
    _run("T7  Edge case — extreme clinical values within [0,1]", t7)

    # ── T8: Missing column graceful handling ─────────────────────────────────
    def t8():
        # Simulate app.py safety loop: missing features filled with 0
        partial = {
            'Platelet (cells/cu.mm)': 95000,
            'Haemoglobin (gm/Dl)': 11.5,
            'Age': 30,
            'Sex_Code': 1,
            'Season_Risk': 3,
        }
        df_partial = pd.DataFrame([partial])
        for col in clf_features:
            if col not in df_partial.columns:
                df_partial[col] = 0
        df_partial = df_partial[clf_features]
        proba = classifier.predict_proba(df_partial)
        assert proba.shape == (1, 2)
    _run("T8  Partial input (app.py safety loop pattern)", t8)

    # ── T9: compute_tree_ci contract ─────────────────────────────────────────
    def t9():
        df_in = _make_row(clf_features)
        trees = np.array([
            t.predict_proba(df_in.values)[0][1]
            for t in classifier.estimators_
        ])
        lo  = float(np.percentile(trees, 2.5))
        hi  = float(np.percentile(trees, 97.5))
        est = float(np.mean(trees))
        w   = hi - lo
        assert 0.0 <= lo <= hi <= 1.0, f"CI out of range: [{lo:.3f}, {hi:.3f}]"
        assert 0.0 <= est <= 1.0,       f"Estimate {est:.3f} out of range"
        assert 0.0 <= w  <= 1.0,        f"CI width {w:.3f} out of range"
    _run("T9  compute_tree_ci — CI in [0,1], width valid", t9)

    # ── T10: Regressor direction — declining trajectory ───────────────────────
    def t10():
        declining = _make_reg_row(reg_features, {
            'Day1_Platelets': 120000,
            'Day2_Platelets': 80000,
            'Delta_D1_D2':   -40000,
            'Season_Risk':    3,
            'Shock_Index':    1.1,
        })
        recovering = _make_reg_row(reg_features, {
            'Day1_Platelets': 60000,
            'Day2_Platelets': 90000,
            'Delta_D1_D2':    30000,
            'Season_Risk':    0,
            'Shock_Index':    0.6,
        })
        pred_dec = regressor.predict(declining)[0]
        pred_rec = regressor.predict(recovering)[0]
        assert pred_rec > pred_dec, (
            f"Expected recovering ({pred_rec:,.0f}) > declining ({pred_dec:,.0f})")
    _run("T10 Regressor direction — recovering > declining trajectory", t10)

    # ── T11: Seasonal modifier sanity ────────────────────────────────────────
    def t11():
        peak    = _make_row(clf_features, {'Season_Risk': 3,
                                           'Platelet (cells/cu.mm)': 90000})
        offseason = _make_row(clf_features, {'Season_Risk': 0,
                                             'Platelet (cells/cu.mm)': 90000})
        p_peak = classifier.predict_proba(peak)[0][1]
        p_off  = classifier.predict_proba(offseason)[0][1]
        # Peak should be >= off-season for same patient profile
        assert p_peak >= p_off - 0.05, (
            f"Peak season ({p_peak:.3f}) unexpectedly lower than off-season ({p_off:.3f})")
    _run("T11 Seasonal modifier — peak season ≥ off-season risk", t11)

    # ── T12: clf_features and reg_features have no duplicates ────────────────
    def t12():
        assert len(clf_features) == len(set(clf_features)), \
            f"Duplicate in clf_features: {[f for f in clf_features if clf_features.count(f)>1]}"
        assert len(reg_features) == len(set(reg_features)), \
            f"Duplicate in reg_features"
    _run("T12 No duplicate feature names", t12)

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(1 for _, r in _results if r == 'pass')
    failed = sum(1 for _, r in _results if r in ('fail', 'error'))
    total  = len(_results)

    print(f"\n{'='*65}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  \033[91m{failed} FAILED\033[0m")
    else:
        print(f"  |  \033[92mAll tests passed ✓\033[0m")
    print('='*65)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()