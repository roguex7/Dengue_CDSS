import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import train_model

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Dengue CDSS AI",
    page_icon="🦟",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #0e1117;}
    div.stButton > button:first-child {
        width: 100%; border-radius: 8px; font-weight: 600; height: 3em;
    }
    div[data-testid="column"] { padding: 0px; }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD MODELS (ROBUST) ---
@st.cache_resource
def load_ai_engine():
    try:
        # We fetch everything, but we ignore the stats (_) since we hid that menu
        # This keeps compatibility with your advanced train_model.py
        results = train_model.main()
        
        # Unpack safely (Grab first 4 items: clf, reg, cols, cols)
        clf = results[0]
        reg = results[1]
        c_cols = results[2]
        r_cols = results[3]
        
        return clf, reg, c_cols, r_cols
    except Exception as e:
        return None, None, None, None

# Initialize Session State
if 'active_page' not in st.session_state: st.session_state.active_page = 'risk'
def set_page_risk(): st.session_state.active_page = 'risk'
def set_page_forecast(): st.session_state.active_page = 'forecast'

# --- HELPER FUNCTION (INPUTS) ---
def render_patient_inputs(key_suffix):
    st.markdown("#### ⚙️ Patient Context & Symptoms")
    c1, c2, c3 = st.columns(3)
    with c1:
        sex = st.selectbox("Sex", ["Male", "Female"], key=f"sex{key_suffix}")
        age = st.slider("Age", 0, 100, 25, key=f"age{key_suffix}")
    
    with c2:
        # Haemoglobin: 0.1 (Fatal) to 25.0 (World Record High)
        # Prevents negative numbers or impossible highs like 50
        hb = st.number_input("Haemoglobin (gm/Dl)", 0.1, 25.0, 13.0, step=0.1, key=f"hb{key_suffix}")
        
        # RBC Count: 0.1 to 14.0
        # Normal is ~5. 14 is the biological limit before blood stops flowing.
        # This BLOCKS 41.50 (Typos) but allows severe polycythemia (9.0)
        rbc = st.number_input("RBC Count", 0.1, 14.0, 4.5, step=0.1, key=f"rbc{key_suffix}")

    with c3:
        # Hematocrit: 5% to 80%
        # < 5% is incompatible with life. > 80% is extremely rare sludge.
        # This keeps inputs realistic.
        hct = st.number_input("Hematocrit (%)", 5.0, 80.0, 40.0, step=0.5, key=f"hct{key_suffix}")

    st.markdown("**Symptoms Observed:**")
    s1, s2, s3, s4 = st.columns(4)
    fever = s1.checkbox("Fever", value=True, key=f"fever{key_suffix}")
    head = s2.checkbox("Headache", key=f"head{key_suffix}")
    pain = s3.checkbox("Joint/Muscle Pain", value=True, key=f"pain{key_suffix}")
    vomit = s4.checkbox("Vomiting", key=f"vomit{key_suffix}")
    
    return {
        'Sex_Code': 1 if sex == "Male" else 0,
        'Haemoglobin (gm/Dl)': hb,
        'Hematocrit (Packed Cell Volume) (%)': hct,
        'Red Blood Cell Count (millions/cu.mm)': rbc,
        'Age': age,
        'Has_Fever': int(fever),
        'Has_Headache': int(head),
        'Has_Pain': int(pain),
        'Has_Vomit': int(vomit)
    }

# --- 3. MAIN APP ---
classifier, regressor, clf_features, reg_features = load_ai_engine()

if classifier is None:
    st.error("❌ Critical Error: Models failed to load. Run 'python train_model.py' first.")
    st.stop()

st.title("🦟 Dengue CDSS: AI Prediction Engine")

# --- NAVIGATION (Cleaned Up) ---
c_nav1, c_nav2, c_spacer = st.columns([1, 1, 3], gap="small")
with c_nav1:
    st.button("🚨 Risk Analyzer", type="primary" if st.session_state.active_page == 'risk' else "secondary", use_container_width=True, on_click=set_page_risk)
with c_nav2:
    st.button("📈 Forecast Engine", type="primary" if st.session_state.active_page == 'forecast' else "secondary", use_container_width=True, on_click=set_page_forecast)

st.divider()

# >>> VIEW 1: RISK ANALYZER <<<
if st.session_state.active_page == 'risk':
    st.subheader("🚨 Immediate Risk Assessment")
    st.info("Enter details to assess the **Current Severity** of the infection.")
    plt_risk = st.number_input("Current Platelet Count", 0, 1000000, 85000, step=1000)
    risk_inputs = render_patient_inputs("_risk")
    
    if st.button("Analyze Risk Now", type="primary", use_container_width=True):
        input_df = pd.DataFrame([risk_inputs])
        input_df['Platelet (cells/cu.mm)'] = plt_risk
        for col in clf_features:
            if col not in input_df.columns: input_df[col] = 0
        input_df = input_df[clf_features]
        
        prediction = classifier.predict(input_df)[0]
        prob = classifier.predict_proba(input_df)[0][1]
        
        st.divider()
        if prediction == 1:
            st.error(f"### ⚠️ HIGH RISK DETECTED")
            st.markdown(f"**Confidence:** {prob*100:.1f}% | **Action:** Immediate Admission.")
        else:
            st.success(f"### ✅ LOW RISK / STABLE")
            st.markdown(f"**Confidence:** {(1-prob)*100:.1f}% | **Action:** Monitor Outpatient.")

# >>> VIEW 2: FORECAST ENGINE <<<
elif st.session_state.active_page == 'forecast':
    st.subheader("📈 24-Hour Trajectory Forecast")
    st.info("Predicts **Day-3 (Tomorrow)** based on momentum.")
    c_hist1, c_hist2 = st.columns(2)
    with c_hist1: day0 = st.number_input("🗓️ Day-0 (Yesterday)", 0, 1000000, 150000, step=1000)
    with c_hist2: day1 = st.number_input("🗓️ Day-1 (Today)", 0, 1000000, 110000, step=1000)
    velocity = day1 - day0
    st.caption(f"Trend Velocity: {velocity} / day")
    st.divider()
    fc_inputs = render_patient_inputs("_forecast")
    
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        input_df = pd.DataFrame([fc_inputs])
        input_df['Day1_Platelets'] = day0
        input_df['Day2_Platelets'] = day1
        input_df['Delta_Day1_Day2'] = velocity
        for col in reg_features:
            if col not in input_df.columns: input_df[col] = 0
        input_df = input_df[reg_features]
        
        pred_day3 = regressor.predict(input_df)[0]
        st.divider()
        c_res, c_graph = st.columns([1, 3], gap="small")
        with c_res:
            st.metric("Tomorrow's Count", f"{int(pred_day3)}", delta=int(pred_day3 - day1))
            if pred_day3 < 20000: st.error("CRITICAL RISK")
            elif pred_day3 < day1: st.warning("Declining")
            else: st.success("Recovering")
        with c_graph:
            x = ["Day-0", "Day-1", "Day-3"]
            y = [day0, day1, pred_day3]
            color = '#FF4B4B' if pred_day3 < day1 else '#09AB3B'
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(x, y, marker='o', color=color, linewidth=3)
            for i, val in enumerate(y):
                ax.annotate(f"{int(val)}", xy=(i, val), xytext=(0, 12), textcoords='offset points', ha='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False); ax.set_yticks([]); plt.tight_layout()
            st.pyplot(fig, use_container_width=True)