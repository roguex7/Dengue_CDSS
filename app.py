import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
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

# --- 2. LOAD MODELS ---
@st.cache_resource
def load_ai_engine():
    try:
        results = train_model.main()
        return results[0], results[1], results[2], results[3]
    except Exception as e:
        return None, None, None, None

# --- 3. SEASONAL LOGIC (UPDATED) ---
def get_season_score(selected_date):
    month = selected_date.month
    if month in [12, 1, 2]: return 0      # Winter
    elif month in [3, 4, 5]: return 1     # Summer
    elif month in [6, 7, 8, 9]: return 3  # Monsoon (High Risk)
    elif month in [10, 11]: return 2      # Post-Monsoon
    return 0

# --- 4. INPUT HELPER ---
def render_patient_inputs(key_suffix):
    st.markdown("#### ⚙️ Patient Context & Symptoms")
    c1, c2, c3 = st.columns(3)
    with c1:
        sex = st.selectbox("Sex", ["Male", "Female"], key=f"sex{key_suffix}")
        age = st.slider("Age", 0, 100, 25, key=f"age{key_suffix}")
    with c2:
        hb = st.number_input("Haemoglobin (gm/Dl)", 0.1, 25.0, 13.0, step=0.1, key=f"hb{key_suffix}")
        rbc = st.number_input("RBC Count", 0.1, 14.0, 4.5, step=0.1, key=f"rbc{key_suffix}")
    with c3:
        hct = st.number_input("Hematocrit (%)", 5.0, 80.0, 40.0, step=0.5, key=f"hct{key_suffix}")
    
    st.markdown("**Symptoms Observed:**")
    s1, s2, s3, s4 = st.columns(4)
    fever = s1.checkbox("Fever", value=True, key=f"fever{key_suffix}")
    head = s2.checkbox("Headache", key=f"head{key_suffix}")
    pain = s3.checkbox("Joint/Muscle Pain", value=True, key=f"pain{key_suffix}")
    vomit = s4.checkbox("Vomiting", key=f"vomit{key_suffix}")
    
    return {
        'Sex_Code': 1 if sex == "Male" else 0,
        'Haemoglobin (gm/Dl)': hb, 'Hematocrit (Packed Cell Volume) (%)': hct,
        'Red Blood Cell Count (millions/cu.mm)': rbc, 'Age': age,
        'Has_Fever': int(fever), 'Has_Headache': int(head),
        'Has_Pain': int(pain), 'Has_Vomit': int(vomit)
    }

# --- 5. MAIN APP ---
if 'active_page' not in st.session_state: st.session_state.active_page = 'risk'
def set_page_risk(): st.session_state.active_page = 'risk'
def set_page_forecast(): st.session_state.active_page = 'forecast'

classifier, regressor, clf_features, reg_features = load_ai_engine()

if classifier is None:
    st.error("❌ Critical Error: Models failed to load.")
    st.stop()

st.title("🦟 Dengue CDSS: AI Prediction Engine")

# Navigation
c_nav1, c_nav2, c_spacer = st.columns([1, 1, 3], gap="small")
with c_nav1: st.button("🚨 Risk Analyzer", type="primary" if st.session_state.active_page == 'risk' else "secondary", use_container_width=True, on_click=set_page_risk)
with c_nav2: st.button("📈 Forecast Engine", type="primary" if st.session_state.active_page == 'forecast' else "secondary", use_container_width=True, on_click=set_page_forecast)

st.divider()

# VIEW 1: RISK
if st.session_state.active_page == 'risk':
    st.subheader("🚨 Immediate Risk Assessment")
    
    # NEW: Date Picker
    c_date, c_plt = st.columns([1, 2])
    with c_date:
        test_date = st.date_input("Date of Test", datetime.date.today())
        season_score = get_season_score(test_date)
        season_name = ["Winter (Low)", "Summer (Mod)", "Post-Monsoon (Mod)", "Monsoon (High)"][season_score if season_score < 3 else 3]
        st.caption(f"Season Risk: **{season_name}**")
        
    with c_plt:
        plt_risk = st.number_input("Current Platelet Count", 0, 1000000, 85000, step=1000)

    risk_inputs = render_patient_inputs("_risk")
    
    if st.button("Analyze Risk Now", type="primary", use_container_width=True):
        input_df = pd.DataFrame([risk_inputs])
        input_df['Platelet (cells/cu.mm)'] = plt_risk
        input_df['Season_Risk'] = season_score 
        
        for col in clf_features:
            if col not in input_df.columns: input_df[col] = 0
        input_df = input_df[clf_features]
        
        prediction = classifier.predict(input_df)[0]
        prob = classifier.predict_proba(input_df)[0][1]
        
        st.divider()
        if prediction == 1:
            st.error(f"### ⚠️ HIGH RISK DETECTED\n**Confidence:** {prob*100:.1f}%")
        else:
            st.success(f"### ✅ LOW RISK / STABLE\n**Confidence:** {(1-prob)*100:.1f}%")

# VIEW 2: FORECAST
elif st.session_state.active_page == 'forecast':
    st.subheader("📈 24-Hour Trajectory Forecast")
    
    # NEW: Date Picker Integration
    c_hist1, c_hist2, c_date = st.columns(3)
    with c_hist1: day0 = st.number_input("🗓️ Day-0 (Yesterday)", 0, 1000000, 150000, step=1000)
    with c_hist2: day1 = st.number_input("🗓️ Day-1 (Today)", 0, 1000000, 110000, step=1000)
    with c_date:
        test_date = st.date_input("Date of Test", datetime.date.today(), key="date_fc")
        season_score = get_season_score(test_date)
        season_name = ["Winter (Low)", "Summer (Mod)", "Post-Monsoon (Mod)", "Monsoon (High)"][season_score if season_score < 3 else 3]
        st.info(f"Season: **{season_name}**")

    velocity = day1 - day0
    st.caption(f"Trend Velocity: {velocity} / day")
    st.divider()
    fc_inputs = render_patient_inputs("_forecast")
    
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        input_df = pd.DataFrame([fc_inputs])
        input_df['Day1_Platelets'] = day0
        input_df['Day2_Platelets'] = day1
        input_df['Delta_Day1_Day2'] = velocity
        input_df['Season_Risk'] = season_score 
        
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
