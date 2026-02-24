import streamlit as st
import joblib
import pandas as pd

# Page Config
st.set_page_config(page_title="NeuroGuard AI", page_icon="🧠", layout="wide")

# Load model and features
model = joblib.load('alzheimers_model.pkl')
features = joblib.load('model_features.pkl')

st.markdown("<h1 style='text-align: center;'>🧠 NeuroGuard AI: Clinical Decision Support</h1>", unsafe_allow_html=True)

st.divider()

# --- TOP SECTION: Balanced Metrics Grid ---
st.markdown("<h3 style='text-align: center;'>📋 Patient Clinical Profile</h3>", unsafe_allow_html=True)

# Create two main columns for the metrics
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("**Symptoms & Observations**")
    memory = st.toggle("Memory Complaints")
    behavior = st.toggle("Behavioral Problems")
    confusion = st.toggle("Confusion")

with right_col:
    st.markdown("**Test Scores & Demographics**")
    # We use a nested grid or just a clean list for these 4 inputs
    mmse = st.number_input("MMSE Score (Cognitive Exam)", 0, 30, 24)
    func_score = st.slider("Functional Assessment", 0.0, 10.0, 7.5)
    adl = st.slider("ADL (Activities of Daily Living)", 0.0, 10.0, 8.0)
    age = st.number_input("Patient Age", 60, 90, 70)

st.divider()

# --- BOTTOM SECTION: Centered Diagnosis ---
# We create 3 columns and use the middle one to center the content
diag_spacer_l, diag_content, diag_spacer_r = st.columns([1, 2, 1])

with diag_content:
    st.subheader("🔍 Diagnostic Results")
    
    # Prepare data
    input_data = {col: 0 for col in features}
    input_data.update({
        'MemoryComplaints': 1 if memory else 0,
        'BehavioralProblems': 1 if behavior else 0,
        'Confusion': 1 if confusion else 0,
        'MMSE': mmse,
        'FunctionalAssessment': func_score,
        'ADL': adl,
        'Age': age
    })
    input_df = pd.DataFrame([input_data])

    if st.button("Generate Final Analysis", type="primary", use_container_width=True):
        prob = model.predict_proba(input_df)[0][1]
        
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if prob > 0.5:
            st.error(f"## High Risk Detected")
            st.metric("Probability of Alzheimer's", f"{prob*100:.1f}%")
            st.warning("Immediate clinical follow-up is recommended.")
        else:
            st.success(f"## Low Risk / Healthy")
            st.metric("Probability of Alzheimer's", f"{prob*100:.1f}%")
            st.info("Results suggest normal cognitive function for this age group.")
        st.markdown("</div>", unsafe_allow_html=True)

st.caption("<p style='text-align: center;'>Developed for Research Purposes | Model Version: XGB-1.0.2</p>", unsafe_allow_html=True)