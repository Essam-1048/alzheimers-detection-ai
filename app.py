import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the saved model and features
model = joblib.load('alzheimers_model.pkl')
feature_names = joblib.load('model_features.pkl')

st.title("🧠 Alzheimer's Early Detection Assistant")
st.write("Enter patient clinical data to assess the risk of Alzheimer's Disease.")

# 2. Create the input UI (Focusing on our most important features)
st.sidebar.header("Patient Symptoms")
mem_complaints = st.sidebar.selectbox("Memory Complaints", ["No", "Yes"])
beh_problems = st.sidebar.selectbox("Behavioral Problems", ["No", "Yes"])
func_assess = st.sidebar.slider("Functional Assessment Score (0-10)", 0.0, 10.0, 5.0)

st.sidebar.header("Clinical Scores")
mmse = st.sidebar.slider("MMSE Score (0-30)", 0.0, 30.0, 20.0)
adl = st.sidebar.slider("ADL Score (0-10)", 0.0, 10.0, 5.0)

# 3. Prepare the data for prediction
# We create a dictionary of all features set to a 'neutral' average
input_dict = {col: 0 for col in feature_names}

# Update the dictionary with our user inputs
input_dict['MemoryComplaints'] = 1 if mem_complaints == "Yes" else 0
input_dict['BehavioralProblems'] = 1 if beh_problems == "Yes" else 0
input_dict['FunctionalAssessment'] = func_assess
input_dict['MMSE'] = mmse
input_dict['ADL'] = adl

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# 4. Make Prediction
if st.button("Analyze Patient Data"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"High Risk Detected: The model is {probability*100:.2f}% confident.")
    else:
        st.success(f"Low Risk: The model is {(1-probability)*100:.2f}% confident the patient is healthy.")

    st.info("Note: This is an AI prototype for research and not a formal medical diagnosis.")