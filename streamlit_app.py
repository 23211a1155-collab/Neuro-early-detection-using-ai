import streamlit as st
import pandas as pd

st.set_page_config(page_title="Neuro Early Detection", layout="centered")

st.title("Neuro Early Detection App")
st.write("This app helps in early detection of neuro-genetic disorders by analyzing patient data.")

# Load datasets
patient_data = pd.read_csv('patient_data.csv')
mental_health_scores = pd.read_csv('mental_health_scores.csv')
model_training_data = pd.read_csv('model_training_data.csv')

# Show dataframes
st.header("Patient Data")
st.dataframe(patient_data)

st.header("Mental Health Scores")
st.dataframe(mental_health_scores)

st.header("Model Training Data")
st.dataframe(model_training_data)

st.info("This is a demo version of the Neuro Early Detection app.")
