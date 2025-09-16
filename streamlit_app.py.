import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Neuro Early Detection", layout="wide")

# Title
st.title("🧠 Neuro Early Detection App")
st.write("This app helps in early detection of neuro-genetic disorders using AI and patient data.")

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Upload Data", "Model Selection", "Prediction", "About"])

# Load model if available
try:
    model = joblib.load('model.pkl')
except:
    model = None

# Navigation logic
if option == "Home":
    st.header("Welcome!")
    st.write("Use the sidebar to navigate between sections.")
    st.image("https://images.openai.com/thumbnails/url/-zLAknicu1mSUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw5yz0qqcPdMSnd0znfyyfbwjPQo0S0occnJDtS11A0188gpzE_PD3ZMd3JyzzV3La8MyivOTDbO83QtK1crBgAYDyoY", caption="Neuro Early Detection")

elif option == "Upload Data":
    st.header("📂 Upload Data Files")
    patient_file = st.file_uploader("Upload Patient Data CSV", type="csv")
    mental_file = st.file_uploader("Upload Mental Health Scores CSV", type="csv")
    model_file = st.file_uploader("Upload Model Training Data CSV", type="csv")

    if patient_file is not None:
        patient_data = pd.read_csv(patient_file)
        st.subheader("Patient Data Preview")
        st.dataframe(patient_data)

    if mental_file is not None:
        mental_data = pd.read_csv(mental_file)
        st.subheader("Mental Health Scores Preview")
        st.dataframe(mental_data)

    if model_file is not None:
        model_data = pd.read_csv(model_file)
        st.subheader("Model Training Data Preview")
        st.dataframe(model_data)

elif option == "Model Selection":
    st.header("⚙ Choose a Model")
    models = ["Random Forest", "Linear Regression", "Decision Tree"]
    chosen_model = st.selectbox("Select Model Type", models)
    st.write(f"You selected: {chosen_model}")

elif option == "Prediction":
    st.header("📈 Make Predictions")
    if model is not None:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=70)
        respiratory_rate = st.number_input("Respiratory Rate", min_value=10, max_value=40, value=18)
        spo2 = st.number_input("SpO2", min_value=50, max_value=100, value=98)
        body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
        systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
        diastolic = st.number_input("Diastolic BP", min_value=40, max_value=120, value=80)

        if st.button("Predict"):
            input_data = np.array([[age, 1 if gender == "Male" else 0, heart_rate,
                                    respiratory_rate, spo2, body_temp, systolic, diastolic]])
            prediction = model.predict(input_data)
            st.success(f"Predicted PHQ-9 Score: {prediction[0]:.2f}")
    else:
        st.warning("Model file not found. Please upload 'model.pkl' to enable predictions.")

elif option == "About":
    st.header("ℹ About this App")
    st.markdown("""
    This app is designed to assist in early detection of neuro-genetic disorders using machine learning models and patient data.
    
    *Features:*
    - Upload patient data and mental health scores
    - Select machine learning models
    - Make predictions for mental health risk
    - Explore relevant metrics like respiratory rate
    
    Built with Streamlit and scikit-learn.
    """)
