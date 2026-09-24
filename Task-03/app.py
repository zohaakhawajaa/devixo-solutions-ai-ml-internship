from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).with_name("best_diabetes_model.pkl")

if not MODEL_PATH.exists():
    st.error("best_diabetes_model.pkl was not found. Run the notebook's Save the Best Model cell first.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
preprocessor = artifact["preprocessor"]

st.title("Diabetes Prediction")
st.write("Enter patient information to predict diabetes.")

gender = st.selectbox("Gender", ["Female", "Male", "Other"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "not current", "ever", "No Info"],
)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5)
blood_glucose = st.number_input(
    "Blood Glucose Level",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
)

if st.button("Predict"):
    input_data = pd.DataFrame(
        {
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "smoking_history": [smoking_history],
            "bmi": [bmi],
            "HbA1c_level": [hba1c],
            "blood_glucose_level": [blood_glucose],
        }
    )

    input_data_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_data_processed)[0]

    if prediction == 1:
        st.error("Prediction: Diabetes")
    else:
        st.success("Prediction: No Diabetes")
