from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.title("Diabetes classification demo")
st.write(
    "A portfolio experiment using the Pima Indians Diabetes dataset. "
    "This model is not validated for medical diagnosis."
)

# Locate the saved model relative to this file.
model_path = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "diabetes_model.joblib"
)
bundle = joblib.load(model_path)

st.write(
    "Enter the measurements below. For glucose, blood pressure, "
    "skin thickness, insulin, and BMI, enter 0 if the value is unknown."
)

with st.form("patient_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input(
        "Glucose, 2-hour oral glucose tolerance test (mg/dL)",
        min_value=0.0,
    )
    blood_pressure = st.number_input(
        "Diastolic blood pressure (mm Hg)", min_value=0.0
    )
    skin_thickness = st.number_input(
        "Triceps skinfold thickness (mm)", min_value=0.0
    )
    insulin = st.number_input(
        "2-hour serum insulin (µU/mL)", min_value=0.0
    )
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0)
    pedigree = st.number_input(
        "Diabetes pedigree function",
        min_value=0.0,
        value=None,
        help="The dataset's numerical family-history score.",
    )
    age = st.number_input("Age (years)", min_value=21, step=1)

    submitted = st.form_submit_button("Run model")

if submitted:
    if pedigree is None or pedigree <= 0:
        st.error("Enter a positive diabetes pedigree function value.")
    elif all(
        value == 0
        for value in [glucose, blood_pressure, skin_thickness, insulin, bmi]
    ):
        st.error("Enter at least one measurement besides age and pregnancies.")
    else:
        patient = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
        }])

        # Match the preprocessing and column order used during training.
        missing_columns = bundle["zero_as_missing"]
        patient[missing_columns] = patient[missing_columns].replace(0, np.nan)
        patient = patient[bundle["columns"]]

        probability = bundle["pipeline"].predict_proba(patient)[0, 1]
        threshold = bundle["threshold"]
        prediction = int(probability >= threshold)

        st.metric("Model score for class 1", f"{probability:.1%}")
        st.write(f"Decision threshold: {threshold:.0%}")
        st.write(f"Predicted class: {prediction}")

        st.caption(
            "Class 0 = no diabetes label; class 1 = diabetes label. "
            "The model score is not an established personal medical risk."
        )