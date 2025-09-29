import streamlit as st
import pickle
import pandas as pd
import os
import joblib # Import joblib
import numpy as np
# ===========================
# STREAMLIT APP
# ===========================
st.title("🩺 Predictive Health Risk App")

st.write("Predict the **cause of death group** based on demographics and location.")

# User input
age = st.number_input("Enter Age:", min_value=0, max_value=120, value=30)
gender = st.selectbox("Select Gender:", data["GENDER"].unique())
location = st.selectbox("Select Location:", data["LOCATION"].unique())

# Derive age group
if age <= 5:
    age_group = "0-5"
elif age <= 18:
    age_group = "6-18"
elif age <= 40:
    age_group = "19-40"
elif age <= 60:
    age_group = "41-60"
else:
    age_group = "60+"

# Prepare input row
input_data = pd.DataFrame([[age, gender, location, age_group]],
                          columns=["AGE", "GENDER", "LOCATION", "AGE_GROUP"])
input_encoded = pd.get_dummies(input_data, drop_first=True)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Choose model
model_choice = st.selectbox("Choose a model:", list(loaded_models.keys()))

# Predict
if st.button("Predict Cause of Death Group"):
    pred = loaded_models[model_choice].predict(input_encoded)[0]
    st.success(f"Predicted Cause Group using {model_choice}: **{pred}**")

# ===========================
# PERFORMANCE REPORT
# ===========================
if st.checkbox("Show performance comparison"):
    reports = {}
    for name, model in loaded_models.items():
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        reports[name] = report["weighted avg"]["f1-score"]

    st.write("### Weighted F1-Scores by Model")
    st.write(pd.DataFrame.from_dict(reports, orient="index", columns=["F1-score"]))
