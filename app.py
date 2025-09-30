# ============================================
# Streamlit app that loads cause_group_pipeline.pkl
# ============================================
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

st.set_page_config(page_title="Predictive Health Risk App", layout="centered")

st.title("🩺 Predictive Health Risk App")
st.markdown("Predict the **cause of death group** from age, gender and location.")

# ---- Load saved assets (pipeline + categories + eval report) ----
try:
    assets = joblib.load("cause_group_pipeline.pkl")
    pipeline = assets["model"]
    categories = assets.get("categories", {})
    eval_report = assets.get("eval_report", None)
except FileNotFoundError:
    st.error("`cause_group_pipeline.pkl` not found. Run the Colab exporter cell and upload the .pkl to this app's repo.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# ---- UI inputs: use categories saved in assets ----
gender_options = categories.get("GENDER", ["M", "F"])
location_options = categories.get("LOCATION", ["Unknown"])
age_group_options = categories.get("AGE_GROUP", ["0-5", "6-18", "19-40", "41-60", "60+"])

age = st.number_input("Enter age", min_value=0, max_value=120, value=30, step=1)
gender = st.selectbox("Gender", gender_options)
location = st.selectbox("Location", location_options)

# ---- Auto or manual age group ----
auto_age_group = st.checkbox("Auto-derive Age Group from age", value=True)

def age_group_func(a):
    try:
        a = float(a)
    except:
        return "Unknown"
    if a <= 5: return "0-5"
    if a <= 18: return "6-18"
    if a <= 40: return "19-40"
    if a <= 60: return "41-60"
    return "60+"

if auto_age_group:
    age_group = age_group_func(age)
    if age_group not in age_group_options:
        age_group = age_group_options[0]
    st.write(f"Derived age group: **{age_group}**")
else:
    age_group = st.selectbox("Age group", age_group_options)

# ---- Prepare input row ----
input_df = pd.DataFrame([[age, gender, location, age_group]],
                        columns=["AGE", "GENDER", "LOCATION", "AGE_GROUP"])

# ---- Prediction ----
if st.button("🔮 Predict Cause Group"):
    try:
        pred = pipeline.predict(input_df)[0]
        st.success(f"Predicted cause group: **{pred}**")

        # ---- Log prediction to CSV ----
        log_file = "predictions_log.csv"
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "age": age,
            "gender": gender,
            "location": location,
            "age_group": age_group,
            "prediction": pred,
        }
        if os.path.exists(log_file):
            df_log = pd.read_csv(log_file)
            df_log = pd.concat([df_log, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            df_log = pd.DataFrame([log_entry])
        df_log.to_csv(log_file, index=False)
        st.info("✅ Prediction logged successfully!")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---- Developer-only debug mode ----
if st.checkbox("🔧 Show backend categories (for developers)"):
    st.json(categories)

# ---- Performance / evaluation (saved during training) ----
if st.checkbox("Show model performance (saved test eval)"):
    if eval_report is None:
        st.info("No evaluation report found in the saved pipeline.")
    else:
        df_report = pd.DataFrame(eval_report).transpose()
        cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in df_report.columns]
        st.write("### Classification report (test set)")
        st.dataframe(df_report[cols].round(3))

# ---- View logs ----
if st.checkbox("📜 View past predictions"):
    log_file = "predictions_log.csv"
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        st.dataframe(df_log.tail(50))  # show last 50 entries
    else:
        st.info("No predictions logged yet.")

# ============================================
# End of app.py
# ============================================

