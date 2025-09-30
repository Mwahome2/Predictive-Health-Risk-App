# ============================================
# Streamlit app that loads cause_and_exact_pipeline.pkl
# ============================================
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="Predictive Health Risk App", layout="centered")

st.title("🩺 Predictive Health Risk App")
st.markdown("Predict both the **exact cause of death** and the broader **cause group** from age, gender and location.")

# ---- Load saved assets (two models + categories + eval reports) ----
try:
    assets = joblib.load("cause_and_exact_pipeline.pkl")
    model_group = assets["model_group"]
    model_exact = assets["model_exact"]
    categories = assets.get("categories", {})
    eval_group = assets.get("eval_group", None)
    eval_exact = assets.get("eval_exact", None)
except FileNotFoundError:
    st.error("`cause_and_exact_pipeline.pkl` not found. Run the Colab exporter cell and upload it to this app's repo.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# ---- UI inputs ----
gender_options = categories.get("GENDER", ["M", "F"])
location_options = categories.get("LOCATION", ["Unknown"])
age_group_options = categories.get("AGE_GROUP", ["0-5", "6-18", "19-40", "41-60", "60+"])

age = st.number_input("Enter age", min_value=0, max_value=120, value=30, step=1)
gender = st.selectbox("Gender", gender_options)
location = st.selectbox("Location", location_options)

auto_age_group = st.checkbox("Auto-derive Age Group from age", value=True)

def derive_age_group(a):
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
    age_group = derive_age_group(age)
    if age_group not in age_group_options:
        age_group = age_group_options[0]
    st.write(f"Derived age group: **{age_group}**")
else:
    age_group = st.selectbox("Age group", age_group_options)

# ---- Prepare input row ----
input_df = pd.DataFrame([[age, gender, location, age_group]],
                        columns=["AGE", "GENDER", "LOCATION", "AGE_GROUP"])

# ---- Prediction ----
if st.button("🔮 Predict Cause & Group"):
    try:
        pred_group = model_group.predict(input_df)[0]
        pred_exact = model_exact.predict(input_df)[0]

        st.success(f"Predicted Cause Group: **{pred_group}**")
        st.success(f"Predicted Exact Cause: **{pred_exact}**")

        # ---- Log prediction to CSV ----
        log_file = "predictions_log.csv"
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "age": age,
            "gender": gender,
            "location": location,
            "age_group": age_group,
            "cause_group": pred_group,
            "cause_exact": pred_exact,
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

# ---- View past predictions ----
if st.checkbox("📜 View past predictions"):
    log_file = "predictions_log.csv"
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
        st.dataframe(df_log.tail(50))  # show last 50
    else:
        st.info("No predictions logged yet.")

# ---- Performance reports ----
if st.checkbox("📊 Show model performance"):
    if eval_group:
        st.write("### Group model performance")
        dfg = pd.DataFrame(eval_group).transpose()
        cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in dfg.columns]
        st.dataframe(dfg[cols].round(3))
    if eval_exact:
        st.write("### Exact cause model performance")
        dfe = pd.DataFrame(eval_exact).transpose()
        cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in dfe.columns]
        st.dataframe(dfe[cols].round(3))


# ============================================
# End of app.py
# ============================================

