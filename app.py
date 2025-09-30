# ============================================
# Streamlit app that loads cause_group_pipeline.pkl
# ============================================
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Predictive Health Risk App", layout="centered")

st.title("🩺 Predictive Health Risk App")
st.markdown("Predict the **cause of death group** from age, gender and location.")

# ---- Load saved assets (pipeline + encoders + eval report) ----
try:
    assets = joblib.load("cause_group_pipeline.pkl")
    pipeline = assets["model"]
    encoders = assets.get("encoders", {})
    eval_report = assets.get("eval_report", None)
except FileNotFoundError:
    st.error("`cause_group_pipeline.pkl` not found. Run the Colab exporter cell and upload the .pkl to this app's repo.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# ---- UI inputs: use encoder classes if available, else simple defaults ----
gender_options = list(encoders.get("GENDER").classes_) if encoders.get("GENDER") is not None else ["Male", "Female", "Other"]
location_options = list(encoders.get("LOCATION").classes_) if encoders.get("LOCATION") is not None else ["Urban", "Rural", "Unknown"]
age_group_options = list(encoders.get("AGE_GROUP").classes_) if encoders.get("AGE_GROUP") is not None else ["0-5", "6-18", "19-40", "41-60", "60+"]

age = st.number_input("Enter age", min_value=0, max_value=120, value=30, step=1)
gender = st.selectbox("Gender", gender_options)
location = st.selectbox("Location", location_options)

# Offer a choice: auto-derive age group or let user pick
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
    # if derived group not in available options, fallback to first option
    if age_group not in age_group_options:
        age_group = age_group_options[0]
    st.write(f"Derived age group: **{age_group}**")
else:
    age_group = st.selectbox("Age group", age_group_options)

# ---- Prepare input row and predict ----
input_df = pd.DataFrame([[age, gender, location, age_group]],
                        columns=["AGE", "GENDER", "LOCATION", "AGE_GROUP"])

if st.button("🔮 Predict Cause Group"):
    try:
        # Note: in previous version, we rebuilt input_df here
        # Keeping it for structure, but reusing existing input_df
        input_df = input_df.copy()

        # 🔄 Previously, we showed encoded values like "1 → Male"
        # That confused users, so we COMMENT it out.
        # You can re-enable for debugging if needed.
        #
        # st.write("### Encoded Values (debug only)")
        # encoded_preview = {}
        # for col in ["GENDER", "LOCATION", "AGE_GROUP"]:
        #     if col in encoders:
        #         le = encoders[col]
        #         encoded_value = le.transform([input_df[col][0]])[0]
        #         encoded_preview[col] = f"{encoded_value} → {input_df[col][0]}"
        # st.json(encoded_preview)

        # ✅ Predict using pipeline (users only see labels, not ints)
        pred = pipeline.predict(input_df)[0]
        st.success(f"Predicted cause group: **{pred}**")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---- Performance / evaluation (saved during training) ----
if st.checkbox("Show model performance (saved test eval)"):
    if eval_report is None:
        st.info("No evaluation report found in the saved pipeline.")
    else:
        # Display a compact table with the 'weighted avg' and class f1's
        df_report = pd.DataFrame(eval_report).transpose()
        cols = [c for c in ["precision", "recall", "f1-score", "support"] if c in df_report.columns]
        st.write("### Classification report (test set)")
        st.dataframe(df_report[cols].round(3))

# ============================================
# End of app.py
# ============================================



