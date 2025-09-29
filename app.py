import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# 1. Load Model + Encoders
# ===============================
assets = None
loaded_model = None
label_encoders = {}

try:
    pipeline_path = "cause_group_pipeline.pkl"  # Make sure this file is in the repo
    assets = joblib.load(open(pipeline_path, "rb"))
    loaded_model = assets["model"]
    label_encoders = assets["encoders"]

    st.success("✅ Model and encoders loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found. Please upload cause_group_pipeline.pkl.")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")

# ===============================
# 2. Prediction Function
# ===============================
def predict_cause_group(input_features):
    try:
        input_array = np.asarray(input_features).reshape(1, -1)
        numerical_prediction = loaded_model.predict(input_array)[0]

        # Decode cause group if encoder exists
        if "CAUSE_GROUP" in label_encoders:
            prediction_label = label_encoders["CAUSE_GROUP"].inverse_transform([numerical_prediction])[0]
        else:
            prediction_label = str(numerical_prediction)

        return prediction_label
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error"

# ===============================
# 3. Streamlit UI
# ===============================
def main():
    st.set_page_config(page_title="🧑‍⚕️ Predictive Health Risk App", layout="centered")
    st.title("🧑‍⚕️ Predictive Health Risk Web App")
    st.markdown("Enter demographic details to predict the likely **Cause of Death Group** (proxy for health risk).")

    # --- Inputs ---
    age = st.slider("Age", min_value=0, max_value=100, value=30, step=1)

    # Gender
    if "GENDER" in label_encoders:
        gender_options = list(label_encoders["GENDER"].classes_)
        gender = st.selectbox("Gender", gender_options)
        gender_encoded = label_encoders["GENDER"].transform([gender])[0]
    else:
        st.warning("⚠️ Gender encoder not loaded.")
        gender_encoded = 0

    # Location
    if "LOCATION" in label_encoders:
        location_options = list(label_encoders["LOCATION"].classes_)
        location = st.selectbox("Location", location_options)
        location_encoded = label_encoders["LOCATION"].transform([location])[0]
    else:
        st.warning("⚠️ Location encoder not loaded.")
        location_encoded = 0

    # Age Group
    if "AGE_GROUP" in label_encoders:
        age_group_options = list(label_encoders["AGE_GROUP"].classes_)
        age_group = st.selectbox("Age Group", age_group_options)
        age_group_encoded = label_encoders["AGE_GROUP"].transform([age_group])[0]
    else:
        st.warning("⚠️ Age group encoder not loaded.")
        age_group_encoded = 0

    st.markdown("---")

    # --- Predict ---
    if st.button("🔮 Predict Health Risk"):
        input_data = [age, gender_encoded, location_encoded, age_group_encoded]
        result = predict_cause_group(input_data)

        st.success(f"✅ Predicted Cause Group: **{result}**")

    st.markdown("---")
    st.caption("⚠️ Disclaimer: This app provides AI-assisted predictions and should not replace professional medical advice.")

if __name__ == "__main__":
    main()


