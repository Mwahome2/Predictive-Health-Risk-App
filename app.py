import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Load Machine Learning Model and Label Encoders ---
loaded_model = None
label_encoders = {}

try:
    # Load trained ML model
    model_path = "cause_group_model.pkl"   # update with your saved model filename
    loaded_model = joblib.load(open(model_path, "rb"))

    # Load label encoders
    label_encoders["GENDER"] = joblib.load(open("gender_encoder.pkl", "rb"))
    label_encoders["LOCATION"] = joblib.load(open("location_encoder.pkl", "rb"))
    label_encoders["AGE_GROUP"] = joblib.load(open("agegroup_encoder.pkl", "rb"))
    label_encoders["CAUSE_GROUP"] = joblib.load(open("causegroup_encoder.pkl", "rb"))

    st.success("✅ Model and encoders loaded successfully")
except FileNotFoundError:
    st.error("❌ Model/encoder files not found. Please ensure they are in the same directory.")
except Exception as e:
    st.error(f"⚠️ Error loading ML assets: {e}")

# --- 2. Prediction Function ---
def predict_cause_group(input_features):
    input_array = np.asarray(input_features).reshape(1, -1)

    if loaded_model is None:
        return "Model not available"

    try:
        prediction_encoded = loaded_model.predict(input_array)
        predicted_group = label_encoders["CAUSE_GROUP"].inverse_transform(prediction_encoded)[0]
        return predicted_group
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return "Prediction error"

# --- 3. Streamlit App UI ---
def main():
    st.set_page_config(page_title="Predictive Health Risk App", layout="centered")

    st.title("🧑‍⚕️ Predictive Health Risk Web App")
    st.markdown("---")
    st.write("Enter demographic details to predict the likely **cause of death group** (proxy for health risk).")

    # --- Input fields ---
    age = st.slider("Age", min_value=0, max_value=100, value=30, step=1)

    if "GENDER" in label_encoders and hasattr(label_encoders["GENDER"], "classes_"):
        gender_options = list(label_encoders["GENDER"].classes_)
        gender = st.selectbox("Gender", gender_options)
        gender_encoded = label_encoders["GENDER"].transform([gender])[0]
    else:
        st.warning("⚠️ Gender encoder not loaded.")
        gender_encoded = None

    if "LOCATION" in label_encoders and hasattr(label_encoders["LOCATION"], "classes_"):
        location_options = list(label_encoders["LOCATION"].classes_)
        location = st.selectbox("Location", location_options)
        location_encoded = label_encoders["LOCATION"].transform([location])[0]
    else:
        st.warning("⚠️ Location encoder not loaded.")
        location_encoded = None

    if "AGE_GROUP" in label_encoders and hasattr(label_encoders["AGE_GROUP"], "classes_"):
        age_group_options = list(label_encoders["AGE_GROUP"].classes_)
        age_group = st.selectbox("Age Group", age_group_options)
        age_group_encoded = label_encoders["AGE_GROUP"].transform([age_group])[0]
    else:
        st.warning("⚠️ Age group encoder not loaded.")
        age_group_encoded = None

    # --- Run prediction ---
    predicted_result = ""
    st.markdown("---")

    input_data = [age, gender_encoded, location_encoded, age_group_encoded]

    if st.button("🔮 Predict Health Risk"):
        if all(v is not None for v in input_data):
            predicted_result = predict_cause_group(input_data)
        else:
            st.warning("⚠️ Please ensure all inputs are valid.")

    # --- Show results ---
    if predicted_result and predicted_result not in ["Model not available", "Prediction error"]:
        st.success(f"**Predicted Health Risk Group:** {predicted_result} ✅")
    elif predicted_result:
        st.error(predicted_result)

    st.markdown("---")
    st.caption("⚠️ Disclaimer: This prediction is AI-assisted and should not be used as a substitute for medical diagnosis.")

if __name__ == "__main__":
    main()

