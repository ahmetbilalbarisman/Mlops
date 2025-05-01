import streamlit as st
import pandas as pd
import mlflow.sklearn
import json

# 🎯 MODEL URI
MODEL_NAME = "TelcoChurnVotingEnsemble"
MODEL_VERSION = 2
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

# 📥 Feature isimlerini yükle
with open("feature_names.json", "r") as f:
    expected_features = json.load(f)

st.title("Telco Customer Churn Prediction")
st.write("This app predicts churn using the VotingClassifier model trained with MLflow.")

# 📋 Kullanıcı giriş formu
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges", 0.0, 200.0, 70.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    
    # 🟦 Buton burada görünür
    submitted = st.form_submit_button("Predict")

if submitted:
    # 🔧 Feature engineering
    input_data = pd.DataFrame({
        "SeniorCitizen": [senior],
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        f"gender_{gender}": [1],
        f"Partner_{partner}": [1],
        f"Dependents_{dependents}": [1],
        f"Contract_{contract}": [1],
        f"InternetService_{internet}": [1],
        f"DeviceProtection_{device_protection}": [1],
        f"MultipleLines_{multiple_lines}": [1],
    })

    # Eksik feature'ları doldur
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_features]

    # 🔮 Prediction
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    st.write("---")
    if prediction == 1:
        st.error(f"⚠️ Churn risk is high (%{proba*100:.1f})")
    else:
        st.success(f"✅ Churn risk is low (%{proba*100:.1f})")
