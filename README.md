# Telco Customer Churn Prediction with MLflow

This project was developed as part of the AIN-3009 MLOps Term Project at Bahçeşehir University. It demonstrates the full lifecycle of a machine learning project — from model development to deployment and monitoring — using **MLflow**.

---

## 📌 Project Description

The goal of this project is to predict customer churn in a telecommunications company using various machine learning models. The project showcases experiment tracking, hyperparameter tuning, model registry usage, a live web interface, and performance monitoring with MLflow.

---

## ⚙️ Technologies Used

- Python 3.9+
- Scikit-learn
- XGBoost, LightGBM
- MLflow
- Streamlit
- Pandas

---

## 📁 Project Structure

 main.py # Trains and logs multiple models with MLflow 
 app.py # Streamlit app for live churn predictions 
 monitor.py # Daily model monitoring simulation 
 feature_names.json # Stores training-time feature order 
 requirements.txt # Python dependencies 
 README.md # Project documentation 
 WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset


---

## 🧪 Models Trained

- Random Forest
- Logistic Regression
- XGBoost
- LightGBM
- VotingClassifier (Ensemble of best 3)

Each model was logged to MLflow along with its hyperparameters and performance metrics (accuracy, precision, recall, F1-score).

---

## 🚀 Deployment (Streamlit)

A live web interface is built using **Streamlit**. Users can:

- Input customer details
- Get churn probability instantly
- View prediction explanation

Run it locally:

```bash
streamlit run app.py
