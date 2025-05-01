import pandas as pd
import json
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === 📁 1. Model ve Feature Yapısını Yükle ===
MODEL_NAME = "TelcoChurnVotingEnsemble"
MODEL_VERSION = 2
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

with open("feature_names.json", "r") as f:
    expected_features = json.load(f)

# === 📊 2. Veri Setini Hazırla ===
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = pd.get_dummies(df.drop(columns=['customerID', 'Churn']), drop_first=True)
y = df['Churn']

# Eksik sütunları tamamla
for col in expected_features:
    if col not in X.columns:
        X[col] = 0
X = X[expected_features]

# === 🔮 3. Tahmin Yap ve Metrikleri Hesapla ===
y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# === 📝 4. MLflow'a Metrikleri Logla ===
with mlflow.start_run(run_name="daily_monitoring"):
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })
    mlflow.set_tag("monitor_type", "simulation")
    mlflow.set_tag("source", "monitor.py")

print("✅ Model performance logged to MLflow.")
