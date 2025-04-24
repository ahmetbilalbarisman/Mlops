import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
import xgboost as xgb
import lightgbm as lgb
from itertools import product

model_configs = [
    {
        "name": "RandomForest",
        "class": RandomForestClassifier,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [10, None],
            "class_weight": ["balanced"]
        }
    },
    {
        "name": "LogisticRegression",
        "class": LogisticRegression,
        "param_grid": {
            "C": [0.1, 1.0],
            "penalty": ["l2"],
            "class_weight": ["balanced"],
            "solver": ["liblinear"],
            "max_iter": [500]
        }
    },
    {
        "name": "XGBoost",
        "class": xgb.XGBClassifier,
        "param_grid": {
            "n_estimators": [100],
            "max_depth": [5, 10],
            "learning_rate": [0.1],
            "eval_metric": ["logloss"],
            "use_label_encoder": [False]
        }
    },
    {
        "name": "LightGBM",
        "class": lgb.LGBMClassifier,
        "param_grid": {
            "n_estimators": [100],
            "max_depth": [5, 10],
            "learning_rate": [0.1],
            "boosting_type": ["gbdt"]
        }
    }
]


def prepare_data(path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    df = df.dropna()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = pd.get_dummies(df.drop(columns=['customerID', 'Churn']), drop_first=True)
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_and_log(model, name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_params(model.get_params())
        mlflow.log_param("model_name", name)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": recall,
            "f1_score": f1
        })
        mlflow.set_tag("type", "ensemble_comparison")
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train.iloc[:1])

        print(f"[{name}] acc: {acc:.4f} | prec: {prec:.4f} | recall: {recall:.4f} | f1: {f1:.4f}")

def main():
    mlflow.set_experiment("TelcoChurn_MultiModel_ParamSearch")
    X_train, X_test, y_train, y_test = prepare_data()

    for config in model_configs:
        keys, values = zip(*config["param_grid"].items())
        for combination in product(*values):
            params = dict(zip(keys, combination))
            model = config["class"](**params)
            run_name = f"{config['name']}_" + "_".join(f"{k}={v}" for k, v in params.items())
            evaluate_and_log(model, run_name, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
