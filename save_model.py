

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="TelcoChurnBestModel",
    version=1,  # registry'de oluşan versiyon numarası
    stage="Production"
)
