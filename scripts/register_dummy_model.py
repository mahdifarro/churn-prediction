import mlflow
from sklearn.dummy import DummyClassifier
import numpy as np
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("./mlruns")
model_name = "churn-prediction-champion"

# Create and train a dummy model
X = np.array([[0], [1]])
y = np.array([0, 1])
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X, y)

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(dummy, "model")
    result = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
    version = int(result.version)

# Transition the registered model version to "Staging"
client = MlflowClient()
client.transition_model_version_stage(name=model_name, version=version, stage="Staging")
