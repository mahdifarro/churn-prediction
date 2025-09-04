# scripts/register_dummy_model.py

import mlflow
from sklearn.dummy import DummyClassifier
import numpy as np
from mlflow.tracking import MlflowClient

# Set the tracking URI to use a local directory
mlflow.set_tracking_uri("./mlruns")
model_name = "churn-prediction-champion"

# --- THIS IS THE NEW LINE ---
# Explicitly create or set the experiment. If it doesn't exist, MLflow will create it.
mlflow.set_experiment("Dummy Model For Tests")

# Create and train a dummy model
X = np.array([[0], [1]])
y = np.array([0, 1])
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X, y)

# Start a run within the specified experiment
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(dummy, "model")
    result = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
    version = int(result.version)

# Transition the registered model version to "Staging"
client = MlflowClient()
client.transition_model_version_stage(name=model_name, version=version, stage="Staging")

print("Dummy model created and registered successfully for testing.")
