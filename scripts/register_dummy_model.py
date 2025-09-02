import mlflow
from sklearn.dummy import DummyClassifier
import numpy as np

mlflow.set_tracking_uri("./mlruns")
model_name = "churn-prediction-champion"

# Create and train a dummy model
X = np.array([[0], [1]])
y = np.array([0, 1])
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X, y)

# Log and register the model
with mlflow.start_run():
    mlflow.sklearn.log_model(dummy, "model")
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)
