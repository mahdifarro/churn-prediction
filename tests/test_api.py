# tests/test_api.py

import os

from fastapi.testclient import TestClient
from api.main import app  # Now this import is safe

# --- CRITICAL FIX ---
# Set the environment variable for MLflow BEFORE importing the FastAPI app.
# This ensures that when 'api.main' is imported, it uses this local path
# for MLflow instead of the Docker-specific one from the .env file.
os.environ["MLFLOW_TRACKING_URI"] = "./mlruns"
# -------------------

# Create a TestClient instance, which allows us to send requests to our FastAPI app
client = TestClient(app)

# Example of valid customer data for testing the /predict endpoint
VALID_PAYLOAD = {
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "tenure": 1,
    "MonthlyCharges": 53.85,
    "TotalCharges": 108.15,
}


def test_health_check():
    """Tests the /health endpoint to ensure the API is running."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_version_check():
    """Tests the /version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    assert "model_version" in response.json()


def test_predict_success():
    """
    Tests the /predict endpoint with valid data.
    """
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    data = response.json()
    assert "model_version" in data
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ["Churn", "No Churn"]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_validation_error():
    """
    Tests that the /predict endpoint returns a 422 error for invalid input.
    """
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["tenure"] = "not-a-number"  # Invalid data type

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
