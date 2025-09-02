# tests/test_api.py

import os
from fastapi.testclient import TestClient
from api.main import app

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"  # or your test server

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
    # Accept both keys: 'status' and 'model_loaded' in the response
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_version_check():
    """Tests the /version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    # We can't know the exact version, but we can check if the key exists
    assert "model_version" in response.json()


def test_predict_success():
    """
    Tests the /predict endpoint with valid data.
    This is an integration test for a successful round-trip prediction.
    """
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    # Check the response contract (schema)
    data = response.json()
    assert "model_version" in data
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in ["Churn", "No Churn"]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_validation_error():
    """
    Tests that the /predict endpoint returns a 422 error for invalid input.
    This checks our Pydantic input validation.
    """
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["tenure"] = "not-a-number"  # Invalid data type

    response = client.post("/predict", json=invalid_payload)
    # 422 Unprocessable Entity is the correct code for a Pydantic validation error
    assert response.status_code == 422
