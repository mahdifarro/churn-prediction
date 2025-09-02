# api/main.py

import os
import uuid
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from api.schemas import (
    CustomerData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
import logging

# NEW: Import MLflow's client to interact with the model registry
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# --- Error Handling ---


class StructuredError(Exception):
    """Custom exception with structured error information."""

    def __init__(
        self, message: str, error_code: str = "INTERNAL_ERROR", trace_id: str = None
    ):
        self.message = message
        self.error_code = error_code
        self.trace_id = trace_id or str(uuid.uuid4())
        super().__init__(self.message)


def create_error_response(error: StructuredError, status_code: int = 500):
    """Create a structured error response with trace ID."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error.error_code,
                "message": error.message,
                "trace_id": error.trace_id,
            }
        },
        headers={"X-Trace-ID": error.trace_id},
    )


def validate_model_signature(model, expected_features: list):
    """Validate that model signature matches expected features."""
    try:
        # Create a dummy dataframe with expected features
        dummy_data = pd.DataFrame({feature: [0] for feature in expected_features})
        # Try to make a prediction to validate the signature
        model.predict_proba(dummy_data)
        # logger.info("Model signature validation passed")
        return True
    except Exception as e:
        raise StructuredError(
            message=f"Model signature validation failed: {str(e)}",
            error_code="MODEL_SIGNATURE_MISMATCH",
        )


# --- App Setup and Model Loading ---

load_dotenv()

# Set the MLflow tracking URI from the environment variable
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Get model configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_STAGE = os.getenv("MODEL_STAGE")

if not MODEL_NAME or not MODEL_STAGE:
    raise ValueError("MODEL_NAME and MODEL_STAGE environment variables must be set.")

# Expected features based on CustomerData schema
EXPECTED_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# --- NEW: Get Model Version and URI from the Registry ---
try:
    # Initialize the MLflow client
    client = MlflowClient()

    # Get the latest version of the model for the specified stage
    latest_version_info = client.get_latest_versions(
        name=MODEL_NAME, stages=[MODEL_STAGE]
    )[0]

    # Extract the run_id (which we use as the model_version) and the correct model URI
    model_version = latest_version_info.run_id
    model_uri = latest_version_info.source

    # logger.info(f"Loading model '{MODEL_NAME}' version '{model_version}' from stage '{MODEL_STAGE}'.")
    # logger.info(f"Model URI: {model_uri}")
    print(
        f"[DEBUG] Model URI being loaded: {model_uri}"
    )  # DEBUG: Print to stdout for container logs

    # Load the model from the specific URI provided by the registry
    model = mlflow.sklearn.load_model(model_uri)
    # logger.info("Model loaded successfully.")

    # Validate model signature
    validate_model_signature(model, EXPECTED_FEATURES)

except Exception as e:
    error = StructuredError(
        message=f"Failed to load model from registry: {str(e)}",
        error_code="MODEL_LOADING_ERROR",
    )
    # logger.error(f"Startup failed: {error.message} (Trace ID: {error.trace_id})")
    raise RuntimeError(error.message)


# --- Global Exception Handler ---
@app.exception_handler(StructuredError)
async def structured_error_handler(request, exc: StructuredError):
    return create_error_response(exc, 400)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    error = StructuredError(message=str(exc.detail), error_code="HTTP_ERROR")
    return create_error_response(error, exc.status_code)


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    error = StructuredError(
        message=f"Unexpected error: {str(exc)}", error_code="INTERNAL_ERROR"
    )
    # logger.error(f"Unhandled exception: {error.message} (Trace ID: {error.trace_id})")
    return create_error_response(error, 500)


# --- API Endpoints ---


@app.get("/health", status_code=200, summary="Health Check")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/version", summary="Get Model Version")
def version():
    """Endpoint to get the version of the currently loaded model."""
    return {
        "model_version": model_version,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict Churn")
def predict_churn(
    customer_data: CustomerData,
    threshold: float = Query(
        0.5, ge=0.0, le=1.0, description="Decision threshold for churn prediction"
    ),
):
    """
    Takes customer data as input and returns the churn prediction and probability.

    Args:
        customer_data: Customer information for prediction
        threshold: Decision threshold (default: 0.5). Customers with probability > threshold are predicted as 'Churn'
    """
    try:
        input_df = pd.DataFrame([customer_data.model_dump()])
        pred_proba = model.predict_proba(input_df)[0]
        churn_probability = pred_proba[1]
        prediction = "Churn" if churn_probability > threshold else "No Churn"

        return PredictionResponse(
            model_version=model_version,
            prediction=prediction,
            probability=float(churn_probability),
        )
    except Exception as e:
        raise StructuredError(
            message=f"Prediction failed: {str(e)}", error_code="PREDICTION_ERROR"
        )


@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    summary="Batch Predict Churn",
)
def predict_churn_batch(
    batch_request: BatchPredictionRequest,
    threshold: float = Query(
        0.5, ge=0.0, le=1.0, description="Decision threshold for churn prediction"
    ),
):
    """
    Takes a list of customer data and returns batch predictions.

    Args:
        batch_request: List of customer information for batch prediction
        threshold: Decision threshold (default: 0.5). Customers with probability > threshold are predicted as 'Churn'
    """
    try:
        if not batch_request.customers:
            raise StructuredError(
                message="No customer data provided in batch request",
                error_code="EMPTY_BATCH_REQUEST",
            )

        if len(batch_request.customers) > 1000:  # Reasonable limit
            raise StructuredError(
                message="Batch size too large. Maximum 1000 customers per request.",
                error_code="BATCH_SIZE_EXCEEDED",
            )

        # Convert all customer data to DataFrame
        input_data = [customer.model_dump() for customer in batch_request.customers]
        input_df = pd.DataFrame(input_data)

        # Get predictions for all customers
        pred_probas = model.predict_proba(input_df)
        churn_probabilities = pred_probas[:, 1]

        # Create predictions list
        predictions = []
        for prob in churn_probabilities:
            prediction = "Churn" if prob > threshold else "No Churn"
            predictions.append({"prediction": prediction, "probability": float(prob)})

        return BatchPredictionResponse(
            model_version=model_version, predictions=predictions
        )

    except StructuredError:
        raise  # Re-raise structured errors as-is
    except Exception as e:
        raise StructuredError(
            message=f"Batch prediction failed: {str(e)}",
            error_code="BATCH_PREDICTION_ERROR",
        )
