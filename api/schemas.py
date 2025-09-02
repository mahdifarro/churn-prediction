# api/schemas.py

# Pydantic is used for data validation. We define the structure and data types
# of the data we expect to receive in an API request.
from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """
    Defines the data schema for a single customer prediction request.
    Each attribute corresponds to a feature our model was trained on.
    Pydantic will automatically validate that the incoming data matches these types.
    For example, it will ensure 'tenure' is an integer and 'gender' is a string.
    """

    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

    # This is an example of what the JSON input for one customer will look like.
    # We can use this for testing our API later.
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "tenure": 1,
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }


# NEW: Define the schema for the prediction response.
# This ensures the API always returns data in a consistent and validated format.
class PredictionResponse(BaseModel):
    model_version: str
    prediction: str
    probability: float = Field(
        ..., ge=0, le=1
    )  # Ensures probability is between 0 and 1


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""

    customers: list[CustomerData]


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""

    model_version: str
    predictions: list[dict]  # List of {prediction, probability} for each customer
