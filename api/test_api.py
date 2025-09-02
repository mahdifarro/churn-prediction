# test_api.py - Quick test script for the enhanced API

import requests
import json

# API base URL (assuming it's running locally)
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_version():
    """Test version endpoint"""
    response = requests.get(f"{BASE_URL}/version")
    print("Version:", response.json())

def test_single_prediction():
    """Test single prediction with custom threshold"""
    customer_data = {
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
        "TotalCharges": 29.85
    }
    
    # Test with default threshold (0.5)
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print("Prediction (default threshold):", response.json())
    
    # Test with custom threshold (0.3)
    response = requests.post(f"{BASE_URL}/predict?threshold=0.3", json=customer_data)
    print("Prediction (threshold=0.3):", response.json())

def test_batch_prediction():
    """Test batch prediction"""
    batch_data = {
        "customers": [
            {
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
                "TotalCharges": 29.85
            },
            {
                "gender": "Female",
                "Partner": "No",
                "Dependents": "Yes",
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "tenure": 36,
                "MonthlyCharges": 85.25,
                "TotalCharges": 3070.00
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict-batch", json=batch_data)
    print("Batch Prediction:", json.dumps(response.json(), indent=2))

def test_error_handling():
    """Test error handling with invalid data"""
    invalid_data = {
        "gender": "Male",
        "tenure": "invalid_number",  # This should cause an error
        "MonthlyCharges": 29.85
        # Missing required fields
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
        print("Error Response:", response.json())
        print("Trace ID from header:", response.headers.get("X-Trace-ID"))
    except Exception as e:
        print("Request failed:", str(e))

if __name__ == "__main__":
    print("Testing Enhanced Churn Prediction API")
    print("="*50)
    
    try:
        test_health()
        print()
        
        test_version()
        print()
        
        test_single_prediction()
        print()
        
        test_batch_prediction()
        print()
        
        print("Testing error handling:")
        test_error_handling()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the API is running on localhost:8000")
    except Exception as e:
        print(f"Test failed: {str(e)}")
