# Customer Churn Prediction - MLOps Pipeline

A complete MLOps pipeline for predicting customer churn using machine learning, featuring automated training, model registry, REST API, and containerized deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [MLflow Integration](#mlflow-integration)
- [Testing](#testing)
- [Project Structure](#project-structure)

## Project Overview

This project implements an end-to-end MLOps pipeline for customer churn prediction with the following capabilities:

- **Multiple ML Models**: Logistic Regression, LightGBM, and XGBoost
- **Automated Training**: Cross-validation with hyperparameter tuning
- **Model Registry**: MLflow model versioning and staging
- **REST API**: FastAPI-based prediction service
- **Containerization**: Docker support for development and production
- **Monitoring**: Comprehensive logging and error handling
- **Testing**: Unit tests for API and data processing

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Training Pipeline │───▶│   MLflow Server │
│  (IBM Dataset)  │    │   (src/train.py)   │    │  (Model Registry)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Frontend    │◀───│   FastAPI API   │◀───│  Model Loading  │
│   (External)    │    │  (api/main.py)  │    │   (Registry)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Git

### 1. Clone and Setup

```bash
git clone <repository-url>
cd churn-prediction

# Create virtual environment
python -m venv .churn_env
# Windows
.churn_env\Scripts\activate
# macOS/Linux
source .churn_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Using make
make train

# Or directly
python -m src.train
```

### 3. Start MLflow Server

```bash
# Using make
make mlflow

# Or directly
mlflow ui
```

### 4. Register a Model (for testing)

```bash
python scripts/register_dummy_model.py
```

### 5. Start API Server

```bash
# Set environment variables
set MODEL_NAME=churn-prediction-champion
set MODEL_STAGE=Staging

# Start API
make serve
# Or: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Test the API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "tenure": 12,
    "MonthlyCharges": 53.85,
    "TotalCharges": 646.2
  }'
```

## Dataset

### Overview

This dataset provides information about customers from a fictional telecommunications company. It includes customer demographics, account information, subscribed services, and whether they have churned or not.

- **Rows:** 7,043
- **Columns:** 21
- **Task:** Binary Classification
- **Target:** Customer Churn (Yes/No)

### Source

- **URL:** [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Provider:** IBM Watson Analytics

### Features

- **Categorical Features (15):** gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
- **Numerical Features (3):** tenure, MonthlyCharges, TotalCharges
- **Target Variable:** Churn (Yes/No)

## Machine Learning Models

The pipeline supports multiple algorithms with automated hyperparameter tuning:

### 1. Logistic Regression
- **Use Case:** Baseline model, interpretable results
- **Features:** L2 regularization, balanced class weights

### 2. LightGBM
- **Use Case:** Gradient boosting, handles categorical features well
- **Hyperparameters:** Multiple configurations for learning rate, depth, features

### 3. XGBoost
- **Use Case:** Robust gradient boosting, often high performance
- **Hyperparameters:** Tuned for learning rate, depth, regularization

### Model Selection Criteria

- **Primary Metric:** ROC-AUC (Area Under the Curve)
- **Cross-Validation:** 5-fold stratified cross-validation
- **Target Performance:** ROC-AUC > 0.82 for production deployment

## API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```http
GET /health
```
Returns API health status and model loading status.

#### Model Version
```http
GET /version
```
Returns current model version and metadata.

#### Single Prediction
```http
POST /predict
```
Predicts churn for a single customer.

**Request Body:**
```json
{
  "gender": "string",
  "Partner": "string",
  "Dependents": "string",
  "PhoneService": "string",
  "MultipleLines": "string",
  "InternetService": "string",
  "OnlineSecurity": "string",
  "OnlineBackup": "string",
  "DeviceProtection": "string",
  "TechSupport": "string",
  "StreamingTV": "string",
  "StreamingMovies": "string",
  "Contract": "string",
  "PaperlessBilling": "string",
  "PaymentMethod": "string",
  "tenure": 0,
  "MonthlyCharges": 0.0,
  "TotalCharges": 0.0
}
```

**Response:**
```json
{
  "model_version": "string",
  "prediction": "Churn|No Churn",
  "probability": 0.85
}
```

#### Batch Prediction
```http
POST /predict-batch
```
Predicts churn for multiple customers in a single request.

### Query Parameters

- `threshold` (float, optional): Decision threshold for churn prediction (0.0-1.0, default: 0.5)

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Start all services (MLflow + API + Training)
docker-compose up --build

# Start only API and MLflow
docker-compose up api mlflow

# Run training in container
docker-compose run --rm training
```

### Manual Docker Build

```bash
# Build the image
docker build -f docker/Dockerfile -t churn-prediction .

# Run API container
docker run -p 8000:8000 \
  -e MODEL_NAME=churn-prediction-champion \
  -e MODEL_STAGE=Staging \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  churn-prediction
```

### Services

- **MLflow Server:** http://localhost:5000
- **API Server:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## MLflow Integration

### Experiment Tracking

- **Experiment Name:** `Churn_Prediction_Baseline`
- **Metrics Logged:** ROC-AUC, Cross-validation scores
- **Artifacts:** Model pipelines, confusion matrices, feature importance plots
- **Parameters:** Model hyperparameters, preprocessing settings

### Model Registry

- **Model Name:** `churn-prediction-champion`
- **Stages:** `None` → `Staging` → `Production`
- **Versioning:** Automatic version incrementing
- **Metadata:** Model metrics, tags, descriptions

### Accessing MLflow UI

```bash
# Local development
mlflow ui --backend-store-uri ./mlruns

# Docker deployment
# Navigate to http://localhost:5000
```

## Testing

### Run All Tests

```bash

# Using pytest directly
pytest

```

### Test Categories

1. **API Tests** (`tests/test_api.py`)
   - Health check endpoint
   - Version endpoint
   - Single prediction endpoint
   - Batch prediction endpoint
   - Input validation
   - Error handling

2. **Data Tests** (`tests/test_data.py`)
   - Data loading and preprocessing
   - Feature validation
   - Data quality checks

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/health

# Get model version
curl http://localhost:8000/version

# Interactive API documentation
# Navigate to http://localhost:8000/docs
```

## Project Structure

```
churn-prediction/
├── api/                      # FastAPI application
│   ├── main.py               # API endpoints and model loading
│   └── schemas.py            # Pydantic data models
├── data/                     # Data storage
│   ├── processed/            # Processed datasets
│   └── raw/                  # Raw datasets
├── docker/                   # Docker configuration
│   └── Dockerfile            # Container definition
├── docs/                     # Documentation
├── mlruns/                   # MLflow tracking data
├── notebooks/                # Jupyter notebooks
│   └── 01-eda.ipynb         # Exploratory data analysis
├── plots/                    # Generated plots and visualizations
├── scripts/                  # Utility scripts
│   └── register_dummy_model.py # Model registration helper
├── src/                      # Core source code
│   ├── __init__.py
│   ├── config.py             # Configuration and parameters
│   ├── data_processing.py    # Data preprocessing utilities
│   ├── main.py               # Entry point
│   ├── model.py              # Model definitions
│   └── train.py              # Training pipeline
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_api.py           # API endpoint tests
│   └── test_data.py          # Data processing tests
├── docker-compose.yml        # Multi-service Docker setup
├── Makefile                  # Build and run commands
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-model

# Make changes
# ... edit files ...

# Run tests
make test

# Train models
make train

# Test API
make serve
```

### 2. Model Experimentation

```bash
# Modify hyperparameters in src/config.py
# Run training with MLflow tracking
python -m src.train

# View results in MLflow UI
make mlflow
```

### 3. Model Deployment

```bash
# Register best model to staging
# (Use MLflow UI or API)

# Test staging model
MODEL_STAGE=Staging make serve

# Promote to production
# (Use MLflow UI to transition model stage)
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# Model Configuration
MODEL_NAME=churn-prediction-champion
MODEL_STAGE=Staging

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Model Parameters

Edit `src/config.py` to modify:

- Data sources and URLs
- Feature definitions
- Model hyperparameters
- Training parameters
- MLflow experiment settings

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
