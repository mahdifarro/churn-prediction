# Makefile for Churn Prediction Project

# Define the python interpreter from the virtual environment
VENV_PYTHON = venv/bin/python

.PHONY: setup test train serve

# Create a virtual environment and install dependencies
setup:
	@echo "Creating virtual environment..."
	python -m venv .churn_env

install:
	@echo "Installing dependencies from requirements.txt..."
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

# Run unit tests
test:
	@echo "Running tests..."
	python -m pytest

# Run the model training pipeline
train:
	@echo "Running model training pipeline..."
	python -m src.train

# Serve the model with FastAPI
serve:
	@echo "Serving model API on http://127.0.0.1:8000..."
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start MLflow server
mlflow:
	@echo "Starting MLflow server on http://127.0.0.1:5000..."
	mlflow ui