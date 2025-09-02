# src/config.py

import os

# --- File Paths and URLs ---
# A single source of truth for the data URL. If it changes, we only update it here.
DATA_URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'

# --- Directory Paths ---
# Base project directory (parent of src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Plots directory for saving training plots
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# --- MLflow Settings ---
# The name of the experiment where all our runs will be logged.
MLFLOW_EXPERIMENT_NAME = "Churn_Prediction_Baseline"
# The name of the folder where the saved model will be stored within the MLflow run.
PIPELINE_ARTIFACT_NAME = "model_pipeline"

# --- Feature lists ---
# Define which columns are categorical and which are numerical
# 'customerID' is an identifier and 'Churn' is the target, so we exclude them from features
TARGET_COLUMN = 'Churn'

CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# --- Model Training ---
# Storing model and cross-validation parameters here makes them easy to change for new experiments.
MODEL_NAME = "LogisticRegression"
CV_FOLDS = 5
# Using a fixed random state ensures that our results are reproducible every time we run the script.
RANDOM_STATE = 42


# # --- Experimentation ---
# # A dictionary of LightGBM hyperparameters to try.
# # We will run one experiment for each set of parameters in this list.
# LGBM_PARAMS = [
#     {
#         'learning_rate': 0.1,
#         'n_estimators': 200,
#         'max_depth': 5,
#         'num_leaves': 31
#     },
#     {
#         'learning_rate': 0.05,
#         'n_estimators': 300,
#         'max_depth': 7,
#         'num_leaves': 50
#     }
# ]


# src/config.py (replace the old LGBM_PARAMS with this)

# --- Experimentation ---
# Let's try a wider range of hyperparameters to find a better combination.
LGBM_PARAMS = [
    # Run 1: Deeper trees, more estimators
    {
        'learning_rate': 0.05,
        'n_estimators': 400,
        'max_depth': 8,
        'num_leaves': 60
    },
    # Run 2: Shallower trees, slower learning rate (often robust)
    {
        'learning_rate': 0.02,
        'n_estimators': 500,
        'max_depth': 5,
        'num_leaves': 20
    },
    # Run 3: A more aggressive, faster-learning model
    {
        'learning_rate': 0.1,
        'n_estimators': 200,
        'max_depth': 6,
        'num_leaves': 40
    }
]


# src/config.py (add this at the end)

# --- XGBoost Experimentation ---
# A dictionary of XGBoost hyperparameters to try.
XGB_PARAMS = [
    {
        'learning_rate': 0.1,
        'n_estimators': 200,
        'max_depth': 5,
        'eval_metric': 'logloss'
    },
    {
        'learning_rate': 0.05,
        'n_estimators': 300,
        'max_depth': 7,
        'eval_metric': 'logloss'
    }
]