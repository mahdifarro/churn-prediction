# src/train.py (version with LogReg, LGBM, and XGBoost)

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier  # NEW: Import XGBoost
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Explicit imports from src.config to avoid F405 errors
from src.config import (
    DATA_URL,
    PLOTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    PIPELINE_ARTIFACT_NAME,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    CV_FOLDS,
    RANDOM_STATE,
    LGBM_PARAMS,
    XGB_PARAMS,
)


# ==============================================================================
# 2. HELPER FUNCTIONS FOR PLOTTING
# ==============================================================================


def ensure_plots_directory():
    """Create plots directory structure if it doesn't exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # Create subdirectory for current training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(PLOTS_DIR, f"training_session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def plot_confusion_matrix(y_true, y_pred, run_id, plots_session_dir):
    """Plot confusion matrix and save to plots directory."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plot_path = os.path.join(plots_session_dir, f"confusion_matrix_{run_id}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(plot_path, "plots")
    print(f"Confusion matrix saved to: {plot_path}")


def plot_feature_importance(pipeline, run_id, plots_session_dir):
    """Plot feature importance and save to plots directory."""
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        # This code works for both LightGBM and XGBoost as they share the .feature_importances_ attribute
        model = pipeline.named_steps["classifier"]
        importances = model.feature_importances_

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        importance_df = importance_df.sort_values("importance", ascending=False).head(
            20
        )

        plt.figure(figsize=(10, 8))
        sns.barplot(x="importance", y="feature", data=importance_df)
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plot_path = os.path.join(plots_session_dir, f"feature_importance_{run_id}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(plot_path, "plots")
        print(f"Feature importance plot saved to: {plot_path}")
    except AttributeError:
        print(
            "Feature importance plot is not available for this model type (e.g., Logistic Regression)."
        )


# ==============================================================================
# 3. MAIN TRAINING FUNCTION (Corrected Version)
# ==============================================================================
def run_training():
    """Main function to run the entire training and experimentation pipeline."""

    # --- THIS IS THE CRITICAL ADDITION ---
    # Set the MLflow tracking URI to connect to the server when running locally.
    # This ensures model paths are saved correctly for Docker.
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"Logging to remote MLflow server: {MLFLOW_TRACKING_URI}")
    else:
        print("Logging to local MLflow directory.")
    # --- END OF CRITICAL ADDITION ---

    # Setup plots directory (your excellent addition)
    plots_session_dir = ensure_plots_directory()
    print(f"Plots will be saved to: {plots_session_dir}")

    # Load and prepare data (your code is correct)
    df = pd.read_csv(DATA_URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    # Define preprocessor (your code is correct)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Define models to run (your code is correct)
    models_to_run = [
        {"name": "LogisticRegression", "params": {}},
        {"name": "LightGBM", "params": LGBM_PARAMS[0]},
        {"name": "LightGBM", "params": LGBM_PARAMS[1]},
        {"name": "LightGBM", "params": LGBM_PARAMS[2]},
        {"name": "XGBoost", "params": XGB_PARAMS[0]},
        {"name": "XGBoost", "params": XGB_PARAMS[1]},
    ]

    # MLflow Experiment (the rest of your logic is correct)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Full_Model_Comparison") as parent_run:
        print(f"Parent Run ID: {parent_run.info.run_id}")

        for model_config in models_to_run:
            model_name = model_config["name"]
            params = model_config["params"]

            with mlflow.start_run(
                run_name=f"{model_name}_run", nested=True
            ) as child_run:
                run_id = child_run.info.run_id
                print(f"\n--- Starting nested run for {model_name}: {run_id} ---")

                if model_name == "LogisticRegression":
                    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
                elif model_name == "LightGBM":
                    model = LGBMClassifier(
                        random_state=RANDOM_STATE, verbosity=-1, **params
                    )
                elif model_name == "XGBoost":
                    model = XGBClassifier(
                        random_state=RANDOM_STATE, use_label_encoder=False, **params
                    )

                full_pipeline = Pipeline(
                    steps=[("preprocessor", preprocessor), ("classifier", model)]
                )

                mlflow.log_param("model_name", model_name)
                mlflow.log_params(params)
                cv = StratifiedKFold(
                    n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
                )
                y_pred_cv = cross_val_predict(
                    full_pipeline, X, y, cv=cv, method="predict"
                )
                y_pred_proba_cv = cross_val_predict(
                    full_pipeline, X, y, cv=cv, method="predict_proba"
                )[:, 1]

                mean_auc = roc_auc_score(y, y_pred_proba_cv)
                mlflow.log_metric("mean_roc_auc", mean_auc)
                print(f"Run {run_id} Mean ROC AUC: {mean_auc:.4f}")

                plot_confusion_matrix(y, y_pred_cv, run_id, plots_session_dir)
                full_pipeline.fit(X, y)
                plot_feature_importance(full_pipeline, run_id, plots_session_dir)

                mlflow.sklearn.log_model(
                    sk_model=full_pipeline, artifact_path=PIPELINE_ARTIFACT_NAME
                )


if __name__ == "__main__":
    run_training()
