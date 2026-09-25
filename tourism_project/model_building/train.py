import pandas as pd
import numpy as np
import os
import joblib
import mlflow

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb

# Load splits
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest  = pd.read_csv("ytest.csv").values.ravel()

# Identify column types
categorical_cols = Xtrain.select_dtypes(include=["object"]).columns.tolist()
numerical_cols   = Xtrain.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numerical_cols),
    (OneHotEncoder(handle_unknown="ignore"), categorical_cols)
)

# Model + pipeline
model = xgb.XGBClassifier(
    use_label_encoder=False, eval_metric="logloss", random_state=42
)
pipeline = make_pipeline(preprocessor, model)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.01, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
}


# ngrok,mflow public tunnel url
#mlflow_public_url = "https://laxative-wolverine-pessimist.ngrok-free.dev/"
# It reads the variable injected by the pipeline.yml
# Falls back to local if the environment variable isn't found
# mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

mlflow_public_url = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow_experiment_name = "test_aml_and_mlops"
  

# MLflow tracking
mlflow.set_tracking_uri(mlflow_public_url)
# mlflow.set_experiment("tourism_wellness_package")
mlflow.set_experiment(mlflow_experiment_name)

print("MLflow experiment name:", mlflow_experiment_name)
print("MLflow UI is available at:", mlflow_public_url)

with mlflow.start_run(run_name="xgb_gridsearch"):
    grid = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
    )
    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_
    ypred = best_model.predict(Xtest)

    acc = accuracy_score(ytest, ypred)
    f1  = f1_score(ytest, ypred)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(best_model, "best_model")

    print("Best params:", grid.best_params_)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(ytest, ypred))

# Save best model for deployment
os.makedirs("tourism_project/deployment", exist_ok=True)
joblib.dump(best_model, "tourism_project/deployment/model.joblib")
print("Model saved to tourism_project/deployment/model.joblib")
