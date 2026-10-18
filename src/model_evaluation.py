"""Stage 4: evaluate on the test set, save metrics, log to the same MLflow run."""
import json

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger, load_params, setup_mlflow

logger = get_logger("model_evaluation")

TEST_FILE = "data/processed/test.csv"
MODEL_FILE = "models/banknote_auth_model.joblib"
METRICS_FILE = "reports/metrics.json"
CONFUSION_FILE = "reports/confusion_matrix.csv"
RUN_ID_FILE = "reports/mlflow_run_id.txt"


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def main():
    params = load_params()
    target = params["data"]["target"]

    logger.info("Starting model evaluation")
    model = joblib.load(MODEL_FILE)
    test = pd.read_csv(TEST_FILE)
    X_test, y_test = test.drop(columns=[target]), test[target]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    with open(METRICS_FILE, "w") as file:
        json.dump(metrics, file, indent=4)
    pd.DataFrame({"actual": y_test.values, "predicted": y_pred}).to_csv(
        CONFUSION_FILE, index=False
    )
    logger.info("Metrics: %s", metrics)

    with open(RUN_ID_FILE) as file:
        run_id = file.read().strip()

    setup_mlflow(params)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_input(
            mlflow.data.from_pandas(
                test, source=TEST_FILE, name="banknote_test", targets=target
            ),
            context="testing",
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(METRICS_FILE)
        mlflow.log_artifact(CONFUSION_FILE)

    # Quality gate: stop the pipeline (and CI) if the model is too weak.
    minimum = params["evaluation"]["min_accuracy"]
    if metrics["accuracy"] < minimum:
        raise ValueError(
            f"Accuracy {metrics['accuracy']} is below the minimum {minimum}"
        )
    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()
