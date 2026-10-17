"""Stage 3: train the model, track it with MLflow, save it with joblib."""
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.utils import get_logger, git_commit, load_params, setup_mlflow

logger = get_logger("model_training")

TRAIN_FILE = "data/processed/train.csv"
MODEL_FILE = "models/banknote_auth_model.joblib"
RUN_ID_FILE = "reports/mlflow_run_id.txt"


def build_model(training: dict):
    kind = training["model_type"]
    seed = training["random_state"]
    if kind == "decision_tree":
        return DecisionTreeClassifier(**training["decision_tree"], random_state=seed)
    if kind == "random_forest":
        return RandomForestClassifier(**training["random_forest"], random_state=seed)
    raise ValueError(f"Unknown model_type: {kind}")


def main():
    params = load_params()
    training, target = params["training"], params["data"]["target"]
    prep = params["preprocessing"]

    logger.info("Starting model training (%s)", training["model_type"])
    train = pd.read_csv(TRAIN_FILE)
    X, y = train.drop(columns=[target]), train[target]

    model = build_model(training)

    cv = StratifiedKFold(
        n_splits=training["cv_folds"],
        shuffle=True,
        random_state=training["random_state"],
    )
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    model.fit(X, y)
    train_accuracy = float(model.score(X, y))

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    logger.info("Model saved to %s", MODEL_FILE)

    setup_mlflow(params)
    with mlflow.start_run() as run:
        mlflow.set_tag("git_commit", git_commit())
        mlflow.log_params(
            {
                "model_type": training["model_type"],
                **{f"model.{k}": v for k, v in training[training["model_type"]].items()},
                "random_state": training["random_state"],
                "cv_folds": training["cv_folds"],
                "test_size": prep["test_size"],
                "drop_duplicates": prep["drop_duplicates"],
                "training_rows": len(X),
            }
        )
        mlflow.log_metrics(
            {
                "cv_accuracy_mean": float(cv_scores.mean()),
                "cv_accuracy_std": float(cv_scores.std()),
                "train_accuracy": train_accuracy,
            }
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                train, source=TRAIN_FILE, name="banknote_train", targets=target
            ),
            context="training",
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X.head(3),
            # MLflow saves with skops; tree models need this type trusted.
            # Safe here because we trained this model ourselves.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            registered_model_name=params["mlflow"]["registered_model_name"],
        )

        with open(RUN_ID_FILE, "w") as file:
            file.write(run.info.run_id)
        logger.info("MLflow run: %s", run.info.run_id)

    logger.info(
        "CV accuracy: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std()
    )


if __name__ == "__main__":
    main()
