import logging
import os
import subprocess

import mlflow
import yaml

PARAMS_FILE = "params.yaml"
LOG_DIR = "logs"


def load_params(path: str = PARAMS_FILE) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to the console and to logs/<name>.log."""
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        for handler in (
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(LOG_DIR, f"{name}.log")),
        ):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


def setup_mlflow(params: dict) -> None:
    """Use MLFLOW_TRACKING_URI if set, otherwise a local sqlite database."""
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment(params["mlflow"]["experiment_name"])


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"
