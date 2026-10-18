import joblib
import pandas as pd
import pytest

from src.model_training import build_model
from src.utils import load_params


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def source_df(params):
    df = pd.read_csv(params["data"]["source_file"])
    df.columns = df.columns.str.strip().str.lower()
    return df


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, params, source_df):
    """Train a small model just for the tests (no need to run the pipeline)."""
    target = params["data"]["target"]
    model = build_model(params["training"])
    model.fit(source_df.drop(columns=[target]), source_df[target])
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(model, path)
    return path
