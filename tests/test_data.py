import pandas as pd
import pytest

from src.data_ingestion import validate
from src.data_preprocessing import clean, split


def test_validate_accepts_source_data(params, source_df):
    data = params["data"]
    df, report = validate(source_df, data["features"], data["target"])
    assert report["rows"] == len(source_df)
    assert report["class_0_rows"] + report["class_1_rows"] == report["rows"]


def test_validate_rejects_missing_column(params, source_df):
    data = params["data"]
    with pytest.raises(ValueError):
        validate(source_df.drop(columns=["entropy"]), data["features"], data["target"])


def test_validate_rejects_bad_labels(params, source_df):
    data = params["data"]
    bad = source_df.copy()
    bad.loc[0, data["target"]] = 5
    with pytest.raises(ValueError):
        validate(bad, data["features"], data["target"])


def test_clean_removes_duplicates_and_missing():
    df = pd.DataFrame({"a": [1, 1, None, 2], "class": [0, 0, 1, 1]})
    assert len(clean(df, drop_duplicates=True)) == 2
    assert len(clean(df, drop_duplicates=False)) == 3


def test_split_is_stratified_and_has_no_overlap(params, source_df):
    target = params["data"]["target"]
    df = clean(source_df, drop_duplicates=True)
    train, test = split(df, target, test_size=0.2, random_state=42)
    assert len(train) + len(test) == len(df)
    assert abs(train[target].mean() - test[target].mean()) < 0.03
