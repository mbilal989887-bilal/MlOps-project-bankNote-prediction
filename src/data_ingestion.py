"""Stage 1: validate the source data and save a clean raw copy."""
import json
import os

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.utils import get_logger, load_params

logger = get_logger("data_ingestion")
REPORT_FILE = "reports/data_report.json"


def validate(df: pd.DataFrame, features: list, target: str):
    """Check columns, types and labels. Returns (clean_df, report)."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    missing_cols = (set(features) | {target}) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    df = df[features + [target]]

    for col in features:
        if not is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")

    labels = set(df[target].dropna().unique())
    if not labels <= {0, 1}:
        raise ValueError(f"Target must contain only 0 and 1, found {labels}")

    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_0_rows": int((df[target] == 0).sum()),
        "class_1_rows": int((df[target] == 1).sum()),
    }
    return df, report


def main():
    params = load_params()
    data = params["data"]

    logger.info("Starting data ingestion")
    if not os.path.exists(data["source_file"]):
        raise FileNotFoundError(f"Dataset not found: {data['source_file']}")

    df, report = validate(
        pd.read_csv(data["source_file"]), data["features"], data["target"]
    )

    os.makedirs(os.path.dirname(data["raw_file"]), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    df.to_csv(data["raw_file"], index=False)
    with open(REPORT_FILE, "w") as file:
        json.dump(report, file, indent=4)

    logger.info("Data report: %s", report)
    logger.info("Data ingestion completed")


if __name__ == "__main__":
    main()
