"""Stage 2: clean the data and split it into train and test files."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger, load_params

logger = get_logger("data_preprocessing")
PROCESSED_DIR = "data/processed"


def clean(df: pd.DataFrame, drop_duplicates: bool) -> pd.DataFrame:
    df = df.dropna()
    if drop_duplicates:
        df = df.drop_duplicates()
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, target: str, test_size: float, random_state: int):
    """Stratified split so train and test keep the same class balance."""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )


def main():
    params = load_params()
    data, prep = params["data"], params["preprocessing"]

    logger.info("Starting data preprocessing")
    df = pd.read_csv(data["raw_file"])
    before = len(df)
    df = clean(df, prep["drop_duplicates"])
    logger.info("Removed %d rows (missing or duplicate)", before - len(df))

    train, test = split(
        df, data["target"], prep["test_size"], prep["random_state"]
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    logger.info("Train shape: %s, test shape: %s", train.shape, test.shape)


if __name__ == "__main__":
    main()
