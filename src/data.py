"""Data loading and feature engineering utilities."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data(filename: str = "student-mat.csv") -> pd.DataFrame:
    """Load the raw student performance dataset."""
    return pd.read_csv(DATA_DIR / filename, sep=";")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to the raw DataFrame.

    - Creates binary target `at_risk` (G3 < 10).
    - Drops G1, G2, G3 to avoid data leakage.
    - Adds combined features: parent_edu, alcohol, studytime_failures.
    """
    df = df.copy()

    # Target
    df["at_risk"] = (df["G3"] < 10).astype(int)

    # New features
    df["parent_edu"] = df["Medu"] + df["Fedu"]
    df["alcohol"] = df["Dalc"] + df["Walc"]
    df["studytime_failures"] = df["studytime"] * df["failures"]

    # Drop period grades to prevent leakage
    df = df.drop(columns=["G1", "G2", "G3"])

    return df
