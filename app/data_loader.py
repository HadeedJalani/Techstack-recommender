import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = [
    "id",
    "name",
    "category",
    "frontend",
    "backend",
    "database",
    "deployment",
    "tags",
    "description",
    "popularity",
]


def load_tech_stacks(path: Path) -> pd.DataFrame:
    """Load and validate the content catalog used by the recommender."""

    df = pd.read_csv(path).fillna("")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty; at least one item is required.")

    if df["id"].duplicated().any():
        raise ValueError("Dataset contains duplicate item ids.")

    return df
