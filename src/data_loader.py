"""
Data loading and preparation utilities for the Global Supply Chain
Risk & Delay Intelligence project.

Dataset: DataCo Smart Supply Chain for Big Data Analysis (Kaggle)
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis

Download the CSV manually (Kaggle requires authentication) and place it at:
    data/raw/DataCoSupplyChainDataset.csv
"""

from pathlib import Path
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "DataCoSupplyChainDataset.csv"
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "orders_clean.parquet"

# Columns known to leak post-outcome information — must NEVER be used as features
# for any model that predicts an outcome at order-creation time (Module 4).
LEAKAGE_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "shipping date (DateOrders)",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw DataCo CSV. Encoding is latin1 for this dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Download 'DataCoSupplyChainDataset.csv' from Kaggle and place it there:\n"
            "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
        )
    return pd.read_csv(path, encoding="latin1")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: parse dates, standardize column names, drop obvious duplicates."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    date_cols = [c for c in df.columns if "Date" in c or "date" in c]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df = df.drop_duplicates()
    return df


def engineer_core_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Derive fields shared across modules: duration, event flags, region/category keys."""
    df = df.copy()

    if "order date (DateOrders)" in df.columns and "shipping date (DateOrders)" in df.columns:
        df["duration_days"] = (
            df["shipping date (DateOrders)"] - df["order date (DateOrders)"]
        ).dt.days
        df["event_observed"] = df["shipping date (DateOrders)"].notna().astype(int)

    return df


def load_dataco(processed: bool = True) -> pd.DataFrame:
    """Main entry point: load, clean, and feature-engineer the dataset."""
    if processed and PROCESSED_PATH.exists():
        return pd.read_parquet(PROCESSED_PATH)

    df = load_raw()
    df = clean(df)
    df = engineer_core_fields(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    return df


def feature_columns(df: pd.DataFrame, exclude_leakage: bool = True) -> list:
    """Return candidate feature columns, excluding leakage columns when requested."""
    cols = list(df.columns)
    if exclude_leakage:
        cols = [c for c in cols if c not in LEAKAGE_COLUMNS]
    return cols
