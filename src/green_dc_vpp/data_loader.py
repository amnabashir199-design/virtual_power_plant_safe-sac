from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


TIMESTAMP_COLUMN = "timestamp_utc"
SPLIT_COLUMN = "split"

REQUIRED_COLUMNS = [
    TIMESTAMP_COLUMN,
    "P_IT_kW",
    "P_cooling_kW",
    "P_total_dc_kW",
    "Ppv_kW",
    "price_buy_EUR_per_kWh",
    "carbon_gCO2_per_kWh_proxy",
    "Pgrid_ref_kW",
    "DR_active",
    SPLIT_COLUMN,
]


def validate_required_columns(df: pd.DataFrame, required: Iterable[str] | None = None) -> None:
    required_cols = list(required or REQUIRED_COLUMNS)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        available = ", ".join(df.columns)
        missing_text = ", ".join(missing)
        raise ValueError(f"Dataset is missing required columns: {missing_text}. Available columns: {available}")


def load_dataset(path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the benchmark CSV/CSV.GZ, parse timestamps, and validate core columns."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".gz" or path.name.lower().endswith(".csv.gz"):
        df = pd.read_csv(path, compression="gzip")
    else:
        df = pd.read_csv(path)

    if TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain timestamp column '{TIMESTAMP_COLUMN}'")

    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="raise")
    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    validate_required_columns(df, required_columns)

    if SPLIT_COLUMN not in df.columns:
        raise ValueError(f"Dataset split column '{SPLIT_COLUMN}' was not found")
    return df


def get_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return a copy of the requested chronological dataset split."""

    if SPLIT_COLUMN not in df.columns:
        raise ValueError(f"Cannot select split: column '{SPLIT_COLUMN}' was not found")
    mask = df[SPLIT_COLUMN].astype(str).str.lower() == str(split_name).lower()
    if not mask.any():
        available = sorted(df[SPLIT_COLUMN].dropna().astype(str).unique())
        raise ValueError(f"Split '{split_name}' not found. Available splits: {available}")
    return df.loc[mask].copy().reset_index(drop=True)
