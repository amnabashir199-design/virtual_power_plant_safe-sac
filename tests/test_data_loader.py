from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.data_loader import TIMESTAMP_COLUMN, get_split, load_dataset


def test_sample_dataset_loads_with_timestamp_and_split() -> None:
    sample_path = ROOT / "data" / "processed" / "green_dc_vpp_publication_v1_2019_15min.parquet"
    assert sample_path.exists(), "Publication dataset is required for the data-loader smoke test"

    df = load_dataset(sample_path)

    assert not df.empty
    assert TIMESTAMP_COLUMN in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COLUMN])
    assert "split" in df.columns


def test_key_columns_exist() -> None:
    df = load_dataset(ROOT / "data" / "processed" / "green_dc_vpp_publication_v1_2019_15min.parquet")
    key_columns = {
        "P_IT_kW",
        "P_cooling_kW",
        "P_total_dc_kW",
        "Ppv_kW",
        "price_buy_EUR_per_kWh",
        "carbon_gCO2_per_kWh_proxy",
        "Pgrid_ref_kW",
        "DR_active",
    }
    assert key_columns.issubset(df.columns)


def test_get_split_returns_non_empty_dataframe() -> None:
    df = load_dataset(ROOT / "data" / "processed" / "green_dc_vpp_publication_v1_2019_15min.parquet")
    split_name = str(df["split"].iloc[0])
    split_df = get_split(df, split_name)

    assert not split_df.empty
    assert set(split_df["split"].astype(str).str.lower()) == {split_name.lower()}
