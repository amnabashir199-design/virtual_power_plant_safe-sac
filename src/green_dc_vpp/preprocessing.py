from __future__ import annotations

import numpy as np
import pandas as pd

from green_dc_vpp.data_loader import TIMESTAMP_COLUMN


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free cyclical time features from timestamp or existing calendar columns."""

    out = df.copy()
    ts = pd.to_datetime(out[TIMESTAMP_COLUMN])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    day = ts.dt.dayofweek
    out["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["sin_day"] = np.sin(2.0 * np.pi * day / 7.0)
    out["cos_day"] = np.cos(2.0 * np.pi * day / 7.0)
    return out


def ensure_forecast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create missing one-step persistence forecasts without future leakage."""

    out = df.copy()
    forecast_sources = {
        "Pdc_forecast_kW": "P_total_dc_kW",
        "Ppv_forecast_kW": "Ppv_kW",
        "price_forecast_EUR_per_kWh": "price_buy_EUR_per_kWh",
    }
    for forecast_col, source_col in forecast_sources.items():
        if forecast_col in out.columns:
            continue
        if source_col not in out.columns:
            raise ValueError(f"Cannot create {forecast_col}: source column {source_col} is missing")
        out[forecast_col] = out[source_col].shift(1).bfill()
    return out
