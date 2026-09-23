from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit_training_normalizers(df: pd.DataFrame, grid_limit_kw: float) -> dict[str, float]:
    """Fit observation scales exclusively from rows labelled ``train``."""

    fit_df = df
    if "split" in df.columns:
        fit_df = df[df["split"].astype(str).str.lower().eq("train")]
    if fit_df.empty:
        raise ValueError("Training normalization requires non-empty rows labelled 'train'.")

    def maximum(column: str, default: float) -> float:
        if column not in fit_df:
            return float(default)
        value = pd.to_numeric(fit_df[column], errors="coerce").max()
        return float(default if not np.isfinite(value) else max(float(value), default))

    ratio = pd.to_numeric(fit_df.get("P_cooling_kW"), errors="coerce") / pd.to_numeric(
        fit_df.get("P_IT_kW"), errors="coerce"
    ).replace(0, np.nan)
    return {
        "P_total_dc_kW": float(grid_limit_kw),
        "P_IT_kW": float(grid_limit_kw),
        "P_cooling_kW": maximum("P_cooling_kW", 1.0),
        "PUE_model": maximum("PUE_model", 1.0),
        "cooling_ratio": max(float(ratio.max()), 0.1),
        "Pdc_forecast_kW": float(grid_limit_kw),
        "Ppv_kW": maximum("Ppv_kW", 1.0),
        "Ppv_forecast_kW": maximum("Ppv_forecast_kW", 1.0),
        "price_buy_EUR_per_kWh": maximum("price_buy_EUR_per_kWh", 1e-6),
        "price_forecast_EUR_per_kWh": maximum("price_forecast_EUR_per_kWh", 1e-6),
        "carbon_gCO2_per_kWh_proxy": 1000.0,
        "Pgrid_ref_kW": float(grid_limit_kw),
        "DR_required_reduction_kW": float(grid_limit_kw),
        "grid_import_contract_limit_kW": float(grid_limit_kw),
        "ambient_temp_C": 50.0,
        "temperature_deviation_C": 20.0,
    }


def save_normalizers(parameters: dict[str, float], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_normalizers(path: str | Path) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in data.items()}
