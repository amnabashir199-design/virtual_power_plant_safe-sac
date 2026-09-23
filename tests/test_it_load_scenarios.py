from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "reference" / "24_build_it_load_scenarios.py"
EXPECTED_SCENARIOS = {
    "baseline_real", "low_it_load", "high_it_load", "ai_training_bursty",
    "smooth_cloud_service", "workload_shiftable", "hot_day_cooling_stress",
    "dr_coincident_high_load",
}
METADATA_COLUMNS = {
    "scenario_name", "scenario_description", "it_load_scaling_factor",
    "cooling_model_note", "scenario_family",
}


def load_scenario_module():
    spec = importlib.util.spec_from_file_location("build_it_load_scenarios", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses inspect the importing module through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_small_dataset(path: Path) -> pd.DataFrame:
    timestamp = pd.date_range("2019-01-01", periods=4 * 96, freq="15min", tz="UTC")
    interval = np.arange(len(timestamp))
    it_load = 400.0 + 80.0 * np.sin(2.0 * np.pi * interval / 96.0) + 15.0 * (interval // 96)
    cooling_ratio = 0.22 + 0.03 * np.sin(2.0 * np.pi * interval / (96.0 * 4.0))
    cooling = it_load * cooling_ratio
    other = 0.05 * it_load
    total = it_load + cooling + other
    pv = np.maximum(0.0, 300.0 * np.sin(np.pi * ((timestamp.hour + timestamp.minute / 60) - 6) / 12))
    net = total - pv
    dr = ((timestamp.hour >= 16) & (timestamp.hour < 18)).astype(int)
    frame = pd.DataFrame({
        "timestamp_utc": timestamp,
        "P_IT_kW": it_load,
        "ai_utilization": np.clip((it_load - 100.0) / 700.0, 0, 1),
        "P_cooling_kW": cooling,
        "P_other_kW": other,
        "P_total_dc_kW": total,
        "PUE_model": total / it_load,
        "cooling_ratio_model": cooling_ratio,
        "Ppv_kW": pv,
        "Pdc_forecast_kW": total * 1.01,
        "price_buy_EUR_per_kWh": 0.03 + 0.02 * ((timestamp.hour >= 17) & (timestamp.hour < 21)),
        "carbon_gCO2_per_kWh_proxy": 500.0 - 100.0 * (pv / max(float(pv.max()), 1.0)),
        "ambient_temp_C": 22.0 + 8.0 * np.sin(2.0 * np.pi * (timestamp.hour - 8) / 24.0),
        "DR_active": dr,
        "P_net_without_battery_kW": net,
        "P_import_without_battery_kW": np.maximum(net, 0),
        "P_export_without_battery_kW": np.maximum(-net, 0),
        "split": "test",
    })
    frame.to_csv(path, index=False)
    return frame


def test_scenario_builder_outputs_and_physical_constraints(tmp_path: Path) -> None:
    module = load_scenario_module()
    source_path = tmp_path / "source.csv"
    source = make_small_dataset(source_path)
    scenario_dir = tmp_path / "scenarios"
    table_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"

    paths, summary_path, figures = module.build_scenarios(
        source_path, scenario_dir, table_dir, figure_dir
    )

    assert len(paths) == 8
    assert {path.stem.replace("green_dc_vpp_2019_15min_", "") for path in paths} == EXPECTED_SCENARIOS
    assert summary_path.exists()
    assert len(figures) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in figures)

    summary = pd.read_csv(summary_path)
    assert set(summary["scenario_name"]) == EXPECTED_SCENARIOS
    assert len(summary) == 8

    for path in paths:
        scenario = pd.read_csv(path)
        assert len(scenario) == len(source)
        assert METADATA_COLUMNS.issubset(scenario.columns)
        assert scenario[list(METADATA_COLUMNS)].notna().all().all()
        assert (scenario[["P_IT_kW", "P_cooling_kW", "P_other_kW", "P_total_dc_kW"]] >= 0).all().all()
        assert scenario["ai_utilization"].between(0, 1).all()
        np.testing.assert_allclose(
            scenario["P_total_dc_kW"],
            scenario["P_IT_kW"] + scenario["P_cooling_kW"] + scenario["P_other_kW"],
            rtol=0,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            scenario["P_net_without_battery_kW"],
            scenario["P_total_dc_kW"] - scenario["Ppv_kW"],
            rtol=0,
            atol=2e-5,
        )

    shifted = pd.read_csv(
        scenario_dir / "green_dc_vpp_2019_15min_workload_shiftable.csv",
        parse_dates=["timestamp_utc"],
    )
    source_dates = pd.to_datetime(source["timestamp_utc"]).dt.date
    shifted_dates = shifted["timestamp_utc"].dt.date
    source_daily_energy = (source["P_IT_kW"] * 0.25).groupby(source_dates).sum()
    shifted_daily_energy = (shifted["P_IT_kW"] * 0.25).groupby(shifted_dates).sum()
    np.testing.assert_allclose(shifted_daily_energy, source_daily_energy, rtol=1e-6, atol=1e-4)
