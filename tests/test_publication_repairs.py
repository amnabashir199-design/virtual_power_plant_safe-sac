from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.metrics import compute_trajectory_metrics
from green_dc_vpp.normalization import fit_training_normalizers
from green_dc_vpp.safety import project_battery_action
from green_dc_vpp.config import load_config
from green_dc_vpp.envs import BatteryVPPEnv


def _trajectory(imports, active, limits) -> pd.DataFrame:
    count = len(imports)
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2019-10-01", periods=count, freq="15min"),
            "P_import_kW": imports,
            "P_export_kW": np.zeros(count),
            "P_grid_kW": imports,
            "P_batt_safe_kW": np.zeros(count),
            "Pgrid_ref_kW": np.zeros(count),
            "grid_import_contract_limit_kW": np.full(count, 1000.0),
            "DR_active": active,
            "DR_import_limit_kW": limits,
            "SOC": np.full(count, 0.55),
            "reward": np.zeros(count),
            "cost_EUR": np.zeros(count),
        }
    )


def test_dr_tolerance_boundaries_and_inactive_interval() -> None:
    tol = 1e-6
    traj = _trajectory(
        [100.0, 100.0 + tol / 2, 100.0 + tol, 100.0 + 2 * tol, 999.0],
        [1, 1, 1, 1, 0],
        [100.0, 100.0, 100.0, 100.0, 1.0],
    )
    result = compute_trajectory_metrics(traj, {"metrics": {"constraint_tolerance_kW": tol}, "battery": {"E_batt_kWh": 4000, "SOC_initial": .55}})
    assert result["DR_interval_compliance_pct"] == 75.0
    assert result["DR_event_count"] == 1.0
    assert result["DR_successful_event_count"] == 0.0
    assert result["DR_event_compliance_pct"] == 0.0
    assert np.isclose(result["DR_max_violation_kW"], 2 * tol)


def test_dr_whole_event_compliance_uses_contiguous_events() -> None:
    traj = _trajectory([90, 95, 0, 100, 101], [1, 1, 0, 1, 1], [100, 100, np.nan, 100, 100])
    result = compute_trajectory_metrics(traj, {"battery": {"E_batt_kWh": 4000, "SOC_initial": .55}})
    assert result["DR_event_count"] == 2.0
    assert result["DR_successful_event_count"] == 1.0
    assert result["DR_event_compliance_pct"] == 50.0


def test_metrics_use_active_battery_configuration() -> None:
    traj = _trajectory([0, 0], [0, 0], [np.nan, np.nan])
    traj["P_batt_safe_kW"] = [100.0, 100.0]
    traj["SOC"] = [0.62, 0.60]
    cfg = {"battery": {"E_batt_kWh": 1000.0, "SOC_initial": 0.60}}
    result = compute_trajectory_metrics(traj, cfg)
    assert np.isclose(result["equivalent_full_cycles"], 0.025)
    assert np.isclose(result["terminal_SOC_error"], 0.0)


def test_normalizers_are_unchanged_when_test_values_change() -> None:
    df = pd.DataFrame(
        {
            "split": ["train", "train", "validation", "test"],
            "P_cooling_kW": [1, 2, 3, 4],
            "P_IT_kW": [10, 10, 10, 10],
            "PUE_model": [1.1, 1.2, 1.3, 1.4],
            "Ppv_kW": [2, 3, 4, 5],
            "Ppv_forecast_kW": [2, 3, 4, 5],
            "price_buy_EUR_per_kWh": [.1, .2, .3, .4],
            "price_forecast_EUR_per_kWh": [.1, .2, .3, .4],
        }
    )
    expected = fit_training_normalizers(df, 700.0)
    changed = df.copy()
    changed.loc[changed["split"] != "train", changed.columns != "split"] = 1e9
    assert fit_training_normalizers(changed, 700.0) == expected


BASE_PARAMS = {
    "battery": {
        "E_batt_kWh": 4000.0,
        "P_batt_max_kW": 750.0,
        "SOC_min": 0.2,
        "SOC_max": 0.9,
        "SOC_reserve": 0.3,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
        "ramp_limit_kW_per_15min": 100.0,
    },
    "system": {"grid_import_contract_limit_kW": 700.0},
    "safety_layer": {
        "enabled": True,
        "enforce_power_limit": True,
        "enforce_soc_bounds": True,
        "enforce_soc_reserve": False,
        "enforce_ramp_limit": True,
        "enforce_dr_limit": True,
        "enforce_grid_limit": True,
    },
}


def _project(raw=1000.0, soc=.55, previous=0.0, row=None, overrides=None):
    params = copy.deepcopy(BASE_PARAMS)
    params["safety_layer"].update(overrides or {})
    row = row or {"P_total_dc_kW": 500.0, "Ppv_kW": 0.0, "DR_active": 0}
    return project_battery_action(raw, soc, previous, row, params, .25)


def test_safety_master_switch_bypasses_projection() -> None:
    assert _project(overrides={"enabled": False})["P_batt_safe_kW"] == 1000.0


def test_power_soc_ramp_and_reserve_switches_are_independent() -> None:
    no_power = _project(previous=1000.0, overrides={"enforce_power_limit": False, "enforce_ramp_limit": False})
    assert no_power["P_batt_safe_kW"] == 1000.0
    no_soc = _project(raw=100.0, soc=.2, overrides={"enforce_soc_bounds": False})
    assert no_soc["P_batt_safe_kW"] == 100.0
    no_ramp = _project(raw=500.0, overrides={"enforce_ramp_limit": False})
    assert no_ramp["P_batt_safe_kW"] == 500.0
    reserve = _project(raw=100.0, soc=.25, overrides={"enforce_ramp_limit": False, "enforce_soc_reserve": True})
    assert reserve["P_batt_safe_kW"] == 0.0


def test_hard_soc_bounds_override_an_infeasible_ramp_corridor() -> None:
    high_soc = _project(raw=-500.0, soc=.8995, previous=-400.0)
    assert high_soc["P_batt_safe_kW"] > -10.0
    assert "ramp_overridden_for_hard_feasibility" in high_soc["clip_reasons"]

    low_soc = _project(raw=500.0, soc=.2, previous=400.0)
    assert low_soc["P_batt_safe_kW"] == 0.0
    assert "ramp_overridden_for_hard_feasibility" in low_soc["clip_reasons"]


def test_dr_and_grid_coaching_switches_are_independent() -> None:
    dr_row = {"P_total_dc_kW": 500.0, "Ppv_kW": 0.0, "DR_active": 1, "DR_import_limit_kW": 450.0}
    assert _project(raw=0, row=dr_row, overrides={"enforce_dr_limit": False, "enforce_grid_limit": False})["P_batt_safe_kW"] == 0
    assert _project(raw=0, row=dr_row, overrides={"enforce_grid_limit": False})["P_batt_safe_kW"] == 50
    grid_row = {"P_total_dc_kW": 800.0, "Ppv_kW": 0.0, "DR_active": 0}
    assert _project(raw=0, row=grid_row, overrides={"enforce_grid_limit": False})["P_batt_safe_kW"] == 0
    assert _project(raw=0, row=grid_row)["P_batt_safe_kW"] == 100


def test_environment_accepts_carry_state_for_continuous_sensitivity() -> None:
    df = pd.read_parquet(ROOT / "data/processed/green_dc_vpp_publication_v1_2019_15min.parquet")
    cfg = load_config(ROOT / "configs/residual_sac/config.yaml")
    env = BatteryVPPEnv(df, cfg, split="test", sequential=True)
    obs, _ = env.reset(options={
        "day_index": 1,
        "initial_soc": 0.61,
        "previous_battery_power_kW": 27.0,
        "previous_grid_import_kW": 412.0,
    })
    assert np.isclose(env._soc, 0.61)
    assert np.isclose(env._prev_p_batt, 27.0)
    assert np.isclose(env._prev_grid_import, 412.0)
    assert np.isclose(obs[0], 0.61)
