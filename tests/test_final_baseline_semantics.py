from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.baselines import build_baseline_controllers, deterministic_baseline_config
from green_dc_vpp.config import load_config
from green_dc_vpp.envs import BatteryVPPEnv
from green_dc_vpp.metrics import compute_trajectory_metrics


def _publication_inputs():
    original = load_config(ROOT / "configs/proposed_residual_safesac/config.yaml")
    data = pd.read_parquet(ROOT / original.paths.processed_parquet)
    return original, deterministic_baseline_config(original), data


def test_deterministic_baseline_config_is_direct_physical_only_and_non_mutating() -> None:
    original, baseline, _ = _publication_inputs()
    assert original.control.mode == "residual_sac"
    assert original.control.greedy_prior_enabled is True
    assert original.safety_layer.enforce_dr_limit is True
    assert original.safety_layer.enforce_grid_limit is True

    assert baseline["control"]["mode"] == "direct_sac"
    assert baseline["control"]["greedy_prior_enabled"] is False
    assert baseline["control"]["greedy_blend_alpha"] == 0.0
    assert baseline["safety_layer"]["enforce_power_limit"] is True
    assert baseline["safety_layer"]["enforce_soc_bounds"] is True
    assert baseline["safety_layer"]["enforce_ramp_limit"] is True
    assert baseline["safety_layer"]["enforce_dr_limit"] is False
    assert baseline["safety_layer"]["enforce_grid_limit"] is False


def test_greedy_baseline_action_is_applied_as_direct_command_without_hidden_prior() -> None:
    _, baseline, data = _publication_inputs()
    env = BatteryVPPEnv(data, baseline, split="test", sequential=True)
    controller = {
        item.name: item for item in build_baseline_controllers(env.df, env.P_batt_max_kW)
    }["greedy_tracking"]
    obs, _ = env.reset(options={"day_index": 0})
    action = controller.act(obs, env.get_action_context())
    _, _, _, _, info = env.step(action)

    intended = float(action[0]) * env.P_batt_max_kW
    assert env.control_mode == "direct_sac"
    assert env.greedy_prior_enabled is False
    assert np.isclose(info["P_batt_raw_kW"], intended, rtol=0.0, atol=1e-5)
    assert info["P_batt_greedy_kW"] == 0.0
    assert info["P_batt_residual_kW"] == 0.0
    assert info["service_coaching"] is False


def test_final_no_battery_pipeline_enforces_all_exact_invariants() -> None:
    _, baseline, data = _publication_inputs()
    env = BatteryVPPEnv(
        data, baseline, split="test", sequential=True, disable_battery=True
    )
    controller = {
        item.name: item for item in build_baseline_controllers(env.df, env.P_batt_max_kW)
    }["no_battery"]
    obs, _ = env.reset(options={"day_index": 0})
    done = False
    while not done:
        action = controller.act(obs, env.get_action_context())
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    trajectory = env.trajectory_dataframe()
    metrics = compute_trajectory_metrics(trajectory, baseline)
    zero_columns = [
        "P_batt_greedy_kW",
        "P_batt_residual_kW",
        "P_batt_raw_kW",
        "P_batt_commanded_kW",
        "P_batt_physically_clipped_kW",
        "P_batt_service_coached_kW",
        "P_batt_safe_kW",
        "degradation_penalty",
    ]
    for column in zero_columns:
        assert np.max(np.abs(trajectory[column].to_numpy(dtype=float))) <= 1e-12
    assert np.allclose(trajectory["SOC"], env.SOC_initial, rtol=0.0, atol=1e-12)
    assert not trajectory["physical_projection"].any()
    assert not trajectory["service_coaching"].any()
    assert metrics["battery_throughput_kWh"] == 0.0
    assert metrics["equivalent_full_cycles"] == 0.0
    assert metrics["physical_projection_fraction_pct"] == 0.0
    assert metrics["service_coaching_fraction_pct"] == 0.0
    assert metrics["terminal_SOC_error"] == 0.0
