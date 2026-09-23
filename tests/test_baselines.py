from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.baselines import build_baseline_controllers
from green_dc_vpp.config import load_config
from green_dc_vpp.envs import BatteryVPPEnv
from green_dc_vpp.metrics import compute_trajectory_metrics


def _load_df() -> pd.DataFrame:
    parquet = ROOT / "data" / "processed" / "green_dc_vpp_publication_v1_2019_15min.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(ROOT / "data" / "green_dc_vpp_sample_7days.csv", parse_dates=["timestamp_utc"])


def test_baseline_actions_are_finite_and_scalar() -> None:
    info = {
        "DR_active": 0,
        "price_buy_EUR_per_kWh": 0.04,
        "carbon_gCO2_per_kWh_proxy": 500.0,
        "Ppv_kW": 200.0,
        "P_total_dc_kW": 500.0,
        "P_net_without_battery_kW": 300.0,
        "Pgrid_ref_kW": 250.0,
    }
    obs = np.zeros(24, dtype=np.float32)
    for controller in build_baseline_controllers(P_batt_max_kW=750.0):
        action = controller.act(obs, info)
        assert action.shape == (1,)
        assert np.isfinite(action).all()
        assert -1.5 <= float(action[0]) <= 1.5


def _run_disabled_battery_episode() -> tuple[BatteryVPPEnv, pd.DataFrame, dict[str, float]]:
    cfg = load_config(ROOT / "configs" / "baselines" / "evaluation_config.yaml")
    env = BatteryVPPEnv(_load_df(), config=cfg, split="all", sequential=True, disable_battery=True)
    obs, info = env.reset(seed=123)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(np.asarray([1.0], dtype=np.float32))
        done = terminated or truncated
    traj = env.trajectory_dataframe()
    return env, traj, compute_trajectory_metrics(traj, env.config)


def test_no_battery_has_zero_throughput() -> None:
    _, traj, metrics = _run_disabled_battery_episode()
    assert np.allclose(traj["P_batt_raw_kW"], 0.0)
    assert np.allclose(traj["P_batt_safe_kW"], 0.0)
    assert metrics["battery_throughput_kWh"] == 0.0
    assert metrics["equivalent_full_cycles"] == 0.0


def test_no_battery_soc_constant() -> None:
    env, traj, metrics = _run_disabled_battery_episode()
    assert np.allclose(traj["SOC"], env.SOC_initial)
    assert metrics["final_SOC"] == env.SOC_initial


def test_no_battery_action_not_modified_by_safety() -> None:
    _, traj, metrics = _run_disabled_battery_episode()
    assert not traj["safety_clipped"].any()
    assert not traj["hard_clip"].any()
    assert not traj["soft_coach"].any()
    assert metrics["hard_clip_fraction_pct"] == 0.0
    assert metrics["soft_coach_fraction_pct"] == 0.0
