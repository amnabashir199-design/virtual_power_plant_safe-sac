from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.baselines import build_baseline_controllers, deterministic_baseline_config
from green_dc_vpp.config import load_config
from green_dc_vpp.envs import BatteryVPPEnv
from green_dc_vpp.metrics import compute_trajectory_metrics

spec = importlib.util.spec_from_file_location("scenario_builder", ROOT / "scripts/24_build_it_load_scenarios.py")
scenario_builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = scenario_builder
spec.loader.exec_module(scenario_builder)

SCENARIOS = list(scenario_builder.DEFINITIONS)
CONTROLLERS = {
    "direct_safesac": ROOT / "configs/publication_v1/direct_safesac.yaml",
    "residual_sac": ROOT / "configs/publication_v1/residual_sac.yaml",
    "proposed_residual_safesac": ROOT / "configs/publication_v1/proposed_residual_safesac.yaml",
}


def evaluate(df, cfg, action_fn, disable_battery: bool = False):
    env = BatteryVPPEnv(df, cfg, split="test", sequential=True, disable_battery=disable_battery)
    trajectories, metrics = [], []
    for day_index in range(len(env.daily_start_indices)):
        obs, _ = env.reset(options={"day_index": day_index})
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(action_fn(obs, env))
            done = terminated or truncated
        traj = env.trajectory_dataframe().copy()
        traj["day_index"] = day_index
        trajectories.append(traj)
        row = compute_trajectory_metrics(traj, cfg)
        row["day_index"] = day_index
        metrics.append(row)
    return pd.concat(trajectories, ignore_index=True), pd.DataFrame(metrics)


def main() -> None:
    output = ROOT / "results/publication_v1/scenarios"
    (output / "datasets").mkdir(parents=True, exist_ok=True)
    (output / "trajectories").mkdir(parents=True, exist_ok=True)
    cfg_base = load_config(CONTROLLERS["proposed_residual_safesac"])
    source = pd.read_parquet(ROOT / cfg_base.paths.processed_parquet)
    all_daily = []
    for scenario_name in SCENARIOS:
        scenario = scenario_builder.recompute_scenario(source, scenario_name)
        scenario_path = output / "datasets" / f"{scenario_name}.parquet"
        scenario.to_parquet(scenario_path, index=False)
        baseline_cfg = deterministic_baseline_config(cfg_base)
        probe = BatteryVPPEnv(scenario, baseline_cfg, split="test", sequential=True)
        baselines = {x.name: x for x in build_baseline_controllers(probe.df, probe.P_batt_max_kW)}
        for name in ["no_battery", "rule_based", "greedy_tracking"]:
            controller = baselines[name]
            traj, daily = evaluate(
                scenario, baseline_cfg, lambda obs, env, c=controller: c.act(obs, env.get_action_context()),
                disable_battery=name == "no_battery",
            )
            traj.to_parquet(output / "trajectories" / f"{scenario_name}_{name}.parquet", index=False)
            daily["scenario"], daily["controller"], daily["seed"] = scenario_name, name, np.nan
            all_daily.append(daily)
        for controller_name, config_path in CONTROLLERS.items():
            cfg = load_config(config_path)
            for seed in range(1, 6):
                model_path = ROOT / f"results/publication_v1/models/{controller_name}/seed_{seed}/selected_validation_checkpoint.zip"
                if not model_path.exists():
                    raise FileNotFoundError(model_path)
                model = SAC.load(model_path, device="cpu")
                traj, daily = evaluate(scenario, cfg, lambda obs, env, m=model: m.predict(obs, deterministic=True)[0])
                traj.to_parquet(output / "trajectories" / f"{scenario_name}_{controller_name}_seed_{seed}.parquet", index=False)
                daily["scenario"], daily["controller"], daily["seed"] = scenario_name, controller_name, seed
                all_daily.append(daily)
    daily = pd.concat(all_daily, ignore_index=True)
    daily.to_csv(output / "scenario_daily_metrics.csv", index=False)
    metric_columns = [c for c in daily.select_dtypes(include="number") if c not in {"day_index", "seed"}]
    seed_day = daily.groupby(["scenario", "controller", "seed"], dropna=False)[metric_columns].mean().reset_index()
    seed_day.to_csv(output / "scenario_seed_means.csv", index=False)
    seed_summary = seed_day.groupby(["scenario", "controller"], dropna=False)[metric_columns].agg(["mean", "std"]).reset_index()
    seed_summary.columns = ["_".join(str(x) for x in col if x).rstrip("_") for col in seed_summary.columns]
    seed_summary.to_csv(output / "scenario_training_seed_summary.csv", index=False)
    scenario_summary = seed_day.groupby("controller", dropna=False)[metric_columns].agg(["mean", "std"]).reset_index()
    scenario_summary.columns = ["_".join(str(x) for x in col if x).rstrip("_") for col in scenario_summary.columns]
    scenario_summary.to_csv(output / "scenario_sensitivity_summary.csv", index=False)
    print(output)


if __name__ == "__main__":
    main()
