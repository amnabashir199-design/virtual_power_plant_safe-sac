from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.config import load_config
from green_dc_vpp.envs import BatteryVPPEnv
from green_dc_vpp.metrics import compute_trajectory_metrics

VARIANTS = [
    "direct_sac_no_safety",
    "residual_prior_no_safety",
    "residual_sac",
    "proposed_residual_safesac",
    "proposed_no_terminal_recovery",
]


def main() -> None:
    base = ROOT / "results/publication_v1"
    dataset = pd.read_parquet(base / "dataset/green_dc_vpp_publication_v1_2019_15min.parquet")
    out = base / "ablations"
    (out / "trajectories").mkdir(parents=True, exist_ok=True)
    daily_rows = []
    for variant in VARIANTS:
        for seed in range(1, 6):
            run = base / f"models/{variant}/seed_{seed}"
            cfg = load_config(run / "config_snapshot.yaml")
            model = SAC.load(run / "selected_validation_checkpoint.zip", device="cpu")
            env = BatteryVPPEnv(dataset, cfg, split="test", sequential=True)
            trajectories = []
            for day_index in range(len(env.daily_start_indices)):
                obs, _ = env.reset(options={"day_index": day_index})
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                trajectory = env.trajectory_dataframe().copy()
                trajectory["day_index"] = day_index
                trajectories.append(trajectory)
                row = compute_trajectory_metrics(trajectory, cfg)
                row.update({"variant": variant, "seed": seed, "day_index": day_index})
                daily_rows.append(row)
            pd.concat(trajectories, ignore_index=True).to_parquet(
                out / "trajectories" / f"{variant}_seed_{seed}.parquet", index=False
            )
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(out / "true_ablation_daily_metrics.csv", index=False)
    metrics = [
        "daily_cost_EUR", "VPP_tracking_RMSE_kW", "DR_interval_compliance_pct",
        "DR_violation_energy_kWh", "battery_throughput_kWh", "terminal_SOC_error",
        "physical_projection_fraction_pct", "service_coaching_fraction_pct",
    ]
    seed_means = daily.groupby(["variant", "seed"])[metrics].mean().reset_index()
    seed_means.to_csv(out / "true_ablation_seed_metrics.csv", index=False)
    summary = seed_means.groupby("variant")[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x).rstrip("_") for col in summary.columns]
    summary.to_csv(out / "TRUE_ABLATION_RESULTS.csv", index=False)
    print(out / "TRUE_ABLATION_RESULTS.csv")


if __name__ == "__main__":
    main()
