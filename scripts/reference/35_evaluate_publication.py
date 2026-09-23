from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import SAC

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.baselines import build_baseline_controllers, deterministic_baseline_config
from green_dc_vpp.config import load_config
from green_dc_vpp.envs import BatteryVPPEnv
from green_dc_vpp.metrics import compute_trajectory_metrics

CONTROLLERS = {
    "direct_safesac": ROOT / "configs/publication_v1/direct_safesac.yaml",
    "residual_sac": ROOT / "configs/publication_v1/residual_sac.yaml",
    "proposed_residual_safesac": ROOT / "configs/publication_v1/proposed_residual_safesac.yaml",
}
BASELINES = ["no_battery", "rule_based", "tou_self_consumption", "carbon_aware", "greedy_tracking"]
DAILY_METRICS = [
    "daily_cost_EUR", "carbon_emissions_kgCO2", "VPP_tracking_RMSE_kW", "VPP_tracking_MAE_kW",
    "DR_violation_energy_kWh", "DR_max_violation_kW", "battery_throughput_kWh",
    "equivalent_full_cycles", "terminal_SOC_error", "mean_daily_peak_import_kW",
    "physical_projection_fraction_pct", "service_coaching_fraction_pct",
]
HORIZON_METRICS = [
    "DR_interval_compliance_pct", "DR_event_compliance_pct", "DR_event_count",
    "DR_successful_event_count", "DR_violation_energy_kWh", "DR_max_violation_kW",
    "horizon_max_grid_import_kW",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def evaluate_policy(df, cfg, action_fn, continuous: bool = False, disable_battery: bool = False):
    env = BatteryVPPEnv(df, cfg, split="test", sequential=True, disable_battery=disable_battery)
    trajectories, daily = [], []
    carry = {"initial_soc": env.SOC_initial, "previous_battery_power_kW": 0.0, "previous_grid_import_kW": 0.0}
    for day_index in range(len(env.daily_start_indices)):
        options = {"day_index": day_index, **carry} if continuous else {"day_index": day_index}
        obs, _ = env.reset(options=options)
        done = False
        while not done:
            action = action_fn(obs, env)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        trajectory = env.trajectory_dataframe().copy()
        trajectory["day_index"] = day_index
        trajectories.append(trajectory)
        metrics = compute_trajectory_metrics(trajectory, cfg)
        metrics["day_index"] = day_index
        metrics["date"] = str(pd.to_datetime(trajectory["timestamp_utc"]).dt.date.iloc[0])
        daily.append(metrics)
        carry = {
            "initial_soc": float(trajectory["SOC"].iloc[-1]),
            "previous_battery_power_kW": float(trajectory["P_batt_safe_kW"].iloc[-1]),
            "previous_grid_import_kW": float(trajectory["P_import_kW"].iloc[-1]),
        }
    combined = pd.concat(trajectories, ignore_index=True)
    return combined, pd.DataFrame(daily), compute_trajectory_metrics(combined, cfg)


def summarize(controller, seed, protocol, daily, horizon):
    row = {"controller": controller, "seed": seed, "protocol": protocol, "test_days": len(daily)}
    for metric in DAILY_METRICS:
        values = pd.to_numeric(daily[metric], errors="coerce")
        row[metric] = float(values.mean())
        row[f"{metric}_std_across_days"] = float(values.std(ddof=1))
        row[f"{metric}_median"] = float(values.median())
        row[f"{metric}_iqr"] = float(values.quantile(.75) - values.quantile(.25))
    for metric in HORIZON_METRICS:
        row[metric] = horizon[metric]
    return row


def bootstrap_ci(values, seed=20260823, iterations=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [.025, .975])


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    count = len(p)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def statistical_comparisons(daily_all: pd.DataFrame) -> pd.DataFrame:
    proposed = daily_all[daily_all["controller"].eq("proposed_residual_safesac")].groupby("day_index")
    proposed_means = proposed[DAILY_METRICS].mean(numeric_only=True)
    rows = []
    for comparator in ["direct_safesac", "residual_sac"]:
        comp = daily_all[daily_all["controller"].eq(comparator)].groupby("day_index")[DAILY_METRICS].mean()
        for metric in ["daily_cost_EUR", "VPP_tracking_RMSE_kW", "battery_throughput_kWh", "terminal_SOC_error"]:
            diff = proposed_means[metric] - comp[metric]
            if np.allclose(diff, 0):
                statistic, pvalue, effect = 0.0, 1.0, 0.0
            else:
                result = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                statistic, pvalue = float(result.statistic), float(result.pvalue)
                ranks = stats.rankdata(np.abs(diff[diff != 0]))
                signed_rank_sum = float(np.sum(ranks * np.sign(diff[diff != 0])))
                effect = signed_rank_sum / float(np.sum(ranks))
            low, high = bootstrap_ci(diff)
            rows.append({
                "controller_a": "proposed_residual_safesac", "controller_b": comparator,
                "metric": metric, "test": "paired Wilcoxon signed-rank across matched test days",
                "statistic": statistic, "p_value": pvalue, "effect_size_rank_biserial": effect,
                "mean_paired_difference": float(diff.mean()), "difference_bootstrap_ci95_low": low,
                "difference_bootstrap_ci95_high": high,
            })
    result = pd.DataFrame(rows)
    result["p_value_holm"] = holm_adjust(result["p_value"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-continuous", action="store_true")
    args = parser.parse_args()
    base = ROOT / "results/publication_v1"
    trajectory_dir, table_dir = base / "trajectories", base / "tables"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    cfg_base = load_config(CONTROLLERS["proposed_residual_safesac"])
    df = pd.read_parquet(ROOT / cfg_base.paths.processed_parquet)
    summary_rows, daily_frames, metadata_rows = [], [], []

    for baseline in BASELINES:
        baseline_cfg = deterministic_baseline_config(cfg_base)
        probe = BatteryVPPEnv(
            df, baseline_cfg, split="test", sequential=True, disable_battery=baseline == "no_battery"
        )
        controller = {x.name: x for x in build_baseline_controllers(probe.df, probe.P_batt_max_kW)}[baseline]
        for protocol in (["daily_reset", "continuous_quarter"] if args.include_continuous else ["daily_reset"]):
            traj, daily, horizon = evaluate_policy(
                df, baseline_cfg, lambda obs, env, c=controller: c.act(obs, env.get_action_context()),
                protocol == "continuous_quarter", disable_battery=baseline == "no_battery"
            )
            traj.to_parquet(trajectory_dir / f"{baseline}_{protocol}.parquet", index=False)
            daily["controller"], daily["seed"], daily["protocol"] = baseline, np.nan, protocol
            daily_frames.append(daily)
            summary_rows.append(summarize(baseline, np.nan, protocol, daily, horizon))

    for controller, config_path in CONTROLLERS.items():
        cfg = load_config(config_path)
        for seed in range(1, 6):
            run_dir = base / "models" / controller / f"seed_{seed}"
            model_path = run_dir / "selected_validation_checkpoint.zip"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing required publication model: {model_path}")
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            model = SAC.load(model_path, device="cpu")
            env_probe = BatteryVPPEnv(df, cfg, split="test")
            if model.observation_space.shape != env_probe.observation_space.shape:
                raise ValueError(f"{model_path}: model/env observation mismatch; fallback is forbidden")
            for protocol in (["daily_reset", "continuous_quarter"] if args.include_continuous else ["daily_reset"]):
                traj, daily, horizon = evaluate_policy(
                    df, cfg, lambda obs, env, m=model: m.predict(obs, deterministic=True)[0], protocol == "continuous_quarter"
                )
                traj_path = trajectory_dir / f"{controller}_seed_{seed}_{protocol}.parquet"
                traj.to_parquet(traj_path, index=False)
                daily["controller"], daily["seed"], daily["protocol"] = controller, seed, protocol
                daily_frames.append(daily)
                summary_rows.append(summarize(controller, seed, protocol, daily, horizon))
                metadata_rows.append({**metadata, "protocol": protocol, "trajectory_path": str(traj_path.relative_to(ROOT)), "trajectory_sha256": sha256(traj_path)})

    daily_all = pd.concat(daily_frames, ignore_index=True)
    summaries = pd.DataFrame(summary_rows)
    daily_all.to_csv(table_dir / "publication_daily_metrics.csv", index=False)
    summaries.to_csv(table_dir / "publication_seed_and_baseline_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(table_dir / "publication_evaluation_metadata.csv", index=False)
    stats_table = statistical_comparisons(daily_all[daily_all["protocol"].eq("daily_reset")])
    stats_table.to_csv(table_dir / "publication_paired_statistics.csv", index=False)

    learned = summaries[summaries["controller"].isin(CONTROLLERS) & summaries["protocol"].eq("daily_reset")]
    baseline = summaries[summaries["controller"].isin(BASELINES) & summaries["protocol"].eq("daily_reset")]
    authoritative = []
    for name, group in pd.concat([baseline, learned]).groupby("controller", sort=False):
        row = {"controller": name, "seeds": int(group["seed"].notna().sum()) or 1}
        for metric in DAILY_METRICS + HORIZON_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std_across_seeds"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            low, high = bootstrap_ci(values, iterations=10000) if len(values) > 1 else (values.iloc[0], values.iloc[0])
            row[f"{metric}_seed_bootstrap_ci95_low"] = float(low)
            row[f"{metric}_seed_bootstrap_ci95_high"] = float(high)
            if metric in DAILY_METRICS:
                matched_days = (
                    daily_all[
                        daily_all["controller"].eq(name) & daily_all["protocol"].eq("daily_reset")
                    ].groupby("day_index")[metric].mean()
                )
                day_low, day_high = bootstrap_ci(matched_days)
                row[f"{metric}_median_across_days"] = float(matched_days.median())
                row[f"{metric}_iqr_across_days"] = float(matched_days.quantile(.75) - matched_days.quantile(.25))
                row[f"{metric}_std_across_days"] = float(matched_days.std(ddof=1))
                row[f"{metric}_day_bootstrap_ci95_low"] = float(day_low)
                row[f"{metric}_day_bootstrap_ci95_high"] = float(day_high)
        authoritative.append(row)
    pd.DataFrame(authoritative).to_csv(table_dir / "AUTHORITATIVE_CONTROLLER_RESULTS.csv", index=False)
    print(table_dir / "AUTHORITATIVE_CONTROLLER_RESULTS.csv")


if __name__ == "__main__":
    main()
