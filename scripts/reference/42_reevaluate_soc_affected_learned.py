"""Reevaluate only frozen learned scenario trajectories proven SOC-invalid.

Phase 3 first identified nine low-IT learned trajectories whose saved SOC
exceeded the configured hard maximum because the old projector preferred an
infeasible ramp corridor.  This script loads the unchanged checkpoints and
frozen scenario dataset, reevaluates exactly those affected controller/seed
pairs, and writes versioned artifacts under ``results/publication_final``.
It performs no training and never writes to ``results/publication_v1``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.config import load_config
from green_dc_vpp.envs import BatteryVPPEnv
from green_dc_vpp.metrics import compute_trajectory_metrics


SCENARIO = "low_it_load"
CONTROLLERS = {
    "direct_safesac": ROOT / "configs/publication_v1/direct_safesac.yaml",
    "residual_sac": ROOT / "configs/publication_v1/residual_sac.yaml",
    "proposed_residual_safesac": ROOT / "configs/publication_v1/proposed_residual_safesac.yaml",
}
OLD_TRAJECTORIES = ROOT / "results/publication_v1/scenarios/trajectories"
OUT = ROOT / "results/publication_final/scenario_trajectories/learned_soc_repair"
VERIFY = ROOT / "results/publication_final/verification"


def discover_invalid() -> list[tuple[str, int, float, float]]:
    invalid = []
    for controller in CONTROLLERS:
        for seed in range(1, 6):
            path = OLD_TRAJECTORIES / f"{SCENARIO}_{controller}_seed_{seed}.parquet"
            soc = pd.read_parquet(path, columns=["SOC"])["SOC"]
            minimum, maximum = float(soc.min()), float(soc.max())
            if minimum < .2 - 1e-9 or maximum > .9 + 1e-9:
                invalid.append((controller, seed, minimum, maximum))
    return invalid


def evaluate(data: pd.DataFrame, config, model: SAC) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = BatteryVPPEnv(data, config, split="test", sequential=True)
    trajectories, daily_rows = [], []
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
        metrics = compute_trajectory_metrics(trajectory, config)
        metrics["day_index"] = day_index
        daily_rows.append(metrics)
    return pd.concat(trajectories, ignore_index=True), pd.DataFrame(daily_rows)


def validate(trajectory: pd.DataFrame, config) -> dict[str, float | int | bool | str]:
    timestamp = pd.to_datetime(trajectory["timestamp_utc"], utc=True, errors="raise")
    power = pd.to_numeric(trajectory["P_batt_safe_kW"], errors="raise")
    soc = pd.to_numeric(trajectory["SOC"], errors="raise")
    balance = (
        pd.to_numeric(trajectory["P_grid_kW"], errors="raise")
        - pd.to_numeric(trajectory["P_total_dc_kW"], errors="raise")
        + pd.to_numeric(trajectory["Ppv_kW"], errors="raise")
        + power
    )
    day_changed = timestamp.dt.floor("D").ne(timestamp.dt.floor("D").shift())
    previous = power.shift(fill_value=0.0).where(~day_changed, 0.0)
    ramp = (power - previous).abs()
    limit = float(config["battery"]["ramp_limit_kW_per_15min"])
    override = trajectory["clip_reasons"].fillna("").str.contains(
        "ramp_overridden_for_hard_feasibility", regex=False
    )
    unjustified = (ramp > limit + 1e-9) & ~override
    result = {
        "interval_count": len(trajectory), "day_count": timestamp.dt.floor("D").nunique(),
        "first_timestamp": timestamp.iloc[0].isoformat(), "last_timestamp": timestamp.iloc[-1].isoformat(),
        "min_SOC": float(soc.min()), "max_SOC": float(soc.max()),
        "max_abs_battery_power_kW": float(power.abs().max()),
        "max_energy_balance_error_kW": float(balance.abs().max()),
        "max_ramp_kW_per_interval": float(ramp.max()),
        "ramp_override_count": int(override.sum()),
        "unjustified_ramp_violation_count": int(unjustified.sum()),
    }
    passed = (
        result["interval_count"] == 8832 and result["day_count"] == 92
        and timestamp.iloc[0] == pd.Timestamp("2019-10-01T00:00:00Z")
        and timestamp.iloc[-1] == pd.Timestamp("2019-12-31T23:45:00Z")
        and result["min_SOC"] >= float(config["battery"]["SOC_min"]) - 1e-9
        and result["max_SOC"] <= float(config["battery"]["SOC_max"]) + 1e-9
        and result["max_abs_battery_power_kW"] <= float(config["battery"]["P_batt_max_kW"]) + 1e-9
        and result["max_energy_balance_error_kW"] <= 1e-9
        and result["unjustified_ramp_violation_count"] == 0
    )
    result["passed"] = bool(passed)
    if not passed:
        raise AssertionError(result)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    VERIFY.mkdir(parents=True, exist_ok=True)
    invalid = discover_invalid()
    expected = {
        ("residual_sac", 1), ("residual_sac", 3), ("residual_sac", 4), ("residual_sac", 5),
        *{("proposed_residual_safesac", seed) for seed in range(1, 6)},
    }
    found = {(controller, seed) for controller, seed, _, _ in invalid}
    if found != expected:
        raise RuntimeError(f"Unexpected SOC-invalid learned trajectory set: {sorted(found)}")
    data = pd.read_parquet(ROOT / f"results/publication_v1/scenarios/datasets/{SCENARIO}.parquet")
    daily_frames, checks, audit = [], [], []
    for controller, seed, old_min, old_max in invalid:
        config = load_config(CONTROLLERS[controller])
        model_path = ROOT / f"results/publication_v1/models/{controller}/seed_{seed}/selected_validation_checkpoint.zip"
        model = SAC.load(model_path, device="cpu")
        trajectory, daily = evaluate(data, config, model)
        check = validate(trajectory, config)
        check.update({"scenario": SCENARIO, "controller": controller, "seed": seed})
        checks.append(check)
        trajectory.to_parquet(OUT / f"{SCENARIO}_{controller}_seed_{seed}_final.parquet", index=False)
        daily["scenario"], daily["controller"], daily["seed"] = SCENARIO, controller, seed
        daily_frames.append(daily)
        audit.append({
            "scenario": SCENARIO, "controller": controller, "seed": seed,
            "old_min_SOC": old_min, "old_max_SOC": old_max,
            "final_min_SOC": check["min_SOC"], "final_max_SOC": check["max_SOC"],
            "model_retrained": False, "passed": True,
        })
    daily = pd.concat(daily_frames, ignore_index=True)
    metric_columns = [c for c in daily.select_dtypes(include="number") if c not in {"day_index", "seed"}]
    seed_means = daily.groupby(["scenario", "controller", "seed"], as_index=False)[metric_columns].mean()
    daily.to_csv(VERIFY / "learned_scenario_soc_repair_daily_metrics.csv", index=False, float_format="%.15g")
    seed_means.to_csv(VERIFY / "learned_scenario_soc_repair_seed_means.csv", index=False, float_format="%.15g")
    pd.DataFrame(checks).to_csv(VERIFY / "learned_scenario_soc_repair_trajectory_checks.csv", index=False, float_format="%.15g")
    pd.DataFrame(audit).to_csv(VERIFY / "LEARNED_SCENARIO_SOC_REPAIR_AUDIT.csv", index=False, float_format="%.15g")
    status = {
        "status": "PASS", "reason": "old ramp/SOC conflict crossed hard SOC_max",
        "scenario": SCENARIO, "reevaluated_trajectories": len(invalid),
        "models_retrained": 0, "models_modified": 0,
        "affected": [{"controller": c, "seed": s} for c, s, _, _ in invalid],
    }
    (VERIFY / "LEARNED_SCENARIO_SOC_REPAIR_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
