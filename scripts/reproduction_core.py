"""Portable, read-only reproduction helpers for the final publication evidence.

The primary workflow replays the included deterministic daily-reset trajectories,
checks every selected checkpoint, and independently recalculates metrics.  It
does not train or modify a controller.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.metrics import compute_trajectory_metrics  # noqa: E402


SCENARIOS = [
    "baseline_real",
    "low_it_load",
    "high_it_load",
    "ai_training_bursty",
    "smooth_cloud_service",
    "workload_shiftable",
    "hot_day_cooling_stress",
    "dr_coincident_high_load",
]
DETERMINISTIC = [
    "no_battery",
    "rule_based",
    "tou_self_consumption",
    "carbon_aware",
    "greedy_tracking",
]
LEARNED = ["direct_safesac", "residual_sac", "proposed_residual_safesac"]
ABLATION_ONLY = [
    "direct_sac_no_safety",
    "residual_prior_no_safety",
    "proposed_no_terminal_recovery",
]
ABLATION_ORDER = [
    "direct_sac_no_safety",
    "residual_prior_no_safety",
    "residual_sac",
    "proposed_no_terminal_recovery",
    "proposed_residual_safesac",
]

DAILY_METRICS = [
    "daily_cost_EUR",
    "carbon_emissions_kgCO2",
    "VPP_tracking_RMSE_kW",
    "VPP_tracking_MAE_kW",
    "battery_throughput_kWh",
    "equivalent_full_cycles",
    "terminal_SOC_error",
    "mean_daily_peak_import_kW",
    "physical_projection_fraction_pct",
    "service_coaching_fraction_pct",
]
POOLED_METRICS = [
    "DR_interval_compliance_pct",
    "DR_event_compliance_pct",
    "DR_violation_energy_kWh",
    "DR_max_violation_kW",
    "horizon_max_grid_import_kW",
]
DERIVED_DAILY_DR_METRICS = [
    "mean_daily_DR_interval_compliance_pct",
    "mean_daily_DR_event_compliance_pct",
    "mean_daily_DR_violation_energy_kWh",
    "mean_daily_max_DR_violation_kW",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_output() -> Path:
    output = ROOT / "results" / "reproduced"
    output.mkdir(parents=True, exist_ok=True)
    return output


def validate_dataset() -> dict:
    dataset_path = ROOT / "data" / "processed" / "green_dc_vpp_publication_v1_2019_15min.parquet"
    frame = pd.read_parquet(dataset_path)
    timestamp_column = "timestamp_utc" if "timestamp_utc" in frame.columns else "timestamp"
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    split_counts = {
        "train": int((timestamps < pd.Timestamp("2019-07-01", tz="UTC")).sum()),
        "validation": int(((timestamps >= pd.Timestamp("2019-07-01", tz="UTC")) &
                           (timestamps < pd.Timestamp("2019-10-01", tz="UTC"))).sum()),
        "test": int((timestamps >= pd.Timestamp("2019-10-01", tz="UTC")).sum()),
    }
    scenario_dir = ROOT / "data" / "processed" / "scenarios"
    scenario_paths = sorted(scenario_dir.glob("*.parquet"))
    baseline = pd.read_parquet(scenario_dir / "baseline_real.parquet", columns=["P_IT_kW"])
    high = pd.read_parquet(scenario_dir / "high_it_load.parquet", columns=["P_IT_kW"])
    ratio = high["P_IT_kW"].to_numpy() / baseline["P_IT_kW"].to_numpy()
    report = {
        "path": dataset_path.relative_to(ROOT).as_posix(),
        "sha256": sha256(dataset_path),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "start_utc": timestamps.min().isoformat(),
        "end_utc": timestamps.max().isoformat(),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "missing_timestamps": int(35040 - timestamps.nunique()),
        "split_rows": split_counts,
        "scenario_dataset_count": len(scenario_paths),
        "high_to_baseline_mean_ratio": float(high["P_IT_kW"].mean() / baseline["P_IT_kW"].mean()),
        "high_to_baseline_interval_ratio_min": float(np.min(ratio)),
        "high_to_baseline_interval_ratio_max": float(np.max(ratio)),
    }
    report["passed"] = bool(
        report["rows"] == 35040
        and report["duplicate_timestamps"] == 0
        and split_counts == {"train": 17376, "validation": 8832, "test": 8832}
        and len(scenario_paths) == 8
        and np.allclose(ratio, 1.25, rtol=0.0, atol=1e-12)
    )
    output = ensure_output() / "DATA_VALIDATION.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"Dataset validation failed: {report}")
    return report


def _day_groups(frame: pd.DataFrame):
    if "day_index" in frame.columns:
        return frame.groupby("day_index", sort=True)
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame.groupby(timestamps.dt.floor("D"), sort=True)


def evaluate_trajectory(path: Path) -> tuple[dict[str, float], pd.DataFrame]:
    frame = pd.read_parquet(path)
    daily_rows = [compute_trajectory_metrics(day) for _, day in _day_groups(frame)]
    if len(daily_rows) != 92:
        raise RuntimeError(f"Expected 92 daily episodes in {path}, found {len(daily_rows)}")
    daily = pd.DataFrame(daily_rows)
    pooled = compute_trajectory_metrics(frame)
    record = {metric: float(daily[metric].mean()) for metric in DAILY_METRICS}
    record.update({metric: float(pooled[metric]) for metric in POOLED_METRICS})
    record["mean_daily_DR_violation_energy_kWh"] = float(
        daily["DR_violation_energy_kWh"].mean()
    )
    record["mean_daily_max_DR_violation_kW"] = float(
        daily["DR_max_violation_kW"].mean()
    )
    record["mean_daily_DR_interval_compliance_pct"] = float(
        daily["DR_interval_compliance_pct"].mean()
    )
    record["mean_daily_DR_event_compliance_pct"] = float(
        daily["DR_event_compliance_pct"].mean()
    )
    record["secondary_daily_average_DR_compliance_pct"] = float(
        daily["DR_interval_compliance_pct"].mean()
    )
    record["test_days"] = int(len(daily))
    return record, daily


def _seed_summary(rows: list[dict], key: str) -> list[dict]:
    frame = pd.DataFrame(rows)
    output = []
    for value, group in frame.groupby(key, sort=False):
        record = {key: value, "seeds": int(group["seed"].nunique())}
        for metric in (
            DAILY_METRICS + POOLED_METRICS + DERIVED_DAILY_DR_METRICS
            + ["secondary_daily_average_DR_compliance_pct"]
        ):
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std_across_seeds"] = float(
                group[metric].std(ddof=1) if len(group) > 1 else 0.0
            )
        output.append(record)
    return output


def evaluate_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    daily_frames = []
    base = ROOT / "trajectories" / "baseline"
    for controller in DETERMINISTIC:
        record, daily = evaluate_trajectory(base / f"{controller}.parquet")
        rows.append({"controller": controller, "seed": 0, **record})
        daily.insert(0, "seed", 0)
        daily.insert(0, "controller", controller)
        daily_frames.append(daily)
    frame = pd.DataFrame(rows)
    output = ensure_output()
    frame.to_csv(output / "BASELINE_REPLAY_METRICS.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(output / "BASELINE_REPLAY_DAILY_METRICS.csv", index=False)
    return frame, pd.concat(daily_frames, ignore_index=True)


def evaluate_learned_controllers() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    daily_frames = []
    base = ROOT / "trajectories" / "learned_controllers"
    for controller in LEARNED:
        for seed in range(1, 6):
            record, daily = evaluate_trajectory(base / f"{controller}_seed_{seed}.parquet")
            rows.append({"controller": controller, "seed": seed, **record})
            daily.insert(0, "seed", seed)
            daily.insert(0, "controller", controller)
            daily_frames.append(daily)
    frame = pd.DataFrame(rows)
    output = ensure_output()
    frame.to_csv(output / "LEARNED_REPLAY_SEED_METRICS.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(output / "LEARNED_REPLAY_DAILY_METRICS.csv", index=False)
    return frame, pd.concat(daily_frames, ignore_index=True)


def evaluate_main() -> tuple[pd.DataFrame, pd.DataFrame]:
    deterministic, _ = evaluate_baselines()
    learned, _ = evaluate_learned_controllers()
    all_seed = pd.concat([deterministic, learned], ignore_index=True)
    summary = pd.DataFrame(_seed_summary(all_seed.to_dict("records"), "controller"))
    output = ensure_output()
    summary.to_csv(output / "TABLE_MAIN_CONTROLLER_RESULTS_REPRODUCED.csv", index=False)
    learned.to_csv(output / "TABLE_SEED_RESULTS_REPRODUCED.csv", index=False)
    return summary, learned


def evaluate_scenarios() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    base = ROOT / "trajectories" / "scenarios"
    for scenario in SCENARIOS:
        for controller in DETERMINISTIC:
            record, _ = evaluate_trajectory(base / f"{scenario}_{controller}.parquet")
            rows.append({"scenario": scenario, "controller": controller, "seed": 0, **record})
        for controller in LEARNED:
            for seed in range(1, 6):
                record, _ = evaluate_trajectory(base / f"{scenario}_{controller}_seed_{seed}.parquet")
                rows.append({"scenario": scenario, "controller": controller, "seed": seed, **record})
    seed_frame = pd.DataFrame(rows)
    summaries = []
    for (scenario, controller), group in seed_frame.groupby(["scenario", "controller"], sort=False):
        record = {"scenario": scenario, "controller": controller, "seeds": int(group["seed"].nunique())}
        for metric in (
            DAILY_METRICS + POOLED_METRICS + DERIVED_DAILY_DR_METRICS
            + ["secondary_daily_average_DR_compliance_pct"]
        ):
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std_across_seeds"] = float(
                group[metric].std(ddof=1) if len(group) > 1 else 0.0
            )
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    output = ensure_output()
    seed_frame.to_csv(output / "SCENARIO_REPLAY_SEED_METRICS.csv", index=False)
    summary.to_csv(output / "TABLE_IT_SCENARIO_RESULTS_REPRODUCED.csv", index=False)
    return summary, seed_frame


def evaluate_ablations() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for variant in ABLATION_ORDER:
        for seed in range(1, 6):
            if variant in LEARNED:
                path = ROOT / "trajectories" / "learned_controllers" / f"{variant}_seed_{seed}.parquet"
            else:
                path = ROOT / "trajectories" / "ablations" / f"{variant}_seed_{seed}.parquet"
            record, _ = evaluate_trajectory(path)
            rows.append({"variant": variant, "seed": seed, **record})
    seed_frame = pd.DataFrame(rows)
    summary = pd.DataFrame(_seed_summary(seed_frame.to_dict("records"), "variant"))
    output = ensure_output()
    seed_frame.to_csv(output / "ABLATION_REPLAY_SEED_METRICS.csv", index=False)
    summary.to_csv(output / "TABLE_TRUE_ABLATION_RESULTS_REPRODUCED.csv", index=False)
    return summary, seed_frame


def _model_entries():
    for controller in LEARNED:
        for seed in range(1, 6):
            yield controller, seed, ROOT / "models" / controller / f"seed_{seed}"
    for controller in ABLATION_ONLY:
        for seed in range(1, 6):
            yield controller, seed, ROOT / "models" / "ablations" / controller / f"seed_{seed}"


def write_model_manifest(load_models: bool = False) -> pd.DataFrame:
    rows = []
    SAC = None
    if load_models:
        from stable_baselines3 import SAC as SACClass
        SAC = SACClass
    for controller, seed, directory in _model_entries():
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        model_path = directory / "selected_validation_checkpoint.zip"
        current_hash = sha256(model_path)
        if current_hash != metadata["model_sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch: {model_path}")
        observation_dimension = int(metadata["observation_dimension"])
        action_dimension = int(metadata["action_dimension"])
        if SAC is not None:
            model = SAC.load(model_path, device="cpu")
            observation_dimension = int(model.observation_space.shape[0])
            action_dimension = int(model.action_space.shape[0])
        rows.append({
            "Controller": controller,
            "Seed": seed,
            "Model path": model_path.relative_to(ROOT).as_posix(),
            "Training steps": int(metadata["training_steps"]),
            "Selected checkpoint step": int(metadata["selected_checkpoint_step"]),
            "Config": (directory / "config_snapshot.yaml").relative_to(ROOT).as_posix(),
            "Observation dimension": observation_dimension,
            "Action dimension": action_dimension,
            "SHA256": current_hash,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "models" / "MODEL_MANIFEST.csv", index=False)
    frame.to_csv(ensure_output() / "MODEL_LOAD_VALIDATION.csv", index=False)
    return frame


def _comparison_rows(reproduced: pd.DataFrame, authoritative: pd.DataFrame,
                     keys: list[str], mapping: dict[str, str], category: str,
                     tolerance: float = 1e-6) -> list[dict]:
    rows = []
    left = authoritative.set_index(keys)
    right = reproduced.set_index(keys)
    for index in left.index:
        if index not in right.index:
            raise RuntimeError(f"Missing reproduced row for {category}: {index}")
        for published_column, reproduced_column in mapping.items():
            published = float(left.loc[index, published_column])
            replayed = float(right.loc[index, reproduced_column])
            difference = replayed - published
            rows.append({
                "Category": category,
                "Item": "|".join(map(str, index if isinstance(index, tuple) else (index,))),
                "Metric": published_column,
                "Published/final value": published,
                "Reproduced value": replayed,
                "Difference": difference,
                "Tolerance": tolerance,
                "Pass": abs(difference) <= tolerance,
            })
    return rows


def compare_outputs(main: pd.DataFrame, seeds: pd.DataFrame,
                    scenarios: pd.DataFrame, ablations: pd.DataFrame) -> pd.DataFrame:
    comparisons: list[dict] = []
    common_map = {
        "daily_cost_EUR_mean": "daily_cost_EUR_mean",
        "carbon_emissions_kgCO2_mean": "carbon_emissions_kgCO2_mean",
        "VPP_tracking_RMSE_kW_mean": "VPP_tracking_RMSE_kW_mean",
        "VPP_tracking_MAE_kW_mean": "VPP_tracking_MAE_kW_mean",
        "DR_interval_compliance_pct_mean": "DR_interval_compliance_pct_mean",
        "DR_event_compliance_pct_mean": "DR_event_compliance_pct_mean",
        "battery_throughput_kWh_mean": "battery_throughput_kWh_mean",
        "terminal_SOC_error_mean": "terminal_SOC_error_mean",
        "physical_projection_fraction_pct_mean": "physical_projection_fraction_pct_mean",
        "service_coaching_fraction_pct_mean": "service_coaching_fraction_pct_mean",
    }
    authoritative_main = pd.read_csv(ROOT / "tables" / "TABLE_MAIN_CONTROLLER_RESULTS.csv")
    comparisons += _comparison_rows(main, authoritative_main, ["controller"], common_map, "main")
    explicit_dr_map = {
        "mean_daily_DR_violation_energy_kWh_mean": "mean_daily_DR_violation_energy_kWh_mean",
        "full_quarter_DR_violation_energy_kWh_mean": "DR_violation_energy_kWh_mean",
        "mean_daily_max_DR_violation_kW_mean": "mean_daily_max_DR_violation_kW_mean",
        "full_quarter_max_DR_violation_kW_mean": "DR_max_violation_kW_mean",
    }
    comparisons += _comparison_rows(
        main, authoritative_main, ["controller"], explicit_dr_map, "main_explicit_DR_aggregation"
    )

    # The frozen manuscript-source table mixed daily deterministic values and
    # full-quarter learned-controller values in the same two columns. Preserve
    # and validate those cells on their actual aggregation basis as audit
    # evidence; publication-facing tables use explicit columns above.
    mixed_source = pd.read_csv(
        ROOT / "results" / "authoritative"
        / "SOURCE_TABLE_MAIN_CONTROLLER_RESULTS_MIXED_DR_AGGREGATION.csv"
    )
    comparisons += _comparison_rows(
        main.loc[main["controller"].isin(DETERMINISTIC)],
        mixed_source.loc[mixed_source["controller"].isin(DETERMINISTIC)],
        ["controller"],
        {
            "DR_violation_energy_kWh_mean": "mean_daily_DR_violation_energy_kWh_mean",
            "DR_max_violation_kW_mean": "mean_daily_max_DR_violation_kW_mean",
        },
        "source_mixed_table_deterministic_daily_basis",
    )
    comparisons += _comparison_rows(
        main.loc[main["controller"].isin(LEARNED)],
        mixed_source.loc[mixed_source["controller"].isin(LEARNED)],
        ["controller"],
        {
            "DR_violation_energy_kWh_mean": "DR_violation_energy_kWh_mean",
            "DR_max_violation_kW_mean": "DR_max_violation_kW_mean",
        },
        "source_mixed_table_learned_full_quarter_basis",
    )

    authoritative_seeds = pd.read_csv(ROOT / "tables" / "TABLE_SEED_RESULTS.csv")
    seed_map = {
        "daily_cost_EUR": "daily_cost_EUR",
        "carbon_emissions_kgCO2": "carbon_emissions_kgCO2",
        "VPP_tracking_RMSE_kW": "VPP_tracking_RMSE_kW",
        "VPP_tracking_MAE_kW": "VPP_tracking_MAE_kW",
        "DR_interval_compliance_pct": "DR_interval_compliance_pct",
        "DR_event_compliance_pct": "DR_event_compliance_pct",
        "DR_violation_energy_kWh": "DR_violation_energy_kWh",
        "battery_throughput_kWh": "battery_throughput_kWh",
        "terminal_SOC_error": "terminal_SOC_error",
        "physical_projection_fraction_pct": "physical_projection_fraction_pct",
        "service_coaching_fraction_pct": "service_coaching_fraction_pct",
    }
    comparisons += _comparison_rows(seeds, authoritative_seeds, ["controller", "seed"], seed_map, "seed")

    authoritative_scenarios = pd.read_csv(ROOT / "tables" / "TABLE_IT_SCENARIO_RESULTS.csv")
    scenario_map = {
        "daily_cost_EUR_mean": "daily_cost_EUR_mean",
        "carbon_emissions_kgCO2_mean": "carbon_emissions_kgCO2_mean",
        "VPP_tracking_RMSE_kW_mean": "VPP_tracking_RMSE_kW_mean",
        "VPP_tracking_MAE_kW_mean": "VPP_tracking_MAE_kW_mean",
        "mean_daily_DR_interval_compliance_pct_mean": "mean_daily_DR_interval_compliance_pct_mean",
        "mean_daily_DR_event_compliance_pct_mean": "mean_daily_DR_event_compliance_pct_mean",
        "pooled_DR_active_interval_compliance_pct_mean": "DR_interval_compliance_pct_mean",
        "pooled_DR_event_compliance_pct_mean": "DR_event_compliance_pct_mean",
        "mean_daily_DR_violation_energy_kWh_mean": "mean_daily_DR_violation_energy_kWh_mean",
        "mean_daily_max_DR_violation_kW_mean": "mean_daily_max_DR_violation_kW_mean",
        "full_quarter_DR_violation_energy_kWh_mean": "DR_violation_energy_kWh_mean",
        "full_quarter_max_DR_violation_kW_mean": "DR_max_violation_kW_mean",
        "battery_throughput_kWh_mean": "battery_throughput_kWh_mean",
        "terminal_SOC_error_mean": "terminal_SOC_error_mean",
    }
    comparisons += _comparison_rows(
        scenarios, authoritative_scenarios, ["scenario", "controller"], scenario_map, "scenario"
    )

    authoritative_ablations = pd.read_csv(ROOT / "tables" / "TABLE_TRUE_ABLATION_RESULTS.csv")
    ablation_map = {
        "daily_cost_EUR_mean": "daily_cost_EUR_mean",
        "VPP_tracking_RMSE_kW_mean": "VPP_tracking_RMSE_kW_mean",
        "mean_daily_DR_interval_compliance_pct_mean": "mean_daily_DR_interval_compliance_pct_mean",
        "pooled_DR_active_interval_compliance_pct_mean": "DR_interval_compliance_pct_mean",
        "mean_daily_DR_violation_energy_kWh_mean": "mean_daily_DR_violation_energy_kWh_mean",
        "mean_daily_max_DR_violation_kW_mean": "mean_daily_max_DR_violation_kW_mean",
        "full_quarter_DR_violation_energy_kWh_mean": "DR_violation_energy_kWh_mean",
        "full_quarter_max_DR_violation_kW_mean": "DR_max_violation_kW_mean",
        "battery_throughput_kWh_mean": "battery_throughput_kWh_mean",
        "terminal_SOC_error_mean": "terminal_SOC_error_mean",
        "physical_projection_fraction_pct_mean": "physical_projection_fraction_pct_mean",
        "service_coaching_fraction_pct_mean": "service_coaching_fraction_pct_mean",
    }
    comparisons += _comparison_rows(
        ablations, authoritative_ablations, ["variant"], ablation_map, "ablation"
    )
    frame = pd.DataFrame(comparisons)
    frame.to_csv(ROOT / "results" / "REPRODUCTION_VALIDATION.csv", index=False)
    if not bool(frame["Pass"].all()):
        failed = frame.loc[~frame["Pass"]]
        raise RuntimeError(f"Reproduction comparison failed for {len(failed)} metrics")
    return frame


def prepare_consistent_authoritative_tables(
    main: pd.DataFrame, scenarios: pd.DataFrame, ablations: pd.DataFrame
) -> None:
    """Create publication-facing DR columns from preserved source tables."""
    source_main = pd.read_csv(
        ROOT / "results" / "authoritative"
        / "SOURCE_TABLE_MAIN_CONTROLLER_RESULTS_MIXED_DR_AGGREGATION.csv"
    )
    ambiguous_main_columns = [
        column for column in source_main.columns
        if column.startswith("DR_violation_energy_kWh_")
        or column.startswith("DR_max_violation_kW_")
    ]
    main_dr = main[[
        "controller",
        "mean_daily_DR_violation_energy_kWh_mean",
        "mean_daily_DR_violation_energy_kWh_std_across_seeds",
        "DR_violation_energy_kWh_mean",
        "DR_violation_energy_kWh_std_across_seeds",
        "mean_daily_max_DR_violation_kW_mean",
        "mean_daily_max_DR_violation_kW_std_across_seeds",
        "DR_max_violation_kW_mean",
        "DR_max_violation_kW_std_across_seeds",
    ]].rename(columns={
        "DR_violation_energy_kWh_mean": "full_quarter_DR_violation_energy_kWh_mean",
        "DR_violation_energy_kWh_std_across_seeds": "full_quarter_DR_violation_energy_kWh_std_across_seeds",
        "DR_max_violation_kW_mean": "full_quarter_max_DR_violation_kW_mean",
        "DR_max_violation_kW_std_across_seeds": "full_quarter_max_DR_violation_kW_std_across_seeds",
    })
    consistent_main = source_main.drop(columns=ambiguous_main_columns).merge(
        main_dr, on="controller", how="left", validate="one_to_one"
    )
    consistent_main.to_csv(ROOT / "tables" / "TABLE_MAIN_CONTROLLER_RESULTS.csv", index=False)

    source_safety = pd.read_csv(
        ROOT / "results" / "authoritative"
        / "SOURCE_TABLE_SAFETY_RESULTS_MIXED_DR_AGGREGATION.csv"
    )
    ambiguous_safety_columns = [
        column for column in source_safety.columns
        if column in {"DR_violation_energy_kWh_mean", "DR_max_violation_kW_mean"}
    ]
    source_safety.drop(columns=ambiguous_safety_columns).merge(
        main_dr, on="controller", how="left", validate="one_to_one"
    ).to_csv(ROOT / "tables" / "TABLE_SAFETY_RESULTS.csv", index=False)

    source_scenarios = pd.read_csv(
        ROOT / "results" / "authoritative"
        / "SOURCE_TABLE_IT_SCENARIO_RESULTS_MIXED_DR_AGGREGATION.csv"
    )
    scenario_dr = scenarios[[
        "scenario", "controller",
        "mean_daily_DR_interval_compliance_pct_mean",
        "mean_daily_DR_interval_compliance_pct_std_across_seeds",
        "mean_daily_DR_event_compliance_pct_mean",
        "mean_daily_DR_event_compliance_pct_std_across_seeds",
        "DR_interval_compliance_pct_mean", "DR_interval_compliance_pct_std_across_seeds",
        "DR_event_compliance_pct_mean", "DR_event_compliance_pct_std_across_seeds",
        "mean_daily_DR_violation_energy_kWh_mean",
        "mean_daily_DR_violation_energy_kWh_std_across_seeds",
        "mean_daily_max_DR_violation_kW_mean",
        "mean_daily_max_DR_violation_kW_std_across_seeds",
        "DR_violation_energy_kWh_mean", "DR_violation_energy_kWh_std_across_seeds",
        "DR_max_violation_kW_mean", "DR_max_violation_kW_std_across_seeds",
    ]].rename(columns={
        "DR_interval_compliance_pct_mean": "pooled_DR_active_interval_compliance_pct_mean",
        "DR_interval_compliance_pct_std_across_seeds": "pooled_DR_active_interval_compliance_pct_std_across_seeds",
        "DR_event_compliance_pct_mean": "pooled_DR_event_compliance_pct_mean",
        "DR_event_compliance_pct_std_across_seeds": "pooled_DR_event_compliance_pct_std_across_seeds",
        "DR_violation_energy_kWh_mean": "full_quarter_DR_violation_energy_kWh_mean",
        "DR_violation_energy_kWh_std_across_seeds": "full_quarter_DR_violation_energy_kWh_std_across_seeds",
        "DR_max_violation_kW_mean": "full_quarter_max_DR_violation_kW_mean",
        "DR_max_violation_kW_std_across_seeds": "full_quarter_max_DR_violation_kW_std_across_seeds",
    })
    ambiguous_scenario_columns = [column for column in source_scenarios if column.startswith("DR_")]
    source_scenarios.drop(columns=ambiguous_scenario_columns).merge(
        scenario_dr, on=["scenario", "controller"], how="left", validate="one_to_one"
    ).to_csv(
        ROOT / "tables" / "TABLE_IT_SCENARIO_RESULTS.csv", index=False
    )

    source_ablation_path = (
        ROOT / "results" / "authoritative" / "SOURCE_TABLE_TRUE_ABLATION_RESULTS_DAILY_AVERAGE.csv"
    )
    if not source_ablation_path.is_file():
        raise FileNotFoundError(source_ablation_path)
    source_ablations = pd.read_csv(source_ablation_path)
    ablation_dr = ablations[[
        "variant",
        "mean_daily_DR_interval_compliance_pct_mean",
        "mean_daily_DR_interval_compliance_pct_std_across_seeds",
        "DR_interval_compliance_pct_mean", "DR_interval_compliance_pct_std_across_seeds",
        "mean_daily_DR_violation_energy_kWh_mean",
        "mean_daily_DR_violation_energy_kWh_std_across_seeds",
        "mean_daily_max_DR_violation_kW_mean",
        "mean_daily_max_DR_violation_kW_std_across_seeds",
        "DR_violation_energy_kWh_mean", "DR_violation_energy_kWh_std_across_seeds",
        "DR_max_violation_kW_mean", "DR_max_violation_kW_std_across_seeds",
    ]].rename(columns={
        "DR_interval_compliance_pct_mean": "pooled_DR_active_interval_compliance_pct_mean",
        "DR_interval_compliance_pct_std_across_seeds": "pooled_DR_active_interval_compliance_pct_std",
        "mean_daily_DR_interval_compliance_pct_std_across_seeds": "mean_daily_DR_interval_compliance_pct_std",
        "mean_daily_DR_violation_energy_kWh_std_across_seeds": "mean_daily_DR_violation_energy_kWh_std",
        "mean_daily_max_DR_violation_kW_std_across_seeds": "mean_daily_max_DR_violation_kW_std",
        "DR_violation_energy_kWh_mean": "full_quarter_DR_violation_energy_kWh_mean",
        "DR_violation_energy_kWh_std_across_seeds": "full_quarter_DR_violation_energy_kWh_std",
        "DR_max_violation_kW_mean": "full_quarter_max_DR_violation_kW_mean",
        "DR_max_violation_kW_std_across_seeds": "full_quarter_max_DR_violation_kW_std",
    })
    ambiguous_ablation_columns = [column for column in source_ablations if column.startswith("DR_")]
    source_ablations.drop(columns=ambiguous_ablation_columns).merge(
        ablation_dr, on="variant", how="left", validate="one_to_one"
    ).to_csv(
        ROOT / "tables" / "TABLE_TRUE_ABLATION_RESULTS.csv", index=False
    )


def run_metric_reproduction(load_models: bool = True) -> dict:
    validation = validate_dataset()
    models = write_model_manifest(load_models=load_models)
    main, seeds = evaluate_main()
    scenarios, _ = evaluate_scenarios()
    ablations, _ = evaluate_ablations()
    prepare_consistent_authoritative_tables(main, scenarios, ablations)
    comparison = compare_outputs(main, seeds, scenarios, ablations)
    report = {
        "dataset_validation": "PASS" if validation["passed"] else "FAIL",
        "model_checkpoints_loaded": int(len(models)) if load_models else 0,
        "main_controllers": int(len(main)),
        "learned_seed_rows": int(len(seeds)),
        "scenario_controller_rows": int(len(scenarios)),
        "ablation_variants": int(len(ablations)),
        "metric_comparisons": int(len(comparison)),
        "metric_comparisons_passed": int(comparison["Pass"].sum()),
        "passed": bool(comparison["Pass"].all() and validation["passed"]),
        "training_performed": False,
    }
    (ensure_output() / "REPRODUCTION_SUMMARY.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
