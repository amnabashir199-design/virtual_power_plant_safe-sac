"""Post-process frozen Phase 2 IT-load scenarios without altering experiments.

This script reads only the frozen publication artifacts in ``results/publication_v1``
and writes new descriptive analysis files below ``results/it_load_analysis``.  It
does not build scenarios, evaluate policies, train models, or edit authoritative
results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
PUBLICATION = ROOT / "results" / "publication_v1"
OUTPUT = ROOT / "results" / "it_load_analysis"
FIGURES = OUTPUT / "figures"

DATASET_DIR = PUBLICATION / "scenarios" / "datasets"
TRAJECTORY_DIR = PUBLICATION / "scenarios" / "trajectories"
SEED_MEANS_PATH = PUBLICATION / "scenarios" / "scenario_seed_means.csv"
DAILY_METRICS_PATH = PUBLICATION / "scenarios" / "scenario_daily_metrics.csv"
TRAINING_SEED_SUMMARY_PATH = PUBLICATION / "scenarios" / "scenario_training_seed_summary.csv"
PUBLICATION_MANIFEST_PATH = PUBLICATION / "PUBLICATION_MANIFEST.json"
PUBLICATION_REPORT_PATH = PUBLICATION / "PUBLICATION_REPAIR_REPORT.md"
FROZEN_DATASET_PATH = PUBLICATION / "dataset" / "green_dc_vpp_publication_v1_2019_15min.parquet"
DATASET_METADATA_PATH = PUBLICATION / "dataset" / "dataset_metadata.json"
METRICS_SOURCE_PATH = ROOT / "src" / "green_dc_vpp" / "metrics.py"
SCENARIO_SOURCE_PATH = ROOT / "scripts" / "24_build_it_load_scenarios.py"
SCENARIO_EVALUATION_SOURCE_PATH = ROOT / "scripts" / "36_evaluate_publication_scenarios.py"
PLOTTING_SOURCE_PATH = ROOT / "scripts" / "37_make_publication_figures.py"
CONFIG_PATH = ROOT / "configs" / "publication_v1" / "proposed_residual_safesac.yaml"

PROPOSED_CONTROLLER = "proposed_residual_safesac"
POWER_COLUMNS = {
    "it": "P_IT_kW",
    "cooling": "P_cooling_kW",
    "other": "P_other_kW",
    "total_facility": "P_total_dc_kW",
    "pv_generation": "Ppv_kW",
    "net_facility": "P_net_without_battery_kW",
    "grid_import_without_battery": "P_import_without_battery_kW",
}


def load_scenario_definitions() -> tuple[list[str], dict[str, Any]]:
    """Read the implemented names and metadata from the frozen scenario generator."""

    spec = importlib.util.spec_from_file_location("frozen_scenario_definitions", SCENARIO_SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load implemented scenario definitions from {SCENARIO_SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    definitions = getattr(module, "DEFINITIONS")
    return list(definitions), definitions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_time_step_hours(timestamps: pd.Series) -> float:
    ordered = pd.to_datetime(timestamps, utc=True).sort_values().reset_index(drop=True)
    diffs = ordered.diff().dropna().dt.total_seconds().to_numpy() / 3600.0
    if len(diffs) == 0 or not np.isfinite(diffs).all() or (diffs <= 0).any():
        raise ValueError("Unable to infer a positive simulation timestep from scenario timestamps.")
    timestep = float(pd.Series(diffs).mode().iloc[0])
    if not np.allclose(diffs, timestep, rtol=0.0, atol=1e-12):
        raise ValueError("Scenario test timestamps are not uniformly spaced; energy/ramp statistics are ambiguous.")
    return timestep


def power_statistics(values: pd.Series, timestamps: pd.Series, timestep_h: float, prefix: str) -> dict[str, float]:
    """Return unrounded load, daily-energy, and ramp statistics for one power series."""

    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if numeric.empty:
        raise ValueError(f"Cannot compute {prefix} characteristics from an empty series.")
    day = pd.to_datetime(timestamps, utc=True).dt.floor("D")
    daily_energy = (numeric * timestep_h).groupby(day).sum()
    ramps = numeric.diff().abs().iloc[1:] / timestep_h
    mean_value = float(numeric.mean())
    std_value = float(numeric.std(ddof=1))
    return {
        f"mean_{prefix}_kW": mean_value,
        f"median_{prefix}_kW": float(numeric.median()),
        f"std_{prefix}_kW": std_value,
        f"min_{prefix}_kW": float(numeric.min()),
        f"max_{prefix}_kW": float(numeric.max()),
        f"p95_{prefix}_kW": float(numeric.quantile(0.95)),
        f"mean_daily_{prefix}_energy_kWh": float(daily_energy.mean()),
        f"std_daily_{prefix}_energy_kWh": float(daily_energy.std(ddof=1)),
        f"min_daily_{prefix}_energy_kWh": float(daily_energy.min()),
        f"max_daily_{prefix}_energy_kWh": float(daily_energy.max()),
        f"{prefix}_peak_to_average_ratio": float(numeric.max() / mean_value) if abs(mean_value) > 1e-12 else np.nan,
        f"{prefix}_coefficient_of_variation": float(std_value / mean_value) if abs(mean_value) > 1e-12 else np.nan,
        f"mean_absolute_{prefix}_ramp_kW_per_h": float(ramps.mean()) if not ramps.empty else np.nan,
        f"max_absolute_{prefix}_ramp_kW_per_h": float(ramps.max()) if not ramps.empty else np.nan,
        f"p95_absolute_{prefix}_ramp_kW_per_h": float(ramps.quantile(0.95)) if not ramps.empty else np.nan,
    }


def scenario_transformation(name: str) -> str:
    """Exact plain-language rendering of the implemented transformation."""

    transformations = {
        "baseline_real": "No IT-load modification; retains the original real/hybrid IT-load profile.",
        "low_it_load": "P_IT,t = 0.75 × baseline P_IT,t at every interval.",
        "high_it_load": "P_IT,t = 1.25 × baseline P_IT,t at every interval.",
        "ai_training_bursty": (
            "P_IT,t = 1.45 × baseline during Tuesday, Thursday, and Saturday 08:00–16:00 UTC; "
            "baseline otherwise."
        ),
        "smooth_cloud_service": (
            "Centered four-hour (16 interval) rolling mean, normalized separately each UTC day to preserve "
            "baseline daily IT energy."
        ),
        "workload_shiftable": (
            "80% inflexible and 20% daily flexible energy redistributed with price, carbon, and DR preference; "
            "daily IT energy preserved."
        ),
        "hot_day_cooling_stress": (
            "IT load retained unchanged; cooling ratio increases progressively by up to 25% in the hottest "
            "ambient-temperature decile."
        ),
        "dr_coincident_high_load": (
            "P_IT,t = 1.25 × baseline during DR-active intervals and a centered ±1-hour shoulder; baseline otherwise."
        ),
    }
    try:
        return transformations[name]
    except KeyError as exc:
        raise KeyError(f"No implemented transformation text for scenario {name!r}") from exc


def differences_from_baseline(frame: pd.DataFrame, baseline: pd.DataFrame, timestep_h: float) -> dict[str, Any]:
    """Describe source-data changes relative to the unmodified frozen baseline."""

    if len(frame) != len(baseline) or not frame["timestamp_utc"].equals(baseline["timestamp_utc"]):
        raise ValueError("Scenario timestamps do not align with the frozen baseline scenario.")

    def changed(column: str, tolerance: float = 1e-9) -> bool:
        return not np.allclose(
            pd.to_numeric(frame[column], errors="raise"),
            pd.to_numeric(baseline[column], errors="raise"),
            rtol=0.0,
            atol=tolerance,
        )

    days = frame["timestamp_utc"].dt.floor("D")
    scenario_energy = (frame["P_IT_kW"] * timestep_h).groupby(days).sum()
    baseline_energy = (baseline["P_IT_kW"] * timestep_h).groupby(days).sum()
    energy_difference = scenario_energy - baseline_energy
    return {
        "it_load_changed_from_baseline": changed("P_IT_kW"),
        "cooling_load_changed_from_baseline": changed("P_cooling_kW"),
        "pv_generation_changed_from_baseline": changed("Ppv_kW"),
        "dr_timing_changed_from_baseline": changed("DR_active"),
        "dr_import_limit_changed_from_baseline": changed("DR_import_limit_kW"),
        "grid_reference_changed_from_baseline": changed("Pgrid_ref_kW"),
        "facility_forecast_changed_from_baseline": changed("Pdc_forecast_kW"),
        "daily_it_energy_preserved_from_baseline": bool(np.allclose(energy_difference, 0.0, rtol=0.0, atol=1e-9)),
        "max_abs_daily_it_energy_difference_from_baseline_kWh": float(energy_difference.abs().max()),
        "mean_abs_daily_it_energy_difference_from_baseline_kWh": float(energy_difference.abs().mean()),
        "mean_abs_daily_it_energy_difference_from_baseline_pct": float(
            (energy_difference.abs() / baseline_energy.replace(0.0, np.nan) * 100.0).mean()
        ),
    }


def load_test_scenarios(scenario_names: list[str]) -> tuple[dict[str, pd.DataFrame], float]:
    frames: dict[str, pd.DataFrame] = {}
    timestep: float | None = None
    required = set(POWER_COLUMNS.values()) | {
        "timestamp_utc", "split", "scenario_name", "scenario_description", "scenario_family", "cooling_model_note",
        "DR_active", "DR_import_limit_kW", "Pgrid_ref_kW", "Pdc_forecast_kW",
    }
    for name in scenario_names:
        path = DATASET_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Frozen scenario dataset is missing: {path}")
        raw = pd.read_parquet(path)
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise ValueError(f"{path} lacks required columns: {missing}")
        raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="raise")
        test = raw.loc[raw["split"].astype(str).str.lower().eq("test")].copy()
        if test.empty:
            raise ValueError(f"{path} has no test split.")
        test = test.sort_values("timestamp_utc").reset_index(drop=True)
        if test["scenario_name"].nunique() != 1 or test["scenario_name"].iloc[0] != name:
            raise ValueError(f"{path} scenario metadata does not identify {name!r}.")
        current_timestep = infer_time_step_hours(test["timestamp_utc"])
        if timestep is None:
            timestep = current_timestep
        elif not np.isclose(timestep, current_timestep, rtol=0.0, atol=1e-12):
            raise ValueError("Frozen scenarios do not use a common timestep.")
        frames[name] = test
    assert timestep is not None
    return frames, timestep


def build_characteristics(
    frames: dict[str, pd.DataFrame], definitions: dict[str, Any], timestep_h: float
) -> pd.DataFrame:
    baseline = frames["baseline_real"]
    rows: list[dict[str, Any]] = []
    for name, frame in frames.items():
        meta = definitions[name]
        row: dict[str, Any] = {
            "scenario": name,
            "scenario_family": str(frame["scenario_family"].iloc[0]),
            "scenario_description": str(frame["scenario_description"].iloc[0]),
            "it_load_transformation": scenario_transformation(name),
            "cooling_model_note": str(frame["cooling_model_note"].iloc[0]),
            "test_start_utc": frame["timestamp_utc"].iloc[0].isoformat(),
            "test_end_utc": frame["timestamp_utc"].iloc[-1].isoformat(),
            "test_interval_count": int(len(frame)),
            "test_day_count": int(frame["timestamp_utc"].dt.floor("D").nunique()),
            "time_step_hours": timestep_h,
            "controller_evaluated": PROPOSED_CONTROLLER,
            "implemented_definition_family": str(meta.family),
        }
        row.update(differences_from_baseline(frame, baseline, timestep_h))
        for prefix, column in POWER_COLUMNS.items():
            row.update(power_statistics(frame[column], frame["timestamp_utc"], timestep_h, prefix))
        rows.append(row)
    return pd.DataFrame(rows)


def build_performance_table(characteristics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate only frozen proposed-controller scenario result rows across five seeds."""

    seed_means = pd.read_csv(SEED_MEANS_PATH)
    daily = pd.read_csv(DAILY_METRICS_PATH)
    proposed = seed_means.loc[seed_means["controller"].eq(PROPOSED_CONTROLLER)].copy()
    proposed_daily = daily.loc[daily["controller"].eq(PROPOSED_CONTROLLER)].copy()
    expected_scenarios = set(characteristics["scenario"])
    if set(proposed["scenario"]) != expected_scenarios:
        raise ValueError("Proposed-controller seed summaries do not cover exactly the frozen scenarios.")
    if proposed.groupby("scenario")["seed"].nunique().nunique() != 1:
        raise ValueError("The number of proposed-controller seeds is inconsistent across scenarios.")
    if proposed.groupby("scenario")["seed"].nunique().iloc[0] != 5:
        raise ValueError("Expected five frozen proposed-controller seeds per scenario.")

    metric_columns = [
        "VPP_tracking_RMSE_kW",
        "VPP_tracking_MAE_kW",
        "DR_interval_compliance_pct",
        "DR_event_compliance_pct",
        "DR_violation_energy_kWh",
        "DR_max_violation_kW",
        "daily_cost_EUR",
        "carbon_emissions_kgCO2",
        "battery_throughput_kWh",
        "terminal_SOC_error",
        "physical_projection_fraction_pct",
        "service_coaching_fraction_pct",
        "mean_daily_peak_import_kW",
        "horizon_max_grid_import_kW",
        "DR_event_count",
        "DR_successful_event_count",
    ]
    missing = sorted(set(metric_columns).difference(proposed.columns))
    if missing:
        raise ValueError(f"Frozen seed summary has no required metrics: {missing}")
    records: list[dict[str, Any]] = []
    for scenario, group in proposed.groupby("scenario", sort=False):
        record: dict[str, Any] = {
            "scenario": scenario,
            "controller": PROPOSED_CONTROLLER,
            "n_seeds": int(group["seed"].nunique()),
            "n_daily_metric_samples": int(
                len(proposed_daily.loc[proposed_daily["scenario"].eq(scenario)])
            ),
            "days_per_seed": int(
                proposed_daily.loc[proposed_daily["scenario"].eq(scenario)].groupby("seed")["day_index"].nunique().iloc[0]
            ),
        }
        for metric in metric_columns:
            record[f"{metric}_mean_across_seeds"] = float(group[metric].mean())
            record[f"{metric}_std_across_seeds"] = float(group[metric].std(ddof=1))
        safety = group["physical_projection_fraction_pct"] + group["service_coaching_fraction_pct"]
        record["safety_intervention_fraction_pct_mean_across_seeds"] = float(safety.mean())
        record["safety_intervention_fraction_pct_std_across_seeds"] = float(safety.std(ddof=1))
        records.append(record)
    performance = pd.DataFrame(records)
    # The frozen scenario summary averages episode-level DR scores, and its
    # no-DR daily episodes receive 100%.  Preserve those rows above, but derive
    # the scientifically appropriate DR values over each full saved test
    # trajectory here, without touching any frozen artifact.
    pooled_dr_rows: list[dict[str, Any]] = []
    for scenario in characteristics["scenario"]:
        trajectory = load_proposed_trajectories(str(scenario))
        per_seed: list[dict[str, Any]] = []
        for seed, seed_trajectory in trajectory.groupby("seed", sort=True):
            ordered = seed_trajectory.sort_values("timestamp_utc").reset_index(drop=True)
            timestep = infer_time_step_hours(ordered["timestamp_utc"])
            active = ordered["DR_active"].astype(bool)
            limit = pd.to_numeric(ordered["DR_import_limit_kW"], errors="raise")
            feasible = active & np.isfinite(limit)
            imported = pd.to_numeric(ordered["P_import_kW"], errors="raise")
            excess = (imported - limit).where(feasible, 0.0)
            violation = feasible & (excess > 1e-6)
            effective_excess = excess.clip(lower=0.0).where(violation, 0.0)
            timestamps = pd.to_datetime(ordered["timestamp_utc"], utc=True)
            contiguous = timestamps.diff().eq(pd.Timedelta(hours=timestep))
            event_start = feasible & (~feasible.shift(fill_value=False) | ~contiguous)
            event_id = event_start.cumsum().where(feasible)
            event_success = (~violation).where(feasible).groupby(event_id).all()
            active_count = int(feasible.sum())
            event_count = int(len(event_success))
            tracking_error = pd.to_numeric(ordered["tracking_error_kW"], errors="raise")
            per_seed.append(
                {
                    "seed": int(seed),
                    "DR_active_interval_count_pooled_test": active_count,
                    "DR_event_count_pooled_test": event_count,
                    "DR_interval_compliance_pct_pooled_test": float(
                        100.0 if active_count == 0 else 100.0 * (1.0 - violation.sum() / active_count)
                    ),
                    "DR_event_compliance_pct_pooled_test": float(
                        100.0 if event_count == 0 else 100.0 * event_success.mean()
                    ),
                    "DR_violation_energy_kWh_pooled_test": float((effective_excess * timestep).sum()),
                    "DR_max_violation_kW_pooled_test": float(effective_excess.max()),
                    "VPP_tracking_RMSE_kW_pooled_test": float(np.sqrt(np.mean(np.square(tracking_error)))),
                }
            )
        pooled = pd.DataFrame(per_seed)
        row: dict[str, Any] = {"scenario": scenario}
        for column in pooled.columns:
            if column == "seed":
                continue
            row[f"{column}_mean_across_seeds"] = float(pooled[column].mean())
            row[f"{column}_std_across_seeds"] = float(pooled[column].std(ddof=1))
        pooled_dr_rows.append(row)
    performance = performance.merge(
        pd.DataFrame(pooled_dr_rows), on="scenario", how="inner", validate="one_to_one"
    )
    master_columns = [
        "scenario",
        "scenario_family",
        "test_interval_count",
        "test_day_count",
        "time_step_hours",
        "mean_it_kW",
        "max_it_kW",
        "std_it_kW",
        "it_coefficient_of_variation",
        "it_peak_to_average_ratio",
        "mean_absolute_it_ramp_kW_per_h",
        "max_absolute_it_ramp_kW_per_h",
        "p95_it_kW",
        "p95_absolute_it_ramp_kW_per_h",
        "mean_daily_it_energy_kWh",
    ]
    master = characteristics.merge(performance, on="scenario", how="inner", validate="one_to_one")
    trailing = [column for column in master.columns if column not in master_columns]
    master = master[master_columns + trailing]
    return performance, master


def compute_correlations(master: pd.DataFrame) -> pd.DataFrame:
    """Generate exploratory Pearson descriptives; n=8 is not inferential evidence."""

    pairs = [
        ("mean_it_kW", "VPP_tracking_RMSE_kW_mean_across_seeds", "mean IT load vs VPP tracking RMSE"),
        ("max_it_kW", "VPP_tracking_RMSE_kW_mean_across_seeds", "peak IT load vs VPP tracking RMSE"),
        ("it_coefficient_of_variation", "VPP_tracking_RMSE_kW_mean_across_seeds", "IT CV vs VPP tracking RMSE"),
        ("mean_absolute_it_ramp_kW_per_h", "VPP_tracking_RMSE_kW_mean_across_seeds", "mean IT ramp vs VPP tracking RMSE"),
        ("mean_absolute_it_ramp_kW_per_h", "battery_throughput_kWh_mean_across_seeds", "mean IT ramp vs battery throughput"),
        ("max_it_kW", "DR_interval_compliance_pct_pooled_test_mean_across_seeds", "peak IT load vs pooled-test DR interval compliance"),
        ("mean_daily_it_energy_kWh", "daily_cost_EUR_mean_across_seeds", "daily IT energy vs daily electricity cost"),
        ("mean_daily_it_energy_kWh", "carbon_emissions_kgCO2_mean_across_seeds", "daily IT energy vs daily carbon emissions"),
    ]
    rows: list[dict[str, Any]] = []
    for x_column, y_column, label in pairs:
        data = master[[x_column, y_column]].dropna()
        correlation = float(data[x_column].corr(data[y_column], method="pearson")) if len(data) >= 3 else np.nan
        rows.append(
            {
                "analysis_type": "exploratory_pearson_correlation",
                "comparison": label,
                "metric_x": x_column,
                "metric_y": y_column,
                "n_scenarios": int(len(data)),
                "pearson_r": correlation,
                "inference_note": (
                    "Descriptive only: eight deterministic, non-independent synthetic scenario families; "
                    "no hypothesis test or causal inference is warranted."
                ),
            }
        )
    return pd.DataFrame(rows)


def percent_change(value: float, baseline: float) -> float:
    return float((value - baseline) / baseline * 100.0) if abs(baseline) > 1e-12 else np.nan


def comparison_rows(master: pd.DataFrame) -> pd.DataFrame:
    """Write explicit low/baseline/high and shape-only descriptive comparisons."""

    indexed = master.set_index("scenario")
    rows: list[dict[str, Any]] = []
    comparison_metrics = [
        "mean_it_kW",
        "max_it_kW",
        "mean_daily_it_energy_kWh",
        "VPP_tracking_RMSE_kW_mean_across_seeds",
        "DR_interval_compliance_pct_pooled_test_mean_across_seeds",
        "DR_violation_energy_kWh_pooled_test_mean_across_seeds",
        "daily_cost_EUR_mean_across_seeds",
        "carbon_emissions_kgCO2_mean_across_seeds",
        "battery_throughput_kWh_mean_across_seeds",
        "safety_intervention_fraction_pct_mean_across_seeds",
    ]
    for scenario in ["low_it_load", "high_it_load"]:
        for metric in comparison_metrics:
            value = float(indexed.loc[scenario, metric])
            reference = float(indexed.loc["baseline_real", metric])
            rows.append(
                {
                    "analysis_type": "IT_load_magnitude_comparison",
                    "comparison": f"{scenario} versus baseline_real",
                    "metric_x": metric,
                    "metric_y": "baseline_real",
                    "n_scenarios": 2,
                    "scenario_value": value,
                    "baseline_value": reference,
                    "absolute_difference": float(value - reference),
                    "percent_difference_from_baseline": percent_change(value, reference),
                    "inference_note": "Descriptive paired scenario contrast; no causal or statistical claim.",
                }
            )
    for scenario in ["smooth_cloud_service", "ai_training_bursty", "workload_shiftable"]:
        for metric in comparison_metrics:
            value = float(indexed.loc[scenario, metric])
            reference = float(indexed.loc["baseline_real", metric])
            rows.append(
                {
                    "analysis_type": "IT_temporal_structure_comparison",
                    "comparison": f"{scenario} versus baseline_real",
                    "metric_x": metric,
                    "metric_y": "baseline_real",
                    "n_scenarios": 2,
                    "scenario_value": value,
                    "baseline_value": reference,
                    "absolute_difference": float(value - reference),
                    "percent_difference_from_baseline": percent_change(value, reference),
                    "inference_note": "Descriptive synthetic-scenario contrast; daily energy preservation varies by scenario.",
                }
            )
    return pd.DataFrame(rows)


def load_proposed_trajectories(scenario: str) -> pd.DataFrame:
    trajectories: list[pd.DataFrame] = []
    for seed in range(1, 6):
        path = TRAJECTORY_DIR / f"{scenario}_{PROPOSED_CONTROLLER}_seed_{seed}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen proposed-controller trajectory: {path}")
        trajectory = pd.read_parquet(path).copy()
        trajectory["timestamp_utc"] = pd.to_datetime(trajectory["timestamp_utc"], utc=True, errors="raise")
        trajectory["seed"] = seed
        trajectories.append(trajectory)
    combined = pd.concat(trajectories, ignore_index=True)
    expected = {"P_import_kW", "P_grid_kW", "P_batt_safe_kW", "Pgrid_ref_kW", "tracking_error_kW", "DR_active"}
    missing = sorted(expected.difference(combined.columns))
    if missing:
        raise ValueError(f"Frozen proposed trajectories lack required fields: {missing}")
    return combined


def dr_coincidence_analysis(
    frames: dict[str, pd.DataFrame], master: pd.DataFrame, timestep_h: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the implemented DR-coincident workload case with baseline on DR intervals."""

    baseline = frames["baseline_real"].copy()
    coincident = frames["dr_coincident_high_load"].copy()
    active = coincident["DR_active"].astype(bool)
    if not active.any():
        raise ValueError("The implemented DR-coincident scenario has no active DR intervals in the test split.")
    baseline_active = baseline.loc[active].copy()
    coincident_active = coincident.loc[active].copy()
    if not baseline_active["DR_active"].astype(bool).all():
        raise ValueError("Baseline DR timing does not align with the DR-coincident scenario.")

    def summarize_trajectory(scenario_name: str) -> pd.DataFrame:
        trajectory = load_proposed_trajectories(scenario_name)
        per_seed_rows: list[dict[str, Any]] = []
        for seed, seed_traj in trajectory.groupby("seed"):
            seed_active = seed_traj.loc[seed_traj["DR_active"].astype(bool)]
            if seed_active.empty:
                raise ValueError(f"Frozen trajectory {scenario_name} seed {seed} has no active DR intervals.")
            battery = pd.to_numeric(seed_active["P_batt_safe_kW"], errors="raise")
            per_seed_rows.append(
                {
                    "seed": int(seed),
                    "mean_grid_import_during_DR_kW": float(seed_active["P_import_kW"].mean()),
                    "mean_grid_power_during_DR_kW": float(seed_active["P_grid_kW"].mean()),
                    "mean_battery_discharge_during_DR_kW": float(battery.clip(lower=0.0).mean()),
                    "mean_battery_power_during_DR_kW": float(battery.mean()),
                    "DR_period_tracking_RMSE_kW": float(np.sqrt(np.mean(np.square(seed_active["tracking_error_kW"])))),
                    "max_grid_import_during_DR_kW": float(seed_active["P_import_kW"].max()),
                }
            )
        return pd.DataFrame(per_seed_rows).drop(columns="seed").agg(["mean", "std"])

    trajectory_summary = summarize_trajectory("dr_coincident_high_load")
    baseline_trajectory_summary = summarize_trajectory("baseline_real")

    def value(metric: str, statistic: str = "mean", baseline_value: bool = False) -> float:
        summary = baseline_trajectory_summary if baseline_value else trajectory_summary
        return float(summary.loc[statistic, metric])

    performance = master.set_index("scenario")
    rows = []
    comparisons = [
        ("mean_IT_load_during_DR_kW", float(coincident_active["P_IT_kW"].mean()), float(baseline_active["P_IT_kW"].mean()), "Frozen scenario datasets"),
        ("mean_facility_load_during_DR_kW", float(coincident_active["P_total_dc_kW"].mean()), float(baseline_active["P_total_dc_kW"].mean()), "Frozen scenario datasets"),
        ("mean_grid_import_during_DR_kW", value("mean_grid_import_during_DR_kW"), value("mean_grid_import_during_DR_kW", baseline_value=True), "Mean across five frozen proposed-controller trajectories"),
        ("mean_battery_discharge_during_DR_kW", value("mean_battery_discharge_during_DR_kW"), value("mean_battery_discharge_during_DR_kW", baseline_value=True), "Mean across five frozen proposed-controller trajectories; positive battery power means discharge"),
        ("DR_period_tracking_RMSE_kW", value("DR_period_tracking_RMSE_kW"), value("DR_period_tracking_RMSE_kW", baseline_value=True), "Calculated from frozen DR-active trajectory rows, then averaged across seeds"),
        ("pooled_test_DR_interval_compliance_pct", float(performance.loc["dr_coincident_high_load", "DR_interval_compliance_pct_pooled_test_mean_across_seeds"]), float(performance.loc["baseline_real", "DR_interval_compliance_pct_pooled_test_mean_across_seeds"]), "Frozen proposed-controller trajectories; recomputed read-only over the pooled test horizon"),
        ("pooled_test_DR_violation_energy_kWh", float(performance.loc["dr_coincident_high_load", "DR_violation_energy_kWh_pooled_test_mean_across_seeds"]), float(performance.loc["baseline_real", "DR_violation_energy_kWh_pooled_test_mean_across_seeds"]), "Frozen proposed-controller trajectories; recomputed read-only over the pooled test horizon"),
        ("authoritative_daily_episode_VPP_tracking_RMSE_kW", float(performance.loc["dr_coincident_high_load", "VPP_tracking_RMSE_kW_mean_across_seeds"]), float(performance.loc["baseline_real", "VPP_tracking_RMSE_kW_mean_across_seeds"]), "Frozen scenario_seed_means.csv"),
    ]
    for metric, scenario_value, baseline_value, source in comparisons:
        rows.append(
            {
                "analysis_type": "IT_load_x_DR_coincidence",
                "comparison": "dr_coincident_high_load versus baseline_real during active DR intervals where applicable",
                "metric_x": metric,
                "metric_y": "baseline_real",
                "n_scenarios": 2,
                "scenario_value": scenario_value,
                "baseline_value": baseline_value,
                "absolute_difference": float(scenario_value - baseline_value) if np.isfinite(baseline_value) else np.nan,
                "percent_difference_from_baseline": percent_change(scenario_value, baseline_value) if np.isfinite(baseline_value) else np.nan,
                "inference_note": source,
            }
        )
    detail = pd.DataFrame(
        {
            "scenario": ["dr_coincident_high_load"],
            "baseline_scenario": ["baseline_real"],
            "DR_active_interval_count": [int(active.sum())],
            "DR_active_duration_hours": [float(active.sum() * timestep_h)],
            "mean_IT_load_during_DR_kW": [float(coincident_active["P_IT_kW"].mean())],
            "mean_baseline_IT_load_during_DR_kW": [float(baseline_active["P_IT_kW"].mean())],
            "mean_facility_load_during_DR_kW": [float(coincident_active["P_total_dc_kW"].mean())],
            "mean_baseline_facility_load_during_DR_kW": [float(baseline_active["P_total_dc_kW"].mean())],
            "mean_grid_import_during_DR_kW": [value("mean_grid_import_during_DR_kW")],
            "std_grid_import_during_DR_kW_across_seeds": [value("mean_grid_import_during_DR_kW", "std")],
            "mean_baseline_grid_import_during_DR_kW": [value("mean_grid_import_during_DR_kW", baseline_value=True)],
            "std_baseline_grid_import_during_DR_kW_across_seeds": [value("mean_grid_import_during_DR_kW", "std", True)],
            "mean_battery_discharge_during_DR_kW": [value("mean_battery_discharge_during_DR_kW")],
            "std_battery_discharge_during_DR_kW_across_seeds": [value("mean_battery_discharge_during_DR_kW", "std")],
            "mean_baseline_battery_discharge_during_DR_kW": [value("mean_battery_discharge_during_DR_kW", baseline_value=True)],
            "std_baseline_battery_discharge_during_DR_kW_across_seeds": [value("mean_battery_discharge_during_DR_kW", "std", True)],
            "DR_period_tracking_RMSE_kW": [value("DR_period_tracking_RMSE_kW")],
            "baseline_DR_period_tracking_RMSE_kW": [value("DR_period_tracking_RMSE_kW", baseline_value=True)],
            "pooled_test_DR_interval_compliance_pct": [float(performance.loc["dr_coincident_high_load", "DR_interval_compliance_pct_pooled_test_mean_across_seeds"])],
            "pooled_test_DR_violation_energy_kWh": [float(performance.loc["dr_coincident_high_load", "DR_violation_energy_kWh_pooled_test_mean_across_seeds"])],
            "authoritative_VPP_tracking_RMSE_kW": [float(performance.loc["dr_coincident_high_load", "VPP_tracking_RMSE_kW_mean_across_seeds"])],
        }
    )
    return pd.DataFrame(rows), detail


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / f"{name}.png", dpi=350, bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def scenario_label(name: str) -> str:
    return name.replace("_", " ").title()


def make_workload_profiles(frames: dict[str, pd.DataFrame]) -> None:
    """Plot all implemented workload profiles across one DR-containing test week."""

    baseline = frames["baseline_real"]
    dr_days = baseline.loc[baseline["DR_active"].astype(bool), "timestamp_utc"].dt.floor("D")
    if dr_days.empty:
        day = baseline["timestamp_utc"].dt.floor("D").iloc[0]
    else:
        day = dr_days.value_counts().idxmax()
    week_start = day - pd.Timedelta(days=int(day.dayofweek))
    week_end = week_start + pd.Timedelta(days=7)
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True, sharey=True)
    for axis, (name, frame) in zip(axes.ravel(), frames.items()):
        week = frame.loc[(frame["timestamp_utc"] >= week_start) & (frame["timestamp_utc"] < week_end)]
        base_week = baseline.loc[(baseline["timestamp_utc"] >= week_start) & (baseline["timestamp_utc"] < week_end)]
        axis.plot(base_week["timestamp_utc"], base_week["P_IT_kW"], color="0.65", linewidth=0.9, label="Baseline")
        axis.plot(week["timestamp_utc"], week["P_IT_kW"], color="#1f5a94", linewidth=1.1, label="Scenario")
        dr = week["DR_active"].astype(bool)
        starts = week.loc[dr & ~dr.shift(fill_value=False), "timestamp_utc"]
        ends = week.loc[dr & ~dr.shift(-1, fill_value=False), "timestamp_utc"] + pd.Timedelta(minutes=15)
        for start, end in zip(starts, ends):
            axis.axvspan(start, end, color="#d95f02", alpha=0.13, linewidth=0)
        axis.set_title(scenario_label(name))
        axis.set_ylabel("IT load (kW)")
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[0, 0].legend(loc="upper right")
    for axis in axes[-1, :]:
        axis.set_xlabel("UTC date")
    fig.suptitle(f"Frozen IT-load scenarios: representative DR-containing test week ({week_start:%d %b %Y})", y=1.01)
    save_figure(fig, "it_workload_profiles")


def make_characteristics_figure(characteristics: pd.DataFrame) -> None:
    labels = [scenario_label(name) for name in characteristics["scenario"]]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    panels = [
        ("mean_it_kW", "Mean IT load (kW)", "#1f5a94"),
        ("max_it_kW", "Peak IT load (kW)", "#d95f02"),
        ("it_coefficient_of_variation", "IT coefficient of variation (-)", "#2a9d8f"),
        ("mean_absolute_it_ramp_kW_per_h", "Mean absolute IT ramp (kW/h)", "#7b2cbf"),
    ]
    for axis, (column, ylabel, color) in zip(axes.ravel(), panels):
        axis.bar(x, characteristics[column], color=color, alpha=0.88)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, labels, rotation=38, ha="right")
    figure.suptitle("IT-load magnitude and temporal characteristics in the frozen test period", y=1.01)
    save_figure(figure, "it_load_characteristics")


def make_peak_rmse_figure(master: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8.3, 6.1))
    colors = plt.cm.tab10(np.linspace(0, 1, len(master)))
    y = master["VPP_tracking_RMSE_kW_mean_across_seeds"]
    yerr = master["VPP_tracking_RMSE_kW_std_across_seeds"]
    for color, (_, row) in zip(colors, master.iterrows()):
        axis.errorbar(
            row["max_it_kW"],
            row["VPP_tracking_RMSE_kW_mean_across_seeds"],
            yerr=row["VPP_tracking_RMSE_kW_std_across_seeds"],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
        axis.annotate(scenario_label(str(row["scenario"])), (row["max_it_kW"], row["VPP_tracking_RMSE_kW_mean_across_seeds"]), xytext=(5, 5), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Peak IT load in test period (kW)")
    axis.set_ylabel("VPP tracking RMSE (kW; mean ± seed SD)")
    axis.set_title("IT peak load and VPP tracking difficulty")
    save_figure(figure, "it_peak_vs_vpp_rmse")


def make_variability_battery_figure(master: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8.3, 6.1))
    colors = plt.cm.tab10(np.linspace(0, 1, len(master)))
    for color, (_, row) in zip(colors, master.iterrows()):
        axis.errorbar(
            row["mean_absolute_it_ramp_kW_per_h"],
            row["battery_throughput_kWh_mean_across_seeds"],
            yerr=row["battery_throughput_kWh_std_across_seeds"],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
        axis.annotate(scenario_label(str(row["scenario"])), (row["mean_absolute_it_ramp_kW_per_h"], row["battery_throughput_kWh_mean_across_seeds"]), xytext=(5, 5), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Mean absolute IT ramp rate (kW/h)")
    axis.set_ylabel("Battery throughput (kWh/day; mean ± seed SD)")
    axis.set_title("IT temporal variability and battery utilization")
    save_figure(figure, "it_variability_vs_battery")


def make_dr_coincidence_figure(frames: dict[str, pd.DataFrame]) -> None:
    """Visualize one implemented DR event using the mean response of five saved trajectories."""

    scenario = frames["dr_coincident_high_load"]
    active = scenario["DR_active"].astype(bool)
    starts = scenario.loc[active & ~active.shift(fill_value=False), "timestamp_utc"]
    if starts.empty:
        return
    event_start = starts.iloc[0]
    window_start = event_start - pd.Timedelta(hours=3)
    window_end = event_start + pd.Timedelta(hours=6)
    load_window = scenario.loc[(scenario["timestamp_utc"] >= window_start) & (scenario["timestamp_utc"] <= window_end)].copy()
    trajectory = load_proposed_trajectories("dr_coincident_high_load")
    numeric = ["P_import_kW", "P_grid_kW", "P_batt_safe_kW", "Pgrid_ref_kW", "DR_import_limit_kW"]
    trajectory_mean = trajectory.groupby("timestamp_utc", as_index=False)[numeric].mean()
    trajectory_window = trajectory_mean.loc[
        (trajectory_mean["timestamp_utc"] >= window_start) & (trajectory_mean["timestamp_utc"] <= window_end)
    ]
    figure, axes = plt.subplots(3, 1, figsize=(11, 8.6), sharex=True)
    axes[0].plot(load_window["timestamp_utc"], load_window["P_IT_kW"], color="#1f5a94", label="IT load")
    axes[0].plot(load_window["timestamp_utc"], load_window["P_total_dc_kW"], color="#2a9d8f", label="Total facility load")
    axes[0].set_ylabel("Load (kW)")
    axes[1].plot(trajectory_window["timestamp_utc"], trajectory_window["P_import_kW"], color="#d95f02", label="Grid import")
    axes[1].plot(trajectory_window["timestamp_utc"], trajectory_window["Pgrid_ref_kW"], color="#264653", linestyle="--", label="VPP reference")
    axes[1].plot(trajectory_window["timestamp_utc"], trajectory_window["DR_import_limit_kW"], color="#e76f51", linestyle=":", label="DR import limit")
    axes[1].set_ylabel("Grid power (kW)")
    axes[1].legend(loc="upper right")
    axes[2].plot(trajectory_window["timestamp_utc"], trajectory_window["P_batt_safe_kW"], color="#7b2cbf", label="Battery power (+ discharge)")
    axes[2].axhline(0.0, color="0.25", linewidth=0.7)
    axes[2].set_ylabel("Battery power (kW)")
    axes[2].set_xlabel("UTC time")
    axes[2].legend(loc="upper right")
    dr = load_window["DR_active"].astype(bool)
    event_starts = load_window.loc[dr & ~dr.shift(fill_value=False), "timestamp_utc"]
    event_ends = load_window.loc[dr & ~dr.shift(-1, fill_value=False), "timestamp_utc"] + pd.Timedelta(minutes=15)
    for axis in axes:
        for start, end in zip(event_starts, event_ends):
            axis.axvspan(start, end, color="#f4a261", alpha=0.22, linewidth=0)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d %b"))
    load_handles, load_labels = axes[0].get_legend_handles_labels()
    load_handles.append(Patch(facecolor="#f4a261", alpha=0.22, label="DR-active period"))
    axes[0].legend(load_handles, load_labels + ["DR-active period"], loc="upper right")
    figure.suptitle("Implemented IT-load/DR coincidence: first active test DR event (mean response across five seeds)", y=1.01)
    save_figure(figure, "it_dr_coincidence")


def make_figures(frames: dict[str, pd.DataFrame], characteristics: pd.DataFrame, master: pd.DataFrame) -> list[str]:
    configure_plotting()
    make_workload_profiles(frames)
    make_characteristics_figure(characteristics)
    make_peak_rmse_figure(master)
    make_variability_battery_figure(master)
    make_dr_coincidence_figure(frames)
    return [
        "figures/it_workload_profiles.png and .pdf",
        "figures/it_load_characteristics.png and .pdf",
        "figures/it_peak_vs_vpp_rmse.png and .pdf",
        "figures/it_variability_vs_battery.png and .pdf",
        "figures/it_dr_coincidence.png and .pdf",
    ]


def format_value(value: float, digits: int = 3) -> str:
    return "not available" if pd.isna(value) else f"{value:,.{digits}f}"


def render_markdown_table(frame: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None) -> str:
    labels = rename or {}
    output = frame[columns].copy()
    output.columns = [labels.get(column, column) for column in columns]
    header = "| " + " | ".join(output.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(output.columns)) + " |"
    rows = []
    for _, row in output.iterrows():
        rendered = []
        for item in row:
            if isinstance(item, (float, np.floating)):
                rendered.append("" if pd.isna(item) else f"{item:.3f}")
            else:
                rendered.append(str(item))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, separator, *rows])


def report_sources(scenario_names: list[str]) -> list[str]:
    sources = [
        FROZEN_DATASET_PATH.relative_to(ROOT).as_posix(),
        DATASET_METADATA_PATH.relative_to(ROOT).as_posix(),
        PUBLICATION_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        PUBLICATION_REPORT_PATH.relative_to(ROOT).as_posix(),
        SCENARIO_SOURCE_PATH.relative_to(ROOT).as_posix(),
        SCENARIO_EVALUATION_SOURCE_PATH.relative_to(ROOT).as_posix(),
        METRICS_SOURCE_PATH.relative_to(ROOT).as_posix(),
        PLOTTING_SOURCE_PATH.relative_to(ROOT).as_posix(),
        CONFIG_PATH.relative_to(ROOT).as_posix(),
        SEED_MEANS_PATH.relative_to(ROOT).as_posix(),
        DAILY_METRICS_PATH.relative_to(ROOT).as_posix(),
        TRAINING_SEED_SUMMARY_PATH.relative_to(ROOT).as_posix(),
    ]
    sources.extend((DATASET_DIR / f"{name}.parquet").relative_to(ROOT).as_posix() for name in scenario_names)
    sources.append(
        f"{TRAJECTORY_DIR.relative_to(ROOT).as_posix()}/{{scenario}}_{PROPOSED_CONTROLLER}_seed_{{1..5}}.parquet "
        "(40 frozen scenario trajectories read; all five seeds for each of eight scenarios)"
    )
    return sources


def write_report(
    characteristics: pd.DataFrame,
    master: pd.DataFrame,
    correlations: pd.DataFrame,
    dr_detail: pd.DataFrame,
    figure_files: list[str],
    scenario_names: list[str],
    timestep_h: float,
) -> None:
    indexed = master.set_index("scenario")
    baseline = indexed.loc["baseline_real"]
    low = indexed.loc["low_it_load"]
    high = indexed.loc["high_it_load"]
    smooth = indexed.loc["smooth_cloud_service"]
    bursty = indexed.loc["ai_training_bursty"]
    shiftable = indexed.loc["workload_shiftable"]
    hot = indexed.loc["hot_day_cooling_stress"]
    dr = indexed.loc["dr_coincident_high_load"]
    correlation_rows = correlations.set_index("comparison")
    source_lines = "\n".join(f"- `{source}`" for source in report_sources(scenario_names))
    scenario_table = render_markdown_table(
        characteristics,
        [
            "scenario", "scenario_family", "test_interval_count", "test_day_count", "time_step_hours",
            "it_load_transformation", "it_load_changed_from_baseline", "cooling_load_changed_from_baseline",
            "pv_generation_changed_from_baseline", "dr_timing_changed_from_baseline", "controller_evaluated",
            "n_seeds", "days_per_seed", "n_daily_metric_samples",
            "facility_forecast_changed_from_baseline", "daily_it_energy_preserved_from_baseline",
        ],
        {
            "scenario": "Scenario",
            "scenario_family": "Family",
            "test_interval_count": "Test intervals",
            "test_day_count": "Test days",
            "time_step_hours": "dt (h)",
            "it_load_transformation": "Implemented IT transformation",
            "it_load_changed_from_baseline": "IT changed?",
            "cooling_load_changed_from_baseline": "Cooling changed?",
            "pv_generation_changed_from_baseline": "PV changed?",
            "dr_timing_changed_from_baseline": "DR timing changed?",
            "controller_evaluated": "Controller",
            "n_seeds": "Seeds",
            "days_per_seed": "Days/seed",
            "n_daily_metric_samples": "Daily samples",
            "facility_forecast_changed_from_baseline": "Facility forecast changed?",
            "daily_it_energy_preserved_from_baseline": "Daily IT energy preserved?",
        },
    )
    master_table = render_markdown_table(
        master,
        [
            "scenario", "mean_it_kW", "max_it_kW", "std_it_kW", "it_coefficient_of_variation",
            "it_peak_to_average_ratio", "mean_absolute_it_ramp_kW_per_h", "max_absolute_it_ramp_kW_per_h",
            "mean_daily_it_energy_kWh", "VPP_tracking_RMSE_kW_mean_across_seeds",
            "DR_interval_compliance_pct_pooled_test_mean_across_seeds",
            "DR_violation_energy_kWh_pooled_test_mean_across_seeds", "daily_cost_EUR_mean_across_seeds",
            "carbon_emissions_kgCO2_mean_across_seeds", "battery_throughput_kWh_mean_across_seeds",
            "safety_intervention_fraction_pct_mean_across_seeds",
        ],
        {
            "scenario": "Scenario",
            "mean_it_kW": "Mean IT (kW)",
            "max_it_kW": "Peak IT (kW)",
            "std_it_kW": "IT SD (kW)",
            "it_coefficient_of_variation": "IT CV (-)",
            "it_peak_to_average_ratio": "IT PAR (-)",
            "mean_absolute_it_ramp_kW_per_h": "Mean IT ramp (kW/h)",
            "max_absolute_it_ramp_kW_per_h": "Max IT ramp (kW/h)",
            "mean_daily_it_energy_kWh": "Daily IT energy (kWh)",
            "VPP_tracking_RMSE_kW_mean_across_seeds": "VPP RMSE (kW)",
            "DR_interval_compliance_pct_pooled_test_mean_across_seeds": "Pooled-test DR compliance (%)",
            "DR_violation_energy_kWh_pooled_test_mean_across_seeds": "Pooled-test DR violation energy (kWh)",
            "daily_cost_EUR_mean_across_seeds": "Cost/day (EUR)",
            "carbon_emissions_kgCO2_mean_across_seeds": "Carbon/day (kgCO2)",
            "battery_throughput_kWh_mean_across_seeds": "Battery throughput (kWh/day)",
            "safety_intervention_fraction_pct_mean_across_seeds": "Safety intervention (%)",
        },
    )
    dr_row = dr_detail.iloc[0]
    document = f"""# IT-Load Characterization & VPP Flexibility Analysis

## Scope and safeguards

This is a strictly post-processing analysis of the frozen Phase 2 publication artifacts. It does not build or alter a scenario, modify the dataset, evaluate or retrain a controller, change a configuration, or overwrite an authoritative result. All newly derived files are confined to `results/it_load_analysis/`.

The analysis uses the test split of each frozen scenario: **{int(characteristics['test_interval_count'].iloc[0]):,} 15-minute intervals**, **{int(characteristics['test_day_count'].iloc[0])} UTC days**, from **{characteristics['test_start_utc'].iloc[0]}** through **{characteristics['test_end_utc'].iloc[0]}**. The inferred timestep is **{timestep_h:.2f} h**, so energy is calculated as $\\sum_t P_t \\Delta t$ and ramp rate as $|P_t-P_{{t-1}}|/\\Delta t$ in kW/h.

Daily cost, carbon, daily-episode tracking RMSE, battery throughput, terminal-SOC, peak-import, and safety values are the existing frozen daily-episode means, averaged across five saved seeds. Thus there are **460 daily metric samples per scenario** (92 days × 5 seeds), although the five seeds—not the repeated daily rows—are the relevant independently initialized controller realizations. In contrast, DR compliance, event compliance, maximum DR violation, and DR violation energy in this new analysis are recomputed read-only over each complete saved 92-day test trajectory before averaging across seeds. This avoids treating no-DR daily episodes (which the frozen episode metric scores as 100%) as DR performance.

## Authoritative source artifacts used

{source_lines}

The frozen source-dataset SHA-256 is checked against the local Parquet source, and every frozen scenario dataset, scenario-result CSV, and proposed-controller trajectory read by this script is checked against `PUBLICATION_MANIFEST.json` before analysis. The scenario definitions and transformation wording are taken from the implemented scenario generator, not recreated here.

## Eight implemented IT-load scenarios

{scenario_table}

The fields above establish that PV, ambient temperature, price, carbon proxy, grid reference, and DR timing/limits are unchanged across the frozen scenario datasets; `Pdc_forecast_kW` is recomputed from each scenario's facility-load trajectory, so it changes when IT or cooling changes. **hot_day_cooling_stress** changes cooling without changing IT load, whereas all other differences follow the implemented scenario generator. The workload transformations are synthetic sensitivity cases except the retained baseline profile.

## IT-load and facility-load characterization

`IT_LOAD_SCENARIO_CHARACTERISTICS.csv` contains unrounded descriptive statistics for IT, cooling, auxiliary, total facility, PV generation, signed net facility demand, and nonnegative grid import without battery. For every available quantity it reports mean, median, standard deviation, minimum, maximum, P95, mean daily energy, peak-to-average ratio, coefficient of variation, and mean/max/P95 absolute ramp rate. The master performance table is reproduced below for direct use in a manuscript review.

{master_table}

Definitions: PAR is maximum divided by mean; CV is sample standard deviation divided by mean. PV generation is reported as a generation magnitude, whereas net facility load is signed after PV and can be negative during export. Safety intervention is physical-projection rate plus service-coaching rate, following the existing frozen metric fields.

## IT-load magnitude: low, baseline, and high cases

Under the simulated scenarios, low IT load has mean IT power **{format_value(low['mean_it_kW'])} kW** and daily IT energy **{format_value(low['mean_daily_it_energy_kWh'])} kWh**, compared with baseline **{format_value(baseline['mean_it_kW'])} kW** and **{format_value(baseline['mean_daily_it_energy_kWh'])} kWh**. High IT load has **{format_value(high['mean_it_kW'])} kW** and **{format_value(high['mean_daily_it_energy_kWh'])} kWh**. Relative to baseline, the frozen proposed-controller results correspond to VPP RMSE changes of **{percent_change(float(low['VPP_tracking_RMSE_kW_mean_across_seeds']), float(baseline['VPP_tracking_RMSE_kW_mean_across_seeds'])):.2f}%** for low load and **{percent_change(float(high['VPP_tracking_RMSE_kW_mean_across_seeds']), float(baseline['VPP_tracking_RMSE_kW_mean_across_seeds'])):.2f}%** for high load. Cost, carbon, battery throughput, and intervention-rate contrasts are available without rounding in `IT_LOAD_ANALYSIS_SUMMARY.csv`.

These contrasts are **associations under the simulated scenarios**, not causal estimates. Low/high scaling also changes cooling and total facility load because the implemented model recalculates them from IT load.

## IT-load temporal structure

The energy-preserving smooth and shiftable cases have the same daily IT energy as baseline to numerical precision, but their shapes differ. Smooth cloud service has mean absolute IT ramp **{format_value(smooth['mean_absolute_it_ramp_kW_per_h'])} kW/h** and VPP RMSE **{format_value(smooth['VPP_tracking_RMSE_kW_mean_across_seeds'])} kW**; the bursty AI-training case has **{format_value(bursty['mean_absolute_it_ramp_kW_per_h'])} kW/h** and **{format_value(bursty['VPP_tracking_RMSE_kW_mean_across_seeds'])} kW**; and the shiftable case has **{format_value(shiftable['mean_absolute_it_ramp_kW_per_h'])} kW/h** and **{format_value(shiftable['VPP_tracking_RMSE_kW_mean_across_seeds'])} kW**. This demonstrates that scenarios with similar or preserved daily IT energy can correspond to different peaks, ramping, tracking error, battery throughput, and intervention rates. It does not establish which workload property causes those outcomes.

The descriptive correlation between mean IT ramp and VPP RMSE across all eight scenario points is **r = {format_value(correlation_rows.loc['mean IT ramp vs VPP tracking RMSE', 'pearson_r'])}**; the correlation between mean IT ramp and battery throughput is **r = {format_value(correlation_rows.loc['mean IT ramp vs battery throughput', 'pearson_r'])}**. Both are exploratory only because the eight scenario families are deterministic and non-independent.

## IT load × DR coincidence

The implemented `dr_coincident_high_load` case contains **{int(dr_row['DR_active_interval_count'])} active DR intervals** ({format_value(dr_row['DR_active_duration_hours'])} h) in the test period. During those active intervals, IT load is **{format_value(dr_row['mean_IT_load_during_DR_kW'])} kW**, compared with **{format_value(dr_row['mean_baseline_IT_load_during_DR_kW'])} kW** in the corresponding baseline intervals; total facility load is **{format_value(dr_row['mean_facility_load_during_DR_kW'])} kW**, compared with **{format_value(dr_row['mean_baseline_facility_load_during_DR_kW'])} kW**. The five-seed proposed-controller mean grid import during DR is **{format_value(dr_row['mean_grid_import_during_DR_kW'])} kW** versus **{format_value(dr_row['mean_baseline_grid_import_during_DR_kW'])} kW** for baseline; mean battery discharge is **{format_value(dr_row['mean_battery_discharge_during_DR_kW'])} kW** versus **{format_value(dr_row['mean_baseline_battery_discharge_during_DR_kW'])} kW** for baseline. The DR-period tracking RMSE is **{format_value(dr_row['DR_period_tracking_RMSE_kW'])} kW** versus **{format_value(dr_row['baseline_DR_period_tracking_RMSE_kW'])} kW**. Read-only pooled-test DR interval compliance is **{format_value(dr_row['pooled_test_DR_interval_compliance_pct'])}%**, pooled-test DR violation energy is **{format_value(dr_row['pooled_test_DR_violation_energy_kWh'])} kWh**, and existing daily-episode VPP tracking RMSE is **{format_value(dr_row['authoritative_VPP_tracking_RMSE_kW'])} kW**.

The scenario definition itself makes IT peaks coincide with DR events and ±1-hour shoulders, so the analysis confirms an implemented adverse coincidence rather than discovering a naturally occurring relationship.

## Exploratory descriptive relationships

{render_markdown_table(correlations, ['comparison', 'n_scenarios', 'pearson_r'], {'comparison': 'Relationship', 'n_scenarios': 'n scenarios', 'pearson_r': 'Pearson r'})}

No p-values, confidence intervals, or significance claims are reported: eight structured synthetic scenario families do not form an inferential sample. The table is only a compact description of the eight frozen points.

## Interpretation limitations and confounding factors

- The scenarios are not independent observations; seven of the eight are deterministic transformations of the same underlying 2019 profile.
- Low/high, bursty, smooth, shiftable, and DR-coincident IT changes recompute cooling and auxiliary load from IT, so IT-only effects cannot be isolated from resulting total-facility-load changes.
- `hot_day_cooling_stress` changes cooling without changing IT, which is useful as a thermal confounder check but is not an IT-load perturbation.
- PV, price, carbon proxy, grid reference, and DR signals remain exogenous; PV and DR timing/limits are unchanged between frozen scenario datasets. `Pdc_forecast_kW` is recalculated from each altered facility-load trajectory. The fixed baseline VPP reference is therefore a material tracking-RMSE confounder. The DR-coincident case changes the IT demand timing relative to those fixed signals.
- Smooth and shiftable scenarios preserve daily IT energy; low/high, bursty, and DR-coincident scenarios do not necessarily do so. Workload shiftability also embeds price/carbon/DR preference, so it is not a pure temporal-variability intervention.
- Only the proposed Residual Safe-SAC controller is summarized in the requested main table; this does not prove controller-general workload effects.
- Existing frozen daily DR summaries score non-DR daily episodes as 100%; this analysis leaves them untouched and reports separate pooled-test DR metrics reconstructed from the saved trajectories.
- The operational model is hybrid/modeled and the synthetic scenarios are sensitivity cases, not independent field deployments or probabilistic forecasts.

## Final scientific interpretation

### A. Main findings

IT-load magnitude and temporal structure are both associated with different total facility demand, VPP tracking, DR outcomes, cost, carbon, battery throughput, and safety intervention in the frozen sensitivity scenarios. The tables and five new figures give the quantitative evidence without changing the underlying experiment.

### B. Evidence that IT-load magnitude matters

The exact 0.75× and 1.25× load-level scenarios shift mean and peak IT load, daily IT energy, cooling/total-facility demand, and the proposed-controller performance summaries. This supports a conservative statement that workload magnitude is an important operating condition for the modeled VPP.

### C. Evidence that IT-load temporal structure matters

Energy-preserving smooth and shiftable transformations, plus the scheduled bursty case, show different IT ramps and peaks alongside different tracking and battery-utilization values. This supports describing temporal structure as a relevant sensitivity dimension, not as an isolated causal driver.

### D. Evidence concerning IT-load/DR coincidence

The DR-coincident scenario deliberately raises IT load around active DR periods. Its observed DR-period IT/facility loads, grid import, battery response, compliance, violation energy, and tracking error are quantified above. It is an adversarial synthetic stress test, not evidence of a measured real-world coincidence rate.

### E. Most strongly associated characteristic with VPP difficulty

Within these eight points, no single characteristic can be identified robustly as *the* dominant driver. Peak IT load and mean IT ramp are useful descriptive indicators, but they co-vary with total facility load, cooling, and scenario construction. The safest conclusion is that **combined load magnitude and timing relative to fixed DR/reference signals** are associated with tracking difficulty.

### F. Safe manuscript claim

It is safe to state: *Under eight transparent, frozen IT-load sensitivity scenarios, the modeled green data-center VPP exhibited workload-dependent differences in tracking, DR, cost, carbon, battery throughput, and safety-intervention metrics. Both load magnitude and temporal structure were examined descriptively.*

### G. Claims that are not safe

Do not claim causal IT-workload effects, statistical generalization, field robustness, an observed AI-training population, an estimated probability of DR coincidence, or superiority of one IT characteristic as a universal predictor.

### H. Remaining limitations

The deterministic scenario construction, shared baseline profile, hybrid facility model, exogenous market/DR signals, limited scenario count, and single-controller emphasis constrain inference.

### I. Recommended manuscript figures

{chr(10).join(f'- `{item}`' for item in figure_files)}

### J. Recommended manuscript table

Use `IT_LOAD_VPP_PERFORMANCE.csv` as the primary manuscript table because it combines exact IT-load characteristics with frozen proposed-controller performance. Use `IT_LOAD_SCENARIO_CHARACTERISTICS.csv` as the full supplementary technical table.

### K. Is this sufficient to make IT load the primary paper contribution?

**No.** The analysis materially strengthens workload-aware sensitivity reporting, but the evidence remains a small set of deterministic synthetic transformations of one base profile. It supports IT load as an important experimental dimension and secondary contribution, not as the sole primary research contribution without additional independently sourced workload datasets, calibrated workload models, or broader controller/field validation.
"""
    (OUTPUT / "IT_LOAD_ANALYSIS_REPORT.md").write_text(document, encoding="utf-8")


def check_frozen_dataset_hash() -> dict[str, Any]:
    manifest = json.loads(PUBLICATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest.get("dataset_sha256")
    actual = sha256(FROZEN_DATASET_PATH)
    if expected and actual != expected:
        raise ValueError(
            f"Frozen publication dataset hash mismatch: manifest={expected}, local={actual}. Analysis halted."
        )
    return {"expected_dataset_sha256": expected, "actual_dataset_sha256": actual}


def check_manifest_artifact_hashes(paths: list[Path]) -> int:
    """Validate every frozen data/result artifact read by this analysis against its manifest hash."""

    manifest = json.loads(PUBLICATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = {
        str(item["path"]).replace("/", "\\").lower(): str(item["sha256"]).lower()
        for item in manifest.get("artifacts", [])
        if "path" in item and "sha256" in item
    }
    checked = 0
    for path in paths:
        relative = str(path.relative_to(ROOT)).replace("/", "\\").lower()
        expected = inventory.get(relative)
        if expected is None:
            raise ValueError(f"Publication manifest has no hash for frozen artifact: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"Frozen artifact hash mismatch for {path}: manifest={expected}, local={actual}. Analysis halted."
            )
        checked += 1
    return checked


def write_outputs() -> dict[str, Any]:
    required_paths = [
        PUBLICATION_MANIFEST_PATH,
        PUBLICATION_REPORT_PATH,
        FROZEN_DATASET_PATH,
        DATASET_METADATA_PATH,
        SEED_MEANS_PATH,
        DAILY_METRICS_PATH,
        TRAINING_SEED_SUMMARY_PATH,
        METRICS_SOURCE_PATH,
        SCENARIO_SOURCE_PATH,
        SCENARIO_EVALUATION_SOURCE_PATH,
        PLOTTING_SOURCE_PATH,
        CONFIG_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required frozen source artifacts are missing:\n" + "\n".join(missing))
    hash_info = check_frozen_dataset_hash()
    scenario_names, definitions = load_scenario_definitions()
    expected_names = {
        "baseline_real", "low_it_load", "high_it_load", "ai_training_bursty", "smooth_cloud_service",
        "workload_shiftable", "hot_day_cooling_stress", "dr_coincident_high_load",
    }
    if set(scenario_names) != expected_names or len(scenario_names) != 8:
        raise ValueError(f"Implemented scenario set is not the expected frozen eight: {scenario_names}")
    frozen_artifacts = [SEED_MEANS_PATH, DAILY_METRICS_PATH, TRAINING_SEED_SUMMARY_PATH]
    frozen_artifacts.extend(DATASET_DIR / f"{name}.parquet" for name in scenario_names)
    frozen_artifacts.extend(
        TRAJECTORY_DIR / f"{name}_{PROPOSED_CONTROLLER}_seed_{seed}.parquet"
        for name in scenario_names
        for seed in range(1, 6)
    )
    manifest_hashes_checked = check_manifest_artifact_hashes(frozen_artifacts)
    frames, timestep_h = load_test_scenarios(scenario_names)
    characteristics = build_characteristics(frames, definitions, timestep_h)
    performance, master = build_performance_table(characteristics)
    characteristics = characteristics.merge(
        performance[["scenario", "n_seeds", "days_per_seed", "n_daily_metric_samples"]],
        on="scenario", how="inner", validate="one_to_one"
    )
    correlations = compute_correlations(master)
    contrasts = comparison_rows(master)
    dr_rows, dr_detail = dr_coincidence_analysis(frames, master, timestep_h)
    summary = pd.concat([correlations, contrasts, dr_rows], ignore_index=True, sort=False)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    characteristics.to_csv(OUTPUT / "IT_LOAD_SCENARIO_CHARACTERISTICS.csv", index=False, float_format="%.15g")
    master.to_csv(OUTPUT / "IT_LOAD_VPP_PERFORMANCE.csv", index=False, float_format="%.15g")
    summary.to_csv(OUTPUT / "IT_LOAD_ANALYSIS_SUMMARY.csv", index=False, float_format="%.15g")
    figure_files = make_figures(frames, characteristics, master)
    write_report(characteristics, master, correlations, dr_detail, figure_files, scenario_names, timestep_h)
    return {
        "scenario_count": len(scenario_names),
        "scenario_test_interval_count": int(characteristics["test_interval_count"].sum()),
        "proposed_daily_metric_samples": int(master["n_daily_metric_samples"].sum()),
        "time_step_hours": timestep_h,
        "manifest_artifacts_hash_checked": manifest_hashes_checked,
        **hash_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate frozen sources without writing analysis outputs.")
    args = parser.parse_args()
    if args.check_only:
        info = check_frozen_dataset_hash()
        names, _ = load_scenario_definitions()
        frames, timestep = load_test_scenarios(names)
        print(json.dumps({**info, "scenario_count": len(frames), "time_step_hours": timestep}, indent=2))
        return
    result = write_outputs()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
