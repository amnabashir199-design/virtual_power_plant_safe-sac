from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "timestamp_utc", "P_IT_kW", "ai_utilization", "P_cooling_kW", "P_other_kW",
    "P_total_dc_kW", "PUE_model", "cooling_ratio_model", "Ppv_kW",
    "price_buy_EUR_per_kWh", "carbon_gCO2_per_kWh_proxy", "ambient_temp_C",
    "DR_active",
}
OTHER_LOAD_FRACTION = 0.05
LOW_SCALE = 0.75
HIGH_SCALE = 1.25
BURST_SCALE = 1.45
DR_SCALE = 1.25
SHIFTABLE_FRACTION = 0.20
HOT_COOLING_MAX_INCREASE = 0.25


@dataclass(frozen=True)
class ScenarioDefinition:
    description: str
    family: str
    cooling_note: str


DEFINITIONS = {
    "baseline_real": ScenarioDefinition(
        "Original real/hybrid 2019 profile; no IT-load modification.", "baseline",
        "Original modeled cooling ratio retained: P_cooling = P_IT * cooling_ratio_model.",
    ),
    "low_it_load": ScenarioDefinition(
        "Synthetic sensitivity case with IT load scaled uniformly to 75% of baseline.", "load_level",
        "Cooling recalculated from scaled IT load and the original time-varying cooling ratio.",
    ),
    "high_it_load": ScenarioDefinition(
        "Synthetic sensitivity case with IT load scaled uniformly to 125% of baseline.", "load_level",
        "Cooling recalculated from scaled IT load and the original time-varying cooling ratio.",
    ),
    "ai_training_bursty": ScenarioDefinition(
        "Synthetic AI-training blocks on Tuesday, Thursday, and Saturday from 08:00 to 16:00 UTC.",
        "workload_shape",
        "Cooling follows the original time-varying ratio after burst-related IT-load modification.",
    ),
    "smooth_cloud_service": ScenarioDefinition(
        "Synthetic stable cloud-service profile formed by a centered four-hour rolling mean with daily energy preservation.",
        "workload_shape",
        "Cooling follows the original time-varying ratio after smoothing and daily energy normalization.",
    ),
    "workload_shiftable": ScenarioDefinition(
        "Synthetic flexible-batch case shifting 20% of daily IT energy toward lower-price, lower-carbon, non-DR intervals.",
        "workload_flexibility",
        "Cooling follows the original time-varying ratio; IT energy is preserved separately for every UTC day.",
    ),
    "hot_day_cooling_stress": ScenarioDefinition(
        "Synthetic thermal-stress case retaining IT load while increasing cooling sensitivity in the hottest ambient-temperature decile.",
        "thermal_stress",
        "Cooling ratio increases progressively by up to 25% within the hottest ambient-temperature decile.",
    ),
    "dr_coincident_high_load": ScenarioDefinition(
        "Synthetic difficult case increasing IT load by 25% during DR events and within one-hour event shoulders.",
        "grid_stress",
        "Cooling follows the original time-varying ratio after DR-coincident IT-load modification.",
    ),
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    if df["timestamp_utc"].isna().any():
        raise ValueError(f"Found {int(df['timestamp_utc'].isna().sum())} invalid timestamps.")
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)
    numeric = REQUIRED_COLUMNS - {"timestamp_utc"}
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[list(numeric)].isna().any().any():
        bad = df[list(numeric)].isna().sum()
        raise ValueError(f"Required columns contain missing values: {bad[bad > 0].to_dict()}")
    if (df[["P_IT_kW", "P_cooling_kW", "P_other_kW", "P_total_dc_kW"]] < 0).any().any():
        raise ValueError("The source dataset contains negative data-center loads.")
    return df


def preserve_daily_energy(values: pd.Series, baseline: pd.Series, timestamps: pd.Series) -> pd.Series:
    day = timestamps.dt.floor("D")
    target = baseline.groupby(day).transform("sum")
    current = values.groupby(day).transform("sum")
    return values * target / current.replace(0, np.nan)


def smooth_cloud_profile(df: pd.DataFrame) -> pd.Series:
    smoothed = df["P_IT_kW"].rolling(16, center=True, min_periods=1).mean()
    return preserve_daily_energy(smoothed, df["P_IT_kW"], df["timestamp_utc"])


def shiftable_profile(df: pd.DataFrame) -> pd.Series:
    day = df["timestamp_utc"].dt.floor("D")
    flexible_energy = df["P_IT_kW"] * SHIFTABLE_FRACTION
    inflexible = df["P_IT_kW"] * (1.0 - SHIFTABLE_FRACTION)

    def daily_preference(series: pd.Series) -> pd.Series:
        spread = series.groupby(day).transform("max") - series.groupby(day).transform("min")
        return (series - series.groupby(day).transform("min")) / spread.replace(0, 1.0)

    price_score = daily_preference(df["price_buy_EUR_per_kWh"])
    carbon_score = daily_preference(df["carbon_gCO2_per_kWh_proxy"])
    penalty = 0.5 * price_score + 0.5 * carbon_score + 2.0 * (df["DR_active"] > 0).astype(float)
    preference = np.exp(-2.0 * penalty)
    daily_flexible_total = flexible_energy.groupby(day).transform("sum")
    daily_weight_total = preference.groupby(day).transform("sum")
    redistributed = preference * daily_flexible_total / daily_weight_total.replace(0, np.nan)
    shifted = inflexible + redistributed
    return preserve_daily_energy(shifted, df["P_IT_kW"], df["timestamp_utc"])


def scenario_it_load(df: pd.DataFrame, name: str) -> tuple[pd.Series, pd.Series]:
    baseline = df["P_IT_kW"]
    cooling_ratio = df["cooling_ratio_model"].copy()
    if name == "baseline_real" or name == "hot_day_cooling_stress":
        it_load = baseline.copy()
    elif name == "low_it_load":
        it_load = baseline * LOW_SCALE
    elif name == "high_it_load":
        it_load = baseline * HIGH_SCALE
    elif name == "ai_training_bursty":
        timestamp = df["timestamp_utc"]
        burst = timestamp.dt.dayofweek.isin([1, 3, 5]) & timestamp.dt.hour.between(8, 15)
        it_load = baseline * np.where(burst, BURST_SCALE, 1.0)
    elif name == "smooth_cloud_service":
        it_load = smooth_cloud_profile(df)
    elif name == "workload_shiftable":
        it_load = shiftable_profile(df)
    elif name == "dr_coincident_high_load":
        active = (df["DR_active"] > 0).astype(int)
        shoulder = active.rolling(9, center=True, min_periods=1).max().astype(bool)
        it_load = baseline * np.where(shoulder, DR_SCALE, 1.0)
    else:
        raise KeyError(f"Unknown scenario: {name}")

    if name == "hot_day_cooling_stress":
        hot_threshold = float(df["ambient_temp_C"].quantile(0.90))
        hot_max = float(df["ambient_temp_C"].max())
        intensity = ((df["ambient_temp_C"] - hot_threshold) / max(hot_max - hot_threshold, 1e-9)).clip(0, 1)
        cooling_ratio = cooling_ratio * (1.0 + HOT_COOLING_MAX_INCREASE * intensity)
    return pd.Series(it_load, index=df.index).clip(lower=0), cooling_ratio.clip(lower=0)


def recompute_scenario(source: pd.DataFrame, name: str) -> pd.DataFrame:
    definition = DEFINITIONS[name]
    scenario = source.copy()
    original_it = source["P_IT_kW"].copy()
    original_total = source["P_total_dc_kW"].copy()
    it_load, cooling_ratio = scenario_it_load(source, name)
    scenario["P_IT_kW"] = it_load
    utilization_scale = it_load / original_it.replace(0, np.nan)
    scenario["ai_utilization"] = (source["ai_utilization"] * utilization_scale).clip(0, 1).fillna(0)
    scenario["cooling_ratio_model"] = cooling_ratio
    scenario["P_cooling_kW"] = scenario["P_IT_kW"] * cooling_ratio
    scenario["P_other_kW"] = scenario["P_IT_kW"] * OTHER_LOAD_FRACTION
    scenario["P_total_dc_kW"] = scenario["P_IT_kW"] + scenario["P_cooling_kW"] + scenario["P_other_kW"]
    scenario["PUE_model"] = scenario["P_total_dc_kW"] / scenario["P_IT_kW"].replace(0, np.nan)
    scenario["P_net_without_battery_kW"] = scenario["P_total_dc_kW"] - scenario["Ppv_kW"]
    scenario["P_import_without_battery_kW"] = scenario["P_net_without_battery_kW"].clip(lower=0)
    scenario["P_export_without_battery_kW"] = (-scenario["P_net_without_battery_kW"]).clip(lower=0)
    if "Pdc_forecast_kW" in scenario.columns:
        train = scenario["split"].astype(str).str.lower().eq("train") if "split" in scenario else pd.Series(True, index=scenario.index)
        fallback = float(scenario.loc[train, "P_total_dc_kW"].median())
        scenario["Pdc_forecast_kW"] = (
            scenario["P_total_dc_kW"].shift(1).rolling(4, min_periods=1).mean().fillna(fallback).clip(lower=0)
        )
    scenario["scenario_name"] = name
    scenario["scenario_description"] = definition.description
    scenario["it_load_scaling_factor"] = scenario["P_IT_kW"] / original_it.replace(0, np.nan)
    scenario["cooling_model_note"] = definition.cooling_note
    scenario["scenario_family"] = definition.family
    if scenario[["P_IT_kW", "P_cooling_kW", "P_other_kW", "P_total_dc_kW"]].isna().any().any():
        raise ValueError(f"Scenario {name} contains missing recomputed load values.")
    if (scenario[["P_IT_kW", "P_cooling_kW", "P_other_kW", "P_total_dc_kW"]] < 0).any().any():
        raise ValueError(f"Scenario {name} contains negative load values.")
    return scenario


def summarize(source: pd.DataFrame, scenario: pd.DataFrame, name: str) -> dict[str, float | str | int]:
    day = scenario["timestamp_utc"].dt.floor("D")
    daily_energy = (scenario["P_IT_kW"] * 0.25).groupby(day).sum()
    daily_peak = scenario["P_IT_kW"].groupby(day).max()
    ramp = scenario["P_IT_kW"].diff()
    baseline_daily = (source["P_IT_kW"] * 0.25).groupby(source["timestamp_utc"].dt.floor("D")).sum()
    return {
        "scenario_name": name,
        "scenario_family": DEFINITIONS[name].family,
        "scenario_description": DEFINITIONS[name].description,
        "interval_count": len(scenario),
        "mean_it_load_kW": scenario["P_IT_kW"].mean(),
        "min_it_load_kW": scenario["P_IT_kW"].min(),
        "max_it_load_kW": scenario["P_IT_kW"].max(),
        "mean_it_scaling_factor": (scenario["P_IT_kW"] / source["P_IT_kW"]).mean(),
        "annual_it_energy_MWh": scenario["P_IT_kW"].sum() * 0.25 / 1000,
        "mean_daily_it_energy_kWh": daily_energy.mean(),
        "mean_daily_peak_it_kW": daily_peak.mean(),
        "maximum_daily_peak_it_kW": daily_peak.max(),
        "mean_absolute_ramp_kW_per_15min": ramp.abs().mean(),
        "maximum_absolute_ramp_kW_per_15min": ramp.abs().max(),
        "mean_cooling_load_kW": scenario["P_cooling_kW"].mean(),
        "mean_total_dc_load_kW": scenario["P_total_dc_kW"].mean(),
        "mean_PUE": scenario["PUE_model"].mean(),
        "maximum_daily_energy_difference_from_baseline_pct":
            ((daily_energy - baseline_daily).abs() / baseline_daily.replace(0, np.nan) * 100).max(),
        "cooling_model_note": DEFINITIONS[name].cooling_note,
        "synthetic_variation": name != "baseline_real",
    }


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_comparison_figures(scenarios: dict[str, pd.DataFrame], figure_dir: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    baseline = scenarios["baseline_real"]
    daily_energy = (baseline["P_IT_kW"] * 0.25).groupby(baseline["timestamp_utc"].dt.floor("D")).sum()
    representative_day = (daily_energy - daily_energy.median()).abs().idxmin()
    week_start = representative_day - pd.Timedelta(days=int(representative_day.dayofweek))
    week_end = week_start + pd.Timedelta(days=7)
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True, sharey=True)
    for ax, (name, frame) in zip(axes.flat, scenarios.items()):
        week = frame[(frame["timestamp_utc"] >= week_start) & (frame["timestamp_utc"] < week_end)]
        base_week = baseline[(baseline["timestamp_utc"] >= week_start) & (baseline["timestamp_utc"] < week_end)]
        ax.plot(base_week["timestamp_utc"], base_week["P_IT_kW"], color="0.65", linewidth=0.8, label="Baseline")
        ax.plot(week["timestamp_utc"], week["P_IT_kW"], color="#24557a", linewidth=1.0, label="Scenario")
        ax.set_title(name.replace("_", " ").title())
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_ylabel("IT load (kW)")
    axes[-1, 0].set_xlabel("Date (UTC)")
    axes[-1, 1].set_xlabel("Date (UTC)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"IT-Load Scenario Comparison: Representative Week of {week_start:%Y-%m-%d}", y=1.0)
    save_figure(fig, figure_dir / "it_load_scenario_comparison")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    for color, (name, frame) in zip(colors, scenarios.items()):
        day = frame["timestamp_utc"].dt.floor("D")
        energy = (frame["P_IT_kW"] * 0.25).groupby(day).sum()
        peak = frame["P_IT_kW"].groupby(day).max()
        ax.scatter(energy, peak, s=10, alpha=0.22, color=color)
        ax.scatter(energy.mean(), peak.mean(), s=75, marker="X", color=color,
                   edgecolor="black", linewidth=0.5, label=name.replace("_", " "))
    ax.set(xlabel="Daily IT energy (kWh)", ylabel="Daily peak IT load (kW)",
           title="Daily Energy and Peak Comparison across IT-Load Scenarios")
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, figure_dir / "scenario_daily_energy_peak_comparison")


def build_scenarios(data_path: Path, scenario_dir: Path, table_dir: Path,
                    figure_dir: Path) -> tuple[list[Path], Path, list[Path]]:
    source = load_dataset(data_path)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    scenarios: dict[str, pd.DataFrame] = {}
    summaries = []
    paths = []
    for name in DEFINITIONS:
        scenario = recompute_scenario(source, name)
        path = scenario_dir / f"green_dc_vpp_2019_15min_{name}.csv"
        scenario.to_csv(path, index=False, float_format="%.6f", date_format="%Y-%m-%d %H:%M:%S%z")
        scenarios[name] = scenario
        summaries.append(summarize(source, scenario, name))
        paths.append(path)
        print(f"Saved {name}: {path}")
    summary_path = table_dir / "it_load_scenario_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    make_comparison_figures(scenarios, figure_dir)
    figures = [
        figure_dir / "it_load_scenario_comparison.png", figure_dir / "it_load_scenario_comparison.pdf",
        figure_dir / "scenario_daily_energy_peak_comparison.png",
        figure_dir / "scenario_daily_energy_peak_comparison.pdf",
    ]
    return paths, summary_path, figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explicit IT-load scenarios for thesis sensitivity analysis.")
    parser.add_argument("--data", default="data/green_dc_vpp_2019_15min.csv")
    parser.add_argument("--scenario-dir", default="data/scenarios")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()
    paths, summary, figures = build_scenarios(
        resolve_path(args.data), resolve_path(args.scenario_dir), resolve_path(args.tables_dir),
        resolve_path(args.figures_dir),
    )
    print(f"Created {len(paths)} scenario files")
    print(f"Created summary table: {summary}")
    print(f"Created {len(figures)} comparison figure files")


if __name__ == "__main__":
    main()
