"""Regenerate publication-style figures from packaged tables and trajectories."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "reproduced" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "green": "#176B5B",
    "orange": "#C76825",
    "blue": "#225F86",
    "gray": "#707785",
    "light": "#F1F6F4",
}
SCENARIOS = [
    "baseline_real", "low_it_load", "high_it_load", "ai_training_bursty",
    "smooth_cloud_service", "workload_shiftable", "hot_day_cooling_stress",
    "dr_coincident_high_load",
]
SCENARIO_LABELS = {
    "baseline_real": "Baseline", "low_it_load": "Low IT load",
    "high_it_load": "High IT load", "ai_training_bursty": "Bursty workload",
    "smooth_cloud_service": "Smooth workload", "workload_shiftable": "Shiftable workload",
    "hot_day_cooling_stress": "Hot-day cooling stress",
    "dr_coincident_high_load": "DR-coincident high load",
}
CONTROLLERS = ["direct_safesac", "residual_sac", "proposed_residual_safesac"]
CONTROLLER_LABELS = {
    "no_battery": "No battery", "rule_based": "Rule-based",
    "tou_self_consumption": "TOU self-consumption", "carbon_aware": "Carbon-aware",
    "greedy_tracking": "Greedy tracking", "direct_safesac": "Direct Safe-SAC",
    "residual_sac": "Residual SAC", "proposed_residual_safesac": "Proposed Residual Safe-SAC",
}


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)


def save(fig, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=320 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture_figures():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    boxes = [
        (0.03, 0.55, "IT workload\n+ cooling + auxiliaries", COLORS["blue"]),
        (0.31, 0.72, "1,000 kWp PV", COLORS["green"]),
        (0.31, 0.37, "4 MWh / 750 kW BESS", COLORS["orange"]),
        (0.62, 0.55, "Data-center VPP\npower balance", COLORS["green"]),
        (0.83, 0.55, "Utility grid\nVPP reference + DR", COLORS["gray"]),
    ]
    for x, y, text, color in boxes:
        patch = FancyBboxPatch((x, y), 0.15, 0.16, boxstyle="round,pad=0.02",
                               facecolor="white", edgecolor=color, linewidth=2)
        ax.add_patch(patch)
        ax.text(x + 0.075, y + 0.08, text, ha="center", va="center", fontsize=10)
    arrows = [((0.18, .63), (.62, .63)), ((.46, .80), (.67, .70)), ((.46, .45), (.62, .58)),
              ((.77, .63), (.83, .63)), ((.83, .58), (.77, .58))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     color=COLORS["gray"], linewidth=1.4))
    ax.text(.5, .12, "Battery-only control boundary; positive battery power denotes discharge",
            ha="center", fontsize=11)
    save(fig, "Fig01_system_architecture")

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.axis("off")
    labels = [
        (0.02, "Observation\n(28 features)"), (0.22, "Greedy VPP-\ntracking prior"),
        (0.42, "Bounded SAC\nresidual (±150 kW)"), (0.62, "Physical\nprojection"),
        (0.80, "DR/grid service\ncoaching"),
    ]
    for x, text in labels:
        patch = FancyBboxPatch((x, .48), .15, .2, boxstyle="round,pad=0.02",
                               facecolor=COLORS["light"], edgecolor=COLORS["green"], linewidth=2)
        ax.add_patch(patch); ax.text(x + .075, .58, text, ha="center", va="center", fontsize=10)
    for x in (.17, .37, .57, .75):
        ax.add_patch(FancyArrowPatch((x, .58), (x + .05, .58), arrowstyle="-|>",
                                     mutation_scale=14, color=COLORS["gray"]))
    ax.text(.5, .26, "Proposed method adds residual-action regularization and terminal SOC recovery",
            ha="center", fontsize=11)
    save(fig, "Fig02_controller_architecture")


def scenario_profiles():
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    for axis, scenario in zip(axes.flat, SCENARIOS):
        frame = pd.read_parquet(ROOT / "data" / "processed" / "scenarios" / f"{scenario}.parquet")
        timestamp = pd.to_datetime(frame["timestamp_utc"], utc=True)
        mask = (timestamp >= pd.Timestamp("2019-10-01", tz="UTC")) & (
            timestamp < pd.Timestamp("2019-10-08", tz="UTC")
        )
        view = frame.loc[mask]
        x = np.arange(len(view)) / 96.0
        axis.plot(x, view["P_IT_kW"], color=COLORS["blue"], lw=1.2, label="IT demand")
        axis.plot(x, view["P_cooling_kW"], color=COLORS["orange"], lw=1.0, label="Cooling")
        axis.set_title(SCENARIO_LABELS[scenario], fontweight="bold")
        axis.set_ylabel("Power (kW)")
        style(axis)
    axes[-1, 0].set_xlabel("Day in representative test week")
    axes[-1, 1].set_xlabel("Day in representative test week")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .97))
    save(fig, "Fig03_scenario_profiles")


def workload_characteristics():
    data = pd.read_csv(ROOT / "tables" / "TABLE_IT_SCENARIO_CHARACTERISTICS.csv").set_index("scenario")
    x = np.arange(len(SCENARIOS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x, [data.loc[s, "mean_it_kW"] for s in SCENARIOS], color=COLORS["blue"])
    axes[1].bar(x, [data.loc[s, "p95_absolute_it_ramp_kW_per_h"] for s in SCENARIOS],
                color=COLORS["orange"])
    axes[0].set_ylabel("Mean IT demand (kW)")
    axes[1].set_ylabel("95th-percentile |IT ramp| (kW/h)")
    for axis in axes:
        axis.set_xticks(x, [SCENARIO_LABELS[s] for s in SCENARIOS], rotation=35, ha="right")
        style(axis)
    fig.tight_layout()
    save(fig, "Fig04_workload_characteristics")


def main_comparison():
    data = pd.read_csv(ROOT / "tables" / "TABLE_MAIN_CONTROLLER_RESULTS.csv")
    x = np.arange(len(data))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fields = [
        ("VPP_tracking_RMSE_kW_mean", "Mean daily VPP RMSE (kW)"),
        ("DR_interval_compliance_pct_mean", "Pooled DR-active compliance (%)"),
        ("daily_cost_EUR_mean", "Mean daily cost (EUR/day)"),
        ("battery_throughput_kWh_mean", "Battery throughput (kWh/day)"),
    ]
    colors = [COLORS["gray"]] * 5 + [COLORS["blue"], COLORS["green"], COLORS["orange"]]
    for axis, (field, label) in zip(axes.flat, fields):
        axis.bar(x, data[field], color=colors)
        axis.set_ylabel(label); style(axis)
        axis.set_xticks(x, [CONTROLLER_LABELS[c] for c in data["controller"]], rotation=30, ha="right")
    fig.tight_layout()
    save(fig, "Fig05_main_controller_comparison")


def sensitivity():
    data = pd.read_csv(ROOT / "tables" / "TABLE_IT_SCENARIO_RESULTS.csv").set_index(["scenario", "controller"])
    x = np.arange(len(SCENARIOS))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    for controller, color in zip(CONTROLLERS, [COLORS["gray"], COLORS["blue"], COLORS["orange"]]):
        rmse = [data.loc[(s, controller), "VPP_tracking_RMSE_kW_mean"] for s in SCENARIOS]
        dr = [data.loc[(s, controller), "pooled_DR_active_interval_compliance_pct_mean"] for s in SCENARIOS]
        axes[0].plot(x, rmse, marker="o", color=color, label=CONTROLLER_LABELS[controller])
        axes[1].plot(x, dr, marker="o", color=color, label=CONTROLLER_LABELS[controller])
    axes[0].set_ylabel("Mean daily VPP RMSE (kW)")
    axes[1].set_ylabel("Pooled DR-active interval\ncompliance (%)")
    axes[1].set_xticks(x, [SCENARIO_LABELS[s] for s in SCENARIOS], rotation=25, ha="right")
    axes[0].legend(ncol=3, frameon=False, loc="upper center")
    for axis in axes: style(axis)
    fig.tight_layout()
    save(fig, "Fig06_learned_workload_sensitivity")


def representative_and_soc():
    frame = pd.read_parquet(
        ROOT / "trajectories" / "learned_controllers" / "proposed_residual_safesac_seed_1.parquet"
    )
    timestamp = pd.to_datetime(frame["timestamp_utc"], utc=True)
    dr_days = timestamp[frame["DR_active"].astype(bool)].dt.floor("D")
    day = dr_days.iloc[0]
    view = frame.loc[timestamp.dt.floor("D") == day].copy()
    hour = np.arange(len(view)) / 4.0
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hour, view["P_import_kW"], label="Grid import", color=COLORS["blue"])
    ax.plot(hour, view["Pgrid_ref_kW"], label="VPP reference", color=COLORS["green"])
    ax.plot(hour, view["DR_import_limit_kW"], label="DR import limit", color=COLORS["orange"])
    active = view["DR_active"].astype(bool).to_numpy()
    ax.fill_between(hour, 0, 1, where=active, transform=ax.get_xaxis_transform(), alpha=.12,
                    color=COLORS["orange"], label="DR active")
    ax.set_xlabel("Hour UTC"); ax.set_ylabel("Power (kW)"); ax.legend(frameon=False, ncol=4)
    style(ax); fig.tight_layout(); save(fig, "Fig07_representative_dr_response")

    first = frame.loc[frame["day_index"] == frame["day_index"].iloc[0]].copy()
    hour = np.arange(len(first)) / 4.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(hour, first["P_batt_safe_kW"], color=COLORS["orange"])
    axes[0].axhline(0, color=COLORS["gray"], lw=.8)
    axes[0].set_ylabel("BESS power (kW)")
    axes[1].plot(hour, first["SOC"], color=COLORS["green"])
    axes[1].axhline(.2, ls="--", color=COLORS["gray"]); axes[1].axhline(.9, ls="--", color=COLORS["gray"])
    axes[1].set_ylabel("SOC (fraction)"); axes[1].set_xlabel("Hour UTC")
    for axis in axes: style(axis)
    fig.tight_layout(); save(fig, "Fig08_soc_and_battery_dispatch")


def multiseed_and_scatter():
    data = pd.read_csv(ROOT / "tables" / "TABLE_SEED_RESULTS.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for axis, (field, label) in zip(axes, [
        ("VPP_tracking_RMSE_kW", "VPP RMSE (kW)"),
        ("daily_cost_EUR", "Cost (EUR/day)"),
        ("terminal_SOC_error", "Terminal SOC error")]):
        groups = [data.loc[data.controller == c, field].to_numpy() for c in CONTROLLERS]
        axis.boxplot(groups, tick_labels=[CONTROLLER_LABELS[c] for c in CONTROLLERS], showmeans=True)
        axis.set_ylabel(label); axis.tick_params(axis="x", rotation=28); style(axis)
    fig.tight_layout(); save(fig, "Fig09_multiseed_stability")

    frame = pd.read_parquet(
        ROOT / "trajectories" / "learned_controllers" / "proposed_residual_safesac_seed_1.parquet"
    )
    physical = frame["physical_projection"].astype(bool)
    coaching = frame["service_coaching"].astype(bool)
    categories = np.select(
        [physical & coaching, physical, coaching],
        ["Combined intervention", "Physical projection", "Service coaching"],
        default="Unchanged",
    )
    fig, ax = plt.subplots(figsize=(6.5, 6))
    palette = {"Unchanged": COLORS["gray"], "Physical projection": COLORS["blue"],
               "Service coaching": COLORS["orange"], "Combined intervention": COLORS["green"]}
    for category in np.unique(categories):
        mask = categories == category
        ax.scatter(frame.loc[mask, "P_batt_commanded_kW"], frame.loc[mask, "P_batt_safe_kW"],
                   s=9, alpha=.45, color=palette[category], label=category)
    limits = [-760, 760]
    ax.plot(limits, limits, "k--", lw=.8); ax.set_xlim(limits); ax.set_ylim(limits)
    ax.set_xlabel("Commanded BESS power (kW)"); ax.set_ylabel("Applied BESS power (kW)")
    ax.legend(frameon=False); style(ax); fig.tight_layout(); save(fig, "FigS1_commanded_applied")


def generate_all_figures() -> list[Path]:
    architecture_figures(); scenario_profiles(); workload_characteristics(); main_comparison()
    sensitivity(); representative_and_soc(); multiseed_and_scatter()
    files = sorted(OUTPUT.glob("*.png")) + sorted(OUTPUT.glob("*.pdf"))
    if len(files) != 20:
        raise RuntimeError(f"Expected 20 reproduced figure files, found {len(files)}")
    return files


if __name__ == "__main__":
    files = generate_all_figures()
    print(f"Generated {len(files)} figure files under {OUTPUT.relative_to(ROOT)}")
