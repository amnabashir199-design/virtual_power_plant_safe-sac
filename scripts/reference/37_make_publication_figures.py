from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/publication_v1/figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    boxes = [
        (.02, .55, .16, .25, "State and\nforecasts"), (.24, .55, .16, .25, "Greedy VPP\ntracking prior"),
        (.24, .12, .16, .25, "SAC residual\npolicy"), (.47, .38, .16, .25, "Bounded command\ncomposition"),
        (.70, .38, .16, .25, "Physical projection\nand service coaching"), (.89, .38, .10, .25, "Battery /\ngrid"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.02", fc="#e8f3ef", ec="#176b55", lw=1.5))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10)
    arrows = [((.18,.675),(.24,.675)),((.18,.60),(.24,.245)),((.40,.675),(.47,.55)),((.40,.245),(.47,.45)),((.63,.505),(.70,.505)),((.86,.505),(.89,.505))]
    for a,b in arrows: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=14,color="#333"))
    ax.text(.55,.82,"Commanded action",ha="center",fontsize=9)
    ax.text(.78,.22,"Logged stages: raw → physical → coached → applied",ha="center",fontsize=9)
    save(fig, "01_proposed_controller_architecture")


def main():
    architecture()
    tables = ROOT / "results/publication_v1/tables"
    trajectories = ROOT / "results/publication_v1/trajectories"
    authoritative = pd.read_csv(tables / "AUTHORITATIVE_CONTROLLER_RESULTS.csv")
    proposed_path = trajectories / "proposed_residual_safesac_seed_1_daily_reset.parquet"
    traj = pd.read_parquet(proposed_path)
    active_dates = pd.to_datetime(traj.loc[traj["DR_active"].gt(0), "timestamp_utc"]).dt.date
    day = active_dates.iloc[0] if len(active_dates) else pd.to_datetime(traj["timestamp_utc"]).dt.date.iloc[0]
    sample = traj[pd.to_datetime(traj["timestamp_utc"]).dt.date.eq(day)]
    ts = pd.to_datetime(sample["timestamp_utc"])

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ts, sample["P_grid_kW"], label="Grid power")
    ax.plot(ts, sample["Pgrid_ref_kW"], label="VPP reference")
    ax.plot(ts, sample["DR_import_limit_kW"], label="DR limit", ls="--")
    ax.set_ylabel("Power (kW)"); ax.legend(ncol=3); ax.grid(alpha=.25)
    save(fig, "02_representative_tracking_dr_day")

    fig, axes = plt.subplots(2,1,figsize=(10,6),sharex=True)
    axes[0].plot(ts, sample["SOC"]); axes[0].set_ylabel("SOC")
    axes[1].plot(ts, sample["P_batt_safe_kW"], label="Applied")
    axes[1].plot(ts, sample["P_batt_commanded_kW"], alpha=.7, label="Commanded")
    axes[1].set_ylabel("Battery power (kW)"); axes[1].legend(); axes[1].grid(alpha=.25)
    save(fig, "03_soc_and_battery_dispatch")

    labels = authoritative["controller"]
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for ax, metric, title in zip(axes,
        ["daily_cost_EUR_mean","VPP_tracking_RMSE_kW_mean","DR_event_compliance_pct_mean"],
        ["Daily operating cost","Tracking RMSE","Event DR compliance"]):
        ax.bar(labels, authoritative[metric]); ax.set_title(title); ax.tick_params(axis="x",rotation=70)
    save(fig, "04_main_controller_comparison")

    fig, ax = plt.subplots(figsize=(7,5))
    sc=ax.scatter(authoritative["daily_cost_EUR_mean"], authoritative["carbon_emissions_kgCO2_mean"],
                  c=authoritative["VPP_tracking_RMSE_kW_mean"], cmap="viridis", s=90)
    for _,r in authoritative.iterrows(): ax.annotate(r["controller"],(r["daily_cost_EUR_mean"],r["carbon_emissions_kgCO2_mean"]),fontsize=7)
    ax.set_xlabel("Daily operating cost (EUR)"); ax.set_ylabel("Daily carbon emissions (kgCO2)")
    fig.colorbar(sc,ax=ax,label="Tracking RMSE (kW)"); save(fig,"05_cost_carbon_tracking_tradeoff")

    seed = pd.read_csv(tables / "publication_seed_and_baseline_summary.csv")
    learned = seed[seed["seed"].notna() & seed["protocol"].eq("daily_reset")]
    grouped=learned.groupby("controller")["VPP_tracking_RMSE_kW"].agg(["mean","std"])
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(grouped.index,grouped["mean"],yerr=grouped["std"],capsize=4)
    ax.set_ylabel("Tracking RMSE (kW)"); ax.tick_params(axis="x",rotation=25); save(fig,"06_multiseed_uncertainty")

    scenario_path=ROOT/"results/publication_v1/scenarios/scenario_training_seed_summary.csv"
    if scenario_path.exists():
        scenario=pd.read_csv(scenario_path)
        fig,ax=plt.subplots(figsize=(11,5))
        for name,g in scenario.groupby("controller"):
            ax.plot(g["scenario"],g["VPP_tracking_RMSE_kW_mean"],marker="o",label=name)
        ax.tick_params(axis="x",rotation=45); ax.set_ylabel("Mean daily tracking RMSE (kW)"); ax.legend(fontsize=7)
        save(fig,"07_scenario_sensitivity")

    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(traj["P_batt_commanded_kW"],traj["P_batt_safe_kW"],c=traj["service_coaching"].astype(int),s=4,alpha=.4)
    lim=np.nanmax(np.abs(traj[["P_batt_commanded_kW","P_batt_safe_kW"]].to_numpy()))
    ax.plot([-lim,lim],[-lim,lim],ls="--",color="black"); ax.set_xlabel("Commanded power (kW)"); ax.set_ylabel("Applied power (kW)")
    save(fig,"08_safety_projection_behavior")


if __name__ == "__main__":
    main()
