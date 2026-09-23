from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_current_figure(path: str | Path, dpi: int = 160) -> None:
    """Save the active matplotlib figure, creating the parent directory first."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")


def _save(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_grid_tracking(traj_df: pd.DataFrame, out_path: str | Path) -> None:
    ts = pd.to_datetime(traj_df["timestamp_utc"]) if "timestamp_utc" in traj_df else traj_df.index
    plt.figure(figsize=(11, 4))
    plt.plot(ts, traj_df["P_grid_kW"], label="Grid power", linewidth=1.8)
    plt.plot(ts, traj_df["Pgrid_ref_kW"], label="VPP reference", linewidth=1.4)
    if "DR_active" in traj_df:
        dr = traj_df["DR_active"].astype(bool)
        if dr.any():
            plt.scatter(pd.Series(ts)[dr], traj_df.loc[dr, "P_grid_kW"], s=14, label="DR active")
    plt.ylabel("kW")
    plt.xlabel("Time")
    plt.legend()
    plt.title("Grid Tracking Example")
    _save(out_path)


def plot_soc_trajectory(traj_df: pd.DataFrame, out_path: str | Path) -> None:
    ts = pd.to_datetime(traj_df["timestamp_utc"]) if "timestamp_utc" in traj_df else traj_df.index
    plt.figure(figsize=(11, 3.5))
    plt.plot(ts, traj_df["SOC"], label="SOC", color="tab:green", linewidth=1.8)
    plt.axhline(0.2, color="tab:red", linestyle="--", linewidth=1.0, label="SOC min")
    plt.axhline(0.9, color="tab:orange", linestyle="--", linewidth=1.0, label="SOC max")
    plt.ylabel("SOC")
    plt.xlabel("Time")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.title("Battery SOC Example")
    _save(out_path)


def plot_battery_action_safety(traj_df: pd.DataFrame, out_path: str | Path) -> None:
    ts = pd.to_datetime(traj_df["timestamp_utc"]) if "timestamp_utc" in traj_df else traj_df.index
    plt.figure(figsize=(11, 4))
    plt.plot(ts, traj_df["P_batt_raw_kW"], label="Raw action", alpha=0.65)
    plt.plot(ts, traj_df["P_batt_safe_kW"], label="Safe action", linewidth=1.8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Battery power kW")
    plt.xlabel("Time")
    plt.legend()
    plt.title("Battery Action Projection")
    _save(out_path)


def plot_baseline_summary(summary_df: pd.DataFrame, out_path: str | Path) -> None:
    if "controller" not in summary_df.columns:
        raise ValueError("summary_df must contain a controller column")
    metrics = [m for m in ["VPP_tracking_RMSE_kW", "daily_cost_EUR", "DR_violation_energy_kWh"] if m in summary_df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        ax.bar(summary_df["controller"], summary_df[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35)
    _save(out_path)
