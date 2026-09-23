from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_CONSTRAINT_TOLERANCE_KW = 1e-6


def _cfg_value(config, section: str, key: str, default: float) -> float:
    if config is None:
        return float(default)
    section_value = config.get(section, {}) if hasattr(config, "get") else {}
    return float(section_value.get(key, default))


def _physical_parameter(traj_df: pd.DataFrame, config, key: str, trajectory_column: str) -> float:
    if config is not None:
        section = config.get("battery", {}) if hasattr(config, "get") else {}
        if key in section:
            return float(section[key])
    if trajectory_column in traj_df.columns:
        values = pd.to_numeric(traj_df[trajectory_column], errors="coerce").dropna().unique()
        if len(values) == 1:
            return float(values[0])
    raise ValueError(
        f"Metric calculation requires battery.{key} in the active configuration "
        f"or a constant {trajectory_column} trajectory column."
    )


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return transposed numeric summary statistics."""

    return df.select_dtypes(include="number").describe().T


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and fractions by column."""

    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "column": missing.index,
            "missing_count": missing.values,
            "missing_fraction": (missing / max(len(df), 1)).values,
            "dtype": [str(dtype) for dtype in df.dtypes],
        }
    )


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def compute_trajectory_metrics(traj_df: pd.DataFrame, config=None) -> dict[str, float]:
    """Compute one-episode dispatch metrics from an environment trajectory."""

    if traj_df.empty:
        return {
            "daily_cost_EUR": 0.0,
            "objective_value": 0.0,
            "grid_energy_cost_EUR": 0.0,
            "imported_energy_kWh": 0.0,
            "exported_energy_kWh": 0.0,
            "export_energy_kWh": 0.0,
            "carbon_emissions_kgCO2": 0.0,
            "carbon_intensity_avg_gCO2_per_kWh": np.nan,
            "pv_generation_kWh": np.nan,
            "pv_used_locally_kWh": np.nan,
            "renewable_self_consumption_ratio_pct": np.nan,
            "PV_self_consumption_kWh": np.nan,
            "renewable_self_consumption_ratio": np.nan,
            "battery_throughput_kWh": 0.0,
            "equivalent_full_cycles": 0.0,
            "mean_daily_peak_import_kW": 0.0,
            "horizon_max_grid_import_kW": 0.0,
            "peak_contract_violation_count": 0.0,
            "peak_contract_violation_energy_kWh": 0.0,
            "VPP_tracking_RMSE_kW": 0.0,
            "VPP_tracking_MAE_kW": 0.0,
            "DR_success_rate_pct": 100.0,
            "DR_interval_compliance_pct": 100.0,
            "DR_event_count": 0.0,
            "DR_successful_event_count": 0.0,
            "DR_event_compliance_pct": 100.0,
            "DR_max_violation_kW": 0.0,
            "feasible_DR_success_pct": 100.0,
            "DR_violation_energy_kWh": 0.0,
            "min_SOC": 0.0,
            "max_SOC": 0.0,
            "final_SOC": 0.0,
            "terminal_SOC_error": 0.0,
            "hard_clip_fraction_pct": 0.0,
            "soft_coach_fraction_pct": 0.0,
            "physical_projection_fraction_pct": 0.0,
            "service_coaching_fraction_pct": 0.0,
            "mean_reward": 0.0,
        }

    dt_h = 0.25
    if "timestamp_utc" in traj_df.columns and len(traj_df) > 1:
        ts = pd.to_datetime(traj_df["timestamp_utc"], errors="coerce")
        diffs = ts.sort_values().diff().dropna()
        if not diffs.empty:
            dt_h = float(diffs.mode().iloc[0].total_seconds() / 3600.0)

    p_grid = _col(traj_df, "P_grid_kW")
    if "P_import_kW" in traj_df.columns:
        p_import = _col(traj_df, "P_import_kW")
    else:
        p_import = p_grid.clip(lower=0.0)
    if "P_export_kW" in traj_df.columns:
        p_export = _col(traj_df, "P_export_kW")
    else:
        p_export = (-p_grid).clip(lower=0.0)
    p_batt = _col(traj_df, "P_batt_safe_kW")
    p_ref = _col(traj_df, "Pgrid_ref_kW")
    grid_limit = _col(traj_df, "grid_import_contract_limit_kW", np.inf).replace(0, np.inf)
    soc = _col(traj_df, "SOC")
    reward = _col(traj_df, "reward")
    cost = _col(traj_df, "cost_EUR")
    imported_energy = float((p_import * dt_h).sum())
    exported_energy = float((p_export * dt_h).sum())
    if "carbon_gCO2_per_kWh_proxy" in traj_df.columns:
        carbon_emissions = (p_import * dt_h * _col(traj_df, "carbon_gCO2_per_kWh_proxy") / 1000.0).sum()
    elif "carbon_kg_proxy" in traj_df.columns:
        carbon_emissions = _col(traj_df, "carbon_kg_proxy").sum()
    else:
        carbon_emissions = np.nan
    carbon_intensity = (
        float(carbon_emissions * 1000.0 / imported_energy)
        if imported_energy > 1e-9 and np.isfinite(carbon_emissions)
        else np.nan
    )

    pv_generation = np.nan
    pv_used_locally = np.nan
    renewable_self_consumption_ratio_pct = np.nan
    if {"Ppv_kW", "P_total_dc_kW", "P_batt_safe_kW"}.issubset(traj_df.columns):
        pv = _col(traj_df, "Ppv_kW")
        dc_load = _col(traj_df, "P_total_dc_kW")
        battery_charge = (-p_batt).clip(lower=0.0)
        pv_generation = float((pv * dt_h).sum())
        pv_used_locally = float((np.minimum(pv, dc_load + battery_charge) * dt_h).sum())
        if pv_generation > 1e-9:
            renewable_self_consumption_ratio_pct = float(100.0 * pv_used_locally / pv_generation)
    elif "Ppv_kW" in traj_df.columns:
        pv_generation = float((_col(traj_df, "Ppv_kW") * dt_h).sum())

    tracking_error = p_grid - p_ref
    peak_excess = (p_import - grid_limit).clip(lower=0.0)

    dr_active = _col(traj_df, "DR_active").astype(bool)
    dr_limit = _col(traj_df, "DR_import_limit_kW", np.nan)
    feasible_dr = dr_active & dr_limit.notna() & np.isfinite(dr_limit)
    tolerance_kw = _cfg_value(
        config, "metrics", "constraint_tolerance_kW", DEFAULT_CONSTRAINT_TOLERANCE_KW
    )
    dr_excess = (p_import - dr_limit).where(feasible_dr, 0.0)
    dr_violation = feasible_dr & (dr_excess > tolerance_kw)
    effective_dr_excess = dr_excess.clip(lower=0.0).where(dr_violation, 0.0)
    dr_violation_energy = (effective_dr_excess * dt_h).sum()

    # A DR event is one contiguous run of feasible active intervals. Timestamp
    # discontinuities also delimit events, preventing unrelated rows from merging.
    if "timestamp_utc" in traj_df.columns:
        timestamps = pd.to_datetime(traj_df["timestamp_utc"], errors="coerce")
        contiguous = timestamps.diff().eq(pd.Timedelta(hours=dt_h))
    else:
        contiguous = pd.Series(True, index=traj_df.index)
        if len(contiguous):
            contiguous.iloc[0] = False
    event_start = feasible_dr & (~feasible_dr.shift(fill_value=False) | ~contiguous)
    event_ids = event_start.cumsum().where(feasible_dr)
    event_success = (~dr_violation).where(feasible_dr).groupby(event_ids).all()
    dr_event_count = int(len(event_success))
    dr_successful_event_count = int(event_success.sum()) if dr_event_count else 0

    throughput = (p_batt.abs() * dt_h).sum()
    battery_capacity = _physical_parameter(traj_df, config, "E_batt_kWh", "battery_capacity_kWh")

    hard_clip = traj_df.get("hard_clip", pd.Series(False, index=traj_df.index)).astype(bool)
    soft_coach = traj_df.get("soft_coach", pd.Series(False, index=traj_df.index)).astype(bool)
    physical_projection = traj_df.get("physical_projection", hard_clip).astype(bool)
    service_coaching = traj_df.get("service_coaching", soft_coach).astype(bool)
    initial_soc = _physical_parameter(traj_df, config, "SOC_initial", "initial_SOC_target")
    terminal_error = abs(float(soc.iloc[-1]) - initial_soc) if not soc.empty else 0.0

    dr_count = int(dr_active.sum())
    feasible_dr_count = int(feasible_dr.sum())
    return {
        "daily_cost_EUR": float(cost.sum()),
        "objective_value": float(-reward.sum()),
        "grid_energy_cost_EUR": float(cost.sum()),
        "imported_energy_kWh": imported_energy,
        "exported_energy_kWh": exported_energy,
        "export_energy_kWh": exported_energy,
        "carbon_emissions_kgCO2": float(carbon_emissions) if np.isfinite(carbon_emissions) else np.nan,
        "carbon_intensity_avg_gCO2_per_kWh": carbon_intensity,
        "pv_generation_kWh": float(pv_generation) if np.isfinite(pv_generation) else np.nan,
        "pv_used_locally_kWh": float(pv_used_locally) if np.isfinite(pv_used_locally) else np.nan,
        "renewable_self_consumption_ratio_pct": (
            float(renewable_self_consumption_ratio_pct)
            if np.isfinite(renewable_self_consumption_ratio_pct)
            else np.nan
        ),
        "PV_self_consumption_kWh": float(pv_used_locally) if np.isfinite(pv_used_locally) else np.nan,
        "renewable_self_consumption_ratio": (
            float(renewable_self_consumption_ratio_pct / 100.0)
            if np.isfinite(renewable_self_consumption_ratio_pct)
            else np.nan
        ),
        "battery_throughput_kWh": float(throughput),
        "equivalent_full_cycles": float(throughput / (2.0 * battery_capacity)),
        "mean_daily_peak_import_kW": float(p_import.max()),
        "horizon_max_grid_import_kW": float(p_import.max()),
        "peak_contract_violation_count": float((peak_excess > 1e-9).sum()),
        "peak_contract_violation_energy_kWh": float((peak_excess * dt_h).sum()),
        "VPP_tracking_RMSE_kW": float(np.sqrt(np.mean(np.square(tracking_error)))),
        "VPP_tracking_MAE_kW": float(np.mean(np.abs(tracking_error))),
        "DR_success_rate_pct": float(100.0 if dr_count == 0 else 100.0 * (1.0 - dr_violation.sum() / dr_count)),
        "DR_interval_compliance_pct": float(100.0 if dr_count == 0 else 100.0 * (1.0 - dr_violation.sum() / dr_count)),
        "DR_event_count": float(dr_event_count),
        "DR_successful_event_count": float(dr_successful_event_count),
        "DR_event_compliance_pct": float(100.0 if dr_event_count == 0 else 100.0 * dr_successful_event_count / dr_event_count),
        "DR_max_violation_kW": float(effective_dr_excess.max()),
        "feasible_DR_success_pct": float(100.0 if feasible_dr_count == 0 else 100.0 * (1.0 - dr_violation.sum() / feasible_dr_count)),
        "DR_violation_energy_kWh": float(dr_violation_energy),
        "min_SOC": float(soc.min()),
        "max_SOC": float(soc.max()),
        "final_SOC": float(soc.iloc[-1]),
        "terminal_SOC_error": float(terminal_error),
        "hard_clip_fraction_pct": float(100.0 * hard_clip.mean()),
        "soft_coach_fraction_pct": float(100.0 * soft_coach.mean()),
        "physical_projection_fraction_pct": float(100.0 * physical_projection.mean()),
        "service_coaching_fraction_pct": float(100.0 * service_coaching.mean()),
        "mean_reward": float(reward.mean()),
    }
