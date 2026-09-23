from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _get(mapping: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _row_get(row: Mapping[str, Any] | pd.Series, key: str, default: Any = 0.0) -> Any:
    try:
        value = row[key]
    except Exception:
        value = default
    if pd.isna(value):
        return default
    return value


def _battery_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return _get(params, "battery", params)


def _system_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return _get(params, "system", {})


def _safety_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return _get(params, "safety_layer", {})


def _soc_bounds(params: Mapping[str, Any]) -> tuple[float, float]:
    battery = _battery_params(params)
    safety = _safety_params(params)
    soc_min = float(_get(battery, "SOC_min", 0.2))
    soc_max = float(_get(battery, "SOC_max", 0.9))
    if bool(_get(safety, "enforce_soc_reserve", False)):
        soc_min = max(soc_min, float(_get(battery, "SOC_reserve", soc_min)))
    return soc_min, soc_max


def _bounds_from_soc(SOC: float, params: Mapping[str, Any], dt_h: float) -> tuple[float, float]:
    battery = _battery_params(params)
    E_batt = float(_get(battery, "E_batt_kWh", 1.0))
    eta_ch = float(_get(battery, "eta_ch", 0.95))
    eta_dis = float(_get(battery, "eta_dis", 0.95))
    soc_min, soc_max = _soc_bounds(params)

    max_discharge = max(0.0, (SOC - soc_min) * E_batt * eta_dis / max(dt_h, 1e-9))
    max_charge = max(0.0, (soc_max - SOC) * E_batt / max(eta_ch * dt_h, 1e-9))
    return -max_charge, max_discharge


def _simulate_import(row: Mapping[str, Any] | pd.Series, p_batt_kw: float) -> float:
    p_net = float(_row_get(row, "P_net_without_battery_kW", _row_get(row, "P_total_dc_kW", 0.0) - _row_get(row, "Ppv_kW", 0.0)))
    return max(p_net - p_batt_kw, 0.0)


def _available_discharge_upper(SOC: float, params: Mapping[str, Any], dt_h: float) -> float:
    _, upper = _bounds_from_soc(SOC, params, dt_h)
    battery = _battery_params(params)
    pmax = float(_get(battery, "P_batt_max_kW", upper))
    return max(0.0, min(pmax, upper))


def project_battery_action(
    P_batt_raw_kW: float,
    SOC: float,
    prev_P_batt_kW: float,
    row: Mapping[str, Any] | pd.Series,
    params: Mapping[str, Any],
    dt_h: float,
) -> dict[str, Any]:
    """Project a scalar battery action into physically safe bounds."""

    battery = _battery_params(params)
    system = _system_params(params)
    safety = _safety_params(params)

    pmax = float(_get(battery, "P_batt_max_kW", 750.0))
    enabled = bool(_get(safety, "enabled", True))
    if not enabled:
        raw = float(P_batt_raw_kW)
        return {
            "P_batt_safe_kW": raw,
            "P_batt_physical_kW": raw,
            "P_batt_service_coached_kW": raw,
            "physical_projection": False,
            "service_coaching": False,
            "hard_clip": False,
            "soft_coach": False,
            "clip_reasons": [],
            "safe_upper_kW": float("inf"),
            "safe_lower_kW": float("-inf"),
        }

    enforce_power = bool(_get(safety, "enforce_power_limit", True))
    lower, upper = (-pmax, pmax) if enforce_power else (-float("inf"), float("inf"))
    reasons: list[str] = []

    if bool(_get(safety, "enforce_soc_bounds", True)):
        soc_lower, soc_upper = _bounds_from_soc(float(SOC), params, dt_h)
        if soc_upper < upper:
            reasons.append("soc_min")
        if soc_lower > lower:
            reasons.append("soc_max")
        lower = max(lower, soc_lower)
        upper = min(upper, soc_upper)

    if bool(_get(safety, "enforce_ramp_limit", True)):
        # Power/SOC bounds are hard feasibility limits.  A previous command can
        # leave the ramp corridor disjoint from those limits (for example,
        # sustained charging immediately before SOC_max).  In that emergency
        # case the closest hard-feasible boundary must override the ramp limit;
        # otherwise the former implementation could deliberately cross SOC_max.
        physical_lower, physical_upper = lower, upper
        ramp = float(_get(battery, "ramp_limit_kW_per_15min", pmax))
        ramp_lower = float(prev_P_batt_kW) - ramp
        ramp_upper = float(prev_P_batt_kW) + ramp
        if ramp_lower > lower or ramp_upper < upper:
            reasons.append("ramp_limit")
        intersection_lower = max(physical_lower, ramp_lower)
        intersection_upper = min(physical_upper, ramp_upper)
        if intersection_lower <= intersection_upper:
            lower, upper = intersection_lower, intersection_upper
        elif ramp_upper < physical_lower:
            lower = upper = physical_lower
            reasons.append("ramp_overridden_for_hard_feasibility")
        else:
            lower = upper = physical_upper
            reasons.append("ramp_overridden_for_hard_feasibility")

    lower = float(lower)
    upper = float(upper)
    raw = float(P_batt_raw_kW)
    p_physical = float(np.clip(raw, lower, upper))

    hard_clip = not np.isclose(raw, p_physical, rtol=0.0, atol=1e-6)
    if enforce_power and abs(raw) > pmax + 1e-6:
        reasons.append("power_limit")

    dr_active = int(float(_row_get(row, "DR_active", 0))) == 1
    grid_limit = float(_row_get(row, "grid_import_contract_limit_kW", _get(system, "grid_import_contract_limit_kW", 1e9)))
    if not np.isfinite(grid_limit) or grid_limit <= 0:
        grid_limit = 1e9

    p_coached = p_physical
    enforce_dr = bool(_get(safety, "enforce_dr_limit", True))
    enforce_grid = bool(_get(safety, "enforce_grid_limit", True))
    if enforce_dr and dr_active and p_coached < 0.0:
        p_coached = 0.0
        reasons.append("dr_no_charge")

    p_import = _simulate_import(row, p_coached)
    if enforce_grid and p_import > grid_limit:
        needed = p_import - grid_limit
        discharge_upper = min(upper, _available_discharge_upper(float(SOC), params, dt_h))
        coached = min(discharge_upper, p_coached + needed)
        if coached > p_coached + 1e-9:
            p_coached = coached
            reasons.append("grid_contract_support")

    if enforce_dr and dr_active:
        dr_limit = float(_row_get(row, "DR_import_limit_kW", _row_get(row, "DR_import_limit_for_model_kW", grid_limit)))
        if np.isfinite(dr_limit) and dr_limit < 1e8 and _simulate_import(row, p_coached) > dr_limit:
            needed = _simulate_import(row, p_coached) - dr_limit
            discharge_upper = min(upper, _available_discharge_upper(float(SOC), params, dt_h))
            coached = min(discharge_upper, p_coached + needed)
            if coached > p_coached + 1e-9:
                p_coached = coached
                reasons.append("dr_support")

    p_safe = float(np.clip(p_coached, lower, upper))
    soft_coach = not np.isclose(p_physical, p_safe, rtol=0.0, atol=1e-6)
    return {
        "P_batt_safe_kW": p_safe,
        "P_batt_physical_kW": p_physical,
        "P_batt_service_coached_kW": p_coached,
        "physical_projection": bool(hard_clip),
        "service_coaching": bool(soft_coach),
        "hard_clip": bool(hard_clip),
        "soft_coach": bool(soft_coach),
        "clip_reasons": sorted(set(reasons)),
        "safe_upper_kW": float(upper),
        "safe_lower_kW": float(lower),
    }
