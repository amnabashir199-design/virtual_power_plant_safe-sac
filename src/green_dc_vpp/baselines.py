from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _get(info: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not info:
        return default
    value = info.get(key, default)
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return float(value)


def _action(value: float) -> np.ndarray:
    return np.asarray([np.clip(value, -1.0, 1.0)], dtype=np.float32)


def deterministic_baseline_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return an evaluation-only config for direct deterministic commands.

    Baseline controllers emit normalized *direct physical battery commands*.
    They therefore must not inherit a residual controller's greedy prior or
    residual action scaling.  They share only the common physical feasibility
    projection (power, SOC, and ramp bounds); DR/grid service coaching remains
    specific to controllers whose methodology explicitly includes it.

    The input configuration is deep-copied and never mutated.
    """

    baseline_config = copy.deepcopy(config)
    control = baseline_config.setdefault("control", {})
    control["mode"] = "direct_sac"
    control["greedy_prior_enabled"] = False
    control["greedy_blend_alpha"] = 0.0

    safety = baseline_config.setdefault("safety_layer", {})
    safety["enabled"] = True
    safety["enforce_power_limit"] = True
    safety["enforce_soc_bounds"] = True
    safety["enforce_ramp_limit"] = True
    safety["enforce_dr_limit"] = False
    safety["enforce_grid_limit"] = False
    return baseline_config


@dataclass
class BaseController:
    name: str
    P_batt_max_kW: float = 750.0
    price_high: float = 0.05
    price_low: float = 0.025
    carbon_high: float = 550.0
    carbon_low: float = 400.0

    def act(self, obs, info: dict[str, Any] | None = None) -> np.ndarray:
        raise NotImplementedError


class NoBatteryController(BaseController):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="no_battery", **kwargs)

    def act(self, obs, info: dict[str, Any] | None = None) -> np.ndarray:
        return _action(0.0)


class RuleBasedController(BaseController):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="rule_based", **kwargs)

    def act(self, obs, info: dict[str, Any] | None = None) -> np.ndarray:
        dr_active = _get(info, "DR_active", 0.0) >= 0.5
        price = _get(info, "price_buy_EUR_per_kWh", 0.0)
        carbon = _get(info, "carbon_gCO2_per_kWh_proxy", 0.0)
        pv = _get(info, "Ppv_kW", 0.0)
        load = _get(info, "P_total_dc_kW", 0.0)
        if dr_active:
            return _action(0.9)
        if price >= self.price_high or carbon >= self.carbon_high:
            return _action(0.55)
        if pv > 0.5 * load and price <= self.price_low and carbon <= self.carbon_low:
            return _action(-0.45)
        return _action(0.0)


class GreedyTrackingController(BaseController):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="greedy_tracking", **kwargs)

    def act(self, obs, info: dict[str, Any] | None = None) -> np.ndarray:
        p_net = _get(info, "P_net_without_battery_kW", _get(info, "P_total_dc_kW", 0.0) - _get(info, "Ppv_kW", 0.0))
        p_ref = _get(info, "Pgrid_ref_kW", p_net)
        target = p_net - p_ref
        return _action(target / max(self.P_batt_max_kW, 1e-9))


class ResidualZeroController(BaseController):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="residual_zero", **kwargs)

    def act(self, obs, info: dict[str, Any] | None = None) -> np.ndarray:
        return _action(0.0)


class TOUSelfConsumptionController(BaseController):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="tou_self_consumption", **kwargs)

    def act(self, obs, info: dict[str, Any] | None = None) -> np.ndarray:
        price = _get(info, "price_buy_EUR_per_kWh", 0.0)
        pv = _get(info, "Ppv_kW", 0.0)
        load = _get(info, "P_total_dc_kW", 0.0)
        if price <= self.price_low and pv > 0.25 * load:
            return _action(-0.55)
        if price >= self.price_high:
            return _action(0.65)
        if pv > load:
            return _action(-0.35)
        return _action(0.0)


class CarbonAwareController(BaseController):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="carbon_aware", **kwargs)

    def act(self, obs, info: dict[str, Any] | None = None) -> np.ndarray:
        carbon = _get(info, "carbon_gCO2_per_kWh_proxy", 0.0)
        dr_active = _get(info, "DR_active", 0.0) >= 0.5
        if dr_active or carbon >= self.carbon_high:
            return _action(0.65)
        if carbon <= self.carbon_low:
            return _action(-0.45)
        return _action(0.0)


def build_baseline_controllers(df: pd.DataFrame | None = None, P_batt_max_kW: float = 750.0) -> list[BaseController]:
    kwargs: dict[str, float] = {"P_batt_max_kW": P_batt_max_kW}
    if df is not None and not df.empty:
        if "price_buy_EUR_per_kWh" in df.columns:
            price = pd.to_numeric(df["price_buy_EUR_per_kWh"], errors="coerce")
            kwargs["price_low"] = float(price.quantile(0.25))
            kwargs["price_high"] = float(price.quantile(0.75))
        if "carbon_gCO2_per_kWh_proxy" in df.columns:
            carbon = pd.to_numeric(df["carbon_gCO2_per_kWh_proxy"], errors="coerce")
            kwargs["carbon_low"] = float(carbon.quantile(0.25))
            kwargs["carbon_high"] = float(carbon.quantile(0.75))
    return [
        NoBatteryController(**kwargs),
        RuleBasedController(**kwargs),
        GreedyTrackingController(**kwargs),
        ResidualZeroController(**kwargs),
        TOUSelfConsumptionController(**kwargs),
        CarbonAwareController(**kwargs),
    ]
