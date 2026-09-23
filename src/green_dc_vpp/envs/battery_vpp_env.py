from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from green_dc_vpp.data_loader import TIMESTAMP_COLUMN
from green_dc_vpp.normalization import fit_training_normalizers, load_normalizers
from green_dc_vpp.safety import project_battery_action


OBSERVATION_COLUMNS = [
    "SOC",
    "prev_battery_power_norm",
    "prev_grid_import_norm",
    "P_total_dc_kW",
    "P_IT_kW",
    "P_cooling_kW",
    "Pdc_forecast_kW",
    "Ppv_kW",
    "Ppv_forecast_kW",
    "price_buy_EUR_per_kWh",
    "price_forecast_EUR_per_kWh",
    "carbon_gCO2_per_kWh_proxy",
    "renewable_share_proxy",
    "Pgrid_ref_kW",
    "DR_active",
    "DR_import_limit_norm",
    "DR_required_reduction_kW",
    "grid_import_contract_limit_kW",
    "ambient_temp_C",
    "temperature_deviation_C",
    "sin_hour",
    "cos_hour",
    "sin_day",
    "cos_day",
]

RESIDUAL_OBSERVATION_COLUMNS = [
    "P_batt_greedy_norm",
    "tracking_error_without_battery_norm",
    "available_discharge_power_norm",
    "available_charge_power_norm",
]

DATA_CENTER_FEATURE_COLUMNS = ["ai_utilization", "PUE_model_norm", "cooling_ratio_norm", "remaining_steps_norm"]
SCENARIO_FAMILIES = ["baseline", "load_level", "workload_shape", "workload_flexibility", "thermal_stress", "grid_stress"]
SCENARIO_FEATURE_COLUMNS = [f"scenario_family_{name}" for name in SCENARIO_FAMILIES]


def _get(mapping: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _nested(mapping: Mapping[str, Any] | Any, key: str) -> Mapping[str, Any]:
    value = _get(mapping, key, {})
    return value if isinstance(value, Mapping) else {}


def _safe_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row[key] if key in row.index else default
    if pd.isna(value):
        return default
    return float(value)


def _soc_power_bounds(
    soc: float,
    soc_min: float,
    soc_max: float,
    e_batt_kwh: float,
    eta_ch: float,
    eta_dis: float,
    dt_h: float,
) -> tuple[float, float]:
    available_discharge = max(0.0, (soc - soc_min) * e_batt_kwh * eta_dis / max(dt_h, 1e-9))
    available_charge = max(0.0, (soc_max - soc) * e_batt_kwh / max(eta_ch * dt_h, 1e-9))
    return available_discharge, available_charge


class BatteryVPPEnv(gym.Env):
    """Battery-only green data-center VPP dispatch environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        config: Mapping[str, Any] | None = None,
        split: str = "train",
        episode_length: int | None = None,
        seed: int | None = None,
        sequential: bool = False,
        disable_battery: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.battery = _nested(self.config, "battery")
        self.system = _nested(self.config, "system")
        self.reward_weights = _nested(self.config, "reward_weights")
        self.reward_cfg = _nested(self.config, "reward")
        self.env_cfg = _nested(self.config, "environment")
        self.control_cfg = _nested(self.config, "control")
        self.terminal_recovery_cfg = _nested(self.config, "terminal_recovery")
        self.observation_cfg = _nested(self.config, "observation")

        self.dt_h = float(_get(self.env_cfg, "timestep_hours", 0.25))
        self.reward_scale = float(_get(self.reward_cfg, "reward_scale", _get(self.env_cfg, "reward_scale", 1.0)))
        self.episode_length = int(episode_length or _get(self.env_cfg, "episode_length", 96))
        self.split = split
        self.sequential = sequential
        self.disable_battery = disable_battery
        self.control_mode = str(_get(self.control_cfg, "mode", "direct_sac")).lower()
        self.greedy_prior_enabled = bool(_get(self.control_cfg, "greedy_prior_enabled", False))

        self.P_batt_max_kW = float(_get(self.battery, "P_batt_max_kW", 750.0))
        self.E_batt_kWh = float(_get(self.battery, "E_batt_kWh", 4000.0))
        self.SOC_initial = float(_get(self.battery, "SOC_initial", 0.55))
        self.SOC_min = float(_get(self.battery, "SOC_min", 0.2))
        self.SOC_max = float(_get(self.battery, "SOC_max", 0.9))
        self.eta_ch = float(_get(self.battery, "eta_ch", 0.95))
        self.eta_dis = float(_get(self.battery, "eta_dis", 0.95))
        self.degradation_cost = float(_get(self.battery, "degradation_cost_EUR_per_kWh_throughput", 0.02))
        self.residual_power_limit_fraction = float(_get(self.control_cfg, "residual_power_limit_fraction", 0.25))
        self.residual_power_limit_kW = self.residual_power_limit_fraction * self.P_batt_max_kW
        self.greedy_blend_alpha = float(_get(self.control_cfg, "greedy_blend_alpha", 1.0))
        self.include_data_center_features = bool(_get(self.observation_cfg, "include_data_center_features", False))
        self.include_scenario_features = bool(_get(self.observation_cfg, "include_scenario_features", False))

        self.full_df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True).copy()
        self.df = self._select_split(self.full_df, split)
        self.daily_start_indices = self._daily_start_indices(self.df)
        if not self.daily_start_indices:
            raise ValueError(f"No complete {self.episode_length}-step daily episodes found for split '{split}'")

        self._rng = np.random.default_rng(seed)
        self._day_cursor = 0
        self._episode_df = self.df.iloc[: self.episode_length].copy()
        self._step = 0
        self._soc = self.SOC_initial
        self._prev_p_batt = 0.0
        self._prev_grid_import = 0.0
        self._records: list[dict[str, Any]] = []

        self.grid_limit_kW = self._infer_grid_limit(self.full_df)
        normalization_cfg = _nested(self.config, "normalization")
        parameters_path = _get(normalization_cfg, "parameters_path", None)
        embedded_parameters = _get(normalization_cfg, "parameters", None)
        if parameters_path:
            self.normalizers = load_normalizers(parameters_path)
        elif isinstance(embedded_parameters, Mapping):
            self.normalizers = {str(k): float(v) for k, v in embedded_parameters.items()}
        else:
            self.normalizers = fit_training_normalizers(self.full_df, self.grid_limit_kW)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_columns = list(OBSERVATION_COLUMNS)
        if self.control_mode == "residual_sac":
            self.observation_columns.extend(RESIDUAL_OBSERVATION_COLUMNS)
        if self.include_data_center_features:
            self.observation_columns.extend(DATA_CENTER_FEATURE_COLUMNS)
        if self.include_scenario_features:
            self.observation_columns.extend(SCENARIO_FEATURE_COLUMNS)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.observation_columns),),
            dtype=np.float32,
        )

    def _select_split(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        if split.lower() == "all" or "split" not in df.columns:
            out = df.copy()
        else:
            out = df[df["split"].astype(str).str.lower() == split.lower()].copy()
        if out.empty:
            raise ValueError(f"Selected split '{split}' is empty")
        return out.reset_index(drop=True)

    def _daily_start_indices(self, df: pd.DataFrame) -> list[int]:
        ts = pd.to_datetime(df[TIMESTAMP_COLUMN])
        starts: list[int] = []
        for _, idx in ts.groupby(ts.dt.date).groups.items():
            positions = np.asarray(idx, dtype=int)
            if len(positions) >= self.episode_length:
                starts.append(int(positions[0]))
        return starts

    def _infer_grid_limit(self, df: pd.DataFrame) -> float:
        if "grid_import_contract_limit_kW" in df.columns:
            value = pd.to_numeric(df["grid_import_contract_limit_kW"], errors="coerce").dropna()
            if not value.empty:
                return float(value.iloc[0])
        system_value = _get(self.system, "grid_import_contract_limit_kW", None)
        if system_value is not None:
            return float(system_value)
        return float(max(df.get("P_import_without_battery_kW", pd.Series([1.0])).max(), 1.0))

    def _build_normalizers(self, df: pd.DataFrame) -> dict[str, float]:
        return fit_training_normalizers(df, self.grid_limit_kW)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options and "day_index" in options:
            start = self.daily_start_indices[int(options["day_index"]) % len(self.daily_start_indices)]
        elif self.sequential:
            start = self.daily_start_indices[self._day_cursor % len(self.daily_start_indices)]
            self._day_cursor += 1
        else:
            start = int(self._rng.choice(self.daily_start_indices))

        self._episode_df = self.df.iloc[start : start + self.episode_length].copy().reset_index(drop=True)
        self._step = 0
        self._soc = float(options.get("initial_soc", self.SOC_initial)) if options else self.SOC_initial
        self._prev_p_batt = float(options.get("previous_battery_power_kW", 0.0)) if options else 0.0
        self._prev_grid_import = float(options.get("previous_grid_import_kW", 0.0)) if options else 0.0
        self._records = []
        obs = self._get_obs()
        return obs, self.get_action_context()

    def step(self, action):
        row = self._current_row()
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        action_value = float(np.clip(action_value, -1.0, 1.0))
        p_net = _safe_float(row, "P_net_without_battery_kW", _safe_float(row, "P_total_dc_kW") - _safe_float(row, "Ppv_kW"))

        if self.disable_battery:
            p_raw = 0.0
            p_greedy = 0.0
            p_residual = 0.0
            p_safe = 0.0
            soc_next = self._soc
            projection = {
                "P_batt_safe_kW": 0.0,
                "P_batt_physical_kW": 0.0,
                "P_batt_service_coached_kW": 0.0,
                "physical_projection": False,
                "service_coaching": False,
                "hard_clip": False,
                "soft_coach": False,
                "clip_reasons": [],
                "safe_upper_kW": 0.0,
                "safe_lower_kW": 0.0,
            }
        else:
            p_greedy = self._greedy_power(row, p_net) if self.control_mode == "residual_sac" or self.greedy_prior_enabled else 0.0
            if self.control_mode == "residual_sac":
                p_residual = action_value * self.residual_power_limit_kW
                p_raw = self.greedy_blend_alpha * p_greedy + p_residual
            else:
                p_residual = 0.0
                p_raw = action_value * self.P_batt_max_kW
            projection = project_battery_action(
                p_raw,
                self._soc,
                self._prev_p_batt,
                row,
                self.config,
                self.dt_h,
            )
            p_safe = float(projection["P_batt_safe_kW"])
            soc_next = self._next_soc(self._soc, p_safe)

        p_grid = p_net - p_safe
        p_import = max(p_grid, 0.0)
        p_export = max(-p_grid, 0.0)

        terminated = self._step >= self.episode_length - 1
        remaining_steps = max(self.episode_length - self._step - 1, 0)
        reward, components = self._compute_reward(
            row,
            p_raw,
            p_residual,
            p_safe,
            p_grid,
            p_import,
            p_export,
            soc_next,
            terminated,
            remaining_steps,
        )
        info = self._build_step_info(row, p_greedy, p_residual, p_raw, p_safe, p_grid, p_import, p_export, soc_next, reward, components, projection)
        self._records.append(info.copy())

        self._soc = soc_next
        self._prev_p_batt = p_safe
        self._prev_grid_import = p_import
        self._step += 1

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), bool(terminated), False, info

    def _current_row(self) -> pd.Series:
        return self._episode_df.iloc[min(self._step, len(self._episode_df) - 1)]

    def _greedy_power(self, row: pd.Series, p_net_without_battery_kw: float | None = None) -> float:
        p_net = (
            float(p_net_without_battery_kw)
            if p_net_without_battery_kw is not None
            else _safe_float(row, "P_net_without_battery_kW", _safe_float(row, "P_total_dc_kW") - _safe_float(row, "Ppv_kW"))
        )
        return p_net - _safe_float(row, "Pgrid_ref_kW", p_net)

    def _next_soc(self, soc: float, p_batt_kw: float) -> float:
        if p_batt_kw >= 0.0:
            discharge_energy = p_batt_kw * self.dt_h / max(self.eta_dis, 1e-9)
            soc -= discharge_energy / max(self.E_batt_kWh, 1e-9)
        else:
            charge_energy = abs(p_batt_kw) * self.dt_h * self.eta_ch
            soc += charge_energy / max(self.E_batt_kWh, 1e-9)
        return float(soc)

    def _weight(self, alias: str, legacy: str, default: float) -> float:
        new_key_map = {
            "w_tracking": "tracking_weight",
            "w_cost": "cost_weight",
            "w_carbon": "carbon_weight",
            "w_degradation": "degradation_weight",
            "w_peak": "peak_violation_weight",
            "w_dr": "dr_violation_weight",
            "w_soc": "soc_violation_weight",
            "w_terminal_soc": "terminal_soc_weight",
        }
        new_key = new_key_map.get(alias)
        if new_key and new_key in self.reward_cfg:
            return float(self.reward_cfg[new_key])
        if alias in self.reward_cfg:
            return float(self.reward_cfg[alias])
        if alias in self.reward_weights:
            return float(self.reward_weights[alias])
        if legacy in self.reward_weights:
            return float(self.reward_weights[legacy])
        return default

    def _reward_value(self, key: str, default: float) -> float:
        return float(_get(self.reward_cfg, key, default))

    def _tracking_loss(self, tracking_error_kw: float, grid_limit_kw: float) -> float:
        if str(_get(self.reward_cfg, "tracking_loss", "squared")).lower() == "huber":
            delta_kw = self._reward_value("tracking_huber_delta_kW", 100.0)
            abs_error = abs(tracking_error_kw)
            if abs_error <= delta_kw:
                loss_kw = 0.5 * tracking_error_kw**2
            else:
                loss_kw = delta_kw * (abs_error - 0.5 * delta_kw)
            return loss_kw / max(grid_limit_kw**2, 1e-9)
        return (tracking_error_kw / max(grid_limit_kw, 1e-9)) ** 2

    def _terminal_recovery_multiplier(self, remaining_steps: int) -> float:
        if not bool(_get(self.terminal_recovery_cfg, "enabled", False)):
            return 1.0
        window = max(int(_get(self.terminal_recovery_cfg, "window_steps", 24)), 1)
        max_multiplier = float(_get(self.terminal_recovery_cfg, "max_weight_multiplier", 5.0))
        if remaining_steps >= window:
            return 1.0
        progress = (window - remaining_steps) / window
        return 1.0 + progress * (max_multiplier - 1.0)

    def _compute_reward(
        self,
        row: pd.Series,
        p_batt_raw_kw: float,
        p_batt_residual_kw: float,
        p_batt_kw: float,
        p_grid_kw: float,
        p_import_kw: float,
        p_export_kw: float,
        soc: float,
        terminal: bool,
        remaining_steps: int = 0,
    ) -> tuple[float, dict[str, float]]:
        grid_limit = max(_safe_float(row, "grid_import_contract_limit_kW", self.grid_limit_kW), 1e-6)
        price_buy = _safe_float(row, "price_buy_EUR_per_kWh", 0.0)
        price_sell = _safe_float(row, "price_sell_EUR_per_kWh", 0.0)
        carbon = _safe_float(row, "carbon_gCO2_per_kWh_proxy", 0.0)
        p_ref = _safe_float(row, "Pgrid_ref_kW", 0.0)
        tracking_error = p_grid_kw - p_ref

        cost_eur = p_import_kw * price_buy * self.dt_h - p_export_kw * price_sell * self.dt_h
        carbon_kg = p_import_kw * carbon * self.dt_h / 1000.0
        degradation = abs(p_batt_kw) * self.dt_h * self.degradation_cost

        tracking_penalty = self._weight("w_tracking", "vpp_tracking", 1.0) * self._tracking_loss(tracking_error, grid_limit)
        cost_penalty = self._weight("w_cost", "energy_cost", 1.0) * cost_eur
        carbon_penalty = self._weight("w_carbon", "carbon", 0.001) * carbon_kg
        degradation_penalty = self._weight("w_degradation", "battery_wear", 1.0) * degradation
        residual_penalty = 0.0
        if self.control_mode == "residual_sac" and self.residual_power_limit_kW > 0:
            residual_penalty = self._reward_value("residual_action_weight", 0.0) * (
                p_batt_residual_kw / self.residual_power_limit_kW
            ) ** 2
        ramp_penalty = self._reward_value("ramp_weight", 0.0) * abs(p_batt_kw - self._prev_p_batt) / max(self.P_batt_max_kW, 1e-9)
        peak_penalty = 0.0
        if p_import_kw > grid_limit:
            peak_penalty = self._weight("w_peak", "peak_import", 50.0) * ((p_import_kw - grid_limit) / grid_limit) ** 2

        dr_penalty = 0.0
        dr_active = int(_safe_float(row, "DR_active", 0.0)) == 1
        dr_limit = _safe_float(row, "DR_import_limit_kW", _safe_float(row, "DR_import_limit_for_model_kW", np.inf))
        if dr_active and np.isfinite(dr_limit) and dr_limit < 1e8 and p_import_kw > dr_limit:
            dr_penalty = self._weight("w_dr", "demand_response", 100.0) * ((p_import_kw - dr_limit) / grid_limit) ** 2

        soc_penalty = 0.0
        if soc < self.SOC_min or soc > self.SOC_max:
            soc_penalty = self._weight("w_soc", "soc_safety", 100.0) * min(abs(soc - self.SOC_min), abs(soc - self.SOC_max))

        reserve_soc = float(_get(self.battery, "SOC_reserve", 0.3))
        reserve_soc_penalty = 0.0
        if soc < reserve_soc:
            reserve_soc_penalty = self._reward_value("reserve_soc_weight", 8.0) * ((reserve_soc - soc) / max(reserve_soc - self.SOC_min, 1e-6)) ** 2

        soc_low_soft_limit = self._reward_value("soc_low_soft_limit", 0.32)
        soc_high_soft_limit = self._reward_value("soc_high_soft_limit", 0.88)
        soc_low_soft_penalty = 0.0
        if soc < soc_low_soft_limit:
            soc_low_soft_penalty = self._reward_value("soc_low_soft_weight", 5.0) * ((soc_low_soft_limit - soc) / max(soc_low_soft_limit - self.SOC_min, 1e-6)) ** 2

        terminal_recovery_multiplier = self._terminal_recovery_multiplier(remaining_steps)
        soc_target_penalty = self._reward_value("soc_target_weight", 2.0) * terminal_recovery_multiplier * abs(soc - self.SOC_initial)

        action_bound_penalty = 0.0
        near_soc_min = soc <= soc_low_soft_limit
        near_soc_max = soc >= soc_high_soft_limit
        if near_soc_min and p_batt_raw_kw > 0:
            action_bound_penalty += (p_batt_raw_kw / max(self.P_batt_max_kW, 1e-9)) ** 2
        if near_soc_max and p_batt_raw_kw < 0:
            action_bound_penalty += (abs(p_batt_raw_kw) / max(self.P_batt_max_kW, 1e-9)) ** 2
        action_bound_penalty *= self._reward_value("soc_target_weight", 2.0)

        terminal_soc_penalty = 0.0
        if terminal:
            terminal_soc_penalty = self._weight("w_terminal_soc", "terminal_soc", 10.0) * abs(soc - self.SOC_initial)

        components = {
            "cost_EUR": float(cost_eur),
            "carbon_kg_proxy": float(carbon_kg),
            "tracking_error_kW": float(tracking_error),
            "tracking_penalty": float(tracking_penalty),
            "cost_penalty": float(cost_penalty),
            "carbon_penalty": float(carbon_penalty),
            "degradation_penalty": float(degradation_penalty),
            "residual_penalty": float(residual_penalty),
            "ramp_penalty": float(ramp_penalty),
            "peak_penalty": float(peak_penalty),
            "dr_penalty": float(dr_penalty),
            "soc_penalty": float(soc_penalty),
            "reserve_soc_penalty": float(reserve_soc_penalty),
            "soc_low_soft_penalty": float(soc_low_soft_penalty),
            "soc_target_penalty": float(soc_target_penalty),
            "action_bound_penalty": float(action_bound_penalty),
            "terminal_soc_penalty": float(terminal_soc_penalty),
            "terminal_recovery_multiplier": float(terminal_recovery_multiplier),
            "remaining_steps": float(remaining_steps),
        }
        total_penalty = sum(v for k, v in components.items() if k.endswith("_penalty"))
        unscaled_reward = -float(total_penalty)
        reward = unscaled_reward * self.reward_scale
        components["unscaled_reward"] = unscaled_reward
        components["reward_scale"] = float(self.reward_scale)
        return float(reward), components

    def _build_step_info(
        self,
        row: pd.Series,
        p_greedy: float,
        p_residual: float,
        p_raw: float,
        p_safe: float,
        p_grid: float,
        p_import: float,
        p_export: float,
        soc: float,
        reward: float,
        components: dict[str, float],
        projection: dict[str, Any],
    ) -> dict[str, Any]:
        info = {
            "timestamp_utc": row[TIMESTAMP_COLUMN],
            "P_batt_greedy_kW": float(p_greedy),
            "P_batt_residual_kW": float(p_residual),
            "P_batt_raw_kW": float(p_raw),
            "P_batt_commanded_kW": float(p_raw),
            "P_batt_physically_clipped_kW": float(projection["P_batt_physical_kW"]),
            "P_batt_service_coached_kW": float(projection["P_batt_service_coached_kW"]),
            "P_batt_safe_kW": float(p_safe),
            "P_grid_kW": float(p_grid),
            "P_import_kW": float(p_import),
            "P_export_kW": float(p_export),
            "P_total_dc_kW": _safe_float(row, "P_total_dc_kW", np.nan),
            "P_IT_kW": _safe_float(row, "P_IT_kW", np.nan),
            "P_cooling_kW": _safe_float(row, "P_cooling_kW", np.nan),
            "PUE_model": _safe_float(row, "PUE_model", np.nan),
            "ai_utilization": _safe_float(row, "ai_utilization", np.nan),
            "Ppv_kW": _safe_float(row, "Ppv_kW", np.nan),
            "carbon_gCO2_per_kWh_proxy": _safe_float(row, "carbon_gCO2_per_kWh_proxy", np.nan),
            "price_buy_EUR_per_kWh": _safe_float(row, "price_buy_EUR_per_kWh", np.nan),
            "SOC": float(soc),
            "battery_capacity_kWh": float(self.E_batt_kWh),
            "initial_SOC_target": float(self.SOC_initial),
            "Pgrid_ref_kW": _safe_float(row, "Pgrid_ref_kW", 0.0),
            "tracking_error_kW": float(p_grid - _safe_float(row, "Pgrid_ref_kW", 0.0)),
            "DR_active": int(_safe_float(row, "DR_active", 0.0)),
            "DR_import_limit_kW": _safe_float(row, "DR_import_limit_kW", np.nan),
            "grid_import_contract_limit_kW": _safe_float(row, "grid_import_contract_limit_kW", self.grid_limit_kW),
            "safety_clipped": bool(projection["hard_clip"] or projection["soft_coach"]),
            "hard_clip": bool(projection["hard_clip"]),
            "soft_coach": bool(projection["soft_coach"]),
            "physical_projection": bool(projection["physical_projection"]),
            "service_coaching": bool(projection["service_coaching"]),
            "clip_reasons": "|".join(projection["clip_reasons"]),
            "safe_upper_kW": float(projection["safe_upper_kW"]),
            "safe_lower_kW": float(projection["safe_lower_kW"]),
            "greedy_blend_alpha": float(self.greedy_blend_alpha),
            "reward": float(reward),
            "step": int(self._step),
            "scenario_name": str(row.get("scenario_name", "baseline_real")),
            "scenario_family": str(row.get("scenario_family", "baseline")),
        }
        info.update(components)
        return info

    def _norm(self, row: pd.Series, col: str) -> float:
        return _safe_float(row, col, 0.0) / max(self.normalizers.get(col, 1.0), 1e-9)

    def _get_obs(self) -> np.ndarray:
        row = self._current_row()
        grid_limit = max(_safe_float(row, "grid_import_contract_limit_kW", self.grid_limit_kW), 1e-6)
        dr_active = int(_safe_float(row, "DR_active", 0.0)) == 1
        if dr_active:
            dr_limit_norm = _safe_float(row, "DR_import_limit_kW", _safe_float(row, "DR_import_limit_for_model_kW", grid_limit)) / grid_limit
        else:
            dr_limit_norm = 1.0

        values = [
            self._soc,
            self._prev_p_batt / max(self.P_batt_max_kW, 1e-9),
            self._prev_grid_import / grid_limit,
            self._norm(row, "P_total_dc_kW"),
            self._norm(row, "P_IT_kW"),
            self._norm(row, "P_cooling_kW"),
            self._norm(row, "Pdc_forecast_kW"),
            self._norm(row, "Ppv_kW"),
            self._norm(row, "Ppv_forecast_kW"),
            self._norm(row, "price_buy_EUR_per_kWh"),
            self._norm(row, "price_forecast_EUR_per_kWh"),
            self._norm(row, "carbon_gCO2_per_kWh_proxy"),
            _safe_float(row, "renewable_share_proxy", 0.0),
            self._norm(row, "Pgrid_ref_kW"),
            float(dr_active),
            float(np.clip(dr_limit_norm, 0.0, 2.0)),
            self._norm(row, "DR_required_reduction_kW"),
            self._norm(row, "grid_import_contract_limit_kW"),
            self._norm(row, "ambient_temp_C"),
            self._norm(row, "temperature_deviation_C"),
            _safe_float(row, "sin_hour", 0.0),
            _safe_float(row, "cos_hour", 0.0),
            _safe_float(row, "sin_day", 0.0),
            _safe_float(row, "cos_day", 0.0),
        ]
        if self.control_mode == "residual_sac":
            p_net = _safe_float(row, "P_net_without_battery_kW", _safe_float(row, "P_total_dc_kW") - _safe_float(row, "Ppv_kW"))
            p_greedy = self._greedy_power(row, p_net)
            tracking_error_without_battery = p_net - _safe_float(row, "Pgrid_ref_kW", p_net)
            available_discharge, available_charge = _soc_power_bounds(
                self._soc,
                self.SOC_min,
                self.SOC_max,
                self.E_batt_kWh,
                self.eta_ch,
                self.eta_dis,
                self.dt_h,
            )
            values.extend(
                [
                    p_greedy / max(self.P_batt_max_kW, 1e-9),
                    tracking_error_without_battery / grid_limit,
                    min(available_discharge, self.P_batt_max_kW) / max(self.P_batt_max_kW, 1e-9),
                    min(available_charge, self.P_batt_max_kW) / max(self.P_batt_max_kW, 1e-9),
                ]
            )
        if self.include_data_center_features:
            p_it = max(_safe_float(row, "P_IT_kW", 0.0), 1e-9)
            cooling_ratio = _safe_float(row, "P_cooling_kW", 0.0) / p_it
            remaining_steps = max(self.episode_length - self._step - 1, 0)
            values.extend([
                float(np.clip(_safe_float(row, "ai_utilization", 0.0), 0.0, 1.0)),
                self._norm(row, "PUE_model"),
                cooling_ratio / max(self.normalizers.get("cooling_ratio", 1.0), 1e-9),
                remaining_steps / max(self.episode_length - 1, 1),
            ])
        if self.include_scenario_features:
            family = str(row.get("scenario_family", "baseline"))
            values.extend([1.0 if family == name else 0.0 for name in SCENARIO_FAMILIES])
        obs = np.asarray(values, dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def get_action_context(self) -> dict[str, Any]:
        row = self._current_row()
        context = {col: row[col] for col in row.index}
        context.update(
            {
                "SOC": self._soc,
                "prev_P_batt_kW": self._prev_p_batt,
                "prev_grid_import_kW": self._prev_grid_import,
                "P_batt_max_kW": self.P_batt_max_kW,
                "grid_limit_kW": self.grid_limit_kW,
            }
        )
        return context

    def trajectory_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)
