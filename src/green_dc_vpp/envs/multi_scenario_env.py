from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from green_dc_vpp.data_loader import load_dataset
from green_dc_vpp.preprocessing import add_time_features, ensure_forecast_columns
from green_dc_vpp.envs.battery_vpp_env import BatteryVPPEnv


class MultiScenarioBatteryVPPEnv(gym.Env):
    """Sample complete daily episodes across explicit IT-load scenarios."""

    metadata = {"render_modes": []}

    def __init__(self, scenario_dir: str | Path, scenario_names: Sequence[str],
                 config: Mapping[str, Any] | None = None, split: str = "train",
                 seed: int | None = None, sequential: bool = False,
                 fixed_scenario: str | None = None) -> None:
        super().__init__()
        if not scenario_names:
            raise ValueError("scenario_names must not be empty")
        self.scenario_dir = Path(scenario_dir)
        self.scenario_names = list(dict.fromkeys(str(name) for name in scenario_names))
        self.config = config or {}
        self.split = split
        self.sequential = sequential
        self.fixed_scenario = fixed_scenario
        if fixed_scenario is not None and fixed_scenario not in self.scenario_names:
            raise ValueError(f"fixed_scenario '{fixed_scenario}' is not in scenario_names")
        self._rng = np.random.default_rng(seed)
        self._scenario_cursor = 0
        self.envs: dict[str, BatteryVPPEnv] = {}
        for offset, name in enumerate(self.scenario_names):
            path = self.scenario_dir / f"green_dc_vpp_2019_15min_{name}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Scenario dataset not found: {path}")
            frame = ensure_forecast_columns(add_time_features(load_dataset(path)))
            self.envs[name] = BatteryVPPEnv(
                frame, config=self.config, split=split,
                seed=None if seed is None else seed + offset, sequential=sequential,
            )
        first = self.envs[self.scenario_names[0]]
        if any(env.observation_space.shape != first.observation_space.shape for env in self.envs.values()):
            raise ValueError("All scenario environments must expose the same observation shape")
        self.action_space = first.action_space
        self.observation_space = first.observation_space
        self.observation_columns = first.observation_columns
        self.current_scenario_name = fixed_scenario or self.scenario_names[0]
        self.current_env = self.envs[self.current_scenario_name]

    def _select_scenario(self, requested: str | None = None) -> str:
        if requested is not None:
            if requested not in self.envs:
                raise ValueError(f"Unknown scenario '{requested}'. Available: {self.scenario_names}")
            return requested
        if self.fixed_scenario is not None:
            return self.fixed_scenario
        if self.sequential:
            name = self.scenario_names[self._scenario_cursor % len(self.scenario_names)]
            self._scenario_cursor += 1
            return name
        return str(self._rng.choice(self.scenario_names))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        child_options = dict(options or {})
        requested = child_options.pop("scenario_name", None)
        self.current_scenario_name = self._select_scenario(requested)
        self.current_env = self.envs[self.current_scenario_name]
        if "day_index" not in child_options and not self.sequential:
            child_options["day_index"] = int(self._rng.integers(len(self.current_env.daily_start_indices)))
        obs, info = self.current_env.reset(seed=seed, options=child_options)
        info = dict(info)
        info["scenario_name"] = self.current_scenario_name
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info = dict(info)
        info["scenario_name"] = self.current_scenario_name
        return obs, reward, terminated, truncated, info

    def get_action_context(self) -> dict[str, Any]:
        context = self.current_env.get_action_context()
        context["scenario_name"] = self.current_scenario_name
        return context

    def trajectory_dataframe(self):
        trajectory = self.current_env.trajectory_dataframe()
        trajectory["scenario_name"] = self.current_scenario_name
        return trajectory

    @property
    def daily_start_indices(self):
        return self.current_env.daily_start_indices

    def close(self) -> None:
        for env in self.envs.values():
            env.close()

