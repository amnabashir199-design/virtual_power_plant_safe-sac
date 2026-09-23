from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.config import load_config
from green_dc_vpp.envs import BatteryVPPEnv


def _load_df() -> pd.DataFrame:
    parquet = ROOT / "data" / "processed" / "green_dc_vpp_publication_v1_2019_15min.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(ROOT / "data" / "green_dc_vpp_sample_7days.csv", parse_dates=["timestamp_utc"])


def test_env_reset_and_step() -> None:
    cfg = load_config(ROOT / "configs" / "baselines" / "evaluation_config.yaml")
    env = BatteryVPPEnv(_load_df(), config=cfg, split="all", sequential=True)

    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape

    obs, reward, terminated, truncated, info = env.step(np.asarray([0.0], dtype=np.float32))
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert "P_grid_kW" in info
    assert "SOC" in info
    assert not truncated
    assert isinstance(terminated, bool)
