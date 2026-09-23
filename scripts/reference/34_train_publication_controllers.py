from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.config import load_config
from green_dc_vpp.envs import BatteryVPPEnv
from green_dc_vpp.metrics import compute_trajectory_metrics

# Small MLP updates are substantially faster and more reproducible on this
# Windows CPU when PyTorch does not create a large intra-op thread pool.
torch.set_num_threads(1)

CONFIGS = {
    "direct_safesac": ROOT / "configs/publication_v1/direct_safesac.yaml",
    "residual_sac": ROOT / "configs/publication_v1/residual_sac.yaml",
    "proposed_residual_safesac": ROOT / "configs/publication_v1/proposed_residual_safesac.yaml",
    "direct_sac_no_safety": ROOT / "configs/publication_v1/direct_safesac.yaml",
    "residual_prior_no_safety": ROOT / "configs/publication_v1/residual_sac.yaml",
    "proposed_no_terminal_recovery": ROOT / "configs/publication_v1/proposed_residual_safesac.yaml",
}
VARIANT_OVERRIDES = {
    "direct_sac_no_safety": {"safety_layer": {"enabled": False}},
    "residual_prior_no_safety": {"safety_layer": {"enabled": False}},
    "proposed_no_terminal_recovery": {"terminal_recovery": {"enabled": False}},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_validation(model: SAC, df: pd.DataFrame, cfg) -> dict[str, float]:
    env = BatteryVPPEnv(df, cfg, split="validation", sequential=True)
    rows = []
    for day_index in range(len(env.daily_start_indices)):
        obs, _ = env.reset(options={"day_index": day_index})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        rows.append(compute_trajectory_metrics(env.trajectory_dataframe(), cfg))
    metrics = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
    return {str(k): float(v) for k, v in metrics.items()}


class PublicationCheckpointCallback(BaseCallback):
    def __init__(self, df, cfg, interval: int, best_path: Path, log_path: Path):
        super().__init__()
        self.df = df
        self.cfg = cfg
        self.interval = interval
        self.best_path = best_path
        self.log_path = log_path
        self.best_score = float("inf")
        self.best_step = 0
        self.rows: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.interval:
            return True
        metrics = evaluate_validation(self.model, self.df, self.cfg)
        score = metrics["objective_value"]
        row = {"training_step": self.num_timesteps, "selection_score_mean_daily_objective": score, **metrics}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.log_path, index=False)
        if score < self.best_score:
            self.best_score = score
            self.best_step = self.num_timesteps
            self.model.save(self.best_path)
        print(f"validation step={self.num_timesteps} score={score:.6f} best_step={self.best_step}", flush=True)
        return True


def train(controller: str, seed: int, timesteps: int | None = None) -> Path:
    config_path = CONFIGS[controller]
    cfg = load_config(config_path)
    for section, values in VARIANT_OVERRIDES.get(controller, {}).items():
        cfg.setdefault(section, {}).update(values)
    total = int(timesteps or cfg.training.total_timesteps)
    dataset_path = ROOT / cfg.paths.processed_parquet
    df = pd.read_parquet(dataset_path)
    run_dir = ROOT / "results/publication_v1/models" / controller / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "selected_validation_checkpoint"
    latest_path = run_dir / "latest_model"
    monitor_base = run_dir / "training_monitor"
    validation_log = run_dir / "validation_checkpoints.csv"
    snapshot = run_dir / "config_snapshot.yaml"
    plain_cfg = json.loads(json.dumps(cfg))
    snapshot.write_text(yaml.safe_dump(plain_cfg, sort_keys=False), encoding="utf-8")

    set_random_seed(seed)
    env = Monitor(BatteryVPPEnv(df, cfg, split="train", seed=seed), filename=str(monitor_base))
    callback = PublicationCheckpointCallback(
        df, cfg, int(cfg.training.checkpoint_interval), best_path, validation_log
    )
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=float(cfg.training.learning_rate),
        buffer_size=int(cfg.training.buffer_size),
        batch_size=int(cfg.training.batch_size),
        gamma=float(cfg.training.gamma),
        tau=float(cfg.training.tau),
        train_freq=int(cfg.training.train_freq),
        gradient_steps=int(cfg.training.gradient_steps),
        learning_starts=int(cfg.training.learning_starts),
        ent_coef=cfg.training.ent_coef,
        device=cfg.training.device,
        seed=seed,
        verbose=0,
    )
    started = time.time()
    model.learn(total_timesteps=total, callback=callback, progress_bar=False)
    model.save(latest_path)
    if not best_path.with_suffix(".zip").exists():
        model.save(best_path)
        callback.best_step = total
    best_zip = best_path.with_suffix(".zip")
    latest_zip = latest_path.with_suffix(".zip")
    metadata = {
        "controller": controller,
        "seed": seed,
        "training_steps": total,
        "train_frequency_environment_steps": int(cfg.training.train_freq),
        "gradient_steps_per_update": int(cfg.training.gradient_steps),
        "planned_gradient_updates_after_learning_starts": int(
            max(total - int(cfg.training.learning_starts), 0)
            // int(cfg.training.train_freq)
            * int(cfg.training.gradient_steps)
        ),
        "selected_checkpoint_step": callback.best_step,
        "selection_rule": "minimum mean daily validation objective across all 92 validation days",
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "config_snapshot_sha256": sha256(snapshot),
        "dataset_path": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": sha256(dataset_path),
        "model_path": str(best_zip.relative_to(ROOT)),
        "model_sha256": sha256(best_zip),
        "latest_model_path": str(latest_zip.relative_to(ROOT)),
        "latest_model_sha256": sha256(latest_zip),
        "observation_dimension": int(model.observation_space.shape[0]),
        "action_dimension": int(model.action_space.shape[0]),
        "elapsed_seconds": time.time() - started,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return best_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=[*CONFIGS, "all"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    controllers = list(CONFIGS) if args.controller == "all" else [args.controller]
    for controller in controllers:
        for seed in args.seeds:
            train(controller, seed, args.timesteps)


if __name__ == "__main__":
    main()
