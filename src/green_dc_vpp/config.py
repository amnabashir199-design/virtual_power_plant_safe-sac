from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Small dictionary-like config with attribute access for convenience."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return value

    def copy(self) -> "Config":
        return Config(super().copy())


def _to_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return Config({k: _to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config(v) for v in value]
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def _load_assumptions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Assumptions file must contain a JSON object: {path}")
    return data


def load_config(config_path: str | Path = "configs/default.yaml") -> Config:
    """Load YAML config and merge battery/system assumptions when available."""

    config_path = Path(config_path)
    cfg = _load_yaml(config_path)
    project_root = config_path.resolve().parents[1] if config_path.parent.name == "configs" else Path.cwd()

    assumptions_rel = cfg.get("paths", {}).get("assumptions_json")
    if assumptions_rel:
        assumptions_path = (project_root / assumptions_rel).resolve()
    else:
        assumptions_path = project_root / "docs" / "battery_and_system_assumptions.json"

    assumptions = _load_assumptions(assumptions_path)
    assumption_overlay: dict[str, Any] = {}
    if "battery" in assumptions:
        assumption_overlay["battery"] = assumptions["battery"]

    system_keys = [
        "pv_capacity_kWp",
        "other_auxiliary_load_fraction_of_IT",
        "grid_import_contract_limit_kW",
        "dispatch_resolution_minutes",
    ]
    system_overlay = {key: assumptions[key] for key in system_keys if key in assumptions}
    if system_overlay:
        assumption_overlay["system"] = system_overlay

    merged = _deep_merge(cfg, assumption_overlay)
    merged.setdefault("metadata", {})
    merged["metadata"]["config_path"] = str(config_path)
    merged["metadata"]["assumptions_path"] = str(assumptions_path)
    merged["metadata"]["assumptions_loaded"] = bool(assumptions)
    return _to_config(merged)
