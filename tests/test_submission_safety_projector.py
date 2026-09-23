"""Publication-freeze regression tests for the final projector hierarchy."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from green_dc_vpp.safety import project_battery_action


PARAMS = {
    "battery": {
        "E_batt_kWh": 4000.0,
        "P_batt_max_kW": 750.0,
        "SOC_min": 0.2,
        "SOC_max": 0.9,
        "SOC_reserve": 0.3,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
        "ramp_limit_kW_per_15min": 300.0,
    },
    "system": {"grid_import_contract_limit_kW": 1.0e9},
    "safety_layer": {
        "enabled": True,
        "enforce_power_limit": True,
        "enforce_soc_bounds": True,
        "enforce_soc_reserve": False,
        "enforce_ramp_limit": True,
        "enforce_dr_limit": False,
        "enforce_grid_limit": False,
    },
}

ROW = {
    "P_net_without_battery_kW": 0.0,
    "grid_import_contract_limit_kW": 1.0e9,
    "DR_active": 0,
}


def project(raw: float, soc: float, previous: float):
    return project_battery_action(raw, soc, previous, ROW, PARAMS, 0.25)


def test_normal_overlap_uses_intersection_of_physical_and_ramp_bounds() -> None:
    result = project(raw=200.0, soc=0.55, previous=0.0)
    assert result["P_batt_safe_kW"] == 200.0
    assert result["safe_lower_kW"] == -300.0
    assert result["safe_upper_kW"] == 300.0
    assert "ramp_overridden_for_hard_feasibility" not in result["clip_reasons"]


def test_soc_limited_discharge_is_clipped_to_hard_upper_bound() -> None:
    result = project(raw=500.0, soc=0.21, previous=0.0)
    expected = (0.21 - 0.2) * 4000.0 * 0.95 / 0.25
    assert math.isclose(result["P_batt_safe_kW"], expected, abs_tol=1e-9)
    assert "soc_min" in result["clip_reasons"]


def test_soc_limited_charge_is_clipped_to_hard_lower_bound() -> None:
    result = project(raw=-500.0, soc=0.89, previous=0.0)
    expected = -(0.9 - 0.89) * 4000.0 / (0.95 * 0.25)
    assert math.isclose(result["P_batt_safe_kW"], expected, abs_tol=1e-9)
    assert "soc_max" in result["clip_reasons"]


def test_ramp_corridor_entirely_above_hard_interval_uses_hard_upper_bound() -> None:
    result = project(raw=500.0, soc=0.2, previous=600.0)
    assert result["safe_lower_kW"] == 0.0
    assert result["safe_upper_kW"] == 0.0
    assert result["P_batt_safe_kW"] == 0.0
    assert "ramp_overridden_for_hard_feasibility" in result["clip_reasons"]


def test_ramp_corridor_entirely_below_hard_interval_uses_hard_lower_bound() -> None:
    result = project(raw=-500.0, soc=0.9, previous=-600.0)
    assert result["safe_lower_kW"] == 0.0
    assert result["safe_upper_kW"] == 0.0
    assert result["P_batt_safe_kW"] == 0.0
    assert "ramp_overridden_for_hard_feasibility" in result["clip_reasons"]
