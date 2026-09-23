from __future__ import annotations

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
        "ramp_limit_kW_per_15min": 1000.0,
    },
    "system": {"grid_import_contract_limit_kW": 700.0},
    "safety_layer": {"enforce_soc_bounds": True, "enforce_ramp_limit": True, "enforce_soc_reserve": False},
}


def test_power_limit_clipping() -> None:
    row = {"P_total_dc_kW": 500.0, "Ppv_kW": 0.0, "DR_active": 0}
    out = project_battery_action(2000.0, 0.55, 0.0, row, PARAMS, 0.25)
    assert out["P_batt_safe_kW"] <= 750.0
    assert out["hard_clip"]


def test_soc_min_prevents_excessive_discharge() -> None:
    row = {"P_total_dc_kW": 500.0, "Ppv_kW": 0.0, "DR_active": 0}
    out = project_battery_action(750.0, 0.2, 0.0, row, PARAMS, 0.25)
    assert out["P_batt_safe_kW"] == 0.0
    assert "soc_min" in out["clip_reasons"]


def test_soc_max_prevents_excessive_charge() -> None:
    row = {"P_total_dc_kW": 500.0, "Ppv_kW": 0.0, "DR_active": 0}
    out = project_battery_action(-750.0, 0.9, 0.0, row, PARAMS, 0.25)
    assert out["P_batt_safe_kW"] == 0.0
    assert "soc_max" in out["clip_reasons"]


def test_dr_active_prevents_harmful_charging() -> None:
    row = {"P_total_dc_kW": 500.0, "Ppv_kW": 0.0, "DR_active": 1, "DR_import_limit_kW": 450.0}
    out = project_battery_action(-300.0, 0.55, 0.0, row, PARAMS, 0.25)
    assert out["P_batt_safe_kW"] >= 0.0
    assert "dr_no_charge" in out["clip_reasons"]
