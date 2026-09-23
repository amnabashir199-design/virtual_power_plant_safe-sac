from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    required = [
        "TABLE_MAIN_CONTROLLER_RESULTS_REPRODUCED.csv",
        "TABLE_SEED_RESULTS_REPRODUCED.csv",
        "TABLE_IT_SCENARIO_RESULTS_REPRODUCED.csv",
        "TABLE_TRUE_ABLATION_RESULTS_REPRODUCED.csv",
    ]
    output = ROOT / "results" / "reproduced"
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise SystemExit(
            "Run `python scripts/reproduce_metrics.py` first; missing: " + ", ".join(missing)
        )
    print(f"Reproduced machine-readable tables are available under {output.relative_to(ROOT)}")
