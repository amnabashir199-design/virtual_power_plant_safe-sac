"""One-command reproduction from frozen checkpoints and daily-reset trajectories."""

from __future__ import annotations

import json

from generate_figures import generate_all_figures
from reproduction_core import run_metric_reproduction


if __name__ == "__main__":
    report = run_metric_reproduction(load_models=True)
    figures = generate_all_figures()
    report["figure_files_generated"] = len(figures)
    report["passed"] = bool(report["passed"] and len(figures) == 20)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
