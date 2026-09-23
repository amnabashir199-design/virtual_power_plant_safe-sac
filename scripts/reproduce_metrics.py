from __future__ import annotations

import json

from reproduction_core import run_metric_reproduction


if __name__ == "__main__":
    print(json.dumps(run_metric_reproduction(load_models=False), indent=2))
