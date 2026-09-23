from __future__ import annotations

from reproduction_core import evaluate_scenarios


if __name__ == "__main__":
    metrics, _ = evaluate_scenarios()
    print(metrics.to_string(index=False))
