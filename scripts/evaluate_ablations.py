from __future__ import annotations

from reproduction_core import evaluate_ablations


if __name__ == "__main__":
    metrics, _ = evaluate_ablations()
    print(metrics.to_string(index=False))
