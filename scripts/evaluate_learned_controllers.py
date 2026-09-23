from __future__ import annotations

from reproduction_core import evaluate_learned_controllers, write_model_manifest


if __name__ == "__main__":
    models = write_model_manifest(load_models=True)
    metrics, _ = evaluate_learned_controllers()
    print(f"Loaded {len(models)} final checkpoints (main and ablation families).")
    print(metrics.to_string(index=False))
