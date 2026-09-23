# Model Checkpoints

`MODEL_MANIFEST.csv` lists every selected checkpoint, seed, training step count, selected validation step, configuration snapshot, observation/action dimension, and SHA256.

- `direct_safesac/`: seeds 1–5.
- `residual_sac/`: seeds 1–5.
- `proposed_residual_safesac/`: seeds 1–5.
- `ablations/direct_sac_no_safety/`: seeds 1–5.
- `ablations/residual_prior_no_safety/`: seeds 1–5.
- `ablations/proposed_no_terminal_recovery/`: seeds 1–5.

Only `selected_validation_checkpoint.zip` is included for each run. Latest/intermediate/backup models and training-monitor logs are excluded. Per-run configuration snapshots, run metadata, and validation checkpoint-selection tables are retained.
