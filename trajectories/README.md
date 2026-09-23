# Trajectory Organization

- `baseline/`: five corrected deterministic baseline trajectories.
- `learned_controllers/`: 15 baseline learned trajectories (three families × five seeds).
- `scenarios/`: 160 trajectories covering five deterministic controllers and three five-seed learned families across eight scenarios. Affected low-IT learned trajectories use the final repaired versions.
- `ablations/`: 15 unique independently trained ablation trajectories. Residual SAC and Proposed Residual Safe-SAC ablation rows reuse `learned_controllers/` and are not duplicated.

All files contain 8,832 intervals representing 92 daily-reset test episodes. Parquet preserves types and keeps the package smaller than CSV. Columns required for independent metric calculation include timestamps, grid/import/export power, battery power, IT/cooling/PV signals, price, carbon proxy, SOC, VPP reference, DR flags/limits, projection/coaching flags, reward, and interval cost.
