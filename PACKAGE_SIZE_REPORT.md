# Package Size Report

Sizes are uncompressed binary mebibytes (MiB, 1 MiB = 1,048,576 bytes). The final checksum/manifest metadata contributes less than 1 MiB.

| Component | Size |
|---|---:|
| Complete package | approximately 590.5 MiB |
| Publication Parquet dataset | 5.746 MiB |
| Eight scenario datasets | 47.244 MiB |
| Data directory including metadata | 53.012 MiB |
| Thirty selected models and metadata | 94.015 MiB |
| All trajectories | 431.748 MiB |
| Baseline trajectories | 8.091 MiB |
| Learned-controller trajectories | 35.899 MiB |
| Scenario trajectories | 352.705 MiB |
| Unique ablation trajectories | 35.052 MiB |
| Frozen manuscript figures | 3.231 MiB |
| Regenerated validation figures | 3.561 MiB |
| Source implementation (`src/`) | 0.068 MiB |
| Reproduction and reference scripts | 0.211 MiB |

## Repository-limit options

The scenario trajectories are the only dominant optional transfer component. If a repository imposes a per-file or total-deposit limit, they may be deposited as a separately versioned companion archive while retaining their checksum paths and citation beside the core package. Removing them from the core archive would prevent complete independent replay of all eight scenario tables, so the split must be documented and both deposits must remain linked.

Model checkpoints and the publication/scenario datasets are required for the advertised frozen-model workflow and should not be removed from the reproducibility deposit.
