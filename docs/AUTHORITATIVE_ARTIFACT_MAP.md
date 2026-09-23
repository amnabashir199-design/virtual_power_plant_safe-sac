# Authoritative Artifact Map

Only the artifacts listed here support the final manuscript. Paths are relative to the package root.

| Manuscript result | Final artifact | Script producing/recalculating it | Input data | Status |
|---|---|---|---|---|
| Eight-controller baseline comparison | `tables/TABLE_MAIN_CONTROLLER_RESULTS.csv` | `scripts/reproduce_metrics.py` | `trajectories/baseline/`, `trajectories/learned_controllers/` | FINAL / AUTHORITATIVE |
| Five learned seeds | `tables/TABLE_SEED_RESULTS.csv` | `scripts/evaluate_learned_controllers.py` | 15 learned daily-reset trajectories and checkpoints | FINAL / AUTHORITATIVE |
| IT scenario characteristics | `tables/TABLE_IT_SCENARIO_CHARACTERISTICS.csv` | `scripts/reference/analyze_it_load.py` | eight datasets in `data/processed/scenarios/` | FINAL / AUTHORITATIVE |
| Eight-scenario controller comparison | `tables/TABLE_IT_SCENARIO_RESULTS.csv` | `scripts/evaluate_scenarios.py` | 160 final scenario trajectories | FINAL / AUTHORITATIVE |
| Primary pooled proposed scenario DR values | pooled columns in `tables/TABLE_IT_SCENARIO_RESULTS.csv` | `scripts/reproduction_core.py` | actual DR-active intervals in final trajectories | FINAL / AUTHORITATIVE |
| True ablation comparison | `tables/TABLE_TRUE_ABLATION_RESULTS.csv` | `scripts/evaluate_ablations.py` | 15 unique ablation trajectories plus 10 main-family trajectories | FINAL / AUTHORITATIVE |
| Safety/intervention diagnostics | `tables/TABLE_SAFETY_RESULTS.csv` | `scripts/reproduce_metrics.py` | baseline deterministic and learned trajectories | FINAL / AUTHORITATIVE |
| Corrected deterministic trajectories | `trajectories/baseline/*.parquet` | `scripts/reference/40_evaluate_corrected_baselines.py` | publication dataset and baseline configuration | FINAL / AUTHORITATIVE |
| Learned baseline trajectories | `trajectories/learned_controllers/*.parquet` | `scripts/reference/35_evaluate_publication.py` | publication dataset and selected checkpoints | FINAL / AUTHORITATIVE |
| Workload-scenario trajectories | `trajectories/scenarios/*.parquet` | `scripts/reference/36_evaluate_publication_scenarios.py` and `42_reevaluate_soc_affected_learned.py` | scenario datasets and selected checkpoints | FINAL / AUTHORITATIVE |
| Independent ablation trajectories | `trajectories/ablations/*.parquet` | `scripts/reference/39_evaluate_publication_ablations.py` | publication dataset and ablation checkpoints | FINAL / AUTHORITATIVE |
| Publication dataset | `data/processed/green_dc_vpp_publication_v1_2019_15min.parquet` | `scripts/reference/32_build_publication_dataset.py` | processed hybrid source layers | FINAL / AUTHORITATIVE |
| Scenario definitions/datasets | `data/processed/scenarios/*.parquet` | `scripts/reference/24_build_it_load_scenarios.py` | publication dataset | FINAL / AUTHORITATIVE |
| Learned checkpoints | `models/**/selected_validation_checkpoint.zip` | optional historical training entry point | publication training/validation split | FINAL / AUTHORITATIVE |
| Main figures | `figures/main/` | `scripts/generate_figures.py` | final tables, scenario data, trajectories | FINAL / AUTHORITATIVE |
| Supplementary intervention scatter | `figures/supplementary/FigS1_commanded_applied.*` | `scripts/generate_figures.py` | proposed seed-1 baseline trajectory | FINAL / AUTHORITATIVE |

The unmodified source tables retaining historically mixed DR aggregation are isolated under `results/authoritative/` and explicitly named `SOURCE_*_MIXED_DR_AGGREGATION.csv` where applicable. The learned-only ablation source table retains its original daily-average name. These files support aggregation auditing but are not the primary package tables.
