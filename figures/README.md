# Figure Map

| Figure | Packaged files | Regeneration script | Primary data |
|---|---|---|---|
| Fig. 1 — VPP system architecture | `main/Fig01_system_architecture.*` | `scripts/generate_figures.py` | documented system parameters |
| Fig. 2 — Proposed controller architecture | `main/Fig02_controller_architecture.*` | `scripts/generate_figures.py` | controller definitions/configurations |
| Fig. 3 — Scenario profiles | `main/Fig03_scenario_profiles.*` | `scripts/generate_figures.py` | `data/processed/scenarios/*.parquet` |
| Fig. 4 — Workload characteristics | `main/Fig04_workload_characteristics.*` | `scripts/generate_figures.py` | `tables/TABLE_IT_SCENARIO_CHARACTERISTICS.csv` |
| Fig. 5 — Main controller comparison | `main/Fig05_main_controller_comparison.*` | `scripts/generate_figures.py` | `tables/TABLE_MAIN_CONTROLLER_RESULTS.csv` |
| Fig. 6 — Learned workload sensitivity | `main/Fig06_learned_workload_sensitivity.*` | `scripts/generate_figures.py` | `tables/TABLE_IT_SCENARIO_RESULTS.csv` |
| Fig. 7 — Representative DR response | `main/Fig07_representative_dr_response.*` | `scripts/generate_figures.py` | proposed seed-1 baseline trajectory |
| Fig. 8 — BESS dispatch and SOC | `main/Fig08_soc_and_battery_dispatch.*` | `scripts/generate_figures.py` | proposed seed-1 baseline trajectory |
| Fig. 9 — Seed stability | `main/Fig09_multiseed_stability.*` | `scripts/generate_figures.py` | `tables/TABLE_SEED_RESULTS.csv` |
| Supplementary Fig. S1 — Commanded/applied action | `supplementary/FigS1_commanded_applied.*` | `scripts/generate_figures.py` | proposed seed-1 projection/coaching flags |

PNG files are publication-resolution raster assets; PDF files provide vector output where applicable. Regenerated files are written under `results/reproduced/figures/` and do not overwrite these final manuscript assets.
