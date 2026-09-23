# Manuscript Evidence Map

| Manuscript section | Claim/topic | Authoritative source file | Exact metric/figure/table |
| --- | --- | --- | --- |
| Introduction evidence | Motivation and tested scope | `MANUSCRIPT_TERMINOLOGY_GUIDE.md`; `FINAL_CLAIM_GUIDE.md` | Qualified benchmark claims only |
| Dataset | Provenance, cadence, split, integrity | `tables/TABLE_DATASET_AND_PROVENANCE.csv`; `DATASET_INTEGRITY_CHECK.txt` | 15 min; 2019 test quarter; SHA256 |
| System model | Battery and grid assumptions | `tables/TABLE_CONTROLLER_DEFINITIONS.csv`; frozen YAML snapshots in `scripts/` | 4 MWh, 750 kW, SOC and ramp limits |
| IT load | Eight workload definitions and characteristics | `tables/TABLE_IT_SCENARIO_CHARACTERISTICS.csv` | load energy, peak, variability, ramps |
| Controller methodology | Direct, residual, proposed and baseline semantics | `tables/TABLE_CONTROLLER_DEFINITIONS.csv`; `verification/FINAL_BASELINE_BUG_DIAGNOSIS.md` | command composition and safety mechanisms |
| Experimental setup | Horizon, seeds, deterministic evaluation | `tables/TABLE_DATASET_AND_PROVENANCE.csv`; `FINAL_REPRODUCIBILITY_REPORT.md` | 92 days, 8832 intervals, learned seeds 1-5 |
| Baseline comparison | Corrected main results | `tables/TABLE_MAIN_CONTROLLER_RESULTS.csv`; Figure 6 | cost, carbon, RMSE, DR, battery and SOC metrics |
| IT-load scenarios | Workload/controller sensitivity | `tables/TABLE_IT_SCENARIO_RESULTS.csv`; `tables/IT_CONTROLLER_INTERACTION_FINAL.csv`; Figure 8 | RMSE, DR compliance, cost, carbon, throughput |
| Seed robustness | Learned-controller uncertainty | `tables/TABLE_SEED_RESULTS.csv`; Figure 7 | five per-seed rows per learned controller |
| Ablation | Contributions of residual prior, safety and terminal recovery | `tables/TABLE_TRUE_ABLATION_RESULTS.csv`; `verification/ABLATION_INTEGRITY_CHECK.csv` | five true learned variants, hashes unchanged |
| Safety | Physical projection and service coaching | `tables/TABLE_SAFETY_RESULTS.csv`; Figure 9 | intervention fractions, DR and SOC metrics |
| Limitations | Hybrid benchmark, finite scenarios, no causal inference | `FINAL_CLAIM_GUIDE.md`; `MANUSCRIPT_TERMINOLOGY_GUIDE.md` | claims 1-10 classifications |
| Conclusion | Defensible summary | `FINAL_CLAIM_GUIDE.md`; `tables/TABLE_MAIN_CONTROLLER_RESULTS.csv` | metric-specific, non-universal statements |
