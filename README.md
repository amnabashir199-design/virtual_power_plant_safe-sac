# GREEN DATA CENTER-BASED VIRTUAL POWER PLANT OPERATION, ENERGY MANAGEMENT USING REINFORCEMENT LEARNING: AN IT-WORKLOAD-AWARE RESIDUAL SAC APPROACH

Authors: **Amna** and **Tao Chen**

This repository-deposition package contains the code, processed hybrid benchmark, selected model checkpoints, daily-reset trajectories, machine-readable results, and figures supporting the final manuscript. It is a computational reproducibility package, not a new experiment and not an operational data-center dataset.

## Study overview

The benchmark represents a green data-center virtual power plant (VPP) comprising IT demand, aggregate cooling and auxiliary demand, 1,000 kWp photovoltaic generation, a 4 MWh/750 kW battery energy storage system (BESS), and grid interaction. Battery dispatch is evaluated for cost, a carbon proxy, VPP-reference tracking, demand-response (DR) service, throughput, terminal state of charge, and intervention frequency.

The proposed controller uses a greedy VPP-tracking prior plus a bounded Soft Actor-Critic (SAC) residual. The resulting battery command passes through explicit physical projection and a separate DR/grid service-coaching stage. Runtime projection enforces the modeled constraints but is not a formal safe-RL proof.

## Main study design

- Chronological split: January–June 2019 training, July–September validation, October–December testing.
- Resolution: 15 minutes; 96-step daily episodes.
- Held-out evaluation: 92 test days with daily SOC reset to 0.55.
- Deterministic controllers: no battery, rule-based, TOU self-consumption, carbon-aware, and greedy tracking.
- Learned families: Direct Safe-SAC, Residual SAC, and Proposed Residual Safe-SAC.
- Five independently trained seeds per learned family.
- Eight workload-centered scenarios.
- Three independently trained ablation families plus the two relevant full-controller families.
- Primary DR statistic: pooled compliance over actual DR-active intervals.

## Quick reproduction

Create an environment, install the dependencies, and run:

```bash
python scripts/validate_data.py
python scripts/run_all_reproduction.py
```

The primary workflow loads all 30 selected checkpoints, replays the included daily-reset trajectories, recalculates the final metrics, regenerates figures, and compares the outputs with the authoritative tables. It does **not** retrain a model. Platform-specific setup commands and expected runtime outputs are in [RUN_REPRODUCTION.md](RUN_REPRODUCTION.md).

## Expected outputs

Authoritative machine-readable tables are under `tables/`:

- `TABLE_MAIN_CONTROLLER_RESULTS.csv`
- `TABLE_IT_SCENARIO_CHARACTERISTICS.csv`
- `TABLE_IT_SCENARIO_RESULTS.csv`
- `TABLE_TRUE_ABLATION_RESULTS.csv`
- `TABLE_SEED_RESULTS.csv`
- `TABLE_SAFETY_RESULTS.csv`

Nine main figures and one supplementary figure are under `figures/`. Recalculated outputs are written to `results/reproduced/`; the final comparison is `results/REPRODUCTION_VALIDATION.csv`.

## Package contents

- `data/`: publication dataset, scenario datasets, provenance, units, and split metadata.
- `src/green_dc_vpp/`: final environment, baseline, safety, metrics, and utility implementation.
- `configs/`: portable evaluation configurations and normalization parameters.
- `models/`: 15 main and 15 independent-ablation selected checkpoints.
- `trajectories/`: daily-reset deterministic, learned, scenario, and non-duplicate ablation trajectories.
- `scripts/`: one-command reproduction and task-specific entry points; exact historical evaluation scripts are retained under `scripts/reference/` for method transparency.
- `docs/`: protocol, controller/scenario/metric definitions, provenance, limitations, security audit, and artifact map.
- `tests/`: relevant final regression tests.

## Data provenance

The included processed benchmark combines source-based, modeled, derived, and generated signals. Raw third-party archives are not redistributed. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md), [data/README.md](data/README.md), and [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md). The submitting authors must confirm redistribution terms for the derived benchmark before public upload.

## Computational environment

The completed project used Python 3.10.11. Tested dependency versions are recorded in `requirements.txt` and `environment.yml`.
