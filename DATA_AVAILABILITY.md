# Data Availability

The reproducibility package prepared for repository deposition contains the processed 2019 hybrid benchmark used in the final experiments, eight scenario-transformed Parquet datasets, selected model checkpoints, daily-reset trajectories, machine-readable tables, and code.

## Public source data

- Open Power System Data (OPSD) European time-series package, version 2020-10-06: `https://doi.org/10.25832/time_series/2020-10-06`. Germany 2019 grid load, load forecast, wind/solar generation, solar profile, and day-ahead price fields were used.
- The workload component was obtained from an NLR-labeled GenAI whole-facility inference archive supplied to the project. A stable public identifier and redistribution terms were not documented in the working materials and must be supplied or confirmed by the authors.
- The cooling context used `cold_source_control_dataset.csv`; a matching source is visible at `https://gitlab.univ-nantes.fr/E25C590R/datacenter/-/blob/main/cold_source_control_dataset.csv`. The authors must confirm that this is the exact upstream version and review its redistribution terms.

Raw third-party archives are **not** included.

## Derived benchmark data

`data/processed/green_dc_vpp_publication_v1_2019_15min.parquet` is the exact 35,040-row publication dataset. It combines source-based, modeled, scaled, derived, and generated variables. It is included to make the frozen-model evaluation independently reproducible, but it must not be described as fully measured operational-site data. The authors must confirm that public redistribution of the derived rows is permitted before repository deposition.

## Scenario data

Eight datasets under `data/processed/scenarios/` were generated deterministically from the publication dataset. Their exact definitions are documented in `docs/SCENARIO_DEFINITIONS.md`; no external scenario data are required.

## Model checkpoints and trajectories

Thirty selected validation checkpoints are included: five seeds for each of the three main learned families and five seeds for each of three independently trained ablation families. Daily-reset trajectories needed to recompute the final baseline, scenario, ablation, safety, and seed results are included.

## Code

Evaluation, metric, plotting, environment, baseline, safety, and scenario code is included. The package has not yet been uploaded to a public repository; no public-access claim or persistent repository identifier is made.
