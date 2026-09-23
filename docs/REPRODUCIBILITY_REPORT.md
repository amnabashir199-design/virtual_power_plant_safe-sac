# Reproducibility Report

## Scope

This package reproduces the final manuscript evidence without retraining. It validates the publication dataset, verifies every selected model checksum and network dimension, independently recalculates metrics from daily-reset trajectories, regenerates figures, and compares reproduced values with the authoritative tables.

## Evidence inventory

- 1 publication dataset and 8 derived scenario datasets.
- 30 selected checkpoints: 15 main learned models and 15 independent ablation models.
- 5 corrected deterministic baseline trajectories.
- 15 learned baseline trajectories.
- 160 scenario trajectories covering 8 controllers × 8 scenarios with five seeds for each learned family.
- 15 non-duplicate unique ablation trajectories; the two full-family ablation rows reference main learned trajectories.
- 6 final machine-readable tables.
- 9 main figure families and 1 supplementary figure family, each supplied as PNG and PDF.

## Primary workflow

`python scripts/run_all_reproduction.py`

The workflow is read-only with respect to data, models, trajectories, and frozen manuscript figures. It writes recalculated artifacts under `results/reproduced/`, refreshes the explicitly labeled machine-readable tables under `tables/`, and writes the comparison to `results/REPRODUCTION_VALIDATION.csv`.

## Consistency rules

- Primary scenario/ablation DR compliance pools actual DR-active intervals.
- DR violation energy and maximum violation are full-quarter quantities where labeled as such.
- Daily-average DR columns are retained only as explicitly named secondary diagnostics.
- The high-IT multiplier is verified directly from scenario data as 1.25.
- Learned-controller DR performance is interpreted as full-stack performance, including service coaching.
- No model training, scenario change, or hyperparameter modification occurs.
- Unmodified source tables with historically mixed DR aggregation are retained under `results/authoritative/`; the primary package tables never compare unlike aggregation bases.

## Expected result

The verified run produced 1,249/1,249 passing numerical comparisons, loaded 30/30 selected checkpoints, and regenerated 20 figure files. The package regression suite produced 30/30 passing tests from the package root without reading the parent working repository.
