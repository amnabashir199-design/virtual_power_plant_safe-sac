# Demand-Response Aggregation Resolution

## Audit finding

The frozen source evidence contained two legacy tables whose generic DR column names represented different aggregation bases in different row groups:

- the main/safety source tables stored mean-daily violation energy and mean-daily maximum for deterministic controllers, but full-quarter quantities for learned controllers;
- the scenario source table stored full-quarter quantities for deterministic controllers, but daily-average quantities for learned controllers.

The trajectories and metric implementation were internally consistent. The problem was confined to ambiguous summary-column assembly and labeling. The original values were not silently changed or discarded.

## Resolution used in this package

The unmodified evidence is preserved as:

- `results/authoritative/SOURCE_TABLE_MAIN_CONTROLLER_RESULTS_MIXED_DR_AGGREGATION.csv`;
- `results/authoritative/SOURCE_TABLE_SAFETY_RESULTS_MIXED_DR_AGGREGATION.csv`;
- `results/authoritative/SOURCE_TABLE_IT_SCENARIO_RESULTS_MIXED_DR_AGGREGATION.csv`.

The primary tables under `tables/` are regenerated from packaged daily-reset trajectories and expose separate fields for:

- mean daily DR interval/event compliance;
- pooled DR-active interval/event compliance;
- mean daily DR violation energy;
- full-quarter DR violation energy;
- mean daily maximum DR violation;
- full-quarter maximum DR violation.

The validation ledger also checks every legacy mixed cell against its actual historical aggregation basis. This preserves traceability while ensuring that no publication-facing package column mixes daily and full-horizon quantities.

## Author review before journal submission

The authors should verify that the manuscript table labels match these explicit definitions, especially any table reporting the legacy deterministic and learned DR values together. The numerical source values are traceable and reproducible; this review concerns labeling and aggregation disclosure, not a new experiment or changed controller result.
