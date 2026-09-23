# Data Directory

## Included processed data

- `processed/green_dc_vpp_publication_v1_2019_15min.parquet`: exact 2019 publication dataset; 35,040 fifteen-minute UTC rows.
- `processed/scenarios/*.parquet`: eight exact workload-centered scenario datasets.
- `metadata/dataset_metadata.json`: dataset checksum, split dates, platform, and integrity summary.
- `metadata/data_dictionary.csv`: variable definitions and units.
- `metadata/source_provenance.csv`: source-layer classification and processing notes.
- `metadata/data_quality_summary.csv`: descriptive and missingness checks.
- `metadata/battery_and_system_assumptions.json`: BESS/PV assumptions.
- `scenario_definitions/IT_LOAD_SCENARIOS_SOURCE.md`: scenario-construction source documentation.

## Chronological split

| Partition | Dates | Rows |
|---|---|---:|
| Training | 2019-01-01 through 2019-06-30 | 17,376 |
| Validation | 2019-07-01 through 2019-09-30 | 8,832 |
| Test | 2019-10-01 through 2019-12-31 | 8,832 |

All timestamps are UTC and the interval length is 15 minutes.

## Classification

- External/source-based: IT profile, grid load/forecast, renewable-generation context, market price, and cooling context.
- Modeled: aggregate cooling, auxiliary demand, local PV scaling, and export-price relationship.
- Derived: renewable-share carbon proxy, forecasts, and facility balance variables.
- Generated: VPP reference, DR event indicator, and DR import limit.
- Scenario-transformed: workload/cooling sensitivity datasets derived from the publication baseline.

Raw third-party archives are excluded. See `../DATA_AVAILABILITY.md` and `../docs/DATA_PROVENANCE.md` before public deposition.
