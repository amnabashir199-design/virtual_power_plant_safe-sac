# Configuration Map

The main evaluation configurations preserve every scientific value from the final runs; only dataset/metadata path fields were adapted to the portable package layout.

- `direct_safesac/config.yaml`
- `residual_sac/config.yaml`
- `proposed_residual_safesac/config.yaml`
- `baselines/evaluation_config.yaml`
- `training_normalization_parameters.json`
- `scenarios/scenario_parameters.json`
- `ablations/ablation_definitions.yaml`

Byte-for-byte per-run configuration snapshots remain beside each selected model checkpoint and retain the original project-relative layout strings for audit provenance. Those strings are not absolute workstation paths and are not used by the primary package workflow.
