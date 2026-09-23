# Final Reproducibility Report

## Publication readiness

**READY FOR MANUSCRIPT WRITING.** The deterministic-baseline semantic defect was confirmed, minimally repaired, rerun only where affected, and independently verified. Historical outputs remain available under `results/publication_v1`.

## 1-7. Frozen experimental basis

- Dataset provenance: hybrid data-center VPP benchmark containing source-based and model-derived signals; see `tables/TABLE_DATASET_AND_PROVENANCE.csv`.
- Dataset SHA256: `e8cfd0456c63797c25e9a16a43423eb783c4f7c8f6a41436748cbf9f02ddff31` (unchanged).
- Chronological split: test period 2019-10-01 through 2019-12-31, 92 days and 8832 15-minute intervals.
- Preprocessing safeguards: the repaired chronological and causal preprocessing artifacts from earlier phases were not modified.
- Controllers: five deterministic baselines plus Direct Safe-SAC, Residual SAC, and proposed Residual Safe-SAC.
- Training configuration: frozen YAML snapshots; 100,000 training steps per learned model. No training occurred in Phase 3.
- Seeds: learned controllers use seeds 1-5; deterministic baselines use one deterministic evaluation.

## 8-10. Baseline repair

The old common evaluator interpreted direct deterministic outputs as residuals and added a hidden greedy prior. It also failed to propagate `disable_battery=True` into the actual no-battery environment. Deterministic baselines now run in direct-action mode with no hidden prior, common power/SOC/ramp feasibility projection, and no proposed-controller service coaching. `disable_battery` is propagated to the actual evaluation environment. Only the five deterministic baselines were rerun.

## 11-15. Verification scope

- Affected: five main deterministic trajectories, 40 deterministic scenario trajectories, nine low-IT learned scenario evaluations with proven SOC violations, their metrics, combined tables, rankings, and dependent figures.
- Unaffected: all 15 learned checkpoint files, all main learned evaluations, the other 111 learned scenario evaluations, scenario definitions/datasets, IT characterization, and true ablations. No learned model was retrained.
- Independent metric reproduction: 85 controller-metric comparisons; all verified; maximum absolute error `1.137e-13`.
- Scenario evaluation: complete 8 scenario x 8 controller final table; only 5 x 8 deterministic cells were reevaluated.
- IT-load analysis: prior Phase 2 characterization files are hash-identical; interpretation remains descriptive.
- Ablations: existing five-variant learned ablation package is hash-identical and unaffected by the deterministic controller path.

## 16. Remaining limitations

Evidence comes from a hybrid benchmark, one test quarter, one storage sizing/configuration, and eight predefined non-independent sensitivity scenarios. Scenario contrasts are not causal inference or external real-data-center validation. Action projection is explicit feasibility handling, not a formal safe-RL guarantee.

## 17-18. Tests and final status

- Tests executed: 65; passed: 65; failed/errors: 0; skipped: 0.
- Main trajectory checks: 45/45 passed.
- No-battery exact-zero checks: passed for main and all eight scenarios.
- Dataset and 15 model hashes: unchanged.
- Authoritative source: this `results/publication_final` directory after `PUBLICATION_FREEZE.json` creation.
