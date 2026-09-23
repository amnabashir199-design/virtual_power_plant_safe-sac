# Experiment Protocol

## Temporal protocol

- Training: 1 January–30 June 2019 (17,376 intervals).
- Validation: 1 July–30 September 2019 (8,832 intervals).
- Testing: 1 October–31 December 2019 (8,832 intervals; 92 days).
- Dispatch resolution: 15 minutes.
- Episode length: 96 steps (one UTC day).
- Initial and daily-reset SOC: 0.55.
- Previous battery power and grid import reset to zero at each episode boundary.

## Training and model selection

Each learned model was trained for 100,000 environment steps. Validation checkpoints were recorded every 20,000 steps and evaluated deterministically over all 92 validation days. The selected checkpoint minimized mean daily validation objective. Five seeds were trained for each of Direct Safe-SAC, Residual SAC, and Proposed Residual Safe-SAC. Three additional five-seed ablation families were independently trained.

## Evaluation

Primary testing uses deterministic policy inference over 92 separate daily-reset episodes. The main learned comparison represents 15 selected models. The same checkpoints are applied to the eight workload-centered scenarios without scenario-specific retraining. Deterministic controllers receive the physical feasibility projection but not the learned-controller service-coaching stage.

## Package reproduction

The default package workflow does not rerun training or modify model weights. It loads all selected checkpoints to verify integrity and dimensions, then independently recalculates results from the exact included daily-reset trajectories. The trajectory replay is the bit-stable route for reproducing the manuscript aggregates; optional fresh inference can be conducted with the retained reference scripts on a working copy.

## Reproducibility caveat

Full RL retraining is stochastic and may produce different checkpoint choices and numerical values even with nominally identical seeds on different hardware/library builds. Retraining is therefore transparency material, not the primary result-reproduction workflow.
