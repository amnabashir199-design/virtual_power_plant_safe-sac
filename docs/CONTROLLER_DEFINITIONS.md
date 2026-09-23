# Controller Definitions

## Deterministic controllers

- **No battery:** battery power fixed to zero.
- **Rule-based:** price/SOC heuristic battery dispatch.
- **TOU self-consumption:** time-of-use and PV/self-consumption heuristic.
- **Carbon-aware:** carbon-proxy/SOC heuristic.
- **Greedy tracking:** battery command directly reduces instantaneous no-battery grid-reference error.

Every battery-equipped deterministic controller receives the common physical power/SOC/change projection. Deterministic controllers do not receive learned-controller service coaching.

## Learned controllers

- **Direct Safe-SAC:** a SAC actor issues the direct battery command. It has no greedy prior, uses physical projection plus service coaching, uses terminal recovery, and has zero residual-action weight. Actor: `24→256→256→1`; twin critics: `25→256→256→1`.
- **Residual SAC:** greedy VPP-tracking prior plus a SAC correction bounded to ±150 kW. It uses the same physical projection and service coaching as the proposed method, but terminal recovery is disabled and residual-action weight is zero. Actor: `28→256→256→1`; twin critics: `29→256→256→1`.
- **Proposed Residual Safe-SAC:** the same prior-guided bounded residual architecture, projection, and coaching, plus residual-action weight 0.10 and the configured terminal SOC-recovery schedule. Actor: `28→256→256→1`; twin critics: `29→256→256→1`.

DR results for learned controllers reflect the full configured stack, including the learned policy and external service-coaching stage; they must not be attributed to the actor alone.

## Ablations

- Direct SAC without projection/coaching.
- Residual prior without projection/coaching.
- Full Residual SAC.
- Proposed controller without terminal recovery.
- Full Proposed Residual Safe-SAC.

The two full-family rows reuse their main baseline trajectories. Only the three unique independently trained ablation trajectory sets are duplicated under `trajectories/ablations/`, avoiding redundant copies.
