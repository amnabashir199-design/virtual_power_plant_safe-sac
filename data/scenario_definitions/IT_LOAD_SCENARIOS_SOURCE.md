# IT-Load Scenarios for Green Data-Center VPP Evaluation

## Purpose

The scenario module extends the 2019 green data-center VPP dataset with explicit IT-workload sensitivity cases. Its purpose is to test whether controller conclusions remain stable when workload level, temporal shape, scheduling flexibility, cooling stress, or coincidence with demand-response events changes.

The scenarios are experimental perturbations for thesis sensitivity analysis. Except for `baseline_real`, they are synthetic and must not be described as measurements from additional facilities or as forecasts of scenario probability.

Generate the datasets and comparison outputs with:

```powershell
.\.venv\Scripts\python.exe scripts\24_build_it_load_scenarios.py --data data\green_dc_vpp_2019_15min.csv
```

Scenario CSV files are written to `data/scenarios/`. The summary table is written to `results/tables/it_load_scenario_summary.csv`, and comparison figures are written to `results/figures/`.

## Common recomputation equations

Let (P_{IT,t}^{(s)}) denote scenario IT power at interval (t), (r_{cool,t}^{(s)}) the cooling ratio, (P_{PV,t}) photovoltaic power, and (s) the scenario. The generator uses the following equations:

\[
P_{cool,t}^{(s)} = P_{IT,t}^{(s)} r_{cool,t}^{(s)},
\]

\[
P_{other,t}^{(s)} = 0.05 P_{IT,t}^{(s)},
\]

\[
P_{DC,t}^{(s)} = P_{IT,t}^{(s)} + P_{cool,t}^{(s)} + P_{other,t}^{(s)},
\]

\[
PUE_t^{(s)} = \frac{P_{DC,t}^{(s)}}{P_{IT,t}^{(s)}},
\]

\[
P_{net,t}^{(s)} = P_{DC,t}^{(s)} - P_{PV,t},
\]

\[
P_{import,t}^{(s)} = \max(P_{net,t}^{(s)},0), \qquad
P_{export,t}^{(s)} = \max(-P_{net,t}^{(s)},0).
\]

The normalized utilization is scaled consistently with the IT-load change and clipped to the physical range ([0,1]). Existing facility forecast error is retained by applying the original total-load forecast ratio to the recomputed total load. PV, price, carbon proxy, grid reference, and DR signals are unchanged.

## Scenario assumptions

### `baseline_real`

Retains the original real/hybrid IT-load profile and time-varying cooling ratio. Recomputed columns provide a consistency check, but no IT-load perturbation is applied.

### `low_it_load`

Uses

\[
P_{IT,t}^{(low)} = 0.75 P_{IT,t}^{(base)}.
\]

It represents operation below the reference workload level. Cooling and auxiliary load are recalculated from the lower IT demand.

### `high_it_load`

Uses

\[
P_{IT,t}^{(high)} = 1.25 P_{IT,t}^{(base)}.
\]

It represents sustained demand growth and increased pressure on grid import, battery dispatch, and DR compliance.

### `ai_training_bursty`

Multiplies IT demand by 1.45 during eight-hour blocks from 08:00 to 16:00 UTC on Tuesday, Thursday, and Saturday. Block boundaries intentionally introduce strong ramp-up and ramp-down behavior associated with stylized GPU training jobs. The schedule is synthetic and is not asserted to represent a particular cluster.

### `smooth_cloud_service`

Applies a centered four-hour rolling mean to IT power. A separate multiplicative correction for every UTC day restores that day's original IT energy:

\[
P_{IT,t}^{(smooth)} = \widetilde{P}_{IT,t}
\frac{\sum_{t\in d}P_{IT,t}^{(base)}}{\sum_{t\in d}\widetilde{P}_{IT,t}}.
\]

This produces lower short-term ramp rates while avoiding an artificial change in daily work served.

### `workload_shiftable`

Treats 80% of interval IT demand as inflexible and 20% of each day's energy as flexible. Flexible energy is redistributed using exponentially decreasing weights based on normalized daily price, normalized daily carbon proxy, and a DR penalty. Energy is then normalized separately for each day, so:

\[
\sum_{t\in d}P_{IT,t}^{(shift)}\Delta t
=
\sum_{t\in d}P_{IT,t}^{(base)}\Delta t.
\]

This represents delay-tolerant batch work. It does not model job deadlines, server capacity, network constraints, or service-level agreements.

### `hot_day_cooling_stress`

Retains baseline IT power. For ambient temperatures in the hottest decile, the cooling ratio increases progressively with temperature, reaching at most 25% above the original modeled ratio. The scenario isolates cooling stress from workload growth.

### `dr_coincident_high_load`

Increases IT demand by 25% during DR-active intervals and within one hour before or after an event. It intentionally creates an adverse coincidence between facility demand and grid constraints.

## Metadata and provenance

Every scenario file contains:

- `scenario_name`
- `scenario_description`
- `it_load_scaling_factor`
- `cooling_model_note`
- `scenario_family`

The source dataset combines dataset-derived signals, modeled facility quantities, and generated VPP context. Scenario metadata does not change that provenance. In thesis writing, use the following distinctions:

- **Dataset-derived:** source IT-load and utilization profiles and source operational/context signals.
- **Modeled:** cooling power, auxiliary load, total facility load, PUE, forecasts, and derived net-import quantities.
- **Synthetic variation:** all scenario perturbations except the unmodified baseline case.

## Connection to the thesis

The scenarios connect workload behavior to the thesis objectives by exposing changes in:

- VPP reference-tracking RMSE and MAE;
- grid import and demand-response success;
- operating cost and carbon emissions;
- battery throughput and equivalent cycling;
- terminal state-of-charge error;
- cooling demand and total data-center power.

They allow the no-battery, rule-based, greedy-tracking, and Residual Safe-SAC v2 controllers to be compared under the same PV, price, carbon, grid-reference, and DR context.

## Limitations

- Scenario factors are transparent sensitivity assumptions, not calibrated probabilities.
- The cooling equation is an aggregate power model and does not simulate airflow, server thermodynamics, or spatial temperature distribution.
- AI training is represented by deterministic blocks rather than a queueing or job-level workload model.
- The shiftable case preserves energy but does not enforce deadlines, computing-resource limits, or migration costs.
- PUE is a derived facility ratio and should not be interpreted as an independent measured response.
- Price and carbon signals are not changed in response to scenario demand; market feedback is outside the present scope.
- DR and grid-reference signals remain exogenous, so the experiments evaluate controller response rather than market clearing.

These limitations should accompany scenario results to avoid presenting synthetic stress cases as measured operating records.
