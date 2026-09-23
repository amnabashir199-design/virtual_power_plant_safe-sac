# Metric Definitions

| Metric | Definition | Aggregation | Unit | Interpretation |
|---|---|---|---|---|
| Operating cost | Import cost minus export revenue | Sum per day; mean over 92 days, then seeds | EUR/day | Lower is better |
| Carbon proxy | Imported energy times carbon-proxy intensity | Sum per day; mean over 92 days, then seeds | kg CO2-proxy/day | Lower proxy burden |
| VPP tracking RMSE | `sqrt(mean_t((P_grid,t - P_ref,t)^2))` | Per day; mean over days, then seeds | kW | Primary tracking error |
| VPP tracking MAE | `mean_t(|P_grid,t - P_ref,t|)` | Per day; mean over days, then seeds | kW | Absolute tracking error |
| DR interval compliance | Active intervals satisfying `P_import ≤ P_DR_limit + 10^-6` | **Primary:** pool all actual active intervals per seed, then average seeds | % | Active-interval service compliance |
| DR event compliance | Contiguous active events with no violating interval | Pool events per seed, then average seeds | % | Whole-event compliance |
| DR violation energy | Positive DR-limit exceedance integrated at 0.25 h | Full 92-day total per seed, then average seeds | kWh/92 days | Violation magnitude over horizon |
| Maximum DR violation | Largest positive active-interval exceedance | Full-quarter maximum per seed, then average seeds | kW | Worst active-interval exceedance |
| Mean daily DR violation energy | Daily violation energy including zero-event days | Per day; mean over 92 days, then seeds | kWh/day | Secondary daily diagnostic |
| Mean daily maximum DR violation | Maximum within each day, zero on non-event days | Per day; mean over 92 days, then seeds | kW/day diagnostic | Secondary daily diagnostic |
| Secondary daily-average DR compliance | Daily compliance, assigning 100% to no-event days | Mean over 92 days, then seeds | % | Not the primary scenario statistic |
| Battery throughput | `sum_t |P_batt,t| Δt` | Per day; mean over days, then seeds | kWh/day | Battery-use burden |
| Equivalent full cycles | Throughput divided by `2E_capacity` | Per day; mean over days, then seeds | cycles/day | Approximate cycling |
| Terminal SOC error | `|SOC_end - 0.55|` | Per day; mean over days, then seeds | fraction | Daily energy neutrality |
| Physical projection fraction | Intervals where hard feasibility changes the command | Pool intervals per day/seed as reported | % | Physical intervention frequency |
| Service-coaching fraction | Intervals where external service coaching changes the action | Pool intervals per day/seed as reported | % | Service intervention frequency |

The main, safety, scenario, and ablation tables explicitly separate daily diagnostics from pooled/full-quarter quantities. All explicit DR columns are recalculated from the packaged trajectories for every controller; ambiguous legacy columns are not reused. They must not be interchanged. See `DR_AGGREGATION_RESOLUTION.md` for the audit trail.
