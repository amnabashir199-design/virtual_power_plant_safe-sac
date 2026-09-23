# Scenario Definitions

All scenarios are deterministic offline sensitivity constructs derived from the same baseline benchmark. They are not online workload schedulers, causal real-time controls, SLA-aware schedulers, or deployment-ready job dispatch.

| Scenario | Exact implemented transformation |
|---|---|
| Baseline | IT demand unchanged; original modeled cooling ratio retained. |
| Low IT load | `P_IT,t = 0.75 × baseline P_IT,t`; cooling and auxiliary demand recomputed. |
| High IT load | `P_IT,t = 1.25 × baseline P_IT,t`; cooling and auxiliary demand recomputed. |
| Bursty workload | `P_IT,t = 1.45 × baseline P_IT,t` Tuesday/Thursday/Saturday 08:00–16:00 UTC; baseline otherwise. |
| Smooth workload | Centered 16-interval (four-hour) rolling mean, with each UTC day rescaled to preserve baseline daily IT energy. |
| Shiftable workload | 80% fixed and 20% redistributed within each day using weights `exp[-2(0.5 price_score + 0.5 carbon_score + 2 DR)]`; daily IT energy preserved. |
| Hot-day cooling stress | Baseline IT unchanged; cooling ratio increases progressively by up to 25% above the ambient-temperature 90th percentile. |
| DR-coincident high load | `P_IT,t = 1.25 × baseline P_IT,t` during DR-active intervals and centered ±1-hour shoulders; baseline otherwise. |

The high-IT to baseline mean ratio is exactly 1.25 in the included scenario data. Scenario generation source is retained at `scripts/reference/24_build_it_load_scenarios.py`.
