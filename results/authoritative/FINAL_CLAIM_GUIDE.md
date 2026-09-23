# Final Claim Guide

| # | Potential claim | Assessment | Evidence and manuscript-safe wording |
| --- | --- | --- | --- |
| 1 | IT workload magnitude materially changes VPP operational difficulty. | SUPPORTED WITH QUALIFICATION | The eight controlled benchmark scenarios show substantial changes in load and controller metrics; state this as scenario sensitivity, not a population-wide effect. |
| 2 | IT workload temporal structure is associated with VPP performance even when daily IT energy is similar. | SUPPORTED WITH QUALIFICATION | Shiftable/bursty/smooth scenarios preserve or control energy while changing timing. Use "is associated within the tested scenarios." |
| 3 | High IT load reduces available flexibility in the tested benchmark. | SUPPORTED WITH QUALIFICATION | Supported for the frozen high-load and DR-coincident cases; limit the statement to this battery, horizon, and benchmark. |
| 4 | Bursty AI-like workloads increase VPP tracking difficulty relative to smoother workloads. | SUPPORTED WITH QUALIFICATION | Descriptive comparison supports workload-structure sensitivity, but not a universal AI-workload effect. |
| 5 | Residual learning improves tracking relative to Direct Safe-SAC. | STRONGLY SUPPORTED | Frozen five-seed mean RMSE is 87.666 kW for Residual SAC versus 158.567 kW for Direct Safe-SAC on the main test horizon. |
| 6 | Safety projection improves DR/constraint performance. | SUPPORTED WITH QUALIFICATION | True ablations support a contribution from the configured projection/coaching stack; distinguish physical projection from service coaching and avoid a formal guarantee. |
| 7 | Proposed Residual Safe-SAC provides a balanced multi-objective operating point. | SUPPORTED WITH QUALIFICATION | It achieves 80.370 kW RMSE and 99.844% interval compliance, while cost, carbon, cycling, and terminal SOC remain conflicting objectives. |
| 8 | Proposed controller outperforms all baselines. | NOT SUPPORTED | Replace with metric-specific comparisons; no controller is best on every cost, carbon, tracking, DR, throughput, and SOC metric. |
| 9 | Proposed controller is universally robust. | NOT SUPPORTED | Replace with "performance was evaluated across eight predefined workload scenarios in the hybrid benchmark." |
| 10 | Results generalize to real operational data centers. | NOT SUPPORTED | Replace with "the results establish benchmark evidence and motivate validation on independent operational deployments." |
