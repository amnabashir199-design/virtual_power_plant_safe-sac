# Known Limitations

- The evidence uses a hybrid benchmark, not a measured operational data-center VPP.
- Several signals are modeled, derived, scaled, or generated.
- Workload scenarios are controlled synthetic sensitivities.
- The VPP reference and DR context are generated from benchmark variables.
- The carbon quantity is a renewable-share-derived proxy, not metered emissions.
- Evaluation covers one held-out 92-day quarter and one BESS size.
- Battery degradation is approximated by a throughput-based cost.
- Cooling is aggregate rather than a detailed thermal plant model.
- The controller dispatches only the battery; no server/job scheduler is included.
- The study includes no hardware-in-the-loop test or operational field deployment.
- Runtime physical projection is not a formal safe-reinforcement-learning proof.
- Scenario comparisons are benchmark-specific associations, not population-wide causal inference.
- Daily SOC reset is an experimental comparison convention, not a physical reset capability.
