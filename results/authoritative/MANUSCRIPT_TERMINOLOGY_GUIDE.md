# Manuscript Terminology Guide

| Topic | Prefer | Avoid |
| --- | --- | --- |
| Dataset | hybrid benchmark; hybrid data-center VPP benchmark; source-based and model-derived signals | fully measured real-world VPP dataset |
| IT baseline | baseline workload; baseline data-center workload profile; baseline hybrid-benchmark workload | real workload, unless provenance fully supports it |
| Scenario analysis | scenario-based sensitivity analysis; workload sensitivity analysis; controlled scenario analysis | universal robustness proof; causal effect; real-world validation |
| Safety | explicit action-level feasibility projection; physical projection; service coaching | formally guaranteed safe RL |
| Proposed controller | prior-guided Residual Safe-SAC with a greedy tracking prior, bounded residual correction, and explicit action projection | universally safe or universally optimal controller |

Physical projection means power, SOC, and ramp feasibility enforcement. Service coaching means the separate high-level DR/grid intervention. Deterministic baselines receive the former only; they do not inherit the proposed controller's service coaching.
