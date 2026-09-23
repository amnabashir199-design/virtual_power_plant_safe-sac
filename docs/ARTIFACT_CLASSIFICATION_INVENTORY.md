# Artifact Classification Inventory

This classification was completed before package copying. It records the disposition of the major artifact families found in the parent research workspace.

| Candidate family | Classification | Disposition and rationale |
|---|---|---|
| `publication_final` tables, corrected baseline trajectories, low-IT learned repairs, freeze/evidence reports | FINAL / AUTHORITATIVE | Included selectively; sole source for corrected deterministic values and corrected low-IT affected trajectories |
| `publication_v1` selected model checkpoints and per-seed config snapshots | FINAL / AUTHORITATIVE | Included for all 15 main and 15 independent-ablation seeds |
| `publication_v1` daily-reset learned trajectories | REQUIRED FOR REPRODUCTION | Included; continuous-quarter variants excluded |
| `publication_v1` scenario datasets and learned trajectories | REQUIRED FOR REPRODUCTION | Included, with affected low-IT files replaced by final repaired versions |
| `publication_final` deterministic scenario trajectories | FINAL / AUTHORITATIVE | Included for five deterministic controllers × eight scenarios |
| `publication_v1` true-ablation trajectories | REQUIRED FOR REPRODUCTION | Included only for three unique independently trained ablation families; duplicate full-family copies omitted |
| Publication dataset Parquet and metadata | FINAL / AUTHORITATIVE | Included; compressed CSV duplicate omitted |
| Raw uploaded third-party archives and source-input copies | EXCLUDE | Redistribution status not established; not required for frozen-model replay |
| Final source package, environment, safety, metrics, baselines, plotting | REQUIRED FOR REPRODUCTION | Included from the final curated source tree |
| Exact historical training/evaluation/scenario scripts | OPTIONAL SUPPORTING | Selected final scripts included under `scripts/reference/`; not called by default |
| Model training monitors | INTERMEDIATE / LOG | Excluded; checkpoint selection tables and run metadata retained |
| Continuous-quarter trajectories | OBSOLETE FOR MANUSCRIPT | Excluded; manuscript uses 92 daily-reset episodes |
| Older `publication_v1` deterministic tables and explicitly labeled audit inputs | OBSOLETE / SUPERSEDED | Excluded from authoritative tables |
| Earlier figures and plot styles | OBSOLETE / DUPLICATE | Excluded; only final manuscript figures retained |
| `results/`, `verifying/`, and `vpp__final/` duplicate archive trees | DUPLICATE | Not copied wholesale; authoritative individual artifacts selected |
| Manuscript drafts, thesis files, literature PDFs | EXCLUDE | Not computational reproduction inputs |
| ZIP archives of working directories/packages | DUPLICATE / EXCLUDE | Excluded from the package |
| `.venv`, `.git`, `.pytest_cache`, `__pycache__`, IDE/system metadata | DEVELOPMENT CLUTTER | Excluded |
| Failed/temporary experiment outputs and debug exports | DEBUG / EXCLUDE | Excluded |
| Intermediate/latest/backup checkpoints | INTERMEDIATE / EXCLUDE | Excluded; only selected validation checkpoints included |

No artifact was selected merely because it had the newest filename. Final manuscript values were traced through the final evidence map, corrected baseline results, learned seed table, scenario table, repaired low-IT trajectories, ablation table, safety table, and the final manuscript evidence verification.
