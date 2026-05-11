# Experiment Registry

| ID | Date | Description | Best Mean Normalized MAE | Decision |
|---|---:|---|---:|---|
| A0 | 2026-05-08 | TinyRadialMPNN baseline, 3 epochs | 0.349324 | Working baseline |
| A1 | 2026-05-08 | Aggressive scheduled Droplet, 3 epochs | 0.374723 | Archive; too aggressive |
| A0-paired | 2026-05-10 | Paired baseline, 5 epochs | 0.322820 | Current smoke baseline |
| A1b | 2026-05-10 | Gentle scheduled Droplet, scripted smoke, 5 epochs | 0.313806 | Tiny single-seed smoke win; proceed to A2 |

## Rules

- One command creates one run.
- Each run saves config snapshots, logs, and per-target MAE.
- Failed schedules are archived, not forgotten.
- No SOTA claim from a single seed or notebook-only code.
| A0-vs-A1b-MS3 | 2026-05-10 | Paired 3-seed smoke test, A0 vs gentle Droplet | mean delta -0.000761 | A1b wins 3/3; promote to candidate regularizer |
| A0-vs-A1b-MS10 | 2026-05-10 | Paired 10-seed smoke test, A0 vs gentle Droplet | mean delta -0.000522; A1b wins 9/10 | Promote A1b to validated smoke regularizer |
| FULL-A0-vs-A1b-MS3 | 2026-05-10 | Full split TinyRadialMPNN, seeds 42-44 | mean delta -0.001098; A1b wins 3/3 | Promote A1b to validated full-split regularizer |
| FULL10-A0-vs-A1b-MS3 | 2026-05-10 | Full split 10-epoch TinyRadialMPNN, seeds 42-44 | mean delta -0.000830; A1b wins 2/3 | Mixed; create slower/capped A1c schedule |
| FULL10-A1c-diagnostic | 2026-05-10 | Capped/slower A1c tested on seeds 42 and 43 | improves seed 43 vs A1b but loses seed 42 | Stop A1c; keep A1b |
| PGA-MV-B0-S43 | 2026-05-10 | Initial scalar PGA/multivector attention scaffold, seed 43 | norm MAE 0.361200 | Runs but weak; add edge-aware geometric attention next |
| PGA-MV-M1-S43 | 2026-05-10 | Edge-aware PGA/MV attention, seed 43 | norm MAE 0.279057 | Large gain over B0; add radial/message value transport next |
