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
