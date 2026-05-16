# System76 recovered U0_atom candidate wave 1 — 2026-05-16

## Claim status

Local proxy/scout only. Not production. Not SOTA.

## Baseline

```text
run: SYS76_RECOVERED_M4_U0atom_100epoch_seed43
best val U0_atom: 258.4153413772583 meV
```

## Results

| Rank | Variant | Best val MAE | Best epoch | Commit | Run |
|---:|---|---:|---:|---|---|
| 1 | `BASE_M4` | 258.415341 meV | 96 | `496740f6a8d3f24b4a6756167128cc0cccde9379` | `SYS76_RECOVERED_M4_U0atom_100epoch_seed43` |
| 2 | `M4_dropout_0p03` | 1552.309036 meV | 92 | `496740f6a8d3f24b4a6756167128cc0cccde9379` | `SYS76_RECOVERED_M4_dropout_0p03_U0atom_100epoch_seed43` |
| 3 | `M4_global_read` | missing | missing | `missing` | `SYS76_RECOVERED_M4_global_read_U0atom_100epoch_seed43` |
| 4 | `M4_global_feedback_s0p5` | missing | missing | `missing` | `SYS76_RECOVERED_M4_global_feedback_s0p5_U0atom_100epoch_seed43` |
| 5 | `M4_cutoff10` | missing | missing | `missing` | `SYS76_RECOVERED_M4_cutoff10_U0atom_100epoch_seed43` |
| 6 | `M4_vch12` | missing | missing | `missing` | `SYS76_RECOVERED_M4_vch12_U0atom_100epoch_seed43` |
| 7 | `M4_family_head` | missing | missing | `missing` | `SYS76_RECOVERED_M4_family_head_U0atom_100epoch_seed43` |

## Decision rule

Promote only candidates that beat the recovered BASE_M4 clearly under this same protocol.

Raw JSON summary:

```text
/home/crod569/qm9-sota-local-results/SYS76_RECOVERED_U0atom_wave1_summary.json
```
