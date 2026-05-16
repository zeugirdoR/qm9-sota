# System76 consolidated local U0_atom results — 2026-05-16

## Claim status

Local proxy/scout/debug results only. Not production. Not SOTA.

## Important decision

The new 20k/2k SYS76 smoke protocol is not a valid promotion anchor. The recovered proxy protocol is the current System76 scout path.

## Key recovered baseline

```text
SYS76_RECOVERED_M4_U0atom_100epoch_seed43
best val U0_atom: 258.4153413772583 meV
role: local candidate-ranking baseline
```

## Consolidated local summary rows

| Rank | Run | Best val MAE | Best epoch | Commit | GPU |
|---:|---|---:|---:|---|---|
| 1 | `LOCAL_M4_U0atom_300epoch_seed43` | 114.43684995174408 meV | 274 | `30a26bd9df9ad536ba16762c937cf17347eb6fc5` | NVIDIA GeForce RTX 4070 Laptop GPU |
| 2 | `LOCAL_M4_U0atom_200epoch_seed43` | 144.1979706287384 meV | 190 | `30a26bd9df9ad536ba16762c937cf17347eb6fc5` | NVIDIA GeForce RTX 4070 Laptop GPU |
| 3 | `LOCAL_M4_U0atom_100epoch_seed43` | 224.65749084949493 meV | 93 | `30a26bd9df9ad536ba16762c937cf17347eb6fc5` | NVIDIA GeForce RTX 4070 Laptop GPU |
| 4 | `SYS76_RECOVERED_M4_U0atom_100epoch_seed43` | 258.4153413772583 meV | 96 | `496740f6a8d3f24b4a6756167128cc0cccde9379` | NVIDIA GeForce RTX 4070 Laptop GPU |
| 5 | `LOCAL_M4S_U0atom_100epoch_seed43` | 415.1037633419037 meV | 95 | `30a26bd9df9ad536ba16762c937cf17347eb6fc5` | NVIDIA GeForce RTX 4070 Laptop GPU |
| 6 | `SYS76_U0atom_BASE_M4_smoke100_seed43_commit30a26bd` | 573.0023980140686 meV | 98 | `30a26bd9df9ad536ba16762c937cf17347eb6fc5` | NVIDIA GeForce RTX 4070 Laptop GPU |
| 7 | `SYS76_U0atom_BASE_M4_smoke100_seed43` | 603.0921339988708 meV | 90 | `496740f6a8d3f24b4a6756167128cc0cccde9379` | NVIDIA GeForce RTX 4070 Laptop GPU |
| 8 | `SYS76_U0atom_BASE_M4_smoke80_seed43` | 691.1085247993469 meV | 77 | `9d9243bdfc008c86bbf80da64d4a51ab91f49924` | NVIDIA GeForce RTX 4070 Laptop GPU |
| 9 | `SYS76_RECOVERED_M4_dropout_0p03_U0atom_100epoch_seed43` | 1552.3090362548828 meV | 92 | `496740f6a8d3f24b4a6756167128cc0cccde9379` | NVIDIA GeForce RTX 4070 Laptop GPU |
| 10 | `LOCAL_M4S_U0atom_smoke_seed43` | 2385.7500553131104 meV | 20 | `30a26bd9df9ad536ba16762c937cf17347eb6fc5` | NVIDIA GeForce RTX 4070 Laptop GPU |

## Do not commit

```text
results_local/
*.pt
checkpoint_epoch_*.pt
latest_checkpoint.pt
best_model.pt
```

