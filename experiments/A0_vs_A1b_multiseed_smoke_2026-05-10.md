# A0 vs A1b Multi-Seed Smoke Test

## Status

A1b wins 3 / 3 paired smoke-test seeds.

## Commit

bdfa4556183686b0f6c7da43e3774d86d80f207d

## Runs

| Seed | A0 best epoch | A0 norm MAE | A1b best epoch | A1b norm MAE | A1b - A0 |
|---:|---:|---:|---:|---:|---:|
| 42 | 4 | 0.314042 | 4 | 0.313806 | -0.000236 |
| 43 | 5 | 0.327969 | 5 | 0.327131 | -0.000838 |
| 44 | 5 | 0.357386 | 5 | 0.356179 | -0.001208 |

## Aggregate

- Mean A0 normalized MAE: 0.3331325650
- Mean A1b normalized MAE: 0.3323719800
- Mean delta A1b - A0: -0.0007605851
- Wins for A1b: 3 / 3
- Approximate relative improvement: 0.23%

## Interpretation

The gentle scheduled Droplet loss gives a small but consistent improvement over the tiny radial MPNN baseline across three paired seeds.

This is not a SOTA-level claim. It is a smoke-test signal that the gentle Droplet schedule is worth carrying forward.

A2 learnable offsets were effectively neutral relative to A1b, so the next priority is not more floating-parameter tuning. The next priority is confirming whether A1b survives more seeds and/or larger training scale.

## Decision

Promote A1b gentle scheduled Droplet to candidate regularizer.

Next steps:

1. Extend A0 vs A1b smoke comparison to 10 seeds.
2. If A1b remains positive, run A0 vs A1b on the full 110k training split.
3. Then move the Droplet overlay to a stronger 3D/equivariant backbone.
