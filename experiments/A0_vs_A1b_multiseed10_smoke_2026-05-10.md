# A0 vs A1b Multi-Seed Smoke Test, 10 Seeds

## Status

A1b gentle scheduled Droplet wins 9 / 10 paired smoke-test seeds.

## Commit

bdfa4556183686b0f6c7da43e3774d86d80f207d

## Aggregate

- Seeds: 42–51
- Mean A0 normalized MAE: 0.3320670746
- Mean A1b normalized MAE: 0.3315447507
- Mean delta A1b - A0: -0.0005223239
- Median delta: -0.0004153699
- Delta std: 0.0005185679
- Wins for A1b: 9 / 10
- Approximate relative improvement: 0.157%

## Per-seed results

| Seed | A0 norm MAE | A1b norm MAE | A1b - A0 | Winner |
|---:|---:|---:|---:|---|
| 42 | 0.314042 | 0.313806 | -0.000236 | A1b |
| 43 | 0.327969 | 0.327131 | -0.000838 | A1b |
| 44 | 0.357386 | 0.356179 | -0.001207 | A1b |
| 45 | 0.341844 | 0.341683 | -0.000161 | A1b |
| 46 | 0.336003 | 0.335889 | -0.000115 | A1b |
| 47 | 0.312940 | 0.312452 | -0.000489 | A1b |
| 48 | 0.356491 | 0.356654 | +0.000164 | A0 |
| 49 | 0.318211 | 0.317767 | -0.000444 | A1b |
| 50 | 0.338421 | 0.336911 | -0.001511 | A1b |
| 51 | 0.317363 | 0.316976 | -0.000387 | A1b |

## Interpretation

A1b produces a small but consistent improvement over the TinyRadialMPNN baseline in the smoke setting.

This is not a SOTA claim. It is a validated smoke-test signal that the gentle Droplet schedule should be carried into larger-scale tests.

A2 learnable offsets remain neutral and should not be tuned further until the scheduled A1b effect survives full-split training.

## Decision

Promote A1b gentle scheduled Droplet to validated smoke regularizer.

Next:

1. Run A0 vs A1b on the full 110k training split for seeds 42, 43, 44.
2. If A1b remains positive, move the Droplet overlay to a stronger 3D/equivariant backbone.
3. Keep A2 on hold.
