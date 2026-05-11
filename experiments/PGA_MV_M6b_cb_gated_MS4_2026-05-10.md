# PGA Multivector Transformer M6b — Scheduled/Gated Cauchy-Binet, 4 Seeds

## Status

Diagnostic only. M6b is not promoted over M4.

## Commit

7e47d275db98f43b8bf2f9b3bc5426cc6e767616

## Results

| Seed | M6b norm MAE | M4 ref | M6b - M4 | Winner |
|---:|---:|---:|---:|---|
| 42 | 0.142046 | 0.128203 | +0.013843 | M4 |
| 43 | 0.138074 | 0.140798 | -0.002724 | M6b |
| 44 | 0.144738 | 0.149065 | -0.004327 | M6b |
| 45 | 0.136058 | 0.131331 | +0.004727 | M4 |

## Aggregate

- Mean M6b normalized MAE: 0.1402289718
- Mean M6b - M4: +0.0028797218
- Wins vs M4: 2 / 4

## Interpretation

Scheduled/gated Cauchy-Binet is much safer than raw Cauchy-Binet concatenation, but it is not robustly better than M4. It helps seeds 43 and 44 but hurts seeds 42 and 45.

## Decision

Do not promote M6b.

Keep M4 as the primary backbone.

Next: test M4 with group-wise or target-wise Droplet rather than more Cauchy-Binet tuning.
