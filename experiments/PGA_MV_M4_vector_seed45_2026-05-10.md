# PGA Multivector Transformer M4 — Vector Channels, Seed 45

## Status

Confirms M4 robustness.

## Commit

1abd14eefc3887ba27a3817cfc1ba08f504ad05a

## Result

- Run: PGA_MV_M4_vector_seed_45
- Best epoch: 10
- Best normalized MAE: 0.1313305348
- Best raw MAE: 29.192764

## Interpretation

M4 remains very strong on an additional seed. This supports promoting M4 as the primary PGA/MV backbone.

## Decision

Promote M4 as the validated primary backbone.

Next: M6b scheduled/gated Cauchy-Binet bias, not raw Cauchy-Binet concatenation.
