# PGA Multivector Transformer M4 — Vector Channels, 3 Seeds

## Status

Major architectural breakthrough. M4 becomes the new primary backbone candidate.

## Commit

7b8e517bfced91c11d17de6986dc891b5b9bcb90

## Results

| Seed | M4 norm MAE | M4 raw MAE | Best epoch | Tiny A0 | Tiny A1b | M4 - Tiny A0 | M4 - Tiny A1b |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.128203 | 27.679562 | 10 | 0.234050 | 0.228921 | -0.105848 | -0.100718 |
| 43 | 0.144428 | 33.640694 | 8 | 0.224260 | 0.227343 | -0.079832 | -0.082915 |
| 44 | 0.149065 | 28.921007 | 10 | 0.239976 | 0.239532 | -0.090910 | -0.090466 |

## Aggregate

- Mean M4 normalized MAE: 0.1405652910
- Mean M4 - Tiny A0: -0.0921967179
- Wins vs Tiny A0: 3 / 3
- Wins vs Tiny A1b: 3 / 3

## Interpretation

Adding equivariant vector channels and invariant vector-magnitude feedback produced a large improvement over the previous TinyRadialMPNN backbone.

This validates the PGA/multivector architecture direction.

## Decision

Promote M4 as the new primary backbone.

Before adding Droplet, JEPA, or Cauchy-Binet features, run:

1. rigid-motion invariance sanity check
2. per-target diagnostics
3. at least one extra seed
