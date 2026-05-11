# PGA Multivector Transformer M5 — Family Heads, 3 Seeds

## Status

Diagnostic only. M5 is not promoted over M4.

## Commit

186a98d8f256f65cf0268865963b91e957b296c4

## Results

| Seed | M5 norm MAE | M5 raw MAE | Best epoch | M4 ref | Tiny A0 ref | M5 - M4 | M5 - Tiny A0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.138740 | 31.095806 | 10 | 0.128203 | 0.234050 | +0.010537 | -0.095310 |
| 43 | 0.129847 | 30.172228 | 10 | 0.144428 | 0.224260 | -0.014581 | -0.094413 |
| 44 | 0.152867 | 33.168186 | 9 | 0.149065 | 0.239976 | +0.003802 | -0.087109 |

## Aggregate

- Mean M5 normalized MAE: 0.1404847354
- Mean M5 - M4: -0.0000805929
- Wins vs M4: 1 / 3
- Wins vs Tiny A0: 3 / 3

## Rigid-motion invariance check

Using M5 seed 43:

- max_abs_diff: 5.125999e-06
- mean_abs_diff: 4.195746e-07

The scalar predictions are effectively invariant to rigid rotations/translations.

## Interpretation

Family heads improve seed 43 but hurt seeds 42 and 44. The result is essentially neutral versus M4 on average and not robust.

The invariance check validates the PGA/MV vector-channel design.

## Decision

Do not promote M5. Promote M4 as the current primary backbone.

Next: M6 = M4 + Cauchy-Binet / local volume features.
