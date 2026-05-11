# PGA Multivector M4D — Group-wise Droplet, 4 Seeds

## Status

Diagnostic only. M4D is not promoted over M4.

## Commit

f0158be93e9953d44fa8c8134ed7a0a8d445aafa

## Results

| Seed | M4D norm MAE | M4 ref | M4D - M4 | Winner |
|---:|---:|---:|---:|---|
| 42 | 0.139204 | 0.128203 | +0.011001 | M4 |
| 43 | 0.137896 | 0.140798 | -0.002902 | M4D |
| 44 | 0.152837 | 0.149065 | +0.003772 | M4 |
| 45 | 0.140590 | 0.131331 | +0.009259 | M4 |

## Aggregate

- Mean M4D normalized MAE: 0.1426319517
- Mean M4D - M4: +0.0052827017
- Wins vs M4: 1 / 4

## Interpretation

Group-wise Droplet helps seed 43 but hurts seeds 42, 44, and 45. It is not robustly beneficial on the strong M4 backbone.

The effective weights stayed close to 1.0, so this was not a collapse; the issue is that even gentle group-wise reweighting perturbs a strong backbone.

## Decision

Do not promote M4D.

Keep M4 as the primary backbone. Future Droplet work should be either much weaker, applied only late, or used diagnostically/per-target rather than as a default training objective.
