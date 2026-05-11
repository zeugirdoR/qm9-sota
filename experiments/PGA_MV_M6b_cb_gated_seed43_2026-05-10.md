# PGA Multivector Transformer M6b — Scheduled/Gated Cauchy-Binet Bias, Seed 43

## Status

Promising candidate. Scheduled/gated CB fixes the raw CB failure mode.

## Commit

7e47d275db98f43b8bf2f9b3bc5426cc6e767616

## Result

- Run: PGA_MV_M6b_cb_gated_seed_43
- Best epoch: 9
- Best normalized MAE: 0.1380743235
- Best raw MAE: 31.054306

## Comparison

- M4 vector backbone seed 43: approximately 0.1408
- M6 raw CB seed 43: 0.1827663034
- M6b scheduled/gated CB seed 43: 0.1380743235

## Interpretation

Raw Cauchy-Binet concatenation degraded M4, but scheduled CB attention bias improves seed 43 slightly. This validates the principle that information-volume terms should be introduced gradually and as a gated residual/bias, not as full-strength raw features from step zero.

## Decision

Run M6b on seeds 42, 44, and 45 before promotion.
