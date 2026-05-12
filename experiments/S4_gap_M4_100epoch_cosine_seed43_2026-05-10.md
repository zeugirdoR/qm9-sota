# S4 Gap Single-Target M4 — 100 Epoch Warmup/Cosine, Seed 43

## Status

Scheduler did not improve over S2.

## Result

- Run: S4_gap_M4_100epoch_cosine_seed_43
- Target: gap
- Validation MAE: 115.863 meV
- Test MAE: 117.853 meV

## Comparison

- S0 10 epochs: ~180.04 meV test
- S1 30 epochs: ~123.00 meV test
- S2 60 epochs, lr=0.0002: ~103.85 meV test
- S3 100 epochs, lr=0.0001: ~105.84 meV test
- S4 100 epochs, warmup/cosine: ~117.85 meV test

## Interpretation

Warmup/cosine scheduling did not help the single-target gap run. The best current gap recipe remains S2.

The gap target appears to need an electronic/global representation improvement rather than more generic training schedule tuning.

## Decision

Keep S2 as best gap result. Next: test M4 + global electronic token.
