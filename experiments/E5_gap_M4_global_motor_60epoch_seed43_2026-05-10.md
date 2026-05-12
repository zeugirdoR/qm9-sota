# E5 Gap Single-Target — E2 + Scheduled Motor Residual, 60 Epochs, Seed 43

## Status

Diagnostic only. E5 is neutral/slightly worse than E2.

## Result

- Run: E5_gap_M4_global_motor_60epoch_seed_43
- Target: gap
- Validation MAE: 92.997 meV
- Test MAE: 91.842 meV

## Comparison

- E2 read-write global token: approximately 91.599 meV test
- E3 two global feedback layers: approximately 92.145 meV test
- E5 scheduled motor residual: 91.842 meV test

## Interpretation

The scheduled motor residual does not destabilize training, but the first motor design does not improve over E2. This validates the safety of scheduled motors but not their usefulness yet.

## Decision

Do not promote E5. Keep E2 as best current gap architecture.

Next: test joint HOMO/LUMO/GAP training.
