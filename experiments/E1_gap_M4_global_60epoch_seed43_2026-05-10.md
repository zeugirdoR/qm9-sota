# E1 Gap Single-Target — M4 + Global Electronic Token, 60 Epochs, Seed 43

## Status

E1 improves over S2 and crosses the first gap milestone below 100 meV.

## Result

- Run: E1_gap_M4_global_60epoch_seed_43
- Target: gap
- Target index: 4
- Validation MAE: 98.239 meV
- Test MAE: 99.747 meV

## Comparison

- Multitask M4D gap test: approximately 224 meV
- S0 M4 10 epochs: approximately 180 meV
- S1 M4 30 epochs: approximately 123 meV
- S2 M4 60 epochs: approximately 103.845 meV
- S3 M4 100 epochs low LR: approximately 105.840 meV
- S4 M4 100 epochs cosine: approximately 117.853 meV
- E1 M4 + global token 60 epochs: 99.747 meV

## Interpretation

The global electronic token provides a modest but real improvement for the gap target. This supports the hypothesis that HOMO/LUMO/GAP need stronger global/electronic context beyond local vector-channel geometry.

## Decision

Promote E1 as the best current gap architecture, but continue to E2: a stronger read-write global token/global attention mechanism.
