# E2 Gap Single-Target — M4 + Read-Write Global Token, 60 Epochs, Seed 43

## Status

Best current gap result.

## Result

- Run: E2_gap_M4_global_feedback_60epoch_seed_43
- Target: gap
- Target index: 4
- Validation MAE: 90.819 meV
- Test MAE: 91.599 meV

## Comparison

- Multitask M4D gap test: approximately 224 meV
- S0 M4 10 epochs: approximately 180 meV
- S1 M4 30 epochs: approximately 123 meV
- S2 M4 60 epochs: approximately 103.845 meV
- E1 M4 + read-only global token: approximately 99.747 meV
- E2 M4 + read-write global token: 91.599 meV

## Interpretation

The read-write global token improves over both the no-global M4 and the read-only global token. This supports the hypothesis that electronic/orbital targets such as gap require stronger global molecular context.

## Decision

Promote E2 as best current gap architecture. Next: test E3 with two global-feedback layers.
