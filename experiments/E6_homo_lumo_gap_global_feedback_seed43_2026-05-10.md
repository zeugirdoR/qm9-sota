# E6 HOMO/LUMO/GAP Joint Training — Global Feedback, Seed 43

## Status

Diagnostic only. E6 does not improve gap over E2.

## Result

- Run: E6_homo_lumo_gap_global_feedback_60epoch_seed_43
- Targets: HOMO, LUMO, GAP
- Indices: [2, 3, 4]
- Best validation normalized group MAE: 0.079614

## Physical-unit test MAE

- GAP: 95.741 meV
- HOMO: 63.642 meV
- LUMO: 75.226 meV

## Comparison

- E2 gap-only read-write global token: approximately 91.599 meV test gap
- E6 joint HOMO/LUMO/GAP: 95.741 meV test gap

## Interpretation

Joint HOMO/LUMO/GAP training gives a useful electronic diagnostic, but it does not improve gap over the gap-only E2 run. The model likely needs stronger electronic architecture or target-specific training to approach SOTA-level orbital/gap errors.

## Decision

Keep E2 as best current gap model.

Next: move to atomization-energy single-target training, starting with U0_atom.
