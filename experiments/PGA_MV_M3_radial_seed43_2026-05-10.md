# PGA Multivector Transformer M3 — Radial Edge Features, Seed 43

## Status

Improves over M2. PGA/MV path is now a plausible experimental backbone.

## Commit

3dd614e9237f003946546b01b2b396444f856f93

## Result

- Run: PGA_MV_M3_radial_seed_43
- Best epoch: 10
- Best normalized MAE: 0.2511508167
- Best raw MAE: 79.823250

## Comparison

- PGA_MV_B0 dense scaffold: 0.3611997366
- PGA_MV_M1 edge-aware attention: 0.2790567875
- PGA_MV_M2 edge message transport: 0.2603588700
- PGA_MV_M3 radial edge features: 0.2511508167
- TinyRadialMPNN FULL10 A0 seed 43: 0.2242598236

## Interpretation

Adding radial basis edge features improves the edge-message PGA/MV scaffold. The model is still behind TinyRadialMPNN but the architecture is progressing consistently.

The best epoch is epoch 10, suggesting M3 may still benefit from longer training or a learning-rate schedule.

## Decision

Proceed to M4: add vector channels and grade-aware value transport.
