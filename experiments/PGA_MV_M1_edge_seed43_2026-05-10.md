# PGA Multivector Transformer M1 — Edge-Aware Attention, Seed 43

## Status

Large improvement over B0 scaffold, but still below TinyRadialMPNN baseline.

## Commit

ddb732eb7e3defabd5311928a3e89a1085957752

## Result

- Run: PGA_MV_M1_edge_seed_43
- Best epoch: 9
- Best normalized MAE: 0.2790567875
- Best raw MAE: 80.286652

## Comparison

- PGA_MV_B0 dense scaffold: 0.3611997366
- PGA_MV_M1 edge-aware: 0.2790567875
- TinyRadialMPNN FULL10 A0 seed 43: 0.2242598236

## Interpretation

Adding edge-restricted molecular attention and geometric edge features greatly improves the PGA/MV scaffold.

The model is still weaker than TinyRadialMPNN, likely because the value/message pathway does not yet depend jointly on source state, destination state, and distance features.

## Decision

Proceed to M2: edge-aware PGA/MV attention with radial/message value transport.
