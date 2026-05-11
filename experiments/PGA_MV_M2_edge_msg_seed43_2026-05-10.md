# PGA Multivector Transformer M2 — Edge Message Transport, Seed 43

## Status

Large improvement over M1, still below TinyRadialMPNN baseline.

## Commit

78d579ee8bd4a42adc2e94f0c6dedb270ec57ef8

## Result

- Run: PGA_MV_M2_edge_msg_seed_43
- Best epoch: 9
- Best normalized MAE: 0.2603588700
- Best raw MAE: 86.350479

## Comparison

- PGA_MV_B0 dense scaffold: 0.3611997366
- PGA_MV_M1 edge-aware attention: 0.2790567875
- PGA_MV_M2 edge message transport: 0.2603588700
- TinyRadialMPNN FULL10 A0 seed 43: 0.2242598236

## Interpretation

Adding MPNN-style value transport to the edge-aware PGA/MV attention significantly improves performance.

The model is still weaker than TinyRadialMPNN. The likely missing pieces are richer radial edge features, actual vector/bivector channel updates, and Cauchy-Binet information-volume features.

## Decision

Proceed to M3: radial basis edge features.
