# PGA Multivector Transformer M4 — Vector Channels, Seed 43

## Status

Major architectural breakthrough. M4 beats the previous TinyRadialMPNN seed-43 baseline by a large margin.

## Commit

7b8e517bfced91c11d17de6986dc891b5b9bcb90

## Result

- Run: PGA_MV_M4_vector_seed_43
- Best epoch: 8
- Best normalized MAE: 0.1444279849
- Best raw MAE: 33.640693

## Comparison

- PGA_MV_B0 dense scaffold: 0.3611997366
- PGA_MV_M1 edge-aware attention: 0.2790567875
- PGA_MV_M2 edge message transport: 0.2603588700
- PGA_MV_M3 radial edge features: 0.2511508167
- PGA_MV_M4 vector channels: 0.1444279849
- TinyRadialMPNN FULL10 A0 seed 43: 0.2242598236

## Interpretation

Adding equivariant vector channels and feeding invariant vector magnitudes back into scalar updates produced the first major backbone improvement.

The model is now a serious candidate backbone. Before adding Droplet, JEPA, or Cauchy-Binet features, validate M4 across seeds and run invariance/per-target diagnostics.

## Decision

Run M4 on seeds 42 and 44. Then run rotation/translation sanity checks and per-target diagnostics.
