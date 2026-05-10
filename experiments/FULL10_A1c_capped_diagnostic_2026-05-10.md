# FULL10 A1c Capped Droplet Diagnostic

## Status

A1c is not promoted.

## Results

### Seed 43

- A0 norm MAE: 0.2242598236
- A1b norm MAE: 0.2273432761
- A1c norm MAE: 0.2251697779

A1c substantially improves over A1b on seed 43, but still loses to A0.

### Seed 42

- A0 norm MAE: 0.2340504974
- A1b norm MAE: 0.2289208174
- A1c norm MAE: 0.2353925258

A1c loses to A0 and loses badly to A1b on the primary normalized metric.

A1c has lower raw MAE on seed 42, but normalized MAE remains the primary metric.

## Interpretation

A1c confirms that the 10-epoch A1b schedule can be too strong for some seeds, but the capped/slower schedule does not improve the overall tradeoff.

## Decision

Stop A1c. Keep A1b as the current Droplet schedule.

Next:

- perform per-target diagnostics for FULL10 seed 42 and seed 43
- stop scalar schedule tuning on TinyRadialMPNN
- move toward stronger backbone experiments
