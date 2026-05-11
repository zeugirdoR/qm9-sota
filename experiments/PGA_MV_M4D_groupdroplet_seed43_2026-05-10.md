# PGA Multivector M4D — Group-wise Droplet, Seed 43

## Status

Promising candidate improvement over M4.

## Commit

f0158be93e9953d44fa8c8134ed7a0a8d445aafa

## Result

- Run: PGA_MV_M4D_groupdroplet_seed_43
- Best epoch: 9
- Best normalized MAE: 0.1378957331
- Best raw MAE: 34.472538

## Comparison

- M4 original seed 43: 0.1444279850
- M4 same-commit-ish rerun seed 43: approximately 0.140798
- M6b gated Cauchy-Binet seed 43: 0.1380743235
- M4D group-wise Droplet seed 43: 0.1378957331

## Interpretation

Group-wise Droplet improves seed 43 slightly while keeping effective weights close to 1.0. The regularizer acts gently and does not collapse training.

The target-family Droplet structure is better aligned with previous diagnostics than a single scalar molecule-level Droplet.

## Decision

Run seeds 42, 44, and 45 before promotion.
