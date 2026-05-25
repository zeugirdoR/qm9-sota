# U0_atom direct stochastic-CB residual ensemble artifact lock — 2026-05-24

## Claim status

Internal artifact-backed local result. No SOTA claim.

## Evaluation contract

target: U0_atom
target_index: 12
metric: target-specific MAE
unit: meV
split: train 110000 / val 10000 / test 10831
seed: 43

## Base checkpoint

results_local/SYS76_FULL_M4_MOTOR_V4A_CONTINUE_U0atom_2400epoch_batch256_seed43/best_model.pt

## Ensemble members

- s8_43
- s8_202
- s8_303

Each member is a post-hoc graph-level stochastic Cauchy-Binet residual head.

## Feature recipe

minor_k: 2
samples_per_center: 8
feature_dim: 51
head_hidden_dim: 64
ensemble: equal-weight average
alpha: 1.0

## Validation result

base_val_mev: 47.23240280151367
corrected_val_mev: 47.10929489135742
delta_val_mev: -0.12310791015625

## Gated test result

base_test_mev: 51.580684661865234
corrected_test_mev: 51.4690055847168
delta_test_mev: -0.1116790771484375

## Decision

The s8_43_202_303 equal-weight ensemble replaces the single s8_h64 residual head as the best current direct stochastic-CB artifact.

It improves over the previous single-head artifact test result of approximately 51.50029 meV.

## Non-claims

No SOTA claim is made.

This is a local artifact-backed result on the project split, not a public leaderboard claim.

Large artifacts are not committed to GitHub.
