# U0_atom direct stochastic-CB residual artifact lock — 2026-05-24

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

## Post-hoc residual artifact

results_local/DIRECT_STOCH_CB_RESIDUAL_HEAD_U0atom_MOTOR_V4A_2400_s8_h64_seed43_TESTGATED/best_residual_head.pt

## Feature cache

results_local/cache/stoch_cb_posneg_minor2_s8_seed43.pt

## Local package

artifacts_local/U0_atom_direct_stoch_cb_residual_2026-05-24/

## Method

Post-hoc graph-level stochastic Cauchy-Binet residual correction.

Feature recipe:

minor_k: 2
samples_per_center: 8
feature_dim: 51
hidden_dim: 64

## Validation result

base_val_mev: 47.232383728027344
cb_val_mev: 47.146610260009766
delta_val_mev: -0.08577346801757812
best_epoch: 82

## Gated test result

base_test_mev: 51.580684661865234
cb_test_mev: 51.50022888183594
delta_test_mev: -0.08045578002929688

## Packaged evaluator reproduction

The packaged evaluator reproduced the test result:

base_mae_mev: 51.58071517944336
corrected_mae_mev: 51.50029373168945
delta_mev: -0.08042144775390625

## Decision

Direct stochastic-CB graph-level residual correction is artifact-locked as the strongest current CB path.

It improves both validation and validation-gated test on the project split.

## Non-claims

No SOTA claim is made.

This is a local artifact-backed result on the project split, not a public leaderboard claim.

Large artifacts are not committed to GitHub.
