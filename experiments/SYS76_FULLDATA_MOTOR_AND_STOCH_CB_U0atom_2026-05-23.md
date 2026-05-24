
## Stochastic CB robustness sweep update

Additional offline validation-only stochastic CB volume residual checks were run on:

- base checkpoint: SYS76_FULL_M4_MOTOR_V4A_CONTINUE_U0atom_2400epoch_batch256_seed43
- base validation MAE: approximately 47.2324 meV

Results:

| Variant | Best epoch | Best val MAE meV | Delta val meV |
|---|---:|---:|---:|
| minor2, samples4, hidden64 | 82 | 47.16499328613281 | -0.06740188598632812 |
| minor2, samples4, linear | 7 | 47.212738037109375 | -0.019672393798828125 |
| minor2, samples4, hidden16 | 93 | 47.21593475341797 | -0.01645660400390625 |
| minor2, samples8, hidden64 | 82 | 47.14659881591797 | -0.08578872680664062 |
| minor3, samples4, hidden64 | 94 | 47.20347213745117 | -0.02892303466796875 |

Interpretation:

- All stochastic CB variants improved validation.
- Linear residual also improved, so the volume summaries themselves carry residual signal.
- minor2 pair-volume summaries currently outperform minor3 triple-volume summaries.
- More stochastic samples helped: samples8 beat samples4.

Decision:

Promote stochastic CB positive-negative volume residual from one-off scout to online auxiliary-loss candidate.

Next implementation target:

loss = supervised_loss + lambda_cb_contrast * softplus(margin - volume_pos + volume_neg)

No test-set claim is made from these offline validation-only residual scouts.

## Critical correction — scheduled motor was inactive in main runner

An audit showed that scripts/train.py and src/qm9sota/train/runner.py do not call model.set_epoch_float(...).
Therefore current_epoch_float likely remains 0.0 throughout main-runner training.

Because motor_lambda() uses current_epoch_float and motor_warmup_epochs, the scheduled motor path is likely inactive in the main-runner motor-v4a experiments.

No-motor control confirmed this:

| Run | Best epoch | Val MAE meV |
|---|---:|---:|
| SYS76_FULL_M4_MOTOR_V4A_FROM400_U0atom_900epoch_batch256_seed43 | 900 | 51.40332505106926 |
| SYS76_FULL_M4_NOMOTOR_FROM400_U0atom_900epoch_batch256_seed43 | 900 | 51.40348523855209 |

Difference: +0.0001601874828338623 meV.

Decision:

Relabel the previous motor-v4a main-runner line as full-data batch256 M4 continuation from M4-400, not confirmed active motor curriculum.

The strong validation result remains real, but its interpretation changes.

Current best validated local line remains:

SYS76_FULL_M4_MOTOR_V4A_CONTINUE_U0atom_2400epoch_batch256_seed43 = 47.232404351234436 meV

Interpreted as: full-data batch256 long M4 continuation from M4-400, likely with scheduled motor inactive.

Next required engineering step:

Patch runner.py to call model.set_epoch_float(...) during training and to evaluate with explicit eval_epoch, then retest active motor and online CB priors.

## Direct stochastic-CB residual head gated test result

The direct stochastic-CB residual-head variant was rerun and then evaluated with a validation-gated test pass.

Base checkpoint:

- SYS76_FULL_M4_MOTOR_V4A_CONTINUE_U0atom_2400epoch_batch256_seed43

Configuration:

- feature cache: results_local/cache/stoch_cb_posneg_minor2_s8_seed43.pt
- minor_k: 2
- samples_per_center: 8
- feature_dim: 51
- hidden_dim: 64
- epochs: 150
- lr: 0.0003
- weight_decay: 0.0001

Validation result:

- base_val_mev: 47.232383728027344
- cb_val_mev: 47.146610260009766
- delta_val_mev: -0.08577346801757812
- best_epoch: 82

Gated test result:

- base_test_mev: 51.580684661865234
- cb_test_mev: 51.50022888183594
- delta_test_mev: -0.08045578002929688

Saved residual head:

- results_local/DIRECT_STOCH_CB_RESIDUAL_HEAD_U0atom_MOTOR_V4A_2400_s8_h64_seed43_TESTGATED/best_residual_head.pt

Decision:

Direct stochastic-CB graph-level residual correction is validated as the strongest current CB path. It improves both validation and gated test. This should be promoted to a model-integrated graph-level residual module or packaged as a post-hoc artifact.

Non-claim:

No SOTA claim is made here. This is a validation-gated local test result on the project split.
