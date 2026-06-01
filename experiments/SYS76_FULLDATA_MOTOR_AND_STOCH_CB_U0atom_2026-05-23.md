
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

## Atomaux900 → U0-only 2700 lr5e-8 CB triage

The 2700 lower-LR continuation improved the base only modestly and weakened the post-hoc stochastic-CB residual signal.

```text
locked atomaux2400 + s8_43_202_303:
  val_corrected_mae_mev: 40.116207122802734
  test_corrected_mae_mev: 44.649662017822266

atomaux2700 base:
  best_epoch: 2698
  best_val_target_converted_mae: 40.17318785190582

atomaux2700 + s8_43:
  best_val_mev: 40.11436080932617
  delta_mev: -0.05881500244140625

atomaux2700 + s8_303:
  best_val_mev: 40.104888916015625
  delta_mev: -0.06827926635742188
```

Decision:

```text
Do not run a 2700 full ensemble.
Do not gated-test 2700.
Keep atomaux2400 + s8_43_202_303 as the locked test-backed best.
Move to reproducibility fix and V20/AGAA motor architecture branch.
```

## V20/AGAA mini scout: Coulomb score bias result

The optional Coulomb score bias did not improve the first V20 motor scaffold.

```text
motor off:
  best_val_target_converted_mae: 1328.7139682611553

motor on:
  best_val_target_converted_mae: 467.13072066044606

motor on + Coulomb:
  best_epoch: 22
  best_val_target_converted_mae: 486.8580607960986
  last_epoch: 25
  last_val_mev: 750.3942857343234
```

Decision:

Do not run Coulomb regularization yet. The direct Coulomb score term appears unstable in the current scaffold. Continue with motor-on/no-Coulomb for a longer 100-epoch 20k mini-scout.

## V20/AGAA motor-on 100epoch/20k result

The V20/AGAA motor scaffold is learning, but is not production-ready yet.

```text
motor off 25epoch/20k:
  best_val_target_converted_mae: 1328.7139682611553

motor on 25epoch/20k:
  best_val_target_converted_mae: 467.13072066044606

motor on 100epoch/20k:
  best_epoch: 98
  best_val_target_converted_mae: 218.08758723279988
```

Decision:

Continue from the best 100epoch checkpoint with lr=1e-4 for another 100 epochs. If the 20k scout does not move below ~150–200 meV, patch a target-only head before scaling to full data.

## V20/AGAA target-only 20k result

The target-only U0_atom head improved the V20 motor-on 20k scout.

```text
motor-on 100epoch/20k:
  best_val_target_converted_mae: 218.08758723279988

motor-on 200epoch/20k lr1e-4 from100:
  best_val_target_converted_mae: 149.28511863183314

motor-on target-only 300epoch/20k lr1e-4 from200:
  best_epoch: 97
  best_val_target_converted_mae: 130.99373319542363
  best_val_target_norm_mae: 0.012681675143539906
```

Decision:

Run a target-only continuation at lr=5e-5 before scaling V20 to full data.

## V20/AGAA target-only full-data scout

The V20/AGAA motor-on target-only branch scaled from the 20k scout to full data and reached a meaningful validation level, but it is not yet production-competitive.

```text
20k target-only lr5e-5:
  best_val_target_converted_mae: 99.4523496524886

full-data target-only lr5e-5:
  run: SCOUT_V20_AGAA_D192_L7_H16_MOTORON_TARGETONLY_U0atom_100epoch_FULL_lr5e5_from20k400_seed43
  train_size: 110000
  val_size: 10000
  best_epoch: 98
  best_val_target_converted_mae: 52.84930991925796
  best_val_target_norm_mae: 0.00511641101911664
```

Decision:

Continue full-data V20 target-only from the best checkpoint at lr=2e-5. Do not gated-test V20 yet.

## V20/AGAA target-only full-data lr2e-5 result

The V20/AGAA motor-on target-only branch became a serious validation challenger after full-data continuation.

```text
V20 target-only full-data lr5e-5 from 20k:
  best_epoch: 98
  best_val_target_converted_mae: 52.84930991925796

V20 target-only full-data lr2e-5 from FULL100:
  best_epoch: 97
  best_val_target_norm_mae: 0.003547437721863389
  best_val_target_converted_mae: 36.64280193313974

M4 atomaux2400 + s8_43_202_303 locked validation:
  corrected_mae_mev: 40.116207122802734

M4 atomaux2400 + s8_43_202_303 locked test:
  corrected_mae_mev: 44.649662017822266
```

Decision:

Continue V20 full-data target-only once more at lr=1e-5 from the FULL200 best checkpoint. Then build a V20-specific stochastic-CB residual evaluator before any gated test.

## LOCKED RESULT — V20/AGAA FULL300 lr1e-5 raw gated test

This replaces the previous M4 atomaux2400 + stochastic-CB artifact as the current project-local test-backed U0_atom best.

```text
model:
  V20/AGAA D192/L7/H16 motor-on target-only

run:
  SCOUT_V20_AGAA_D192_L7_H16_MOTORON_TARGETONLY_U0atom_300epoch_FULL_lr1e5_fromFULL200_seed43

validation:
  mae_mev: 30.566791534423828
  p50_mev: 22.564002990722656
  p90_mev: 63.08757781982422
  p95_mev: 81.65106201171875
  p99_mev: 137.33062744140625

gated test:
  mae_mev: 31.027469635009766
  p50_mev: 22.876150131225586
  p90_mev: 65.04935455322266
  p95_mev: 83.0776596069336
  p99_mev: 149.4949188232422

## LOCKED RESULT — V20/AGAA FULL400 lr5e-6 raw gated test

This replaces the previous V20 FULL300 lock and the older M4 atomaux2400 + stochastic-CB artifact.

```text
model:
  V20/AGAA D192/L7/H16 motor-on target-only

run:
  SCOUT_V20_AGAA_D192_L7_H16_MOTORON_TARGETONLY_U0atom_400epoch_FULL_lr5e6_fromFULL300_seed43

validation:
  mae_mev: 27.909137725830078
  p50_mev: 20.468231201171875
  p90_mev: 57.328834533691406
  p95_mev: 74.0921630859375
  p99_mev: 130.9029083251953

gated test:
  mae_mev: 28.700634002685547
  p50_mev: 20.857187271118164
  p90_mev: 59.39204406738281
  p95_mev: 77.71495056152344
  p99_mev: 141.37136840820312

## LOCKED RESULT — V20/AGAA FULL500 lr2e-6 raw gated test

This replaces the previous V20 FULL400 lock.

```text
model:
  V20/AGAA D192/L7/H16 motor-on target-only

run:
  SCOUT_V20_AGAA_D192_L7_H16_MOTORON_TARGETONLY_U0atom_500epoch_FULL_lr2e6_fromFULL400_seed43

summary:
  best_epoch: 99
  best_val_target_converted_mae: 26.367141928239633
  best_val_target_norm_mae: 0.0025526375975459814

## LOCKED RESULT — V20/AGAA FULL600 lr1e-6 raw gated test

This replaces the previous V20 FULL500 lock.

```text
model:
  V20/AGAA D192/L7/H16 motor-on target-only

run:
  SCOUT_V20_AGAA_D192_L7_H16_MOTORON_TARGETONLY_U0atom_600epoch_FULL_lr1e6_fromFULL500_seed43

summary:
  best_epoch: 89
  best_val_target_converted_mae: 25.62825315401618
  best_val_target_norm_mae: 0.0024811048060655594
```

Validation eval:

```text
mae_mev: 25.628252029418945
p50_mev: 18.498905181884766
p90_mev: 52.64536666870117
p95_mev: 69.37175750732422
p99_mev: 122.52284240722656
```

Gated test:

```text
mae_mev: 26.401905059814453
p50_mev: 18.93088150024414
p90_mev: 54.98998260498047
p95_mev: 71.69291687011719
p99_mev: 129.79782104492188
```

Previous locked V20 result:

```text
V20 FULL500 lr2e-6 test_mae_mev: 27.07649040222168
```

Improvement over FULL500:

```text
0.6745853424072266 meV
```

Previous M4+CB locked result:

```text
M4 atomaux2400 + s8_43_202_303 test_mae_mev: 44.649662017822266
```

Improvement over M4+CB:

```text
18.247756958007812 meV
```

Decision:

```text
Lock raw V20 FULL600 lr1e-6 as current project-local test-backed best.
Do not rerun V20 stochastic-CB until the residual scout includes zero-correction baseline, zero-initialized final residual layer, and alpha=0 guard.
No public SOTA claim is made; this is project-local split/artifact-backed.
```

## V20/AGAA FULL800 lr2e-7 validation-only result

FULL800 improved the validation checkpoint, but not enough for a new gated test.

```text
model:
  V20/AGAA D192/L7/H16 motor-on target-only

run:
  SCOUT_V20_AGAA_D192_L7_H16_MOTORON_TARGETONLY_U0atom_800epoch_FULL_lr2e7_fromFULL700_seed43

summary:
  best_epoch: 89
  best_val_target_converted_mae: 25.165587584074387
  best_val_target_norm_mae: 0.002436313545331359
```

Current locked test-backed result:

```text
V20 FULL600 lr1e-6 test_mae_mev: 26.401905059814453
```

Decision:

```text
Keep FULL600 as locked test-backed best.
Keep FULL800 as best validation checkpoint.
Launch FULL900 lr1e-7 validation-only from FULL800 best.
```

## V22/PGA-GP-M0 and triplet-plane scout plan

The first V22/PGA geometric-product motor attention scout is alive and improves the early 20k motor-attention regime.

```text
V20 motor-on 25epoch/20k:
  best_val_target_converted_mae: 467.13072066044606

V21-M2 25epoch/20k:
  best_val_target_converted_mae: 388.23448176621156

V22-PGA-GP-M0 25epoch/20k:
  run: SCOUT_V22_PGA_GP_MOTOR_D192_L7_H16_TARGETONLY_U0atom_25epoch_20k_seed43
  best_epoch: 25
  best_val_target_converted_mae: 372.58154813093824
  best_val_target_norm_mae: 0.03607010841369629
```

Decision:

```text
Continue V22-PGA-GP-M0 to 100epoch/20k.
Start PGA triplet-plane cache work.
Compare two plane-native options:
  1. deterministic pair-conditioned triplet-plane summary attention bias
  2. stochastic sampled plane-token attention
Do not introduce CB until the plane/PGA attention path is stable.
```

PGA convention:

```text
Atom point:
  P = e123 + x e023 + y e031 + z e012

Plane:
  pi = d e0 + nx e1 + ny e2 + nz e3
```

## V22 stochastic sampled triplet-plane context scout

The stochastic triplet-plane cache is ready and the first stochastic-plane model adds sampled plane context at graph level.

```text
cache:
  results_local/cache/pga_stochastic_triplet_planes_k128_seed43_fp16.pt
  n: 130831
  k: 128 sampled triplet planes per molecule

model:
  V22-PGA-STOCH-PLANE-M0
  base: V22-PGA-GP motor-product attention
  injection: sampled plane-token context added before final U0 head
  guardrail: plane_context_proj zero-initialized
```

Decision rule:

```text
Compare against V22-GP-M0 25epoch/20k = 372.58154813093824 meV.
Compare against deterministic plane-summary 25epoch/20k = 427.38663300663404 meV.
Continue stochastic-plane if it improves over the V22-GP baseline or gives a late/stable curve.
```

## Small-capacity PGA geometry exploration bench

Motivation:

```text
The D192/L7/H16 models have ~3M parameters and train slowly on Sys76.
To test whether Chem-aware PGA/stochastic-plane features are truly useful, run smaller ~1M-parameter scouts.
If the geometry is right, smaller models should learn faster and expose the signal more clearly.
```

Planned small scouts:

```text
V22-GP-small:
  D128 / L5 / H8
  true PGA motor-product attention
  no plane features

V22-STOCH-PLANE-XATTN-small:
  D128 / L5 / H8
  true PGA motor-product attention
  K=128 sampled triplet planes
  atom-to-plane cross attention

Optional V20-small:
  D128 / L5 / H8
  free q_screw/k_screw motor attention reference
```

Decision logic:

```text
If stochastic-plane-small beats V22-GP-small, sampled planes are useful.
If V20-small beats V22-small badly, PGA motor-product parameterization is harder to optimize.
If small models are competitive with large scouts, continue feature discovery at small capacity before scaling.
```

## Small-capacity stochastic atom-to-plane xattn result

The sampled triplet-plane atom-to-plane cross-attention branch improved the ~1M parameter V22 small baseline.

```text
V22-GP-small D128/L5/H8 50epoch/20k:
  best_val_target_converted_mae: 387.4879715653563

V22-STOCH-PLANE-XATTN-small D128/L5/H8 50epoch/20k b256:
  run: SCOUT_V22_STOCH_PLANE_XATTN_SMALL_D128_L5_H8_TARGETONLY_U0atom_50epoch_20k_b256_seed43
  best_epoch: 43
  best_val_target_converted_mae: 332.13243161629794
  best_val_target_norm_mae: 0.03215417638421059
  trainable_params: 1041891
```

Interpretation:

```text
Sampled triplet-plane information helps under low capacity.
The graph-level plane context and deterministic plane-summary variants were worse, but atom-to-plane attention preserves local geometry and gives a clear gain.
Continue this branch from the best checkpoint with lower LR.
```

## Small-capacity stochastic atom-to-plane xattn continuation result

The sampled triplet-plane atom-to-plane cross-attention branch strongly improves the small-capacity V22 baseline and beats the V20-small reference.

```text
V22-GP-small D128/L5/H8 50epoch/20k:
  best_val_target_converted_mae: 387.4879715653563

V22-STOCH-PLANE-XATTN-small D128/L5/H8 50epoch/20k:
  best_val_target_converted_mae: 332.13243161629794

V22-STOCH-PLANE-XATTN-small D128/L5/H8 100epoch/20k:
  best_epoch: 42
  best_val_target_converted_mae: 240.89310830980892
  best_val_target_norm_mae: 0.023321177810430527

V20-small D128/L5/H8 50epoch/20k:
  best_val_target_converted_mae: 498.16403474786597
```

Interpretation:

```text
Sampled triplet-plane information helps under low capacity.
The useful structure appears to be atom-to-plane relation, not crude plane-summary or graph-level pooling.
Continue the small xattn branch with lower LR and test D192/L7/H16 xattn scale-up.
```

## V22 no-plane small full-data control

The no-plane D128/L5/H8 V22-GP control confirms that sampled triplet-plane atom-to-plane attention gives a substantial full-data advantage.

```text
No-plane V22-GP-small FULL100:
  run: SCOUT_V22_PGA_GP_SMALL_D128_L5_H8_TARGETONLY_U0atom_100epoch_FULL_lr2e5_from20k50_seed43
  best_epoch: 99
  best_val_target_converted_mae: 115.39951027865314
  best_val_target_norm_mae: 0.01117197796702385
  trainable_params: 955281

Plane-xattn V22-small FULL100:
  best_val_target_converted_mae: 96.64120964754375

Plane-xattn V22-small FULL400:
  best_val_target_converted_mae: 67.70952860503287
```

Decision:

```text
Continue the plane-xattn small full-data branch as primary.
Run no-plane FULL200 later only as a control continuation.
No gated test; V22-small remains validation-only and is not production-competitive yet.
```

## V22 motor-curriculum FULL200 and cloud handoff decision

The V22 D192/L7/H16 motor-curriculum branch transferred to the full split and continued improving, but full-data epochs are now too slow for Sys76 iteration.

```text
V22 motor-curriculum StageG 20k:
  best_val_target_converted_mae: 130.5739174072107

V22 motor-curriculum FULL100 from StageG:
  best_val_target_converted_mae: 85.56586501994268

V22 motor-curriculum FULL200 from FULL100:
  run: SCOUT_V22_MOTORCURR_D192_L7_H16_TARGETONLY_U0atom_200epoch_FULL_lr2e6_fromFULL100_seed43
  best_epoch: 98
  best_val_target_converted_mae: 75.5125996390551
  best_val_target_norm_mae: 0.0073104738257825375
  trainable_params: 3036001
```

Current references:

```text
Locked V20 FULL600 raw test:
  26.401905059814453 meV

Best V20 FULL900 validation:
  25.085482037134675 meV

Best V22 small plane-xattn validation:
  65.19399124766379 meV
```

Decision:

```text
Move long full-data continuations to A100/H100-class cloud.
Use Sys76 for smoke tests, 20k scouts, cache generation, documentation, and small validation checks.
Next cloud runs:
  1. V22 motor-curriculum FULL300 lr1e-6 from FULL200 best.
  2. Delayed plane-onset D192 run from stabilized motor checkpoint with plane_context_scale=0.25 or 0.10.
  3. Optional V22 small plane-xattn continuation/control runs.
```
