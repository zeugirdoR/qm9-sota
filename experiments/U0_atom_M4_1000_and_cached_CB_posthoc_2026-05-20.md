# U0_atom M4-1000 and cached-CB post-hoc result — 2026-05-20

## Claim status

Internal artifact-backed result only. Not SOTA.

## Evaluation contract

target: U0_atom
target_index: 12
metric: target-specific MAE
unit: meV
split: train 110000 / val 10000 / test 10831
official evaluator: scripts/eval_single_target.py

All offline cached-CB results must reproduce official base test MAE before being trusted.

## Plain M4-1000 result

Run family: M4_U0atom_1000epoch_lr5e7_warmbest

Plain M4 test results:

seed42:
  800 test:  31.896921727875462 meV
  1000 test: 31.651799516569554 meV
  delta:    -0.24512221130590844 meV

seed43:
  800 test:  34.00328980182503 meV
  1000 test: 33.59587196277544 meV
  delta:    -0.40741783904958595 meV

seed44:
  800 test:  30.71318341032332 meV
  1000 test: 30.4091491149784 meV
  delta:    -0.30403429534491977 meV

Aggregate:

800 test mean: approximately 32.204465 meV
1000 test mean: approximately 31.885607 meV
mean delta: approximately -0.318858 meV

Decision: M4_U0atom_1000epoch_lr5e7_warmbest is the current internal plain-M4 best. No SOTA claim.

## Plain M4-1100 plateau probe

Seed43:
  1000 val:  29.929138374328613 meV
  1100 val:  29.86149959564209 meV
  delta val: -0.06763877868652202 meV
  1000 test:  33.59587196277544 meV
  1100 test:  33.53205899124209 meV
  delta test: -0.06381297153335197 meV

Seed42:
  1000 val:  35.45434856414795 meV
  1100 val:  35.40510540008545 meV
  delta val: -0.04924316406249574 meV
  1000 test:  31.651799516569554 meV
  1100 test:  31.650671766533126 meV
  delta test: -0.0011277500364279547 meV

Decision: 1100 LR=2e-7 is a plateau probe. Do not promote yet.

## Cached-CB post-hoc residual on M4-1000

Method:
  base model: M4_U0atom_1000epoch_lr5e7_warmbest
  CB features: cached graph_cb_summary
  k: 4
  feature_dim: 16
  residual model: offline MLP on frozen M4 normalized residuals
  hidden_dim: 16
  lr: 3e-4
  weight_decay: 1e-4

Result:
  base test mean: 31.885604858398438 meV
  CB test mean:   31.856914520263672 meV
  mean delta:    -0.028690338134765625 meV
  seed42 delta: -0.0142364501953125 meV
  seed43 delta: -0.04449462890625 meV
  seed44 delta: -0.027339935302734375 meV

Decision: Cached CB remains consistently positive but tiny on M4-1000.
Use cached CB only. Do not use on-the-fly CB inside forward pass.
Not enough yet to justify full architecture integration.

## Next direction

System76 late motor-v2 scout from strongest local M4-1000 seed43 checkpoint.

Motor-v1 result:
  Plain continuation 350: 76.32052153348923 meV
  Late motor 350:         78.52908968925476 meV

Interpretation: Motor-v1 improved over the old 300 checkpoint but lost to matched plain continuation.
Next motor test should use a much smaller/later gate from a stronger checkpoint.
