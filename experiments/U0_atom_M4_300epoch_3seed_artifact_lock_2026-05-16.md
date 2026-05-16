# U0_atom M4 300-epoch 3-seed artifact lock — 2026-05-16

## Facts

Repository:

```text
repo: github.com/zeugirdoR/qm9-sota
branch: feature/pga-multivector-attention
git_commit recorded in run metadata: 30a26bd9df9ad536ba16762c937cf17347eb6fc5
```

Target:

```text
target: U0_atom
target_index: 12
metric: target-specific MAE
raw PyG units: eV
reported units: meV
conversion: raw eV × 1000
```

Dataset/split protocol used by project configs:

```text
dataset: PyG QM9
train_size: 110000
val_size: 10000
test_size: 10831
```

Model:

```text
plain M4 / pga_multivector_transformer
hidden_dim: 128
num_layers: 4
num_rbf: 32
cutoff: 8.0
vector_channels: 8
head_mode: single
attention_mode: edge
edge_feature_mode: radial
no Cauchy-Binet bias
no motor residual
no bigger head
```

Input information:

```text
atom/node graph features
molecular graph edges
3D positions
radial edge features
vector-channel geometric message passing
```

Runtime/artifact storage:

```text
production GPU recorded in metadata: NVIDIA A100-SXM4-40GB
torch_cuda_version recorded in metadata: 12.8
artifact storage: /content/drive/MyDrive/qm9-sota-results
```

## Experimental results

### M4_U0atom_300epoch_seed_42

```text
val U0_atom MAE: 81.06921977996826 meV
val source: single_target_U0_atom_val_eval.json
test U0_atom MAE: 77.10036442513369 meV
test n: 10831
test source: single_target_U0_atom_test_eval.json
status: artifact-backed, val/test evaluated
```

### M4_U0atom_300epoch_seed_43

```text
val U0_atom MAE: 72.68887758255005 meV
val source: summary.json
test U0_atom MAE: 75.60250677898226 meV
test n: 10831
test source: single_target_U0_atom_test_eval.json
status: artifact-backed, test evaluated; val from summary
```

### M4_U0atom_300epoch_seed_44

```text
val U0_atom MAE: 77.79751682281494 meV
val source: single_target_U0_atom_val_eval.json
test U0_atom MAE: 77.33998612544839 meV
test n: 10831
test source: single_target_U0_atom_test_eval.json
status: artifact-backed, val/test evaluated
```

Final 3-seed test aggregate:

```text
seed42 test: 77.10036442513369 meV
seed43 test: 75.60250677898226 meV
seed44 test: 77.33998612544839 meV

mean test MAE: 76.68095244318812 meV
sample std: 0.9416147979926457 meV
population std: 0.7688252631119467 meV
```

Validation aggregate:

```text
seed42 val: 81.06921977996826 meV
seed43 val: 72.68887758255005 meV
seed44 val: 77.79751682281494 meV

mean val MAE: 77.18520472844442 meV
sample std: 4.2235918851306415 meV
population std: 3.4485483334432576 meV
```

## Decisions

1. This is the current artifact-backed internal M4 baseline for U0_atom.
2. Do not describe this as SOTA.
3. Use target-specific physical MAE in meV for U0_atom decisions.
4. Keep plain M4 as the locked production baseline before testing new mechanisms.
5. Use Colab/A100 for production confirmation.
6. Use System76 for local scout/proxy/debug runs only.

## Hypotheses

1. Late-checkpoint averaging may improve U0_atom test MAE.
2. Continuing beyond 300 epochs may help, but it may also overfit.
3. Lower-LR continuation from epoch 300 may be more stable than continuing at the original LR.
4. System76 smoke/proxy rankings may help choose mechanisms for A100 confirmation, but this is not yet validated.

## System76 local scout note

System76 local machine:

```text
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
VRAM: about 8.19 GB
torch: 2.5.1+cu121
PyG: 2.7.0
```

Operational note:

```text
The System76 machine can run small/local tests at roughly 3x Colab speed for small tasks.
Use it for smoke tests, debugging, config validation, checkpoint-averaging scripts, and proxy ranking.
Do not treat System76 smoke/proxy results as production-equivalent to full QM9/A100 results.
```

Known local issue:

```text
Python may hang after printing final training summary.
Use timeout wrappers or kill scripts/train.py from another terminal if needed.
QM9_FORCE_EXIT=1 may be useful.
```

## Reproducibility status

```text
3 production seeds complete.
All reported test values are artifact-backed.
Val/test eval JSONs are written for seeds 42 and 44.
Seed43 test eval JSON is written.
Seed43 validation number currently comes from summary.json; a separate val eval JSON is optional for symmetry.
```

## Open questions

1. Should seed43 val eval JSON be generated for symmetry with seeds 42 and 44?
2. Does late-checkpoint averaging improve the 3-seed mean?
3. Does continuation beyond 300 epochs improve or overfit?
4. Does a lower-LR continuation from epoch 300 help?
5. What verified external SOTA table should be used for final comparison?
6. Does System76 proxy ranking reliably predict full QM9/A100 ranking across mechanisms?

## Claim status

No SOTA claim is made here.

This document reports an internal artifact-backed result only.
