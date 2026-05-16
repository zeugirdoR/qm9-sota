# U0_atom M4 300-epoch 3-seed artifact lock — 2026-05-16

## Facts

```text
repo: github.com/zeugirdoR/qm9-sota
branch: feature/pga-multivector-attention
git_commit recorded in run metadata: 30a26bd9df9ad536ba16762c937cf17347eb6fc5
```

## Target / metric

```text
target: U0_atom
target_index: 12
metric: target-specific MAE
raw PyG units: eV
reported units: meV
conversion: raw eV × 1000
split sizes: train 110000 / val 10000 / test 10831
```

## Model

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
input information: atom/node graph features + graph edges + 3D positions + radial edge features + vector channels
no Cauchy-Binet bias
no motor residual
no bigger head
```

## Experimental results

### M4_U0atom_300epoch_seed_42

```text
val U0_atom MAE: 81.06921977996826 meV
val source: single_target_U0_atom_val_eval.json
test U0_atom MAE: 77.10036442513369 meV
test source: single_target_U0_atom_test_eval.json
test n: 10831
status: artifact-backed, val/test evaluated
```

### M4_U0atom_300epoch_seed_43

```text
val U0_atom MAE: 72.68891277313233 meV
val source: single_target_U0_atom_val_eval.json
test U0_atom MAE: 75.60250677898226 meV
test source: single_target_U0_atom_test_eval.json
test n: 10831
status: artifact-backed, val/test evaluated
```

### M4_U0atom_300epoch_seed_44

```text
val U0_atom MAE: 77.79751682281494 meV
val source: single_target_U0_atom_val_eval.json
test U0_atom MAE: 77.33998612544839 meV
test source: single_target_U0_atom_test_eval.json
test n: 10831
status: artifact-backed, val/test evaluated
```

## Final aggregates

```text
mean val MAE: 77.18521645863851 meV
sample std val: 4.223573153644176 meV
population std val: 3.4485330392486024 meV

mean test MAE: 76.68095244318812 meV
sample std test: 0.9416147979926457 meV
population std test: 0.7688252631119467 meV
```

## Decisions

1. This is the current artifact-backed internal M4 baseline for U0_atom.
2. Do not describe this as SOTA.
3. Use target-specific physical MAE in meV for U0_atom decisions.
4. Keep plain M4 as the locked production baseline before testing new mechanisms.
5. Use Colab/A100 for production confirmation.
6. Use System76 for local scout/proxy/debug runs only.

## System76 local scout note

```text
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
VRAM: about 8.19 GB
torch: 2.5.1+cu121
PyG: 2.7.0
role: fast local smoke/proxy/debug machine
caveat: not production-equivalent to full QM9/A100 runs
```

## Open questions

1. Does late-checkpoint averaging improve the 3-seed mean?
2. Does continuation beyond 300 epochs improve or overfit?
3. Does a lower-LR continuation from epoch 300 help?
4. What verified external SOTA table should be used for final comparison?
5. Does System76 proxy ranking reliably predict full QM9/A100 ranking across mechanisms?

## Claim status


No SOTA claim is made here.

This document reports an internal artifact-backed result only.
