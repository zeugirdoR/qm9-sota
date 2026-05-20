# U0_atom M4 1000epoch LR=5e-7 warmbest 3-seed artifact lock — 2026-05-20

## Claim status

Internal artifact-backed result only. Not SOTA.

## Setup

```text
target: U0_atom
target_index: 12
split: train 110000 / val 10000 / test 10831
metric: target-specific MAE
unit: meV
model: plain M4 / pga_multivector_transformer
input: graph + 3D coordinates + radial edge features + vector channels
continuation: from M4_U0atom_800epoch_lr2e6_warmbest
optimizer: fresh AdamW
lr: 5e-7
epochs: 1000
CB/CGA/motor: off
```

## Results

| Seed | Best epoch | Val MAE | Test MAE | Test delta vs 800 | GPU |
|---:|---:|---:|---:|---:|---|
| 42 | 997 | 35.454348564148 | 31.651799516570 | -0.245122211306 | NVIDIA A100-SXM4-40GB |
| 43 | 1000 | 29.929138374329 | 33.595871962775 | -0.407417839050 | NVIDIA A100-SXM4-40GB |
| 44 | 985 | 31.831889152527 | 30.409149114978 | -0.304034295345 | NVIDIA A100-SXM4-40GB |

## Aggregate

```text
val mean: 32.40512536366781
val sample std: 2.806855309014571
test mean: 31.885606864774466
test sample std: 1.606175592587522
mean test delta vs 800: -0.3188581152334713
```

## Decision

```text
Promote M4_U0atom_1000epoch_lr5e7_warmbest to current internal A100/H100 best.
No SOTA claim.
Next: seed43 1100epoch LR=2e-7 pilot, validation first.
```

