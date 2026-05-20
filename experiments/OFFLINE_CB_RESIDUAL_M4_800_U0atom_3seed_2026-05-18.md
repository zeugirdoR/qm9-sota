# Offline cached-CB residual on M4-800 U0_atom — 3-seed result — 2026-05-18

## Claim status

Internal post-hoc residual result only. Not SOTA. Not an end-to-end architecture claim.

## Setup

```text
target: U0_atom
target_index: 12
metric: target-specific MAE
unit: meV
split: train 110000 / val 10000 / test 10831
base model: M4_U0atom_800epoch_lr2e6_warmbest
seeds: 42, 43, 44
CB features: cached graph_cb_summary, k=4, 16 features
residual model: offline MLP on frozen M4 normalized residuals
hidden_dim: 16
lr: 3e-4
weight_decay: 1e-4
