# System76 CGA residual POC negative result — 2026-05-18

## Claim status

Local System76 proxy result only. Not production. Not SOTA.

## Setup

```text
target: U0_atom
metric: target-specific validation MAE
unit: meV
commit: d2ec9b61676186ea0883a8fab499bcb18824d93c
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
protocol: recovered System76 local proxy
```

## Results

```text
Plain M4 20epoch:
  best_epoch: 20
  best_val_norm_mae: 0.12577497959136963
  best_val_U0_atom: 1299.1760969161987 meV

CGA residual v2 20epoch:
  best_epoch: 20
  best_val_norm_mae: 0.14507952332496643
  best_val_U0_atom: 1498.5798597335815 meV

Plain M4 50epoch:
  best_epoch: 49
  best_val_norm_mae: 0.05167662724852562
  best_val_U0_atom: 533.787190914154 meV

CGA residual v2 50epoch:
  best_epoch: 46
  best_val_norm_mae: 0.0729803740978241
  best_val_U0_atom: 753.8415193557739 meV
```

## Decision

```text
CGA graph_h fusion v1:
  rejected earlier as catastrophic

CGA scalar residual v2:
  stable but not promoted
  50epoch result is 220.0543284416199 meV worse than matched plain M4
  about 41.2% worse relative to matched plain M4
```

## Interpretation

The current CGA graph-level six-feature residual is too weak or too coarse. It does not justify 100epoch local continuation or A100 promotion.

Possible future CGA directions:

```text
1. Per-atom CGA features instead of only graph-level summaries.
2. Cached CGA features to avoid runtime overhead.
3. CGA residual trained on top of a frozen strong M4 checkpoint.
4. Better local blade/volume features.
```

## Next step

Try a CB / information-volume residual head with the same safe design:

```text
pred = M4_pred + tiny_gate * CB_residual_head(CB_features)
```

Do not use CB attention bias yet. Do not enable motors yet.
