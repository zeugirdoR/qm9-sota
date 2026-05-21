# System76 CB residual POC positive result — 2026-05-18

## Claim status

Local System76 proxy result only. Not production. Not SOTA.

## Setup

```text
target: U0_atom
metric: target-specific validation MAE
unit: meV
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
protocol: recovered System76 local proxy
CB method: scalar residual head
CB fusion: pred = M4_pred + sigmoid(gate) * CB_residual
gate_init: -10.0
