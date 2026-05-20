# QM9-SOTA State Packet v0.2 — 2026-05-18

## Claim status

Internal project-state document. No SOTA claim is made here.

A result may not be called SOTA unless it includes: target, split, metric, units, input information used, source, and reproducibility status.

## 1. Objective

Target: U0_atom  
Target index: 12  
Metric: target-specific MAE  
Unit: meV  
Primary model: M4 / pga_multivector_transformer

## 2. Evaluation contract

Official evaluator: `scripts/eval_single_target.py`

Official split:

```text
seeded torch.randperm
train: first 110000
val: next 10000
test: final 10831

Then press:

```text
Ctrl-D
