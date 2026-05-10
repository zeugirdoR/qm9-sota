# A2 Gentle Droplet + Learnable Offsets — Scripted Smoke Run

## Status

Stable but effectively neutral relative to A1b.

## Runtime

- Device: cuda
- GPU: NVIDIA A100-SXM4-40GB
- Torch CUDA version: 12.8

## Dataset

- PyG QM9 preprocessed dataset
- Molecules: 130,831
- Split: 110,000 train / 10,000 validation / 10,831 test
- Smoke mode:
  - Train batches: 157
  - Validation batches: 16

## Result

A2 gentle learnable offsets:

- Best epoch: 4
- Best mean normalized validation MAE: 0.3138059378
- Best mean raw validation MAE: 82.662048

## Comparison

A0 scripted baseline:

- Best mean normalized validation MAE: 0.3140424788

A1b gentle scheduled Droplet:

- Best mean normalized validation MAE: 0.3138061166

A2 vs A0:

- Normalized MAE delta: -0.0002365410

A2 vs A1b:

- Normalized MAE delta: -0.0000001788

## Epoch summary

| Epoch | Train loss | Validation normalized MAE | active_fraction | effective_weight_mean | rho_nu_alpha | lambda_droplet | temperature | offset_reg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.230733 | 0.455442 | 1.000000 | 1.000000 | 0.002500 | 0.000000 | 1.000000 | 0.0 |
| 2 | 0.165209 | 0.360163 | 1.000000 | 0.999129 | 0.003127 | 0.004421 | 1.088420 | 0.0 |
| 3 | 0.136931 | 0.353744 | 1.000000 | 0.995921 | 0.008114 | 0.030305 | 1.606105 | 0.0 |
| 4 | 0.125081 | 0.313806 | 0.999950 | 0.994762 | 0.023116 | 0.078218 | 2.564359 | 3.67e-09 |
| 5 | 0.114953 | 0.343123 | 0.999950 | 0.995491 | 0.054036 | 0.140865 | 3.817298 | 6.94e-08 |

## Interpretation

The learnable offsets did not materially move during the 5-epoch smoke run. A2 is essentially identical to A1b.

This confirms that bounded learnable offsets are safe under the current settings, but it does not show that they help.

## Decision

Do not tune A2 further yet.

Next step:

- run paired multi-seed smoke tests for A0 vs A1b
- only revisit learnable offsets if A1b is consistently competitive or better across seeds
