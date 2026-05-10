# A1b Gentle Scheduled Droplet — Scripted Smoke Run

## Status

Single-seed smoke-test win.

## Commit

f040db8b39699daec26eda2c98eb9d0051a1634f

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

## Baseline comparison

A0 scripted baseline:

- Best epoch: 4
- Best mean normalized validation MAE: 0.3140424788
- Best mean raw validation MAE: 82.759056

A1b gentle Droplet:

- Best epoch: 4
- Best mean normalized validation MAE: 0.3138061166
- Best mean raw validation MAE: 82.662117

Difference:

- Normalized MAE delta: -0.0002363622
- Approximate relative improvement: 0.075%
- Raw MAE delta: -0.096939

## Epoch summary

| Epoch | Train loss | Validation normalized MAE | active_fraction | effective_weight_mean | rho_nu_alpha | lambda_droplet | temperature |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.230733 | 0.455442 | 1.000000 | 1.000000 | 0.002500 | 0.000000 | 1.000000 |
| 2 | 0.165209 | 0.360163 | 1.000000 | 0.999129 | 0.003127 | 0.004421 | 1.088420 |
| 3 | 0.136931 | 0.353744 | 1.000000 | 0.995921 | 0.008114 | 0.030305 | 1.606105 |
| 4 | 0.125081 | 0.313806 | 0.999950 | 0.994763 | 0.023116 | 0.078218 | 2.564359 |
| 5 | 0.114954 | 0.343125 | 0.999950 | 0.995494 | 0.054033 | 0.140865 | 3.817298 |

## Interpretation

The gentle Droplet schedule produced a tiny improvement over the scripted A0 baseline. The effect is too small to treat as a robust result, but the schedule is stable and does not collapse the training weights.

Unlike the archived aggressive A1 schedule, this run kept rho, lambda, and temperature modest during the 5-epoch smoke test.

## Decision

Proceed to A2:

- gentle Droplet schedule
- bounded learnable offsets for delta, nu, and alpha
- offsets activated only after warmup
- continue comparing best validation checkpoints
