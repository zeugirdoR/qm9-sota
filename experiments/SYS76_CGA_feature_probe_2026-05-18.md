# System76 CGA/Gram feature probe — 2026-05-18

## Claim status

Feature sanity probe only. Not a model result. Not production. Not SOTA.

## Purpose

Test whether simple CGA-inspired Gram/volume features are finite and nontrivially related to `U0_atom` before modifying M4.

## Setup

```text
dataset: PyG QM9
data_root: /home/crod569/data/QM9
target: U0_atom
target_index: 12
molecules_scanned: 5000
CGA point Gram convention: P_i · P_j = -0.5 ||x_i - x_j||^2
```

## Feature summary

| Feature | Mean | Std | Min | Max | Corr with U0_atom raw |
|---|---:|---:|---:|---:|---:|
| `mean_neighbor_dist` | 1.70271 | 0.0722445 | 1.14585 | 2.43503 | 0.529828 |
| `std_neighbor_dist` | 0.419142 | 0.0672788 | 0.18376 | 1.08268 | 0.481259 |
| `min_neighbor_dist` | 1.11833 | 0.0268303 | 0.962089 | 1.32142 | 0.318218 |
| `max_neighbor_dist` | 2.18015 | 0.179281 | 1.32961 | 3.89664 | 0.495898 |
| `logdet_I_plus_Gram` | 4.0839 | 0.184169 | 1.5087 | 4.63722 | -0.36307 |
| `local_volume_proxy` | 1.21813 | 0.565569 | 3.4642e-12 | 2.0377 | -0.463488 |

## Decision

If features are finite and non-degenerate, proceed to a gated M4+CGA readout proof of concept on System76 recovered proxy protocol.

