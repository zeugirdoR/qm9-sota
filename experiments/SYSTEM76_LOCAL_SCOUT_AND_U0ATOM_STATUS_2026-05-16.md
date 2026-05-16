# System76 local scout + U0_atom M4 300-epoch status — 2026-05-16

## Facts

Repository:

```text
repo: github.com/zeugirdoR/qm9-sota
branch: feature/pga-multivector-attention
```

Project source-of-truth policy:

```text
GitHub: code, configs, documentation, small result summaries
Colab: disposable GPU runner
Google Drive: checkpoints and large outputs
```

System76 local scout machine:

```text
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
VRAM: about 8.19 GB
torch: 2.5.1+cu121
torch cuda: 12.1
PyG: 2.7.0
driver observed: 595.58.03
CUDA driver observed: 13.2
```

User operational note:

```text
System76 can run small/local tests at roughly 3x Colab speed for small tasks.
Treat this as an operational observation, not as a full-production benchmark.
```

System76 caveat:

```text
Local proxy tests use smoke/smaller splits and local disk.
They are useful for ranking trends, debugging, config validation, and wrapper testing.
They are not numerically equivalent to full QM9/A100 production runs.
```

Known local issue:

```text
Python may hang after printing final training summary.
Workaround: run with timeout and/or kill scripts/train.py from another terminal.
QM9_FORCE_EXIT=1 may be useful.
```

## Decisions

1. Use System76 aggressively as the local scout/proxy/debug machine.
2. Use Colab/A100 for production confirmation and full-split long runs.
3. Long production runs must remain Drive-backed and checkpointed.
4. Document all meaningful scout/proxy outcomes in GitHub before using them to justify A100 runs.
5. Do not make SOTA claims from System76 smoke/proxy results.

## Current U0_atom production line

Active target:

```text
target: U0_atom
target_index: 12
metric: target-specific MAE
units: meV
conversion: raw PyG eV × 1000
```

Active model:

```text
plain M4 / pga_multivector_transformer
hidden_dim: 128
num_layers: 4
num_rbf: 32
cutoff: 8.0
vector_channels: 8
head_mode: single
no CB
no motor
no bigger head
```

Completed or partially completed artifact-backed runs as of this note:

```text
M4_U0atom_300epoch_seed_42:
  val U0_atom MAE: 81.06921977996826 meV
  test U0_atom MAE: 77.10036442513369 meV
  test n: 10831
  status: artifact-backed, val/test evaluated

M4_U0atom_300epoch_seed_43:
  best epoch: 248
  best val U0_atom MAE: 72.68887758255005 meV
  test U0_atom MAE: 75.60250677898226 meV
  test n: 10831
  status: artifact-backed, test evaluated; separate val eval JSON still optional

M4_U0atom_300epoch_seed_44:
  val U0_atom MAE: 77.79751682281494 meV
  val n: 10000
  status: validation evaluated; test eval pending
```

Interim production aggregates:

```text
2-seed test aggregate, seeds 42 and 43 only:
  seed42 test: 77.10036442513369 meV
  seed43 test: 75.60250677898226 meV
  mean: about 76.3514 meV
  sample std: about 1.0591 meV

3-seed validation aggregate, using seed43 summary val:
  seed42 val: 81.06921977996826 meV
  seed43 val: 72.68887758255005 meV
  seed44 val: 77.79751682281494 meV
  mean: about 77.1852 meV
  sample std: about 4.2236 meV
```

## Hypotheses

1. System76 can reduce iteration latency for small debugging/scouting tasks.
2. Local smoke/proxy rankings may help decide which mechanisms deserve A100 confirmation.
3. The proxy-to-full-run correlation still needs explicit validation.

## Open questions

1. What is the seed44 test MAE for the completed 300-epoch M4 U0_atom run?
2. Does the 3-seed test mean fall below 75 meV?
3. Does checkpoint averaging over late 300-epoch checkpoints improve U0_atom test MAE?
4. Should System76 have a standardized smoke config and timeout wrapper committed to the repo?
5. Does System76 proxy ranking reliably predict full-QM9/A100 ranking across mechanisms?

## Reproducibility status

No SOTA claim is made here.

Current status:

```text
U0_atom M4 300-epoch:
  seed 42: artifact-backed, val/test evaluated
  seed 43: artifact-backed, test evaluated
  seed 44: val evaluated, test pending
  3-seed test mean/std not complete
```

System76 status:

```text
documented operational resource
useful for scout/proxy/debugging
not yet validated as production-equivalent
```
