# System76 U0_atom smoke wave 1 — 2026-05-16

## Purpose

Scout candidate changes locally on System76 before promoting anything to Colab/A100.

## Claim status

No SOTA claim. No production claim. These are local smoke/proxy tests only.

## Target

```text
target: U0_atom
target_index: 12
metric: best validation target MAE
units: meV
raw PyG units: eV
conversion: eV × 1000
```

## Local smoke protocol

```text
seed: 43
epochs: 80
smoke_train_size: 20000
smoke_val_size: 2000
batch_size: 64
results_dir: /home/crod569/qm9-sota-local-results
data_root: /home/crod569/data/QM9
```

## Candidate configs

- `BASE_M4`
  - config: `configs/train/sys76_smoke/U0_atom_BASE_M4_smoke80_seed43.yaml`
  - run_name: `SYS76_U0atom_BASE_M4_smoke80_seed43`
- `M4_dropout_0p03`
  - config: `configs/train/sys76_smoke/U0_atom_M4_dropout_0p03_smoke80_seed43.yaml`
  - run_name: `SYS76_U0atom_M4_dropout_0p03_smoke80_seed43`
- `M4_global_read`
  - config: `configs/train/sys76_smoke/U0_atom_M4_global_read_smoke80_seed43.yaml`
  - run_name: `SYS76_U0atom_M4_global_read_smoke80_seed43`
- `M4_global_feedback_s0p5`
  - config: `configs/train/sys76_smoke/U0_atom_M4_global_feedback_s0p5_smoke80_seed43.yaml`
  - run_name: `SYS76_U0atom_M4_global_feedback_s0p5_smoke80_seed43`
- `M4_cutoff10`
  - config: `configs/train/sys76_smoke/U0_atom_M4_cutoff10_smoke80_seed43.yaml`
  - run_name: `SYS76_U0atom_M4_cutoff10_smoke80_seed43`
- `M4_vch12`
  - config: `configs/train/sys76_smoke/U0_atom_M4_vch12_smoke80_seed43.yaml`
  - run_name: `SYS76_U0atom_M4_vch12_smoke80_seed43`
- `M4_family_head`
  - config: `configs/train/sys76_smoke/U0_atom_M4_family_head_smoke80_seed43.yaml`
  - run_name: `SYS76_U0atom_M4_family_head_smoke80_seed43`

## Promotion rule

A candidate must beat the local BASE_M4 smoke validation MAE clearly before it deserves A100 confirmation.
Known non-promoted branches such as CB bias, motor residual, M4H, M4L, and M4S are not included in this first wave.

## Results

| Rank | Variant | Best val MAE | Best epoch | Run dir |
|---:|---|---:|---:|---|
| 1 | `BASE_M4` | 691.108525 meV | 77 | `/home/crod569/qm9-sota-local-results/SYS76_U0atom_BASE_M4_smoke80_seed43` |
| 2 | `M4_dropout_0p03` | missing | missing | `/home/crod569/qm9-sota-local-results/SYS76_U0atom_M4_dropout_0p03_smoke80_seed43` |
| 3 | `M4_global_read` | missing | missing | `/home/crod569/qm9-sota-local-results/SYS76_U0atom_M4_global_read_smoke80_seed43` |
| 4 | `M4_global_feedback_s0p5` | missing | missing | `/home/crod569/qm9-sota-local-results/SYS76_U0atom_M4_global_feedback_s0p5_smoke80_seed43` |
| 5 | `M4_cutoff10` | missing | missing | `/home/crod569/qm9-sota-local-results/SYS76_U0atom_M4_cutoff10_smoke80_seed43` |
| 6 | `M4_vch12` | missing | missing | `/home/crod569/qm9-sota-local-results/SYS76_U0atom_M4_vch12_smoke80_seed43` |
| 7 | `M4_family_head` | missing | missing | `/home/crod569/qm9-sota-local-results/SYS76_U0atom_M4_family_head_smoke80_seed43` |

Raw JSON summary:

```text
/home/crod569/qm9-sota-local-results/SYS76_U0atom_smoke_wave1_summary.json
```

