# Phase 1 — PaiNN equivariant backbone (DeltaAI GH200, 2026-06-28)

Replaced the toy `TinyRadialMPNN` (dist² + mean-pool) with **PaiNN** (Schütt et al. 2021): scalar +
equivariant *vector* features, message/update blocks over the molecular bond graph, learned atomwise
readout. Pure PyTorch+PyG (no `torch_cluster`/`radius_graph`, no new aarch64 dep). Reproducible:
`sbatch deltaai/qm9_painn.sbatch`.

## Result (full QM9, 110k train, 20 epochs, ~4.5 min on one GH200)

- **Equivariance self-test:** rotation+translation max|Δpred| = **2.7e-7** → PASS (checked on the GH200 before training).
- **Best val mean-normalized MAE = 0.0676** (epoch 17); best raw MAE 15.0.

| backbone | val norm-MAE (full data) |
|---|---|
| TinyRadialMPNN (toy) | ~0.238 |
| **PaiNN (equivariant)** | **0.068** (~3.5× better) |

- Training is noisy (constant lr=5e-4, no schedule): val MAE oscillates ~0.07–0.10 over epochs 8–20. A
  cosine schedule + more epochs (and a larger cutoff/radius graph) would push it lower and smoother — but
  0.068 is already a strong, competitive backbone and a ~3.5× upgrade.
- ~0.16 M params at F=64 / ~0.6 M at F=128; trained in minutes — negligible compute.

## Significance

A real equivariant backbone with a learned readout, replacing the debug toy. Beyond the accuracy jump, this
**unblocks the latent OOD certificate** (`OOD_certificate_2026-06-27.md`), whose first attempt failed
because the toy's mean-pooled dist²-embedding was geometry-blind and not error-predictive. PaiNN's
vector features depend on bond directions (so they move under geometry corruption) and its learned readout
should make distance-from-manifold error-predictive. Next: re-run `scripts/ood_certificate.py` with
`OOD_MODEL=painn`.
