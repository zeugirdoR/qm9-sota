# Latent OOD certificate — first attempt (NEGATIVE result), DeltaAI GH200, 2026-06-27

`scripts/ood_certificate.py`: fit `droplet_core` on training graph embeddings (PCA→16, Student-t budget),
score the val set, test calibration + geometry-OOD. **Both metrics came back null — an honest negative.**

## Result (toy `TinyRadialMPNN`, 10 epochs, full data)

| metric | value | verdict |
|---|---|---|
| calibration: spearman(score, err) | 0.154 | near-zero |
| MAE full → keep-80% (abstain top-20% score) | 0.2382 → 0.2414 | **worse** (abstention hurt) |
| OOD AUC (real vs jittered geometry, σ=1.0 Å) | 0.518 | chance |
| flagged real / corrupted | 0.17 / 0.08 | corrupted flagged *less* |
| err real / corrupted | 0.238 / 0.280 | (corrupted is harder, as expected) |

## Diagnosis — representation + probe, not the droplet

- **Geometry jitter is the wrong OOD probe for this model.** `TinyRadialMPNN` uses **squared distances** +
  **mean-pooling**, a pipeline nearly invariant to position noise: jitter washes out under averaging and
  even pulls the pooled embedding toward the centroid (so corrupted are flagged *less*). The corrupted
  molecules never leave the training manifold in embedding space → AUC ≈ 0.5.
- **The toy embedding isn't error-predictive.** A 3-layer mean-pooled 128-dim embedding doesn't encode
  enough structure for distance-from-manifold to track error (spearman ≈ noise).
- The droplet itself was well-behaved (R²=135, ν=9.6, clean fit). **Garbage representation in → garbage
  certificate out.**

## What this tells us / next

1. **A genuine structural OOD test** instead of geometry jitter: a **train-restricted split** — train on
   molecules with ≤ k atoms, test the held-out larger ones (a real distribution shift the embedding magnitude
   should reflect). And/or a composition/scaffold hold-out.
2. **A real backbone (Phase 1).** The certificate's quality is bounded by the representation; an equivariant
   PaiNN/MACE-class embedding (and a learned readout, not mean-pool) should make distance-from-manifold both
   error-predictive and OOD-sensitive. The OOD certificate is effectively **blocked on Phase 1**.

So: the certificate idea is not refuted — it was tested on a representation too weak to support it. The
informative outcome is that the OOD certificate should follow the real backbone, not precede it.
