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

## Update — RESOLVED with the PaiNN backbone (Phase 1, 2026-06-28)

Re-ran with `OOD_MODEL=painn` (the equivariant backbone, val MAE 0.068). The certificate comes alive.

| metric | toy (TinyRadialMPNN) | **PaiNN** |
|---|---|---|
| geometry-OOD AUC | 0.518 | **0.998** |
| flagged real / corrupted | 0.17 / 0.08 | 0.11 / **1.00** |
| err real / corrupted | 0.238 / 0.280 | 0.074 / **7.50** |
| calibration spearman(score, err) | 0.154 | **0.254** |
| MAE full → keep-80% | worse | 0.0736 → **0.0696 (−5.4%)** |

- **OOD detection is essentially perfect (AUC 0.998):** every off-manifold (jittered-geometry) molecule is
  flagged (corrupted flag 1.00, real 0.11), and the flagged set is exactly where the model is
  catastrophically wrong (error 7.50 vs 0.074, ~100×). On a real equivariant embedding the droplet gives a
  hard, **derived** abstain boundary in chemical-representation space.
- **Selective-prediction calibration is now positive** (spearman 0.254; abstaining on the top-20%
  OOD-score cuts retained MAE 5.4%) — directionally fixed vs the toy, where abstention *hurt*. Still modest
  (the harder in-distribution error-ranking task).
- **Confirms the diagnosis:** the certificate's quality was bounded by representation, exactly as the
  negative result predicted (AUC 0.52 → 0.998 from swapping the backbone alone). The QM9 contribution — a
  derived, invariant OOD/abstain certificate that SOTA regressors lack — now stands.
- *To strengthen calibration further:* a train-restricted **structural** OOD split (size/composition, a
  genuine distribution shift rather than geometry jitter) and a cosine-LR PaiNN for a cleaner embedding.
