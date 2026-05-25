# U0_atom Droplet Heat-Bath local-surface POC — 2026-05-25

## Claim status

Internal diagnostic result. No SOTA claim.

## Motivation

We tested a low-temperature Droplet Heat-Bath idea: perturb equilibrium QM9 geometries slightly, amputate outlier droplets, and compare the model local energy response against a small local emulator/teacher.

The goal is not to assign the true U0 label to perturbed molecules. The goal is to anchor true U0 at equilibrium and use the local neighborhood to regularize the learned energy surface.

## POC setup

- base model: SYS76_FULL_M4_NOMOTOR_ATOMAUX900_THEN_U0ONLY_1500epoch_batch256_seed43
- script: scripts/local_scout/tiny_droplet_heatbath_poc_u0atom.py
- split: train
- n_graphs: 128
- k_droplets: 8
- sigma: 0.01 Å
- droplet keep quantile: 0.75
- max bond stretch threshold: 0.05 Å
- toy emulator: local harmonic bond energy

## Result

All droplets:

- n: 1024
- teacher_delta_abs_mean_mev: 1.9023323059082031
- teacher_delta_abs_p95_mev: 3.2489078044891357
- model_delta_abs_mean_mev: 106.40933227539062
- model_delta_abs_p95_mev: 262.7963562011719
- surface_abs_error_mean_mev: 105.7242202758789
- surface_abs_error_p95_mev: 261.2129211425781
- rmsd_mean_a: 0.016748245805501938
- rmsd_p95_a: 0.01970871351659298
- max_bond_stretch_p95_a: 0.042302463203668594

Kept droplets after amputation:

- n: 768
- teacher_delta_abs_mean_mev: 1.8164868354797363
- teacher_delta_abs_p95_mev: 2.7338919639587402
- model_delta_abs_mean_mev: 105.21273803710938
- model_delta_abs_p95_mev: 260.5122375488281
- surface_abs_error_mean_mev: 104.61212921142578
- surface_abs_error_p95_mev: 258.6097717285156
- rmsd_mean_a: 0.016564423218369484
- rmsd_p95_a: 0.018594130873680115
- max_bond_stretch_p95_a: 0.037862036377191544
- amputated_fraction: 0.25

## Interpretation

The local harmonic teacher says the perturbed droplets are small, with mean local energy shifts around 1.8–1.9 meV. The trained model changes by about 105 meV on the same tiny geometry perturbations. Therefore the model is not locally smooth as an energy surface around the equilibrium geometry.

Droplet amputation successfully reduces geometric and teacher-energy outliers, but it does not by itself reduce the model/teacher local-surface mismatch. This means the next training version should use a very small local-surface regularizer, not high-weight augmentation.

## Decision

Proceed to a sigma sweep and only then a tiny training scout.

Recommended next diagnostics:

- sigma = 0.0025 Å
- sigma = 0.005 Å
- sigma = 0.010 Å

Recommended first loss form:

```text
L = L_U0_anchor
  + lambda_atomaux * L_[13,14,15]
  + lambda_surface * SmoothL1(M(x0 + delta) - M(x0), E_teacher(x0 + delta) - E_teacher(x0))
```

The local-surface contribution should start around 1–3% of the primary U0 loss.

## Non-claims

This POC does not show improved validation/test. It only shows that the local heat-bath signal is measurable and that the current model has large local geometry sensitivity.

## Sigma sweep result

The sigma sweep shows a clear mismatch between the toy local harmonic teacher and the trained model local response.

Kept-droplet summary:

```text
sigma   amput   teacher_mean  teacher_p95  model_mean  model_p95  surferr_mean  surferr_p95  rmsd_p95  stretch_p95
0.0025  0.250   0.114073      0.171863     24.472815   60.974133  24.456366     60.929279    0.004649  0.009473
0.0050  0.250   0.454751      0.683472     49.891872   124.092880 49.794849     124.157784   0.009297  0.018948
0.0100  0.250   1.816487      2.733892     105.212738  260.512238 104.612129    258.609772   0.018594  0.037862
0.0200  0.829   5.415902      7.915277     219.300735  512.925171 216.309525    507.260742   0.036096  0.049513
```

Interpretation:

- The toy teacher scales approximately quadratically with sigma, as expected for a harmonic local energy surface.
- The model response scales approximately linearly with sigma and is orders of magnitude larger than the toy teacher response.
- Therefore the model has a large local first-order geometry sensitivity around equilibrium.
- Droplet amputation cleans geometry/teacher outliers, but does not remove the model local-surface mismatch.

Decision:

The first training version should not use a high-weight full local-surface matching loss. Instead, use a tiny symmetric ±delta stationarity regularizer:

```text
odd_model  = 0.5 * (M(x0 + delta) - M(x0 - delta))
even_model = 0.5 * (M(x0 + delta) + M(x0 - delta)) - M(x0)

L_stationary = SmoothL1(odd_model, 0)
L_curvature  = SmoothL1(even_model, teacher_delta)  # optional, after stationarity is stable
```

Recommended first scout:

- sigma: 0.0025 Å
- perturb_fraction: 0.10–0.25
- one ±delta pair per selected graph
- contribution target: about 1% of primary U0 loss
- stage: 400→500 only, with atomaux [13,14,15] still gentle
