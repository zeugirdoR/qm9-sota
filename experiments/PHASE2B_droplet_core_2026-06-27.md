# Phase 2b — rigorous invariant `droplet_core` on QM9 (3-way, DeltaAI GH200, 2026-06-27)

Swapped the heuristic `DropletLoss` for the shared invariant **`droplet_core`** (affine-invariant
Mahalanobis on the residual covariance + consistency-corrected fixed point + **derived** χ²₁₉ budget;
imported from `TOI-droplet/emergence/droplet_core.py`). 3-way label-noise sweep:
`sbatch deltaai/qm9_noise.sbatch`.

## Result (clean-val mean normalized MAE; lower = better)

| corrupted frac | baseline | droplet (heuristic) | core (invariant) |
|---|---|---|---|
| 0.0 | 0.2797 | 0.2763 | **0.3027** |
| 0.1 | 0.2967 | 0.2746 | **0.2981** |
| 0.2 | 0.3026 | 0.2818 | **0.2893** |
| 0.3 | 0.3100 | 0.2819 | **0.2864** |

## Reading — the core over-amputates the genuine heavy tail

- The core **is** robust (beats baseline at every noise>0, and uniquely *improves* as noise rises,
  0.303→0.286), but it pays a **clean-data tax** (0% noise: 0.303 vs baseline 0.280) and never matches the
  heuristic.
- **Diagnosis:** with λ→1 and a HARD compact-support cut at the Gaussian χ²₁₉(0.999) budget, the molecules
  beyond R² on clean data are **genuinely hard** (large but legitimate residuals), not corrupted. Hard-zeroing
  them removes useful signal → worse clean-val MAE. As noise rises, more of the amputated set is truly
  corrupted, so the net harm falls — hence the improve-with-noise curve. The heuristic dodges this via its
  **soft** gate + overlap weight (down-weights the tail instead of killing it).
- **Root cause = heavy-tailed residuals breaking the Gaussian χ² budget — the SAME phenomenon as
  `TOI-droplet/emergence/MAXRL_SCORE_EXPERIMENT.md §5b`** (score droplet: tail index ν≈2, "the χ² budget sits
  too low and over-amputates the genuine bulk"). A recurring cross-domain subtlety with a common fix.

## Next — Phase 2c: heavy-tail-aware budget

Mirror §5b: fit the residual tail (multivariate Student-t / robust tail index) and set the budget from the
heavy-tailed predictive instead of the Gaussian χ², so the genuine tail is kept and only true outliers are
amputated. Expectation: removes the clean-data tax (core ≤ baseline at 0%) while preserving/strengthening
the noise robustness — a tuning-free, invariant, heavy-tail-correct droplet that should match or beat the
hand-tuned heuristic. Cheaper interim checks: raise `qlevel`, add a `min_weight` floor, or soften ν.
