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

## Phase 2c — heavy-tail (Student-t) budget (`budget: student_t`, R²=D·F_{D,ν}(q), ν-floor 3)

| corrupted frac | baseline | heuristic | core χ² (2b) | **core heavy-tail (2c)** |
|---|---|---|---|---|
| 0.0 | 0.2797 | 0.2763 | 0.3027 | **0.2905** |
| 0.1 | 0.2967 | 0.2746 | 0.2981 | **0.2785** |
| 0.2 | 0.3026 | 0.2818 | 0.2893 | **0.2747** (beats heuristic) |
| 0.3 | 0.3100 | 0.2819 | 0.2864 | **0.2971** (regressed) |

- The heavy-tail budget **cut the clean-data tax** (0.303→0.291) and made the core **competitive**: beats
  baseline at every noise>0, and beats the hand-tuned heuristic at 20%.
- But the tax isn't fully gone (0.291 vs heuristic 0.276 — hard compact support vs the heuristic's soft gate),
  and **30% regressed** (0.286→0.297): at heavy contamination the wider budget + an outlier-inflated robust
  covariance lets some corrupted leak past R².
- **Net:** the rigorous, **tuning-free**, invariant core is now *competitive* with the hand-tuned heuristic
  (both beat baseline; core wins at moderate noise, heuristic more uniformly flat) — without any ρ/temp/λ
  tuning. Its edge is principle (derived budget + invariance + shared-with-alignment code), not yet a
  dominant MAE. Confirms the §5b heavy-tail correction transfers to QM9.

### Remaining calibration (Phase 2d, optional)
- **Residual clean tax** → soften the cut: a `min_weight` floor or a redescending soft weight (raise ν /
  sigmoid gate like the heuristic) instead of the hard `[·]_+`.
- **30% leak** → tighter robust scatter: lower the fixed-point `trim` (e.g. 0.5) and/or shrink the ν-fit
  `keep` fraction so near-boundary corrupted don't inflate the budget.
- Then the higher-impact items: the **latent OOD certificate** (calibrated abstain — a contribution the
  loss-reweighting can't make), a real equivariant backbone (Phase 1), droplet-gated generation (Phase 3).
