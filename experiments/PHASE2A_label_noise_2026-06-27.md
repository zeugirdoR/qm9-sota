# Phase 2a — Droplet robustness to label noise (DeltaAI GH200, 2026-06-27)

**First demonstrable droplet win on QM9.** With a fraction of training targets corrupted, the
bounded-influence droplet loss amputates the corrupted molecules and holds clean-validation MAE roughly
flat, while the baseline degrades.

## Setup

- Smoke split: 20,000 train / 2,000 **clean** validation; `TinyRadialMPNN`; 10 epochs; seed 42.
- Label noise: a fixed fraction of **train** targets corrupted by **+5σ** Gaussian (per-target, in
  normalized space). `QM9_NOISE_SEED=0` → the *same* molecules corrupted for baseline vs droplet at each
  level (fair). Validation is never corrupted.
- Losses: **baseline** = smooth-L1; **droplet** = `droplet_robust` (fully engaged: λ_end = ρ_end = 1.0, so
  the compact-support weight actually amputates).
- Reproducible: `sbatch deltaai/qm9_noise.sbatch` → `results/noise_robustness.png` + `deltaai/plot_noise.py`.

## Result (clean-val mean normalized MAE; lower = better)

| corrupted frac | baseline | droplet | droplet advantage |
|---|---|---|---|
| 0.0 | 0.2797 | 0.2763 | +0.0034 (no tax) |
| 0.1 | 0.2967 | 0.2746 | **+0.0221** |
| 0.2 | 0.3026 | 0.2818 | **+0.0209** |
| 0.3 | 0.3100 | 0.2819 | **+0.0280** |

- **Droplet ~flat** (0.276 → 0.282) across 0→30% corruption; **baseline degrades monotonically**
  (0.280 → 0.310). The advantage grows with the corruption rate.
- **Headline:** at **30% of training labels corrupted (5σ)**, the droplet (0.282) ≈ the baseline on **clean**
  data (0.280) — bounded influence absorbs the corruption.
- **No tax at 0% noise** (droplet ≤ baseline), so the robustness is not bought with clean-data accuracy.

## Honest scope

- Smoke scale + toy `TinyRadialMPNN` backbone; numbers are not competitive QM9 accuracy (that's Phase 1).
- This is still the **heuristic** `DropletLoss` (residual-energy `I_δ`, *tuned* ρ budget), **not** the
  rigorous invariant droplet core (affine-invariant Mahalanobis + consistency fixed point + *derived* χ²).
  Phase 2b swaps that in — expected to sharpen the amputation and let us drop the tuned schedule.
- The baseline is smooth-L1 (already mildly outlier-robust), so this gap is a **conservative lower bound**;
  an MSE baseline would widen it.
- Mirrors the LLM-alignment reward-hack result (`TOI-droplet/emergence/MAXRL_SCORE_EXPERIMENT.md`): the same
  compact-support amputation, in a new domain — "alignment of a different kind."
