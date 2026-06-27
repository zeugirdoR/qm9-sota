#!/usr/bin/env python3
"""Collate the label-noise robustness sweep into a table + figure.

Reads results/noise{frac}_{baseline,droplet}/summary.json (clean-val mean normalized MAE) and plots
baseline vs droplet as the corrupted-target fraction grows. The droplet should stay flatter (bounded
influence amputates the corrupted molecules) while the baseline degrades.
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "results"
FRACS = [0.0, 0.1, 0.2, 0.3]


def best(run: str):
    p = ROOT / run / "summary.json"
    if not p.exists():
        return None
    return float(json.load(open(p))["best_val_mean_norm_mae"])


base = [best(f"noise{f}_baseline") for f in FRACS]
drop = [best(f"noise{f}_droplet") for f in FRACS]

print("=" * 60)
print("Label-noise robustness (clean-val mean normalized MAE; lower=better)")
print("=" * 60)
print(f"{'noise_frac':>10}  {'baseline':>10}  {'droplet':>10}  {'droplet_advantage':>18}")
for f, b, d in zip(FRACS, base, drop):
    adv = (b - d) if (b is not None and d is not None) else None
    print(f"{f:>10}  {str(round(b,4) if b else b):>10}  {str(round(d,4) if d else d):>10}  {str(round(adv,4) if adv is not None else adv):>18}")

if all(v is not None for v in base + drop):
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(FRACS, base, "o-", color="C3", lw=2, label="baseline (smooth-L1)")
    plt.plot(FRACS, drop, "o-", color="C0", lw=2, label="droplet (bounded-influence)")
    plt.xlabel("fraction of training targets corrupted (5$\\sigma$)")
    plt.ylabel("clean validation mean normalized MAE")
    plt.title("Droplet robustness to label noise on QM9 (GH200 smoke)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    out = ROOT / "noise_robustness.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)
else:
    print("(some runs missing — figure skipped)")
