#!/usr/bin/env python
"""Latent OOD certificate for QM9 via droplet_core on the graph embedding.

Fit the invariant droplet on TRAINING molecular embeddings -> a hard, DERIVED compact-support boundary in
chemical-representation space. A molecule's Mahalanobis^2 from the predictive t* is its OOD score. Two
results the regression loss cannot provide:
  (1) CALIBRATION / selective prediction: on the real val set, abstaining on the highest-score
      (out-of-support) molecules monotonically reduces MAE -- the certificate knows where the model errs.
  (2) OOD DETECTION: real geometries vs position-corrupted (off-manifold) geometries are separated by the
      score (AUC), and the model's error is higher on the flagged ones.

Env: OOD_EPOCHS, OOD_PCA, OOD_NFIT, OOD_QLEVEL, OOD_BUDGET (gaussian|student_t), OOD_JITTER, OOD_SMOKE.
Output: results/ood_certificate.png + results/ood_certificate.json.
"""
from __future__ import annotations
import os, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qm9sota.data.qm9 import load_qm9_bundle
from qm9sota.models.tiny_radial_mpnn import build_model
from qm9sota.train.evaluate import normalize_y
from qm9sota.losses.droplet_core import toi_fixed_point, _maha2

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)

EPOCHS = int(os.environ.get("OOD_EPOCHS", "10"))
PCA_DIM = int(os.environ.get("OOD_PCA", "16"))
NFIT = int(os.environ.get("OOD_NFIT", "4000"))
QLEVEL = float(os.environ.get("OOD_QLEVEL", "0.999"))
BUDGET = os.environ.get("OOD_BUDGET", "student_t")
JITTER = float(os.environ.get("OOD_JITTER", "1.0"))

_MODEL = os.environ.get("OOD_MODEL", "painn")     # PaiNN by default — the toy backbone was the bottleneck
_model_cfg = {"name": _MODEL, "hidden_dim": 128, "num_layers": 3, "out_dim": 19}
if _MODEL == "painn":
    _model_cfg.update({"num_rbf": 20, "cutoff": 5.0})
cfg = {
    "data": {"train_size": 110000, "val_size": 10000,
             "smoke": bool(int(os.environ.get("OOD_SMOKE", "0"))),
             "smoke_train_size": 20000, "smoke_val_size": 2000,
             "batch_size": 128, "num_workers": 0},
    "model": _model_cfg,
}

torch.manual_seed(0)
bundle = load_qm9_bundle(cfg, seed=42)
mean = bundle.target_mean.to(DEV); std = bundle.target_std.to(DEV)
model = build_model(cfg).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)

print(f"device={DEV}  training {EPOCHS} epochs on {len(bundle.train_loader.dataset)} molecules ...")
for ep in range(EPOCHS):
    model.train()
    for batch in bundle.train_loader:
        batch = batch.to(DEV); opt.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(batch), normalize_y(batch.y, mean, std))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    print(f"  epoch {ep + 1}/{EPOCHS}")


@torch.no_grad()
def embed(loader, jitter: float = 0.0):
    model.eval(); E, ERR = [], []
    for batch in loader:
        batch = batch.to(DEV)
        if jitter > 0.0:
            batch.pos = batch.pos + jitter * torch.randn_like(batch.pos)
        out = model(batch, return_embeddings=True)
        E.append(out["graph_embedding"].cpu().numpy())
        ERR.append((out["pred"] - normalize_y(batch.y, mean, std)).abs().mean(-1).cpu().numpy())
    return np.concatenate(E), np.concatenate(ERR)


# fit-set embeddings (stable subsample of train), real val, and geometry-corrupted val (same molecules)
fit_items = [bundle.train_loader.dataset[i] for i in range(min(NFIT, len(bundle.train_loader.dataset)))]
Etr, _ = embed(DataLoader(fit_items, batch_size=128, shuffle=False))
Eval, err_val = embed(bundle.val_loader)
Eood, err_ood = embed(bundle.val_loader, jitter=JITTER)

# PCA on train embeddings; project all three
mu_e = Etr.mean(0); Etr_c = Etr - mu_e
P = np.linalg.svd(Etr_c, full_matrices=False)[2][:PCA_DIM].T
Xtr = Etr_c @ P; Xval = (Eval - mu_e) @ P; Xood = (Eood - mu_e) @ P

# droplet fit on TRAIN embeddings -> predictive t* + derived budget; score val + corrupted
w, mu, C, _, R2, nu = toi_fixed_point(Xtr, qlevel=QLEVEL, budget=BUDGET)
Cinv = np.linalg.inv(C)
s_val = _maha2(Xval, mu, Cinv); s_ood = _maha2(Xood, mu, Cinv)

# (1) calibration / risk-coverage on real val
order = np.argsort(s_val)
n = len(order)
risk_droplet = np.cumsum(err_val[order]) / np.arange(1, n + 1)
risk_oracle = np.cumsum(np.sort(err_val)) / np.arange(1, n + 1)
cov = np.arange(1, n + 1) / n
full_mae = float(err_val.mean())
mae80 = float(err_val[order[:int(0.8 * n)]].mean())
rho = float(spearmanr(s_val, err_val).correlation)

# (2) OOD detection: real vs geometry-corrupted (off-manifold)
def auc(neg, pos):
    return float((pos[:, None] > neg[None, :]).mean())
ood_auc = auc(s_val, s_ood)
flag_val = float((s_val > R2).mean()); flag_ood = float((s_ood > R2).mean())

print("=" * 72)
print(f"QM9 latent OOD certificate  (PCA{PCA_DIM}, budget={BUDGET}, R2={R2:.1f}, nu={nu:.1f})")
print("=" * 72)
print(f"(1) calibration: spearman(score, err)={rho:.3f} | MAE full={full_mae:.4f} -> keep-80%={mae80:.4f} "
      f"({100 * (full_mae - mae80) / max(full_mae, 1e-9):.1f}% lower)")
print(f"(2) OOD detection (real vs jittered geometry, sigma={JITTER}A): AUC={ood_auc:.3f} | "
      f"flagged real={flag_val:.2f} corrupted={flag_ood:.2f} | "
      f"err real={err_val.mean():.3f} corrupted={err_ood.mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
a = ax[0]
a.plot(cov, risk_droplet, color="C0", lw=2, label="droplet certificate")
a.plot(cov, risk_oracle, color="0.5", ls="--", lw=1.5, label="oracle (sort by true error)")
a.axhline(full_mae, color="C3", ls=":", lw=1.5, label="no abstention")
a.set_xlabel("coverage (retained, lowest OOD-score first)"); a.set_ylabel("MAE of retained set")
a.set_title(f"(a) selective prediction — spearman(score,err)={rho:.2f}"); a.legend(fontsize=8)
a = ax[1]
hi = float(np.percentile(np.r_[s_val, s_ood], 99))
bins = np.linspace(0, max(hi, R2 * 1.1), 40)
a.hist(s_val, bins=bins, alpha=.7, color="C0", label="real geometries")
a.hist(s_ood, bins=bins, alpha=.7, color="C3", label="corrupted geometries (OOD)")
a.axvline(R2, color="k", ls="--", label=f"derived budget R$^2$={R2:.0f}")
a.set_xlabel("droplet OOD score (Mahalanobis$^2$ from $t^\\star$)"); a.set_ylabel("count")
a.set_title(f"(b) OOD detection — AUC={ood_auc:.3f}"); a.legend(fontsize=8)
fig.suptitle("QM9 latent OOD certificate: droplet_core on the molecular embedding", y=1.02)
fig.tight_layout()
out = RESULTS / "ood_certificate.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
json.dump({"pca": PCA_DIM, "budget": BUDGET, "R2": float(R2), "nu": float(nu), "spearman": rho,
           "mae_full": full_mae, "mae_keep80": mae80, "ood_auc": ood_auc,
           "flagged_real": flag_val, "flagged_ood": flag_ood,
           "err_real": float(err_val.mean()), "err_ood": float(err_ood.mean())},
          open(HERE / "results" / "ood_certificate.json", "w"), indent=2)
print("saved", out)
