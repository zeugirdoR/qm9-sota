"""DropletCoreLoss — QM9 regression loss built on the RIGOROUS shared droplet (droplet_core.py).

Replaces the heuristic DropletLoss (residual-energy I_delta + tuned rho budget) with the invariant object:
each molecule's residual vector r in R^19 is weighted by the compact-support weight of the
consistency-corrected predictive t* (robust residual mean+covariance), with a DERIVED chi^2_19 budget.
Affine-invariant (whitens by the residual covariance, so correlated targets are handled correctly) and
tuning-free (no rho/temperature schedule — only a warmup + lambda blend so we don't amputate before the
residuals concentrate). Weights are computed on DETACHED residuals (stop-gradient).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .droplet_core import toi_fixed_point
from .droplet import _resolve_steps


class DropletCoreLoss(nn.Module):
    def __init__(self, base_loss: str = "smooth_l1", warmup_steps: int = 200, anneal_steps: int = 600,
                 qlevel: float = 0.999, nu: float = 0.5, lambda_end: float = 1.0):
        super().__init__()
        self.base_loss = base_loss
        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        self.qlevel = qlevel
        self.nu = nu
        self.lambda_end = lambda_end

    def _lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return 0.0
        return self.lambda_end * min(1.0, (step - self.warmup_steps) / max(1, self.anneal_steps))

    def _base_per_sample(self, residual: torch.Tensor) -> torch.Tensor:
        if self.base_loss == "l1":
            per = residual.abs()
        elif self.base_loss == "mse":
            per = residual.pow(2)
        else:
            per = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")
        return per.mean(dim=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, step: int):
        residual = pred - target
        base_per = self._base_per_sample(residual)
        lam = self._lambda(step)

        if lam <= 0.0 or pred.shape[0] <= pred.shape[-1] + 1:
            return base_per.mean(), {"lambda_droplet": torch.tensor(lam),
                                     "active_fraction": torch.tensor(1.0)}

        with torch.no_grad():
            w_np, _, _, _, R2 = toi_fixed_point(residual.detach().cpu().numpy(),
                                                qlevel=self.qlevel, nu=self.nu)
            w = torch.as_tensor(w_np, dtype=pred.dtype, device=pred.device)

        eff = (1.0 - lam) + lam * w
        loss = (base_per * eff).sum() / eff.sum().clamp_min(1e-8)
        stats = {
            "lambda_droplet": torch.tensor(lam),
            "active_fraction": (w > 0).float().mean().detach(),
            "droplet_weight_mean": w.mean().detach(),
            "R2_budget": torch.tensor(float(R2)),
        }
        return loss, stats


def build_droplet_core_loss(loss_cfg: dict, steps_per_epoch: int, device: torch.device) -> DropletCoreLoss:
    c = dict(loss_cfg["loss"])
    for k in ("name", "status", "reason"):
        c.pop(k, None)
    base_loss = c.pop("base_loss", "smooth_l1")
    warmup = _resolve_steps(c.get("warmup_steps", 200), steps_per_epoch)
    anneal = _resolve_steps(c.get("anneal_steps", 600), steps_per_epoch)
    return DropletCoreLoss(
        base_loss=base_loss,
        warmup_steps=warmup,
        anneal_steps=anneal,
        qlevel=float(c.get("qlevel", 0.999)),
        nu=float(c.get("nu", 0.5)),
        lambda_end=float(c.get("lambda_end", 1.0)),
    ).to(device)
