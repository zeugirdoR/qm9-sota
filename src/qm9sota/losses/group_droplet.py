from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_GROUPS = {
    "electronic": [0, 1, 2, 3, 4],
    "geometry_thermal": [5, 6, 11, 16, 17, 18],
    "energy": [7, 8, 9, 10, 12, 13, 14, 15],
}


@dataclass
class GroupDropletConfig:
    warmup_steps: int = 860
    anneal_steps: int = 860 * 8

    delta_start: float = 0.05
    delta_end: float = 0.35

    nu_start: float = 0.05
    nu_end: float = 0.50

    alpha_start: float = 0.05
    alpha_end: float = 0.50

    temp_start: float = 1.0
    temp_end: float = 8.0

    lambda_start: float = 0.0
    lambda_end: float = 0.35

    soft_gate: bool = True
    min_weight: float = 0.20
    normalize_by_weight: bool = True

    base_loss: str = "smooth_l1"
    eps: float = 1e-8


def _cosine_interp(start: float, end: float, frac: float) -> float:
    frac = max(0.0, min(1.0, frac))
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * frac))


def _resolve_steps(value, *, steps_per_epoch: int, default: int) -> int:
    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if value == "auto_one_epoch":
        return max(1, steps_per_epoch)

    if value == "auto_four_epochs":
        return max(1, steps_per_epoch * 4)

    if value == "auto_eight_epochs":
        return max(1, steps_per_epoch * 8)

    if value == "auto_sixteen_epochs":
        return max(1, steps_per_epoch * 16)

    return default


class GroupDropletLoss(nn.Module):
    """
    Group-wise Droplet loss for QM9 multitarget regression.

    Instead of one scalar molecule-level weight over all 19 targets, this computes
    one Droplet weight per target family:
      electronic
      geometry_thermal
      energy

    The model still predicts all 19 normalized targets.
    """

    def __init__(self, cfg: GroupDropletConfig):
        super().__init__()
        self.cfg = cfg

        for name, idx in TARGET_GROUPS.items():
            self.register_buffer(
                f"{name}_idx",
                torch.tensor(idx, dtype=torch.long),
                persistent=False,
            )

    def _schedule_frac(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return 0.0
        denom = max(1, self.cfg.anneal_steps)
        return max(0.0, min(1.0, (step - self.cfg.warmup_steps) / denom))

    def current_params(self, step: int, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        frac = self._schedule_frac(step)

        delta = _cosine_interp(self.cfg.delta_start, self.cfg.delta_end, frac)
        nu = _cosine_interp(self.cfg.nu_start, self.cfg.nu_end, frac)
        alpha = _cosine_interp(self.cfg.alpha_start, self.cfg.alpha_end, frac)
        temp = _cosine_interp(self.cfg.temp_start, self.cfg.temp_end, frac)
        lam = _cosine_interp(self.cfg.lambda_start, self.cfg.lambda_end, frac)

        delta = torch.tensor(delta, dtype=ref.dtype, device=ref.device).clamp(0.01, 0.99)
        nu = torch.tensor(nu, dtype=ref.dtype, device=ref.device).clamp_min(self.cfg.eps)
        alpha = torch.tensor(alpha, dtype=ref.dtype, device=ref.device).clamp_min(self.cfg.eps)
        temp = torch.tensor(temp, dtype=ref.dtype, device=ref.device)
        lam = torch.tensor(lam, dtype=ref.dtype, device=ref.device)

        return {
            "delta": delta,
            "nu": nu,
            "alpha": alpha,
            "rho": nu * alpha,
            "temperature": temp,
            "lambda": lam,
        }

    def _base_per_target_loss(self, residual: torch.Tensor) -> torch.Tensor:
        if self.cfg.base_loss == "l1":
            return residual.abs()

        if self.cfg.base_loss == "mse":
            return residual.pow(2)

        if self.cfg.base_loss == "smooth_l1":
            return F.smooth_l1_loss(
                residual,
                torch.zeros_like(residual),
                reduction="none",
            )

        raise ValueError(f"Unknown base_loss: {self.cfg.base_loss}")

    def _group_loss_and_stats(
        self,
        *,
        residual: torch.Tensor,
        per_target_loss: torch.Tensor,
        idx: torch.Tensor,
        params: Dict[str, torch.Tensor],
        prefix: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        r = residual[:, idx]
        base = per_target_loss[:, idx]

        base_per_sample = base.mean(dim=-1)

        residual_energy = r.pow(2).mean(dim=-1)

        delta = params["delta"]
        rho = params["rho"]
        temperature = params["temperature"]
        lam = params["lambda"]

        i_delta = 0.5 * delta * (1.0 - delta) * residual_energy
        boundary_score = 1.0 - rho * i_delta

        if self.cfg.soft_gate:
            gate = torch.sigmoid(temperature * boundary_score)
        else:
            gate = (boundary_score >= 0.0).to(dtype=residual.dtype)

        overlap = torch.exp(-i_delta).clamp_min(self.cfg.eps)
        droplet_weight = gate * overlap

        if self.cfg.min_weight > 0.0:
            droplet_weight = self.cfg.min_weight + (
                1.0 - self.cfg.min_weight
            ) * droplet_weight

        effective_weight = (1.0 - lam) + lam * droplet_weight
        weighted = base_per_sample * effective_weight

        if self.cfg.normalize_by_weight:
            group_loss = weighted.sum() / effective_weight.sum().clamp_min(self.cfg.eps)
        else:
            group_loss = weighted.mean()

        stats = {
            f"{prefix}_base_loss": base_per_sample.mean().detach(),
            f"{prefix}_i_delta": i_delta.mean().detach(),
            f"{prefix}_active_fraction": (boundary_score >= 0.0).float().mean().detach(),
            f"{prefix}_droplet_weight": droplet_weight.mean().detach(),
            f"{prefix}_effective_weight": effective_weight.mean().detach(),
        }

        return group_loss, base_per_sample.detach(), stats

    def forward(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        params = self.current_params(step, pred)

        residual = pred - target
        per_target_loss = self._base_per_target_loss(residual)

        group_losses = []
        stats: Dict[str, torch.Tensor] = {}

        for name in ["electronic", "geometry_thermal", "energy"]:
            idx = getattr(self, f"{name}_idx")
            g_loss, _, g_stats = self._group_loss_and_stats(
                residual=residual,
                per_target_loss=per_target_loss,
                idx=idx,
                params=params,
                prefix=name,
            )
            group_losses.append(g_loss)
            stats.update(g_stats)

        loss = torch.stack(group_losses).mean()

        stats.update(
            {
                "loss": loss.detach(),
                "group_droplet_loss": loss.detach(),
                "delta": params["delta"].detach(),
                "nu": params["nu"].detach(),
                "alpha": params["alpha"].detach(),
                "rho_nu_alpha": params["rho"].detach(),
                "temperature": params["temperature"].detach(),
                "lambda_droplet": params["lambda"].detach(),
            }
        )

        return loss, stats


def build_group_droplet_loss(
    loss_cfg: dict,
    *,
    steps_per_epoch: int,
    device: torch.device,
) -> GroupDropletLoss:
    block = dict(loss_cfg.get("loss", loss_cfg))

    cfg = GroupDropletConfig(
        warmup_steps=_resolve_steps(
            block.get("warmup_steps", "auto_one_epoch"),
            steps_per_epoch=steps_per_epoch,
            default=steps_per_epoch,
        ),
        anneal_steps=_resolve_steps(
            block.get("anneal_steps", "auto_eight_epochs"),
            steps_per_epoch=steps_per_epoch,
            default=steps_per_epoch * 8,
        ),
        delta_start=float(block.get("delta_start", 0.05)),
        delta_end=float(block.get("delta_end", 0.35)),
        nu_start=float(block.get("nu_start", 0.05)),
        nu_end=float(block.get("nu_end", 0.50)),
        alpha_start=float(block.get("alpha_start", 0.05)),
        alpha_end=float(block.get("alpha_end", 0.50)),
        temp_start=float(block.get("temp_start", 1.0)),
        temp_end=float(block.get("temp_end", 8.0)),
        lambda_start=float(block.get("lambda_start", 0.0)),
        lambda_end=float(block.get("lambda_end", 0.20)),
        soft_gate=bool(block.get("soft_gate", True)),
        min_weight=float(block.get("min_weight", 0.20)),
        normalize_by_weight=bool(block.get("normalize_by_weight", True)),
        base_loss=str(block.get("base_loss", "smooth_l1")),
    )

    return GroupDropletLoss(cfg).to(device)
