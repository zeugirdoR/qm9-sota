from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DropletAnnealConfig:
    warmup_steps: int = 2_000
    anneal_steps: int = 30_000

    delta_start: float = 0.05
    delta_end: float = 0.50

    nu_start: float = 0.10
    nu_end: float = 1.00

    alpha_start: float = 0.10
    alpha_end: float = 1.00

    temp_start: float = 1.0
    temp_end: float = 25.0

    lambda_start: float = 0.0
    lambda_end: float = 1.0

    soft_gate: bool = True
    min_weight: float = 0.0
    normalize_by_weight: bool = True

    learnable_offsets: bool = False
    learn_after_step: Optional[int] = None

    delta_offset_scale: float = 0.15
    log_param_offset_scale: float = 0.50
    offset_reg: float = 1e-3

    eps: float = 1e-8


def cosine_interp(start: float, end: float, frac: float) -> float:
    frac = max(0.0, min(1.0, frac))
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * frac))


class DropletLoss(nn.Module):
    """Bounded-information weighted loss for normalized QM9 regression targets."""

    def __init__(self, target_scale: Optional[torch.Tensor] = None, cfg: Optional[DropletAnnealConfig] = None, base_loss: str = "smooth_l1"):
        super().__init__()
        self.cfg = cfg or DropletAnnealConfig()
        self.base_loss = base_loss

        if target_scale is None:
            self.register_buffer("target_scale", torch.ones(1, 1))
        else:
            self.register_buffer("target_scale", target_scale.view(1, -1).float())

        if self.cfg.learnable_offsets:
            self.delta_offset = nn.Parameter(torch.zeros(()))
            self.log_nu_offset = nn.Parameter(torch.zeros(()))
            self.log_alpha_offset = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("delta_offset", None)
            self.register_parameter("log_nu_offset", None)
            self.register_parameter("log_alpha_offset", None)

    def _schedule_frac(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return 0.0
        return max(0.0, min(1.0, (step - self.cfg.warmup_steps) / max(1, self.cfg.anneal_steps)))

    def _learn_gain(self, step: int) -> float:
        if not self.cfg.learnable_offsets or self.cfg.learn_after_step is None or step < self.cfg.learn_after_step:
            return 0.0
        return max(0.0, min(1.0, (step - self.cfg.learn_after_step) / max(1, self.cfg.anneal_steps // 4)))

    def current_params(self, step: int, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        frac = self._schedule_frac(step)
        delta = torch.tensor(cosine_interp(self.cfg.delta_start, self.cfg.delta_end, frac), dtype=ref.dtype, device=ref.device)
        nu = torch.tensor(cosine_interp(self.cfg.nu_start, self.cfg.nu_end, frac), dtype=ref.dtype, device=ref.device)
        alpha = torch.tensor(cosine_interp(self.cfg.alpha_start, self.cfg.alpha_end, frac), dtype=ref.dtype, device=ref.device)
        temp = torch.tensor(cosine_interp(self.cfg.temp_start, self.cfg.temp_end, frac), dtype=ref.dtype, device=ref.device)
        lam = torch.tensor(cosine_interp(self.cfg.lambda_start, self.cfg.lambda_end, frac), dtype=ref.dtype, device=ref.device)

        gain = self._learn_gain(step)
        if self.cfg.learnable_offsets and gain > 0.0:
            gain_t = torch.tensor(gain, dtype=ref.dtype, device=ref.device)
            delta = delta + gain_t * self.cfg.delta_offset_scale * torch.tanh(self.delta_offset)
            nu = nu * torch.exp(gain_t * self.cfg.log_param_offset_scale * torch.tanh(self.log_nu_offset))
            alpha = alpha * torch.exp(gain_t * self.cfg.log_param_offset_scale * torch.tanh(self.log_alpha_offset))

        delta = torch.clamp(delta, min=0.01, max=0.99)
        nu = torch.clamp(nu, min=self.cfg.eps)
        alpha = torch.clamp(alpha, min=self.cfg.eps)
        return {"delta": delta, "nu": nu, "alpha": alpha, "rho": nu * alpha, "temperature": temp, "lambda": lam}

    def _base_per_sample_loss(self, scaled_residual: torch.Tensor) -> torch.Tensor:
        if self.base_loss == "l1":
            per_target = scaled_residual.abs()
        elif self.base_loss == "mse":
            per_target = scaled_residual.pow(2)
        elif self.base_loss == "smooth_l1":
            per_target = F.smooth_l1_loss(scaled_residual, torch.zeros_like(scaled_residual), reduction="none")
        else:
            raise ValueError(f"Unknown base_loss: {self.base_loss}")
        return per_target.mean(dim=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        params = self.current_params(step, pred)
        scale = self.target_scale.to(dtype=pred.dtype, device=pred.device)
        scaled_residual = (pred - target) / scale.clamp_min(self.cfg.eps)

        base_per_sample = self._base_per_sample_loss(scaled_residual)
        target_count = torch.full((pred.shape[0],), pred.shape[-1], dtype=pred.dtype, device=pred.device)
        residual_energy = scaled_residual.pow(2).sum(dim=-1) / target_count

        delta = params["delta"]
        rho = params["rho"]
        temperature = params["temperature"]
        lam = params["lambda"]

        i_delta = 0.5 * delta * (1.0 - delta) * residual_energy
        boundary_score = 1.0 - rho * i_delta

        if self.cfg.soft_gate:
            gate = torch.sigmoid(temperature * boundary_score)
        else:
            gate = (boundary_score >= 0.0).to(dtype=pred.dtype)

        overlap = torch.exp(-i_delta).clamp_min(self.cfg.eps)
        droplet_weight = gate * overlap

        if self.cfg.min_weight > 0.0:
            droplet_weight = self.cfg.min_weight + (1.0 - self.cfg.min_weight) * droplet_weight

        effective_weight = (1.0 - lam) + lam * droplet_weight
        weighted_loss = base_per_sample * effective_weight

        if self.cfg.normalize_by_weight:
            loss = weighted_loss.sum() / effective_weight.sum().clamp_min(self.cfg.eps)
        else:
            loss = weighted_loss.mean()

        reg = pred.new_tensor(0.0)
        if self.cfg.learnable_offsets:
            gain = self._learn_gain(step)
            if gain > 0.0:
                reg = self.cfg.offset_reg * gain * (
                    torch.tanh(self.delta_offset).pow(2)
                    + torch.tanh(self.log_nu_offset).pow(2)
                    + torch.tanh(self.log_alpha_offset).pow(2)
                )
                loss = loss + reg

        stats = {
            "loss": loss.detach(),
            "base_loss_mean": base_per_sample.mean().detach(),
            "i_delta_mean": i_delta.mean().detach(),
            "boundary_score_mean": boundary_score.mean().detach(),
            "active_fraction": (boundary_score >= 0.0).float().mean().detach(),
            "droplet_weight_mean": droplet_weight.mean().detach(),
            "effective_weight_mean": effective_weight.mean().detach(),
            "delta": params["delta"].detach(),
            "nu": params["nu"].detach(),
            "alpha": params["alpha"].detach(),
            "rho_nu_alpha": params["rho"].detach(),
            "temperature": params["temperature"].detach(),
            "lambda_droplet": params["lambda"].detach(),
            "offset_reg": reg.detach(),
        }
        return loss, stats


def _resolve_steps(value, steps_per_epoch: int):
    if isinstance(value, int):
        return value
    if value == "auto_half_epoch":
        return max(25, steps_per_epoch // 2)
    if value == "auto_one_epoch":
        return steps_per_epoch
    if value == "auto_three_epochs":
        return max(100, steps_per_epoch * 3)
    if value == "auto_eight_epochs":
        return max(100, steps_per_epoch * 8)
    if value == "auto_sixteen_epochs":
        return max(100, steps_per_epoch * 16)
    if value == "auto_three_epochs_start":
        return steps_per_epoch * 3
    raise ValueError(f"Unknown step schedule value: {value}")


def build_droplet_loss(loss_cfg: dict, steps_per_epoch: int, device: torch.device) -> DropletLoss:
    cfg_dict = dict(loss_cfg["loss"])
    base_loss = cfg_dict.pop("base_loss", "smooth_l1")
    cfg_dict.pop("name", None)
    cfg_dict.pop("status", None)
    cfg_dict.pop("reason", None)

    if "warmup_steps" in cfg_dict:
        cfg_dict["warmup_steps"] = _resolve_steps(cfg_dict["warmup_steps"], steps_per_epoch)
    if "anneal_steps" in cfg_dict:
        cfg_dict["anneal_steps"] = _resolve_steps(cfg_dict["anneal_steps"], steps_per_epoch)
    if "learn_after_step" in cfg_dict and cfg_dict["learn_after_step"] is not None:
        if cfg_dict["learn_after_step"] == "auto_three_epochs":
            cfg_dict["learn_after_step"] = steps_per_epoch * 3
        else:
            cfg_dict["learn_after_step"] = _resolve_steps(cfg_dict["learn_after_step"], steps_per_epoch)

    anneal_cfg = DropletAnnealConfig(**cfg_dict)
    return DropletLoss(target_scale=torch.ones(19), cfg=anneal_cfg, base_loss=base_loss).to(device)
