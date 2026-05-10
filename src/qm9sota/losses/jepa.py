from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JEPAConfig:
    mask_ratio_low: float = 0.15
    mask_ratio_high: float = 0.40
    predictor_hidden: int = 128
    ema_tau_start: float = 0.99
    ema_tau_end: float = 0.999
    lambda_max: float = 0.5
    lambda_warmup_epochs: float = 1.0
    lambda_ramp_epochs: float = 2.0
    base_loss: str = "smooth_l1"
    eps: float = 1e-8


def _cosine_interp(start: float, end: float, frac: float) -> float:
    frac = max(0.0, min(1.0, frac))
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * frac))


def sample_atom_mask(
    batch_idx: torch.Tensor,
    *,
    ratio_low: float,
    ratio_high: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a per-atom boolean mask, choosing a per-graph ratio uniformly in [low, high]."""
    device = batch_idx.device
    num_nodes = batch_idx.numel()
    num_graphs = int(batch_idx.max().item()) + 1 if num_nodes > 0 else 0

    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if num_nodes == 0 or num_graphs == 0:
        return mask

    counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
    counts.index_add_(0, batch_idx, torch.ones_like(batch_idx))

    rand_node = torch.rand(num_nodes, device=device, generator=generator)
    ratios = torch.rand(num_graphs, device=device, generator=generator) * (ratio_high - ratio_low) + ratio_low
    ratios = ratios.to(dtype=rand_node.dtype)
    target_k = torch.maximum(torch.ones_like(counts), torch.round(counts.float() * ratios).long())
    target_k = torch.minimum(target_k, counts)

    offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    for g in range(num_graphs):
        start = int(offsets[g].item())
        end = int(offsets[g + 1].item())
        k = int(target_k[g].item())
        if k <= 0 or end <= start:
            continue
        slot = rand_node[start:end]
        kth = torch.topk(slot, k, largest=False).indices
        mask[start + kth] = True
    return mask


def apply_atom_mask(data, mask: torch.Tensor):
    """Return a shallow-cloned batch with ``data.x`` rows zeroed at masked positions.

    Coordinates and edge_index are untouched, so the message-passing graph is preserved
    and only atom identity is hidden.
    """
    masked = data.clone()
    x = masked.x.float().clone()
    x[mask] = 0.0
    masked.x = x
    return masked


class EMATargetEncoder:
    """Stop-gradient EMA copy of the online encoder used as the JEPA target.

    Holds a deepcopy of the online model in eval mode with ``requires_grad_(False)``.
    ``update`` does ``theta_t <- tau * theta_t + (1 - tau) * theta_o`` in-place.
    """

    def __init__(self, online_model: nn.Module):
        self.module = copy.deepcopy(online_model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()

    def buffers(self):
        return self.module.buffers()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        self.module.load_state_dict(state)

    @torch.no_grad()
    def update(self, online_model: nn.Module, tau: float) -> None:
        for p_t, p_o in zip(self.module.parameters(), online_model.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
        for b_t, b_o in zip(self.module.buffers(), online_model.buffers()):
            if b_t.dtype.is_floating_point:
                b_t.data.mul_(tau).add_(b_o.data, alpha=1.0 - tau)
            else:
                b_t.data.copy_(b_o.data)

    @torch.no_grad()
    def encode_nodes(self, data) -> torch.Tensor:
        return self.module.encode_nodes(data)


class JEPAPredictor(nn.Module):
    """Per-node MLP that predicts target embeddings from online context embeddings."""

    def __init__(self, hidden_dim: int, predictor_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, predictor_hidden),
            nn.SiLU(),
            nn.Linear(predictor_hidden, hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class JEPALoss(nn.Module):
    """JEPA auxiliary loss for the online MPNN encoder.

    Caller is expected to manage:
      - the masked forward pass through the online model (returns context node embeddings)
      - the unmasked forward pass through the EMA target encoder (returns target node embeddings)
      - the mask boolean tensor over nodes
      - the current epoch (float, fractional epochs allowed) for lambda/tau schedules

    This module owns the predictor parameters and the schedule logic. EMA updates and
    augmentation live outside (see ``EMATargetEncoder``, ``sample_atom_mask``).
    """

    def __init__(self, hidden_dim: int, cfg: Optional[JEPAConfig] = None):
        super().__init__()
        self.cfg = cfg or JEPAConfig()
        self.predictor = JEPAPredictor(hidden_dim=hidden_dim, predictor_hidden=self.cfg.predictor_hidden)

    def lambda_at(self, epoch: float) -> float:
        warm = self.cfg.lambda_warmup_epochs
        ramp = self.cfg.lambda_ramp_epochs
        if epoch < warm:
            return 0.0
        if ramp <= 0:
            return self.cfg.lambda_max
        frac = max(0.0, min(1.0, (epoch - warm) / ramp))
        return frac * self.cfg.lambda_max

    def tau_at(self, epoch: float, total_epochs: float) -> float:
        if total_epochs <= 0:
            return self.cfg.ema_tau_end
        frac = max(0.0, min(1.0, epoch / total_epochs))
        return _cosine_interp(self.cfg.ema_tau_start, self.cfg.ema_tau_end, frac)

    def _per_node_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cfg.base_loss == "smooth_l1":
            return F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
        if self.cfg.base_loss == "mse":
            return (pred - target).pow(2).mean(dim=-1)
        if self.cfg.base_loss == "cosine":
            pred_n = F.normalize(pred, dim=-1, eps=self.cfg.eps)
            tgt_n = F.normalize(target, dim=-1, eps=self.cfg.eps)
            return 1.0 - (pred_n * tgt_n).sum(dim=-1)
        raise ValueError(f"Unknown JEPA base_loss: {self.cfg.base_loss}")

    def forward(
        self,
        *,
        context_node_h: torch.Tensor,
        target_node_h: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        target_node_h = target_node_h.detach()
        if mask.sum() == 0:
            zero = context_node_h.new_zeros(())
            return zero, {
                "jepa_loss": zero.detach(),
                "jepa_masked_count": torch.tensor(0, device=context_node_h.device),
                "jepa_target_std": target_node_h.std(dim=0).mean().detach(),
            }

        pred = self.predictor(context_node_h[mask])
        target = target_node_h[mask]
        per_node = self._per_node_loss(pred, target)
        loss = per_node.mean()

        stats = {
            "jepa_loss": loss.detach(),
            "jepa_masked_count": mask.sum().detach(),
            "jepa_target_std": target_node_h.std(dim=0).mean().detach(),
            "jepa_pred_std": pred.detach().std(dim=0).mean(),
        }
        return loss, stats


def build_jepa_loss(jepa_cfg: dict, hidden_dim: int, device: torch.device) -> JEPALoss:
    cfg_dict = dict(jepa_cfg)
    cfg = JEPAConfig(**cfg_dict)
    return JEPALoss(hidden_dim=hidden_dim, cfg=cfg).to(device)
