from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from qm9sota.geometry.multivector import Multivector


@dataclass
class MVEdgeAttentionConfig:
    scalar_dim: int
    edge_dim: int = 35
    hidden_dim: int = 128
    vector_channels: int = 8
    dropout: float = 0.0
    cb_dim: int = 32
    use_cb_bias: bool = False


class MultivectorEdgeAttention(nn.Module):
    """
    Edge-restricted scalar + vector multivector attention.

    Main path is the validated M4 pathway:
      - scalar q/k edge attention
      - radial edge features
      - MPNN-style scalar message
      - equivariant vector message from learned gates times edge directions
      - invariant vector magnitudes fed back into scalar state

    Optional M6b path:
      - Cauchy-Binet / local Gram-volume features enter only as a scheduled
        additive attention bias:
            score += lambda_cb * cb_bias(cb_features)

    This keeps the strong M4 path intact and introduces CB information gently.
    """

    def __init__(self, cfg: MVEdgeAttentionConfig):
        super().__init__()
        self.cfg = cfg

        h = cfg.hidden_dim
        d = cfg.scalar_dim
        cv = cfg.vector_channels

        self.q_s = nn.Linear(d, h)
        self.k_s = nn.Linear(d, h)

        self.edge_bias = nn.Sequential(
            nn.Linear(cfg.edge_dim, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )

        self.edge_msg = nn.Sequential(
            nn.Linear(2 * d + cfg.edge_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )

        self.vector_gate = nn.Sequential(
            nn.Linear(2 * d + cfg.edge_dim, h),
            nn.SiLU(),
            nn.Linear(h, cv),
        )

        self.vector_to_scalar = nn.Sequential(
            nn.Linear(cv, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )

        if cfg.use_cb_bias:
            self.cb_bias = nn.Sequential(
                nn.LayerNorm(cfg.cb_dim),
                nn.Linear(cfg.cb_dim, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
        else:
            self.cb_bias = None

        self.out_s = nn.Linear(h, d)
        self.norm_s = nn.LayerNorm(d)

        self.vector_norm = nn.LayerNorm(cv)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        mv: Multivector,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        pos: torch.Tensor,
        cb_features: torch.Tensor | None = None,
        lambda_cb: float | torch.Tensor = 0.0,
    ) -> Multivector:
        if mv.s is None:
            raise ValueError("MultivectorEdgeAttention currently requires scalar channels mv.s")

        s = mv.s
        src, dst = edge_index
        num_nodes = s.shape[0]

        q = self.q_s(s)
        k = self.k_s(s)

        score = (q[dst] * k[src]).sum(dim=-1) / math.sqrt(q.shape[-1])
        score = score + self.edge_bias(edge_features).squeeze(-1)

        if self.cb_bias is not None and cb_features is not None:
            if not torch.is_tensor(lambda_cb):
                lambda_cb_t = score.new_tensor(float(lambda_cb))
            else:
                lambda_cb_t = lambda_cb.to(dtype=score.dtype, device=score.device)

            score = score + lambda_cb_t * self.cb_bias(cb_features).squeeze(-1)

        max_per_dst = s.new_full((num_nodes,), -float("inf"))
        max_per_dst.scatter_reduce_(0, dst, score, reduce="amax", include_self=True)

        exp_score = torch.exp(score - max_per_dst[dst])
        denom = s.new_zeros(num_nodes)
        denom.index_add_(0, dst, exp_score)

        attn = exp_score / denom[dst].clamp_min(1e-12)
        attn = self.dropout(attn)

        edge_context = torch.cat([s[src], s[dst], edge_features], dim=-1)

        # Scalar message pathway.
        edge_s = self.edge_msg(edge_context)
        scalar_msg = edge_s * attn.unsqueeze(-1)

        scalar_agg = s.new_zeros(num_nodes, edge_s.shape[-1])
        scalar_agg.index_add_(0, dst, scalar_msg)

        # Equivariant vector message pathway.
        rel = pos[src].float() - pos[dst].float()
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = rel / rel_norm

        gates = self.vector_gate(edge_context)
        vector_msg = gates.unsqueeze(-1) * direction.unsqueeze(1)
        vector_msg = vector_msg * attn.view(-1, 1, 1)

        cv = gates.shape[-1]
        vector_agg = s.new_zeros(num_nodes, cv, 3)
        vector_agg.index_add_(0, dst, vector_msg)

        if mv.v is not None and mv.v.shape[1] == cv:
            vector_agg = vector_agg + mv.v

        # Invariant vector magnitude feedback.
        vector_mag = vector_agg.pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
        vector_mag = self.vector_norm(vector_mag)

        scalar_plus_vector = scalar_agg + self.vector_to_scalar(vector_mag)
        out_s = self.norm_s(s + self.out_s(scalar_plus_vector))

        return Multivector(s=out_s, v=vector_agg, b=mv.b, p=mv.p)
