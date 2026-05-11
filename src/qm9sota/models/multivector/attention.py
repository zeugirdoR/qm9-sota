from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from qm9sota.geometry.multivector import Multivector, pairwise_scalar_score


@dataclass
class MVAttentionConfig:
    scalar_dim: int
    vector_channels: int = 4
    bivector_channels: int = 4
    pseudoscalar_dim: int = 4
    hidden_dim: int = 128
    dropout: float = 0.0


class MultivectorAttention(nn.Module):
    """
    Prototype dense multivector attention.

    Attention scores are scalar invariants built from grade-wise contractions.
    Values update grade channels separately.

    This is intentionally dense and simple for small QM9 molecules.
    """

    def __init__(self, cfg: MVAttentionConfig):
        super().__init__()
        self.cfg = cfg

        h = cfg.hidden_dim

        self.q_s = nn.Linear(cfg.scalar_dim, h)
        self.k_s = nn.Linear(cfg.scalar_dim, h)
        self.v_s = nn.Linear(cfg.scalar_dim, h)
        self.out_s = nn.Linear(h, cfg.scalar_dim)

        self.dropout = nn.Dropout(cfg.dropout)
        self.norm_s = nn.LayerNorm(cfg.scalar_dim)

    def forward(
        self,
        mv: Multivector,
        attn_bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> Multivector:
        if mv.s is None:
            raise ValueError("Current prototype requires scalar channels mv.s")

        q = Multivector(s=self.q_s(mv.s))
        k = Multivector(s=self.k_s(mv.s))

        scores = pairwise_scalar_score(q, k) / math.sqrt(q.s.shape[-1])

        if attn_bias is not None:
            scores = scores + attn_bias

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        v_s = self.v_s(mv.s)
        upd_s = attn @ v_s
        out_s = self.norm_s(mv.s + self.out_s(upd_s))

        # For M0/M1 we update only scalar channels. Vector/bivector updates come next.
        return Multivector(s=out_s, v=mv.v, b=mv.b, p=mv.p)
