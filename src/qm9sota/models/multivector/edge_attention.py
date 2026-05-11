from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from qm9sota.geometry.multivector import Multivector


@dataclass
class MVEdgeAttentionConfig:
    scalar_dim: int
    edge_dim: int = 3
    hidden_dim: int = 128
    dropout: float = 0.0


class MultivectorEdgeAttention(nn.Module):
    """
    Edge-restricted scalar multivector attention.

    This is M1:
      - attention is restricted to molecular edges
      - attention score uses scalar q/k contraction + geometric edge bias
      - value update uses source scalar state and edge features
      - aggregation is softmax-normalized per destination node

    Later stages will add vector/bivector value transport.
    """

    def __init__(self, cfg: MVEdgeAttentionConfig):
        super().__init__()
        self.cfg = cfg

        h = cfg.hidden_dim
        d = cfg.scalar_dim

        self.q_s = nn.Linear(d, h)
        self.k_s = nn.Linear(d, h)
        self.v_s = nn.Linear(d, h)

        self.edge_bias = nn.Sequential(
            nn.Linear(cfg.edge_dim, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )

        self.edge_value = nn.Sequential(
            nn.Linear(cfg.edge_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
        )

        self.out_s = nn.Linear(h, d)
        self.norm_s = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        mv: Multivector,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> Multivector:
        if mv.s is None:
            raise ValueError("MultivectorEdgeAttention currently requires scalar channels mv.s")

        s = mv.s
        src, dst = edge_index

        q = self.q_s(s)
        k = self.k_s(s)
        v = self.v_s(s)

        # Edge score for message src -> dst.
        score = (q[dst] * k[src]).sum(dim=-1) / math.sqrt(q.shape[-1])
        score = score + self.edge_bias(edge_features).squeeze(-1)

        # Stable softmax per destination node.
        num_nodes = s.shape[0]

        max_per_dst = s.new_full((num_nodes,), -float("inf"))
        max_per_dst.scatter_reduce_(0, dst, score, reduce="amax", include_self=True)

        exp_score = torch.exp(score - max_per_dst[dst])
        denom = s.new_zeros(num_nodes)
        denom.index_add_(0, dst, exp_score)
        attn = exp_score / denom[dst].clamp_min(1e-12)
        attn = self.dropout(attn)

        edge_v = v[src] + self.edge_value(edge_features)
        msg = edge_v * attn.unsqueeze(-1)

        agg = s.new_zeros(num_nodes, edge_v.shape[-1])
        agg.index_add_(0, dst, msg)

        out_s = self.norm_s(s + self.out_s(agg))

        return Multivector(s=out_s, v=mv.v, b=mv.b, p=mv.p)
