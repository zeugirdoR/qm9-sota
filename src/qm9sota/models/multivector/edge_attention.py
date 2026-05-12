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

    # Scheduled motor residual path.
    use_motor: bool = False
    motor_hidden_dim: int = 128


class MultivectorEdgeAttention(nn.Module):
    """
    Edge-restricted scalar + vector multivector attention.

    Main validated M4 path:
      - scalar q/k edge attention
      - radial edge features
      - MPNN-style scalar message
      - equivariant vector message from learned gates times edge directions
      - invariant vector magnitudes fed back into scalar state

    Optional scheduled Cauchy-Binet path:
      score += lambda_cb * cb_bias(cb_features)

    Optional scheduled Motor path:
      vector_msg += lambda_motor * motor_vector_msg

    The motor residual is deliberately off at the beginning of training and
    enters only through lambda_motor supplied by the model schedule.
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

        if cfg.use_motor:
            mh = cfg.motor_hidden_dim

            # Motor-inspired edge branch. It predicts:
            #   - channel gates
            #   - a local axis vector
            #   - three mixing coefficients for direction/axis/cross components
            self.motor_core = nn.Sequential(
                nn.LayerNorm(2 * d + cfg.edge_dim),
                nn.Linear(2 * d + cfg.edge_dim, mh),
                nn.SiLU(),
                nn.Linear(mh, mh),
                nn.SiLU(),
            )
            self.motor_gate = nn.Linear(mh, cv)
            self.motor_axis = nn.Linear(mh, 3)
            self.motor_mix = nn.Linear(mh, 3)
        else:
            self.motor_core = None
            self.motor_gate = None
            self.motor_axis = None
            self.motor_mix = None

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
        lambda_motor: float | torch.Tensor = 0.0,
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

        # Base equivariant vector message pathway.
        rel = pos[src].float() - pos[dst].float()
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = rel / rel_norm

        gates = self.vector_gate(edge_context)
        vector_msg = gates.unsqueeze(-1) * direction.unsqueeze(1)

        # Optional scheduled motor residual.
        if self.motor_core is not None:
            if not torch.is_tensor(lambda_motor):
                lambda_motor_t = vector_msg.new_tensor(float(lambda_motor))
            else:
                lambda_motor_t = lambda_motor.to(dtype=vector_msg.dtype, device=vector_msg.device)

            if float(lambda_motor_t.detach().cpu()) != 0.0:
                motor_h = self.motor_core(edge_context)

                motor_gate = torch.sigmoid(self.motor_gate(motor_h))  # [E, Cv]

                axis = self.motor_axis(motor_h)
                axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

                # Remove the component of axis parallel to direction, giving a local perpendicular axis.
                axis_perp = axis - (axis * direction).sum(dim=-1, keepdim=True) * direction
                axis_perp = axis_perp / axis_perp.norm(dim=-1, keepdim=True).clamp_min(1e-8)

                cross = torch.cross(direction, axis_perp, dim=-1)
                cross = cross / cross.norm(dim=-1, keepdim=True).clamp_min(1e-8)

                mix = torch.tanh(self.motor_mix(motor_h))  # [E, 3]

                motor_dir = (
                    mix[:, 0:1] * direction
                    + mix[:, 1:2] * axis_perp
                    + mix[:, 2:3] * cross
                )
                motor_dir = motor_dir / motor_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)

                motor_msg = motor_gate.unsqueeze(-1) * motor_dir.unsqueeze(1)
                vector_msg = vector_msg + lambda_motor_t * motor_msg

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
