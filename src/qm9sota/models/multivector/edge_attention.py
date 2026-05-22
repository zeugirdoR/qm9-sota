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

    # Scheduled motor paths.
    use_motor: bool = False
    motor_hidden_dim: int = 128

    # "vector_residual": original M4 motor path, post-attention vector residual.
    # "attention_bias": recovered V20/AGAA-style score perturbation.
    # "both": use both paths.
    motor_mode: str = "vector_residual"

    # "mlp": score from MLP(edge_context).
    # "screw": score from q_screw/k_screw sparse edge attention.
    motor_score_mode: str = "mlp"
    motor_score_heads: int = 8
    motor_score_dim: int = 8


class MultivectorEdgeAttention(nn.Module):
    """
    Edge-restricted scalar + vector multivector attention.

    Main M4 path:
      - scalar q/k edge attention
      - radial edge features
      - scalar edge message
      - equivariant vector message from learned gates times edge directions
      - invariant vector magnitudes fed back into scalar state

    Optional Cauchy-Binet path:
      score += lambda_cb * cb_bias(cb_features)

    Optional original scheduled motor path:
      vector_msg += lambda_motor * motor_vector_msg

    Optional recovered V20/AGAA-style motor path:
      score += lambda_motor * motor_score

    The score-path motor changes attention itself.
    """

    def __init__(self, cfg: MVEdgeAttentionConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.motor_mode not in {"vector_residual", "attention_bias", "both"}:
            raise ValueError(f"Unknown motor_mode: {cfg.motor_mode}")
        if cfg.motor_score_mode not in {"mlp", "screw"}:
            raise ValueError(f"Unknown motor_score_mode: {cfg.motor_score_mode}")

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

            # Shared motor edge branch for original vector residual and MLP score motor.
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

            if cfg.motor_mode in {"attention_bias", "both"}:
                self.motor_score = nn.Linear(mh, 1)
                # MLP attention-bias path starts as exact no-op.
                nn.init.zeros_(self.motor_score.weight)
                nn.init.zeros_(self.motor_score.bias)

                hs = int(cfg.motor_score_heads)
                ds = int(cfg.motor_score_dim)
                self.motor_q_screw = nn.Linear(d, hs * ds, bias=False)
                self.motor_k_screw = nn.Linear(d, hs * ds, bias=False)
                self.motor_edge_screw = nn.Linear(cfg.edge_dim, hs, bias=False)
                self.motor_rbf_gate = nn.Linear(cfg.edge_dim, hs, bias=True)
                self.motor_rbf_bias = nn.Linear(cfg.edge_dim, hs, bias=False)
                self.motor_screw_scale = nn.Parameter(torch.tensor(0.5))
                metric_init = torch.ones(ds)
                if ds == 6:
                    metric_init = torch.tensor([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
                self.motor_screw_metric = nn.Parameter(metric_init)
                nn.init.orthogonal_(self.motor_q_screw.weight, gain=2.0)
                nn.init.orthogonal_(self.motor_k_screw.weight, gain=2.0)
                nn.init.constant_(self.motor_rbf_gate.bias, 1.0)
                self.motor_score_heads = hs
                self.motor_score_dim = ds
            else:
                self.motor_score = None
                self.motor_q_screw = None
                self.motor_k_screw = None
                self.motor_edge_screw = None
                self.motor_rbf_gate = None
                self.motor_rbf_bias = None
                self.motor_screw_scale = None
                self.motor_screw_metric = None
                self.motor_score_heads = 0
                self.motor_score_dim = 0
        else:
            self.motor_core = None
            self.motor_gate = None
            self.motor_axis = None
            self.motor_mix = None
            self.motor_score = None
            self.motor_q_screw = None
            self.motor_k_screw = None
            self.motor_edge_screw = None
            self.motor_rbf_gate = None
            self.motor_rbf_bias = None
            self.motor_screw_scale = None
            self.motor_screw_metric = None
            self.motor_score_heads = 0
            self.motor_score_dim = 0

        self.out_s = nn.Linear(h, d)
        self.norm_s = nn.LayerNorm(d)

        self.vector_norm = nn.LayerNorm(cv)
        self.dropout = nn.Dropout(cfg.dropout)
        self.last_motor_aux = None

    def _lambda_tensor(self, value, ref: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(value):
            return ref.new_tensor(float(value))
        return value.to(dtype=ref.dtype, device=ref.device)

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
        self.last_motor_aux = s.new_tensor(0.0)
        src, dst = edge_index
        num_nodes = s.shape[0]

        q = self.q_s(s)
        k = self.k_s(s)

        score = (q[dst] * k[src]).sum(dim=-1) / math.sqrt(q.shape[-1])
        score = score + self.edge_bias(edge_features).squeeze(-1)

        if self.cb_bias is not None and cb_features is not None:
            lambda_cb_t = self._lambda_tensor(lambda_cb, score)
            score = score + lambda_cb_t * self.cb_bias(cb_features).squeeze(-1)

        edge_context = torch.cat([s[src], s[dst], edge_features], dim=-1)

        # Recovered V20/AGAA-style motor attention-bias path:
        #   score += lambda_motor * motor_score
        if (
            self.motor_core is not None
            and self.motor_score is not None
            and self.cfg.motor_mode in {"attention_bias", "both"}
        ):
            lambda_motor_t_score = self._lambda_tensor(lambda_motor, score)
            if float(lambda_motor_t_score.detach().cpu()) != 0.0:
                if self.cfg.motor_score_mode == "screw":
                    hs = self.motor_score_heads
                    ds = self.motor_score_dim
                    q_m = self.motor_q_screw(s[dst]).view(-1, hs, ds)
                    k_m = self.motor_k_screw(s[src]).view(-1, hs, ds)

                    metric = self.motor_screw_metric.to(dtype=q_m.dtype, device=q_m.device)
                    q_m = q_m * metric.view(1, 1, ds)
                    k_m = k_m * metric.view(1, 1, ds)

                    # Old V20/AGAA-style motor score: asinh(<q,k>_metric) * scale.
                    G = (q_m * k_m).sum(dim=-1)
                    motor_score_h = torch.asinh(G) * self.motor_screw_scale

                    # Old locality modulation: score_h = motor_score_h * gate + bias.
                    gate_h = self.motor_rbf_gate(edge_features)
                    bias_h = self.motor_rbf_bias(edge_features)
                    score_h = motor_score_h * gate_h + bias_h

                    motor_score = score_h.mean(dim=-1)

                    # Exact recovered coupling statistic, but kept differentiable for v9.
                    # Old notebook logged: mean(abs(motor_score * gate)).detach()
                    # Training comment says: maximize motor coupling / avoid kinematic lock.
                    self.last_motor_aux = -torch.mean(torch.abs(motor_score_h * gate_h))
                else:
                    motor_h_score = self.motor_core(edge_context)
                    motor_score = torch.tanh(self.motor_score(motor_h_score).squeeze(-1))
                    self.last_motor_aux = -torch.mean(torch.abs(motor_score))

                score = score + lambda_motor_t_score * motor_score

        max_per_dst = s.new_full((num_nodes,), -float("inf"))
        max_per_dst.scatter_reduce_(0, dst, score, reduce="amax", include_self=True)

        exp_score = torch.exp(score - max_per_dst[dst])
        denom = s.new_zeros(num_nodes)
        denom.index_add_(0, dst, exp_score)

        attn = exp_score / denom[dst].clamp_min(1e-12)
        attn = self.dropout(attn)

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

        # Original scheduled motor vector-residual path.
        if self.motor_core is not None and self.cfg.motor_mode in {"vector_residual", "both"}:
            lambda_motor_t_vec = self._lambda_tensor(lambda_motor, vector_msg)

            if float(lambda_motor_t_vec.detach().cpu()) != 0.0:
                motor_h = self.motor_core(edge_context)

                motor_gate = torch.sigmoid(self.motor_gate(motor_h))  # [E, Cv]

                axis = self.motor_axis(motor_h)
                axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

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
                vector_msg = vector_msg + lambda_motor_t_vec * motor_msg

        vector_msg = vector_msg * attn.view(-1, 1, 1)

        cv = gates.shape[-1]
        vector_agg = s.new_zeros(num_nodes, cv, 3)
        vector_agg.index_add_(0, dst, vector_msg)

        if mv.v is not None and mv.v.shape[1] == cv:
            vector_agg = vector_agg + mv.v

        vector_mag = vector_agg.pow(2).sum(dim=-1).clamp_min(1e-12).sqrt()
        vector_mag = self.vector_norm(vector_mag)

        scalar_plus_vector = scalar_agg + self.vector_to_scalar(vector_mag)
        out_s = self.norm_s(s + self.out_s(scalar_plus_vector))

        return Multivector(s=out_s, v=vector_agg, b=mv.b, p=mv.p)
