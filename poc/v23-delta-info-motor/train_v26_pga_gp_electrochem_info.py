#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TARGET_INDEX = 12
TARGET_NAME = "U0_atom"
MEV = 1000.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_splits(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return perm[:110000], perm[110000:120000], perm[120000:]


def normalize_y(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (y.float() - mean) / std.clamp_min(1e-8)


def scheduled_lambda(epoch_float: float, *, warmup: float, ramp: float, end: float) -> float:
    if epoch_float < warmup:
        return 0.0
    if ramp <= 0:
        return float(end)
    frac = max(0.0, min(1.0, (epoch_float - warmup) / ramp))
    return float(end) * 0.5 * (1.0 - math.cos(math.pi * frac))


def delta_exp(x, delta):
    if abs(delta) < 1e-5:
        return torch.exp(x)
    return torch.relu(1.0 + delta * x) ** (1.0 / delta)

def delta_softmax(logits, delta, dim=-1):
    shifted_logits = logits - logits.max(dim=dim, keepdim=True)[0]
    weights = delta_exp(shifted_logits, delta)
    return weights / (weights.sum(dim=dim, keepdim=True) + 1e-8)


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int = 20, cutoff: float = 8.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = 10.0 / float(cutoff)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers).pow(2))


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _dq_mul(ar: torch.Tensor, ad: torch.Tensor, br: torch.Tensor, bd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rr = _quat_mul(ar, br)
    dd = _quat_mul(ar, bd) + _quat_mul(ad, br)
    return rr, dd


def _dq_reverse(r: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _quat_conj(r), _quat_conj(d)


def _pga_points_cyclic(pos: torch.Tensor) -> torch.Tensor:
    one = torch.ones_like(pos[..., :1])
    return torch.cat([one, pos[..., 0:1], pos[..., 1:2], pos[..., 2:3]], dim=-1)


class ElectroChemicallyConditionedMetric(nn.Module):
    def __init__(self, hidden_dim=192, n_heads=16, rank=8, delta=0.5, metric_scale=2.0, max_z=100):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads
        self.rank = rank
        self.delta = delta
        self.metric_scale = metric_scale

        self.chem_emb = nn.Embedding(max_z, 16)
        
        self.lambda_proj = nn.Linear(33, 1)
        self.u_gate = nn.Linear(33, 8 * rank)

        self.log_temp = nn.Parameter(torch.zeros(1))
        self.log_vol_weight = nn.Parameter(torch.tensor([-2.0]))
        self.log_lmbda_base = nn.Parameter(torch.tensor([math.log(0.005)]))
        
        self.grade_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]))
        self.U_proj = nn.Linear(self.d_head, 8 * rank)

    def forward(self, x_head, rij_r, rij_d, pair_mask, z_dense, coulomb_feat):
        B, N, H, D = x_head.shape
        
        M_ij = torch.cat([rij_r, rij_d], dim=-1)
        M_ij = M_ij * self.grade_weights

        chem_i = self.chem_emb(z_dense)
        chem_pair = torch.cat([
            chem_i.unsqueeze(2).expand(-1, -1, N, -1),
            chem_i.unsqueeze(1).expand(-1, N, -1, -1)
        ], dim=-1)

        electrochem_prior = torch.cat([chem_pair, coulomb_feat.expand(-1, -1, -1, 1)], dim=-1)

        log_lmbda_local = self.lambda_proj(electrochem_prior).view(B, 1, N, N, 1, 1)
        lmbda = torch.exp(self.log_lmbda_base + log_lmbda_local)

        U_raw = self.U_proj(x_head).view(B, N, H, 8, self.rank).permute(0, 2, 1, 3, 4)
        gate = torch.sigmoid(self.u_gate(electrochem_prior)).view(B, 1, N, N, 8, self.rank)
        
        U_ij = U_raw.unsqueeze(3) * gate
        U = torch.tanh(U_ij) * self.metric_scale
        U_T = U.transpose(4, 5)

        projected_motor = torch.einsum('bhijrd, bhijd -> bhijr', U_T, M_ij)
        
        D_ij = torch.sum(projected_motor**2, dim=-1) + lmbda.squeeze(-1).squeeze(-1) * torch.sum(M_ij**2, dim=-1)

        I_R = torch.eye(self.rank, device=x_head.device).view(1, 1, 1, 1, self.rank, self.rank)
        U_T_U = torch.einsum('bhijrd, bhijdc -> bhijrc', U_T, U)
        
        logdet = torch.linalg.slogdet(lmbda * I_R + U_T_U)[1]
        log_vol_ij = 0.5 * logdet
        
        temp = torch.exp(self.log_temp) + 1e-4
        vol_weight = torch.exp(self.log_vol_weight)
        
        attention_logits = (-0.5 * D_ij / temp) - (vol_weight * log_vol_ij)
        return attention_logits


class MotorAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_rbf: int = 20):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.dim_motor = 8

        self.q_screw = nn.Linear(d_model, n_heads * self.dim_motor, bias=False)
        self.k_screw = nn.Linear(d_model, n_heads * self.dim_motor, bias=False)

        self.rbf_gate = nn.Linear(n_rbf, n_heads, bias=True)
        self.rbf_bias = nn.Linear(n_rbf, n_heads, bias=False)

        self.info_metric = ElectroChemicallyConditionedMetric(
            hidden_dim=d_model, n_heads=n_heads, rank=8, delta=0.5
        )

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rbf_feat: torch.Tensor,
        coulomb_feat: torch.Tensor,
        pos_dense: torch.Tensor | None,
        mask: torch.Tensor,
        z_dense: torch.Tensor,
        motor_strength: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        B, N, D = x.shape
        x_f32 = x.float()

        q_raw = self.q_screw(x_f32).view(B, N, self.n_heads, self.dim_motor).permute(0, 2, 1, 3)
        k_raw = self.k_screw(x_f32).view(B, N, self.n_heads, self.dim_motor).permute(0, 2, 1, 3)

        q_r = q_raw[..., :4]
        q_d = q_raw[..., 4:]
        k_r = k_raw[..., :4]
        k_d = k_raw[..., 4:]

        q_r = q_r / q_r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        k_r = k_r / k_r.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        q_d = q_d - q_r * (q_r * q_d).sum(dim=-1, keepdim=True)
        k_d = k_d - k_r * (k_r * k_d).sum(dim=-1, keepdim=True)

        qi_r = q_r.unsqueeze(3)
        qi_d = q_d.unsqueeze(3)
        kj_r = k_r.unsqueeze(2)
        kj_d = k_d.unsqueeze(2)

        if pos_dense is None:
            t_r = torch.zeros_like(qi_r + kj_r)
            t_r[..., 0] = 1.0
            t_d = torch.zeros_like(t_r)
        else:
            _ = _pga_points_cyclic(pos_dense)
            rel = pos_dense[:, :, None, :].float() - pos_dense[:, None, :, :].float()
            rel = rel.unsqueeze(1)
            t_r = torch.zeros(rel.shape[:-1] + (4,), device=rel.device, dtype=x_f32.dtype)
            t_r[..., 0] = 1.0
            t_d = torch.cat([torch.zeros_like(rel[..., :1]), 0.5 * rel.to(dtype=x_f32.dtype)], dim=-1)

        qt_r, qt_d = _dq_mul(qi_r, qi_d, t_r, t_d)
        kr_rev, kd_rev = _dq_reverse(kj_r, kj_d)
        rij_r, rij_d = _dq_mul(qt_r, qt_d, kr_rev, kd_rev)

        x_head = x_f32.view(B, N, self.n_heads, self.d_head)
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        
        motor_score = self.info_metric(x_head, rij_r, rij_d, pair_mask, z_dense, coulomb_feat)

        geo_bias = self.rbf_bias(rbf_feat).permute(0, 3, 1, 2)
        geo_gate = self.rbf_gate(rbf_feat).permute(0, 3, 1, 2)

        scores = geo_bias + float(motor_strength) * geo_gate * motor_score
        scores = scores.masked_fill(~pair_mask, -1e9)

        attn = delta_softmax(scores, delta=self.info_metric.delta, dim=-1)

        v = self.v_proj(x_f32).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        out = self.out_proj(out)

        pair_mask_f = pair_mask.float()
        motor_sig = (motor_score.abs() * pair_mask_f).sum() / pair_mask_f.sum().clamp_min(1.0)
        return out, motor_sig


class V20BlockMotor(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_rbf: int = 20, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MotorAttention(d_model=d_model, n_heads=n_heads, n_rbf=n_rbf)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rbf_feat: torch.Tensor,
        coulomb_feat: torch.Tensor,
        pos_dense: torch.Tensor | None,
        mask: torch.Tensor,
        z_dense: torch.Tensor,
        motor_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out, sig = self.attn(h, rbf_feat, coulomb_feat, pos_dense=pos_dense, mask=mask, z_dense=z_dense, motor_strength=motor_strength)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, sig


class V20AGAAMotor(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int = 7,
        d_model: int = 192,
        n_heads: int = 16,
        max_z: int = 100,
        n_rbf: int = 20,
        dropout: float = 0.0,
        out_dim: int = 19,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)

        self.emb_z = nn.Embedding(max_z, d_model)
        self.emb_geo = nn.Linear(4, d_model)
        self.emb_fuse = nn.Linear(2 * d_model, d_model)

        self.rbf = GaussianRBF(n_rbf=n_rbf, cutoff=8.0)

        self.layers = nn.ModuleList([
            V20BlockMotor(d_model=d_model, n_heads=n_heads, n_rbf=n_rbf, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, data, motor_strength: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        z_dense, mask = to_dense_batch(data.z.long(), data.batch)
        pos_dense, _ = to_dense_batch(data.pos.float(), data.batch)

        mask_f = mask.float().unsqueeze(-1)
        center = (pos_dense * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        pos_dense = pos_dense - center

        z_dense = z_dense.clamp(min=0, max=99)
        h_z = self.emb_z(z_dense)

        dist = torch.cdist(pos_dense, pos_dense).clamp_min(1e-8)
        rbf_feat = self.rbf(dist)

        B, N = z_dense.shape
        pair_mask = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = pair_mask & (~eye)

        z_f = z_dense.float() * mask.float()
        zz = z_f[:, :, None] * z_f[:, None, :]
        coulomb = (zz / dist.clamp_min(0.40)) / 20.0
        coulomb = coulomb.masked_fill(~valid_pair, 0.0)
        coulomb_feat = coulomb.unsqueeze(-1)

        n_atoms = mask.sum(dim=1).view(-1, 1, 1).clamp_min(1)
        dist_masked = dist.masked_fill(~mask[:, None, :], 0.0)
        mean_dist = dist_masked.sum(dim=-1, keepdim=True) / n_atoms

        radius = pos_dense.norm(dim=-1, keepdim=True)
        geo = torch.cat(
            [
                radius,
                radius.pow(2),
                mean_dist,
                mask.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        h_geo = self.emb_geo(geo)
        h = self.emb_fuse(torch.cat([h_z, h_geo], dim=-1))

        motor_sig_total = h.new_tensor(0.0)
        for layer in self.layers:
            h, sig = layer(h, rbf_feat, coulomb_feat, pos_dense, mask, z_dense, motor_strength=float(motor_strength))
            motor_sig_total = motor_sig_total + sig

        h = self.norm_final(h)
        graph_h = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        out = self.head(graph_h)

        return out, motor_sig_total / max(len(self.layers), 1)
