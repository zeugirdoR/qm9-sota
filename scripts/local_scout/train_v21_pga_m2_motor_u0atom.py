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


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int = 20, cutoff: float = 8.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = 10.0 / float(cutoff)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers).pow(2))


class MotorAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_rbf: int = 20):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.dim_motor = 8

        self.q_screw = nn.Linear(d_model, n_heads * self.dim_motor, bias=False)
        self.k_screw = nn.Linear(d_model, n_heads * self.dim_motor, bias=False)

        self.rbf_gate = nn.Linear(n_rbf, n_heads, bias=True)
        self.rbf_bias = nn.Linear(n_rbf, n_heads, bias=False)
        self.coulomb_proj = nn.Linear(1, n_heads, bias=False)

        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rbf_feat: torch.Tensor,
        coulomb_feat: torch.Tensor | None,
        mask: torch.Tensor,
        motor_strength: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, D], rbf_feat: [B, N, N, R], mask: [B, N]
        B, N, D = x.shape
        x_f32 = x.float()

        q_raw = self.q_screw(x_f32).view(B, N, self.n_heads, self.dim_motor).permute(0, 2, 1, 3)
        k_raw = self.k_screw(x_f32).view(B, N, self.n_heads, self.dim_motor).permute(0, 2, 1, 3)

        # V21-PGA-M1:
        # Interpret each 8D q/k as a motor-like dual quaternion:
        #   motor = rotor r[4] + eps * dual d[4]
        # Enforce:
        #   ||r|| = 1
        #   <r, d> = 0
        # But unlike M0, do NOT normalize d.  This keeps useful
        # magnitude/confidence information in the dual part.
        q_r = q_raw[..., :4]
        q_d = q_raw[..., 4:]
        k_r = k_raw[..., :4]
        k_d = k_raw[..., 4:]

        q_r = q_r / q_r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        k_r = k_r / k_r.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        q_d = q_d - q_r * (q_r * q_d).sum(dim=-1, keepdim=True)
        k_d = k_d - k_r * (k_r * k_d).sum(dim=-1, keepdim=True)

        rot_score = (q_r.unsqueeze(3) * k_r.unsqueeze(2)).sum(dim=-1)
        dual_score = (q_d.unsqueeze(3) * k_d.unsqueeze(2)).sum(dim=-1) / math.sqrt(4.0)
        dual_score = dual_score.clamp(min=-5.0, max=5.0)

        # Restore logit dynamic range lost by hard normalization.
        motor_score = 8.0 * (rot_score + 0.010 * dual_score)

        geo_bias = self.rbf_bias(rbf_feat).permute(0, 3, 1, 2)
        geo_gate = self.rbf_gate(rbf_feat).permute(0, 3, 1, 2)

        scores = geo_bias

        if coulomb_feat is not None:
            scores = scores + self.coulomb_proj(coulomb_feat).permute(0, 3, 1, 2)

        scores = scores + float(motor_strength) * geo_gate * motor_score

        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        scores = scores.masked_fill(~pair_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)

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
        coulomb_feat: torch.Tensor | None,
        mask: torch.Tensor,
        motor_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out, sig = self.attn(h, rbf_feat, coulomb_feat=coulomb_feat, mask=mask, motor_strength=motor_strength)
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
        use_coulomb: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.use_coulomb = bool(use_coulomb)

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

        coulomb_feat = None
        if self.use_coulomb:
            B, N = z_dense.shape
            pair_mask = mask[:, :, None] & mask[:, None, :]
            eye = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)
            valid_pair = pair_mask & (~eye)

            z_f = z_dense.float() * mask.float()
            zz = z_f[:, :, None] * z_f[:, None, :]
            # Tame scale: raw Z_i Z_j / r can be large for short bonds.
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
            h, sig = layer(h, rbf_feat, coulomb_feat, mask, motor_strength=float(motor_strength))
            motor_sig_total = motor_sig_total + sig

        h = self.norm_final(h)
        graph_h = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        out = self.head(graph_h)

        return out, motor_sig_total / max(len(self.layers), 1)


@torch.no_grad()
def evaluate(model, loader, device, target_mean, target_std, motor_strength: float, target_only: bool = False):
    model.eval()

    target_errs = []
    raw_all_sum = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        pred_norm, _ = model(batch, motor_strength=motor_strength)
        y_norm = normalize_y(batch.y, target_mean, target_std)

        if target_only:
            target_err = (pred_norm.view(-1) - y_norm[:, TARGET_INDEX]).abs()
            target_errs.append(target_err.detach().cpu())
            raw_all_sum += float((target_err * target_std[TARGET_INDEX]).sum().detach().cpu())
        else:
            err_norm_all = (pred_norm - y_norm).abs()
            err_raw_all = err_norm_all * target_std
            target_errs.append(err_norm_all[:, TARGET_INDEX].detach().cpu())
            raw_all_sum += float(err_raw_all.mean(dim=1).sum().detach().cpu())

        total_graphs += batch.y.size(0)

    target_norm = torch.cat(target_errs).mean().item()
    target_raw = target_norm * float(target_std[TARGET_INDEX].detach().cpu())
    return {
        "val_mean_norm_mae": target_norm,
        "val_target_mev": target_raw * MEV,
        "val_mean_raw_mae": raw_all_sum / max(total_graphs, 1),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data-root", default=str(Path.home() / "data/QM9"))
    ap.add_argument("--resume", default=None, help="Optional model checkpoint to warmstart from.")
    ap.add_argument("--seed", type=int, default=43)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--train-size", type=int, default=110000)
    ap.add_argument("--val-size", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pred-batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)

    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=7)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--n-rbf", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--use-coulomb", action="store_true")
    ap.add_argument("--target-only", action="store_true", help="Use a single U0_atom output head.")

    ap.add_argument("--atomaux-indices", default="13,14,15")
    ap.add_argument("--atomaux-lambda-end", type=float, default=0.0003)
    ap.add_argument("--atomaux-warmup", type=float, default=0.0)
    ap.add_argument("--atomaux-ramp", type=float, default=100.0)

    ap.add_argument("--motor-strength", type=float, default=0.0)
    ap.add_argument("--lambda-motor-reg", type=float, default=0.0)

    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = QM9(root=args.data_root)
    train_idx_full, val_idx_full, test_idx = make_splits(len(dataset), args.seed)

    train_idx = train_idx_full[: int(args.train_size)]
    val_idx = val_idx_full[: int(args.val_size)]

    y_full = torch.cat([dataset[int(i)].y for i in train_idx_full], dim=0).float()
    target_mean = y_full.mean(dim=0).to(device)
    target_std = y_full.std(dim=0).clamp_min(1e-8).to(device)

    train_loader = DataLoader(
        [dataset[int(i)] for i in train_idx],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        [dataset[int(i)] for i in val_idx],
        batch_size=args.pred_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    out_dim = 1 if bool(args.target_only) else 19

    model = V20AGAAMotor(
        num_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_rbf=args.n_rbf,
        dropout=args.dropout,
        out_dim=out_dim,
        use_coulomb=bool(args.use_coulomb),
    ).to(device)

    resume_info = None
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        obj = torch.load(ckpt_path, map_location=device)
        state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
        current = model.state_dict()
        filtered = {}
        skipped_shape_keys = []
        for k, v in state.items():
            if k in current and tuple(current[k].shape) == tuple(v.shape):
                filtered[k] = v
            else:
                skipped_shape_keys.append(k)

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        resume_info = {
            "resumed_from": str(ckpt_path),
            "loaded_keys": len(filtered),
            "skipped_shape_keys": skipped_shape_keys,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        print(resume_info, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    aux_indices = [int(x) for x in args.atomaux_indices.split(",") if x.strip()]
    aux_indices = [i for i in aux_indices if i != TARGET_INDEX]

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metadata = {
        "run_name": args.run_name,
        "method": "standalone V21/PGA-M2 scaled motor scout",
        "target": TARGET_NAME,
        "target_index": TARGET_INDEX,
        "split_sizes": {
            "train_used": int(len(train_idx)),
            "val_used": int(len(val_idx)),
            "test_full": int(len(test_idx)),
            "train_full": int(len(train_idx_full)),
            "val_full": int(len(val_idx_full)),
        },
        "args": vars(args),
        "trainable_params": int(trainable_params),
        "resume_info": resume_info,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("Runtime:", {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_cuda_version": torch.version.cuda,
    }, flush=True)
    print("dataset_size:", len(dataset), flush=True)
    print("train_batches:", len(train_loader), flush=True)
    print("val_batches:", len(val_loader), flush=True)
    print("split_sizes:", len(train_idx), len(val_idx), len(test_idx), flush=True)
    print("trainable_params:", int(trainable_params), flush=True)

    best = {
        "best_epoch": None,
        "best_val_target_norm_mae": float("inf"),
        "best_val_target_converted_mae": None,
    }
    best_state = None
    logs = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_graphs = 0
        stat_sums = defaultdict(float)
        stat_count = 0

        for step, batch in enumerate(tqdm(train_loader, leave=False)):
            batch = batch.to(device)

            pred_norm, motor_sig = model(batch, motor_strength=args.motor_strength)
            y_norm = normalize_y(batch.y, target_mean, target_std)

            pred_u0 = pred_norm.view(-1) if bool(args.target_only) else pred_norm[:, TARGET_INDEX]
            primary = F.smooth_l1_loss(pred_u0, y_norm[:, TARGET_INDEX])
            loss = primary

            epoch_float = float(epoch) + float(step) / max(float(len(train_loader)), 1.0)
            aux_lam = scheduled_lambda(
                epoch_float,
                warmup=args.atomaux_warmup,
                ramp=args.atomaux_ramp,
                end=args.atomaux_lambda_end,
            )

            aux_loss = None
            if (not bool(args.target_only)) and aux_indices and aux_lam > 0.0:
                idx_t = torch.tensor(aux_indices, dtype=torch.long, device=device)
                aux_loss = F.smooth_l1_loss(
                    pred_norm.index_select(1, idx_t),
                    y_norm.index_select(1, idx_t),
                )
                loss = loss + float(aux_lam) * aux_loss

            if args.lambda_motor_reg > 0.0 and args.motor_strength > 0.0:
                loss = loss + float(args.lambda_motor_reg) * motor_sig

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = batch.y.size(0)
            total_loss += float(loss.detach().cpu()) * bs
            total_graphs += bs

            stat_sums["primary_sup_loss"] += float(primary.detach().cpu())
            stat_sums["aux_energy_lambda"] += float(aux_lam)
            stat_sums["motor_sig"] += float(motor_sig.detach().cpu())

            if aux_loss is not None:
                stat_sums["aux_energy_loss"] += float(aux_loss.detach().cpu())
                stat_sums["aux_energy_weighted"] += float(aux_lam) * float(aux_loss.detach().cpu())

            stat_count += 1

        val = evaluate(model, val_loader, device, target_mean, target_std, args.motor_strength, target_only=bool(args.target_only))

        if val["val_mean_norm_mae"] < best["best_val_target_norm_mae"]:
            best = {
                "best_epoch": epoch,
                "best_val_target_norm_mae": val["val_mean_norm_mae"],
                "best_val_target_converted_mae": val["val_target_mev"],
            }
            best_state = copy.deepcopy(model.state_dict())

        row = {
            "run": args.run_name,
            "epoch": epoch,
            "train_loss": total_loss / max(total_graphs, 1),
            **val,
            **{k: v / max(stat_count, 1) for k, v in stat_sums.items()},
            **best,
        }
        logs.append(row)

        print(row, flush=True)
        (out_dir / "epoch_log.jsonl").write_text("\n".join(json.dumps(x) for x in logs) + "\n")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best": best,
            },
            out_dir / "latest_checkpoint.pt",
        )

    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "best": best,
                "target": TARGET_NAME,
                "target_index": TARGET_INDEX,
            },
            out_dir / "best_model.pt",
        )

    summary = {
        "run_name": args.run_name,
        "target": TARGET_NAME,
        "target_index": TARGET_INDEX,
        **best,
        "trainable_params": int(trainable_params),
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("SUMMARY")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
