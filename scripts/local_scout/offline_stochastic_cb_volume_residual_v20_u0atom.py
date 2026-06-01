#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.local_scout.train_v20_agaa_motor_u0atom import (
    V20AGAAMotor,
    make_splits,
    normalize_y,
)

TARGET_INDEX = 12
TARGET_NAME = "U0_atom"
MEV = 1000.0


class ResidualHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        if int(hidden_dim) <= 0:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 1),
            )
        else:
            h = int(hidden_dim)
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def load_feature_cache(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for key in ("features", "x", "cache", "feat"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key].float()
    raise TypeError(f"Unsupported feature cache format: {path}")


def load_v20_model(run_dir: Path, checkpoint: Path | None, device: torch.device, motor_strength: float):
    summary_path = run_dir / "summary.json"
    meta_path = run_dir / "metadata.json"

    cfg_args = {}
    if summary_path.exists():
        obj = json.loads(summary_path.read_text())
        cfg_args.update(obj.get("args", {}) or {})
    elif meta_path.exists():
        obj = json.loads(meta_path.read_text())
        cfg_args.update(obj.get("args", {}) or {})

    n_layers = int(cfg_args.get("n_layers", 7))
    d_model = int(cfg_args.get("d_model", 192))
    n_heads = int(cfg_args.get("n_heads", 16))
    n_rbf = int(cfg_args.get("n_rbf", 20))
    dropout = float(cfg_args.get("dropout", 0.0))
    use_coulomb = bool(cfg_args.get("use_coulomb", False))

    model = V20AGAAMotor(
        num_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_rbf=n_rbf,
        dropout=dropout,
        out_dim=1,
        use_coulomb=use_coulomb,
    ).to(device)

    ckpt_path = checkpoint or (run_dir / "best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    obj = torch.load(ckpt_path, map_location=device)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    missing, unexpected = model.load_state_dict(state, strict=False)

    print({
        "loaded_checkpoint": str(ckpt_path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_rbf": n_rbf,
        "dropout": dropout,
        "use_coulomb": use_coulomb,
        "motor_strength": motor_strength,
    }, flush=True)

    return model, ckpt_path, cfg_args


@torch.no_grad()
def compute_preds(
    model: nn.Module,
    dataset,
    indices: torch.Tensor,
    device: torch.device,
    pred_batch_size: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    motor_strength: float,
):
    model.eval()
    loader = DataLoader(
        [dataset[int(i)] for i in indices],
        batch_size=int(pred_batch_size),
        shuffle=False,
        num_workers=0,
    )

    pred_norms = []
    y_raws = []

    for batch in loader:
        batch = batch.to(device)
        pred_norm, _ = model(batch, motor_strength=float(motor_strength))
        pred_norms.append(pred_norm.view(-1).detach().cpu())
        y_raws.append(batch.y[:, TARGET_INDEX].detach().cpu().float())

    pred_norm = torch.cat(pred_norms, dim=0).float()
    y_raw = torch.cat(y_raws, dim=0).float()
    y_norm = ((y_raw.to(device) - target_mean[TARGET_INDEX]) / target_std[TARGET_INDEX].clamp_min(1e-8)).detach().cpu()
    return pred_norm, y_norm, y_raw


@torch.no_grad()
def mae_mev_from_norm(pred_norm: torch.Tensor, y_norm: torch.Tensor, target_std_scalar: float) -> float:
    return float((pred_norm - y_norm).abs().mean().item() * target_std_scalar * MEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--feature-cache", required=True)
    ap.add_argument("--data-root", default=str(Path.home() / "data/QM9"))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--feature-seed", type=int, default=43)
    ap.add_argument("--minor-k", type=int, default=2)
    ap.add_argument("--samples-per-center", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--weight-decay", type=float, default=0.0001)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--pred-batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--motor-strength", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset = QM9(root=args.data_root)
    train_idx, val_idx, test_idx = make_splits(len(dataset), args.seed)

    y_full = torch.cat([dataset[int(i)].y for i in train_idx], dim=0).float()
    target_mean = y_full.mean(dim=0).to(device)
    target_std = y_full.std(dim=0).clamp_min(1e-8).to(device)
    target_std_scalar = float(target_std[TARGET_INDEX].detach().cpu())

    feature_cache = load_feature_cache(Path(args.feature_cache))
    if feature_cache.size(0) != len(dataset):
        raise ValueError(f"feature cache rows {feature_cache.size(0)} != dataset size {len(dataset)}")

    model, ckpt_path, model_args = load_v20_model(
        Path(args.run_dir),
        Path(args.checkpoint) if args.checkpoint else None,
        device,
        args.motor_strength,
    )

    print("Computing frozen V20 predictions...", flush=True)
    pred_train_norm, y_train_norm, _ = compute_preds(
        model, dataset, train_idx, device, args.pred_batch_size, target_mean, target_std, args.motor_strength
    )
    pred_val_norm, y_val_norm, _ = compute_preds(
        model, dataset, val_idx, device, args.pred_batch_size, target_mean, target_std, args.motor_strength
    )

    base_val_mev = mae_mev_from_norm(pred_val_norm, y_val_norm, target_std_scalar)
    base_train_mev = mae_mev_from_norm(pred_train_norm, y_train_norm, target_std_scalar)

    x_train = feature_cache[train_idx].float()
    x_val = feature_cache[val_idx].float()

    residual_train = (y_train_norm - pred_train_norm).float()
    residual_val = (y_val_norm - pred_val_norm).float()

    head = ResidualHead(x_train.shape[1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {
        "best_epoch": None,
        "best_val_mev": float("inf"),
        "best_delta_val_mev": None,
    }
    best_state = None
    logs = []

    n = x_train.size(0)
    generator = torch.Generator().manual_seed(args.seed + int(args.feature_seed) + 909)

    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        perm = torch.randperm(n, generator=generator)
        total_loss = 0.0
        total_seen = 0

        for start in range(0, n, int(args.batch_size)):
            idx = perm[start:start + int(args.batch_size)]
            xb = x_train[idx].to(device)
            rb = residual_train[idx].to(device)

            pred_res = head(xb)
            loss = F.smooth_l1_loss(pred_res, rb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu()) * idx.numel()
            total_seen += idx.numel()

        head.eval()
        with torch.no_grad():
            corr_val = []
            for start in range(0, x_val.size(0), int(args.batch_size)):
                xb = x_val[start:start + int(args.batch_size)].to(device)
                corr_val.append(head(xb).detach().cpu())
            corr_val = torch.cat(corr_val, dim=0).float()

            corrected_val_norm = pred_val_norm + corr_val
            val_mev = mae_mev_from_norm(corrected_val_norm, y_val_norm, target_std_scalar)
            delta = val_mev - base_val_mev

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_seen, 1),
            "val_mev": val_mev,
            "delta_val_mev": delta,
        }
        logs.append(row)

        if val_mev < best["best_val_mev"]:
            best = {
                "best_epoch": epoch,
                "best_val_mev": val_mev,
                "best_delta_val_mev": delta,
            }
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        if epoch <= 5 or epoch % 10 == 0 or epoch == int(args.epochs):
            print({**row, **best}, flush=True)

    best_path = out_dir / "best_residual_head.pt"
    if best_state is not None:
        torch.save(
            {
                "model_state_dict": best_state,
                "feature_dim": int(x_train.shape[1]),
                "hidden_dim": int(args.hidden_dim),
                "target": TARGET_NAME,
                "target_index": TARGET_INDEX,
                "base_checkpoint": str(ckpt_path),
                "feature_cache": str(args.feature_cache),
                "motor_strength": float(args.motor_strength),
            },
            best_path,
        )

    summary = {
        "target": TARGET_NAME,
        "target_index": TARGET_INDEX,
        "seed": int(args.seed),
        "feature_seed": int(args.feature_seed),
        "method": "V20 target-only offline stochastic CB residual",
        "base_run_dir": str(args.run_dir),
        "base_checkpoint": str(ckpt_path),
        "feature_cache": str(args.feature_cache),
        "feature_dim": int(x_train.shape[1]),
        "minor_k": int(args.minor_k),
        "samples_per_center": int(args.samples_per_center),
        "hidden_dim": int(args.hidden_dim),
        "base_train_mev": base_train_mev,
        "base_val_mev": base_val_mev,
        **best,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "motor_strength": float(args.motor_strength),
        "best_residual_head": str(best_path),
        "model_args": model_args,
    }

    (out_dir / "offline_stochastic_cb_volume_residual_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "epoch_log.jsonl").write_text("\n".join(json.dumps(x) for x in logs) + "\n")

    print("RESULT")
    print(json.dumps(summary, indent=2), flush=True)
    print("Wrote:", out_dir / "offline_stochastic_cb_volume_residual_summary.json", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
