
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qm9sota.models.tiny_radial_mpnn import build_model
from qm9sota.utils.config import load_yaml

MEV = 1000.0


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(in_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def resolve_path(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def make_splits(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return {
        "train": perm[:110000],
        "val": perm[110000:120000],
        "test": perm[120000:],
    }


def parse_member(text: str):
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError("member must be name:head_path:feature_cache_path")
    return parts[0], resolve_path(parts[1]), resolve_path(parts[2])


@torch.no_grad()
def compute_base_preds(model, dataset, indices, device, batch_size: int, target_index: int):
    subset = [dataset[int(i)] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    preds = []
    ys = []
    model.eval()

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        preds.append(out[:, target_index].detach().cpu())
        ys.append(batch.y[:, target_index].detach().cpu())

    return torch.cat(preds), torch.cat(ys)


def mae_mev(pred_norm, y_norm, y_mu, y_sd):
    pred_raw = pred_norm * y_sd + y_mu
    y_raw = y_norm * y_sd + y_mu
    return float((pred_raw - y_raw).abs().mean() * MEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pred-batch-size", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    manifest_path = resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text())

    members = [parse_member(m) for m in args.member]
    if args.weights:
        weights = torch.tensor([float(x) for x in args.weights.split(",")], dtype=torch.float32)
        if len(weights) != len(members):
            raise ValueError("weights length must match member count")
    else:
        weights = torch.ones(len(members), dtype=torch.float32)
    weights = weights / weights.sum()

    data_root = manifest.get("data_root", str(Path.home() / "data/QM9"))
    seed = int(manifest.get("split", {}).get("seed", manifest.get("seed", 43)))
    target_index = int(manifest.get("target_index", 12))

    config_path = resolve_path(manifest["config"])
    base_checkpoint = resolve_path(manifest["base_checkpoint"])

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    cfg = load_yaml(config_path)
    cfg.setdefault("data", {})
    cfg["data"]["root"] = data_root
    cfg["data"]["smoke"] = False
    cfg["data"]["batch_size"] = args.pred_batch_size

    dataset = QM9(root=data_root)
    splits = make_splits(len(dataset), seed)
    idx = splits[args.split]

    y_full = torch.cat([data.y for data in dataset], dim=0).float()
    train_idx = splits["train"]
    target_mean = y_full[train_idx].mean(dim=0)
    target_std = y_full[train_idx].std(dim=0).clamp_min(1e-8)
    y_mu = target_mean[target_index]
    y_sd = target_std[target_index]

    model = build_model(cfg).to(device)
    obj = torch.load(base_checkpoint, map_location=device)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state, strict=False)

    pred_norm, y_raw = compute_base_preds(model, dataset, idx, device, args.pred_batch_size, target_index)
    y_norm = (y_raw - y_mu) / y_sd

    total_corr = torch.zeros_like(pred_norm)
    member_results = []

    for w, member in zip(weights, members):
        name, head_path, cache_path = member

        cache = torch.load(cache_path, map_location="cpu")
        X = cache["features"].float()

        head_ckpt = torch.load(head_path, map_location="cpu")
        x_mu = head_ckpt["feature_mean"].float()
        x_sd = head_ckpt["feature_std"].float().clamp_min(1e-8)
        hidden_dim = int(head_ckpt.get("hidden_dim", 64))

        residual = ResidualMLP(X.shape[1], hidden_dim=hidden_dim)
        residual.load_state_dict(head_ckpt["model_state_dict"])
        residual.eval()

        Xn = (X[idx] - x_mu) / x_sd
        with torch.no_grad():
            corr = residual(Xn)

        total_corr += float(w) * corr
        member_results.append({
            "name": name,
            "weight": float(w),
            "head": str(head_path.relative_to(ROOT) if head_path.is_relative_to(ROOT) else head_path),
            "feature_cache": str(cache_path.relative_to(ROOT) if cache_path.is_relative_to(ROOT) else cache_path),
            "corr_abs_mean_norm": float(corr.abs().mean()),
        })

    corrected_pred_norm = pred_norm + float(args.alpha) * total_corr

    base_mev = mae_mev(pred_norm, y_norm, y_mu, y_sd)
    corrected_mev = mae_mev(corrected_pred_norm, y_norm, y_mu, y_sd)

    result = {
        "artifact_manifest": str(manifest_path.relative_to(ROOT) if manifest_path.is_relative_to(ROOT) else manifest_path),
        "split": args.split,
        "target": manifest.get("target", "U0_atom"),
        "target_index": target_index,
        "n": int(len(idx)),
        "base_mae_mev": base_mev,
        "corrected_mae_mev": corrected_mev,
        "delta_mev": corrected_mev - base_mev,
        "alpha": float(args.alpha),
        "members": member_results,
        "config": str(config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path),
        "base_checkpoint": str(base_checkpoint.relative_to(ROOT) if base_checkpoint.is_relative_to(ROOT) else base_checkpoint),
        "device": str(device),
        "pred_batch_size": args.pred_batch_size,
    }

    text = json.dumps(result, indent=2)
    print(text)

    if args.out:
        out = resolve_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        print("Wrote:", out)


if __name__ == "__main__":
    main()
