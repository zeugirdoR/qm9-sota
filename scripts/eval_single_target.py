#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from qm9sota.models.tiny_radial_mpnn import build_model
from qm9sota.utils.config import load_yaml
from qm9sota.utils.device import get_device
from qm9sota.utils.seed import seed_everything


REPO_ROOT = Path(__file__).resolve().parents[1]

TARGETS = {
    "mu": {"index": 0, "unit": "D", "conversion": 1.0},
    "alpha": {"index": 1, "unit": "a0^3", "conversion": 1.0},
    "homo": {"index": 2, "unit": "meV", "conversion": 1000.0},
    "lumo": {"index": 3, "unit": "meV", "conversion": 1000.0},
    "gap": {"index": 4, "unit": "meV", "conversion": 1000.0},
    "r2": {"index": 5, "unit": "a0^2", "conversion": 1.0},
    "zpve": {"index": 6, "unit": "meV", "conversion": 1000.0},
    "U0": {"index": 7, "unit": "meV_total", "conversion": 1000.0},
    "U": {"index": 8, "unit": "meV_total", "conversion": 1000.0},
    "H": {"index": 9, "unit": "meV_total", "conversion": 1000.0},
    "G": {"index": 10, "unit": "meV_total", "conversion": 1000.0},
    "Cv": {"index": 11, "unit": "cal/mol/K", "conversion": 1.0},
    "U0_atom": {"index": 12, "unit": "meV", "conversion": 1000.0},
    "U_atom": {"index": 13, "unit": "meV", "conversion": 1000.0},
    "H_atom": {"index": 14, "unit": "meV", "conversion": 1000.0},
    "G_atom": {"index": 15, "unit": "meV", "conversion": 1000.0},
}


def resolve_path(x: str | Path) -> Path:
    p = Path(x)
    return p if p.is_absolute() else REPO_ROOT / p


def make_split_indices(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return perm[:110_000], perm[110_000:120_000], perm[120_000:]


def load_checkpoint(path: Path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    target_name = args.target or cfg.get("target", {}).get("name")
    if target_name not in TARGETS:
        raise ValueError(f"Unknown target {target_name}. Known: {sorted(TARGETS)}")

    target_info = TARGETS[target_name]
    target_idx = int(target_info["index"])

    device = get_device(cfg.get("device", "cuda"))

    dataset = QM9(root=str(cfg.get("data", {}).get("root", "/content/data/QM9")))
    train_idx, val_idx, test_idx = make_split_indices(len(dataset), seed)

    ys = torch.cat([data.y for data in dataset], dim=0)
    target_mean = ys[train_idx].mean(dim=0).to(device)
    target_std = ys[train_idx].std(dim=0).clamp_min(1e-8).to(device)

    idx = val_idx if args.split == "val" else test_idx
    ds = dataset.index_select(idx.tolist())

    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("data", {}).get("batch_size", 128)),
        shuffle=False,
        num_workers=0,
    )

    model = build_model(cfg).to(device)
    run_dir = resolve_path(args.run_dir)
    ckpt = run_dir / "best_model.pt"

    model.load_state_dict(load_checkpoint(ckpt, device))
    model.eval()

    err_sum = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)
        pred = pred_norm * target_std + target_mean

        err = (pred[:, target_idx] - batch.y[:, target_idx]).abs()
        err_sum += float(err.sum().detach().cpu())
        n += batch.y.size(0)

    mae_raw = err_sum / max(n, 1)
    mae_units = mae_raw * float(target_info["conversion"])

    result = {
        "run_dir": str(run_dir),
        "config": str(resolve_path(args.config)),
        "seed": seed,
        "split": args.split,
        "target": target_name,
        "target_index": target_idx,
        "mae_raw_pyg_units": mae_raw,
        "unit": target_info["unit"],
        "mae_converted_units": mae_units,
        "n": n,
    }

    out = run_dir / f"single_target_{target_name}_{args.split}_eval.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    main()
