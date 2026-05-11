#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from qm9sota.models.tiny_radial_mpnn import build_model
from qm9sota.utils.config import load_yaml
from qm9sota.utils.device import get_device
from qm9sota.utils.seed import seed_everything


REPO_ROOT = Path(__file__).resolve().parents[1]


# PyG QM9 target order:
# 0 mu, 1 alpha, 2 homo, 3 lumo, 4 gap, 5 r2, 6 zpve,
# 7 U0, 8 U, 9 H, 10 G, 11 Cv,
# 12 U0_atom, 13 U_atom, 14 H_atom, 15 G_atom,
# 16 A, 17 B, 18 C

SOTA12: List[Dict] = [
    {
        "target": "alpha",
        "pyg_index": 1,
        "unit": "a0^3",
        "conversion": 1.0,
        "equiformer_v2": 0.050,
    },
    {
        "target": "gap",
        "pyg_index": 4,
        "unit": "meV",
        "conversion": 1000.0,
        "equiformer_v2": 29.0,
    },
    {
        "target": "homo",
        "pyg_index": 2,
        "unit": "meV",
        "conversion": 1000.0,
        "equiformer_v2": 14.0,
    },
    {
        "target": "lumo",
        "pyg_index": 3,
        "unit": "meV",
        "conversion": 1000.0,
        "equiformer_v2": 13.0,
    },
    {
        "target": "mu",
        "pyg_index": 0,
        "unit": "D",
        "conversion": 1.0,
        "equiformer_v2": 0.010,
    },
    {
        "target": "Cv",
        "pyg_index": 11,
        "unit": "cal/mol/K",
        "conversion": 1.0,
        "equiformer_v2": 0.023,
    },
    {
        "target": "G",
        "pyg_index": 10,
        "unit": "meV",
        "conversion": 1000.0,
        "equiformer_v2": 7.57,
    },
    {
        "target": "H",
        "pyg_index": 9,
        "unit": "meV",
        "conversion": 1000.0,
        "equiformer_v2": 6.22,
    },
    {
        "target": "r2",
        "pyg_index": 5,
        "unit": "a0^2",
        "conversion": 1.0,
        "equiformer_v2": 0.186,
    },
    {
        "target": "U",
        "pyg_index": 8,
        "unit": "meV",
        "conversion": 1000.0,
        "equiformer_v2": 6.49,
    },
    {
        "target": "U0",
        "pyg_index": 7,
        "unit": "meV",
        "conversion": 1000.0,
        "equiformer_v2": 6.17,
    },
    {
        "target": "zpve",
        "pyg_index": 6,
        "unit": "meV",
        "conversion": 1000.0,
        "equiformer_v2": 1.47,
    },
]


def resolve_path(path_like: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return base / path


def make_split_indices(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)

    train_idx = perm[:110_000]
    val_idx = perm[110_000:120_000]
    test_idx = perm[120_000:]

    return train_idx, val_idx, test_idx


def load_checkpoint(path: Path, device: torch.device):
    obj = torch.load(path, map_location=device)

    # Current runner appears to save raw state_dict.
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]

    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]

    return obj


@torch.no_grad()
def evaluate_raw_mae(model, loader, target_mean, target_std, device):
    model.eval()

    abs_err_sum = torch.zeros(19, device=device)
    n = 0

    for batch in loader:
        batch = batch.to(device)

        pred_norm = model(batch)
        pred_raw = pred_norm * target_std + target_mean

        err = (pred_raw - batch.y).abs()
        abs_err_sum += err.sum(dim=0)
        n += batch.y.size(0)

    return (abs_err_sum / max(n, 1)).detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a QM9 checkpoint on the 12-target Equiformer-style SOTA table."
    )
    parser.add_argument("--config", required=True, help="Training config used for the model.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing best_model.pt.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to run-dir/best_model.pt.")
    parser.add_argument("--seed", type=int, default=None, help="Split/model seed override. Defaults to config seed.")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size override.")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to run-dir/qm9_sota12_eval.csv.")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    run_dir = resolve_path(args.run_dir)

    cfg = load_yaml(config_path)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    device = get_device(cfg.get("device", "cuda"))

    data_root = cfg.get("data", {}).get("root", "/content/data/QM9")
    dataset = QM9(root=str(data_root))

    train_idx, val_idx, test_idx = make_split_indices(len(dataset), seed)

    ys = torch.cat([data.y for data in dataset], dim=0)
    target_mean = ys[train_idx].mean(dim=0).to(device)
    target_std = ys[train_idx].std(dim=0).clamp_min(1e-8).to(device)

    test_dataset = dataset.index_select(test_idx.tolist())

    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else cfg.get("data", {}).get("batch_size", 128)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg).to(device)

    ckpt_path = resolve_path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pt"
    state = load_checkpoint(ckpt_path, device)
    model.load_state_dict(state)

    raw_mae = evaluate_raw_mae(
        model=model,
        loader=test_loader,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
    )

    rows = []

    for item in SOTA12:
        idx = int(item["pyg_index"])
        raw = float(raw_mae[idx].item())
        converted = raw * float(item["conversion"])
        sota = float(item["equiformer_v2"])

        rows.append(
            {
                "target": item["target"],
                "pyg_index": idx,
                "unit": item["unit"],
                "test_mae_pyg_units": raw,
                "conversion": float(item["conversion"]),
                "test_mae_sota_units": converted,
                "equiformer_v2_mae": sota,
                "ours_minus_equiformer_v2": converted - sota,
                "ratio_ours_to_equiformer_v2": converted / sota if sota != 0 else float("nan"),
            }
        )

    df = pd.DataFrame(rows)

    output_path = resolve_path(args.output) if args.output else run_dir / "qm9_sota12_eval.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = {
        "config": str(config_path),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "seed": seed,
        "dataset_size": len(dataset),
        "test_size": len(test_idx),
        "output": str(output_path),
        "mean_ratio_ours_to_equiformer_v2": float(df["ratio_ours_to_equiformer_v2"].mean()),
        "num_targets_beating_equiformer_v2": int((df["ours_minus_equiformer_v2"] < 0).sum()),
    }

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:", output_path)
    print("Wrote:", summary_path)
    print()
    print(df.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
