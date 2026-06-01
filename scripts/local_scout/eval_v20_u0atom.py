#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
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


def load_model(run_dir: Path, checkpoint: Path | None, device: torch.device):
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"

    args = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        args.update(summary.get("args", {}) or {})
    elif metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        args.update(metadata.get("args", {}) or {})

    d_model = int(args.get("d_model", 192))
    n_layers = int(args.get("n_layers", 7))
    n_heads = int(args.get("n_heads", 16))
    n_rbf = int(args.get("n_rbf", 20))
    dropout = float(args.get("dropout", 0.0))
    use_coulomb = bool(args.get("use_coulomb", False))
    target_only = bool(args.get("target_only", True))
    out_dim = 1 if target_only else 19

    model = V20AGAAMotor(
        num_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_rbf=n_rbf,
        dropout=dropout,
        out_dim=out_dim,
        use_coulomb=use_coulomb,
    ).to(device)

    ckpt = checkpoint or (run_dir / "best_model.pt")
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    obj = torch.load(ckpt, map_location=device)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    missing, unexpected = model.load_state_dict(state, strict=False)

    print({
        "loaded_checkpoint": str(ckpt),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_rbf": n_rbf,
        "target_only": target_only,
        "use_coulomb": use_coulomb,
    }, flush=True)

    return model, args, ckpt


@torch.no_grad()
def evaluate(model, dataset, indices, device, target_mean, target_std, pred_batch_size, motor_strength):
    model.eval()

    loader = DataLoader(
        [dataset[int(i)] for i in indices],
        batch_size=int(pred_batch_size),
        shuffle=False,
        num_workers=0,
    )

    abs_err_norm = []
    abs_err_mev = []

    for batch in loader:
        batch = batch.to(device)

        pred_norm, _ = model(batch, motor_strength=float(motor_strength))
        pred_u0 = pred_norm.view(-1)

        y_norm = normalize_y(batch.y, target_mean, target_std)
        y_u0 = y_norm[:, TARGET_INDEX]

        err_norm = (pred_u0 - y_u0).abs()
        err_mev = err_norm * target_std[TARGET_INDEX] * MEV

        abs_err_norm.append(err_norm.detach().cpu())
        abs_err_mev.append(err_mev.detach().cpu())

    abs_err_norm = torch.cat(abs_err_norm)
    abs_err_mev = torch.cat(abs_err_mev)

    return {
        "n": int(abs_err_mev.numel()),
        "mae_norm": float(abs_err_norm.mean().item()),
        "mae_mev": float(abs_err_mev.mean().item()),
        "p50_mev": float(abs_err_mev.quantile(0.50).item()),
        "p90_mev": float(abs_err_mev.quantile(0.90).item()),
        "p95_mev": float(abs_err_mev.quantile(0.95).item()),
        "p99_mev": float(abs_err_mev.quantile(0.99).item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--data-root", default=str(Path.home() / "data/QM9"))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--pred-batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--motor-strength", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    dataset = QM9(root=args.data_root)
    train_idx, val_idx, test_idx = make_splits(len(dataset), int(args.seed))

    y_train = torch.cat([dataset[int(i)].y for i in train_idx], dim=0).float()
    target_mean = y_train.mean(dim=0).to(device)
    target_std = y_train.std(dim=0).clamp_min(1e-8).to(device)

    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    model, model_args, ckpt = load_model(run_dir, checkpoint, device)

    indices = val_idx if args.split == "val" else test_idx
    metrics = evaluate(
        model=model,
        dataset=dataset,
        indices=indices,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        pred_batch_size=int(args.pred_batch_size),
        motor_strength=float(args.motor_strength),
    )

    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "split": args.split,
        "target": TARGET_NAME,
        "target_index": TARGET_INDEX,
        "seed": int(args.seed),
        "motor_strength": float(args.motor_strength),
        "device": str(device),
        "pred_batch_size": int(args.pred_batch_size),
        **metrics,
        "model_args": model_args,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2), flush=True)
    print("Wrote:", out, flush=True)


if __name__ == "__main__":
    main()
