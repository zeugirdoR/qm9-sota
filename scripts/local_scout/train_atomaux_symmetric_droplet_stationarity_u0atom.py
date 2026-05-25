#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qm9sota.models.tiny_radial_mpnn import build_model
from qm9sota.utils.config import load_yaml
from qm9sota.utils.seed import seed_everything

TARGET_INDEX = 12
TARGET_NAME = "U0_atom"
MEV = 1000.0


def make_splits(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return perm[:110000], perm[110000:120000], perm[120000:]


def normalize_y(y, mean, std):
    return (y.float() - mean) / std.clamp_min(1e-8)


def scheduled_lambda(epoch_float: float, *, warmup: float, ramp: float, end: float) -> float:
    if epoch_float < warmup:
        return 0.0
    if ramp <= 0:
        return float(end)
    frac = max(0.0, min(1.0, (epoch_float - warmup) / ramp))
    return float(end) * 0.5 * (1.0 - math.cos(math.pi * frac))


def remove_translation(delta: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    out = delta.clone()
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    for g in range(n_graphs):
        mask = batch == g
        if mask.any():
            out[mask] = out[mask] - out[mask].mean(dim=0, keepdim=True)
    return out


def graph_perturb_stats(pos0, pos1, edge_index, batch):
    device = pos0.device
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    delta = pos1 - pos0

    rmsd = torch.zeros(n_graphs, device=device)
    max_disp = torch.zeros(n_graphs, device=device)

    for g in range(n_graphs):
        mask = batch == g
        d = delta[mask].norm(dim=-1)
        if d.numel():
            rmsd[g] = d.pow(2).mean().clamp_min(1e-12).sqrt()
            max_disp[g] = d.max()

    src, dst = edge_index
    # Deduplicate if both directions are present.
    keep_edge = src < dst
    src = src[keep_edge]
    dst = dst[keep_edge]

    if src.numel() == 0:
        return rmsd, max_disp, torch.zeros(n_graphs, device=device)

    edge_graph = batch[src]
    r0 = (pos0[src] - pos0[dst]).norm(dim=-1).clamp_min(1e-8)
    r1 = (pos1[src] - pos1[dst]).norm(dim=-1).clamp_min(1e-8)
    stretch = (r1 - r0).abs()

    max_stretch = torch.zeros(n_graphs, device=device)
    for g in range(n_graphs):
        m = edge_graph == g
        if m.any():
            max_stretch[g] = stretch[m].max()

    return rmsd, max_disp, max_stretch


def droplet_stationarity_loss(
    *,
    model,
    batch,
    target_index: int,
    sigma: float,
    graph_fraction: float,
    keep_quantile: float,
    max_bond_stretch: float,
    generator: torch.Generator,
):
    n_graphs = batch.y.size(0)
    device = batch.pos.device

    selected = torch.rand(n_graphs, device=device, generator=generator) < float(graph_fraction)
    if selected.sum().item() == 0:
        return None, {}

    node_selected = selected[batch.batch]

    pos0 = batch.pos.float()
    delta = torch.randn(pos0.shape, device=device, generator=generator) * float(sigma)
    delta = remove_translation(delta, batch.batch)
    delta = delta * node_selected.unsqueeze(-1)

    pos_plus = pos0 + delta
    pos_minus = pos0 - delta

    rmsd, max_disp, max_stretch = graph_perturb_stats(pos0, pos_plus, batch.edge_index, batch.batch)

    # Amputate outlier droplets among selected graphs.
    selected_scores = rmsd[selected] + 0.1 * max_disp[selected] + 0.5 * max_stretch[selected]
    if selected_scores.numel() == 0:
        return None, {}

    cutoff = torch.quantile(selected_scores, float(keep_quantile))
    score = rmsd + 0.1 * max_disp + 0.5 * max_stretch
    kept = selected & (score <= cutoff) & (max_stretch <= float(max_bond_stretch))

    if kept.sum().item() == 0:
        return None, {
            "droplet_selected_graphs": float(selected.sum().item()),
            "droplet_kept_graphs": 0.0,
        }

    batch_plus = batch.clone()
    batch_minus = batch.clone()
    batch_plus.pos = pos_plus
    batch_minus.pos = pos_minus

    pred_plus = model(batch_plus)[:, target_index]
    pred_minus = model(batch_minus)[:, target_index]

    odd = 0.5 * (pred_plus - pred_minus)
    loss = F.smooth_l1_loss(odd[kept], torch.zeros_like(odd[kept]))

    stats = {
        "droplet_stationarity_loss": loss.detach(),
        "droplet_selected_graphs": float(selected.sum().item()),
        "droplet_kept_graphs": float(kept.sum().item()),
        "droplet_kept_frac": float(kept.sum().item() / max(float(selected.sum().item()), 1.0)),
        "droplet_rmsd_mean_a": rmsd[kept].mean().detach(),
        "droplet_rmsd_p95_a": torch.quantile(rmsd[kept], 0.95).detach(),
        "droplet_max_stretch_p95_a": torch.quantile(max_stretch[kept], 0.95).detach(),
        "droplet_odd_abs_mean_norm": odd[kept].abs().mean().detach(),
    }
    return loss, stats


@torch.no_grad()
def evaluate(model, loader, device, target_mean, target_std, target_index):
    model.eval()
    abs_norm = []
    abs_raw = []
    all_raw_mae_sum = 0.0
    n_graphs_total = 0

    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)
        y_norm = normalize_y(batch.y, target_mean, target_std)

        err_norm_all = (pred_norm - y_norm).abs()
        err_raw_all = err_norm_all * target_std.to(device)

        abs_norm.append(err_norm_all[:, target_index].detach().cpu())
        abs_raw.append(err_raw_all[:, target_index].detach().cpu())

        all_raw_mae_sum += float(err_raw_all.mean(dim=1).sum().detach().cpu())
        n_graphs_total += batch.y.size(0)

    target_norm = torch.cat(abs_norm).mean().item()
    target_raw = torch.cat(abs_raw).mean().item()
    mean_raw = all_raw_mae_sum / max(n_graphs_total, 1)

    return {
        "val_mean_raw_mae": mean_raw,
        "val_mean_norm_mae": target_norm,
        "val_target_raw_mae": target_raw,
        "val_target_mev": target_raw * MEV,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data-root", default=str(Path.home() / "data/QM9"))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pred-batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=5.0)

    # Gentle atomization auxiliary, matching the successful scout.
    ap.add_argument("--atomaux-indices", default="13,14,15")
    ap.add_argument("--atomaux-lambda-end", type=float, default=0.0003)
    ap.add_argument("--atomaux-warmup", type=float, default=400.0)
    ap.add_argument("--atomaux-ramp", type=float, default=100.0)

    # Symmetric droplet stationarity.
    ap.add_argument("--droplet-sigma", type=float, default=0.0025)
    ap.add_argument("--droplet-every-steps", type=int, default=16)
    ap.add_argument("--droplet-graph-fraction", type=float, default=0.25)
    ap.add_argument("--droplet-keep-quantile", type=float, default=0.75)
    ap.add_argument("--droplet-max-bond-stretch", type=float, default=0.015)
    ap.add_argument("--droplet-ratio-end", type=float, default=0.01)
    ap.add_argument("--droplet-ratio-warmup", type=float, default=400.0)
    ap.add_argument("--droplet-ratio-ramp", type=float, default=50.0)
    ap.add_argument("--droplet-lambda-cap", type=float, default=1.0)

    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(Path(args.config))
    cfg.setdefault("data", {})
    cfg["data"]["root"] = args.data_root
    cfg["data"]["batch_size"] = args.batch_size
    cfg["data"]["smoke"] = False

    dataset = QM9(root=args.data_root)
    train_idx, val_idx, test_idx = make_splits(len(dataset), args.seed)

    y_full = torch.cat([data.y for data in dataset], dim=0).float()
    target_mean = y_full[train_idx].mean(dim=0).to(device)
    target_std = y_full[train_idx].std(dim=0).clamp_min(1e-8).to(device)

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

    model = build_model(cfg).to(device)
    obj = torch.load(args.resume, map_location=device)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    missing, unexpected = model.load_state_dict(state, strict=False)

    start_epoch = int(obj.get("epoch", 400)) + 1 if isinstance(obj, dict) else 401
    global_step = int(obj.get("global_step", 0)) if isinstance(obj, dict) and obj.get("global_step") is not None else 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    aux_indices = [int(x) for x in args.atomaux_indices.split(",") if x.strip()]
    aux_indices = [i for i in aux_indices if i != TARGET_INDEX]

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 424242)

    best = {
        "best_epoch": None,
        "best_val_target_norm_mae": float("inf"),
        "best_val_target_converted_mae": None,
    }
    best_state = None
    logs = []

    metadata = {
        "run_name": args.run_name,
        "method": "atomaux plus symmetric droplet stationarity scout",
        "config": args.config,
        "resume": args.resume,
        "seed": args.seed,
        "target": TARGET_NAME,
        "target_index": TARGET_INDEX,
        "split_sizes": [len(train_idx), len(val_idx), len(test_idx)],
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "args": vars(args),
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
    print({
        "resumed_from": args.resume,
        "start_epoch": start_epoch,
        "global_step": global_step,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }, flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        stat_sums = defaultdict(float)
        stat_count = 0

        for step_in_epoch, batch in enumerate(tqdm(train_loader, leave=False)):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)

            if hasattr(model, "set_epoch_float"):
                epoch_float_for_model = float(epoch) + float(step_in_epoch) / max(float(len(train_loader)), 1.0)
                model.set_epoch_float(epoch_float_for_model)

            pred_norm = model(batch)
            y_norm = normalize_y(batch.y, target_mean, target_std)

            primary = F.smooth_l1_loss(pred_norm[:, TARGET_INDEX], y_norm[:, TARGET_INDEX])
            loss = primary

            epoch_float = float(epoch) + float(step_in_epoch) / max(float(len(train_loader)), 1.0)

            aux_lam = scheduled_lambda(
                epoch_float,
                warmup=args.atomaux_warmup,
                ramp=args.atomaux_ramp,
                end=args.atomaux_lambda_end,
            )
            aux_loss = None
            if aux_indices and aux_lam > 0.0:
                idx_t = torch.tensor(aux_indices, dtype=torch.long, device=device)
                aux_loss = F.smooth_l1_loss(
                    pred_norm.index_select(1, idx_t),
                    y_norm.index_select(1, idx_t),
                )
                loss = loss + float(aux_lam) * aux_loss

            droplet_loss = None
            droplet_lambda = 0.0
            droplet_ratio = scheduled_lambda(
                epoch_float,
                warmup=args.droplet_ratio_warmup,
                ramp=args.droplet_ratio_ramp,
                end=args.droplet_ratio_end,
            )

            droplet_stats = {}
            if args.droplet_every_steps > 0 and (global_step % args.droplet_every_steps == 0) and droplet_ratio > 0.0:
                droplet_loss, droplet_stats = droplet_stationarity_loss(
                    model=model,
                    batch=batch,
                    target_index=TARGET_INDEX,
                    sigma=args.droplet_sigma,
                    graph_fraction=args.droplet_graph_fraction,
                    keep_quantile=args.droplet_keep_quantile,
                    max_bond_stretch=args.droplet_max_bond_stretch,
                    generator=gen,
                )
                if droplet_loss is not None:
                    target_contrib = float(droplet_ratio) * primary.detach()
                    droplet_lambda_t = target_contrib / droplet_loss.detach().clamp_min(1e-12)
                    droplet_lambda_t = droplet_lambda_t.clamp(max=float(args.droplet_lambda_cap))
                    droplet_lambda = float(droplet_lambda_t.detach().cpu())
                    loss = loss + droplet_lambda_t * droplet_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            bs = batch.y.size(0)
            total_loss += float(loss.detach().cpu()) * bs
            total_graphs += bs
            global_step += 1

            stats = {
                "primary_sup_loss": float(primary.detach().cpu()),
                "aux_energy_lambda": float(aux_lam),
            }
            if aux_loss is not None:
                stats["aux_energy_loss"] = float(aux_loss.detach().cpu())
                stats["aux_energy_weighted"] = float(aux_lam) * float(aux_loss.detach().cpu())
            if droplet_loss is not None:
                stats["droplet_stationarity_loss"] = float(droplet_loss.detach().cpu())
                stats["droplet_lambda"] = float(droplet_lambda)
                stats["droplet_ratio"] = float(droplet_ratio)
                stats["droplet_weighted"] = float(droplet_lambda) * float(droplet_loss.detach().cpu())
                for k, v in droplet_stats.items():
                    stats[k] = float(v.detach().cpu()) if torch.is_tensor(v) else float(v)

            for k, v in stats.items():
                stat_sums[k] += float(v)
            stat_count += 1

        avg_stats = {k: v / max(stat_count, 1) for k, v in stat_sums.items()}
        val = evaluate(model, val_loader, device, target_mean, target_std, TARGET_INDEX)

        is_best = val["val_mean_norm_mae"] < best["best_val_target_norm_mae"]
        if is_best:
            best = {
                "best_epoch": epoch,
                "best_val_target_norm_mae": val["val_mean_norm_mae"],
                "best_val_target_converted_mae": val["val_target_mev"],
            }
            best_state = copy.deepcopy(model.state_dict())

        log = {
            "run": args.run_name,
            "epoch": epoch,
            "train_loss": total_loss / max(total_graphs, 1),
            "global_step": global_step,
            **val,
            **avg_stats,
            **best,
            "current_is_best": is_best,
        }
        logs.append(log)
        print(log, flush=True)
        (out_dir / "epoch_log.jsonl").write_text("\n".join(json.dumps(r) for r in logs) + "\n")

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best": best,
            },
            out_dir / "latest_checkpoint.pt",
        )

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "best": best,
                },
                out_dir / f"checkpoint_epoch_{epoch}.pt",
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
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
