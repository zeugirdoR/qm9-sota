from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml

from qm9sota.data.qm9 import TARGET_NAMES
from qm9sota.train.evaluate import evaluate_both, normalize_y


def make_optimizer(model, loss_fn, cfg: dict):
    opt_cfg = cfg["optimizer"]
    lr = float(opt_cfg.get("lr", 3e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-6))

    params = [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}]
    if loss_fn is not None:
        loss_params = list(loss_fn.parameters())
        if loss_params:
            params.append({"params": loss_params, "lr": lr * 0.1, "weight_decay": 0.0})

    return torch.optim.AdamW(params)


def train_one_epoch(model, loader, optimizer, device, target_mean, target_std, global_step: int, droplet_loss=None, grad_clip: float = 5.0):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    stat_sums = defaultdict(float)
    stat_count = 0

    for batch in tqdm(loader, leave=False):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred_norm = model(batch)
        y_norm = normalize_y(batch.y, target_mean, target_std)

        if droplet_loss is None:
            loss = F.smooth_l1_loss(pred_norm, y_norm)
            stats = {}
        else:
            loss, stats = droplet_loss(pred=pred_norm, target=y_norm, step=global_step)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch.y.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_graphs += batch_size
        global_step += 1

        if stats:
            for k, v in stats.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    stat_sums[k] += float(v.detach().cpu())
                elif isinstance(v, (float, int)):
                    stat_sums[k] += float(v)
            stat_count += 1

    avg_stats = {k: v / stat_count for k, v in stat_sums.items()} if stat_count else {}
    return total_loss / max(total_graphs, 1), global_step, avg_stats


def run_training(
    *,
    cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    run_name: str,
    model,
    bundle,
    device,
    output_root: Path,
    config_paths: dict[str, str],
    extra_metadata: dict[str, Any] | None = None,
):
    target_mean = bundle.target_mean.to(device)
    target_std = bundle.target_std.to(device)
    model = model.to(device)

    loss_name = loss_cfg.get("loss", {}).get("name", "baseline")
    droplet_loss = None
    if loss_name == "droplet":
        from qm9sota.losses.droplet import build_droplet_loss
        droplet_loss = build_droplet_loss(loss_cfg, steps_per_epoch=len(bundle.train_loader), device=device)
    elif loss_name != "baseline":
        raise ValueError(f"Unknown loss name: {loss_name}")

    optimizer = make_optimizer(model, droplet_loss, cfg)
    grad_clip = float(cfg["optimizer"].get("grad_clip", 5.0))
    epochs = int(cfg["train"].get("epochs", 5))

    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "train_config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    with (run_dir / "loss_config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(loss_cfg, f)
    metadata = {
        "run_name": run_name,
        "config_paths": config_paths,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    global_step = 0
    logs = []
    best_norm = float("inf")
    best_epoch = None
    best_raw_mae = None
    best_norm_mae = None
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, global_step, stats = train_one_epoch(
            model=model,
            loader=bundle.train_loader,
            optimizer=optimizer,
            device=device,
            target_mean=target_mean,
            target_std=target_std,
            global_step=global_step,
            droplet_loss=droplet_loss,
            grad_clip=grad_clip,
        )

        raw_mae, norm_mae = evaluate_both(model, bundle.val_loader, device, target_mean, target_std)
        mean_raw = float(raw_mae.mean())
        mean_norm = float(norm_mae.mean())

        log = {
            "run": run_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "global_step": global_step,
            "val_mean_raw_mae": mean_raw,
            "val_mean_norm_mae": mean_norm,
        }
        log.update(stats)
        logs.append(log)
        print(log)

        if mean_norm < best_norm:
            best_norm = mean_norm
            best_epoch = epoch
            best_raw_mae = raw_mae
            best_norm_mae = norm_mae
            best_state = copy.deepcopy(model.state_dict())

        pd.DataFrame(logs).to_csv(run_dir / "epoch_log.csv", index=False)

    if best_state is not None:
        torch.save(best_state, run_dir / "best_model.pt")

    per_target = pd.DataFrame({
        "target": TARGET_NAMES,
        "raw_mae": best_raw_mae.numpy(),
        "normalized_mae": best_norm_mae.numpy(),
    })
    per_target.to_csv(run_dir / "best_per_target_mae.csv", index=False)

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_mean_norm_mae": float(best_norm),
        "best_val_mean_raw_mae": float(best_raw_mae.mean()),
    }

    if extra_metadata:
        summary.update(
            {
                "git_commit": extra_metadata.get("git_commit", "unknown"),
                "runtime": extra_metadata.get("runtime", {}),
            }
        )
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(summary)
    return summary
