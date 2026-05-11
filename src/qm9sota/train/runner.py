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


def make_optimizer(model, loss_fn, cfg: dict, jepa_loss=None):
    opt_cfg = cfg["optimizer"]
    lr = float(opt_cfg.get("lr", 3e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-6))

    params = [{"params": model.parameters(), "lr": lr, "weight_decay": weight_decay}]
    if loss_fn is not None:
        loss_params = list(loss_fn.parameters())
        if loss_params:
            params.append({"params": loss_params, "lr": lr * 0.1, "weight_decay": 0.0})
    if jepa_loss is not None:
        jepa_params = list(jepa_loss.parameters())
        if jepa_params:
            params.append({"params": jepa_params, "lr": lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(params)



def get_target_index(cfg: dict) -> int | None:
    target_cfg = cfg.get("target", {})
    mode = target_cfg.get("mode", "all")

    if mode == "all":
        return None

    if mode == "single":
        if "index" not in target_cfg:
            raise ValueError("target.mode='single' requires target.index")
        return int(target_cfg["index"])

    raise ValueError(f"Unknown target mode: {mode}")


def select_target_if_needed(tensor, target_index: int | None):
    if target_index is None:
        return tensor
    return tensor[:, target_index:target_index + 1]


def primary_mae_for_target(norm_mae, target_index: int | None) -> float:
    if target_index is None:
        return float(norm_mae.mean())
    return float(norm_mae[target_index])



def make_scheduler(optimizer, cfg: dict):
    sched_cfg = cfg.get("scheduler", {})
    name = sched_cfg.get("name", "none")

    if name in {"none", None}:
        return None

    if name != "warmup_cosine":
        raise ValueError(f"Unknown scheduler: {name}")

    epochs = int(cfg["train"].get("epochs", 5))
    warmup_epochs = float(sched_cfg.get("warmup_epochs", 5.0))
    min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.03))

    def lr_lambda(epoch_idx: int):
        # epoch_idx is 0-based for the upcoming epoch.
        e = float(epoch_idx)

        if warmup_epochs > 0 and e < warmup_epochs:
            return max(min_lr_ratio, (e + 1.0) / warmup_epochs)

        denom = max(1.0, float(epochs) - warmup_epochs)
        t = min(1.0, max(0.0, (e - warmup_epochs) / denom))

        # cosine from 1.0 to min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    target_mean,
    target_std,
    global_step: int,
    droplet_loss=None,
    grad_clip: float = 5.0,
    jepa_ctx: dict | None = None,
    target_index: int | None = None,
):
    from qm9sota.losses.jepa import apply_atom_mask, sample_atom_mask

    model.train()
    total_loss = 0.0
    total_graphs = 0
    stat_sums = defaultdict(float)
    stat_count = 0

    steps_per_epoch = len(loader)

    for step_in_epoch, batch in enumerate(tqdm(loader, leave=False)):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred_norm = model(batch)
        y_norm = normalize_y(batch.y, target_mean, target_std)

        pred_for_loss = select_target_if_needed(pred_norm, target_index)
        y_for_loss = select_target_if_needed(y_norm, target_index)

        if droplet_loss is None:
            sup_loss = F.smooth_l1_loss(pred_for_loss, y_for_loss)
            stats = {}
        else:
            if target_index is not None:
                # Keep first single-target pass simple: supervised only.
                sup_loss = F.smooth_l1_loss(pred_for_loss, y_for_loss)
                stats = {}
            else:
                sup_loss, stats = droplet_loss(pred=pred_norm, target=y_norm, step=global_step)
        stats = dict(stats)

        tau = None
        if jepa_ctx is not None:
            jepa_loss_mod = jepa_ctx["loss"]
            ema_target = jepa_ctx["ema"]
            jcfg = jepa_loss_mod.cfg
            total_epochs = jepa_ctx["total_epochs"]
            epoch = jepa_ctx["epoch"]

            epoch_float = (epoch - 1) + (step_in_epoch + 1) / max(steps_per_epoch, 1)
            lam = jepa_loss_mod.lambda_at(epoch_float)
            tau = jepa_loss_mod.tau_at(epoch_float, total_epochs)

            mask = sample_atom_mask(
                batch.batch,
                ratio_low=jcfg.mask_ratio_low,
                ratio_high=jcfg.mask_ratio_high,
            )
            masked_batch = apply_atom_mask(batch, mask)
            online_out = model(masked_batch, return_embeddings=True)
            h_online = online_out["node_embeddings"]
            h_target = ema_target.encode_nodes(batch)
            j_loss, j_stats = jepa_loss_mod(
                context_node_h=h_online,
                target_node_h=h_target,
                mask=mask,
            )

            total_step_loss = sup_loss + lam * j_loss
            stats.update(j_stats)
            stats["sup_loss"] = sup_loss.detach()
            stats["jepa_lambda"] = torch.tensor(lam, device=device)
            stats["jepa_tau"] = torch.tensor(tau, device=device)
        else:
            total_step_loss = sup_loss

        total_step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if jepa_ctx is not None and tau is not None:
            jepa_ctx["ema"].update(model, tau=tau)

        batch_size = batch.y.size(0)
        total_loss += float(total_step_loss.detach()) * batch_size
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

    loss_block = loss_cfg.get("loss", {})
    loss_name = loss_block.get("name", "baseline")

    droplet_loss = None
    jepa_loss_mod = None
    ema_target = None

    def _build_supervised(sup_block: dict):
        nonlocal droplet_loss
        sup_name = sup_block.get("name", "baseline")
        if sup_name == "droplet":
            from qm9sota.losses.droplet import build_droplet_loss
            droplet_loss = build_droplet_loss(
                {"loss": sup_block},
                steps_per_epoch=len(bundle.train_loader),
                device=device,
            )
        elif sup_name != "baseline":
            raise ValueError(f"Unknown supervised loss name: {sup_name}")

    if loss_name == "droplet":
        _build_supervised({**loss_block, "name": "droplet"})
    elif loss_name == "group_droplet":
        from qm9sota.losses.group_droplet import build_group_droplet_loss
        droplet_loss = build_group_droplet_loss(
            {"loss": loss_block},
            steps_per_epoch=len(bundle.train_loader),
            device=device,
        )
    elif loss_name == "baseline":
        pass
    elif loss_name == "jepa_aux":
        sup_block = dict(loss_block.get("supervised", {"name": "baseline"}))
        _build_supervised(sup_block)
        from qm9sota.losses.jepa import EMATargetEncoder, build_jepa_loss
        hidden_dim = int(cfg["model"].get("hidden_dim", 128))
        jepa_loss_mod = build_jepa_loss(
            dict(loss_block.get("jepa", {})),
            hidden_dim=hidden_dim,
            device=device,
        )
        ema_target = EMATargetEncoder(model).to(device)
    elif loss_name == "jepa":
        from qm9sota.losses.jepa import EMATargetEncoder, build_jepa_loss
        hidden_dim = int(cfg["model"].get("hidden_dim", 128))
        jepa_loss_mod = build_jepa_loss(
            dict(loss_block.get("jepa", {})),
            hidden_dim=hidden_dim,
            device=device,
        )
        ema_target = EMATargetEncoder(model).to(device)
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")

    optimizer = make_optimizer(model, droplet_loss, cfg, jepa_loss=jepa_loss_mod)
    scheduler = make_scheduler(optimizer, cfg)
    grad_clip = float(cfg["optimizer"].get("grad_clip", 5.0))
    epochs = int(cfg["train"].get("epochs", 5))
    target_index = get_target_index(cfg)

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
        jepa_ctx = None
        if jepa_loss_mod is not None:
            jepa_ctx = {
                "loss": jepa_loss_mod,
                "ema": ema_target,
                "epoch": epoch,
                "total_epochs": epochs,
            }

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
            target_index=target_index,
            jepa_ctx=jepa_ctx,
        )

        raw_mae, norm_mae = evaluate_both(model, bundle.val_loader, device, target_mean, target_std)
        mean_raw = float(raw_mae.mean())
        mean_norm = primary_mae_for_target(norm_mae, target_index)

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
        "target_mode": cfg.get("target", {}).get("mode", "all"),
        "target_name": cfg.get("target", {}).get("name", None),
        "target_index": target_index,
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
