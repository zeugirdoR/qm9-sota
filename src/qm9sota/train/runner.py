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




TARGET_UNIT_INFO = {
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
    "A": {"index": 16, "unit": "GHz", "conversion": 1.0},
    "B": {"index": 17, "unit": "GHz", "conversion": 1.0},
    "C": {"index": 18, "unit": "GHz", "conversion": 1.0},
}


def target_unit_info_from_cfg(cfg: dict):
    target_cfg = cfg.get("target", {})
    name = target_cfg.get("name", None)

    if name in TARGET_UNIT_INFO:
        return TARGET_UNIT_INFO[name]

    idx = target_cfg.get("index", None)
    if idx is not None:
        idx = int(idx)
        for info in TARGET_UNIT_INFO.values():
            if int(info["index"]) == idx:
                return info

    return None


def target_summary_metrics(raw_mae, norm_mae, cfg: dict, target_index):
    """
    Returns target-specific metrics for single-target runs.

    raw_mae and norm_mae are vectors of length 19.
    """
    if not isinstance(target_index, int):
        return {}

    info = target_unit_info_from_cfg(cfg)
    if info is None:
        conversion = 1.0
        unit = None
    else:
        conversion = float(info["conversion"])
        unit = info["unit"]

    raw = float(raw_mae[target_index])
    norm = float(norm_mae[target_index])
    converted = raw * conversion

    return {
        "best_val_target_norm_mae": norm,
        "best_val_target_raw_mae": raw,
        "best_val_target_unit": unit,
        "best_val_target_conversion": conversion,
        "best_val_target_converted_mae": converted,
    }


def get_target_index(cfg: dict):
    """
    Returns:
      None for all-target mode
      int for single-target mode
      list[int] for multi-target mode
    """
    target_cfg = cfg.get("target", {})
    mode = target_cfg.get("mode", "all")

    if mode == "all":
        return None

    if mode == "single":
        if "index" not in target_cfg:
            raise ValueError("target.mode='single' requires target.index")
        return int(target_cfg["index"])

    if mode == "multi":
        if "indices" not in target_cfg:
            raise ValueError("target.mode='multi' requires target.indices")
        return [int(i) for i in target_cfg["indices"]]

    raise ValueError(f"Unknown target mode: {mode}")


def select_target_if_needed(tensor, target_index):
    if target_index is None:
        return tensor

    if isinstance(target_index, int):
        return tensor[:, target_index:target_index + 1]

    if isinstance(target_index, (list, tuple)):
        idx = torch.tensor(target_index, dtype=torch.long, device=tensor.device)
        return tensor.index_select(dim=1, index=idx)

    raise TypeError(f"Unsupported target_index type: {type(target_index)}")


def primary_mae_for_target(norm_mae, target_index) -> float:
    if target_index is None:
        return float(norm_mae.mean())

    if isinstance(target_index, int):
        return float(norm_mae[target_index])

    if isinstance(target_index, (list, tuple)):
        idx = torch.tensor(target_index, dtype=torch.long, device=norm_mae.device)
        return float(norm_mae.index_select(dim=0, index=idx).mean())

    raise TypeError(f"Unsupported target_index type: {type(target_index)}")



def make_scheduler(optimizer, cfg: dict):
    sched_cfg = cfg.get("scheduler", {})
    name = sched_cfg.get("name", "none")

    if name in {"none", None}:
        return None

    if name != "warmup_cosine":
        raise ValueError(f"Unknown scheduler: {name}")

    epochs = int(cfg["train"].get("epochs", 5))
    checkpoint_every = int(cfg["train"].get("checkpoint_every", 0))
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

        if hasattr(model, "motor_aux_loss"):
            motor_aux_loss = model.motor_aux_loss()
            if motor_aux_loss is not None:
                total_step_loss = total_step_loss + motor_aux_loss

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



def save_training_checkpoint(
    *,
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_epoch,
    best_norm,
    best_raw_mae,
    best_norm_mae,
):
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_norm": None if best_norm is None else float(best_norm),
        "best_raw_mae": best_raw_mae.detach().cpu() if best_raw_mae is not None else None,
        "best_norm_mae": best_norm_mae.detach().cpu() if best_norm_mae is not None else None,
    }
    torch.save(payload, path)


def load_training_checkpoint(
    *,
    path,
    model,
    optimizer,
    scheduler,
    device,
):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "best_epoch": ckpt.get("best_epoch", None),
        "best_norm": ckpt.get("best_norm", None),
        "best_raw_mae": ckpt.get("best_raw_mae", None),
        "best_norm_mae": ckpt.get("best_norm_mae", None),
    }


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
    resume_path=None,
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
    checkpoint_every = int(cfg["train"].get("checkpoint_every", 0))
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
    rows = []
    start_epoch = 1

    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        resume_state = load_training_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        last_epoch = int(resume_state["epoch"])
        global_step = int(resume_state["global_step"])
        best_epoch = resume_state["best_epoch"]
        best_norm = float("inf") if resume_state["best_norm"] is None else float(resume_state["best_norm"])

        best_raw_mae = resume_state["best_raw_mae"]
        if best_raw_mae is not None:
            best_raw_mae = best_raw_mae.to(device)

        best_norm_mae = resume_state["best_norm_mae"]
        if best_norm_mae is not None:
            best_norm_mae = best_norm_mae.to(device)

        start_epoch = last_epoch + 1
        print({
            "resumed_from": str(resume_path),
            "start_epoch": start_epoch,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_norm": best_norm,
        })

    for epoch in range(start_epoch, epochs + 1):
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

        if checkpoint_every > 0 and (epoch % checkpoint_every == 0 or epoch == epochs):
            ckpt_path = run_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            latest_path = run_dir / "latest_checkpoint.pt"

            save_training_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_epoch=best_epoch,
                best_norm=best_norm,
                best_raw_mae=best_raw_mae,
                best_norm_mae=best_norm_mae,
            )

            save_training_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_epoch=best_epoch,
                best_norm=best_norm,
                best_raw_mae=best_raw_mae,
                best_norm_mae=best_norm_mae,
            )

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
        "target_index": target_index if not isinstance(target_index, tuple) else list(target_index),
    }

    summary.update(
        target_summary_metrics(
            raw_mae=best_raw_mae,
            norm_mae=best_norm_mae,
            cfg=cfg,
            target_index=target_index,
        )
    )

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

    # Optional local hard-exit after final summary is printed.
    # Useful for local Linux/PyG/CUDA environments that hang during shutdown.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()

    if os.environ.get("QM9_FORCE_EXIT", "0") == "1":
        os._exit(0)
    return summary
