from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import importlib.util

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


TARGET_NAMES = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv",
    "U0_atom", "U_atom", "H_atom", "G_atom", "A", "B", "C",
]


@dataclass
class QM9Bundle:
    dataset: QM9
    train_loader: DataLoader
    val_loader: DataLoader
    target_mean: torch.Tensor
    target_std: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor


def load_qm9_bundle(cfg: dict, seed: int) -> QM9Bundle:
    data_cfg = cfg["data"]
    # QM9_ROOT env override lets the same configs run on Colab (/content/...) and DeltaAI (/work/...)
    root = Path(os.environ.get("QM9_ROOT", data_cfg.get("root", "/content/data/QM9")))

    if importlib.util.find_spec("rdkit") is not None:
        print(
            "WARNING: RDKit is importable. The initial benchmark contract uses the PyG "
            "preprocessed QM9 path with no RDKit. If QM9 processing fails, restart the runtime "
            "or uninstall RDKit."
        )

    dataset = QM9(root=str(root))
    n = len(dataset)

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)

    train_size = int(data_cfg.get("train_size", 110_000))
    val_size = int(data_cfg.get("val_size", 10_000))

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    smoke = bool(data_cfg.get("smoke", True))
    if smoke:
        train_use = train_idx[: int(data_cfg.get("smoke_train_size", 20_000))]
        val_use = val_idx[: int(data_cfg.get("smoke_val_size", 2_000))]
    else:
        train_use = train_idx
        val_use = val_idx

    ys = torch.cat([data.y for data in dataset], dim=0)
    target_mean = ys[train_idx].mean(dim=0)
    target_std = ys[train_idx].std(dim=0).clamp_min(1e-8)

    val_dataset = dataset.index_select(val_use.tolist())

    # Optional label noise on a FIXED fraction of TRAIN molecules (clean val untouched). This is the
    # robustness regime where a bounded-influence/droplet loss should beat the baseline: corrupted
    # targets are large (scale*sigma) outliers the droplet amputates. Env-overridable for sweeps.
    noise_cfg = data_cfg.get("label_noise") or {}
    noise_frac = float(os.environ.get("QM9_NOISE_FRAC", noise_cfg.get("frac", 0.0)))
    if noise_frac > 0.0:
        scale = float(os.environ.get("QM9_NOISE_SCALE", noise_cfg.get("scale", 5.0)))
        nseed = int(os.environ.get("QM9_NOISE_SEED", noise_cfg.get("seed", 0)))
        ng = torch.Generator().manual_seed(nseed)
        train_list = [dataset[i] for i in train_use.tolist()]
        k = int(round(noise_frac * len(train_list)))
        corrupt_pos = torch.randperm(len(train_list), generator=ng)[:k].tolist()
        std = target_std.view(-1)
        for p in corrupt_pos:
            d = train_list[p]
            d.y = d.y + (scale * std * torch.randn(std.shape, generator=ng)).view_as(d.y)
        train_dataset = train_list
        print(f"label noise: corrupted {k}/{len(train_list)} train targets at {scale} sigma "
              f"(frac={noise_frac}, seed={nseed})")
    else:
        train_dataset = dataset.index_select(train_use.tolist())

    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 0))

    loader_gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=loader_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return QM9Bundle(
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        target_mean=target_mean,
        target_std=target_std,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
