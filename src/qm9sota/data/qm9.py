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

    train_dataset = dataset.index_select(train_use.tolist())
    val_dataset = dataset.index_select(val_use.tolist())

    ys = torch.cat([data.y for data in dataset], dim=0)
    target_mean = ys[train_idx].mean(dim=0)
    target_std = ys[train_idx].std(dim=0).clamp_min(1e-8)

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
