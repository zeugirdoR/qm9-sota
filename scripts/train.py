#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import torch

from qm9sota.data.qm9 import load_qm9_bundle
from qm9sota.models.tiny_radial_mpnn import build_model
from qm9sota.train.runner import run_training
from qm9sota.utils.config import load_yaml
from qm9sota.utils.device import get_device, describe_device
from qm9sota.utils.seed import seed_everything


def git_commit_or_unknown() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QM9 smoke experiments.")
    parser.add_argument("--config", required=True, help="Path to train config YAML")
    parser.add_argument("--loss", required=True, help="Path to loss config YAML")
    parser.add_argument("--run-name", required=True, help="Name for result directory")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    loss_cfg = load_yaml(args.loss)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    device = get_device(cfg.get("device", "cuda"))
    print("Runtime:", json.dumps(describe_device(device), indent=2))
    print("git_commit:", git_commit_or_unknown())

    bundle = load_qm9_bundle(cfg, seed=seed)
    print("dataset_size:", len(bundle.dataset))
    print("train_batches:", len(bundle.train_loader))
    print("val_batches:", len(bundle.val_loader))
    print("split_sizes:", len(bundle.train_idx), len(bundle.val_idx), len(bundle.test_idx))

    # Important: seed again immediately before model construction so baseline and droplet runs
    # start from identical weights when using the same config/seed.
    seed_everything(seed + 123)
    model = build_model(cfg)

    output_root = Path(cfg.get("paths", {}).get("results_dir", "results"))
    run_training(
        cfg=cfg,
        loss_cfg=loss_cfg,
        run_name=args.run_name,
        model=model,
        bundle=bundle,
        device=device,
        output_root=output_root,
        config_paths={"train_config": args.config, "loss_config": args.loss},
    )


if __name__ == "__main__":
    main()
