#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from qm9sota.data.qm9 import load_qm9_bundle
from qm9sota.models.tiny_radial_mpnn import build_model
from qm9sota.train.runner import run_training
from qm9sota.utils.config import load_yaml
from qm9sota.utils.device import get_device, describe_device
from qm9sota.utils.seed import seed_everything


REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit_or_unknown(repo_root: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def resolve_path(path_like: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return base / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QM9 smoke experiments.")
    parser.add_argument("--config", required=True, help="Path to train config YAML")
    parser.add_argument("--loss", required=True, help="Path to loss config YAML")
    parser.add_argument("--run-name", required=True, help="Name for result directory")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    loss_path = resolve_path(args.loss)

    cfg = load_yaml(config_path)
    loss_cfg = load_yaml(loss_path)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    device = get_device(cfg.get("device", "cuda"))

    runtime = describe_device(device)
    git_commit = git_commit_or_unknown()

    print("Runtime:", json.dumps(runtime, indent=2))
    print("repo_root:", str(REPO_ROOT))
    print("git_commit:", git_commit)

    bundle = load_qm9_bundle(cfg, seed=seed)

    print("dataset_size:", len(bundle.dataset))
    print("train_batches:", len(bundle.train_loader))
    print("val_batches:", len(bundle.val_loader))
    print("split_sizes:", len(bundle.train_idx), len(bundle.val_idx), len(bundle.test_idx))

    # Seed again immediately before model construction so baseline and
    # droplet runs start from identical weights when using the same config/seed.
    seed_everything(seed + 123)
    model = build_model(cfg)

    output_root = resolve_path(cfg.get("paths", {}).get("results_dir", "results"))

    run_training(
        cfg=cfg,
        loss_cfg=loss_cfg,
        run_name=args.run_name,
        model=model,
        bundle=bundle,
        device=device,
        output_root=output_root,
        config_paths={
            "train_config": str(config_path),
            "loss_config": str(loss_path),
        },
        extra_metadata={
            "repo_root": str(REPO_ROOT),
            "git_commit": git_commit,
            "runtime": runtime,
        },
        resume_path=Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()

    # Some local Linux/PyG/CUDA environments can hang during Python shutdown
    # after all results have already been written. Enable this for local smoke
    # tests with: QM9_FORCE_EXIT=1 python scripts/train.py ...
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()

    if os.environ.get("QM9_FORCE_EXIT", "0") == "1":
        os._exit(0)
