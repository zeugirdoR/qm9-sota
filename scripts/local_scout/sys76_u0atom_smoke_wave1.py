from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. Run: python -m pip install pyyaml") from e


REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "train" / "sys76_smoke"
EXPERIMENT_DOC = REPO / "experiments" / "SYS76_U0atom_smoke_wave1_2026-05-16.md"

RESULTS_ROOT = Path(os.environ.get("QM9_LOCAL_RESULTS", str(Path.home() / "qm9-sota-local-results")))
DATA_ROOT = Path(os.environ.get("QM9_DATA_ROOT", str(Path.home() / "data" / "QM9")))

SEED = 43
EPOCHS = 80

BASE_MODEL = {
    "name": "pga_multivector_transformer",
    "node_in_dim": 11,
    "hidden_dim": 128,
    "out_dim": 19,
    "num_layers": 4,
    "dropout": 0.0,
    "attention_mode": "edge",
    "edge_feature_mode": "radial",
    "num_rbf": 32,
    "cutoff": 8.0,
    "vector_channels": 8,
    "head_mode": "single",
}

VARIANTS = {
    "BASE_M4": {},
    "M4_dropout_0p03": {
        "dropout": 0.03,
    },
    "M4_global_read": {
        "use_global_token": True,
    },
    "M4_global_feedback_s0p5": {
        "global_feedback": True,
        "global_feedback_layers": 1,
        "global_feedback_scale": 0.5,
    },
    "M4_cutoff10": {
        "cutoff": 10.0,
    },
    "M4_vch12": {
        "vector_channels": 12,
    },
    "M4_family_head": {
        "head_mode": "family",
    },
}


def config_for(name: str, patch: dict) -> dict:
    model = dict(BASE_MODEL)
    model.update(patch)

    return {
        "seed": SEED,
        "device": "cuda",
        "paths": {
            "results_dir": str(RESULTS_ROOT),
        },
        "data": {
            "root": str(DATA_ROOT),
            "train_size": 110000,
            "val_size": 10000,
            "smoke": True,
            "smoke_train_size": 20000,
            "smoke_val_size": 2000,
            "batch_size": 64,
            "num_workers": 2,
        },
        "target": {
            "mode": "single",
            "name": "U0_atom",
            "index": 12,
        },
        "model": model,
        "optimizer": {
            "name": "adamw",
            "lr": 0.0002,
            "weight_decay": 0.000001,
            "grad_clip": 5.0,
        },
        "train": {
            "epochs": EPOCHS,
            "checkpoint_every": 20,
        },
        "metrics": {
            "primary": "U0_atom_normalized_mae",
            "save_per_target": True,
        },
    }


def run(cmd: list[str], *, env: dict | None = None) -> subprocess.CompletedProcess:
    print()
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO), env=env)


def write_configs() -> list[tuple[str, Path, str]]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    written = []

    for name, patch in VARIANTS.items():
        cfg = config_for(name, patch)
        cfg_path = CONFIG_DIR / f"U0_atom_{name}_smoke80_seed43.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        run_name = f"SYS76_U0atom_{name}_smoke80_seed43"
        written.append((name, cfg_path, run_name))

    return written


def write_initial_doc(written: list[tuple[str, Path, str]]) -> None:
    lines = []
    add = lines.append

    add("# System76 U0_atom smoke wave 1 — 2026-05-16")
    add("")
    add("## Purpose")
    add("")
    add("Scout candidate changes locally on System76 before promoting anything to Colab/A100.")
    add("")
    add("## Claim status")
    add("")
    add("No SOTA claim. No production claim. These are local smoke/proxy tests only.")
    add("")
    add("## Target")
    add("")
    add("```text")
    add("target: U0_atom")
    add("target_index: 12")
    add("metric: best validation target MAE")
    add("units: meV")
    add("raw PyG units: eV")
    add("conversion: eV × 1000")
    add("```")
    add("")
    add("## Local smoke protocol")
    add("")
    add("```text")
    add(f"seed: {SEED}")
    add(f"epochs: {EPOCHS}")
    add("smoke_train_size: 20000")
    add("smoke_val_size: 2000")
    add("batch_size: 64")
    add(f"results_dir: {RESULTS_ROOT}")
    add(f"data_root: {DATA_ROOT}")
    add("```")
    add("")
    add("## Candidate configs")
    add("")
    for name, cfg_path, run_name in written:
        add(f"- `{name}`")
        add(f"  - config: `{cfg_path.relative_to(REPO)}`")
        add(f"  - run_name: `{run_name}`")
    add("")
    add("## Promotion rule")
    add("")
    add("A candidate must beat the local BASE_M4 smoke validation MAE clearly before it deserves A100 confirmation.")
    add("Known non-promoted branches such as CB bias, motor residual, M4H, M4L, and M4S are not included in this first wave.")
    add("")
    add("## Results")
    add("")
    add("Pending.")

    EXPERIMENT_DOC.write_text("\n".join(lines) + "\n")
    print("Wrote doc:", EXPERIMENT_DOC)


def parse_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {
            "exists": False,
            "run_dir": str(run_dir),
        }

    data = json.loads(summary_path.read_text())
    return {
        "exists": True,
        "run_dir": str(run_dir),
        "best_epoch": data.get("best_epoch"),
        "best_val_target_converted_mae": data.get("best_val_target_converted_mae"),
        "best_val_target_unit": data.get("best_val_target_unit"),
        "best_val_target_raw_mae": data.get("best_val_target_raw_mae"),
        "git_commit": data.get("git_commit"),
        "runtime": data.get("runtime"),
    }


def summarize(written: list[tuple[str, Path, str]]) -> dict:
    rows = []

    for name, cfg_path, run_name in written:
        run_dir = RESULTS_ROOT / run_name
        row = {
            "variant": name,
            "config": str(cfg_path),
            "run_name": run_name,
        }
        row.update(parse_summary(run_dir))
        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: float("inf") if r.get("best_val_target_converted_mae") is None else r["best_val_target_converted_mae"],
    )

    out = {
        "target": "U0_atom",
        "target_index": 12,
        "metric": "best validation target MAE",
        "unit": "meV",
        "claim_status": "local smoke/proxy only; not production; not SOTA",
        "seed": SEED,
        "epochs": EPOCHS,
        "results_root": str(RESULTS_ROOT),
        "rows": rows_sorted,
    }

    out_path = RESULTS_ROOT / "SYS76_U0atom_smoke_wave1_summary.json"
    out_path.write_text(json.dumps(out, indent=2))

    lines = EXPERIMENT_DOC.read_text().splitlines()
    marker = "## Results"
    if marker in lines:
        idx = lines.index(marker)
        lines = lines[: idx + 1]
    else:
        lines.append("")
        lines.append(marker)

    lines.append("")
    lines.append("| Rank | Variant | Best val MAE | Best epoch | Run dir |")
    lines.append("|---:|---|---:|---:|---|")

    for i, row in enumerate(rows_sorted, start=1):
        mae = row.get("best_val_target_converted_mae")
        mae_text = "missing" if mae is None else f"{mae:.6f} meV"
        epoch = row.get("best_epoch")
        epoch_text = "missing" if epoch is None else str(epoch)
        lines.append(f"| {i} | `{row['variant']}` | {mae_text} | {epoch_text} | `{row['run_dir']}` |")

    lines.append("")
    lines.append("Raw JSON summary:")
    lines.append("")
    lines.append(f"```text")
    lines.append(str(out_path))
    lines.append("```")
    lines.append("")

    EXPERIMENT_DOC.write_text("\n".join(lines) + "\n")

    print()
    print("SUMMARY")
    print(json.dumps(out, indent=2)[:6000])
    print()
    print("Wrote:", out_path)
    print("Updated doc:", EXPERIMENT_DOC)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-min", type=int, default=240)
    args = parser.parse_args()

    written = write_configs()
    write_initial_doc(written)

    if args.make_only:
        print("Configs/docs created. Not running.")
        return

    if args.run:
        env = os.environ.copy()
        env["QM9_FORCE_EXIT"] = "1"

        for name, cfg_path, run_name in written:
            run_dir = RESULTS_ROOT / run_name
            if (run_dir / "summary.json").exists() and not args.force:
                print("Skipping existing completed run:", run_name)
                continue

            seconds = str(args.timeout_min * 60)
            cmd = [
                "timeout",
                "--kill-after=60",
                seconds,
                sys.executable,
                str(REPO / "scripts" / "train.py"),
                "--config",
                str(cfg_path),
                "--loss",
                str(REPO / "configs" / "loss" / "baseline.yaml"),
                "--run-name",
                run_name,
                "--seed",
                str(SEED),
            ]

            rc = run(cmd, env=env).returncode
            print("returncode:", rc)

        summarize(written)
        return

    if args.summarize:
        summarize(written)
        return

    print("Nothing to do. Use --make-only, --run, or --summarize.")


if __name__ == "__main__":
    main()
