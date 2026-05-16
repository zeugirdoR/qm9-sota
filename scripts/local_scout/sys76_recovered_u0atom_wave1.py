from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

import yaml

REPO = Path.home() / "qm9-sota"
RESULTS = Path(os.environ.get("QM9_LOCAL_RESULTS", str(Path.home() / "qm9-sota-local-results")))
BASE_CONFIG = REPO / "configs/train/sys76_proxy_recovered/U0_atom_LOCAL_M4_recovered_100epoch_seed43.yaml"
CONFIG_DIR = REPO / "configs/train/sys76_proxy_recovered_wave1"
DOC = REPO / "experiments/SYS76_RECOVERED_U0atom_wave1_2026-05-16.md"

SEED = 43

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

BASELINE_RUN_NAME = "SYS76_RECOVERED_M4_U0atom_100epoch_seed43"

def load_base():
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(BASE_CONFIG)
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg.setdefault("paths", {})
    cfg["paths"]["results_dir"] = str(RESULTS)
    cfg["seed"] = SEED
    return cfg

def run_name_for(variant):
    if variant == "BASE_M4":
        return BASELINE_RUN_NAME
    return f"SYS76_RECOVERED_{variant}_U0atom_100epoch_seed43"

def write_configs():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    base = load_base()
    written = []

    for variant, patch in VARIANTS.items():
        cfg = json.loads(json.dumps(base))
        cfg.setdefault("model", {})
        cfg["model"].update(patch)
        cfg["seed"] = SEED

        out = CONFIG_DIR / f"U0_atom_{variant}_recovered_100epoch_seed43.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        written.append((variant, out, run_name_for(variant)))

    lines = []
    add = lines.append
    add("# System76 recovered U0_atom candidate wave 1 — 2026-05-16")
    add("")
    add("## Claim status")
    add("")
    add("Local proxy/scout only. Not production. Not SOTA.")
    add("")
    add("## Baseline")
    add("")
    add("```text")
    add("run: SYS76_RECOVERED_M4_U0atom_100epoch_seed43")
    add("best val U0_atom: 258.4153413772583 meV")
    add("```")
    add("")
    add("## Candidate configs")
    add("")
    for variant, cfg_path, run_name in written:
        add(f"- `{variant}`")
        add(f"  - config: `{cfg_path.relative_to(REPO)}`")
        add(f"  - run_name: `{run_name}`")
    add("")
    add("## Promotion rule")
    add("")
    add("Candidate must beat the recovered BASE_M4 under the same local protocol before A100 confirmation.")
    add("")
    add("## Results")
    add("")
    add("Pending.")
    DOC.write_text("\n".join(lines) + "\n")

    print("Wrote configs:")
    for item in written:
        print(item)
    print("Wrote doc:", DOC)
    return written

def parse_summary(run_dir):
    p = run_dir / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def run_training(written, force=False, timeout_min=360):
    env = os.environ.copy()
    env["QM9_FORCE_EXIT"] = "1"
    env["QM9_LOCAL_RESULTS"] = str(RESULTS)

    for variant, cfg_path, run_name in written:
        run_dir = RESULTS / run_name
        if (run_dir / "summary.json").exists() and not force:
            print("Skipping existing:", run_name)
            continue

        cmd = [
            "timeout",
            "--kill-after=60",
            f"{timeout_min}m",
            sys.executable,
            str(REPO / "scripts/train.py"),
            "--config",
            str(cfg_path),
            "--loss",
            str(REPO / "configs/loss/baseline.yaml"),
            "--run-name",
            run_name,
            "--seed",
            str(SEED),
        ]

        print()
        print("$", " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
        print("returncode:", rc)

def summarize(written):
    rows = []

    for variant, cfg_path, run_name in written:
        run_dir = RESULTS / run_name
        s = parse_summary(run_dir)
        row = {
            "variant": variant,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "config": str(cfg_path),
        }
        if s:
            row.update({
                "best_epoch": s.get("best_epoch"),
                "best_val_target_converted_mae": s.get("best_val_target_converted_mae"),
                "best_val_target_unit": s.get("best_val_target_unit"),
                "git_commit": s.get("git_commit"),
                "gpu": (s.get("runtime") or {}).get("gpu_name"),
            })
        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: float("inf") if r.get("best_val_target_converted_mae") is None else r["best_val_target_converted_mae"],
    )

    summary = {
        "target": "U0_atom",
        "target_index": 12,
        "metric": "best validation target MAE",
        "unit": "meV",
        "claim_status": "local proxy only; not production; not SOTA",
        "baseline_run": BASELINE_RUN_NAME,
        "baseline_val_mev": 258.4153413772583,
        "rows": rows_sorted,
    }

    out = RESULTS / "SYS76_RECOVERED_U0atom_wave1_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    lines = []
    add = lines.append
    add("# System76 recovered U0_atom candidate wave 1 — 2026-05-16")
    add("")
    add("## Claim status")
    add("")
    add("Local proxy/scout only. Not production. Not SOTA.")
    add("")
    add("## Baseline")
    add("")
    add("```text")
    add("run: SYS76_RECOVERED_M4_U0atom_100epoch_seed43")
    add("best val U0_atom: 258.4153413772583 meV")
    add("```")
    add("")
    add("## Results")
    add("")
    add("| Rank | Variant | Best val MAE | Best epoch | Commit | Run |")
    add("|---:|---|---:|---:|---|---|")

    for i, r in enumerate(rows_sorted, start=1):
        mae = r.get("best_val_target_converted_mae")
        mae_txt = "missing" if mae is None else f"{mae:.6f} meV"
        epoch = r.get("best_epoch")
        epoch_txt = "missing" if epoch is None else str(epoch)
        commit = r.get("git_commit") or "missing"
        add(f"| {i} | `{r['variant']}` | {mae_txt} | {epoch_txt} | `{commit}` | `{r['run_name']}` |")

    add("")
    add("## Decision rule")
    add("")
    add("Promote only candidates that beat the recovered BASE_M4 clearly under this same protocol.")
    add("")
    add("Raw JSON summary:")
    add("")
    add("```text")
    add(str(out))
    add("```")

    DOC.write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    print("Wrote:", out)
    print("Updated:", DOC)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-min", type=int, default=360)
    args = parser.parse_args()

    written = write_configs()

    if args.run:
        run_training(written, force=args.force, timeout_min=args.timeout_min)
        summarize(written)
    elif args.summarize:
        summarize(written)
    elif args.make:
        pass
    else:
        print("Use --make, --run, or --summarize")

if __name__ == "__main__":
    main()
