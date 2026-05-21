from __future__ import annotations

from pathlib import Path
import json
import torch
from torch_geometric.datasets import QM9

from qm9sota.geometry.cga_features import graph_cga_summary


DATA_ROOT = Path.home() / "data" / "QM9"
OUT = Path.home() / "awsgit" / "qm9-sota" / "experiments" / "SYS76_CGA_feature_probe_2026-05-18.md"

TARGET_INDEX = 12
TARGET_NAME = "U0_atom"
MAX_MOLS = 5000


def corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if denom.item() == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def main() -> None:
    dataset = QM9(root=str(DATA_ROOT))
    n = min(MAX_MOLS, len(dataset))

    xs = []
    ys = []

    for idx in range(n):
        data = dataset[idx]
        feat = graph_cga_summary(data.pos, batch=None, k=4).squeeze(0).detach().cpu()
        y = data.y.view(-1)[TARGET_INDEX].detach().cpu()
        xs.append(feat)
        ys.append(y)

        if (idx + 1) % 500 == 0:
            print("processed", idx + 1)

    X = torch.stack(xs)
    y = torch.stack(ys)

    names = [
        "mean_neighbor_dist",
        "std_neighbor_dist",
        "min_neighbor_dist",
        "max_neighbor_dist",
        "logdet_I_plus_Gram",
        "local_volume_proxy",
    ]

    rows = []
    for j, name in enumerate(names):
        rows.append(
            {
                "feature": name,
                "mean": float(X[:, j].mean()),
                "std": float(X[:, j].std(unbiased=True)),
                "min": float(X[:, j].min()),
                "max": float(X[:, j].max()),
                "corr_with_U0_atom_raw": corrcoef(X[:, j], y),
            }
        )

    lines = []
    add = lines.append

    add("# System76 CGA/Gram feature probe — 2026-05-18")
    add("")
    add("## Claim status")
    add("")
    add("Feature sanity probe only. Not a model result. Not production. Not SOTA.")
    add("")
    add("## Purpose")
    add("")
    add("Test whether simple CGA-inspired Gram/volume features are finite and nontrivially related to `U0_atom` before modifying M4.")
    add("")
    add("## Setup")
    add("")
    add("```text")
    add(f"dataset: PyG QM9")
    add(f"data_root: {DATA_ROOT}")
    add(f"target: {TARGET_NAME}")
    add(f"target_index: {TARGET_INDEX}")
    add(f"molecules_scanned: {n}")
    add("CGA point Gram convention: P_i · P_j = -0.5 ||x_i - x_j||^2")
    add("```")
    add("")
    add("## Feature summary")
    add("")
    add("| Feature | Mean | Std | Min | Max | Corr with U0_atom raw |")
    add("|---|---:|---:|---:|---:|---:|")

    for r in rows:
        add(
            f"| `{r['feature']}` | "
            f"{r['mean']:.6g} | {r['std']:.6g} | "
            f"{r['min']:.6g} | {r['max']:.6g} | "
            f"{r['corr_with_U0_atom_raw']:.6g} |"
        )

    add("")
    add("## Decision")
    add("")
    add("If features are finite and non-degenerate, proceed to a gated M4+CGA readout proof of concept on System76 recovered proxy protocol.")
    add("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")

    print(json.dumps(rows, indent=2))
    print("Wrote:", OUT)


if __name__ == "__main__":
    main()
