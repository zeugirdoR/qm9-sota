#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qm9sota.models.tiny_radial_mpnn import build_model
from qm9sota.utils.config import load_yaml

TARGET_INDEX = 12
MEV = 1000.0


def make_splits(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return {
        "train": perm[:110000],
        "val": perm[110000:120000],
        "test": perm[120000:],
    }


def load_model(config_path: Path, run_dir: Path, data_root: str, device: torch.device):
    cfg = load_yaml(config_path)
    cfg.setdefault("data", {})
    cfg["data"]["root"] = data_root
    cfg["data"]["smoke"] = False

    model = build_model(cfg).to(device)

    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    model.eval()
    return model, missing, unexpected


def target_stats(dataset, train_idx):
    y_full = torch.cat([data.y for data in dataset], dim=0).float()
    target_mean = y_full[train_idx].mean(dim=0)
    target_std = y_full[train_idx].std(dim=0).clamp_min(1e-8)
    return target_mean[TARGET_INDEX], target_std[TARGET_INDEX]


def remove_translation(delta: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    out = delta.clone()
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    for g in range(num_graphs):
        mask = batch == g
        if mask.any():
            out[mask] = out[mask] - out[mask].mean(dim=0, keepdim=True)
    return out


def graph_stats_and_harmonic_delta(
    *,
    pos0: torch.Tensor,
    pos1: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    k_bond_ev_per_a2: float,
):
    """Tiny local harmonic-bond emulator.

    E_delta ~= 0.5 * k * sum_unique_bonds (|r_ij'| - |r_ij|)^2

    This is intentionally simple. It is a placeholder for xTB/AIQM1/local emulator.
    """
    device = pos0.device
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0

    delta = pos1 - pos0

    rmsd = torch.zeros(num_graphs, device=device)
    max_disp = torch.zeros(num_graphs, device=device)
    n_atoms = torch.zeros(num_graphs, device=device)

    for g in range(num_graphs):
        mask = batch == g
        d = delta[mask].norm(dim=-1)
        n_atoms[g] = mask.sum()
        rmsd[g] = d.pow(2).mean().clamp_min(1e-12).sqrt()
        max_disp[g] = d.max() if d.numel() else 0.0

    src, dst = edge_index
    # Deduplicate PyG undirected bonds if both directions are present.
    mask = src < dst
    src = src[mask]
    dst = dst[mask]

    edge_graph = batch[src]
    r0 = (pos0[src] - pos0[dst]).norm(dim=-1).clamp_min(1e-8)
    r1 = (pos1[src] - pos1[dst]).norm(dim=-1).clamp_min(1e-8)
    stretch = r1 - r0

    e_edge_ev = 0.5 * float(k_bond_ev_per_a2) * stretch.pow(2)

    delta_e_ev = torch.zeros(num_graphs, device=device)
    delta_e_ev.index_add_(0, edge_graph, e_edge_ev)

    max_abs_stretch = torch.zeros(num_graphs, device=device)
    for g in range(num_graphs):
        sm = edge_graph == g
        if sm.any():
            max_abs_stretch[g] = stretch[sm].abs().max()

    return {
        "rmsd_a": rmsd,
        "max_disp_a": max_disp,
        "max_abs_bond_stretch_a": max_abs_stretch,
        "teacher_delta_mev": delta_e_ev * MEV,
        "n_atoms": n_atoms,
    }


def robust_z(x: torch.Tensor) -> torch.Tensor:
    med = x.median()
    mad = (x - med).abs().median().clamp_min(1e-12)
    return (x - med).abs() / (1.4826 * mad)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", default=str(Path.home() / "data/QM9"))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--n-graphs", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--k-droplets", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.01, help="Gaussian coordinate noise in Angstrom")
    ap.add_argument("--k-bond", type=float, default=1.0, help="harmonic bond stiffness in eV/A^2")
    ap.add_argument("--keep-quantile", type=float, default=0.75)
    ap.add_argument("--max-bond-stretch", type=float, default=0.05)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results_local/TINY_DROPLET_HEATBATH_POC_U0atom/summary.json")
    args = ap.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    dataset = QM9(root=args.data_root)
    splits = make_splits(len(dataset), args.seed)
    idx = splits[args.split][: args.n_graphs]

    y_mu, y_sd = target_stats(dataset, splits["train"])

    model, missing, unexpected = load_model(
        Path(args.config),
        Path(args.run_dir),
        args.data_root,
        device,
    )

    subset = [dataset[int(i)] for i in idx]
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 12345)

    records = []

    for batch_id, batch in enumerate(loader):
        batch = batch.to(device)

        pred0_norm = model(batch)[:, TARGET_INDEX]
        pos0 = batch.pos.float()

        for k in range(args.k_droplets):
            delta = torch.randn(pos0.shape, device=device, generator=gen) * float(args.sigma)
            delta = remove_translation(delta, batch.batch)
            pos1 = pos0 + delta

            batch_pert = batch.clone()
            batch_pert.pos = pos1

            pred1_norm = model(batch_pert)[:, TARGET_INDEX]
            model_delta_mev = (pred1_norm - pred0_norm) * y_sd.to(device) * MEV

            st = graph_stats_and_harmonic_delta(
                pos0=pos0,
                pos1=pos1,
                edge_index=batch.edge_index,
                batch=batch.batch,
                k_bond_ev_per_a2=args.k_bond,
            )

            num_graphs = int(batch.batch.max().item()) + 1 if batch.batch.numel() else 0
            for g in range(num_graphs):
                records.append({
                    "batch_id": batch_id,
                    "droplet_id": k,
                    "local_graph_id": g,
                    "rmsd_a": float(st["rmsd_a"][g].cpu()),
                    "max_disp_a": float(st["max_disp_a"][g].cpu()),
                    "max_abs_bond_stretch_a": float(st["max_abs_bond_stretch_a"][g].cpu()),
                    "teacher_delta_mev": float(st["teacher_delta_mev"][g].cpu()),
                    "model_delta_mev": float(model_delta_mev[g].cpu()),
                    "n_atoms": int(st["n_atoms"][g].cpu()),
                })

    # Robust droplet amputation.
    rmsd = torch.tensor([r["rmsd_a"] for r in records])
    max_disp = torch.tensor([r["max_disp_a"] for r in records])
    stretch = torch.tensor([r["max_abs_bond_stretch_a"] for r in records])
    teacher_abs = torch.tensor([abs(r["teacher_delta_mev"]) for r in records])

    score = robust_z(rmsd) + robust_z(max_disp) + robust_z(stretch) + robust_z(teacher_abs)
    cutoff = torch.quantile(score, float(args.keep_quantile))

    keep = (score <= cutoff) & (stretch <= float(args.max_bond_stretch))

    for i, r in enumerate(records):
        r["droplet_score"] = float(score[i])
        r["keep"] = bool(keep[i])

    def summarize(mask: torch.Tensor):
        inds = mask.nonzero(as_tuple=False).view(-1)
        if inds.numel() == 0:
            return {}

        teacher = torch.tensor([records[int(i)]["teacher_delta_mev"] for i in inds])
        model = torch.tensor([records[int(i)]["model_delta_mev"] for i in inds])
        rmsd_i = torch.tensor([records[int(i)]["rmsd_a"] for i in inds])
        stretch_i = torch.tensor([records[int(i)]["max_abs_bond_stretch_a"] for i in inds])

        surface_abs_err = (model - teacher).abs()

        return {
            "n": int(inds.numel()),
            "teacher_delta_abs_mean_mev": float(teacher.abs().mean()),
            "teacher_delta_abs_p95_mev": float(torch.quantile(teacher.abs(), 0.95)),
            "model_delta_abs_mean_mev": float(model.abs().mean()),
            "model_delta_abs_p95_mev": float(torch.quantile(model.abs(), 0.95)),
            "surface_abs_error_mean_mev": float(surface_abs_err.mean()),
            "surface_abs_error_p95_mev": float(torch.quantile(surface_abs_err, 0.95)),
            "rmsd_mean_a": float(rmsd_i.mean()),
            "rmsd_p95_a": float(torch.quantile(rmsd_i, 0.95)),
            "max_bond_stretch_p95_a": float(torch.quantile(stretch_i, 0.95)),
            "false_same_label_conflict_mean_mev": float(teacher.abs().mean()),
        }

    out = {
        "method": "tiny droplet heat-bath proof of concept",
        "note": "Harmonic bond emulator is a placeholder for xTB/AIQM1/local emulator.",
        "config": args.config,
        "run_dir": args.run_dir,
        "split": args.split,
        "n_graphs": args.n_graphs,
        "k_droplets": args.k_droplets,
        "sigma_a": args.sigma,
        "k_bond_ev_per_a2": args.k_bond,
        "keep_quantile": args.keep_quantile,
        "max_bond_stretch_a": args.max_bond_stretch,
        "target_index": TARGET_INDEX,
        "device": str(device),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "all_droplets": summarize(torch.ones(len(records), dtype=torch.bool)),
        "kept_droplets": summarize(keep.cpu()),
        "amputated_fraction": float(1.0 - keep.float().mean()),
        "top_outliers": sorted(records, key=lambda r: r["droplet_score"], reverse=True)[:10],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")

    print(json.dumps(out, indent=2))
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
