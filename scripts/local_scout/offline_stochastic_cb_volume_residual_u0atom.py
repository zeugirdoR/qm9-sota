#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qm9sota.models.tiny_radial_mpnn import build_model
from qm9sota.utils.config import load_yaml
from qm9sota.utils.seed import seed_everything

TARGET_INDEX = 12
TARGET_NAME = "U0_atom"
MEV = 1000.0


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(in_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        last = self.net if isinstance(self.net, nn.Linear) else self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_splits(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return perm[:110000], perm[110000:120000], perm[120000:]


def undirected_neighbors(edge_index: torch.Tensor, n: int):
    neigh = [set() for _ in range(n)]
    ei = edge_index.cpu()
    for a, b in ei.t().tolist():
        if a == b:
            continue
        neigh[a].add(b)
        neigh[b].add(a)
    return neigh


def sample_logvol(vectors: torch.Tensor, k: int, samples: int, gen: torch.Generator, eps: float):
    """
    Stochastic Cauchy-Binet minor estimate.

    For k vectors in R^3, det(V V^T) is the squared k-volume.
    Sampling many k-subsets approximates a Cauchy-Binet sum of squared minors.
    """
    m = int(vectors.shape[0])
    if m < k:
        return torch.empty(0)

    vals = []
    # Enumerate all pairs/triples if cheap; otherwise sample.
    if k == 2 and m <= 16:
        combos = [(i, j) for i in range(m) for j in range(i + 1, m)]
        if len(combos) > samples:
            perm = torch.randperm(len(combos), generator=gen)[:samples].tolist()
            combos = [combos[i] for i in perm]
        for idx in combos:
            V = vectors[list(idx)]
            G = V @ V.t()
            vals.append(torch.logdet(G + eps * torch.eye(k)))
    elif k == 3 and m <= 12:
        combos = [(i, j, l) for i in range(m) for j in range(i + 1, m) for l in range(j + 1, m)]
        if len(combos) > samples:
            perm = torch.randperm(len(combos), generator=gen)[:samples].tolist()
            combos = [combos[i] for i in perm]
        for idx in combos:
            V = vectors[list(idx)]
            G = V @ V.t()
            vals.append(torch.logdet(G + eps * torch.eye(k)))
    else:
        for _ in range(samples):
            idx = torch.randperm(m, generator=gen)[:k]
            V = vectors[idx]
            G = V @ V.t()
            vals.append(torch.logdet(G + eps * torch.eye(k)))

    if not vals:
        return torch.empty(0)
    return torch.stack(vals)


def stats(x: torch.Tensor):
    if x.numel() == 0:
        return torch.zeros(7)

    x = x.float()
    qs = torch.quantile(x, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return torch.tensor([
        float(x.mean()),
        float(x.std(unbiased=False)),
        float(x.min()),
        float(qs[0]),
        float(qs[2]),
        float(qs[4]),
        float(x.max()),
    ])


def graph_stochastic_cb_features(data, *, minor_k: int, samples_per_center: int, seed: int, eps: float):
    pos = data.pos.cpu().float()
    z = data.z.cpu().float() if hasattr(data, "z") and data.z is not None else data.x[:, 0].cpu().float()
    n = int(pos.shape[0])

    neigh = undirected_neighbors(data.edge_index, n)
    gen = torch.Generator().manual_seed(int(seed))

    pos_raw, neg_raw, gap_raw = [], [], []
    pos_unit, neg_unit, gap_unit = [], [], []
    pos_counts, neg_counts = [], []

    all_nodes = set(range(n))

    for c in range(n):
        p_neighbors = sorted(neigh[c])
        non_neighbors = sorted(all_nodes - {c} - set(p_neighbors))

        pos_counts.append(len(p_neighbors))
        neg_counts.append(len(non_neighbors))

        if len(p_neighbors) >= minor_k:
            v_pos = pos[p_neighbors] - pos[c]
            v_pos_unit = v_pos / v_pos.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            lp_raw = sample_logvol(v_pos, minor_k, samples_per_center, gen, eps)
            lp_unit = sample_logvol(v_pos_unit, minor_k, samples_per_center, gen, eps)
        else:
            lp_raw = torch.empty(0)
            lp_unit = torch.empty(0)

        if len(non_neighbors) >= minor_k:
            v_neg = pos[non_neighbors] - pos[c]
            v_neg_unit = v_neg / v_neg.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            ln_raw = sample_logvol(v_neg, minor_k, samples_per_center, gen, eps)
            ln_unit = sample_logvol(v_neg_unit, minor_k, samples_per_center, gen, eps)
        else:
            ln_raw = torch.empty(0)
            ln_unit = torch.empty(0)

        if lp_raw.numel():
            pos_raw.append(lp_raw)
        if ln_raw.numel():
            neg_raw.append(ln_raw)
        if lp_unit.numel():
            pos_unit.append(lp_unit)
        if ln_unit.numel():
            neg_unit.append(ln_unit)

        if lp_raw.numel() and ln_raw.numel():
            gap_raw.append(lp_raw.mean() - ln_raw.mean())
        if lp_unit.numel() and ln_unit.numel():
            gap_unit.append(lp_unit.mean() - ln_unit.mean())

    pos_raw = torch.cat(pos_raw) if pos_raw else torch.empty(0)
    neg_raw = torch.cat(neg_raw) if neg_raw else torch.empty(0)
    pos_unit = torch.cat(pos_unit) if pos_unit else torch.empty(0)
    neg_unit = torch.cat(neg_unit) if neg_unit else torch.empty(0)
    gap_raw = torch.stack(gap_raw) if gap_raw else torch.empty(0)
    gap_unit = torch.stack(gap_unit) if gap_unit else torch.empty(0)

    degree = torch.tensor(pos_counts, dtype=torch.float32)
    neg_count = torch.tensor(neg_counts, dtype=torch.float32)

    # Simple chemistry/topology stabilizers.
    atom_stats = torch.tensor([
        float(n),
        float(len(data.edge_index.t()) / 2.0),
        float(degree.mean()),
        float(degree.std(unbiased=False)),
        float(degree.max()),
        float(neg_count.mean()),
        float(z.mean()),
        float(z.std(unbiased=False)),
        float(z.max()),
    ])

    feats = torch.cat([
        stats(pos_raw),
        stats(neg_raw),
        stats(gap_raw),
        stats(pos_unit),
        stats(neg_unit),
        stats(gap_unit),
        atom_stats,
    ])

    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_cache(dataset, cache_path: Path, *, minor_k: int, samples_per_center: int, seed: int, eps: float):
    X = []
    for i, data in enumerate(dataset):
        if i % 500 == 0:
            print("features processed", i, flush=True)
        X.append(graph_stochastic_cb_features(
            data,
            minor_k=minor_k,
            samples_per_center=samples_per_center,
            seed=seed + i * 1009,
            eps=eps,
        ))
    X = torch.stack(X).float()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": X,
        "minor_k": minor_k,
        "samples_per_center": samples_per_center,
        "seed": seed,
        "eps": eps,
        "feature_dim": int(X.shape[1]),
        "method": "stochastic CB positive/negative local volume summaries",
    }
    torch.save(payload, cache_path)
    print("Wrote feature cache:", cache_path, X.shape)
    return X


@torch.no_grad()
def compute_preds(model, dataset, indices, device, batch_size=256):
    subset = [dataset[int(i)] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    preds, ys = [], []
    model.eval()

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        preds.append(out[:, TARGET_INDEX].detach().cpu())
        ys.append(batch.y[:, TARGET_INDEX].detach().cpu())

    return torch.cat(preds), torch.cat(ys)


def mae_mev(pred_norm, y_norm, y_mu, y_sd):
    pred_raw = pred_norm * y_sd + y_mu
    y_raw = y_norm * y_sd + y_mu
    return float((pred_raw - y_raw).abs().mean() * MEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--feature-cache", required=True)
    ap.add_argument("--data-root", default=str(Path.home() / "data/QM9"))
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--minor-k", type=int, default=2)
    ap.add_argument("--samples-per-center", type=int, default=32)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--pred-batch-size", type=int, default=256)
    ap.add_argument("--eval-test", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(Path(args.config))
    cfg["seed"] = args.seed
    cfg.setdefault("data", {})
    cfg["data"]["root"] = args.data_root
    cfg["data"]["smoke"] = False
    cfg["data"]["batch_size"] = args.pred_batch_size

    dataset = QM9(root=args.data_root)
    n = len(dataset)
    train_idx, val_idx, test_idx = make_splits(n, args.seed)

    cache_path = Path(args.feature_cache)
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
        X = cache["features"].float()
        print("Loaded feature cache:", cache_path, X.shape)
    else:
        X = build_feature_cache(
            dataset,
            cache_path,
            minor_k=args.minor_k,
            samples_per_center=args.samples_per_center,
            seed=args.seed,
            eps=args.eps,
        )

    assert X.shape[0] == n, (X.shape, n)

    # Standardize features on train split.
    x_mu = X[train_idx].mean(dim=0)
    x_sd = X[train_idx].std(dim=0).clamp_min(1e-8)
    Xn = (X - x_mu) / x_sd

    # Target normalization, matching training.
    ys_full = torch.cat([data.y for data in dataset], dim=0).float()
    target_mean = ys_full[train_idx].mean(dim=0)
    target_std = ys_full[train_idx].std(dim=0).clamp_min(1e-8)
    y_mu = target_mean[TARGET_INDEX]
    y_sd = target_std[TARGET_INDEX]

    # Load frozen base model.
    model = build_model(cfg).to(device)
    ckpt_p = Path(args.run_dir) / "best_model.pt"
    assert ckpt_p.exists(), ckpt_p
    obj = torch.load(ckpt_p, map_location=device)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state)
    model.eval()

    print("Computing frozen model predictions...", flush=True)
    pred_train_norm, y_train_raw = compute_preds(model, dataset, train_idx, device, args.pred_batch_size)
    pred_val_norm, y_val_raw = compute_preds(model, dataset, val_idx, device, args.pred_batch_size)

    y_train_norm = (y_train_raw - y_mu) / y_sd
    y_val_norm = (y_val_raw - y_mu) / y_sd

    base_val_mev = mae_mev(pred_val_norm, y_val_norm, y_mu, y_sd)
    resid_train = y_train_norm - pred_train_norm

    print("base_val_mev:", base_val_mev)

    net = ResidualMLP(Xn.shape[1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    Xtr = Xn[train_idx].to(device)
    rtr = resid_train.to(device)
    Xv = Xn[val_idx].to(device)

    pred_val_norm_dev = pred_val_norm.to(device)

    best = {
        "best_epoch": None,
        "best_val_mev": float("inf"),
        "best_delta_val_mev": None,
    }
    best_state = None

    m = Xtr.shape[0]

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(m, device=device)

        net.train()
        total = 0.0
        for start in range(0, m, args.batch_size):
            idx = perm[start:start + args.batch_size]
            pred_r = net(Xtr[idx])
            loss = F.mse_loss(pred_r, rtr[idx])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu()) * idx.numel()

        net.eval()
        with torch.no_grad():
            val_r = net(Xv)
            pred_val_corr = pred_val_norm_dev + val_r
            val_mev = mae_mev(pred_val_corr.cpu(), y_val_norm, y_mu, y_sd)

        if val_mev < best["best_val_mev"]:
            best = {
                "best_epoch": epoch,
                "best_val_mev": val_mev,
                "best_delta_val_mev": val_mev - base_val_mev,
            }
            best_state = copy.deepcopy(net.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print({
                "epoch": epoch,
                "train_loss": total / m,
                "val_mev": val_mev,
                "delta_val_mev": val_mev - base_val_mev,
                "best": best,
            }, flush=True)

    if best_state is not None:
        net.load_state_dict(best_state)

    residual_ckpt = out_dir / "best_residual_head.pt"
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "feature_mean": x_mu,
            "feature_std": x_sd,
            "target_mean": target_mean,
            "target_std": target_std,
            "target_index": TARGET_INDEX,
            "feature_cache": str(cache_path),
            "best": best,
            "hidden_dim": args.hidden_dim,
        },
        residual_ckpt,
    )

    result = {
        "target": TARGET_NAME,
        "target_index": TARGET_INDEX,
        "seed": args.seed,
        "method": "offline stochastic CB positive/negative local volume residual",
        "base_run_dir": str(args.run_dir),
        "config": str(args.config),
        "feature_cache": str(cache_path),
        "feature_dim": int(X.shape[1]),
        "minor_k": args.minor_k,
        "samples_per_center": args.samples_per_center,
        "base_val_mev": base_val_mev,
        **best,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "weight_decay": args.weight_decay,
        "best_residual_head": str(residual_ckpt),
    }

    # Optional test is for final inspection only, not tuning.
    if args.eval_test:
        pred_test_norm, y_test_raw = compute_preds(model, dataset, test_idx, device, args.pred_batch_size)
        y_test_norm = (y_test_raw - y_mu) / y_sd
        Xt = Xn[test_idx].to(device)
        with torch.no_grad():
            test_r = net(Xt)
            pred_test_corr = pred_test_norm.to(device) + test_r
        base_test_mev = mae_mev(pred_test_norm, y_test_norm, y_mu, y_sd)
        cb_test_mev = mae_mev(pred_test_corr.cpu(), y_test_norm, y_mu, y_sd)
        result.update({
            "base_test_mev": base_test_mev,
            "cb_test_mev": cb_test_mev,
            "delta_test_mev": cb_test_mev - base_test_mev,
        })

    out = out_dir / "offline_stochastic_cb_volume_residual_summary.json"
    out.write_text(json.dumps(result, indent=2))
    print("\nRESULT")
    print(json.dumps(result, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    main()
