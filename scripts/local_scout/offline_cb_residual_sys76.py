from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

REPO = Path.home() / "awsgit" / "qm9-sota"
sys.path.insert(0, str(REPO / "src"))

from qm9sota.utils.config import load_yaml
from qm9sota.models.tiny_radial_mpnn import build_model
from qm9sota.geometry.cb_features import graph_cb_summary

TARGET_INDEX = 12
TARGET_NAME = "U0_atom"
MEV = 1000.0


class CBResidualMLP(nn.Module):
    def __init__(self, in_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Exact no-op start.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_splits(n: int, seed: int = 43):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    train_idx = perm[:110000]
    val_idx = perm[110000:120000]
    test_idx = perm[120000:]
    return train_idx, val_idx, test_idx


def precompute_cb_cache(dataset, out_path: Path, k: int = 4):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs = []
    ys = []

    n = len(dataset)
    print("Precomputing CB features for", n, "molecules")
    print("Output:", out_path)

    for i in range(n):
        data = dataset[i]
        feat = graph_cb_summary(data.pos, batch=None, k=k).squeeze(0).cpu()
        xs.append(feat)
        ys.append(data.y.view(-1).cpu())

        if (i + 1) % 5000 == 0:
            print("processed", i + 1)

    X = torch.stack(xs).float()
    Y = torch.stack(ys).float()

    payload = {
        "features": X,
        "targets": Y,
        "feature_name": "graph_cb_summary",
        "k": k,
        "target_index_U0_atom": TARGET_INDEX,
    }

    torch.save(payload, out_path)
    print("Wrote:", out_path)
    print("features:", X.shape)
    print("targets:", Y.shape)


@torch.no_grad()
def compute_m4_preds(model, dataset, indices, device, batch_size=128):
    subset = [dataset[int(i)] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    preds = []
    ys = []

    model.eval()
    for data in loader:
        data = data.to(device)
        out = model(data)
        preds.append(out[:, TARGET_INDEX].detach().cpu())
        ys.append(data.y[:, TARGET_INDEX].detach().cpu())

    return torch.cat(preds), torch.cat(ys)


def mae_mev(pred_norm, y_norm, y_mu, y_sd):
    pred_raw = pred_norm * y_sd + y_mu
    y_raw = y_norm * y_sd + y_mu
    return float((pred_raw - y_raw).abs().mean() * MEV)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cb-cache", required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pred-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-root", default=str(Path.home() / "data" / "QM9"))
    parser.add_argument("--precompute-only", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset = QM9(root=str(data_root))
    n = len(dataset)
    print("dataset:", n)

    cb_cache = Path(args.cb_cache)
    if not cb_cache.exists():
        precompute_cb_cache(dataset, cb_cache, k=4)

    if args.precompute_only:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_idx, val_idx, test_idx = make_splits(n, seed=args.seed)
    print("splits:", len(train_idx), len(val_idx), len(test_idx))

    cache = torch.load(cb_cache, map_location="cpu")
    X_cb = cache["features"].float()
    Y_raw_all = cache["targets"].float()[:, TARGET_INDEX]

    assert X_cb.shape[0] == n, (X_cb.shape, n)

    # Normalize CB features by training split.
    mu_x = X_cb[train_idx].mean(dim=0)
    sd_x = X_cb[train_idx].std(dim=0).clamp_min(1e-8)
    Xn = (X_cb - mu_x) / sd_x

    # Build/load frozen M4.
    cfg = load_yaml(Path(args.config))
    model = build_model(cfg).to(device)

    run_dir = Path(args.run_dir)
    best = run_dir / "best_model.pt"
    assert best.exists(), best

    obj = torch.load(best, map_location=device)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(state)
    model.eval()

    y_train_raw = Y_raw_all[train_idx]
    y_mu = y_train_raw.mean()
    y_sd = y_train_raw.std(unbiased=True).clamp_min(1e-8)
    print("y_mu:", float(y_mu), "y_sd:", float(y_sd))

    print("computing M4 train preds...")
    pred_train_norm, y_train_raw2 = compute_m4_preds(
        model, dataset, train_idx, device, batch_size=args.pred_batch_size
    )
    print("computing M4 val preds...")
    pred_val_norm, y_val_raw = compute_m4_preds(
        model, dataset, val_idx, device, batch_size=args.pred_batch_size
    )
    print("computing M4 test preds...")
    pred_test_norm, y_test_raw = compute_m4_preds(
        model, dataset, test_idx, device, batch_size=args.pred_batch_size
    )

    y_train_norm = (y_train_raw2 - y_mu) / y_sd
    y_val_norm = (y_val_raw - y_mu) / y_sd
    y_test_norm = (y_test_raw - y_mu) / y_sd

    resid_train = y_train_norm - pred_train_norm
    resid_val = y_val_norm - pred_val_norm
    resid_test = y_test_norm - pred_test_norm

    base_val_mev = mae_mev(pred_val_norm, y_val_norm, y_mu, y_sd)
    base_test_mev = mae_mev(pred_test_norm, y_test_norm, y_mu, y_sd)

    print("base val meV:", base_val_mev)
    print("base test meV:", base_test_mev)

    net = CBResidualMLP(in_dim=Xn.shape[1], hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)

    Xtr = Xn[train_idx].to(device)
    rtr = resid_train.to(device)
    Xv = Xn[val_idx].to(device)
    Xt = Xn[test_idx].to(device)

    y_val_norm_dev = y_val_norm.to(device)
    y_test_norm_dev = y_test_norm.to(device)
    pred_val_norm_dev = pred_val_norm.to(device)
    pred_test_norm_dev = pred_test_norm.to(device)

    best_state = None
    best_val = float("inf")
    best_epoch = None

    m = Xtr.shape[0]
    batch = args.batch_size

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(m, device=device)
        losses = []

        net.train()
        for start in range(0, m, batch):
            idx = perm[start:start + batch]
            pred_r = net(Xtr[idx])
            loss = F.mse_loss(pred_r, rtr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        net.eval()
        with torch.no_grad():
            val_r = net(Xv)
            test_r = net(Xt)

            pred_val_corr = pred_val_norm_dev + val_r
            pred_test_corr = pred_test_norm_dev + test_r

            val_mev = mae_mev(pred_val_corr.cpu(), y_val_norm, y_mu, y_sd)
            test_mev = mae_mev(pred_test_corr.cpu(), y_test_norm, y_mu, y_sd)

        if val_mev < best_val:
            best_val = val_mev
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0:
            print({
                "epoch": epoch,
                "train_loss": sum(losses) / max(1, len(losses)),
                "val_mev": val_mev,
                "test_mev": test_mev,
                "best_val": best_val,
                "best_epoch": best_epoch,
            })

    net.load_state_dict(best_state)
    net.eval()

    with torch.no_grad():
        val_r = net(Xv)
        test_r = net(Xt)
        pred_val_corr = pred_val_norm_dev + val_r
        pred_test_corr = pred_test_norm_dev + test_r

        best_val_mev = mae_mev(pred_val_corr.cpu(), y_val_norm, y_mu, y_sd)
        best_test_mev = mae_mev(pred_test_corr.cpu(), y_test_norm, y_mu, y_sd)

    result = {
        "target": TARGET_NAME,
        "target_index": TARGET_INDEX,
        "seed": args.seed,
        "method": "offline CB residual MLP on frozen M4 normalized residuals",
        "cb_cache": str(cb_cache),
        "m4_run_dir": str(run_dir),
        "base_val_mev": base_val_mev,
        "base_test_mev": base_test_mev,
        "best_epoch": best_epoch,
        "best_val_mev": best_val_mev,
        "best_test_mev": best_test_mev,
        "delta_val_mev": best_val_mev - base_val_mev,
        "delta_test_mev": best_test_mev - base_test_mev,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
    }

    torch.save(
        {
            "model_state_dict": best_state,
            "result": result,
            "x_mean": mu_x,
            "x_std": sd_x,
            "y_mean": y_mu,
            "y_std": y_sd,
        },
        out_dir / "offline_cb_residual_model.pt",
    )

    (out_dir / "offline_cb_residual_summary.json").write_text(json.dumps(result, indent=2))

    print("RESULT")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
