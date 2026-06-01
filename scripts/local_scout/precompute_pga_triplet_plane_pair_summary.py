#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.datasets import QM9

MAX_N = 29


def pair_plane_summary(pos: torch.Tensor, z: torch.Tensor, max_n: int = MAX_N) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each ordered pair (i,j), summarize all triplet planes (i,j,k), k != i,j.

    Coordinates are centered by centroid.

    Plane:
      n = (x_j - x_i) x (x_k - x_i)
      d = - n dot x_i

    Summary per pair:
      0 mean_area
      1 max_area
      2 std_area
      3 mean_abs_d
      4 max_abs_d
      5 normal_mean_norm
      6 normal_cov_trace
      7 near_collinear_frac
      8 heavy_k_frac
      9 count_norm
    """
    n = int(pos.size(0))
    out = pos.new_zeros((max_n, max_n, 10))
    mask = torch.zeros((max_n, max_n), dtype=torch.bool)

    if n < 3:
        return out, mask

    x = pos.float()
    x = x - x.mean(dim=0, keepdim=True)

    heavy = (z[:n].long() > 1).float()

    for i in range(n):
        xi = x[i]
        for j in range(n):
            if i == j:
                continue
            xj = x[j]
            areas = []
            abs_ds = []
            normals = []
            heavy_flags = []
            for k in range(n):
                if k == i or k == j:
                    continue
                xk = x[k]
                normal = torch.cross(xj - xi, xk - xi, dim=0)
                area = normal.norm()
                d = -(normal * xi).sum()
                normals.append(normal)
                areas.append(area)
                abs_ds.append(d.abs())
                heavy_flags.append(heavy[k])

            if not areas:
                continue

            areas_t = torch.stack(areas)
            abs_d_t = torch.stack(abs_ds)
            normals_t = torch.stack(normals)
            count = float(areas_t.numel())

            near_col = (areas_t < 1e-4).float().mean()
            normal_mean = normals_t.mean(dim=0)
            normal_centered = normals_t - normal_mean
            cov_trace = (normal_centered.pow(2).sum(dim=1).mean()).sqrt()

            out[i, j, 0] = areas_t.mean()
            out[i, j, 1] = areas_t.max()
            out[i, j, 2] = areas_t.std(unbiased=False)
            out[i, j, 3] = abs_d_t.mean()
            out[i, j, 4] = abs_d_t.max()
            out[i, j, 5] = normal_mean.norm()
            out[i, j, 6] = cov_trace
            out[i, j, 7] = near_col
            out[i, j, 8] = torch.stack(heavy_flags).float().mean()
            out[i, j, 9] = count / max(float(max_n), 1.0)
            mask[i, j] = True

    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path.home() / "data/QM9"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-n", type=int, default=MAX_N)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = ap.parse_args()

    ds = QM9(root=args.data_root)
    max_n = int(args.max_n)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    summaries = torch.zeros((len(ds), max_n, max_n, 10), dtype=dtype)
    masks = torch.zeros((len(ds), max_n, max_n), dtype=torch.bool)
    n_atoms = torch.zeros((len(ds),), dtype=torch.int16)

    for idx in range(len(ds)):
        data = ds[idx]
        pos = data.pos.float()
        z = data.z.long()
        n_atoms[idx] = int(pos.size(0))
        s, m = pair_plane_summary(pos, z, max_n=max_n)
        summaries[idx] = s.to(dtype)
        masks[idx] = m
        if idx % 5000 == 0:
            print({"idx": idx, "n": len(ds)}, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pair_plane_summary": summaries,
            "pair_mask": masks,
            "n_atoms": n_atoms,
            "max_n": max_n,
            "feature_names": [
                "mean_area",
                "max_area",
                "std_area",
                "mean_abs_d",
                "max_abs_d",
                "normal_mean_norm",
                "normal_cov_trace",
                "near_collinear_frac",
                "heavy_k_frac",
                "count_norm",
            ],
            "description": "PGA triplet-plane pair summaries. Plane convention: pi=d e0 + nx e1 + ny e2 + nz e3. Atom point convention: P=e123+x e023+y e031+z e012.",
        },
        out,
    )
    print("Wrote:", out)
    print(json.dumps({"n": len(ds), "out": str(out), "max_n": max_n, "dtype": args.dtype}, indent=2))


if __name__ == "__main__":
    main()
