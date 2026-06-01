#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.datasets import QM9


def all_triplet_planes(pos: torch.Tensor, z: torch.Tensor):
    n = int(pos.size(0))
    x = pos.float()
    x = x - x.mean(dim=0, keepdim=True)

    planes = []
    areas = []
    zstats = []
    indices = []

    if n < 3:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int16),
        )

    for i in range(n - 2):
        xi = x[i]
        for j in range(i + 1, n - 1):
            xj = x[j]
            for k in range(j + 1, n):
                xk = x[k]
                normal = torch.cross(xj - xi, xk - xi, dim=0)
                area = normal.norm()
                d = -(normal * xi).sum()
                plane = torch.cat([d.view(1), normal], dim=0)

                zi, zj, zk = float(z[i]), float(z[j]), float(z[k])
                zs = torch.tensor(
                    [
                        zi + zj + zk,
                        max(zi, zj, zk),
                        float((zi > 1) + (zj > 1) + (zk > 1)),
                        area.item(),
                    ],
                    dtype=torch.float32,
                )

                planes.append(plane)
                areas.append(area)
                zstats.append(zs)
                indices.append(torch.tensor([i, j, k], dtype=torch.int16))

    return torch.stack(planes), torch.stack(areas), torch.stack(zstats), torch.stack(indices)


def sample_planes(planes, areas, zstats, indices, k: int, generator: torch.Generator):
    n = int(planes.size(0))
    if n == 0:
        return (
            torch.zeros((k, 4), dtype=torch.float16),
            torch.zeros((k,), dtype=torch.float16),
            torch.zeros((k, 4), dtype=torch.float16),
            torch.full((k, 3), -1, dtype=torch.int16),
            torch.zeros((k,), dtype=torch.bool),
        )

    k_rand = k // 2
    k_top = k - k_rand

    rand_idx = torch.randint(0, n, (k_rand,), generator=generator)

    if n >= k_top:
        top_idx = torch.topk(areas, k_top, largest=True).indices
    else:
        top_idx = torch.randint(0, n, (k_top,), generator=generator)

    sel = torch.cat([rand_idx, top_idx], dim=0)

    sel_planes = planes[sel]
    sel_areas = areas[sel]
    sel_zstats = zstats[sel]
    sel_indices = indices[sel]

    # Normalize plane vector but keep area separately.
    norm = sel_planes.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sel_planes_unit = sel_planes / norm

    valid = torch.ones((k,), dtype=torch.bool)
    return (
        sel_planes_unit.to(torch.float16),
        sel_areas.to(torch.float16),
        sel_zstats.to(torch.float16),
        sel_indices.to(torch.int16),
        valid,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path.home() / "data/QM9"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    ds = QM9(root=args.data_root)
    k = int(args.k)
    gen = torch.Generator().manual_seed(int(args.seed))

    planes_out = torch.zeros((len(ds), k, 4), dtype=torch.float16)
    areas_out = torch.zeros((len(ds), k), dtype=torch.float16)
    zstats_out = torch.zeros((len(ds), k, 4), dtype=torch.float16)
    idx_out = torch.full((len(ds), k, 3), -1, dtype=torch.int16)
    mask_out = torch.zeros((len(ds), k), dtype=torch.bool)
    n_atoms = torch.zeros((len(ds),), dtype=torch.int16)

    for idx in range(len(ds)):
        data = ds[idx]
        n_atoms[idx] = int(data.pos.size(0))
        planes, areas, zstats, indices = all_triplet_planes(data.pos, data.z)
        p, a, zs, ti, m = sample_planes(planes, areas, zstats, indices, k, gen)
        planes_out[idx] = p
        areas_out[idx] = a
        zstats_out[idx] = zs
        idx_out[idx] = ti
        mask_out[idx] = m
        if idx % 5000 == 0:
            print({"idx": idx, "n": len(ds)}, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "planes": planes_out,
            "areas": areas_out,
            "zstats": zstats_out,
            "triplet_indices": idx_out,
            "mask": mask_out,
            "n_atoms": n_atoms,
            "k": k,
            "description": "Stochastic PGA triplet plane samples. Plane pi=[d,nx,ny,nz], atom point convention P=e123+x e023+y e031+z e012.",
        },
        out,
    )
    print("Wrote:", out)
    print(json.dumps({"n": len(ds), "k": k, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
