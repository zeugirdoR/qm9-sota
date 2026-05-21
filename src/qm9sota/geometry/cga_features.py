from __future__ import annotations

import torch
from torch import Tensor


def pairwise_sq_dists(pos: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Pairwise squared Euclidean distances.

    In standard CGA point embedding P(x), the conformal inner product satisfies
        P_i · P_j = -0.5 ||x_i - x_j||^2
    up to convention.

    So this is the numerically stable core of the CGA point Gramian.
    """
    diff = pos[:, None, :] - pos[None, :, :]
    d2 = (diff * diff).sum(dim=-1)
    return torch.clamp(d2, min=0.0)


def cga_point_gram(pos: Tensor) -> Tensor:
    """
    CGA conformal point Gramian for Euclidean points.

    Returns G_ij = -0.5 ||x_i - x_j||^2.

    Shape:
        pos: [N, 3]
        G:   [N, N]
    """
    return -0.5 * pairwise_sq_dists(pos)


def local_gram_volume_features(
    pos: Tensor,
    k: int = 4,
    eps: float = 1e-6,
) -> Tensor:
    """
    Per-atom local Gram/volume features.

    For each atom i:
      - find k nearest neighbors by Euclidean distance,
      - center their displacement vectors at atom i,
      - form local Gram matrix D D^T,
      - compute stable scalar summaries.

    This is not a full CGA transformer. It is a small, safe, invariant
    geometric feature branch meant to test whether local Gram/volume
    information helps M4.

    Returns [N, 6]:
      0. mean neighbor distance
      1. std neighbor distance
      2. min neighbor distance
      3. max neighbor distance
      4. logdet(I + Gram)
      5. normalized local volume proxy
    """
    n = pos.size(0)
    device = pos.device
    dtype = pos.dtype

    if n <= 1:
        return torch.zeros((n, 6), device=device, dtype=dtype)

    d2 = pairwise_sq_dists(pos)
    d = torch.sqrt(d2 + eps)

    # Exclude self.
    d_for_knn = d + torch.eye(n, device=device, dtype=dtype) * 1.0e6
    kk = min(k, n - 1)

    knn_dist, knn_idx = torch.topk(d_for_knn, k=kk, largest=False, dim=-1)

    feats = []
    eye = torch.eye(kk, device=device, dtype=dtype)

    for i in range(n):
        neigh = pos[knn_idx[i]]
        disp = neigh - pos[i : i + 1]  # [kk, 3]
        gram = disp @ disp.transpose(0, 1)  # [kk, kk]

        # Stable logdet. I + Gram avoids singularity and scale blowup.
        sign, logabsdet = torch.linalg.slogdet(eye + gram)
        logdet = torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))

        # Product of singular values proxy, normalized by k.
        # This is a soft local volume/area content summary.
        s = torch.linalg.svdvals(disp)
        vol_proxy = torch.log1p(torch.prod(s + eps))

        row = torch.stack(
            [
                knn_dist[i].mean(),
                knn_dist[i].std(unbiased=False),
                knn_dist[i].min(),
                knn_dist[i].max(),
                logdet,
                vol_proxy,
            ]
        )
        feats.append(row)

    return torch.stack(feats, dim=0)


def graph_cga_summary(pos: Tensor, batch: Tensor | None = None, k: int = 4) -> Tensor:
    """
    Graph-level CGA/Gram summary.

    If batch is None, returns [1, 6].
    If batch is provided, returns [num_graphs, 6].

    Uses mean local Gram/volume features over atoms.
    """
    if batch is None:
        return local_gram_volume_features(pos, k=k).mean(dim=0, keepdim=True)

    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    outs = []

    for g in range(num_graphs):
        mask = batch == g
        if mask.sum() == 0:
            outs.append(torch.zeros((6,), device=pos.device, dtype=pos.dtype))
        else:
            outs.append(local_gram_volume_features(pos[mask], k=k).mean(dim=0))

    return torch.stack(outs, dim=0)
