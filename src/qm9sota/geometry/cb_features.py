from __future__ import annotations

import torch
from torch import Tensor


def pairwise_sq_dists(pos: Tensor) -> Tensor:
    diff = pos[:, None, :] - pos[None, :, :]
    return torch.clamp((diff * diff).sum(dim=-1), min=0.0)


def local_cb_volume_features(
    pos: Tensor,
    k: int = 4,
    eps: float = 1e-6,
) -> Tensor:
    """
    Local Cauchy-Binet / Gram-volume summary features.

    For each atom i:
      - take k nearest neighbors,
      - form displacement matrix D,
      - compute Gram = D D^T,
      - use determinant/eigenvalue/volume summaries.

    Returns [N, 8]:
      0 mean neighbor distance
      1 std neighbor distance
      2 logdet(I + Gram)
      3 logdet(eps I + Gram)
      4 log product singular values
      5 max singular value
      6 min singular value
      7 condition log-ratio
    """
    n = pos.size(0)
    device = pos.device
    dtype = pos.dtype

    if n <= 1:
        return torch.zeros((n, 8), device=device, dtype=dtype)

    d2 = pairwise_sq_dists(pos)
    d = torch.sqrt(d2 + eps)

    d_for_knn = d + torch.eye(n, device=device, dtype=dtype) * 1.0e6
    kk = min(k, n - 1)
    knn_dist, knn_idx = torch.topk(d_for_knn, k=kk, largest=False, dim=-1)

    eye = torch.eye(kk, device=device, dtype=dtype)
    feats = []

    for i in range(n):
        neigh = pos[knn_idx[i]]
        disp = neigh - pos[i : i + 1]  # [kk, 3]
        gram = disp @ disp.transpose(0, 1)

        sign1, logdet1 = torch.linalg.slogdet(eye + gram)
        logdet1 = torch.where(sign1 > 0, logdet1, torch.zeros_like(logdet1))

        sign2, logdet2 = torch.linalg.slogdet(eps * eye + gram)
        logdet2 = torch.where(sign2 > 0, logdet2, torch.zeros_like(logdet2))

        s = torch.linalg.svdvals(disp)
        s_safe = s + eps

        log_prod_s = torch.log(s_safe).sum()
        max_s = s_safe.max()
        min_s = s_safe.min()
        log_cond = torch.log(max_s / min_s)

        row = torch.stack(
            [
                knn_dist[i].mean(),
                knn_dist[i].std(unbiased=False),
                logdet1,
                logdet2,
                log_prod_s,
                max_s,
                min_s,
                log_cond,
            ]
        )
        feats.append(row)

    return torch.stack(feats, dim=0)


def graph_cb_summary(pos: Tensor, batch: Tensor | None = None, k: int = 4) -> Tensor:
    """
    Graph-level CB / information-volume summary.

    Returns [num_graphs, 16]:
      mean and std over atoms of the 8 local CB features.
    """
    if batch is None:
        local = local_cb_volume_features(pos, k=k)
        return torch.cat(
            [local.mean(dim=0), local.std(dim=0, unbiased=False)],
            dim=0,
        ).unsqueeze(0)

    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    outs = []

    for g in range(num_graphs):
        mask = batch == g
        if mask.sum() == 0:
            outs.append(torch.zeros((16,), device=pos.device, dtype=pos.dtype))
            continue

        local = local_cb_volume_features(pos[mask], k=k)
        outs.append(
            torch.cat(
                [local.mean(dim=0), local.std(dim=0, unbiased=False)],
                dim=0,
            )
        )

    return torch.stack(outs, dim=0)
