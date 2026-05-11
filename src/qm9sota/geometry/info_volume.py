from __future__ import annotations

import torch


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., k, d]
    returns: [..., k, k] Gram matrix x x^T
    """
    return x @ x.transpose(-1, -2)


def stable_logdet_gram(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Stable logdet of Gram matrix.

    x: [..., k, d]
    returns: [...]

    This approximates squared k-volume via det(X X^T).
    """
    g = gram_matrix(x)
    eye = torch.eye(g.shape[-1], dtype=g.dtype, device=g.device)
    g = g + eps * eye
    sign, logabsdet = torch.linalg.slogdet(g)

    return torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, -30.0))


def pair_area_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Squared area spanned by two vectors using Gram determinant.

    a, b: [..., 3]
    returns: [...]
    """
    aa = (a * a).sum(dim=-1)
    bb = (b * b).sum(dim=-1)
    ab = (a * b).sum(dim=-1)
    return (aa * bb - ab * ab).clamp_min(0.0)


def triple_volume_sq(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Squared volume of parallelepiped spanned by a,b,c.

    a,b,c: [..., 3]
    returns: [...]
    """
    x = torch.stack([a, b, c], dim=-2)
    g = gram_matrix(x)
    return torch.linalg.det(g).clamp_min(0.0)


def local_edge_volume_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Prototype edge features from relative positions.

    Returns [E, 3]:
      distance
      distance^2
      log(1 + distance^2)
    """
    src, dst = edge_index
    rel = pos[src] - pos[dst]
    dist2 = (rel * rel).sum(dim=-1).clamp_min(1e-12)
    dist = dist2.sqrt()
    return torch.stack([dist, dist2, torch.log1p(dist2)], dim=-1)


def gaussian_rbf(
    dist: torch.Tensor,
    *,
    num_basis: int = 32,
    cutoff: float = 8.0,
    gamma: float | None = None,
) -> torch.Tensor:
    """
    Gaussian radial basis expansion.

    dist: [E]
    returns: [E, num_basis]
    """
    centers = torch.linspace(
        0.0,
        cutoff,
        num_basis,
        dtype=dist.dtype,
        device=dist.device,
    )

    if gamma is None:
        spacing = cutoff / max(num_basis - 1, 1)
        gamma = 1.0 / max(spacing * spacing, 1e-6)

    return torch.exp(-gamma * (dist.unsqueeze(-1) - centers.unsqueeze(0)).pow(2))


def radial_edge_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_basis: int = 32,
    cutoff: float = 8.0,
) -> torch.Tensor:
    """
    Rich radial edge features.

    Returns [E, 3 + num_basis]:
      distance
      distance^2
      log(1 + distance^2)
      Gaussian RBF(distance)
    """
    src, dst = edge_index
    rel = pos[src] - pos[dst]

    dist2 = (rel * rel).sum(dim=-1).clamp_min(1e-12)
    dist = dist2.sqrt()

    base = torch.stack(
        [
            dist,
            dist2,
            torch.log1p(dist2),
        ],
        dim=-1,
    )

    rbf = gaussian_rbf(
        dist,
        num_basis=num_basis,
        cutoff=cutoff,
    )

    return torch.cat([base, rbf], dim=-1)


def node_cauchy_binet_summaries(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Local Cauchy-Binet / Gram-volume summaries per node.

    For node i, gather outgoing local relative vectors r_ij over molecular edges.
    We avoid explicit triple enumeration and use stable scatter summaries that are
    related to Gram/Cauchy-Binet quantities.

    Returns [N, 8]:
      degree
      mean distance
      mean distance^2
      radial variance proxy
      trace covariance proxy
      squared Frobenius covariance proxy
      anisotropy proxy
      log volume proxy

    These are invariant scalar summaries of the local geometric neighborhood.
    """
    src, dst = edge_index
    num_nodes = pos.shape[0]

    # Treat each directed edge src -> dst as a neighbor vector centered at dst.
    rel = pos[src].float() - pos[dst].float()
    dist2 = (rel * rel).sum(dim=-1).clamp_min(eps)
    dist = dist2.sqrt()

    one = pos.new_ones(src.shape[0], 1)
    deg = pos.new_zeros(num_nodes, 1)
    deg.index_add_(0, dst, one)
    deg_safe = deg.clamp_min(1.0)

    sum_rel = pos.new_zeros(num_nodes, 3)
    sum_rel.index_add_(0, dst, rel)
    mean_rel = sum_rel / deg_safe

    centered = rel - mean_rel[dst]

    sum_dist = pos.new_zeros(num_nodes, 1)
    sum_dist.index_add_(0, dst, dist.unsqueeze(-1))

    sum_dist2 = pos.new_zeros(num_nodes, 1)
    sum_dist2.index_add_(0, dst, dist2.unsqueeze(-1))

    mean_dist = sum_dist / deg_safe
    mean_dist2 = sum_dist2 / deg_safe
    radial_var = (mean_dist2 - mean_dist.pow(2)).clamp_min(0.0)

    # Covariance-like local Gram summary C_i = mean sum r r^T.
    outer = centered.unsqueeze(-1) * centered.unsqueeze(-2)  # [E, 3, 3]
    cov = pos.new_zeros(num_nodes, 3, 3)
    cov.index_add_(0, dst, outer)
    cov = cov / deg_safe.view(num_nodes, 1, 1)

    trace = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    frob_sq = (cov * cov).sum(dim=(-1, -2), keepdim=True).view(num_nodes, 1)

    # Isotropy/anisotropy proxy: ||C||_F^2 / tr(C)^2.
    anisotropy = frob_sq / trace.pow(2).clamp_min(eps)

    eye = torch.eye(3, dtype=pos.dtype, device=pos.device).unsqueeze(0)
    cov_reg = cov + eps * eye
    sign, logdet = torch.linalg.slogdet(cov_reg)
    logvol = torch.where(sign > 0, logdet, torch.full_like(logdet, -30.0)).unsqueeze(-1)

    return torch.cat(
        [
            deg,
            mean_dist,
            mean_dist2,
            radial_var,
            trace,
            frob_sq,
            anisotropy,
            logvol,
        ],
        dim=-1,
    )


def cauchy_binet_edge_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Lift node-local Cauchy-Binet summaries to edges.

    Returns [E, 8 * 4]:
      src_summary
      dst_summary
      abs(src - dst)
      src * dst

    This lets edge attention see local geometry around both incident atoms.
    """
    src, dst = edge_index
    node_feat = node_cauchy_binet_summaries(pos, edge_index)

    src_feat = node_feat[src]
    dst_feat = node_feat[dst]

    return torch.cat(
        [
            src_feat,
            dst_feat,
            (src_feat - dst_feat).abs(),
            src_feat * dst_feat,
        ],
        dim=-1,
    )


def radial_cauchy_binet_edge_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    num_basis: int = 32,
    cutoff: float = 8.0,
) -> torch.Tensor:
    """
    M6 edge features.

    Returns [E, 3 + num_basis + 32]:
      radial edge features
      Cauchy-Binet local volume edge summaries
    """
    radial = radial_edge_features(
        pos,
        edge_index,
        num_basis=num_basis,
        cutoff=cutoff,
    )
    cb = cauchy_binet_edge_features(pos, edge_index)
    return torch.cat([radial, cb], dim=-1)
