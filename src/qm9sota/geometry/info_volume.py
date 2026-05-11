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

    # If numerical sign fails, return very negative log volume rather than NaN.
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

    This is a placeholder for richer Cauchy-Binet local neighborhood features.
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
    centers = torch.linspace(0.0, cutoff, num_basis, dtype=dist.dtype, device=dist.device)
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

    base = torch.stack([dist, dist2, torch.log1p(dist2)], dim=-1)
    rbf = gaussian_rbf(dist, num_basis=num_basis, cutoff=cutoff)
    return torch.cat([base, rbf], dim=-1)


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
    centers = torch.linspace(0.0, cutoff, num_basis, dtype=dist.dtype, device=dist.device)
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

    base = torch.stack([dist, dist2, torch.log1p(dist2)], dim=-1)
    rbf = gaussian_rbf(dist, num_basis=num_basis, cutoff=cutoff)
    return torch.cat([base, rbf], dim=-1)


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
