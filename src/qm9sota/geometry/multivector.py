from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Multivector:
    """
    Practical grade-separated multivector representation for 3D molecular models.

    We represent:
      s: scalar channels              [N, Cs]
      v: vector channels              [N, Cv, 3]
      b: bivector channels, dualized  [N, Cb, 3]
      p: pseudoscalar channels        [N, Cp]

    In 3D Euclidean/PGA-inspired implementation, bivectors can be represented
    by their dual axial-vector coordinates. This is not yet full PGA algebra,
    but it gives us a stable stepping stone for multivector attention.
    """

    s: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    p: Optional[torch.Tensor] = None

    def to(self, device: torch.device | str) -> "Multivector":
        return Multivector(
            s=None if self.s is None else self.s.to(device),
            v=None if self.v is None else self.v.to(device),
            b=None if self.b is None else self.b.to(device),
            p=None if self.p is None else self.p.to(device),
        )

    @property
    def device(self):
        for x in (self.s, self.v, self.b, self.p):
            if x is not None:
                return x.device
        return None

    @property
    def num_nodes(self) -> int:
        for x in (self.s, self.v, self.b, self.p):
            if x is not None:
                return x.shape[0]
        return 0


def vector_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Dot product over last coordinate dimension.

    a, b: [..., C, 3]
    returns: [..., C]
    """
    return (a * b).sum(dim=-1)


def vector_norm_sq(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vector_dot(a, a).clamp_min(eps)


def safe_normalize_vector(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return a / vector_norm_sq(a, eps=eps).sqrt().unsqueeze(-1)


def scalar_invariant_inner(q: Multivector, k: Multivector) -> torch.Tensor:
    """
    Node-wise scalar invariant inner product.

    This is the basic invariant used for attention scores.
    It contracts matching grades and returns [N] if all channel counts match.
    """

    terms = []

    if q.s is not None and k.s is not None:
        terms.append((q.s * k.s).sum(dim=-1))

    if q.v is not None and k.v is not None:
        terms.append(vector_dot(q.v, k.v).sum(dim=-1))

    if q.b is not None and k.b is not None:
        terms.append(vector_dot(q.b, k.b).sum(dim=-1))

    if q.p is not None and k.p is not None:
        terms.append((q.p * k.p).sum(dim=-1))

    if not terms:
        raise ValueError("No matching grades for invariant inner product.")

    return sum(terms)


def pairwise_scalar_score(q: Multivector, k: Multivector) -> torch.Tensor:
    """
    Dense pairwise invariant score.

    Returns [N, N]. This is for prototypes/small molecules.
    Later we should edge-restrict it for efficiency.
    """

    terms = []

    if q.s is not None and k.s is not None:
        terms.append(q.s @ k.s.T)

    if q.v is not None and k.v is not None:
        # q.v: [N, C, 3], k.v: [M, C, 3] -> [N, M]
        terms.append(torch.einsum("ncd,mcd->nm", q.v, k.v))

    if q.b is not None and k.b is not None:
        terms.append(torch.einsum("ncd,mcd->nm", q.b, k.b))

    if q.p is not None and k.p is not None:
        terms.append(q.p @ k.p.T)

    if not terms:
        raise ValueError("No matching grades for pairwise scalar score.")

    return sum(terms)
