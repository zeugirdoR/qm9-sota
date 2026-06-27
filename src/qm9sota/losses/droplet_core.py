"""droplet_core.py — the invariant droplet, as a shared, domain-agnostic core.

VENDORED from TOI-droplet/emergence/droplet_core.py — keep in sync. The SAME invariant object powers
LLM-alignment reward-hack amputation (TOI-droplet) and QM9 robust regression (here): one reparametrization-
invariant divergence, a derived chi^2 budget, two domains.

  * closed-form genuine divergence  I_delta = 4(1 - exp(-maha^2/8))   (monotone in Mahalanobis^2),
  * DERIVED compact-support budget  R^2 = chi^2_D(qlevel)             (not tuned),
  * consistency-corrected coupled fixed point -> robust predictive t*=(mu,C) and the compact-support
    weight  w = [1 - maha^2/R^2]_+^(1/nu)  that amputates points outside t*'s support.

Pure NumPy + SciPy; call on any [N, D] matrix of vectors. Weights are a stop-gradient quantity.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def derived_budget(D: int, qlevel: float = 0.999) -> float:
    """Compact-support boundary R^2 = chi^2_D quantile — the DERIVED budget, not a tuned trim."""
    return float(chi2.ppf(qlevel, D))


def idelta_gaussian(maha2):
    """Closed-form genuine well-separated-mode divergence I_delta = 4(1 - exp(-maha^2/8))."""
    return 4.0 * (1.0 - np.exp(-np.asarray(maha2) / 8.0))


def _maha2(x: np.ndarray, mu: np.ndarray, Cinv: np.ndarray) -> np.ndarray:
    d = x - mu
    return np.einsum("id,de,ie->i", d, Cinv, d)


def toi_fixed_point(X, qlevel: float = 0.999, nu: float = 0.5,
                    ridge_frac: float = 1e-3, trim: float = 0.6, iters: int = 20):
    """Consistency-corrected coupled fixed point -> (weights, mu, C, maha2, R2).

    Robust trim warm-start (keep the central `trim` fraction), MCD-style consistency scale correction
    (inlier median maha^2 -> chi^2 median), an inlier refit for the final predictive t*, then the
    compact-support genuine weight w = [1 - maha^2/R^2]_+^(1/nu). `X` is [N, D]; returns weights in
    [0, 1] (exactly 0 beyond the chi^2 support).
    """
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    R2 = derived_budget(D, qlevel)
    rid = ridge_frac * float(np.mean(np.var(X, 0))) + 1e-12
    eye = np.eye(D)

    mu = np.median(X, 0)
    C = np.cov(X.T) + rid * eye
    h = max(int(trim * N), D + 1)
    for _ in range(iters):
        d2 = _maha2(X, mu, np.linalg.inv(C))
        core = X[np.argsort(d2)[:h]]
        mn = core.mean(0)
        Cn = np.cov(core.T) + rid * eye
        if np.linalg.norm(mn - mu) < 1e-7:
            mu, C = mn, Cn
            break
        mu, C = mn, Cn

    med = np.median(_maha2(X, mu, np.linalg.inv(C)))
    C = C * (med / chi2.ppf(0.5, D) + 1e-12)

    inl = _maha2(X, mu, np.linalg.inv(C)) < chi2.ppf(0.975, D)
    if int(inl.sum()) > D:
        mu = X[inl].mean(0)
        C = np.cov(X[inl].T) + rid * eye
        med = np.median(_maha2(X, mu, np.linalg.inv(C)))
        C = C * (med / chi2.ppf(0.5, D) + 1e-12)

    maha2 = _maha2(X, mu, np.linalg.inv(C))
    w = np.clip(1.0 - maha2 / R2, 0.0, None) ** (1.0 / nu)
    return w, mu, C, maha2, R2
