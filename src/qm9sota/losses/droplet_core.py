"""droplet_core.py — the invariant droplet, as a shared, domain-agnostic core.

VENDORED from TOI-droplet/emergence/droplet_core.py — keep in sync. The SAME invariant object powers
LLM-alignment reward-hack amputation (TOI-droplet) and QM9 robust regression (here): one reparametrization-
invariant divergence, a derived budget, two domains.

  * closed-form genuine divergence  I_delta = 4(1 - exp(-maha^2/8))   (monotone in Mahalanobis^2),
  * DERIVED compact-support budget  R^2 = chi^2_D(qlevel) (Gaussian) OR R^2 = D*F_{D,nu}(qlevel)
    (heavy-tail / Student-t -- the MAXRL_SCORE_EXPERIMENT.md sec 5b correction),
  * consistency-corrected coupled fixed point -> robust predictive t*=(mu,C) and the compact-support
    weight  w = [1 - maha^2/R^2]_+^(1/nu)  that amputates points outside t*'s support.

Pure NumPy + SciPy; call on any [N, D] matrix of vectors. Weights are a stop-gradient quantity.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, f as fdist
from scipy.special import digamma
from scipy.optimize import brentq


def derived_budget(D: int, qlevel: float = 0.999) -> float:
    """Gaussian compact-support boundary R^2 = chi^2_D quantile — the DERIVED budget, not a tuned trim."""
    return float(chi2.ppf(qlevel, D))


def idelta_gaussian(maha2):
    """Closed-form genuine well-separated-mode divergence I_delta = 4(1 - exp(-maha^2/8))."""
    return 4.0 * (1.0 - np.exp(-np.asarray(maha2) / 8.0))


def fit_nu(maha2, D: int, numin: float = 2.0, numax: float = 60.0, iters: int = 60) -> float:
    """EM fit of the multivariate-Student-t dof nu from (inlier) Mahalanobis^2 (heavy tail -> small nu)."""
    m2 = np.asarray(maha2, dtype=np.float64)
    if m2.size < D + 2:
        return numax
    nu = 8.0
    for _ in range(iters):
        u = (nu + D) / (nu + m2)
        c = 1.0 + np.mean(np.log(u) - u) + digamma((nu + D) / 2.0) - np.log((nu + D) / 2.0)
        g = lambda v: np.log(v / 2.0) - digamma(v / 2.0) + c
        try:
            nu = float(np.clip(brentq(g, 1e-2, 1e8), numin, numax))
        except Exception:
            nu = numax
    return nu


def student_t_budget(D: int, nu: float, qlevel: float = 0.999) -> float:
    """Heavy-tail boundary R^2 = D * F_{D,nu}(qlevel): the Mahalanobis^2 quantile of a multivariate-t."""
    return float(D * fdist.ppf(qlevel, D, nu))


def _maha2(x: np.ndarray, mu: np.ndarray, Cinv: np.ndarray) -> np.ndarray:
    d = x - mu
    return np.einsum("id,de,ie->i", d, Cinv, d)


def toi_fixed_point(X, qlevel: float = 0.999, nu: float = 0.5, budget: str = "gaussian",
                    ridge_frac: float = 1e-3, trim: float = 0.6, iters: int = 20):
    """Consistency-corrected coupled fixed point -> (weights, mu, C, maha2, R2, nu_fit).

    Robust trim warm-start, MCD-style consistency scale correction, inlier refit -> predictive t*, then
    w = [1 - maha^2/R^2]_+^(1/nu). Budget R^2 is Gaussian chi^2_D (default) or the heavy-tail
    multivariate-t quantile (`budget="student_t"`, fitting nu on the genuine bulk). `nu_fit` = fitted dof.
    """
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
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

    nu_fit = float("inf")
    if budget == "student_t":
        order = np.argsort(maha2)
        keep = order[: max(D + 2, int(0.7 * N))]   # genuine bulk (drops the largest = gross outliers)
        nu_fit = fit_nu(maha2[keep], D, numin=3.0)  # floor nu>=3: F_{D,2} has an infinite-variance tail -> budget blows up
        R2 = student_t_budget(D, nu_fit, qlevel)
    else:
        R2 = derived_budget(D, qlevel)

    w = np.clip(1.0 - maha2 / R2, 0.0, None) ** (1.0 / nu)
    return w, mu, C, maha2, R2, nu_fit
