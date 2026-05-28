"""Utility functions for paired synthetic-control fairness-audit experiments."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return expit(np.clip(x, -35, 35))


def logit(p: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logit."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def bh_flags(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg rejection mask."""
    reject, _, _, _ = multipletests(np.asarray(pvals), alpha=alpha, method="fdr_bh")
    return reject.astype(bool)


def fit_logit_irls(
    X: np.ndarray,
    y: np.ndarray,
    *,
    beta0: np.ndarray | None = None,
    max_iter: int = 60,
    tol: float = 1e-8,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit logistic regression by IRLS and return beta, Hessian inverse, info.

    The returned covariance is the model-based Fisher inverse. For cluster-robust
    standard errors, pass beta to :func:`cluster_covariance`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p, dtype=float) if beta0 is None else np.asarray(beta0, dtype=float).copy()
    converged = False
    for it in range(1, max_iter + 1):
        eta = X @ beta
        mu = sigmoid(eta)
        W = np.maximum(mu * (1 - mu), 1e-10)
        H = X.T @ (X * W[:, None])
        H.flat[:: p + 1] += ridge
        g = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    eta = X @ beta
    mu = sigmoid(eta)
    W = np.maximum(mu * (1 - mu), 1e-10)
    H = X.T @ (X * W[:, None])
    H.flat[:: p + 1] += ridge
    try:
        h_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        h_inv = np.linalg.pinv(H)
    return beta, h_inv, {"iterations": it, "converged": converged}


def prepare_cluster(groups) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare cluster indices for sandwich covariance."""
    import pandas as pd

    codes = pd.Categorical(groups).codes.astype(np.int32)
    order = np.argsort(codes)
    sorted_codes = codes[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_codes))]
    return order, starts, len(starts)


def cluster_covariance(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    cluster_info: tuple[np.ndarray, np.ndarray, int],
    h_inv: np.ndarray | None = None,
) -> np.ndarray:
    """Cluster-robust sandwich covariance for a fitted logistic model."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n, p = X.shape
    if h_inv is None:
        _, h_inv, _ = fit_logit_irls(X, y, beta0=beta, max_iter=1)
    order, starts, G = cluster_info
    mu = sigmoid(X @ beta)
    score = X * (y - mu)[:, None]
    sums = np.add.reduceat(score[order], starts, axis=0)
    meat = sums.T @ sums
    cov = h_inv @ meat @ h_inv
    if G > 1 and n > p:
        cov *= (G / (G - 1)) * ((n - 1) / (n - p))
    return cov


def fit_logit_with_covariance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    clusters=None,
    beta0: np.ndarray | None = None,
    max_iter: int = 60,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit logit and return beta, standard errors, info."""
    beta, h_inv, info = fit_logit_irls(X, y, beta0=beta0, max_iter=max_iter, tol=tol)
    if clusters is None:
        cov = h_inv
    else:
        cov = cluster_covariance(X, y, beta, prepare_cluster(clusters), h_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return beta, se, info


def expected_cluster_covariance_independent_bernoulli(
    X: np.ndarray,
    p: np.ndarray,
    clusters=None,
    h_inv: np.ndarray | None = None,
    *,
    finite_correction: bool = True,
) -> np.ndarray:
    """Expected cluster-sandwich covariance under row-independent Bernoulli DGP.

    In exact paired-control runs, cluster-robust SEs are computed on each
    synthetic outcome vector. For the full ACS experiment this is expensive, so
    the asymptotic coefficient-draw mode uses the expected sandwich covariance
    under the synthetic-control data-generating process. Because the DGP draws
    rows independently conditional on X, cross-row score covariances inside a
    cluster are zero; therefore the expected sandwich meat equals X'WX. The
    cluster IDs still enter through the usual finite-cluster correction, keeping
    the reported asymptotic mode aligned with the cluster-robust audit target.
    """
    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float)
    n, k = X.shape
    W = np.maximum(p * (1 - p), 1e-10)
    H = X.T @ (X * W[:, None])
    if h_inv is None:
        try:
            h_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            h_inv = np.linalg.pinv(H)
    cov = h_inv @ H @ h_inv
    if clusters is not None and finite_correction:
        import pandas as pd
        G = int(pd.Series(clusters).nunique())
        if G > 1 and n > k:
            cov *= (G / (G - 1)) * ((n - 1) / (n - k))
    return cov


def pvalues_from_beta(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    return 2 * norm.sf(np.abs(z))


def solve_intercept_for_rate(offset: np.ndarray, target_rate: float) -> float:
    """Find alpha so mean(sigmoid(alpha + offset)) equals target_rate."""
    lo, hi = -30.0, 30.0
    for _ in range(120):
        mid = (lo + hi) / 2
        rate = sigmoid(mid + offset).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
