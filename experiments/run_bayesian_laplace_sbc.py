#!/usr/bin/env python3
"""Bayesian/Laplace positive-control sanity check.

This is not full MCMC. It fits a Gaussian-prior logistic MAP estimator and uses
Laplace posterior intervals, then compares coverage to cluster-robust
frequentist intervals on the same injected synthetic outcomes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paired_synth_controls.design import DesignSpec, make_designs
from paired_synth_controls.utils import cluster_covariance, fit_logit_irls, prepare_cluster, sigmoid, solve_intercept_for_rate


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def fit_map_laplace(X, y, prior_sd, beta0=None, max_iter=60, tol=1e-8):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    prior_prec = 1.0 / (np.asarray(prior_sd, dtype=float) ** 2)
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    converged = False
    for it in range(1, max_iter + 1):
        eta = X @ beta
        mu = sigmoid(eta)
        W = np.maximum(mu * (1 - mu), 1e-10)
        H = X.T @ (X * W[:, None])
        H.flat[:: p + 1] += prior_prec
        rhs = X.T @ (W * (eta + (y - mu) / W))
        try:
            beta_new = np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(H, rhs, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    eta = X @ beta
    mu = sigmoid(eta)
    W = np.maximum(mu * (1 - mu), 1e-10)
    H = X.T @ (X * W[:, None])
    H.flat[:: p + 1] += prior_prec
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return beta, se, {"iterations": it, "converged": converged}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_hiring_example.csv")
    parser.add_argument("--config", default="configs/hiring_schema_template.json")
    parser.add_argument("--B", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--out", default="results/bayesian_laplace_summary.json")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    spec = DesignSpec(
        outcome_col=cfg["outcome_col"],
        q_col=cfg["q_col"],
        cluster_col=cfg.get("cluster_col"),
        categorical_cols=cfg["categorical_cols"],
        references=cfg["references"],
    )
    df = read_table(Path(args.data))
    y, q, clusters, orders, codes, X, names, _, _ = make_designs(df, spec)
    name_to_idx = {n: i for i, n in enumerate(names)}
    injections = cfg["injections"]
    injected_names = list(injections.keys())
    injected_idx = [name_to_idx[n] for n in injected_names]
    theta = np.zeros(len(names))
    for n, v in injections.items():
        theta[name_to_idx[n]] = float(v)

    beta_base, h_inv_base, _ = fit_logit_irls(X, y)
    alpha_idx = name_to_idx["const"]
    q_idx = name_to_idx["Q"]
    offset = beta_base[q_idx] * q + X[:, 2:] @ theta[2:]
    alpha_star = solve_intercept_for_rate(offset, y.mean())
    eta = alpha_star + offset
    p_inject = sigmoid(eta)
    cluster_info = prepare_cluster(clusters) if clusters is not None else None
    rng = np.random.default_rng(args.seed)

    prior_sd = np.full(len(names), 2.5)
    prior_sd[alpha_idx] = 5.0
    rows_bayes = []
    rows_freq = []
    cdf_values = []
    beta0_map = None
    beta0_freq = None
    for b in range(1, args.B + 1):
        ysyn = rng.binomial(1, p_inject).astype(float)
        bm, sem, _ = fit_map_laplace(X, ysyn, prior_sd, beta0=beta0_map)
        beta0_map = bm
        bf, h_inv, _ = fit_logit_irls(X, ysyn, beta0=beta0_freq)
        beta0_freq = bf
        covf = cluster_covariance(X, ysyn, bf, cluster_info, h_inv) if cluster_info is not None else h_inv
        sef = np.sqrt(np.maximum(np.diag(covf), 0))
        for n, idx in zip(injected_names, injected_idx):
            true = injections[n]
            for method, beta, se, store in [("Bayes_Laplace", bm, sem, rows_bayes), ("Frequentist_cluster", bf, sef, rows_freq)]:
                lo = beta[idx] - 1.96 * se[idx]
                hi = beta[idx] + 1.96 * se[idx]
                store.append({
                    "draw": b,
                    "coef": n,
                    "theta": float(true),
                    "target_OR": float(np.exp(true)),
                    "OR_hat": float(np.exp(beta[idx])),
                    "covered": bool(lo <= true <= hi),
                    "method": method,
                })
            cdf_values.append(float(norm.cdf((true - bm[idx]) / sem[idx])))

    bayes = pd.DataFrame(rows_bayes)
    freq = pd.DataFrame(rows_freq)
    def per_draw_mae(df_):
        return df_.assign(abs_err=lambda d: np.abs(d.OR_hat - d.target_OR)).groupby("draw").abs_err.mean()
    bmae = per_draw_mae(bayes)
    fmae = per_draw_mae(freq)
    cov_b = bayes.groupby("coef").covered.mean()
    cov_f = freq.groupby("coef").covered.mean()
    ks_stat, ks_p = kstest(np.asarray(cdf_values), "uniform")
    summary = {
        "data": {"n": int(len(df)), "positives": int(y.sum()), "positive_rate": float(y.mean()), "clusters": int(pd.Series(clusters).nunique()) if clusters is not None else None},
        "B": args.B,
        "method_note": "Gaussian-prior logistic regression with Laplace/MAP posterior approximation; not full MCMC.",
        "injections": injections,
        "bayes_laplace": {
            "mean_coverage": float(bayes.covered.mean()),
            "MAE_OR_mean": float(bmae.mean()),
            "MAE_OR_MC95": [float(np.quantile(bmae, 0.025)), float(np.quantile(bmae, 0.975))],
            "coverage_by_group": {k: float(v) for k, v in cov_b.items()},
        },
        "frequentist_cluster": {
            "mean_coverage": float(freq.covered.mean()),
            "MAE_OR_mean": float(fmae.mean()),
            "MAE_OR_MC95": [float(np.quantile(fmae, 0.025)), float(np.quantile(fmae, 0.975))],
            "coverage_by_group": {k: float(v) for k, v in cov_f.items()},
        },
        "coverage_max_abs_difference_pp": float((cov_b - cov_f).abs().max() * 100),
        "posterior_cdf_uniformity_KS_stat": float(ks_stat),
        "posterior_cdf_uniformity_KS_p": float(ks_p),
        "alpha_star": float(alpha_star),
        "base_gamma_Q": float(beta_base[q_idx]),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
