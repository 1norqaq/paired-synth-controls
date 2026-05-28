#!/usr/bin/env python3
"""Glass-ceiling breakpoint sweep with parametric bootstrap.

This reproduces the Section 6 selective-inference code path on a schema-compatible
hiring dataset. The private data are not included; use the synthetic example data for a quick
run and the private file locally for the paper number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paired_synth_controls.design import DesignSpec, make_designs  # noqa: E402
from paired_synth_controls.utils import cluster_covariance, fit_logit_irls, prepare_cluster, sigmoid  # noqa: E402


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def loglik(X, y, beta):
    eta = X @ beta
    return float(np.sum(y * eta - np.logaddexp(0, eta)))


def fit_ll(X, y, beta0=None):
    beta, h_inv, info = fit_logit_irls(X, y, beta0=beta0)
    return beta, h_inv, loglik(X, y, beta), info


def add_step_interactions(X, q, group_cols, tau):
    high = (q >= tau).astype(float)
    inter = X[:, group_cols] * high[:, None]
    return np.column_stack([X, inter])


def sweep_sup_lr(X_null, y, q, group_cols, taus):
    beta_null, _, ll_null, _ = fit_ll(X_null, y)
    best = {"tau": None, "LR": -np.inf, "beta_alt": None, "h_inv_alt": None, "ll_alt": None}
    beta0_alt = None
    for tau in taus:
        X_alt = add_step_interactions(X_null, q, group_cols, tau)
        beta_alt, h_inv_alt, ll_alt, _ = fit_ll(X_alt, y, beta0=beta0_alt)
        beta0_alt = beta_alt
        LR = 2 * (ll_alt - ll_null)
        if LR > best["LR"]:
            best = {"tau": float(tau), "LR": float(LR), "beta_alt": beta_alt, "h_inv_alt": h_inv_alt, "ll_alt": float(ll_alt)}
    return beta_null, ll_null, best


def parse_grid(grid: str):
    vals = [float(x) for x in grid.split(",") if x.strip()]
    return np.array(sorted(set(vals)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_hiring_example.csv")
    parser.add_argument("--config", default="configs/hiring_schema_template.json")
    parser.add_argument("--groups", nargs="+", default=["ethnicity_E2", "ethnicity_E3"], help="Group coefficients whose high-Q interactions are swept")
    parser.add_argument("--tau-grid", default="0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--B-bootstrap", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--out", default="results/glass_ceiling_summary.json")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    spec = DesignSpec(cfg["outcome_col"], cfg["q_col"], cfg.get("cluster_col"), cfg["categorical_cols"], cfg["references"])
    df = read_table(Path(args.data))
    y, q, clusters, orders, codes, X, names, _, _ = make_designs(df, spec)
    name_to_idx = {n: i for i, n in enumerate(names)}
    group_cols = [name_to_idx[g] for g in args.groups if g in name_to_idx]
    if not group_cols:
        raise ValueError(f"None of --groups {args.groups} found in design names")
    taus = parse_grid(args.tau_grid)
    beta_null, ll_null, best = sweep_sup_lr(X, y, q, group_cols, taus)
    p_null = sigmoid(X @ beta_null)
    rng = np.random.default_rng(args.seed)
    boot_lr = np.zeros(args.B_bootstrap)
    for b in range(args.B_bootstrap):
        yb = rng.binomial(1, p_null).astype(float)
        _, _, bbest = sweep_sup_lr(X, yb, q, group_cols, taus)
        boot_lr[b] = bbest["LR"]
    p_boot = float((1 + np.sum(boot_lr >= best["LR"])) / (args.B_bootstrap + 1))

    X_best = add_step_interactions(X, q, group_cols, best["tau"])
    beta_alt = best["beta_alt"]
    cov = cluster_covariance(X_best, y, beta_alt, prepare_cluster(clusters), best["h_inv_alt"]) if clusters is not None else best["h_inv_alt"]
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    regime = {}
    offset = X.shape[1]
    for k, g in enumerate(args.groups[: len(group_cols)]):
        idx = group_cols[k]
        int_idx = offset + k
        below = beta_alt[idx]
        above = beta_alt[idx] + beta_alt[int_idx]
        se_above = np.sqrt(max(cov[idx, idx] + cov[int_idx, int_idx] + 2 * cov[idx, int_idx], 0))
        regime[g] = {
            "OR_below": float(np.exp(below)),
            "CI_below": [float(np.exp(below - 1.96 * se[idx])), float(np.exp(below + 1.96 * se[idx]))],
            "OR_above": float(np.exp(above)),
            "CI_above": [float(np.exp(above - 1.96 * se_above)), float(np.exp(above + 1.96 * se_above))],
            "interaction_p": float(2 * norm.sf(abs(beta_alt[int_idx] / se[int_idx]))) if se[int_idx] > 0 else None,
        }
    summary = {
        "data": {"n": int(len(df)), "positives": int(y.sum()), "positive_rate": float(y.mean()), "clusters": int(pd.Series(clusters).nunique()) if clusters is not None else None},
        "method_note": "Breakpoint sweep over high-Q group interactions; parametric bootstrap of supremum LR under the single-regime null.",
        "groups": args.groups,
        "tau_grid": taus.tolist(),
        "observed_sup_LR": float(best["LR"]),
        "selected_tau": float(best["tau"]),
        "bootstrap_B": int(args.B_bootstrap),
        "bootstrap_p": p_boot,
        "bootstrap_mean": float(boot_lr.mean()),
        "bootstrap_q95": float(np.quantile(boot_lr, 0.95)),
        "bootstrap_max": float(boot_lr.max()),
        "regime_ORs": regime,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
