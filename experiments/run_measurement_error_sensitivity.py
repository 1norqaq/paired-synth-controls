#!/usr/bin/env python3
"""Sensitive-attribute measurement-error sensitivity experiment.

Inject a known disparity, corrupt one sensitive attribute at controlled rates,
and refit the audit using the corrupted attribute. This reproduces the Section 8
measurement-error blind-spot code path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paired_synth_controls.design import DesignSpec, make_designs  # noqa: E402
from paired_synth_controls.utils import cluster_covariance, fit_logit_irls, prepare_cluster, sigmoid, solve_intercept_for_rate  # noqa: E402


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def fit_or_for_coef(X, y, clusters, coef_idx):
    beta, h_inv, _ = fit_logit_irls(X, y)
    cov = cluster_covariance(X, y, beta, prepare_cluster(clusters), h_inv) if clusters is not None else h_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return float(np.exp(beta[coef_idx])), [float(np.exp(beta[coef_idx] - 1.96 * se[coef_idx])), float(np.exp(beta[coef_idx] + 1.96 * se[coef_idx]))]


def corrupt_attribute(df, col, rate, rng):
    out = df.copy()
    vals = sorted(out[col].dropna().astype(str).unique().tolist())
    arr = out[col].astype(str).to_numpy().copy()
    mask = rng.random(len(arr)) < rate
    for i in np.flatnonzero(mask):
        choices = [v for v in vals if v != arr[i]]
        if choices:
            arr[i] = rng.choice(choices)
    out[col] = arr
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_hiring_example.csv")
    parser.add_argument("--config", default="configs/hiring_schema_template.json")
    parser.add_argument("--attribute", default="ethnicity")
    parser.add_argument("--group-coef", default="ethnicity_E2")
    parser.add_argument("--theta", type=float, default=-0.6931471805599453, help="Injected log-OR; default OR=0.5")
    parser.add_argument("--rates", default="0,0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--out", default="results/measurement_error_summary.json")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    spec = DesignSpec(cfg["outcome_col"], cfg["q_col"], cfg.get("cluster_col"), cfg["categorical_cols"], cfg["references"])
    df = read_table(Path(args.data))
    y, q, clusters, orders, codes, X, names, _, _ = make_designs(df, spec)
    name_to_idx = {n: i for i, n in enumerate(names)}
    if args.group_coef not in name_to_idx:
        raise ValueError(f"{args.group_coef!r} not found in design")
    gidx = name_to_idx[args.group_coef]
    beta_base, _, _ = fit_logit_irls(X, y)
    theta = np.zeros(len(names))
    theta[gidx] = args.theta
    offset = beta_base[1] * q + X[:, 2:] @ theta[2:]
    alpha = solve_intercept_for_rate(offset, float(y.mean()))
    p = sigmoid(alpha + offset)
    rng = np.random.default_rng(args.seed)
    ysyn = rng.binomial(1, p).astype(float)

    rows = []
    for rate in [float(x) for x in args.rates.split(",") if x.strip()]:
        cdf = corrupt_attribute(df, args.attribute, rate, rng) if rate > 0 else df.copy()
        _, _, cclusters, _, _, Xc, names_c, _, _ = make_designs(cdf, spec)
        idx = {n: i for i, n in enumerate(names_c)}[args.group_coef]
        OR, CI = fit_or_for_coef(Xc, ysyn, cclusters, idx)
        rows.append({"misclassification_rate": rate, "target_OR": float(np.exp(args.theta)), "recovered_OR": OR, "CI": CI})
    summary = {
        "data": {"n": int(len(df)), "positives": int(y.sum()), "positive_rate": float(y.mean()), "clusters": int(pd.Series(clusters).nunique()) if clusters is not None else None},
        "attribute": args.attribute,
        "group_coef": args.group_coef,
        "target_OR": float(np.exp(args.theta)),
        "rows": rows,
        "method_note": "Non-differential random corruption of one sensitive attribute followed by the same cluster-robust audit.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
