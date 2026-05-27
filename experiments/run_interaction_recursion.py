#!/usr/bin/env python3
"""Interaction-aware control check for the working-model recursion limitation.

The script constructs a non-linear group-by-Q truth (bias only above a threshold)
and compares a main-effects audit to an interaction-aware audit. It supports the
paper's Section 8 limitation that main-effect controls do not prove the functional
family is correct, while interaction-aware controls recover regime-specific truth.
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


def fit_se(X, y, clusters=None, beta0=None):
    beta, h_inv, _ = fit_logit_irls(X, y, beta0=beta0)
    cov = cluster_covariance(X, y, beta, prepare_cluster(clusters), h_inv) if clusters is not None else h_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return beta, se


def add_group_high_interaction(X, q, group_col, threshold):
    high = (q >= threshold).astype(float)
    inter = X[:, group_col] * high
    return np.column_stack([X, inter])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_hiring_example.csv")
    parser.add_argument("--config", default="configs/hiring_schema_template.json")
    parser.add_argument("--group", default="ethnicity_E3")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--theta-high", type=float, default=-0.9)
    parser.add_argument("--B", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--out", default="results/interaction_recursion_summary.json")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    spec = DesignSpec(cfg["outcome_col"], cfg["q_col"], cfg.get("cluster_col"), cfg["categorical_cols"], cfg["references"])
    df = read_table(Path(args.data))
    y, q, clusters, orders, codes, X, names, _, _ = make_designs(df, spec)
    name_to_idx = {n: i for i, n in enumerate(names)}
    if args.group not in name_to_idx:
        raise ValueError(f"Group {args.group!r} not found in design columns")
    gidx = name_to_idx[args.group]
    beta_base, _, _ = fit_logit_irls(X, y)
    truth = np.zeros(len(y))
    truth[(X[:, gidx] > 0) & (q >= args.threshold)] = args.theta_high
    offset = beta_base[1] * q + truth
    alpha = solve_intercept_for_rate(offset, float(y.mean()))
    p = sigmoid(alpha + offset)
    X_int = add_group_high_interaction(X, q, gidx, args.threshold)
    int_idx = X.shape[1]

    rng = np.random.default_rng(args.seed)
    main_or = []
    main_covers_low = []
    main_covers_high = []
    int_covers_low = []
    int_covers_high = []
    int_covers_param = []
    beta0_main = None
    beta0_int = None
    for _ in range(args.B):
        yb = rng.binomial(1, p).astype(float)
        bm, sem = fit_se(X, yb, clusters, beta0_main)
        beta0_main = bm
        lo_m = bm[gidx] - 1.96 * sem[gidx]
        hi_m = bm[gidx] + 1.96 * sem[gidx]
        main_or.append(float(np.exp(bm[gidx])))
        main_covers_low.append(bool(lo_m <= 0.0 <= hi_m))
        main_covers_high.append(bool(lo_m <= args.theta_high <= hi_m))

        bi, sei = fit_se(X_int, yb, clusters, beta0_int)
        beta0_int = bi
        low = bi[gidx]
        high = bi[gidx] + bi[int_idx]
        # Delta-method SE for high-regime sum.
        _, h_inv, _ = fit_logit_irls(X_int, yb, beta0=bi, max_iter=1)
        cov = cluster_covariance(X_int, yb, bi, prepare_cluster(clusters), h_inv) if clusters is not None else h_inv
        se_high = np.sqrt(max(cov[gidx, gidx] + cov[int_idx, int_idx] + 2 * cov[gidx, int_idx], 0))
        int_covers_low.append(bool(low - 1.96 * sei[gidx] <= 0.0 <= low + 1.96 * sei[gidx]))
        int_covers_high.append(bool(high - 1.96 * se_high <= args.theta_high <= high + 1.96 * se_high))
        int_covers_param.append(bool(bi[int_idx] - 1.96 * sei[int_idx] <= args.theta_high <= bi[int_idx] + 1.96 * sei[int_idx]))

    summary = {
        "data": {"n": int(len(df)), "positives": int(y.sum()), "positive_rate": float(y.mean()), "clusters": int(pd.Series(clusters).nunique()) if clusters is not None else None},
        "B": args.B,
        "group": args.group,
        "threshold": args.threshold,
        "truth_low_log_OR": 0.0,
        "truth_high_log_OR": args.theta_high,
        "truth_high_OR": float(np.exp(args.theta_high)),
        "main_effect_audit": {
            "mean_OR": float(np.mean(main_or)),
            "coverage_low_truth": float(np.mean(main_covers_low)),
            "coverage_high_truth": float(np.mean(main_covers_high)),
        },
        "interaction_aware_audit": {
            "coverage_low_truth": float(np.mean(int_covers_low)),
            "coverage_high_truth": float(np.mean(int_covers_high)),
            "coverage_interaction_parameter": float(np.mean(int_covers_param)),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
