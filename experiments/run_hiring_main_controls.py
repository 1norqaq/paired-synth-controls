#!/usr/bin/env python3
"""Private hiring main paired-control and omitted-Q diagnostic experiment.

This script implements the analyses reported in Tables 4--5 of the paper for a
schema-compatible hiring dataset. The proprietary row-level data are not in the
repository; the script can be checked on ``data/synthetic_hiring_example.csv``
and rerun by the authors on the private file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paired_synth_controls.design import DesignSpec, design_from_codes, make_designs  # noqa: E402
from paired_synth_controls.utils import (  # noqa: E402
    bh_flags,
    cluster_covariance,
    fit_logit_irls,
    prepare_cluster,
    sigmoid,
    solve_intercept_for_rate,
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def make_spec(cfg: dict) -> DesignSpec:
    return DesignSpec(
        outcome_col=cfg["outcome_col"],
        q_col=cfg["q_col"],
        cluster_col=cfg.get("cluster_col"),
        categorical_cols=cfg["categorical_cols"],
        references=cfg["references"],
    )


def fit_cov(X, y, clusters=None, beta0=None, use_cluster=True):
    beta, h_inv, info = fit_logit_irls(X, y, beta0=beta0)
    if clusters is not None and use_cluster:
        cov = cluster_covariance(X, y, beta, prepare_cluster(clusters), h_inv)
    else:
        cov = h_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return beta, se, info


def pvals_flags(beta, se, start, alpha):
    pvals = 2 * norm.sf(np.abs(beta[start:] / se[start:]))
    flags = bh_flags(pvals, alpha=alpha)
    return pvals, flags


def summarize_flags(S, flag_mat, names):
    rates = flag_mat.mean(axis=0) if len(flag_mat) else np.zeros(len(names))
    top = sorted(zip(names, rates), key=lambda x: x[1], reverse=True)
    if len(top) == 0 or top[0][1] < 0.10:
        most = "— (diffuse)"
    else:
        most = ", ".join(f"{n} ({100*r:.1f}%)" for n, r in top if r >= max(0.10, top[0][1] - 1e-12))
    return {
        "mean_S": float(np.mean(S)) if len(S) else 0.0,
        "fail_rate": float(np.mean(np.asarray(S) >= 1)) if len(S) else 0.0,
        "P_S_eq_0": float(np.mean(np.asarray(S) == 0)) if len(S) else 0.0,
        "max_S": int(np.max(S)) if len(S) else 0,
        "flag_rates": {names[i]: float(rates[i]) for i in range(len(names))},
        "most_flagged": most,
    }


def make_coarse_design(codes, q, orders, categorical_cols):
    q_bin = (q >= np.median(q)).astype(float)
    parts = [np.ones(len(q)), q_bin]
    names = ["const", "Q_median_split"]
    for col in categorical_cols:
        cats = orders[col]
        c = codes[col]
        D = (c[:, None] == np.arange(1, len(cats), dtype=np.int16)[None, :]).astype(float)
        parts.append(D)
        names.extend([f"{col}_{cat}" for cat in cats[1:]])
    return np.column_stack(parts), names


def run_table3(y, q, clusters, X_good, names_good, cfg, B, seed, alpha):
    rng = np.random.default_rng(seed)
    beta_base, _, _ = fit_logit_irls(X_good, y)
    beta_clean = beta_base.copy()
    beta_clean[2:] = 0.0
    p_clean = sigmoid(X_good @ beta_clean)

    injections = cfg.get("injections", {})
    name_to_idx = {n: i for i, n in enumerate(names_good)}
    theta_full = np.zeros(len(names_good))
    injected = []
    for name, val in injections.items():
        if name in name_to_idx:
            theta_full[name_to_idx[name]] = float(val)
            injected.append(name)
    offset = beta_base[1] * q + X_good[:, 2:] @ theta_full[2:]
    alpha_star = solve_intercept_for_rate(offset, float(y.mean()))
    beta_inj = np.zeros_like(beta_base)
    beta_inj[0] = alpha_star
    beta_inj[1] = beta_base[1]
    beta_inj[2:] = theta_full[2:]
    p_inj = sigmoid(X_good @ beta_inj)

    test_names = names_good[2:]
    S_neg = np.zeros(B, dtype=int)
    flags_neg = np.zeros((B, len(test_names)), dtype=bool)
    pos_rows = []
    beta0_neg = None
    beta0_pos = None
    for b in range(1, B + 1):
        yc = rng.binomial(1, p_clean).astype(float)
        bn, sen, _ = fit_cov(X_good, yc, clusters, beta0=beta0_neg, use_cluster=True)
        beta0_neg = bn
        _, fg = pvals_flags(bn, sen, 2, alpha)
        flags_neg[b - 1] = fg
        S_neg[b - 1] = int(fg.sum())

        yi = rng.binomial(1, p_inj).astype(float)
        bp, sep, _ = fit_cov(X_good, yi, clusters, beta0=beta0_pos, use_cluster=True)
        beta0_pos = bp
        pvals, fpos = pvals_flags(bp, sep, 2, alpha)
        for j, name in enumerate(test_names):
            true = theta_full[j + 2]
            lo = bp[j + 2] - 1.96 * sep[j + 2]
            hi = bp[j + 2] + 1.96 * sep[j + 2]
            pos_rows.append({
                "draw": b,
                "coef": name,
                "theta": float(true),
                "target_OR": float(math.exp(true)),
                "OR_hat": float(math.exp(bp[j + 2])),
                "covered": bool(lo <= true <= hi),
                "flag": bool(fpos[j]),
                "p_value": float(pvals[j]),
            })
    pos = pd.DataFrame(pos_rows)
    inj_names = [n for n in injected if n in test_names]
    null_names = [n for n in test_names if abs(theta_full[name_to_idx[n]]) < 1e-12]
    if inj_names:
        mae_by_draw = pos[pos.coef.isin(inj_names)].assign(abs_err=lambda d: np.abs(d.OR_hat - d.target_OR)).groupby("draw").abs_err.mean()
    else:
        mae_by_draw = pd.Series([np.nan])
    return {
        "negative_control": summarize_flags(S_neg, flags_neg, test_names),
        "positive_control": {
            "injected_coefficients": inj_names,
            "coverage_by_group": {k: float(v) for k, v in pos[pos.coef.isin(inj_names)].groupby("coef").covered.mean().items()},
            "mean_coverage_injected": float(pos[pos.coef.isin(inj_names)].covered.mean()) if inj_names else None,
            "mean_recovered_OR": {k: float(v) for k, v in pos[pos.coef.isin(inj_names)].groupby("coef").OR_hat.mean().items()},
            "MAE_OR_mean": float(mae_by_draw.mean()) if inj_names else None,
            "MAE_OR_MC95": [float(np.quantile(mae_by_draw, 0.025)), float(np.quantile(mae_by_draw, 0.975))] if inj_names else None,
            "null_false_flags_per_draw_mean": float(pos[pos.coef.isin(null_names)].groupby("draw").flag.sum().mean()) if null_names else None,
            "null_false_flags_P0": float((pos[pos.coef.isin(null_names)].groupby("draw").flag.sum() == 0).mean()) if null_names else None,
        },
        "positive_draws": pos_rows,
        "beta_base": beta_base.tolist(),
        "p_clean": p_clean,
    }


def run_table4_diagnostic(y, clusters, X_good, names_good, X_noq, names_noq, X_coarse, names_coarse, p_clean, B, seed, alpha):
    rng = np.random.default_rng(seed)
    pipelines = {
        "A_good": (X_good, names_good, 2, True),
        "A_iid": (X_good, names_good, 2, False),
        "A_coarseQ": (X_coarse, names_coarse, 2, True),
        "A_noQ": (X_noq, names_noq, 1, True),
    }
    S = {k: np.zeros(B, dtype=int) for k in pipelines}
    flag_mats = {k: np.zeros((B, len(v[1]) - v[2]), dtype=bool) for k, v in pipelines.items()}
    beta0 = {k: None for k in pipelines}
    rows = []
    for b in range(1, B + 1):
        yb = rng.binomial(1, p_clean).astype(float)
        for pname, (X, names, start, use_cluster) in pipelines.items():
            beta, se, _ = fit_cov(X, yb, clusters, beta0=beta0[pname], use_cluster=use_cluster)
            beta0[pname] = beta
            pvals, flags = pvals_flags(beta, se, start, alpha)
            S[pname][b - 1] = int(flags.sum())
            flag_mats[pname][b - 1] = flags
            rows.append({"draw": b, "pipeline": pname, "S": int(flags.sum()), "flags": ";".join([names[start + i] for i, f in enumerate(flags) if f])})
    return {k: summarize_flags(S[k], flag_mats[k], pipelines[k][1][pipelines[k][2]:]) for k in pipelines}, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_hiring_example.csv")
    parser.add_argument("--config", default="configs/hiring_schema_template.json")
    parser.add_argument("--B-controls", type=int, default=400)
    parser.add_argument("--B-diagnostic", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="results/hiring_main_controls_summary.json")
    parser.add_argument("--draws-dir", default="")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    spec = make_spec(cfg)
    df = read_table(Path(args.data))
    y, q, clusters, orders, codes, X_good, names_good, X_noq, names_noq = make_designs(df, spec)
    X_coarse, names_coarse = make_coarse_design(codes, q, orders, spec.categorical_cols)

    controls = run_table3(y, q, clusters, X_good, names_good, cfg, args.B_controls, args.seed, args.alpha)
    diagnostic, diag_rows = run_table4_diagnostic(y, clusters, X_good, names_good, X_noq, names_noq, X_coarse, names_coarse, np.asarray(controls.pop("p_clean")), args.B_diagnostic, args.seed + 1, args.alpha)
    summary = {
        "data": {"n": int(len(df)), "positives": int(y.sum()), "positive_rate": float(y.mean()), "clusters": int(pd.Series(clusters).nunique()) if clusters is not None else None},
        "B_controls": args.B_controls,
        "B_diagnostic": args.B_diagnostic,
        "method_note": "Exact row-wise Bernoulli paired controls with cluster-robust logit audits; diagnostic includes A_good, A_iid, A_coarseQ, and A_noQ.",
        "controls": {k: v for k, v in controls.items() if k not in {"positive_draws", "beta_base"}},
        "diagnostic": diagnostic,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    if args.draws_dir:
        d = Path(args.draws_dir)
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(controls["positive_draws"]).to_csv(d / "hiring_main_positive_draws.csv", index=False)
        pd.DataFrame(diag_rows).to_csv(d / "hiring_main_diagnostic_draws.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
