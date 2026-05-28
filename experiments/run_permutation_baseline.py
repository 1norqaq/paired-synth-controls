#!/usr/bin/env python3
"""Within-Q-decile permutation negative-control baseline.

This script is data-schema driven. It is safe to publish because the proprietary
hiring data are not included; use --data data/synthetic_hiring_example.csv for a
check, or a private local file with the same columns/config for reproduction.
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
from paired_synth_controls.design import DesignSpec, make_designs, design_from_codes
from paired_synth_controls.utils import bh_flags, fit_logit_irls, cluster_covariance, prepare_cluster


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def audit(beta, se, names, include_q: bool, alpha: float):
    start = 2 if include_q else 1
    coefs = names[start:]
    pvals = 2 * norm.sf(np.abs(beta[start:] / se[start:]))
    flags = bh_flags(pvals, alpha=alpha)
    return [coefs[i] for i, f in enumerate(flags) if f]


def permute_codes_within_deciles(codes, q_decile, rng, mode: str = "tuple"):
    """Permute sensitive attributes within Q-deciles.

    ``mode=tuple`` keeps the joint distribution of the sensitive-attribute tuple
    A intact by shuffling rows of the full tuple. This matches the paper text.
    ``mode=separate`` is retained only as a sensitivity option; it shuffles each
    categorical column independently and therefore breaks cross-attribute
    dependence.
    """
    out = {k: v.copy() for k, v in codes.items()}
    for d in np.unique(q_decile):
        idx = np.flatnonzero(q_decile == d)
        if mode == "tuple":
            src = rng.permutation(idx)
            for col, vals in codes.items():
                out[col][idx] = vals[src]
        elif mode == "separate":
            for col, vals in out.items():
                out[col][idx] = rng.permutation(vals[idx])
        else:
            raise ValueError("mode must be 'tuple' or 'separate'")
    return out


def fit_cluster(X, y, clusters, beta0=None):
    beta, h_inv, info = fit_logit_irls(X, y, beta0=beta0)
    cov = cluster_covariance(X, y, beta, prepare_cluster(clusters), h_inv) if clusters is not None else h_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return beta, se, info


def summarize(draws, flag_counts, B, label_map):
    out = {}
    df = pd.DataFrame(draws)
    for pipeline in ["A_good", "A_noQ"]:
        sub = df[df.pipeline == pipeline]
        top = sorted(flag_counts[pipeline].items(), key=lambda kv: kv[1], reverse=True)
        out[pipeline] = {
            "mean_S": float(sub.S.mean()),
            "fail_rate": float((sub.S >= 1).mean()),
            "P_S_eq_0": float((sub.S == 0).mean()),
            "max_S": int(sub.S.max()),
            "top_flagged": [
                {"coef": k, "label": label_map.get(k, k), "count": int(v), "rate": float(v / B)}
                for k, v in top if v > 0
            ][:20],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_hiring_example.csv")
    parser.add_argument("--config", default="configs/hiring_schema_template.json")
    parser.add_argument("--B", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="results/permutation_summary.json")
    parser.add_argument("--permute-mode", choices=["tuple", "separate"], default="tuple", help="tuple preserves joint A distribution; separate is a sensitivity option")
    parser.add_argument("--draws-out", default="", help="Optional CSV path for per-draw outputs")
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
    y, q, clusters, orders, codes, _, names_good, _, names_noq = make_designs(df, spec)
    q_decile = np.asarray(pd.qcut(q, 10, labels=False, duplicates="drop"), dtype=int)
    rng = np.random.default_rng(args.seed)
    label_map = cfg.get("label_map", {})

    flag_counts = {"A_good": {n: 0 for n in names_good[2:]}, "A_noQ": {n: 0 for n in names_noq[1:]}}
    draws = []
    beta0_good = None
    beta0_noq = None
    for b in range(1, args.B + 1):
        pcodes = permute_codes_within_deciles(codes, q_decile, rng, mode=args.permute_mode)
        Xg, ng = design_from_codes(pcodes, q, orders, spec.categorical_cols, include_q=True)
        Xn, nn = design_from_codes(pcodes, q, orders, spec.categorical_cols, include_q=False)
        bg, seg, _ = fit_cluster(Xg, y, clusters, beta0=beta0_good)
        beta0_good = bg
        fg = audit(bg, seg, ng, include_q=True, alpha=args.alpha)
        bn, sen, _ = fit_cluster(Xn, y, clusters, beta0=beta0_noq)
        beta0_noq = bn
        fn = audit(bn, sen, nn, include_q=False, alpha=args.alpha)
        for f in fg:
            flag_counts["A_good"][f] += 1
        for f in fn:
            flag_counts["A_noQ"][f] += 1
        draws.append({"draw": b, "pipeline": "A_good", "S": len(fg), "flags": ";".join(fg)})
        draws.append({"draw": b, "pipeline": "A_noQ", "S": len(fn), "flags": ";".join(fn)})

    summary = {
        "data": {"n": int(len(df)), "positives": int(y.sum()), "positive_rate": float(y.mean()), "clusters": int(pd.Series(clusters).nunique()) if clusters is not None else None},
        "B": args.B,
        "method": f"within-Q-decile {args.permute_mode}-level permutation of sensitive attributes; audit refit on historical outcome",
        **summarize(draws, flag_counts, args.B, label_map),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    if args.draws_out:
        dpath = Path(args.draws_out)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(draws).to_csv(dpath, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
