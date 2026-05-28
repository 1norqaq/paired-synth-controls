#!/usr/bin/env python3
"""Public ACS/Folktables replication.

This script can run either:

* ``--method asymptotic``: the fast full-sample coefficient-draw approximation
  used for the reported paper numbers on ACS CA 2018 5-year; or
* ``--method exact``: row-wise Bernoulli synthetic-outcome draws followed by
  refitting the audit model on each draw. Exact mode is slower, so the README
  recommends using a small ``--B-*`` or ``--subsample`` for verification.

The script can consume either a preprocessed parquet/CSV with columns

    INCOME_GT_50K, Q, race_group, race_label, PUMA

or build such a table from Folktables/Census if --download is passed.
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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paired_synth_controls.utils import (  # noqa: E402
    bh_flags,
    cluster_covariance,
    expected_cluster_covariance_independent_bernoulli,
    fit_logit_irls,
    logit,
    pvalues_from_beta,
    prepare_cluster,
    sigmoid,
    solve_intercept_for_rate,
)

RACE_LABELS = ["NH-White", "NH-Black", "NH-Asian", "NH-AmInd-AKNat", "Hispanic", "NH-Other-Multi"]
NONREF_LABELS = RACE_LABELS[1:]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def recode_race_group(df: pd.DataFrame) -> pd.DataFrame:
    """Six race/ethnicity groups: White ref, Black, Asian, AmInd/AKNat, Hispanic, Other/Multi."""
    out = df.copy()
    hisp = out.get("HISP", pd.Series(np.zeros(len(out)), index=out.index)).fillna(0).astype(float).to_numpy()
    rac = out["RAC1P"].fillna(-1).astype(int).to_numpy()
    group = np.full(len(out), 5, dtype=int)  # other / multi
    group[(rac == 1)] = 0  # White
    group[(rac == 2)] = 1  # Black
    group[(rac == 6)] = 2  # Asian
    group[(rac == 3)] = 3  # American Indian / Alaska Native
    group[hisp > 1] = 4    # Hispanic overrides RAC1P
    out["race_group"] = group
    out["race_label"] = [RACE_LABELS[i] for i in group]
    return out


def build_from_folktables(year: int, horizon: str, state: str) -> pd.DataFrame:
    from folktables import ACSDataSource

    data_source = ACSDataSource(survey_year=str(year), horizon=horizon, survey="person")
    raw = data_source.get_data(states=[state], download=True)
    # ACSIncome-like filters. This avoids dependency on folktables internals and
    # keeps HISP/PUMA available for clustering and recoding.
    df = raw.copy()
    filters = (
        (df["AGEP"] > 16)
        & (df["PINCP"] > 100)
        & (df["WKHP"] > 0)
        & (df["PWGTP"] >= 1)
    )
    df = df.loc[filters].copy()
    df["INCOME_GT_50K"] = (df["PINCP"] > 50000).astype(int)
    df = recode_race_group(df)

    # Qualification signal Q: logistic prediction from non-race/non-ethnicity features.
    feature_cols = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "RELP", "SEX", "WKHP"]
    X = df[feature_cols].copy()
    y = df["INCOME_GT_50K"].to_numpy()
    categorical = ["COW", "SCHL", "MAR", "OCCP", "RELP", "SEX"]
    numeric = ["AGEP", "WKHP"]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    pre = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical),
            ("num", StandardScaler(with_mean=False), numeric),
        ]
    )
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    df["Q"] = pipe.predict_proba(X)[:, 1]
    keep = ["INCOME_GT_50K", "Q", "race_group", "race_label", "PUMA"]
    return df[keep].reset_index(drop=True)


def prepare_processed(args) -> pd.DataFrame:
    if args.input:
        df = read_table(Path(args.input))
        required = {"INCOME_GT_50K", "Q", "race_group", "race_label", "PUMA"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input is missing preprocessed columns: {sorted(missing)}")
        return df.copy()
    if not args.download:
        raise ValueError("Pass --input preprocessed_file or use --download to fetch/build Folktables data.")
    return build_from_folktables(args.year, args.horizon, args.state)


def asymptotic_draws(beta0, cov, B, rng, coef_start):
    draws = rng.multivariate_normal(beta0, cov, size=B, method="svd", check_valid="ignore")
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    pvals = 2 * norm.sf(np.abs(draws[:, coef_start:] / se[coef_start:]))
    flags = np.vstack([bh_flags(pvals[i]) for i in range(B)])
    return draws, se, pvals, flags, flags.sum(axis=1)


def summarize_negative(S, flags, labels):
    fr = flags.mean(axis=0)
    if fr.max() < 0.10:
        most = "— (diffuse)"
    else:
        most = ", ".join(f"{labels[i]} ({100 * fr[i]:.1f}%)" for i in np.where(fr >= max(0.10, fr.max() - 1e-12))[0])
    return {
        "mean_S": float(S.mean()),
        "fail_rate": float((S >= 1).mean()),
        "P_S_eq_0": float((S == 0).mean()),
        "max_S": int(S.max()),
        "flag_rates": {labels[i]: float(fr[i]) for i in range(len(labels))},
        "most_flagged": most,
    }


def negative_rows(pvals_good, flags_good, S_good, pvals_noq, flags_noq, S_noq):
    rows = []
    for pipeline, pvals, flags, S in [("A_good", pvals_good, flags_good, S_good), ("A_noQ", pvals_noq, flags_noq, S_noq)]:
        for b in range(len(S)):
            row = {"draw": b, "pipeline": pipeline, "S": int(S[b])}
            for j, label in enumerate(NONREF_LABELS):
                row[f"flag_{label}"] = bool(flags[b, j])
                row[f"p_{label}"] = float(pvals[b, j])
            rows.append(row)
    return pd.DataFrame(rows)


def positive_rows(beta_r, se_r, flags, pvals, covered, mae):
    rows = []
    for b in range(beta_r.shape[0]):
        row = {"draw": b, "S": int(flags[b].sum()), "mae_or_nonnull": float(mae[b])}
        for j, label in enumerate(NONREF_LABELS):
            row[f"beta_{label}"] = float(beta_r[b, j])
            row[f"se_{label}"] = float(se_r[j])
            row[f"OR_{label}"] = float(np.exp(beta_r[b, j]))
            row[f"covered_{label}"] = bool(covered[b, j])
            row[f"flag_{label}"] = bool(flags[b, j])
            row[f"p_{label}"] = float(pvals[b, j])
        rows.append(row)
    return pd.DataFrame(rows)


def run_asymptotic(df: pd.DataFrame, B_neg: int, B_pos: int, seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y, Q, race, D, X_good, X_noq = make_folktables_matrices(df)
    beta_base, _, info = fit_logit_irls(X_good, y)
    beta_clean, p_clean, beta_noq_clean, p_noq, beta_inj, p_inj, theta, alpha_star = synthetic_dgps(
        y, Q, race, D, X_good, X_noq, beta_base
    )

    _, Hinv_clean, _ = fit_logit_irls(X_good, p_clean, beta0=beta_clean, max_iter=1)
    _, Hinv_noq, _ = fit_logit_irls(X_noq, p_noq, beta0=beta_noq_clean, max_iter=1)
    _, Hinv_inj, _ = fit_logit_irls(X_good, p_inj, beta0=beta_inj, max_iter=1)
    clusters = df["PUMA"].to_numpy() if "PUMA" in df.columns else None
    cov_clean = expected_cluster_covariance_independent_bernoulli(X_good, p_clean, clusters, Hinv_clean)
    cov_noq = expected_cluster_covariance_independent_bernoulli(X_noq, p_noq, clusters, Hinv_noq)
    cov_inj = expected_cluster_covariance_independent_bernoulli(X_good, p_inj, clusters, Hinv_inj)

    rng = np.random.default_rng(seed)
    _, _, p_good, flags_good, S_good = asymptotic_draws(beta_clean, cov_clean, B_neg, rng, 2)
    _, _, p_noq_draw, flags_noq, S_noq = asymptotic_draws(beta_noq_clean, cov_noq, B_neg, rng, 1)
    draws_inj, se_inj, p_inj_draw, flags_inj, _ = asymptotic_draws(beta_inj, cov_inj, B_pos, rng, 2)
    beta_r = draws_inj[:, 2:]
    se_r = se_inj[2:]
    covered = (beta_r - 1.96 * se_r <= theta) & (theta <= beta_r + 1.96 * se_r)
    OR = np.exp(beta_r)
    inj = [0, 1, 3]
    null = [2, 4]
    mae = np.mean(np.abs(OR[:, inj] - np.exp(theta[inj])), axis=1)

    summary = make_summary(
        df, y, Q, race, beta_base, info, alpha_star, theta, B_neg, B_pos, seed,
        summarize_negative(S_good, flags_good, NONREF_LABELS),
        summarize_negative(S_noq, flags_noq, NONREF_LABELS),
        beta_r, covered, flags_inj, mae, method="asymptotic",
    )
    neg_df = negative_rows(p_good, flags_good, S_good, p_noq_draw, flags_noq, S_noq)
    pos_df = positive_rows(beta_r, se_r, flags_inj, p_inj_draw, covered, mae)
    return summary, neg_df, pos_df


def run_exact(df: pd.DataFrame, B_neg: int, B_pos: int, seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y, Q, race, D, X_good, X_noq = make_folktables_matrices(df)
    cluster_info = prepare_cluster(df["PUMA"].to_numpy()) if "PUMA" in df else None
    beta_base, _, info = fit_logit_irls(X_good, y)
    beta_clean, p_clean, beta_noq_clean, p_noq, beta_inj, p_inj, theta, alpha_star = synthetic_dgps(
        y, Q, race, D, X_good, X_noq, beta_base
    )
    rng = np.random.default_rng(seed)

    neg_rows_out = []
    S_good = np.zeros(B_neg, dtype=int)
    S_noq = np.zeros(B_neg, dtype=int)
    flags_good = np.zeros((B_neg, 5), dtype=bool)
    flags_noq = np.zeros((B_neg, 5), dtype=bool)

    for b in range(B_neg):
        yb = rng.binomial(1, p_clean).astype(float)
        beta_g, h_g, _ = fit_logit_irls(X_good, yb, beta0=beta_clean)
        cov_g = cluster_covariance(X_good, yb, beta_g, cluster_info, h_g) if cluster_info is not None else h_g
        se_g = np.sqrt(np.maximum(np.diag(cov_g), 0))
        pg = pvalues_from_beta(beta_g, se_g)[2:]
        fg = bh_flags(pg)
        flags_good[b] = fg
        S_good[b] = int(fg.sum())

        beta_n, h_n, _ = fit_logit_irls(X_noq, yb, beta0=beta_noq_clean)
        cov_n = cluster_covariance(X_noq, yb, beta_n, cluster_info, h_n) if cluster_info is not None else h_n
        se_n = np.sqrt(np.maximum(np.diag(cov_n), 0))
        pn = pvalues_from_beta(beta_n, se_n)[1:]
        fn = bh_flags(pn)
        flags_noq[b] = fn
        S_noq[b] = int(fn.sum())

        for pipeline, S, flags, pvals in [("A_good", S_good[b], fg, pg), ("A_noQ", S_noq[b], fn, pn)]:
            row = {"draw": b, "pipeline": pipeline, "S": int(S)}
            for j, label in enumerate(NONREF_LABELS):
                row[f"flag_{label}"] = bool(flags[j])
                row[f"p_{label}"] = float(pvals[j])
            neg_rows_out.append(row)

    beta_r = np.zeros((B_pos, 5))
    se_r = np.zeros((B_pos, 5))
    p_pos = np.zeros((B_pos, 5))
    flags_pos = np.zeros((B_pos, 5), dtype=bool)
    covered = np.zeros((B_pos, 5), dtype=bool)
    mae = np.zeros(B_pos)
    inj = [0, 1, 3]
    null = [2, 4]

    for b in range(B_pos):
        yb = rng.binomial(1, p_inj).astype(float)
        beta_p, h_p, _ = fit_logit_irls(X_good, yb, beta0=beta_inj)
        cov_p = cluster_covariance(X_good, yb, beta_p, cluster_info, h_p) if cluster_info is not None else h_p
        se_p = np.sqrt(np.maximum(np.diag(cov_p), 0))
        beta_r[b] = beta_p[2:]
        se_r[b] = se_p[2:]
        p_pos[b] = pvalues_from_beta(beta_p, se_p)[2:]
        flags_pos[b] = bh_flags(p_pos[b])
        covered[b] = (beta_r[b] - 1.96 * se_r[b] <= theta) & (theta <= beta_r[b] + 1.96 * se_r[b])
        mae[b] = np.mean(np.abs(np.exp(beta_r[b, inj]) - np.exp(theta[inj])))

    summary = make_summary(
        df, y, Q, race, beta_base, info, alpha_star, theta, B_neg, B_pos, seed,
        summarize_negative(S_good, flags_good, NONREF_LABELS),
        summarize_negative(S_noq, flags_noq, NONREF_LABELS),
        beta_r, covered, flags_pos, mae, method="exact",
    )
    pos_df = positive_rows(beta_r, se_r.mean(axis=0), flags_pos, p_pos, covered, mae)
    return summary, pd.DataFrame(neg_rows_out), pos_df


def make_folktables_matrices(df: pd.DataFrame):
    y = df["INCOME_GT_50K"].astype(float).to_numpy()
    Q = df["Q"].astype(float).to_numpy()
    race = df["race_group"].astype(int).to_numpy()
    D = np.column_stack([(race == k).astype(float) for k in range(1, 6)])
    X_good = np.column_stack([np.ones(len(df)), Q, D])
    X_noq = np.column_stack([np.ones(len(df)), D])
    return y, Q, race, D, X_good, X_noq


def synthetic_dgps(y, Q, race, D, X_good, X_noq, beta_base):
    beta_clean = np.r_[beta_base[0], beta_base[1], np.zeros(5)]
    p_clean = sigmoid(X_good @ beta_clean)

    mean_p = np.array([p_clean[race == k].mean() for k in range(6)])
    beta_noq_clean = np.r_[logit(mean_p[0]), logit(mean_p[1:]) - logit(mean_p[0])]
    p_noq = mean_p[race]

    theta = np.array([-0.5, -0.2, 0.0, -0.3, 0.0])
    alpha_star = solve_intercept_for_rate(beta_base[1] * Q + D @ theta, y.mean())
    beta_inj = np.r_[alpha_star, beta_base[1], theta]
    p_inj = sigmoid(X_good @ beta_inj)
    return beta_clean, p_clean, beta_noq_clean, p_noq, beta_inj, p_inj, theta, alpha_star


def make_summary(
    df, y, Q, race, beta_base, info, alpha_star, theta, B_neg, B_pos, seed,
    neg_good, neg_noq, beta_r, covered, flags_inj, mae, *, method: str,
):
    inj = [0, 1, 3]
    null = [2, 4]
    OR = np.exp(beta_r)
    return {
        "meta": {
            "N": int(len(df)),
            "PUMA_clusters": int(df["PUMA"].nunique()) if "PUMA" in df else None,
            "empirical_positive_rate": float(y.mean()),
            "mean_Q_by_group": {str(k): float(v) for k, v in df.groupby("race_label")["Q"].mean().items()},
            "race_counts": {str(k): int(v) for k, v in df["race_label"].value_counts().items()},
            "B_negative": B_neg,
            "B_positive": B_pos,
            "seed": seed,
            "method": method,
            "method_note": (
                "Fast asymptotic Monte Carlo of logistic audit estimates using the expected PUMA-cluster sandwich covariance under the row-independent Bernoulli synthetic-control DGP. Use --method exact for row-wise refits with cluster-robust SEs."
                if method == "asymptotic"
                else "Exact row-wise Bernoulli synthetic-control draws followed by refitting the logistic audit on each draw."
            ),
            "base_irls_iterations": info["iterations"],
            "alpha_star_positive": float(alpha_star),
        },
        "negative_control": {"A_good": neg_good, "A_noQ": neg_noq},
        "positive_control": {
            "target_theta": {NONREF_LABELS[i]: float(theta[i]) for i in range(5)},
            "target_OR": {NONREF_LABELS[i]: float(math.exp(theta[i])) for i in range(5)},
            "mean_recovered_OR": {NONREF_LABELS[i]: float(OR[:, i].mean()) for i in range(5)},
            "coverage_by_group": {NONREF_LABELS[i]: float(covered[:, i].mean()) for i in range(5)},
            "mean_coverage_nonnull": float(covered[:, inj].mean()),
            "mean_coverage_all_groups": float(covered.mean()),
            "MAE_OR_nonnull_mean": float(mae.mean()),
            "MAE_OR_nonnull_MC95": [float(np.quantile(mae, 0.025)), float(np.quantile(mae, 0.975))],
            "null_false_flags_per_draw_mean": float(flags_inj[:, null].sum(axis=1).mean()),
            "null_false_flags_P0": float((flags_inj[:, null].sum(axis=1) == 0).mean()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Preprocessed parquet/csv with INCOME_GT_50K,Q,race_group,race_label,PUMA")
    parser.add_argument("--download", action="store_true", help="Download/build ACS data with folktables")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--horizon", default="5-Year")
    parser.add_argument("--state", default="CA")
    parser.add_argument("--method", choices=["asymptotic", "exact"], default="asymptotic")
    parser.add_argument("--subsample", type=int, default=0, help="Optional random subsample size, useful for exact-mode checks")
    parser.add_argument("--B-neg", type=int, default=300)
    parser.add_argument("--B-pos", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--out", default="results/folktables_summary.json")
    parser.add_argument("--draws-dir", default="", help="Optional directory for per-draw CSV outputs")
    args = parser.parse_args()

    df = prepare_processed(args)
    if args.subsample and args.subsample < len(df):
        df = df.sample(n=args.subsample, random_state=args.seed).reset_index(drop=True)

    if args.method == "asymptotic":
        summary, neg_df, pos_df = run_asymptotic(df, args.B_neg, args.B_pos, args.seed)
    else:
        summary, neg_df, pos_df = run_exact(df, args.B_neg, args.B_pos, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    if args.draws_dir:
        draws_dir = Path(args.draws_dir)
        draws_dir.mkdir(parents=True, exist_ok=True)
        neg_df.to_csv(draws_dir / "folktables_negative_draws.csv", index=False)
        pos_df.to_csv(draws_dir / "folktables_positive_draws.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
