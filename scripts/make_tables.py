#!/usr/bin/env python3
"""Generate paper-ready result snippets from stored JSON summaries.

This script intentionally consumes only aggregate/redacted outputs for the
proprietary hiring experiments. It is meant to make the paper numbers auditable
without exposing row-level private data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Directory containing summary JSON files")
    parser.add_argument("--out", default="results/table_snippets.md", help="Markdown output path")
    args = parser.parse_args()

    root = Path(args.results_dir)
    folktables = load_json(root / "folktables_summary.json")
    main_controls_path = root / "hiring_main_controls_summary_redacted.json"
    main_controls = load_json(main_controls_path) if main_controls_path.exists() else None
    perm = load_json(root / "hiring_exp2_permutation_summary_redacted.json")
    bayes = load_json(root / "hiring_exp3_bayesian_laplace_summary_redacted.json")
    glass_path = root / "hiring_glass_ceiling_summary_redacted.json"
    glass = load_json(glass_path) if glass_path.exists() else None

    lines: list[str] = []
    lines.append("# Result snippets from JSON summaries\n")
    lines.append("Produced by `scripts/make_tables.py`.\n")

    lines.append("## Table 6: ACS/Folktables replication\n")
    lines.append("| Pipeline | mean S | fail rate | most-flagged |")
    lines.append("|---|---:|---:|---|")
    for name in ["A_good", "A_noQ"]:
        block = folktables["negative_control"][name]
        lines.append(
            f"| `{name}` | {fmt3(block['mean_S'])} | {pct(block['fail_rate'])} | {block.get('most_flagged', '')} |"
        )
    pc = folktables["positive_control"]
    mae_lo, mae_hi = pc["MAE_OR_nonnull_MC95"]
    lines.append("")
    lines.append(
        "Positive control: "
        f"mean non-null coverage = {pct(pc['mean_coverage_nonnull'])}; "
        f"MAE_OR = {pc['MAE_OR_nonnull_mean']:.3f} [{mae_lo:.3f}, {mae_hi:.3f}]."
    )
    lines.append(
        f"N = {folktables['meta']['N']:,}; method = {folktables['meta'].get('method', 'asymptotic')}"
    )
    lines.append("")

    if main_controls is not None:
        lines.append("## Tables 4--5: private hiring paired controls and diagnostic\n")
        neg = main_controls["controls"]["negative_control"]
        pos = main_controls["controls"]["positive_control"]
        mae_lo, mae_hi = pos["MAE_OR_MC95"]
        lines.append(f"Negative control: P(S=0) = {neg['P_S_eq_0']:.3f}; mean S = {neg['mean_S']:.2f}; max S = {neg['max_S']}.")
        lines.append(f"Positive control: mean injected-group coverage = {pct(pos['mean_coverage_injected'])}; MAE_OR = {pos['MAE_OR_mean']:.3f} [{mae_lo:.3f}, {mae_hi:.3f}].")
        lines.append("")
        lines.append("Diagnostic negative control:")
        lines.append("| Pipeline | mean S | fail rate | most-flagged |")
        lines.append("|---|---:|---:|---|")
        for name in ["A_good", "A_iid", "A_coarseQ", "A_noQ"]:
            block = main_controls["diagnostic"][name]
            lines.append(f"| `{name}` | {block['mean_S']:.2f} | {pct(block['fail_rate'])} | {block['most_flagged']} |")
        lines.append("")

    lines.append("## Section 5: permutation negative-control baseline\n")
    lines.append("| Pipeline | mean S | fail rate | P(S=0) | max S |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ["A_good", "A_noQ"]:
        block = perm[name]
        lines.append(
            f"| `{name}` | {fmt3(block['mean_S'])} | {pct(block['fail_rate'])} | {pct(block['P_S_eq_0'])} | {block['max_S']} |"
        )
    top = ", ".join(f"{x['label']} ({pct(x['rate'])})" for x in perm.get("top_flagged_A_noQ", []))
    lines.append(f"Most-flagged omitted-Q groups: {top}.\n")

    lines.append("## Section 5: Bayesian/Laplace positive-control sanity check\n")
    bl = bayes["bayes_laplace"]
    fr = bayes["frequentist_cluster"]
    bl_lo, bl_hi = bl["MAE_OR_MC_95_interval"]
    fr_lo, fr_hi = fr["MAE_OR_MC_95_interval"]
    lines.append("| Method | mean coverage | MAE_OR |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Bayesian/Laplace | {pct(bl['mean_coverage'])} | {bl['MAE_OR_mean']:.3f} [{bl_lo:.3f}, {bl_hi:.3f}] |")
    lines.append(f"| Frequentist cluster | {pct(fr['mean_coverage'])} | {fr['MAE_OR_mean']:.3f} [{fr_lo:.3f}, {fr_hi:.3f}] |")
    lines.append("")
    lines.append(
        f"Max per-group coverage difference = {bayes['coverage_max_abs_difference_pp']:.1f} pp; "
        f"posterior-CDF uniformity KS p = {bayes['posterior_cdf_uniformity_KS_p']:.3f}."
    )
    lines.append("")

    if glass is not None:
        lines.append("## Section 6: glass-ceiling breakpoint bootstrap\n")
        lines.append(f"Selected tau = {glass['selected_tau']:.2f}; observed sup-LR = {glass['observed_sup_LR']:.1f}; bootstrap p = {glass['bootstrap_p']:.3g}; bootstrap q95 = {glass['bootstrap_q95']:.1f}.")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
