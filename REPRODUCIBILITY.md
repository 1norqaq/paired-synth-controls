# Reproducibility Notes

This document summarizes the information needed to rerun the experiments and to
interpret which numbers are fully public versus proprietary-data dependent.

## Environment

Recommended environment:

- Python 3.10 or later
- CPU only; no GPU required
- Packages listed in `requirements.txt`
- Tested with NumPy/SciPy/pandas/scikit-learn/statsmodels on Linux

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Random seeds

The public Folktables script defaults to seed `20260527`. The hiring scripts use
fixed defaults printed in their JSON outputs. All scripts accept a `--seed`
argument where stochastic draws are used.

## Public ACS/Folktables replication

Fully public. The script can download ACS data through `folktables`, or consume a
preprocessed file with columns:

```text
INCOME_GT_50K, Q, race_group, race_label, PUMA
```

Reported full-sample command:

```bash
python experiments/run_folktables.py \
  --download \
  --year 2018 \
  --horizon "5-Year" \
  --state CA \
  --method asymptotic \
  --B-neg 300 \
  --B-pos 400 \
  --out results/folktables_summary.json \
  --draws-dir results
```

The reported public result uses the fast asymptotic coefficient-draw
approximation on the full ACS CA 2018 5-year sample (`N = 950,197` in our local
run), using the expected PUMA-cluster sandwich covariance under the row-independent Bernoulli synthetic DGP. Because the synthetic DGP draws rows independently conditional on X, the expected cluster-sandwich meat equals the Fisher information up to the usual finite-cluster correction; exact mode computes realized cluster-robust SEs on every synthetic outcome.
This mode completes much faster than refitting 700 full-sample logistic models.

Exact row-wise Bernoulli refit mode is included to verify that the code path
matches the paired-control algorithm and applies cluster-robust standard errors. It is intentionally run with small B or a
subsample unless substantial CPU time is available:

```bash
python experiments/run_folktables.py \
  --input /path/to/preprocessed_folktables_ca_2018.parquet \
  --method exact \
  --subsample 50000 \
  --B-neg 20 \
  --B-pos 20 \
  --out results/folktables_exact_check_summary.json \
  --draws-dir results/exact_check
```

Expected full-sample asymptotic numbers:

- `A_good`: mean `S = 0.027`, fail rate `2.7%`
- `A_noQ`: mean `S = 5.000`, fail rate `100.0%`
- positive-control non-null coverage `94.2%`
- `MAE_OR = 0.005 [0.001, 0.011]`


## Private hiring main controls and omitted-Q diagnostic

Requires the proprietary hiring dataset. The full code path is implemented in
`experiments/run_hiring_main_controls.py`. With a schema-compatible local file,
run:

```bash
python experiments/run_hiring_main_controls.py \
  --data /path/to/hiring_cleaned.csv \
  --config configs/hiring_schema_template.json \
  --B-controls 400 \
  --B-diagnostic 300 \
  --out results/hiring_main_controls_summary.json \
  --draws-dir results/hiring_main_draws
```

This reproduces the main negative/positive paired controls and the weakened
pipeline diagnostic (`A_good`, `A_iid`, `A_coarseQ`, `A_noQ`). Redacted aggregate
outputs are provided in `results/hiring_main_controls_summary_redacted.json`.

## Permutation negative-control baseline

Requires the proprietary hiring dataset, which cannot be released. The required
schema is documented in `data/README.md` and `configs/hiring_schema_template.json`.
The code path can be tested with the included synthetic example file:

```bash
python experiments/run_permutation_baseline.py \
  --data data/synthetic_hiring_example.csv \
  --config configs/hiring_schema_template.json \
  --B 10 \
  --permute-mode tuple \
  --out results/permutation_smoke_test.json
```

Reported private-data aggregate result:

- `A_good`: mean `S = 0.073`, fail rate `6.7%`
- `A_noQ`: mean `S = 1.120`, fail rate `59.7%`
- most-flagged omitted-Q groups: E2, E3, E6

Only redacted aggregate and per-draw outputs are included for this experiment.
They contain no row-level hiring records. The default is tuple-level permutation
within Q-deciles to preserve the joint distribution of A; `--permute-mode
separate` is included as a sensitivity option.

## Bayesian/Laplace positive-control sanity check

Requires the same proprietary hiring dataset. The included code uses a
Gaussian-prior logistic regression with a Laplace/MAP posterior approximation; it
is **not** full MCMC. This distinction should be retained in the paper.

Toy check:

```bash
python experiments/run_bayesian_laplace_sbc.py \
  --data data/synthetic_hiring_example.csv \
  --config configs/hiring_schema_template.json \
  --B 10 \
  --out results/bayesian_laplace_smoke_test.json
```

Reported private-data aggregate result over 200 positive-control draws:

- Bayesian/Laplace mean coverage: `94.9%`
- frequentist cluster mean coverage: `95.2%`
- max per-group coverage difference: `0.5` percentage points
- posterior-CDF/rank uniformity KS p-value: `0.998`
- `MAE_OR = 0.037 [0.016, 0.066]` for both methods up to rounding


## Sections 6 and 8 supporting analyses

The glass-ceiling breakpoint bootstrap, working-model recursion / interaction-aware
variant, and sensitive-attribute measurement-error sensitivity are implemented in:

- `experiments/run_glass_ceiling_bootstrap.py`
- `experiments/run_interaction_recursion.py`
- `experiments/run_measurement_error_sensitivity.py`

These require the same proprietary hiring schema for paper-number regeneration.
They also run on the included synthetic example data via:

```bash
bash scripts/extended_analysis_test.sh
```

Redacted aggregate outputs are included in `results/hiring_glass_ceiling_summary_redacted.json`,
`results/hiring_interaction_recursion_summary_redacted.json`, and
`results/hiring_measurement_error_summary_redacted.json`.

## Generating paper snippets

The paper tables and prose numbers can be regenerated from the JSON summaries:

```bash
python scripts/make_tables.py --results-dir results --out results/table_snippets.md
```

## Checks

Run the following from the repository root:

```bash
bash scripts/smoke_test.sh
bash scripts/extended_analysis_test.sh
```

This uses only the included synthetic example data and checks the private-data scripts plus the
summary-to-table generation path.

## Privacy and excluded files

The repository deliberately excludes:

- row-level proprietary hiring data
- profile/candidate identifiers
- vendor, company, jurisdiction, or observation-window identifiers
- raw `.pkl` extracts and cleaned private hiring CSV files

Private-data outputs included under `results/` are aggregate or redacted per-draw
statistics only.
