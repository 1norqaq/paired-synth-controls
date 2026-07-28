# Auditing the Auditor: Paired Synthetic Controls for Calibrating Fairness Audits

This repository contains code for the experiments in the paper
**"Auditing the Auditor: Paired Synthetic Controls for Calibrating Fairness Audits"**.

The repository has two parts:

1. **Public reproducibility:** the ACS/Folktables replication can be rerun without proprietary data.
2. **Proprietary hiring experiments:** the exact scripts, schema, synthetic example data, and redacted aggregate outputs are provided, but the row-level hiring dataset and private per-draw outputs are not released.

See `REPRODUCIBILITY.md` for seeds, runtime expectations, exact/asymptotic modes,
and privacy details.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick check

The check uses only the included synthetic example data and checks the private-data code
paths plus table generation:

```bash
bash scripts/smoke_test.sh
```

## Public ACS/Folktables replication

Run from the repository root:

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

If you already have a preprocessed ACS/Folktables file with columns
`INCOME_GT_50K`, `Q`, `race_group`, `race_label`, and `PUMA`, use:

```bash
python experiments/run_folktables.py \
  --input /path/to/folktables_ca_2018.parquet \
  --method asymptotic \
  --B-neg 300 \
  --B-pos 400 \
  --out results/folktables_summary.json \
  --draws-dir results
```

Reported output on the ACS CA 2018 5-year file used in the paper:

| quantity | value |
|---|---:|
| N | 950,197 |
| `A_good` negative-control mean S | 0.027 |
| `A_good` failure rate | 2.7% |
| `A_noQ` negative-control mean S | 5.000 |
| `A_noQ` failure rate | 100.0% |
| positive-control non-null mean coverage | 94.2% |
| positive-control `MAE_OR` | 0.005 [0.001, 0.011] |

The reported full-sample result uses `--method asymptotic`, a fast coefficient-draw
approximation using the expected PUMA-cluster sandwich covariance under the
row-independent Bernoulli synthetic-control DGP. The exact row-wise
Bernoulli/refit code path, which uses PUMA cluster-robust standard errors on
each synthetic outcome, is also included:

```bash
python experiments/run_folktables.py \
  --input /path/to/folktables_ca_2018.parquet \
  --method exact \
  --subsample 50000 \
  --B-neg 20 \
  --B-pos 20 \
  --out results/folktables_exact_check_summary.json \
  --draws-dir results/exact_check
```

Use exact mode with a smaller B or subsample unless substantial CPU time is
available.


## Private hiring main paired-control and diagnostic analyses

The core hiring analyses reported in Tables 4--5 of the paper can be regenerated on a local
schema-compatible private file. The repository includes the full script but not
the row-level proprietary data.

```bash
python experiments/run_hiring_main_controls.py \
  --data /path/to/hiring_cleaned.csv \
  --config configs/hiring_schema_template.json \
  --B-controls 400 \
  --B-diagnostic 300 \
  --out results/hiring_main_controls_summary.json \
  --draws-dir results/hiring_main_draws
```

The script implements the exact row-wise Bernoulli/refit paired-control
procedure with cluster-robust logit audits, including the `A_good`, `A_iid`,
`A_coarseQ`, and `A_noQ` diagnostic variants. Redacted private-data aggregate
outputs are stored in `results/hiring_main_controls_summary_redacted.json`. Private per-draw CSVs are not included in the release; use `--draws-dir` only to regenerate them locally when running on an authorized private file.

## Permutation negative-control baseline

The proprietary hiring data are not included. The required schema is described in
`data/README.md` and `configs/hiring_schema_template.json`. A synthetic example file
is included for checks:

```bash
python experiments/run_permutation_baseline.py \
  --data data/synthetic_hiring_example.csv \
  --config configs/hiring_schema_template.json \
  --B 10 \
  --permute-mode tuple \
  --out results/permutation_smoke_test.json
```

Reported aggregate result on the private hiring dataset:

| pipeline | mean S | failure rate | P(S = 0) | max S |
|---|---:|---:|---:|---:|
| `A_good` | 0.073 | 6.7% | 93.3% | 2 |
| `A_noQ` | 1.120 | 59.7% | 40.3% | 6 |

The most-flagged omitted-Q groups were E2 (46.7%), E3 (31.0%), and E6 (11.7%).
Only redacted aggregate summary outputs are included under `results/`; private per-draw outputs are intentionally not included. They can be regenerated locally with `--draws-dir` when a schema-compatible private file is available.

## Bayesian/Laplace positive-control sanity check

This is a Gaussian-prior logistic regression with a Laplace/MAP posterior
approximation, not full MCMC.

Check on synthetic example data:

```bash
python experiments/run_bayesian_laplace_sbc.py \
  --data data/synthetic_hiring_example.csv \
  --config configs/hiring_schema_template.json \
  --B 10 \
  --out results/bayesian_laplace_smoke_test.json
```

Reported aggregate result on the private hiring dataset:

| method | mean coverage | `MAE_OR` |
|---|---:|---:|
| Bayesian/Laplace | 94.9% | 0.037 [0.016, 0.066] |
| frequentist cluster | 95.2% | 0.037 [0.016, 0.066] |

The maximum per-group coverage difference was 0.5 percentage points, and the
posterior-CDF/rank uniformity diagnostic gave KS p = 0.998. Only redacted
aggregate summary outputs are included under `results/`; private per-draw
outputs are intentionally not included. They can be regenerated locally when a
schema-compatible private file is available.


## Sections 6 and 8 supporting scripts

The empirical and limitation analyses beyond the baseline experiments are also
implemented:

```bash
python experiments/run_glass_ceiling_bootstrap.py \
  --data /path/to/hiring_cleaned.csv \
  --config configs/hiring_schema_template.json \
  --groups ethnicity_E2 ethnicity_E3 \
  --tau-grid 0.75,0.80,0.85,0.90,0.95 \
  --B-bootstrap 250 \
  --out results/glass_ceiling_summary.json

python experiments/run_interaction_recursion.py \
  --data /path/to/hiring_cleaned.csv \
  --config configs/hiring_schema_template.json \
  --group ethnicity_E3 \
  --threshold 0.85 \
  --theta-high -0.9 \
  --B 200 \
  --out results/interaction_recursion_summary.json

python experiments/run_measurement_error_sensitivity.py \
  --data /path/to/hiring_cleaned.csv \
  --config configs/hiring_schema_template.json \
  --attribute ethnicity \
  --group-coef ethnicity_E2 \
  --theta -0.6931471805599453 \
  --rates 0,0.10,0.15 \
  --out results/measurement_error_summary.json
```

Redacted private-data aggregate outputs for these analyses are included under
`results/`.

## Generate paper table snippets

```bash
python scripts/make_tables.py --results-dir results --out results/table_snippets.md
```

This regenerates the Table 6 and Section 5 construction-comparison numbers from the stored JSON summaries.

## Repository layout

```text
src/paired_synth_controls/        shared design/audit utilities
experiments/run_folktables.py     public ACS replication; asymptotic and exact modes
experiments/run_hiring_main_controls.py
experiments/run_permutation_baseline.py
experiments/run_bayesian_laplace_sbc.py
experiments/run_glass_ceiling_bootstrap.py
experiments/run_interaction_recursion.py
experiments/run_measurement_error_sensitivity.py
scripts/make_tables.py            regenerate paper table snippets
scripts/smoke_test.sh             synthetic-example B=5 check
scripts/extended_analysis_test.sh  synthetic-example checks for Sections 6 and 8 scripts
configs/hiring_schema_template.json
data/synthetic_hiring_example.csv synthetic example data only
results/                          public and redacted outputs
REPRODUCIBILITY.md                detailed reproducibility notes
```

## Privacy note

Do **not** commit the proprietary hiring row-level data or private per-draw
outputs. The private experiments are reproducible by the authors with a local
file matching the documented schema; reviewers can run the same code path on the
synthetic example data and can fully reproduce the public Folktables experiment.
