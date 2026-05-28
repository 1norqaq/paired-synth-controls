#!/usr/bin/env bash
set -euo pipefail

# Run from repository root. Uses only included synthetic example data. This is deliberately
# stronger than a B=2 import check while staying short enough for reviewers.
mkdir -p results/smoke_draws
python - <<'PY'
import pandas as pd
src = 'data/synthetic_hiring_example.csv'
df = pd.read_csv(src).sample(n=min(1200, len(pd.read_csv(src))), random_state=20260526)
df.to_csv('results/smoke_hiring_subset.csv', index=False)
print(f'Wrote results/smoke_hiring_subset.csv with n={len(df)}')
PY
SMOKE_DATA=results/smoke_hiring_subset.csv

python experiments/run_hiring_main_controls.py \
  --data "$SMOKE_DATA" \
  --config configs/hiring_schema_template.json \
  --B-controls 5 \
  --B-diagnostic 5 \
  --out results/hiring_main_controls_smoke_test.json \
  --draws-dir results/smoke_draws

python experiments/run_permutation_baseline.py \
  --data "$SMOKE_DATA" \
  --config configs/hiring_schema_template.json \
  --B 5 \
  --permute-mode tuple \
  --out results/permutation_smoke_test.json \
  --draws-out results/smoke_draws/permutation_smoke_draws.csv

python experiments/run_bayesian_laplace_sbc.py \
  --data "$SMOKE_DATA" \
  --config configs/hiring_schema_template.json \
  --B 5 \
  --out results/bayesian_laplace_smoke_test.json

python scripts/make_tables.py --results-dir results --out results/table_snippets.md

echo "Checks completed. For glass-ceiling / interaction / measurement-error code paths, run scripts/extended_analysis_test.sh."
