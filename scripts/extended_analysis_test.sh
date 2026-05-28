#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
SMOKE_DATA=${SMOKE_DATA:-results/smoke_hiring_subset.csv}
if [ ! -f "$SMOKE_DATA" ]; then
  python - <<'PY'
import pandas as pd
src = 'data/synthetic_hiring_example.csv'
df = pd.read_csv(src).sample(n=min(1200, len(pd.read_csv(src))), random_state=20260526)
df.to_csv('results/smoke_hiring_subset.csv', index=False)
PY
  SMOKE_DATA=results/smoke_hiring_subset.csv
fi
python experiments/run_glass_ceiling_bootstrap.py \
  --data "$SMOKE_DATA" \
  --config configs/hiring_schema_template.json \
  --B-bootstrap 5 \
  --tau-grid 0.75,0.90 \
  --out results/glass_ceiling_smoke_test.json
python experiments/run_interaction_recursion.py \
  --data "$SMOKE_DATA" \
  --config configs/hiring_schema_template.json \
  --B 5 \
  --out results/interaction_recursion_smoke_test.json
python experiments/run_measurement_error_sensitivity.py \
  --data "$SMOKE_DATA" \
  --config configs/hiring_schema_template.json \
  --rates 0,0.10,0.20 \
  --out results/measurement_error_smoke_test.json
echo "Extended analysis tests completed."
