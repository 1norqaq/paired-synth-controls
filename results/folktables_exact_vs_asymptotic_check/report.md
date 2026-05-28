# Folktables exact-vs-asymptotic verification

Small-scale verification run on the same random subsample of ACS CA 2018 5-year.

- Input: `<path>/folktables_ca_2018.parquet`
- Subsample: 50,000 rows
- PUMA clusters: 265
- Seed: 20260528
- Negative-control draws: 100
- Positive-control draws: 100
- Asymptotic path: expected PUMA-cluster sandwich covariance
- Exact path: row-wise Bernoulli synthetic outcomes + refit + PUMA cluster-robust SE

## Summary

| Method | A_good mean S | A_good fail | A_noQ mean S | A_noQ fail | Non-null coverage | MAE_OR |
|---|---:|---:|---:|---:|---:|---:|
| Asymptotic | 0.060 | 6.0% | 4.960 | 100.0% | 94.0% | 0.0230 [0.0061, 0.0425] |
| Exact | 0.040 | 3.0% | 5.000 | 100.0% | 96.3% | 0.0219 [0.0070, 0.0462] |

## Interpretation

The row-wise exact run agrees with the asymptotic approximation on the substantive verdict:

- the correct audit stays near nominal calibration in both paths;
- the omitted-Q audit fails essentially every draw in both paths;
- positive-control recovery is close: 94.0% non-null coverage for asymptotic versus 96.3% for exact;
- MAE_OR is nearly identical: 0.0230 versus 0.0219.

The remaining differences are within the expected Monte Carlo variation for B=100 on a 50k subsample.
