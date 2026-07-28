# Result snippets from JSON summaries

Produced by `scripts/make_tables.py`.

## Table 6: ACS/Folktables replication

| Pipeline | mean S | fail rate | most-flagged |
|---|---:|---:|---|
| `A_good` | 0.027 | 2.7% | — (diffuse) |
| `A_noQ` | 5.000 | 100.0% | NH-Black (100.0%), NH-Asian (100.0%), NH-AmInd-AKNat (100.0%), Hispanic (100.0%), NH-Other-Multi (100.0%) |

Positive control: mean non-null coverage = 94.2%; MAE_OR = 0.005 [0.001, 0.011].
N = 950,197; method = asymptotic

## Tables 4--5: private hiring paired controls and diagnostic

Negative control: P(S=0) = 0.950; mean S = 0.06; max S = 2.
Positive control: mean injected-group coverage = 94.6%; MAE_OR = 0.037 [0.016, 0.068].

Diagnostic negative control:
| Pipeline | mean S | fail rate | most-flagged |
|---|---:|---:|---|
| `A_good` | 0.07 | 5.7% | diffuse |
| `A_iid` | 0.07 | 5.7% | diffuse |
| `A_coarseQ` | 0.10 | 8.0% | diffuse |
| `A_noQ` | 0.78 | 48.3% | E2 (39%), E3 (18%) |

## Section 5.3: permutation negative-control baseline

| Pipeline | mean S | fail rate | P(S=0) | max S |
|---|---:|---:|---:|---:|
| `A_good` | 0.073 | 6.7% | 93.3% | 2 |
| `A_noQ` | 1.120 | 59.7% | 40.3% | 6 |
Most-flagged omitted-Q groups: E2 (46.7%), E3 (31.0%), E6 (11.7%).

## Section 5.3: Bayesian/Laplace positive-control sanity check

| Method | mean coverage | MAE_OR |
|---|---:|---:|
| Bayesian/Laplace | 94.9% | 0.037 [0.016, 0.066] |
| Frequentist cluster | 95.2% | 0.037 [0.016, 0.066] |

Max per-group coverage difference = 0.5 pp; posterior-CDF uniformity KS p = 0.998.

## Section 6: glass-ceiling breakpoint bootstrap

Selected tau = 0.90; observed sup-LR = 33.3; bootstrap p = 0.004; bootstrap q95 = 6.2.
