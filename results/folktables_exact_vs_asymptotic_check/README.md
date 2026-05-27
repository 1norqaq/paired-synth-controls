# Folktables exact-vs-asymptotic verification

This folder contains a small-scale verification run on a 50,000-row ACS CA 2018 five-year subsample with 100 negative-control and 100 positive-control draws. The exact path uses row-wise Bernoulli synthetic outcomes, audit refits, and PUMA cluster-robust standard errors. The asymptotic path uses the expected PUMA-cluster sandwich covariance.

The two paths agree on the substantive verdict: the correct audit stays near nominal, the omitted-Q audit fails on all draws, and positive-control recovery is close.
