# Data

The proprietary hiring dataset used in the paper cannot be released. The scripts
expect a row-level table with the following schema:

| column | description |
|---|---|
| `job_id` | cluster identifier, e.g. requisition/job id |
| `Y_hist` | binary historical outcome |
| `Q` | continuous qualification score in `[0, 1]` |
| `gender` | anonymized categorical attribute |
| `ethnicity` | anonymized categorical attribute, e.g. `E1`--`E13` |
| `age_band` | anonymized age band |
| `location` | anonymized location code, e.g. `R1`--`R14` |

The category references and injected effects are configured in
`configs/hiring_schema_template.json`. To rerun on a private non-anonymized local
file, create a private config file with the corresponding reference category names
and coefficient names; do not commit that config if it reveals private labels.

`synthetic_hiring_example.csv` is a synthetic example dataset for checks only; it is not
used for the paper's reported hiring numbers.
