#!/usr/bin/env python3
"""Generate the small synthetic hiring file included in data/."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

rng = np.random.default_rng(12345)
N = 6000
J = 300
job_id = rng.integers(0, J, size=N)
eth = rng.choice(["E1", "E2", "E3", "E4", "E5", "E6"], size=N, p=[.28,.20,.16,.14,.12,.10])
gender = rng.choice(["Male", "Female"], size=N, p=[.55,.45])
age = rng.choice(["18-29", "30-39", "40-49", "50-64"], size=N, p=[.30,.35,.23,.12])
loc = rng.choice(["R1", "R2", "R3", "R4"], size=N, p=[.45,.25,.20,.10])
base_q = rng.beta(2.2, 3.2, size=N)
q_shift = np.select([eth=="E2", eth=="E3", eth=="E6", eth=="E5"], [-.08, -.06, -.05, .03], default=0.0)
Q = np.clip(base_q + q_shift + rng.normal(0, .03, size=N), 0.01, 0.99)
eta = -3.2 + 4.8 * Q
eta += np.select([eth=="E2", eth=="E3", eth=="E4", eth=="E5"], [-.4, -.25, -.1, .15], default=0.0)
eta += np.where(gender=="Female", -.05, 0.0) + np.where(loc=="R2", -.15, 0.0)
Y_hist = rng.binomial(1, expit(eta))
df = pd.DataFrame({"job_id": job_id, "Y_hist": Y_hist, "Q": Q, "gender": gender, "ethnicity": eth, "age_band": age, "location": loc})
out = Path(__file__).resolve().parents[1] / "data" / "synthetic_hiring_example.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(out, df.shape, df.Y_hist.mean())
