"""Create a SYNTHETIC dataset with the same schema as 'Default of Credit Card Clients'.

Only for testing the pipeline when you don't have the real data yet.
Metrics produced from this file are NOT meaningful - use the real UCI dataset.
"""
import numpy as np
import pandas as pd

from .data_utils import DATA_DIR, BILL_COLS, PAYAMT_COLS, PAY_COLS


def generate(n: int = 30000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    limit = np.clip(rng.lognormal(11.8, 0.8, n) // 10_000 * 10_000, 10_000, 800_000)
    age = np.clip(rng.normal(35, 9, n).round(), 21, 75).astype(int)
    df = pd.DataFrame({
        "LIMIT_BAL": limit.astype(int),
        "SEX": rng.choice([1, 2], n, p=[0.4, 0.6]),
        "EDUCATION": rng.choice([1, 2, 3, 4], n, p=[0.35, 0.47, 0.16, 0.02]),
        "MARRIAGE": rng.choice([1, 2, 3], n, p=[0.45, 0.53, 0.02]),
        "AGE": age,
    })
    risk = rng.normal(0, 1, n) - 0.4 * np.log(limit / 100_000)
    base = np.clip(np.round(risk * 0.9), -2, 8).astype(int)
    for i, col in enumerate(PAY_COLS):
        noise = rng.integers(-1, 2, n)
        df[col] = np.clip(base + noise - (i // 2), -2, 8)
    util = rng.uniform(0.05, 1.0, n)
    for col in BILL_COLS:
        df[col] = (limit * util * rng.uniform(0.7, 1.2, n)).astype(int)
    for col in PAYAMT_COLS:
        paid = np.where(base <= 0, rng.uniform(0.02, 0.2, n), rng.uniform(0, 0.03, n))
        df[col] = (limit * util * paid).astype(int)
    logit = (-1.6 + 0.75 * np.clip(df["PAY_0"], -2, 6) + 0.1 * np.clip(df["PAY_2"], -2, 6)
             - 0.000002 * limit + rng.normal(0, 0.6, n))
    df["DEFAULT"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


if __name__ == "__main__":
    out = DATA_DIR / "sample_credit_data.csv"
    out.parent.mkdir(exist_ok=True)
    d = generate()
    d.to_csv(out, index=False)
    print(f"Wrote synthetic data ({len(d):,} rows, default rate {d['DEFAULT'].mean():.1%}) -> {out}")
