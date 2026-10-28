"""Synthetic card-transaction generator.

Real bank transaction data is confidential, and the popular public fraud datasets
ship already-anonymised PCA features, which hides the most interesting part of the
problem: turning raw transactions into behavioural signals. This module simulates
raw transactions for a population of customers so the full pipeline can be built
end to end.

Fraud is injected as "account takeover" episodes: a burst of transactions shortly
after a compromise, often on a new device, far from home, at odd hours and in
categories that are easy to resell (electronics, gift cards). Legitimate customers
also travel, shop late and buy electronics, so no single signal gives fraud away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = [
    "grocery",
    "fuel",
    "restaurant",
    "transport",
    "retail",
    "online_retail",
    "electronics",
    "gift_cards",
    "travel",
    "utilities",
]

# How often an ordinary customer shops in each category.
LEGIT_CATEGORY_WEIGHTS = np.array([0.24, 0.10, 0.16, 0.10, 0.12, 0.13, 0.04, 0.01, 0.03, 0.07])
# What fraudsters tend to buy.
FRAUD_CATEGORY_WEIGHTS = np.array([0.03, 0.02, 0.03, 0.02, 0.08, 0.27, 0.27, 0.20, 0.06, 0.02])
# Typical spend multiplier per category, relative to the customer's usual spend.
CATEGORY_SCALE = np.array([1.0, 1.2, 0.8, 0.4, 1.5, 1.3, 4.0, 2.0, 6.0, 2.5])


def generate_transactions(
    n_customers: int = 2000,
    n_days: int = 120,
    fraud_episode_rate: float = 0.06,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate raw transactions.

    Args:
        n_customers: number of cardholders.
        n_days: length of the simulated period.
        fraud_episode_rate: share of customers whose card is compromised at some point.
        seed: random seed for reproducibility.

    Returns:
        DataFrame sorted by timestamp with one row per transaction and an ``is_fraud`` label.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
    rows: list[dict] = []
    txn_id = 0

    for cust in range(n_customers):
        typical_spend = rng.lognormal(mean=3.4, sigma=0.5)  # roughly $15-$80 median spend
        daily_rate = rng.uniform(0.6, 3.0)
        n_devices = rng.integers(1, 3)
        devices = [f"dev_{cust}_{d}" for d in range(n_devices)]
        night_owl = rng.random() < 0.12

        n_txn = rng.poisson(daily_rate * n_days)
        times = np.sort(rng.uniform(0, n_days * 86400, size=n_txn))
        for t in times:
            # Shift timestamps into realistic waking hours for most people.
            day = int(t // 86400)
            if night_owl:
                hour = rng.integers(0, 24)
            else:
                hour = int(np.clip(rng.normal(14, 3.5), 6, 23))
            ts = start + pd.Timedelta(days=day, hours=int(hour), minutes=int(rng.integers(0, 60)))
            cat_idx = rng.choice(len(CATEGORIES), p=LEGIT_CATEGORY_WEIGHTS)
            amount = typical_spend * CATEGORY_SCALE[cat_idx] * rng.lognormal(0, 0.6)
            travelling = rng.random() < 0.03
            dist = rng.exponential(300 if travelling else 6)
            device = devices[rng.integers(0, n_devices)] if rng.random() > 0.02 else f"dev_{cust}_new{txn_id}"
            rows.append(
                dict(
                    transaction_id=txn_id,
                    customer_id=cust,
                    timestamp=ts,
                    amount=round(float(amount), 2),
                    merchant_category=CATEGORIES[cat_idx],
                    distance_from_home_km=round(float(dist), 1),
                    device_id=device,
                    is_online=bool(cat_idx in (5, 7) or rng.random() < 0.15),
                    is_fraud=0,
                )
            )
            txn_id += 1

        # Account takeover episode.
        if rng.random() < fraud_episode_rate:
            compromise = start + pd.Timedelta(seconds=float(rng.uniform(5 * 86400, n_days * 86400 - 86400)))
            fraud_device = f"dev_fraud_{cust}"
            far_away = rng.random() < 0.6
            n_fraud = rng.integers(2, 9)
            offset = 0.0
            for _ in range(n_fraud):
                offset += rng.exponential(25 * 60)  # minutes apart, a burst
                ts = compromise + pd.Timedelta(seconds=offset)
                # Some fraud happens at night, but not all of it.
                if rng.random() < 0.4:
                    ts = ts.normalize() + pd.Timedelta(hours=int(rng.integers(0, 5)), minutes=int(rng.integers(0, 60)))
                cat_idx = rng.choice(len(CATEGORIES), p=FRAUD_CATEGORY_WEIGHTS)
                amount = typical_spend * CATEGORY_SCALE[cat_idx] * rng.lognormal(0.5, 0.7)
                rows.append(
                    dict(
                        transaction_id=txn_id,
                        customer_id=cust,
                        timestamp=ts,
                        amount=round(float(amount), 2),
                        merchant_category=CATEGORIES[cat_idx],
                        distance_from_home_km=round(float(rng.exponential(800) if far_away else rng.exponential(10)), 1),
                        device_id=fraud_device if rng.random() < 0.75 else devices[0],
                        is_online=bool(rng.random() < 0.7),
                        is_fraud=1,
                    )
                )
                txn_id += 1

    df = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = generate_transactions()
    print(data.head())
    print(f"{len(data):,} transactions, fraud rate {data.is_fraud.mean():.3%}")
