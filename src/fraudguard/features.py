"""Feature engineering: raw transactions -> behavioural signals.

Every feature for a transaction is computed only from that customer's *earlier*
transactions, so nothing from the future leaks into training. In production these
history-based values would come from a feature store; the API accepts them directly.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "log_amount",
    "amount_ratio_to_customer_avg",
    "hour",
    "is_night",
    "day_of_week",
    "log_distance_from_home_km",
    "is_online",
    "is_new_device",
    "is_new_category",
    "txns_last_1h",
    "txns_last_24h",
    "log_seconds_since_last_txn",
    "customer_history_length",
]
CATEGORICAL_FEATURES = ["merchant_category"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Used when a customer has no history yet.
DEFAULT_SECONDS_SINCE_LAST = 30 * 86400


def add_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add history-based features to a raw transaction table.

    Expects columns: customer_id, timestamp, amount, merchant_category,
    distance_from_home_km, device_id, is_online.
    """
    out = df.sort_values(["customer_id", "timestamp"], kind="stable").copy()
    g = out.groupby("customer_id", sort=False)

    # Customer's average spend before this transaction.
    prior_sum = g["amount"].cumsum() - out["amount"]
    prior_n = g.cumcount()
    global_median = float(out["amount"].median())
    out["customer_avg_amount"] = np.where(prior_n > 0, prior_sum / prior_n.replace(0, 1), global_median)
    out["customer_history_length"] = prior_n

    # First time we see this device / category for this customer.
    out["is_new_device"] = (~out.duplicated(["customer_id", "device_id"])).astype(int)
    out["is_new_category"] = (~out.duplicated(["customer_id", "merchant_category"])).astype(int)
    # A customer's very first transaction is not "new" in a suspicious sense.
    out.loc[prior_n == 0, ["is_new_device", "is_new_category"]] = 0

    # Velocity: how many earlier transactions in the last hour / day.
    # Seconds since epoch (independent of the datetime resolution pandas uses).
    secs = ((out["timestamp"] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy()
    cust = out["customer_id"].to_numpy()
    last_1h = np.zeros(len(out), dtype=int)
    last_24h = np.zeros(len(out), dtype=int)
    since_last = np.full(len(out), float(DEFAULT_SECONDS_SINCE_LAST))
    boundaries = np.flatnonzero(np.r_[True, cust[1:] != cust[:-1], True])
    for s, e in pairwise(boundaries):
        t = secs[s:e]
        idx = np.arange(e - s)
        last_1h[s:e] = idx - np.searchsorted(t, t - 3600, side="left")
        last_24h[s:e] = idx - np.searchsorted(t, t - 86400, side="left")
        if e - s > 1:
            since_last[s + 1 : e] = np.diff(t)
    out["txns_last_1h"] = last_1h
    out["txns_last_24h"] = last_24h
    out["seconds_since_last_txn"] = since_last

    return finalise_features(out).sort_index()


def finalise_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw + context columns into the exact model inputs.

    Shared by training and the API so both compute features the same way.
    Expects: amount, customer_avg_amount, timestamp, distance_from_home_km, is_online,
    is_new_device, is_new_category, txns_last_1h, txns_last_24h,
    seconds_since_last_txn, customer_history_length, merchant_category.
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["log_amount"] = np.log1p(out["amount"])
    out["amount_ratio_to_customer_avg"] = out["amount"] / out["customer_avg_amount"].clip(lower=1.0)
    out["hour"] = ts.dt.hour
    out["is_night"] = ts.dt.hour.between(0, 5).astype(int)
    out["day_of_week"] = ts.dt.dayofweek
    out["log_distance_from_home_km"] = np.log1p(out["distance_from_home_km"].clip(lower=0))
    out["is_online"] = out["is_online"].astype(int)
    out["log_seconds_since_last_txn"] = np.log1p(out["seconds_since_last_txn"].clip(lower=0))
    return out
