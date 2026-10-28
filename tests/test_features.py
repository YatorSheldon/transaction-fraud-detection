import pandas as pd

from fraudguard.data import generate_transactions
from fraudguard.features import ALL_FEATURES, add_behavioural_features


def _toy():
    t0 = pd.Timestamp("2026-01-01 10:00")
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 1, 1, 2],
            "timestamp": [t0, t0 + pd.Timedelta(minutes=10), t0 + pd.Timedelta(minutes=30), t0 + pd.Timedelta(days=2), t0],
            "amount": [10.0, 20.0, 300.0, 15.0, 50.0],
            "merchant_category": ["grocery", "grocery", "electronics", "grocery", "fuel"],
            "distance_from_home_km": [1.0, 2.0, 500.0, 1.0, 3.0],
            "device_id": ["a", "a", "b", "a", "z"],
            "is_online": [False, False, True, False, False],
        }
    )


def test_velocity_counts_only_past_transactions():
    out = add_behavioural_features(_toy())
    assert out["txns_last_1h"].tolist() == [0, 1, 2, 0, 0]
    assert out["txns_last_24h"].tolist() == [0, 1, 2, 0, 0]


def test_customer_average_uses_only_history():
    out = add_behavioural_features(_toy())
    # Third transaction: average of the two before it, not including itself.
    assert out.loc[2, "customer_avg_amount"] == 15.0
    assert out.loc[2, "amount_ratio_to_customer_avg"] == 20.0


def test_new_device_and_category_flags():
    out = add_behavioural_features(_toy())
    assert out["is_new_device"].tolist() == [0, 0, 1, 0, 0]
    assert out["is_new_category"].tolist() == [0, 0, 1, 0, 0]


def test_output_keeps_original_order_and_all_features():
    raw = _toy()
    out = add_behavioural_features(raw)
    assert out.index.tolist() == raw.index.tolist()
    assert set(ALL_FEATURES).issubset(out.columns)
    assert not out[ALL_FEATURES].isna().any().any()


def test_generator_is_reproducible_and_has_fraud():
    a = generate_transactions(n_customers=50, n_days=20, seed=1)
    b = generate_transactions(n_customers=50, n_days=20, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert 0 < a["is_fraud"].sum() < len(a)
