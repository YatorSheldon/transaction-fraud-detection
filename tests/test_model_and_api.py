"""Trains a small model once, then checks quality, the scoring function and the API."""

import json

import pytest
from fastapi.testclient import TestClient

import fraudguard.model as model_module
from fraudguard.train import main as train

NORMAL = {
    "amount": 35.0,
    "timestamp": "2026-04-15T13:10:00",
    "merchant_category": "grocery",
    "distance_from_home_km": 2.0,
    "is_online": False,
    "is_new_device": False,
    "is_new_category": False,
    "customer_avg_amount": 40.0,
    "txns_last_1h": 0,
    "txns_last_24h": 2,
    "seconds_since_last_txn": 14400,
    "customer_history_length": 200,
}
SUSPICIOUS = {
    **NORMAL,
    "amount": 950.0,
    "timestamp": "2026-04-15T03:05:00",
    "merchant_category": "electronics",
    "distance_from_home_km": 900.0,
    "is_online": True,
    "is_new_device": True,
    "is_new_category": True,
    "txns_last_1h": 3,
    "txns_last_24h": 4,
    "seconds_since_last_txn": 300,
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    summary = train(n_customers=600, n_days=90, seed=7, out_dir=out)
    model_module.load_bundle.cache_clear()
    original_path = model_module.MODEL_PATH
    model_module.MODEL_PATH = out / "models" / "fraud_model.joblib"  # load_bundle() reads this at call time
    yield out, summary
    model_module.MODEL_PATH = original_path
    model_module.load_bundle.cache_clear()


def test_model_beats_random_by_a_wide_margin(trained):
    _, summary = trained
    best = summary["models"][summary["best_model"]]
    assert best["pr_auc"] > 10 * summary["data"]["fraud_rate"]
    assert best["roc_auc"] > 0.9
    assert best["total_cost"] < best["cost_if_no_model"]


def test_reports_written(trained):
    out, _ = trained
    assert (out / "reports" / "figures" / "pr_curve.png").exists()
    metrics = json.loads((out / "reports" / "metrics.json").read_text())
    assert metrics["best_model"] in metrics["models"]


def test_suspicious_scores_higher_than_normal(trained):
    bundle = model_module.load_bundle(str(model_module.MODEL_PATH))
    normal = model_module.score_transaction(NORMAL, bundle)
    suspicious = model_module.score_transaction(SUSPICIOUS, bundle)
    assert suspicious["fraud_probability"] > normal["fraud_probability"]
    assert not normal["flagged"]


def test_api_predict_and_validation(trained):
    from fraudguard.api import app

    model_module.load_bundle.cache_clear()
    client = TestClient(app)

    assert client.get("/health").json()["status"] == "ok"
    resp = client.post("/predict", json=SUSPICIOUS)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert isinstance(body["flagged"], bool)

    bad = client.post("/predict", json={**NORMAL, "amount": -5})
    assert bad.status_code == 422
    bad_cat = client.post("/predict", json={**NORMAL, "merchant_category": "casino"})
    assert bad_cat.status_code == 422
