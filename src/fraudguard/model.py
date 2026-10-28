"""Load the trained model and score a single transaction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from fraudguard.features import finalise_features

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "fraud_model.joblib"


@lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> dict:
    path = Path(path) if path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Train one first: python -m fraudguard.train")
    return joblib.load(path)


def score_transaction(txn: dict, bundle: dict | None = None) -> dict:
    """Score one transaction.

    ``txn`` holds the raw transaction plus customer context (see ``fraudguard.api.Transaction``).
    Returns the fraud probability, the decision and the threshold used.
    """
    bundle = bundle or load_bundle()
    frame = finalise_features(pd.DataFrame([txn]))
    proba = float(bundle["model"].predict_proba(frame[bundle["features"]])[:, 1][0])
    threshold = float(bundle["threshold"])
    return {
        "fraud_probability": round(proba, 4),
        "flagged": proba >= threshold,
        "threshold": threshold,
        "model": bundle["model_name"],
        "model_version": bundle["version"],
    }
