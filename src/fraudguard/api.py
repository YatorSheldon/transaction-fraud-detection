"""REST API for real-time transaction scoring.

Run with:  uvicorn fraudguard.api:app --reload   (then open http://localhost:8000/docs)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraudguard import __version__
from fraudguard.data import CATEGORIES
from fraudguard.model import load_bundle, score_transaction

app = FastAPI(
    title="FraudGuard API",
    description="Scores card transactions for fraud risk in real time.",
    version=__version__,
)


class Transaction(BaseModel):
    """A transaction plus the customer's recent behaviour (normally served by a feature store)."""

    amount: float = Field(..., gt=0, examples=[349.99])
    timestamp: datetime = Field(..., examples=["2026-04-12T02:41:00"])
    merchant_category: Literal[tuple(CATEGORIES)] = Field(..., examples=["electronics"])  # type: ignore[valid-type]
    distance_from_home_km: float = Field(..., ge=0, examples=[640.0])
    is_online: bool = Field(..., examples=[True])
    is_new_device: bool = Field(False, description="First time this customer uses this device")
    is_new_category: bool = Field(False, description="First time this customer shops in this category")
    customer_avg_amount: float = Field(..., gt=0, description="Customer's average past spend", examples=[42.5])
    txns_last_1h: int = Field(0, ge=0)
    txns_last_24h: int = Field(0, ge=0)
    seconds_since_last_txn: float = Field(86400, ge=0)
    customer_history_length: int = Field(100, ge=0, description="Number of past transactions on the card")


class Score(BaseModel):
    fraud_probability: float
    flagged: bool
    threshold: float
    model: str
    model_version: str


@app.get("/health")
def health() -> dict:
    try:
        bundle = load_bundle()
    except FileNotFoundError:
        return {"status": "no_model"}
    return {"status": "ok", "model": bundle["model_name"], "version": bundle["version"]}


@app.post("/predict", response_model=Score)
def predict(txn: Transaction) -> Score:
    try:
        result = score_transaction(txn.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Score(**result)
