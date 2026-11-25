"""Scoring transactions with the saved model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudshield.config import SETTINGS, RiskThresholds
from fraudshield.risk.tiers import assign_tiers

SCORE_CHUNK = 50_000  # rows scored at a time, to keep memory flat on large datasets


def score_frame(
    model,
    df: pd.DataFrame,
    decision_threshold: float,
    thresholds: RiskThresholds = SETTINGS.risk_thresholds,
    chunk: int = SCORE_CHUNK,
    progress=None,
) -> pd.DataFrame:
    """Fraud probability, yes/no decision and risk tier for every row of ``df``.

    The result has the same index as ``df``.
    """
    if df.empty:
        return pd.DataFrame(columns=["fraud_probability", "predicted_fraud", "risk_tier"])
    parts = []
    for start in range(0, len(df), chunk):
        parts.append(model.predict_proba(df.iloc[start : start + chunk])[:, 1])
        if progress:
            progress(min(start + chunk, len(df)) / len(df))
    proba = np.clip(np.concatenate(parts), 0.0, 1.0)
    return pd.DataFrame(
        {
            "fraud_probability": proba,
            "predicted_fraud": (proba >= decision_threshold).astype("int8"),
            "risk_tier": assign_tiers(proba, thresholds),
        },
        index=df.index,
    )


def manual_transaction(
    amount: float,
    timestamp: pd.Timestamp,
    category: str,
    card_id: str = "",
    age_years: float | None = None,
    distance_km: float | None = None,
    city_pop: float | None = None,
) -> pd.DataFrame:
    """Build a one-row dataset from form inputs, in the canonical schema.

    The model computes distance from coordinates, so the customer is placed at
    (0, 0) and the merchant due north at the requested distance
    (1 degree of latitude is about 111.195 km).
    """
    ts = pd.Timestamp(timestamp)
    row = {
        "transaction_id": "manual-entry",
        "timestamp": ts,
        "amount": float(amount),
        "card_id": card_id or "unknown-card",
        "category": category,
        "city_pop": float(city_pop) if city_pop is not None else np.nan,
        "dob": ts - pd.Timedelta(days=age_years * 365.25) if age_years is not None else pd.NaT,
        "lat": 0.0 if distance_km is not None else np.nan,
        "long": 0.0 if distance_km is not None else np.nan,
        "merch_lat": distance_km / 111.195 if distance_km is not None else np.nan,
        "merch_long": 0.0 if distance_km is not None else np.nan,
    }
    df = pd.DataFrame([row])
    df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
    df["dob"] = pd.to_datetime(df["dob"]).astype("datetime64[ns]")
    return df
