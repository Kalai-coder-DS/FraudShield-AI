"""Row-level features that need no training data.

Each feature here is calculated from a single row only (for example, the hour
of the timestamp). Nothing is learned from other rows, so these features cannot
leak information between training and test data. They are shared by the EDA
module (Phase 3) and the model pipeline (Phase 4), so they are defined once.

Features that DO depend on other rows (for example "how many transactions did
this card make in the last 24 hours") are built in Phase 4, using past rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometres between two sets of coordinates."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(x, dtype="float64")) for x in (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with time, age, distance and log-amount columns.

    Columns are only added when the inputs they need are present.
    """
    out = df.copy()
    ts = out["timestamp"]
    out["hour"] = ts.dt.hour.astype("int8")
    out["weekday"] = ts.dt.day_of_week.astype("int8")  # 0 = Monday
    out["date"] = ts.dt.normalize()
    out["log_amount"] = np.log1p(out["amount"])

    if "dob" in out:
        out["age_years"] = (ts - out["dob"]).dt.days / 365.25

    if {"lat", "long", "merch_lat", "merch_long"} <= set(out.columns):
        out["distance_km"] = haversine_km(
            out["lat"], out["long"], out["merch_lat"], out["merch_long"]
        )
    return out
