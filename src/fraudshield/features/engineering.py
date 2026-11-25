"""Features used by the fraud models.

Two kinds of features, handled differently to prevent data leakage:

1. **Row features** (``add_model_features``): calculated from one row only,
   for example the hour of the transaction. Nothing is learned from other
   rows, so they are safe to calculate anywhere.

2. **Card profile features** (``CardProfile``): compare a transaction with the
   card's usual spending ("is this amount unusual *for this card*?"). The
   "usual spending" is a statistic learned from data, so - exactly like a
   scaler - it is learned in ``fit`` from the TRAINING period only and then
   applied to validation, test and new data. Cards never seen in training get
   the overall average profile and ``card_known = 0``.

Why not "transactions by this card in the last 24 hours"? That count depends
on having every transaction of the card. A sample (or an uploaded extract)
contains only some of them, so the same feature would mean something different
at training time and scoring time (train/serve skew). The card profile is
learned once from complete training data, so it does not have that problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fraudshield.features.basic import add_basic_features

# Late night, 22:00-03:59: a common fraud-domain pattern; EDA shows the effect per hour.
NIGHT_HOURS = frozenset({22, 23, 0, 1, 2, 3})

NUMERIC_FEATURES: tuple[str, ...] = (
    "log_amount",
    "hour",
    "is_night",
    "weekday",
    "age_years",
    "distance_km",
    "log_city_pop",
    "card_amount_z",
    "card_known",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("category",)

# Columns deliberately NOT given to the model, with the reason (shown in the app).
EXCLUDED_FEATURES: dict[str, str] = {
    "gender": "Protected attribute. Using it could make the model treat people differently "
    "by gender; it is kept for analysis only.",
    "job": "Hundreds of distinct values and close to personal data; high risk of overfitting.",
    "merchant": "Hundreds of distinct merchant names; the model would memorise names "
    "instead of learning patterns.",
    "city, state": "Location is already represented by distance and city population; "
    "raw names would overfit.",
    "transaction_id": "An identifier carries no information about fraud.",
    "card_id": "Used only to look up the card's spending profile, never as a feature itself "
    "(the model must not memorise individual cards).",
}

_MIN_CARD_HISTORY = 5  # fewer training transactions than this -> use the overall profile
_MIN_STD = 0.05


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features for the model. Missing optional inputs become NaN columns."""
    out = add_basic_features(df)
    out["is_night"] = out["hour"].isin(NIGHT_HOURS).astype("int8")
    out["log_city_pop"] = np.log1p(out["city_pop"]) if "city_pop" in out else np.nan
    for col in ("age_years", "distance_km"):
        if col not in out:
            out[col] = np.nan
    return out


class CardProfile(BaseEstimator, TransformerMixin):
    """Learns each card's typical (log) amount from training data.

    Adds:
      card_amount_z  how many standard deviations the amount is above/below the
                     card's usual amount (overall profile for unknown cards)
      card_known     1 if the card had at least ``min_history`` training transactions
    """

    def __init__(self, min_history: int = _MIN_CARD_HISTORY):
        self.min_history = min_history

    def fit(self, X: pd.DataFrame, y=None):
        log_amount = np.log1p(X["amount"].astype("float64"))
        cards = X["card_id"].astype(str)
        stats = log_amount.groupby(cards).agg(["mean", "std", "size"])
        stats = stats[stats["size"] >= self.min_history]
        self.global_mean_ = float(log_amount.mean())
        self.global_std_ = float(max(log_amount.std(), _MIN_STD))
        self.card_mean_ = stats["mean"].to_dict()
        self.card_std_ = stats["std"].clip(lower=_MIN_STD).fillna(self.global_std_).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        cards = out["card_id"].astype(str)
        mean = cards.map(self.card_mean_)
        std = cards.map(self.card_std_)
        known = mean.notna()
        mean = mean.fillna(self.global_mean_).astype("float64")
        std = std.fillna(self.global_std_).astype("float64")
        out["card_amount_z"] = (np.log1p(out["amount"].astype("float64")) - mean) / std
        out["card_known"] = known.astype("int8")
        return out

    def get_feature_names_out(self, input_features=None):  # pragma: no cover - sklearn API
        return np.asarray(
            list(input_features or []) + ["card_amount_z", "card_known"], dtype=object
        )
