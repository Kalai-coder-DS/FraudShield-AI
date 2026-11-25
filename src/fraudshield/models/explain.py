"""Why did the model give this transaction its score?

Method (a simple, model-agnostic version of what SHAP does):

    For one feature at a time, replace the transaction's value with values
    taken from typical transactions (a "background" sample), re-score, and
    see how much the fraud probability drops or rises.

    contribution(feature) = P(this transaction) - average P(with that feature
                                                   replaced by typical values)

A large positive contribution means "this feature pushed the score up".
The numbers are in probability points, so they are easy to read, but they are
an approximation: features also interact, so contributions need not add up to
the total score exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudshield.features.basic import WEEKDAY_NAMES, haversine_km

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "Amount": ("log_amount",),
    "Amount vs. card's usual": ("card_amount_z", "card_known"),
    "Time of day": ("hour", "is_night"),
    "Day of week": ("weekday",),
    "Spending category": ("category_",),  # prefix: all one-hot columns
    "Customer age": ("age_years",),
    "Distance to merchant": ("distance_km",),
    "City population": ("log_city_pop",),
}


@dataclass
class ModelParts:
    """The pieces inside the saved CalibratedClassifierCV(FrozenEstimator(pipeline))."""

    features: object  # pipeline steps before the model: raw rows -> numeric matrix
    classifier: object  # e.g. RandomForestClassifier
    calibrator: object  # isotonic regression: raw score -> calibrated probability
    feature_names: list[str]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.features.transform(df), dtype="float64")

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Use the same raw score the calibrator was fitted on: scikit-learn calibrates
        # decision_function when the model has one (e.g. Logistic Regression) and
        # predict_proba otherwise (trees, forests).
        if hasattr(self.classifier, "decision_function"):
            raw = self.classifier.decision_function(X)
        else:
            raw = self.classifier.predict_proba(X)[:, 1]
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)


def model_parts(model) -> ModelParts:
    calibrated = model.calibrated_classifiers_[0]
    pipeline = calibrated.estimator.estimator  # FrozenEstimator -> Pipeline
    return ModelParts(
        features=pipeline[:-1],
        classifier=pipeline[-1],
        calibrator=calibrated.calibrators[0],
        feature_names=list(pipeline.named_steps["preprocess"].get_feature_names_out()),
    )


def _group_columns(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {}
    for group, members in FEATURE_GROUPS.items():
        idx = [
            i
            for i, name in enumerate(feature_names)
            if any(name == m or (m.endswith("_") and name.startswith(m)) for m in members)
        ]
        if idx:
            groups[group] = idx
    return groups


def explain(
    parts: ModelParts, row: pd.DataFrame, background: np.ndarray
) -> tuple[float, pd.DataFrame]:
    """Return (probability, table of contributions) for a single transaction row.

    ``background`` is an already transformed matrix of typical transactions
    (see ``background_matrix``); 100-300 rows are plenty.
    """
    if len(row) != 1:
        raise ValueError("explain() takes exactly one transaction.")
    x = parts.transform(row)
    probability = float(parts.predict(x)[0])
    rows = []
    for group, cols in _group_columns(parts.feature_names).items():
        replaced = np.repeat(x, len(background), axis=0)
        replaced[:, cols] = background[:, cols]
        typical = float(parts.predict(replaced).mean())
        rows.append({"feature": group, "contribution": probability - typical})
    table = pd.DataFrame(rows).sort_values("contribution", ascending=False, key=abs)
    return probability, table.reset_index(drop=True)


def background_matrix(parts: ModelParts, df: pd.DataFrame, size: int = 200, seed: int = 0):
    """Transformed sample of typical (non-fraud where known) transactions."""
    pool = df
    if "is_fraud" in df and (df["is_fraud"] == 0).sum() >= size:
        pool = df[df["is_fraud"] == 0]
    sample = pool.sample(min(size, len(pool)), random_state=seed)
    return parts.transform(sample)


def describe_value(group: str, row: pd.Series, card_amount_z: float | None = None) -> str:
    """Human-readable value of a feature group for one transaction."""
    ts = row.get("timestamp")
    if group == "Amount":
        return f"{row['amount']:,.2f}"
    if group == "Amount vs. card's usual" and card_amount_z is not None:
        if abs(card_amount_z) < 0.5:
            return "close to this card's usual amount"
        direction = "above" if card_amount_z > 0 else "below"
        return f"{abs(card_amount_z):.1f} standard deviations {direction} this card's usual"
    if group == "Time of day" and pd.notna(ts):
        return f"{pd.Timestamp(ts):%H:%M}"
    if group == "Day of week" and pd.notna(ts):
        return WEEKDAY_NAMES[pd.Timestamp(ts).day_of_week]
    if group == "Spending category":
        return str(row.get("category", ""))
    if group == "Customer age" and pd.notna(row.get("dob")) and pd.notna(ts):
        return f"{(pd.Timestamp(ts) - pd.Timestamp(row['dob'])).days / 365.25:.0f} years"
    if group == "Distance to merchant" and pd.notna(row.get("merch_lat")):
        km = haversine_km(row["lat"], row["long"], row["merch_lat"], row["merch_long"])
        return f"{float(km):,.0f} km"
    if group == "City population" and pd.notna(row.get("city_pop")):
        return f"{row['city_pop']:,.0f}"
    return "—"


def reasons(table: pd.DataFrame, values: dict[str, str], top: int = 3, min_points: float = 0.02):
    """Plain sentences for the features that raised the score the most."""
    up = table[table["contribution"] >= min_points].head(top)
    return [
        f"{r.feature} ({values.get(r.feature, '—')}) raised the fraud probability by "
        f"about {r.contribution * 100:.0f} percentage points."
        for r in up.itertuples()
    ]
