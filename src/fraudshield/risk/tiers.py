"""Turning a calibrated fraud probability into a risk tier and a recommended action.

The cut-offs come from ``config.RiskThresholds`` (30% / 60% / 85% by default).
They are business choices, not model output: the Model Performance page shows
what fraud rate and workload each cut-off produces on the test period, so they
can be justified or changed with data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraudshield.config import RISK_TIERS, SETTINGS, RiskThresholds


@dataclass(frozen=True)
class TierInfo:
    name: str
    action: str
    meaning: str


TIER_INFO: dict[str, TierInfo] = {
    "Low": TierInfo(
        "Low",
        "Approve normally",
        "Looks like normal behaviour. No review needed.",
    ),
    "Medium": TierInfo(
        "Medium",
        "Monitor / batch review",
        "Some unusual signals. Review in a daily batch or watch for repeats on the card.",
    ),
    "High": TierInfo(
        "High",
        "Analyst review today",
        "Strong fraud signals. An analyst should check it the same day.",
    ),
    "Critical": TierInfo(
        "Critical",
        "Hold and review urgently",
        "Very likely fraud. Hold the transaction and contact the cardholder.",
    ),
}


def assign_tiers(probability, thresholds: RiskThresholds = SETTINGS.risk_thresholds) -> np.ndarray:
    """Vectorised: probability array -> array of tier names."""
    p = np.asarray(probability, dtype="float64")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be between 0 and 1.")
    edges = [thresholds.medium, thresholds.high, thresholds.critical]
    return np.asarray(RISK_TIERS, dtype=object)[np.searchsorted(edges, p, side="right")]


def assign_tier(probability: float, thresholds: RiskThresholds = SETTINGS.risk_thresholds) -> str:
    return str(assign_tiers([probability], thresholds)[0])
