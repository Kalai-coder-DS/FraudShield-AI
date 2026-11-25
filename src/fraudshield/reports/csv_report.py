"""CSV export of scored transactions."""

from __future__ import annotations

import pandas as pd

from fraudshield.ui.components import mask_card

EXPORT_COLUMNS = (
    "transaction_id",
    "timestamp",
    "amount",
    "category",
    "merchant",
    "card_id",
    "fraud_probability",
    "risk_tier",
    "predicted_fraud",
    "is_fraud",
)


def scored_transactions_csv(
    data: pd.DataFrame,
    scores: pd.DataFrame,
    tiers: tuple[str, ...] | None = None,
    flagged_only: bool = False,
) -> bytes:
    """Scored transactions as CSV bytes, riskiest first.

    Card numbers are masked to the last four digits, so the file can be shared
    with less risk. ``tiers`` limits the export, e.g. ("High", "Critical");
    ``flagged_only`` keeps only transactions the model flags for review.
    The file starts with a byte-order mark so Excel opens it as UTF-8.
    """
    df = data.join(scores)
    if tiers:
        df = df[df["risk_tier"].isin(tiers)]
    if flagged_only:
        df = df[df["predicted_fraud"] == 1]
    df = df.sort_values("fraud_probability", ascending=False, kind="stable")
    out = df[[c for c in EXPORT_COLUMNS if c in df]].copy()
    if "card_id" in out:
        out["card_id"] = out["card_id"].astype(object).map(mask_card)
    out["fraud_probability"] = out["fraud_probability"].round(4)
    return out.to_csv(index=False).encode("utf-8-sig")
