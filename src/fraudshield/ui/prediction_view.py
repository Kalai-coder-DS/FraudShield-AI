"""The "result card" for one scored transaction, shared by the Fraud Prediction
and Transaction Investigation pages."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from fraudshield.models import explain
from fraudshield.models.charts import contribution_chart
from fraudshield.reports.summary import format_probability
from fraudshield.risk.tiers import TIER_INFO, assign_tier
from fraudshield.ui import theme
from fraudshield.ui.components import escape_markdown, mask_card, render_kpi

DISCLAIMER = (
    "This is a model's estimate based on past patterns, not a confirmed fraud decision. "
    "Use it to decide what to review first."
)


def tier_badge(tier: str) -> str:
    color = theme.RISK_COLORS[tier]
    return (
        f"<span style='background:{color}26;color:{color};padding:.15rem .6rem;"
        f"border-radius:999px;font-weight:600;font-size:.85rem'>{escape(tier)}</span>"
    )


def show_result(
    row: pd.DataFrame,
    parts: explain.ModelParts,
    background,
    decision_threshold: float,
    show_details: bool = True,
) -> None:
    """Score one transaction and explain the score."""
    probability, table = explain.explain(parts, row, background)
    tier = assign_tier(probability)
    info = TIER_INFO[tier]
    flagged = probability >= decision_threshold
    record = row.iloc[0]

    cols = st.columns(3)
    with cols[0]:
        render_kpi(
            "Fraud probability",
            format_probability(probability),
            "Calibrated: of transactions scored like this, about this share were fraud.",
        )
    with cols[1]:
        render_kpi(
            "Model decision",
            "Flag for review" if flagged else "Approve",
            f"Flagged when the probability is at least {decision_threshold:.0%}.",
        )
    with cols[2]:
        render_kpi("Risk tier", tier, f"{info.action}. {info.meaning}")

    label = record.get("is_fraud")
    if label is not None and pd.notna(label):
        actual = "fraud" if int(label) == 1 else "legitimate"
        correct = bool(flagged) == (int(label) == 1)
        message = f"Known label in the data: **{actual}**. The model's decision was " + (
            "**correct**." if correct else "**wrong**."
        )
        (st.success if correct else st.warning)(message, icon=":material/fact_check:")

    st.caption(DISCLAIMER)

    z = parts.features[:-1].transform(row)["card_amount_z"].iloc[0]
    values = {g: explain.describe_value(g, record, z) for g in explain.FEATURE_GROUPS}
    st.markdown("##### Why this score?")
    reasons = explain.reasons(table, values)
    if reasons:
        for reason in reasons:
            st.markdown(f"- {escape_markdown(reason)}")
    else:
        st.markdown(
            "- No single feature raised the score noticeably; it looks like a typical transaction."
        )
    st.plotly_chart(contribution_chart(table, values), config={"displaylogo": False})
    st.caption(
        "Each bar shows how the probability changes when that feature is replaced by values "
        "from typical transactions. Red bars push towards fraud, teal bars towards legitimate. "
        "This is an approximation: features also work together."
    )

    if show_details:
        st.markdown("##### Transaction details")
        shown = record.drop(labels=[c for c in ("is_fraud",) if c in record.index])
        details = pd.DataFrame(
            {
                "Field": shown.index,
                "Value": [
                    mask_card(v) if k == "card_id" else ("" if pd.isna(v) else str(v))
                    for k, v in shown.items()
                ],
            }
        )
        st.dataframe(details, hide_index=True)


def transaction_columns() -> dict:
    """Column labels and widths for scored-transaction tables (no cut-off values)."""
    cc = st.column_config
    return {
        "transaction_id": cc.TextColumn("Transaction ID", width="medium"),
        "timestamp": cc.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm", width="medium"),
        "amount": cc.NumberColumn("Amount", format="%.2f", width="small"),
        "category": cc.TextColumn("Category", width="small"),
        "merchant": cc.TextColumn("Merchant", width="medium"),
        "card_id": cc.TextColumn("Card", width="small"),
        "fraud_probability": cc.ProgressColumn(
            "Fraud probability", format="%.1f%%", min_value=0, max_value=100, width="medium"
        ),
        "risk_tier": cc.TextColumn("Risk tier", width="small"),
        "predicted_fraud": cc.CheckboxColumn("Flagged", width="small"),
        "is_fraud": cc.NumberColumn("Known fraud (1 = yes)", width="small"),
    }
