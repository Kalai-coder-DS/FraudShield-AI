"""Module 7 - Transaction Investigation: search, filter and inspect transactions."""

from __future__ import annotations

import math

import streamlit as st

from fraudshield.config import RISK_TIERS, SETTINGS
from fraudshield.storage import db
from fraudshield.ui import model_state as ms
from fraudshield.ui.components import escape_markdown, mask_card, render_header
from fraudshield.ui.prediction_view import show_result

render_header(
    "Transaction Investigation",
    "Search and filter scored transactions, then open one to see its details and "
    "why the model rated it that way.",
    eyebrow="Module 7 · Investigation",
)

model, meta, parts = ms.get_model()
active = ms.get_active_dataset()
ms.ensure_scores(active, model, meta)
data = ms.load_dataset(active.id)
labelled = "is_fraud" in data and data["is_fraud"].notna().any()

# --- Filters ----------------------------------------------------------------------
with st.form("filters"):
    c1, c2, c3 = st.columns(3)
    text = c1.text_input("Transaction ID contains", max_chars=100)
    tiers = c2.multiselect("Risk tier", list(RISK_TIERS))
    categories = c3.multiselect("Category", sorted(data["category"].astype(str).unique()))
    c4, c5, c6, c7 = st.columns(4)
    min_amount = c4.number_input("Min amount", min_value=0.0, value=0.0, step=10.0)
    max_amount = c5.number_input(
        "Max amount", min_value=0.0, value=float(math.ceil(data["amount"].max())), step=10.0
    )
    first, last = data["timestamp"].min().date(), data["timestamp"].max().date()
    dates = c6.date_input("Date range", value=(first, last), min_value=first, max_value=last)
    label_choice = c7.selectbox(
        "Known label", ["Any", "Fraud", "Legitimate"], disabled=not labelled
    )
    flagged_only = st.checkbox("Only transactions the model flags for review")
    st.form_submit_button("Apply filters", type="primary")

date_from, date_to = dates if isinstance(dates, tuple) and len(dates) == 2 else (first, last)
f = db.TransactionFilter(
    text=text,
    tiers=tuple(tiers),
    categories=tuple(categories),
    min_amount=min_amount or None,
    max_amount=max_amount,
    date_from=str(date_from),
    date_to=str(date_to),
    label={"Any": None, "Fraud": 1, "Legitimate": 0}[label_choice],
    flagged_only=flagged_only,
)
if min_amount > max_amount:
    st.error("Min amount is larger than max amount.")
    st.stop()

total = db.count_transactions(active.id, f)
st.caption(f"{total:,} of {len(data):,} transactions match · riskiest first")
if total == 0:
    st.info("No transactions match these filters.", icon=":material/search_off:")
    st.stop()

size = SETTINGS.table_page_size
pages = max(1, math.ceil(total / size))
page = st.number_input(f"Page (of {pages:,})", 1, pages, 1)
rows = db.search_transactions(active.id, f, int(page), size)
columns = [
    c
    for c in (
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
    if c in rows
]
shown = rows[columns].copy()
if "card_id" in shown:
    shown["card_id"] = shown["card_id"].astype(object).map(mask_card)
shown["fraud_probability"] = shown["fraud_probability"] * 100
event = st.dataframe(
    shown,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "fraud_probability": st.column_config.ProgressColumn(
            "Fraud probability", format="%.1f%%", min_value=0, max_value=100
        ),
        "predicted_fraud": st.column_config.CheckboxColumn("Flagged"),
        "is_fraud": st.column_config.NumberColumn("Known label (1 = fraud)"),
    },
    key=f"table-{page}",
)
st.caption("Click a row to inspect it.")

# --- Detail ------------------------------------------------------------------------
selected = event.selection.rows if event and event.selection else []
if selected:
    row_number = rows.index[selected[0]]
    row = data.loc[[row_number]]
    st.divider()
    st.markdown(f"#### Transaction {escape_markdown(row['transaction_id'].iloc[0])}")
    show_result(
        row,
        parts,
        ms.background(active.id, meta["model_sha256"], parts),
        meta["decision_threshold"],
    )
