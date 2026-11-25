"""Module 5 - Fraud Prediction: score one transaction and explain the score."""

from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd
import streamlit as st

from fraudshield.models import scoring
from fraudshield.storage import db
from fraudshield.ui import model_state as ms
from fraudshield.ui.components import escape_markdown, render_empty_state, render_header
from fraudshield.ui.prediction_view import DISCLAIMER, show_result

render_header(
    "Fraud Prediction",
    "Score a single transaction and see which of its details made the model "
    "more or less suspicious.",
    eyebrow="Module 5 · Machine learning",
)

model, meta, parts = ms.get_model()
threshold = meta["decision_threshold"]
st.info(
    f"Model: {meta['selected_model']}, trained {meta['created_at'][:10]}. {DISCLAIMER}",
    icon=":material/info:",
)

pick_tab, manual_tab = st.tabs(["Pick a transaction", "Enter details"])

# --- Pick from the active dataset ---------------------------------------------
with pick_tab:
    active = db.get_active_dataset()
    if active is None:
        render_empty_state(
            "No active dataset",
            "Save a dataset in Dataset Management to pick transactions from it, "
            "or use the 'Enter details' tab.",
        )
    else:
        scores = ms.ensure_scores(active, model, meta)
        data = ms.load_dataset(active.id)
        st.caption(f"Dataset: {escape_markdown(active.name)} · {len(data):,} transactions")

        rng = np.random.default_rng()
        buttons = st.columns(4)
        if buttons[0].button("Riskiest transaction"):
            st.session_state["pick_row"] = int(scores["fraud_probability"].idxmax())
        if buttons[1].button("Random transaction"):
            st.session_state["pick_row"] = int(rng.choice(data.index))
        labelled = "is_fraud" in data and data["is_fraud"].notna().any()
        if labelled:
            fraud_rows = data.index[data["is_fraud"] == 1]
            if buttons[2].button("Random known fraud") and len(fraud_rows):
                st.session_state["pick_row"] = int(rng.choice(fraud_rows))
            legit_rows = data.index[data["is_fraud"] == 0]
            if buttons[3].button("Random legitimate") and len(legit_rows):
                st.session_state["pick_row"] = int(rng.choice(legit_rows))

        with st.form("find-id"):
            typed = st.text_input("Or enter a transaction ID", max_chars=100)
            if st.form_submit_button("Find"):
                found = db.get_transaction(active.id, typed.strip())
                if found.empty:
                    st.error("No transaction with that ID in the active dataset.")
                else:
                    st.session_state["pick_row"] = int(found.index[0])

        row_id = st.session_state.get("pick_row")
        if row_id is not None and row_id in data.index:
            row = data.loc[[row_id]]
            st.markdown(f"#### Transaction {escape_markdown(row['transaction_id'].iloc[0])}")
            show_result(
                row, parts, ms.background(active.id, meta["model_sha256"], parts), threshold
            )
        else:
            st.caption("Choose a transaction with the buttons or by ID to see its score.")

# --- Manual entry ----------------------------------------------------------------
with manual_tab:
    st.caption(
        "Describe a transaction. Fields you leave at their defaults use typical values. "
        "Personal details such as names are never needed."
    )
    categories = ms.model_categories(parts)
    with st.form("manual"):
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Amount", min_value=0.0, value=120.0, step=10.0)
        day = c2.date_input("Date", value=date(2020, 7, 15))
        at = c3.time_input("Time", value=time(14, 30))
        c4, c5, c6 = st.columns(3)
        category = c4.selectbox("Spending category", categories)
        age = c5.number_input("Customer age (years)", min_value=16, max_value=110, value=45)
        distance = c6.number_input(
            "Distance to merchant (km)", min_value=0.0, max_value=500.0, value=75.0
        )
        c7, c8 = st.columns(2)
        city_pop = c7.number_input("City population", min_value=0, value=20_000, step=1_000)
        card = c8.text_input(
            "Card ID (optional)",
            help="If the card appeared in the training data, the model compares the amount "
            "with that card's usual spending. Otherwise the overall average is used.",
            max_chars=40,
        )
        submitted = st.form_submit_button("Score transaction", type="primary")

    if submitted:
        row = scoring.manual_transaction(
            amount,
            pd.Timestamp.combine(day, at),
            category,
            card.strip(),
            float(age),
            float(distance),
            float(city_pop),
        )
        known = card.strip() in ms.card_profile(parts).card_mean_
        if card.strip() and not known:
            st.caption("This card was not in the training data, so the overall profile is used.")
        background_source = db.get_active_dataset()
        if background_source is None:
            st.warning(
                "Explanations need typical transactions for comparison. Save a dataset in "
                "Dataset Management to see them; the score below is still exact.",
                icon=":material/warning:",
            )
            p = float(model.predict_proba(row)[:, 1][0])
            st.metric("Fraud probability", f"{p:.1%}")
        else:
            bg = ms.background(background_source.id, meta["model_sha256"], parts)
            show_result(row, parts, bg, threshold, show_details=False)
