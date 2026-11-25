"""Module 6 - Risk Intelligence: risk tiers, patterns among risky transactions,
and what a different decision threshold would mean."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraudshield.config import RISK_TIERS, SETTINGS
from fraudshield.eda.analysis import amount_band
from fraudshield.eda.charts import _style
from fraudshield.risk.tiers import TIER_INFO
from fraudshield.storage import db
from fraudshield.ui import model_state as ms
from fraudshield.ui import theme
from fraudshield.ui.components import escape_markdown, mask_card, render_header, render_kpi
from fraudshield.ui.prediction_view import transaction_columns

PLOT = {"displaylogo": False}

render_header(
    "Risk Intelligence",
    "Every transaction in the active dataset grouped into Low, Medium, High and "
    "Critical risk, with the patterns behind the risky ones.",
    eyebrow="Module 6 · Investigation",
)

model, meta, parts = ms.get_model()
active = ms.get_active_dataset()
scores = ms.ensure_scores(active, model, meta)
data = ms.load_dataset(active.id)
df = data.join(scores)
labelled = "is_fraud" in df and df["is_fraud"].notna().any()
rt = SETTINGS.risk_thresholds

st.caption(
    f"Dataset: {escape_markdown(active.name)} · {len(df):,} transactions · "
    f"model {meta['selected_model']}"
)

# --- Tier KPIs --------------------------------------------------------------------
counts = df["risk_tier"].value_counts().reindex(list(RISK_TIERS), fill_value=0)
ranges = {
    "Low": f"below {rt.medium:.0%}",
    "Medium": f"{rt.medium:.0%} – {rt.high:.0%}",
    "High": f"{rt.high:.0%} – {rt.critical:.0%}",
    "Critical": f"{rt.critical:.0%} and above",
}
for col, tier in zip(st.columns(4), RISK_TIERS, strict=True):
    with col:
        render_kpi(
            f"{tier} risk",
            f"{counts[tier]:,}",
            f"{counts[tier] / len(df):.2%} of transactions · probability {ranges[tier]}. "
            f"{TIER_INFO[tier].action}.",
        )

overview, patterns, threshold_tab, table_tab = st.tabs(
    ["Risk distribution", "Patterns in risky transactions", "Decision threshold", "High-risk list"]
)

# --- Distribution -----------------------------------------------------------------
with overview:
    fig = go.Figure(
        go.Bar(
            x=list(RISK_TIERS),
            y=counts.to_numpy(),
            marker_color=[theme.RISK_COLORS[t] for t in RISK_TIERS],
            text=[f"{n:,}" for n in counts],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:,} transactions<extra></extra>",
        )
    )
    fig.update_yaxes(type="log")
    st.plotly_chart(
        _style(fig, "Transactions per risk tier", "Risk tier", "Transactions (log scale)"),
        config=PLOT,
    )
    st.caption(
        "**What this shows:** how many transactions fall in each tier. The scale is "
        "logarithmic because almost all transactions are Low risk."
    )
    tier_rows = []
    for tier in RISK_TIERS:
        in_tier = df[df["risk_tier"] == tier]
        row = {
            "Tier": tier,
            "Probability": ranges[tier],
            "Recommended action": TIER_INFO[tier].action,
            "Transactions": len(in_tier),
            "Total amount": in_tier["amount"].sum(),
        }
        if labelled:
            known = in_tier["is_fraud"].dropna()
            row["Known fraud rate"] = (known == 1).mean() if len(known) else np.nan
        tier_rows.append(row)
    st.dataframe(
        pd.DataFrame(tier_rows).style.format(
            {"Total amount": "{:,.2f}", "Known fraud rate": "{:.1%}"}, na_rep="—"
        ),
        hide_index=True,
    )
    st.caption(
        "Tier cut-offs are set in config.py (RiskThresholds). They are applied to calibrated "
        "probabilities, so a tier's known fraud rate should roughly match its probability range."
    )

# --- Patterns -----------------------------------------------------------------------
with patterns:
    risky = df["risk_tier"].isin(["High", "Critical"])
    if risky.sum() == 0:
        st.info("No High or Critical transactions in this dataset.", icon=":material/info:")
    else:
        st.markdown(
            f"{int(risky.sum()):,} transactions are High or Critical. The charts compare how "
            "they are spread with all transactions. **Lift** above 1 means a group is "
            "over-represented among risky transactions (e.g. 3.0 = three times as common)."
        )

        def lift_table(key: pd.Series, label: str) -> pd.DataFrame:
            share_all = key.value_counts(normalize=True)
            share_risky = key[risky].value_counts(normalize=True)
            out = pd.DataFrame({"all": share_all, "risky": share_risky}).fillna(0)
            out["lift"] = np.where(out["all"] > 0, out["risky"] / out["all"], np.nan)
            out = out[out["all"] >= 0.005].sort_values("lift", ascending=False)
            return out.rename_axis(label).reset_index()

        def lift_chart(table: pd.DataFrame, label: str, title: str) -> go.Figure:
            data_ = table.head(12).iloc[::-1]
            fig = go.Figure(
                go.Bar(
                    x=data_["lift"],
                    y=data_[label].astype(str),
                    orientation="h",
                    marker_color=[theme.FRAUD if v > 1 else theme.PRIMARY for v in data_["lift"]],
                    customdata=np.stack([data_["risky"] * 100, data_["all"] * 100], axis=-1),
                    hovertemplate="%{y}: lift %{x:.1f}<br>%{customdata[0]:.1f}% of risky vs "
                    "%{customdata[1]:.1f}% of all<extra></extra>",
                )
            )
            fig.add_vline(x=1, line_dash="dash", line_color=theme.TEXT_MUTED)
            return _style(fig, title, "Lift (1 = same as all transactions)", label, height=420)

        left, right = st.columns(2)
        with left:
            cat = lift_table(df["category"].astype(str), "Category")
            st.plotly_chart(
                lift_chart(cat, "Category", "Categories over-represented in high risk"), config=PLOT
            )
        with right:
            hours = lift_table(df["timestamp"].dt.hour.map(lambda h: f"{h:02d}:00"), "Hour")
            st.plotly_chart(
                lift_chart(hours, "Hour", "Hours over-represented in high risk"), config=PLOT
            )
        bands = lift_table(amount_band(df["amount"]).astype(str), "Amount range")
        st.plotly_chart(
            lift_chart(bands, "Amount range", "Amount ranges over-represented in high risk"),
            config=PLOT,
        )
        top_cat = cat.iloc[0]
        st.caption(
            f"**Reading this:** '{escape_markdown(top_cat['Category'])}' makes up "
            f"{top_cat['risky']:.1%} of "
            f"high-risk transactions but only {top_cat['all']:.1%} of all transactions. "
            "Groups below 0.5% of "
            "all transactions are hidden to avoid noise."
        )

# --- Threshold what-if ----------------------------------------------------------
with threshold_tab:
    st.markdown(
        "The model flags a transaction when its probability reaches the **decision threshold**. "
        f"The saved model uses **{meta['decision_threshold']:.2f}**, chosen on validation data. "
        "Move the slider to see the trade-off on this dataset. This does not change the model."
    )
    chosen = st.slider(
        "Decision threshold", 0.05, 0.95, float(round(meta["decision_threshold"], 2)), 0.01
    )
    p = df["fraud_probability"].to_numpy()
    flagged = p >= chosen
    cols = st.columns(4)
    with cols[0]:
        render_kpi(
            "Flagged for review",
            f"{int(flagged.sum()):,}",
            f"{flagged.mean():.2%} of transactions.",
        )
    if labelled:
        known = df["is_fraud"].notna().to_numpy()
        y = (df["is_fraud"] == 1).fillna(False).to_numpy()
        tp = int((flagged & y & known).sum())
        fp = int((flagged & ~y & known).sum())
        fn = int((~flagged & y & known).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        with cols[1]:
            render_kpi(
                "Fraud caught (recall)", f"{recall:.1%}", f"{tp:,} of {tp + fn:,} known fraud."
            )
        with cols[2]:
            render_kpi("Alerts that are fraud", f"{precision:.1%}", f"{fp:,} false alarms.")
        with cols[3]:
            render_kpi("Fraud missed", f"{fn:,}", "Known fraud below the threshold.")
        st.caption(
            "Lower threshold: more fraud caught, more false alarms for analysts. Higher threshold: "
            "fewer alarms, more fraud missed. If this dataset came from the training period, "
            "these numbers are optimistic; the honest test results are on Model Performance."
        )
    else:
        st.caption("This dataset has no fraud labels, so precision and recall cannot be shown.")

# --- High-risk list ------------------------------------------------------------------
with table_tab:
    f = db.TransactionFilter(tiers=("High", "Critical"))
    total = db.count_transactions(active.id, f)
    if total == 0:
        st.info("No High or Critical transactions.", icon=":material/info:")
    else:
        size = SETTINGS.table_page_size
        pages = max(1, math.ceil(total / size))
        page = st.number_input(f"Page (of {pages:,})", 1, pages, 1)
        rows = db.search_transactions(active.id, f, int(page), size)
        shown = rows[
            [
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
                    "is_fraud",
                )
                if c in rows
            ]
        ].copy()
        if "card_id" in shown:
            shown["card_id"] = shown["card_id"].astype(object).map(mask_card)
        shown["fraud_probability"] = shown["fraud_probability"] * 100
        st.dataframe(
            shown,
            hide_index=True,
            column_config=transaction_columns(),
        )
        st.caption(
            f"{total:,} High and Critical transactions, riskiest first. Open Transaction "
            "Investigation to filter further and see why each one was flagged."
        )
