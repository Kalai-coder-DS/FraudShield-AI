"""Module 1 - Executive Dashboard: one screen, plain language, real numbers."""

from __future__ import annotations

from html import escape

import streamlit as st

from fraudshield.eda import charts
from fraudshield.reports.summary import risk_level_text
from fraudshield.ui import model_state as ms
from fraudshield.ui.components import escape_markdown, render_header, render_kpi, render_page_link

PLOT = {"displaylogo": False}

render_header(
    "Executive Dashboard",
    "How many transactions were checked, how many look risky, and what that is worth.",
    eyebrow="Overview",
)

model, meta, _ = ms.get_model()
active = ms.get_active_dataset()
summary = ms.get_summary(active, model, meta)
ds, sc, lab = summary.dataset, summary.scoring, summary.labelled

st.caption(
    f"Dataset: {escape_markdown(ds['name'])} · {ds['start']} to {ds['end']} · "
    f"model {summary.model['name']} (trained {summary.model['trained']})"
)

st.markdown(
    f"<div class='fs-card' style='border-left:3px solid #E5484D;border-radius:0'>"
    f"<h3>{escape(risk_level_text(summary))}</h3>"
    "<p>Scores are model estimates that help decide what to review first, "
    "not confirmed fraud decisions.</p></div>",
    unsafe_allow_html=True,
)

# --- KPIs ------------------------------------------------------------------------------
row1 = [
    (
        "Transactions analysed",
        f"{ds['transactions']:,}",
        f"Made with {summary.totals['cards']:,} cards.",
    ),
    (
        "Total transaction value",
        f"{summary.totals['total_amount']:,.0f}",
        "Sum of all amounts in the dataset.",
    ),
    (
        "Flagged for review",
        f"{sc['flagged']:,}",
        f"{sc['flagged_share']:.2%} of transactions reach the model's threshold.",
    ),
    ("Value flagged", f"{sc['flagged_amount']:,.0f}", "Total amount of the flagged transactions."),
]
row2 = [
    (
        "High + Critical risk",
        f"{sc['high_or_critical']:,}",
        f"Worth {sc['high_or_critical_amount']:,.0f}. Review today or hold.",
    ),
]
if lab:
    row2 += [
        (
            "Known fraud in data",
            f"{lab['known_fraud']:,}",
            f"{lab['known_fraud_rate']:.2%} of labelled transactions.",
        ),
        (
            "Known fraud flagged",
            f"{lab['recall']:.1%}" if lab["recall"] is not None else "—",
            f"{lab['caught']:,} caught, {lab['missed']:,} missed, "
            f"{lab['false_alarms']:,} false alarms.",
        ),
    ]
row2.append(
    (
        "Model quality (PR-AUC)",
        f"{summary.model['pr_auc']:.3f}",
        f"On unseen test data; random guessing scores {summary.model['fraud_rate']:.3f}.",
    ),
)
for row in (row1, row2):
    for col, (label, value, caption) in zip(st.columns(4), row, strict=False):
        with col:
            render_kpi(label, value, caption)
    st.write("")

for warning in summary.warnings:
    st.warning(warning, icon=":material/warning:")

# --- Charts ------------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.plotly_chart(charts.tier_bar_chart(summary.tiers), config=PLOT)
    st.caption("Almost all transactions are Low risk; the log scale keeps the small tiers visible.")
with right:
    if summary.top_categories:
        st.plotly_chart(charts.category_lift_chart(summary.top_categories), config=PLOT)
        st.caption("Red bars: categories more common among High/Critical than overall.")
    else:
        st.info("No High or Critical transactions, so there are no risky categories.")

weekly = summary.weekly.drop(columns=["fraud"], errors="ignore")
st.plotly_chart(
    charts.weekly_activity_chart(weekly, "Weekly activity: all transactions vs. flagged"),
    config=PLOT,
)

# --- Findings ------------------------------------------------------------------------------
st.markdown("##### What the numbers say")
for finding in summary.findings:
    st.markdown(f"- {escape_markdown(finding)}")

st.markdown("##### Next steps")
links = st.columns(3)
with links[0]:
    render_page_link("views/risk.py", "Review risk tiers", ":material/shield:")
with links[1]:
    render_page_link("views/investigation.py", "Investigate transactions", ":material/search:")
with links[2]:
    render_page_link("views/reports.py", "Download a report", ":material/description:")
