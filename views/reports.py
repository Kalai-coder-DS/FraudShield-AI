"""Module 8 - Reports: download scored transactions (CSV) and a summary report (PDF)."""

from __future__ import annotations

import re
from datetime import date

import streamlit as st

from fraudshield.reports.csv_report import scored_transactions_csv
from fraudshield.ui import model_state as ms
from fraudshield.ui.components import escape_markdown, render_header, render_kpi

render_header(
    "Reports",
    "Download the results: every scored transaction as a CSV file, or a summary "
    "report as a PDF to share.",
    eyebrow="Module 8 · Investigation",
)

model, meta, _ = ms.get_model()
active = ms.get_active_dataset()
summary = ms.get_summary(active, model, meta)
sha = meta["model_sha256"]
slug = re.sub(r"[^A-Za-z0-9]+", "_", active.name).strip("_").lower() or "dataset"
stamp = date.today().isoformat()

st.caption(
    f"Dataset: {escape_markdown(active.name)} · {summary.dataset['transactions']:,} "
    f"transactions · model {summary.model['name']}"
)

# --- CSV ----------------------------------------------------------------------------------
st.markdown("##### Scored transactions (CSV)")
scopes = {
    "All transactions": ({}, "all"),
    "Flagged for review only": ({"flagged_only": True}, "flagged"),
    "High and Critical risk only": ({"tiers": ("High", "Critical")}, "high_critical"),
}
scope = st.radio("Which transactions?", list(scopes), horizontal=True)
options, suffix = scopes[scope]


@st.cache_data(show_spinner="Preparing CSV...", max_entries=3)
def make_csv(dataset_id: int, sha256: str, suffix: str, flagged_only=False, tiers=None) -> bytes:
    data = ms.load_dataset(dataset_id)
    scores = ms._load_scores(dataset_id, sha256)
    return scored_transactions_csv(data, scores, tiers=tiers, flagged_only=flagged_only)


csv_bytes = make_csv(active.id, sha, suffix, **options)
rows = max(csv_bytes.count(b"\n") - 1, 0)
st.download_button(
    f"Download CSV ({rows:,} transactions, {len(csv_bytes) / 1e6:,.1f} MB)",
    data=csv_bytes,
    file_name=f"fraudshield_{slug}_{suffix}_{stamp}.csv",
    mime="text/csv",
    icon=":material/download:",
)
st.caption(
    "Columns: transaction ID, time, amount, category, merchant, card (masked to the last 4 "
    "digits), fraud probability, risk tier, flagged yes/no and the known label if available. "
    "Riskiest first."
)

st.divider()

# --- PDF -----------------------------------------------------------------------------------
st.markdown("##### Summary report (PDF)")
cols = st.columns(4)
for col, (label, value) in zip(
    cols,
    [
        ("Transactions", f"{summary.dataset['transactions']:,}"),
        ("Flagged", f"{summary.scoring['flagged']:,}"),
        ("High + Critical", f"{summary.scoring['high_or_critical']:,}"),
        ("Model PR-AUC", f"{summary.model['pr_auc']:.3f}"),
    ],
    strict=True,
):
    with col:
        render_kpi(label, value)
st.write("")
st.markdown(
    "The PDF contains: key figures, plain-language findings, the risk tier table and chart, "
    "weekly activity, the categories most common among high-risk transactions, and model "
    "performance on unseen test data. Every number comes from the current results."
)

key = f"pdf-{active.id}-{sha}"
if st.button("Create PDF report", type="primary", icon=":material/picture_as_pdf:"):
    from fraudshield.reports.pdf_report import build_pdf

    with st.spinner("Building the PDF..."):
        st.session_state[key] = build_pdf(summary)

if key in st.session_state:
    pdf = st.session_state[key]
    st.download_button(
        f"Download PDF ({len(pdf) / 1e3:,.0f} KB)",
        data=pdf,
        file_name=f"fraudshield_report_{slug}_{stamp}.pdf",
        mime="application/pdf",
        icon=":material/download:",
    )
    st.success("Report ready.", icon=":material/check_circle:")
