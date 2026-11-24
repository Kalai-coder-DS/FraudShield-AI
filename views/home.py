"""Home page: explains the product in plain language and shows build status."""

from __future__ import annotations

from html import escape

import streamlit as st

from fraudshield.ui.components import build_badge_html, render_header
from fraudshield.ui.modules import MODULES

render_header(
    "FraudShield AI",
    "Find suspicious financial transactions with machine learning, understand why "
    "they look risky, and decide which ones to investigate first.",
    eyebrow="Fraud detection & risk intelligence",
)

STEPS = (
    ("1 · Load data", "Upload a CSV or use the sample dataset; its quality is checked."),
    ("2 · Explore", "Charts show how fraudulent transactions differ from normal ones."),
    ("3 · Score", "Trained models give every transaction a fraud probability and a risk tier."),
    ("4 · Investigate", "Review the riskiest transactions, see why, and export a report."),
)
for column, (title, text) in zip(st.columns(len(STEPS)), STEPS, strict=True):
    with column:
        st.markdown(
            f'<div class="fs-card"><h3>{escape(title)}</h3><p>{escape(text)}</p></div>',
            unsafe_allow_html=True,
        )

st.info(
    "A fraud probability is a model's estimate based on past patterns. It is a signal to "
    "prioritise review, not a confirmed fraud decision.",
    icon=":material/info:",
)

st.subheader("Modules")
rows = []
for module in MODULES:
    status = build_badge_html("Ready" if module.ready else f"Phase {module.phase}", module.ready)
    rows.append(
        f"<tr><td style='padding:.5rem .75rem'><b>{escape(module.title)}</b></td>"
        f"<td style='padding:.5rem .75rem;color:#A99BB3'>{escape(module.summary)}</td>"
        f"<td style='padding:.5rem .75rem'>{status}</td></tr>"
    )
st.markdown(
    "<div class='fs-card'><table style='width:100%;border-collapse:collapse'>"
    + "".join(rows)
    + "</table></div>",
    unsafe_allow_html=True,
)
