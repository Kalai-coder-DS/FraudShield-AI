"""Honest "coming in Phase N" page for modules that are not built yet.

It shows what the module will do, without any fake numbers, charts or buttons.
"""

from __future__ import annotations

from html import escape

from fraudshield.ui.components import render_empty_state, render_header
from fraudshield.ui.modules import get_module


def render_module_placeholder(key: str) -> None:
    import streamlit as st

    module = get_module(key)
    render_header(module.title, module.summary, eyebrow=f"Module · Phase {module.phase}")
    render_empty_state(
        "Not built yet",
        f"This module is developed in Phase {module.phase}. "
        "No results are shown until it is implemented and tested.",
    )
    items = "".join(f"<li>{escape(f)}</li>" for f in module.features)
    st.markdown(
        f'<div class="fs-card" style="margin-top:1rem"><h3>What this module will do</h3>'
        f"<ul>{items}</ul></div>",
        unsafe_allow_html=True,
    )
