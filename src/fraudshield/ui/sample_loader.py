"""One-click loading of the bundled sample dataset (for empty pages)."""

from __future__ import annotations

import streamlit as st

from fraudshield.data.sample import load_sample, sample_available


def render_load_sample_button(key: str) -> None:
    """Show a 'Load sample dataset' button; loads it and refreshes the page when clicked."""
    if not sample_available():
        return
    if st.button(
        "Load sample dataset", type="primary", key=f"load-sample-{key}", icon=":material/download:"
    ):
        with st.spinner("Checking and saving the sample dataset..."):
            load_sample()
        st.session_state["dataset_flash"] = "Loaded the sample dataset and made it active."
        st.rerun()
    st.caption("Loads the bundled 50,000-row sample of real Sparkov test data.")
