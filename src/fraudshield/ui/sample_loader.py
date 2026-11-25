"""One-click loading of the bundled sample dataset (for empty pages)."""

from __future__ import annotations

import streamlit as st

from fraudshield.config import SAMPLE_DATASET_FILE
from fraudshield.data.loader import read_columns
from fraudshield.data.schema import suggest_mapping
from fraudshield.data.validation import process_dataset
from fraudshield.storage import db


def sample_available() -> bool:
    return SAMPLE_DATASET_FILE.is_file()


def load_sample() -> int:
    """Validate the bundled sample with automatic column matching and make it active."""
    result = process_dataset(
        SAMPLE_DATASET_FILE, suggest_mapping(read_columns(SAMPLE_DATASET_FILE))
    )
    return db.save_dataset("Sparkov sample", "Bundled sample (Sparkov)", result.data, result.report)


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
