"""Shared Streamlit helpers: load the model once, score the active dataset once.

* The model is cached per file fingerprint, so it is loaded from disk only once
  and reloaded automatically after retraining.
* Scores are stored in SQLite together with the model fingerprint. A dataset is
  re-scored only when it has no scores yet or the model has changed.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fraudshield.features.engineering import CardProfile
from fraudshield.models import explain, registry, scoring
from fraudshield.storage import db
from fraudshield.ui.components import render_empty_state, render_page_link
from fraudshield.ui.sample_loader import render_load_sample_button


@st.cache_resource(show_spinner="Loading the model...")
def _load_model(sha256: str):
    return registry.load_model()


def get_model():
    """Return (model, metadata, parts) or show an explanation and stop the page."""
    if not registry.metadata_exists():
        render_empty_state(
            "No trained model yet",
            "Run 'python scripts/train_models.py' in the terminal first (see docs/models.md).",
        )
        st.stop()
    meta = registry.load_metadata()
    try:
        model = _load_model(meta["model_sha256"])
    except registry.ModelNotFoundError as err:
        render_empty_state("Model file missing", str(err))
        st.stop()
    except registry.ModelIntegrityError as err:
        st.error(str(err), icon=":material/gpp_bad:")
        st.stop()
    return model, meta, explain.model_parts(model)


def get_active_dataset() -> db.DatasetInfo:
    active = db.get_active_dataset()
    if active is None:
        render_empty_state(
            "No active dataset",
            "Load the sample dataset with one click, or add your own data in Dataset Management.",
        )
        render_load_sample_button("model-state")
        render_page_link("views/dataset.py", "Go to Dataset Management", ":material/table_view:")
        st.stop()
    return active


@st.cache_data(show_spinner="Loading transactions...", max_entries=2)
def load_dataset(dataset_id: int) -> pd.DataFrame:
    return db.load_transactions(dataset_id)


def ensure_scores(dataset: db.DatasetInfo, model, meta: dict) -> pd.DataFrame:
    """Scores for the dataset, computing and storing them if needed."""
    sha = meta["model_sha256"]
    if db.scores_model(dataset.id) != sha:
        data = load_dataset(dataset.id)
        bar = st.progress(0.0, text=f"Scoring {len(data):,} transactions with the model...")
        scores = scoring.score_frame(
            model,
            data,
            meta["decision_threshold"],
            progress=lambda share: bar.progress(share, text=f"Scoring... {share:.0%}"),
        )
        db.save_scores(dataset.id, scores, sha)
        bar.empty()
        _load_scores.clear()
    return _load_scores(dataset.id, sha)


@st.cache_data(max_entries=2)
def _load_scores(dataset_id: int, sha256: str) -> pd.DataFrame:
    return db.load_scores(dataset_id)


@st.cache_resource(max_entries=2)
def background(dataset_id: int, sha256: str, _parts: explain.ModelParts):
    """Typical transactions used as the reference for explanations."""
    return explain.background_matrix(_parts, load_dataset(dataset_id))


def card_profile(parts: explain.ModelParts) -> CardProfile:
    return parts.features.named_steps["card_profile"]


def model_categories(parts: explain.ModelParts) -> list[str]:
    encoder = parts.features.named_steps["preprocess"].named_transformers_["cat"]
    return [str(c) for c in encoder.named_steps["encode"].categories_[0]]


@st.cache_data(show_spinner="Summarising results...", max_entries=2)
def _summary(dataset_id: int, dataset_name: str, sha256: str, _meta: dict):
    from fraudshield.reports.summary import build_summary

    data = load_dataset(dataset_id)
    scores = _load_scores(dataset_id, sha256)
    quality = db.load_report(dataset_id).quality_score
    return build_summary(data, scores, _meta, dataset_name, quality)


def get_summary(dataset: db.DatasetInfo, model, meta: dict):
    """Summary of the active dataset (scores it first if needed). Cached per dataset+model."""
    ensure_scores(dataset, model, meta)
    return _summary(dataset.id, dataset.name, meta["model_sha256"], meta)
