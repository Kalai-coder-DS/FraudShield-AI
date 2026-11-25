"""UI flow tests: run the real Streamlit pages and click through them.

Every test uses an isolated workspace (temporary database, model folder and sample
file), so it never touches your real data. Needs Streamlit; skipped where it is
not installed.
"""

import pytest
from conftest import make_signal_rows

testing = pytest.importorskip("streamlit.testing.v1")

import streamlit as st  # noqa: E402

from fraudshield import config  # noqa: E402
from fraudshield.data.sample import load_sample  # noqa: E402
from fraudshield.models import registry  # noqa: E402
from fraudshield.storage import db  # noqa: E402

VIEWS = config.PROJECT_ROOT / "views"
ALL_PAGES = [
    "home",
    "dashboard",
    "dataset",
    "eda",
    "models",
    "prediction",
    "risk",
    "investigation",
    "reports",
]


@pytest.fixture
def app_env(tmp_path, monkeypatch, small_model):
    """Temporary database + trained model + sample file, with Streamlit caches cleared."""
    monkeypatch.setattr(config, "DATABASE_PATH", tmp_path / "app.db")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    sample = tmp_path / "sample" / "transactions_sample.csv"
    sample.parent.mkdir()
    make_signal_rows(800, "2020-08-01", 123).to_csv(sample, index=False)
    monkeypatch.setattr(config, "SAMPLE_DATASET_FILE", sample)
    registry.save(small_model[2])
    st.cache_data.clear()
    st.cache_resource.clear()
    yield tmp_path
    st.cache_data.clear()
    st.cache_resource.clear()


def open_page(name: str):
    at = testing.AppTest.from_file(str(VIEWS / f"{name}.py"), default_timeout=90)
    at.run()
    assert not at.exception, f"{name}: {at.exception}"
    return at


def text_of(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [s.value for s in at.success] + [w.value for w in at.warning]
    return " ".join(str(p) for p in parts)


def button(at, label: str):
    matches = [b for b in at.button if b.label == label]
    assert matches, f"No button labelled {label!r}; found {[b.label for b in at.button]}"
    return matches[0]


@pytest.mark.parametrize("page", ALL_PAGES)
def test_every_page_opens_without_a_dataset(app_env, page):
    open_page(page)


def test_one_click_sample_then_every_page_works(app_env):
    at = open_page("eda")
    button(at, "Load sample dataset").click().run()
    assert not at.exception
    active = db.get_active_dataset()
    assert active is not None and active.rows_clean == 800
    for page in ALL_PAGES:
        open_page(page)


def test_dashboard_shows_real_numbers(app_env):
    load_sample()
    text = text_of(open_page("dashboard"))
    assert "need an analyst's attention" in text  # plain apostrophe, not &#x27;
    assert "800" in text  # transactions analysed


def test_prediction_explains_the_riskiest_transaction(app_env):
    load_sample()
    at = open_page("prediction")
    button(at, "Riskiest transaction").click().run()
    assert not at.exception
    assert "Why this score?" in text_of(at)


def test_manual_entry_scores_a_transaction(app_env):
    load_sample()
    at = open_page("prediction")
    button(at, "Score transaction").click().run()
    assert not at.exception
    assert "Fraud probability" in text_of(at)


def test_reports_build_a_pdf(app_env):
    load_sample()
    at = open_page("reports")
    button(at, "Create PDF report").click().run()
    assert not at.exception
    assert "Report ready." in text_of(at)


def test_dataset_page_checks_and_saves_the_sample(app_env):
    at = open_page("dataset")
    button(at, "Check data quality").click().run()
    assert not at.exception
    assert "Quality score" in text_of(at)
    button(at, "Save and use this dataset").click().run()
    assert not at.exception
    assert db.get_active_dataset() is not None


def test_investigation_filters_to_critical(app_env):
    load_sample()
    at = open_page("investigation")
    at.multiselect[0].select("Critical")
    button(at, "Apply filters").click().run()
    assert not at.exception
    assert "transactions match" in text_of(at)
