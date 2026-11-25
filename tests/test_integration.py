"""End-to-end: the whole path from a CSV file to the reports, on an isolated workspace.

Runs every layer together (no UI): sample file -> validation -> SQLite -> training ->
saved model (with fingerprint check) -> scoring -> stored scores -> search ->
explanation -> summary -> CSV and PDF.
"""

import io

import pandas as pd
import pytest
from conftest import SPARKOV_COLUMNS, make_signal_rows

from fraudshield import config
from fraudshield.data.sample import load_sample, sample_available
from fraudshield.data.schema import suggest_mapping
from fraudshield.data.validation import process_dataset
from fraudshield.models import explain, registry, scoring
from fraudshield.models.training import train_and_evaluate
from fraudshield.reports.csv_report import scored_transactions_csv
from fraudshield.reports.pdf_report import build_pdf
from fraudshield.reports.summary import build_summary
from fraudshield.storage import db


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the app's database, models folder and sample file at a temp folder."""
    monkeypatch.setattr(config, "DATABASE_PATH", tmp_path / "processed" / "app.db")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "SAMPLE_DATASET_FILE", tmp_path / "sample" / "sample.csv")
    return tmp_path


def test_full_pipeline(workspace):
    # 1. Train on an earlier period, keep the model in the (temporary) models folder
    frames = []
    for i, (n, start) in enumerate(((5000, "2020-01-01"), (2000, "2020-06-01"))):
        path = workspace / f"{i}.csv"
        make_signal_rows(n, start, 90 + i).to_csv(path, index=False)
        frames.append(process_dataset(path, suggest_mapping(SPARKOV_COLUMNS)).data)
    result = train_and_evaluate(*frames, n_jobs=1, rf_trees=20, log=lambda *_: None)
    registry.save(result)  # default location = patched config.MODELS_DIR
    assert (workspace / "models" / registry.MODEL_FILE).is_file()
    meta = registry.load_metadata()
    model = registry.load_model()

    # 2. The bundled sample comes from a later period, loaded with one call
    config.SAMPLE_DATASET_FILE.parent.mkdir(parents=True)
    make_signal_rows(1500, "2020-09-01", 99).to_csv(config.SAMPLE_DATASET_FILE, index=False)
    assert sample_available()
    dataset_id = load_sample()
    active = db.get_active_dataset()
    assert active.id == dataset_id and active.rows_clean == 1500
    assert (workspace / "processed" / "app.db").is_file()  # nothing written to the real DB

    # 3. Score, store and search
    data = db.load_transactions(dataset_id)
    scores = scoring.score_frame(model, data, meta["decision_threshold"])
    db.save_scores(dataset_id, scores, meta["model_sha256"])
    assert db.scores_model(dataset_id) == meta["model_sha256"]
    riskiest = db.search_transactions(dataset_id, db.TransactionFilter(), 1, 5)
    assert riskiest["fraud_probability"].iloc[0] == pytest.approx(scores["fraud_probability"].max())

    # 4. Explain the riskiest transaction
    parts = explain.model_parts(model)
    row = data.loc[[riskiest.index[0]]]
    p, table = explain.explain(parts, row, explain.background_matrix(parts, data))
    assert p == pytest.approx(scores["fraud_probability"].max())
    assert table["contribution"].abs().max() > 0

    # 5. Summary, CSV and PDF agree with each other
    summary = build_summary(data, db.load_scores(dataset_id), meta, active.name, 100.0)
    csv = pd.read_csv(
        io.BytesIO(scored_transactions_csv(data, scores, flagged_only=True)), encoding="utf-8-sig"
    )
    assert len(csv) == summary.scoring["flagged"]
    assert build_pdf(summary).startswith(b"%PDF")

    # 6. Deleting the dataset removes everything belonging to it
    db.delete_dataset(dataset_id)
    assert db.get_active_dataset() is None and db.scores_model(dataset_id) is None


def test_tampered_model_is_refused_end_to_end(workspace, small_model):
    _, _, result = small_model
    registry.save(result)
    path = workspace / "models" / registry.MODEL_FILE
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(registry.ModelIntegrityError):
        registry.load_model()
