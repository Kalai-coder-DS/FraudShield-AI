"""Phase 4: features, training workflow, evaluation and model registry."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import SPARKOV_COLUMNS, make_signal_rows

from fraudshield.config import PROJECT_ROOT, RiskThresholds
from fraudshield.data.schema import suggest_mapping
from fraudshield.data.validation import process_dataset
from fraudshield.features.engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    CardProfile,
    add_model_features,
)
from fraudshield.models import evaluation as ev
from fraudshield.models import registry
from fraudshield.models.training import (
    ANOMALY_MODEL,
    SUPERVISED_MODELS,
    build_pipeline,
    time_split,
    train_and_evaluate,
)

MAPPING = suggest_mapping(SPARKOV_COLUMNS)


def processed(rows: pd.DataFrame, tmp_path, name: str) -> pd.DataFrame:
    path = tmp_path / f"{name}.csv"
    rows.to_csv(path, index=False)
    return process_dataset(path, MAPPING).data


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("signal")
    train = processed(make_signal_rows(6000, "2020-01-01", 1), tmp, "train")
    test = processed(make_signal_rows(3000, "2020-06-01", 2), tmp, "test")
    return train, test


@pytest.fixture(scope="module")
def trained(datasets):
    train, test = datasets
    return train_and_evaluate(train, test, n_jobs=1, rf_trees=30, log=lambda *_: None)


# --- Features -----------------------------------------------------------------


def test_model_features_exist_even_when_inputs_are_missing():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 23:00", "2024-01-01 12:00"]),
            "amount": [10.0, 20.0],
        }
    )
    out = add_model_features(df)
    assert out["is_night"].tolist() == [1, 0]
    assert out["age_years"].isna().all() and out["distance_km"].isna().all()
    assert out["log_city_pop"].isna().all()


def test_card_profile_learns_only_from_training_data():
    train = pd.DataFrame({"card_id": ["a"] * 5, "amount": [np.e - 1] * 5})  # log1p = 1.0
    test = pd.DataFrame({"card_id": ["a", "new"], "amount": [1000.0, 1000.0]})
    profile = CardProfile().fit(train)
    before = dict(profile.card_mean_)
    out = profile.transform(test)
    assert profile.card_mean_ == before  # transforming test data changed nothing learned
    assert profile.card_mean_["a"] == pytest.approx(1.0)
    assert out["card_known"].tolist() == [1, 0]
    assert out["card_amount_z"].iloc[0] > 10  # 1000 is far above card a's usual ~1.7


def test_card_with_too_little_history_uses_overall_profile():
    train = pd.DataFrame({"card_id": ["a", "a", "b", "b", "b", "b", "b"], "amount": [5.0] * 7})
    profile = CardProfile(min_history=5).fit(train)
    assert "a" not in profile.card_mean_ and "b" in profile.card_mean_


def test_time_split_never_puts_future_rows_in_training(datasets):
    fit, val = time_split(datasets[0])
    assert fit["timestamp"].max() <= val["timestamp"].min()
    assert len(val) == pytest.approx(0.2 * len(datasets[0]), abs=1)


@pytest.mark.parametrize("name", [*SUPERVISED_MODELS, ANOMALY_MODEL])
def test_each_pipeline_fits_and_scores(name, datasets):
    train, test = datasets
    pipe = build_pipeline(name, n_jobs=1, rf_trees=10)
    if name == ANOMALY_MODEL:
        pipe.fit(train)
        assert len(pipe.score_samples(test)) == len(test)
    else:
        pipe.fit(train, train["is_fraud"].astype(int))
        proba = pipe.predict_proba(test)[:, 1]
        assert len(proba) == len(test) and ((proba >= 0) & (proba <= 1)).all()


def test_preprocessor_uses_declared_features_only(datasets):
    pipe = build_pipeline("Logistic Regression")
    pipe.fit(datasets[0], datasets[0]["is_fraud"].astype(int))
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    assert set(NUMERIC_FEATURES) <= set(names)
    assert all(n in NUMERIC_FEATURES or n.startswith("category_") for n in names)
    assert not any(n.startswith(("gender", "job", "merchant", "card_id")) for n in names)
    assert CATEGORICAL_FEATURES == ("category",)


# --- Evaluation -----------------------------------------------------------------


def test_threshold_metrics_by_hand():
    y = [1, 1, 0, 0, 0]
    p = [0.9, 0.4, 0.6, 0.1, 0.2]
    m = ev.threshold_metrics(y, p, 0.5)
    assert (m["true_positives"], m["false_positives"], m["false_negatives"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["accuracy"] == 0.6


def test_perfect_ranking_scores_one():
    m = ev.ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0


def test_best_f1_threshold_separates_classes():
    assert ev.best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.8)


def test_recall_at_alert_rate():
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    score = [0.9, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0]
    assert ev.recall_at_alert_rate(y, score, 0.1) == 0.5  # top 1 of 10 catches 1 of 2 fraud


def test_tier_table_covers_every_transaction():
    rng = np.random.default_rng(0)
    proba = rng.random(1000)
    y = (rng.random(1000) < proba).astype(int)
    rows = ev.tier_table(y, proba, RiskThresholds())
    assert [r["tier"] for r in rows] == ["Low", "Medium", "High", "Critical"]
    assert sum(r["transactions"] for r in rows) == 1000
    assert rows[0]["recall_if_flagged_from_here"] == 1.0


# --- Full workflow --------------------------------------------------------------


def test_workflow_finds_the_planted_pattern(trained, datasets):
    best = trained.results[trained.selected]
    fraud_rate = best.test["fraud_rate"]
    assert trained.selected in SUPERVISED_MODELS
    assert best.test["pr_auc"] > 3 * fraud_rate  # far better than random guessing
    assert best.test["roc_auc"] > 0.8
    top = [r["feature"] for r in trained.importance[:3]]
    assert "card_amount_z" in top or "is_night" in top


def test_selection_uses_validation_not_test(trained):
    supervised = [r for r in trained.results.values() if r.kind == "supervised"]
    best_val = max(supervised, key=lambda r: r.validation["pr_auc_uncalibrated"])
    assert trained.selected == best_val.name


def test_anomaly_model_has_no_threshold_metrics(trained):
    iso = trained.results[ANOMALY_MODEL]
    assert iso.kind == "anomaly" and "precision" not in iso.test
    assert 0 <= iso.test["recall_at_best_alert_rate"] <= 1


def test_periods_do_not_overlap(trained):
    d = trained.data_info
    assert d["train"]["end"] <= d["validation"]["start"]
    assert d["validation"]["end"] <= d["test"]["start"]


# --- Registry -------------------------------------------------------------------


def test_save_and_load_model(trained, datasets, tmp_path):
    meta = registry.save(trained, tmp_path)
    assert meta["selected_model"] == trained.selected
    json.loads((tmp_path / registry.METADATA_FILE).read_text())  # valid JSON
    model = registry.load_model(tmp_path)
    proba = model.predict_proba(datasets[1].head(50))[:, 1]
    expected = trained.results[trained.selected].model.predict_proba(datasets[1].head(50))[:, 1]
    np.testing.assert_allclose(proba, expected)


def test_changed_model_file_is_refused(trained, tmp_path):
    registry.save(trained, tmp_path)
    path = tmp_path / registry.MODEL_FILE
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(registry.ModelIntegrityError):
        registry.load_model(tmp_path)


def test_missing_model_gives_clear_message(tmp_path):
    with pytest.raises(registry.ModelNotFoundError, match="train_models.py"):
        registry.load_metadata(tmp_path)


def test_training_script_end_to_end(tmp_path):
    make_signal_rows(4000, "2020-01-01", 3).to_csv(tmp_path / "train.csv", index=False)
    make_signal_rows(2000, "2020-06-01", 4).to_csv(tmp_path / "test.csv", index=False)
    out = tmp_path / "models"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_models.py"),
            "--train",
            str(tmp_path / "train.csv"),
            "--test",
            str(tmp_path / "test.csv"),
            "--output",
            str(out),
            "--rf-trees",
            "10",
            "--jobs",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Selected:" in result.stdout
    assert (out / registry.MODEL_FILE).is_file() and (out / registry.METADATA_FILE).is_file()
