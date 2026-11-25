"""Phase 5: risk tiers, scoring, explanations and score storage/search."""

import numpy as np
import pandas as pd
import pytest

from fraudshield.config import RiskThresholds
from fraudshield.models import explain, scoring
from fraudshield.risk.tiers import TIER_INFO, assign_tier, assign_tiers
from fraudshield.storage import db

# --- Risk tiers -------------------------------------------------------------------


@pytest.mark.parametrize(
    "p, tier",
    [
        (0.0, "Low"),
        (0.2999, "Low"),
        (0.30, "Medium"),
        (0.5999, "Medium"),
        (0.60, "High"),
        (0.8499, "High"),
        (0.85, "Critical"),
        (1.0, "Critical"),
    ],
)
def test_tier_boundaries(p, tier):
    assert assign_tier(p) == tier


def test_custom_thresholds():
    t = RiskThresholds(medium=0.1, high=0.2, critical=0.3)
    assert list(assign_tiers([0.05, 0.15, 0.25, 0.35], t)) == ["Low", "Medium", "High", "Critical"]


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan])
def test_invalid_probability_is_rejected(bad):
    with pytest.raises(ValueError):
        assign_tiers([bad])


def test_every_tier_has_an_action():
    assert set(TIER_INFO) == {"Low", "Medium", "High", "Critical"}


# --- Scoring -----------------------------------------------------------------------


def test_score_frame_is_chunk_independent(small_model):
    model, test, _ = small_model
    a = scoring.score_frame(model, test, 0.5, chunk=10_000)
    b = scoring.score_frame(model, test, 0.5, chunk=333)
    pd.testing.assert_frame_equal(a, b)
    assert a.index.equals(test.index)
    assert ((a["fraud_probability"] >= 0.5) == (a["predicted_fraud"] == 1)).all()


def test_manual_transaction_has_requested_distance_and_scores(small_model):
    model, _, _ = small_model
    row = scoring.manual_transaction(
        500, pd.Timestamp("2020-07-01 23:30"), "shopping_net", "", 40, 30.0, 5000
    )
    from fraudshield.features.basic import add_basic_features

    assert add_basic_features(row)["distance_km"].item() == pytest.approx(30.0, abs=0.01)
    assert add_basic_features(row)["age_years"].item() == pytest.approx(40, abs=0.01)
    p = model.predict_proba(row)[:, 1]
    assert 0 <= p[0] <= 1


def test_manual_transaction_without_optional_details(small_model):
    model, _, _ = small_model
    row = scoring.manual_transaction(20, pd.Timestamp("2020-07-01 12:00"), "home")
    assert 0 <= model.predict_proba(row)[:, 1][0] <= 1


# --- Explanations ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["Logistic Regression", "Decision Tree", "Random Forest"])
def test_model_parts_reproduce_every_model(small_model, name):
    # Regression test: Logistic Regression is calibrated on decision_function, trees on
    # predict_proba. Explanations must use the same score as the saved model.
    _, test, result = small_model
    model = result.results[name].model
    parts = explain.model_parts(model)
    np.testing.assert_allclose(
        parts.predict(parts.transform(test)), model.predict_proba(test)[:, 1]
    )


def test_explanation_finds_planted_signals(small_model):
    model, test, _ = small_model
    parts = explain.model_parts(model)
    proba = model.predict_proba(test)[:, 1]
    row = test.iloc[[int(np.argmax(proba))]]
    p, table = explain.explain(parts, row, explain.background_matrix(parts, test))
    assert p == pytest.approx(proba.max())
    top3 = set(table["feature"].head(3))
    assert top3 & {"Amount", "Amount vs. card's usual", "Time of day", "Spending category"}
    # City population carries no signal in the test data: its effect must be ~0
    city = table.set_index("feature").loc["City population", "contribution"]
    assert abs(city) < 0.05


def test_reasons_only_mention_features_that_raise_the_score():
    table = pd.DataFrame({"feature": ["Amount", "Time of day"], "contribution": [0.4, -0.2]})
    text = explain.reasons(table, {"Amount": "900.00"})
    assert len(text) == 1 and "Amount (900.00)" in text[0] and "40 percentage points" in text[0]


def test_explain_needs_exactly_one_row(small_model):
    model, test, _ = small_model
    parts = explain.model_parts(model)
    with pytest.raises(ValueError):
        explain.explain(parts, test.head(2), explain.background_matrix(parts, test))


# --- Score storage and search -----------------------------------------------------------


@pytest.fixture
def scored_db(small_model, tmp_path):
    from fraudshield.data.validation import DataReport

    model, test, _ = small_model
    data = test.reset_index(drop=True)
    report = DataReport(
        len(data),
        len(data),
        1.0,
        len(data),
        100.0,
        {c: c for c in data.columns},
        [],
        True,
        0,
        0,
        None,
        None,
        [],
        [],
    )
    path = tmp_path / "s.db"
    ds = db.save_dataset("s", "test", data, report, db_path=path)
    scores = scoring.score_frame(model, db.load_transactions(ds, db_path=path), 0.5)
    db.save_scores(ds, scores, "sha-1", path)
    return ds, path, db.load_transactions(ds, db_path=path), scores


def test_scores_round_trip(scored_db):
    ds, path, _, scores = scored_db
    assert db.scores_model(ds, path) == "sha-1"
    loaded = db.load_scores(ds, path)
    np.testing.assert_allclose(loaded["fraud_probability"], scores["fraud_probability"])
    assert (loaded["risk_tier"].to_numpy() == scores["risk_tier"].to_numpy()).all()


def test_filters_match_pandas(scored_db):
    ds, path, data, scores = scored_db
    df = data.join(scores)
    F = db.TransactionFilter
    assert db.count_transactions(ds, F(), path) == len(df)
    assert db.count_transactions(ds, F(tiers=("High", "Critical")), path) == int(
        df["risk_tier"].isin(["High", "Critical"]).sum()
    )
    assert db.count_transactions(ds, F(categories=("misc_net",)), path) == int(
        (df["category"] == "misc_net").sum()
    )
    assert db.count_transactions(ds, F(min_amount=50, max_amount=100), path) == int(
        df["amount"].between(50, 100).sum()
    )
    assert db.count_transactions(ds, F(label=1), path) == int((df["is_fraud"] == 1).sum())
    assert db.count_transactions(ds, F(flagged_only=True), path) == int(df["predicted_fraud"].sum())
    day = df["timestamp"].min().date()
    assert db.count_transactions(ds, F(date_from=str(day), date_to=str(day)), path) == int(
        (df["timestamp"].dt.date == day).sum()
    )


def test_search_is_sorted_and_paginated(scored_db):
    ds, path, _, _ = scored_db
    page1 = db.search_transactions(ds, db.TransactionFilter(), 1, 20, path)
    page2 = db.search_transactions(ds, db.TransactionFilter(), 2, 20, path)
    probs = list(page1["fraud_probability"]) + list(page2["fraud_probability"])
    assert probs == sorted(probs, reverse=True)
    assert not set(page1.index) & set(page2.index)


def test_search_text_is_safe(scored_db):
    ds, path, data, _ = scored_db
    F = db.TransactionFilter
    assert db.count_transactions(ds, F(text="'; DROP TABLE transactions; --"), path) == 0
    assert db.count_transactions(ds, F(text="%"), path) == 0  # wildcard is literal
    assert db.count_transactions(ds, F(), path) == len(data)  # table still intact
    one = data["transaction_id"].iloc[3]
    assert db.count_transactions(ds, F(text=one), path) >= 1
    assert len(db.get_transaction(ds, one, path)) == 1


def test_deleting_dataset_removes_scores(scored_db):
    ds, path, _, _ = scored_db
    db.delete_dataset(ds, path)
    assert db.scores_model(ds, path) is None
