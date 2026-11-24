"""EDA calculations, checked against values worked out by hand."""

import numpy as np
import pandas as pd
import pytest
from conftest import SPARKOV_COLUMNS

from fraudshield.data.schema import suggest_mapping
from fraudshield.data.validation import process_dataset
from fraudshield.eda import analysis as a
from fraudshield.features.basic import add_basic_features, haversine_km


def frame(amounts, labels, categories=None, hours=None):
    """Small hand-made test table."""
    n = len(amounts)
    hours = hours if hours is not None else [10] * n
    df = pd.DataFrame(
        {
            "transaction_id": [f"t{i}" for i in range(n)],
            "timestamp": [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h) for h in hours],
            "amount": np.asarray(amounts, dtype="float64"),
            "card_id": pd.Categorical(["c1"] * n),
            "category": pd.Categorical(categories or ["x"] * n),
        }
    )
    if labels is not None:
        df["is_fraud"] = pd.array(labels, dtype="Int8")
    return add_basic_features(df)


# --- Row features -----------------------------------------------------------


def test_haversine_known_distance():
    # London to Paris is about 344 km
    assert haversine_km(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(343.6, abs=0.5)
    assert haversine_km(10, 20, 10, 20) == pytest.approx(0.0)


def test_basic_features():
    df = frame([9.0], [0], hours=[13])
    assert df["hour"].item() == 13
    assert df["weekday"].item() == 0  # 1 Jan 2024 was a Monday
    assert df["log_amount"].item() == pytest.approx(np.log(10))
    assert "age_years" not in df and "distance_km" not in df


def test_features_do_not_modify_input():
    raw = frame([1.0], [0]).drop(columns=["hour"])
    add_basic_features(raw)
    assert "hour" not in raw


# --- Statistics -------------------------------------------------------------


def test_wilson_interval_known_values():
    low, high = a.wilson_interval([0, 5], [10, 10])
    assert low == pytest.approx([0.0, 0.2366], abs=1e-4)
    assert high == pytest.approx([0.2775, 0.7634], abs=1e-4)


def test_wilson_interval_empty_group_is_nan():
    low, high = a.wilson_interval([0], [0])
    assert np.isnan(low[0]) and np.isnan(high[0])


def test_overview_and_balance():
    df = frame([10, 20, 30, 40], [0, 0, 1, None])
    ov = a.overview(df)
    assert ov["transactions"] == 4 and ov["labelled"] == 3
    assert ov["fraud"] == 1 and ov["fraud_rate"] == pytest.approx(1 / 3)
    assert ov["total_amount"] == 100 and ov["fraud_amount"] == 30
    bal = a.class_balance(df).set_index("class")
    assert bal.loc["Fraud", "transactions"] == 1 and bal.loc["Legitimate", "amount"] == 30


def test_fraud_rate_by_group():
    df = frame([1] * 6, [1, 0, 0, 1, 1, 0], categories=list("aaabbb"))
    table = a.fraud_rate_by_category(df).set_index("group")
    assert table.loc["a", "fraud_rate"] == pytest.approx(1 / 3)
    assert table.loc["b", "fraud_rate"] == pytest.approx(2 / 3)
    assert list(table.index) == ["b", "a"]  # sorted by rate
    assert not table["reliable"].any()  # only 3 transactions each


def test_rates_ignore_unlabelled_rows():
    df = frame([1, 1, 1], [1, None, 0])
    table = a.fraud_rate_by_category(df)
    assert table["transactions"].item() == 2


def test_hour_table_has_all_24_hours():
    table = a.fraud_rate_by_hour(frame([1, 1], [0, 1], hours=[0, 23]))
    assert list(table["group"]) == [str(h) for h in range(24)]
    assert table["transactions"].sum() == 2


def test_amount_bands():
    df = frame([5, 10, 999, 1000], [0, 0, 0, 1])
    table = a.fraud_rate_by_amount_band(df).set_index("group")
    assert table.loc["0–10", "transactions"] == 1  # bands include the lower edge only
    assert table.loc["10–25", "transactions"] == 1
    assert table.loc["500–1,000", "transactions"] == 1
    assert table.loc["1,000+", "fraud"] == 1


def test_amount_histogram_shares_sum_to_one():
    rng = np.random.default_rng(0)
    df = frame(rng.lognormal(3, 1, 500), (rng.random(500) < 0.1).astype(int).tolist())
    hist = a.amount_histogram(df, bins=20)
    assert len(hist) == 20
    assert hist["Fraud"].sum() == pytest.approx(1.0)
    assert hist["Legitimate"].sum() == pytest.approx(1.0)
    assert (hist["bin_end"] > hist["bin_start"]).all()


def test_outliers_iqr():
    df = frame([10, 10, 10, 10, 1000], [0, 0, 0, 0, 1])
    out = a.outliers_iqr(df)
    assert out["iqr"] == 0 and out["outliers"] == 1
    assert out["fraud_rate_outliers"] == 1.0 and out["fraud_rate_others"] == 0.0


def test_correlation_drops_constant_columns():
    df = frame([1, 2, 3, 4], [0, 0, 1, 1])  # hour is constant
    corr = a.correlation_matrix(df)
    assert "hour" not in corr
    assert corr.loc["amount", "is_fraud"] == pytest.approx(
        np.corrcoef([1, 2, 3, 4], [1.5, 1.5, 3.5, 3.5])[0, 1]
    )


def test_box_stats():
    stats = a.amount_box_stats(frame([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])).iloc[0]
    assert (stats["q1"], stats["median"], stats["q3"]) == (2, 3, 4)
    assert stats["lower_fence"] == 1 and stats["upper_fence"] == 5


def test_daily_activity():
    df = frame([1, 2, 3], [1, 0, 0], hours=[1, 2, 30])
    daily = a.daily_activity(df)
    assert daily["transactions"].tolist() == [2, 1]
    assert daily["fraud"].tolist() == [1, 0]
    assert daily["fraud_rate"].tolist() == [0.5, 0.0]


# --- Findings ---------------------------------------------------------------


def test_findings_call_out_a_clear_category():
    n = 400
    cats = ["risky"] * 100 + ["safe"] * 300
    labels = [1] * 30 + [0] * 70 + [1] * 3 + [0] * 297
    findings = " ".join(a.key_findings(frame([10.0] * n, labels, categories=cats)))
    assert "'risky' has the highest fraud rate" in findings
    assert "30.00%" in findings


def test_findings_do_not_report_noise():
    rng = np.random.default_rng(1)
    n = 2000
    df = frame(
        [10.0] * n,
        (rng.random(n) < 0.05).astype(int).tolist(),
        categories=rng.choice(list("abcd"), n).tolist(),
        hours=rng.integers(0, 24, n).tolist(),
    )
    findings = " ".join(a.key_findings(df))
    assert "No spending category has a fraud rate clearly above" in findings


def test_unlabelled_data_works():
    df = frame([1, 2, 3], None)
    assert a.overview(df)["fraud_rate"] is None
    assert a.key_findings(df) == []
    assert list(a.amount_histogram(df).columns[-1:]) == ["All"]
    assert set(a.amount_box_stats(df)["class"]) == {"All"}


# --- End to end on processed data ---------------------------------------------


def test_eda_runs_on_processed_dataset(messy_csv):
    data = process_dataset(messy_csv, suggest_mapping(SPARKOV_COLUMNS)).data
    df = add_basic_features(data)
    assert {"age_years", "distance_km"} <= set(df.columns)
    assert a.overview(df)["fraud"] == 3
    assert len(a.fraud_rate_by_age_band(df)) == len(a.AGE_LABELS)
    assert a.correlation_matrix(df).shape[0] >= 2
    assert a.key_findings(df)
