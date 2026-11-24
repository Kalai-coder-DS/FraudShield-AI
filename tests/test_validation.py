from datetime import datetime

import pandas as pd
import pytest
from conftest import SPARKOV_COLUMNS, make_row

from fraudshield.data.schema import suggest_mapping
from fraudshield.data.validation import (
    DataReport,
    DatasetValidationError,
    process_dataset,
    quality_score,
)

MAPPING = suggest_mapping(SPARKOV_COLUMNS)


def issue_codes(report):
    return {(i.code, i.column): i for i in report.issues}


def test_clean_file_has_full_quality(clean_csv):
    result = process_dataset(clean_csv, MAPPING)
    r = result.report
    assert r.rows_in_file == r.rows_analysed == r.rows_clean == 40
    assert r.quality_score == 100.0
    assert r.fraud_rows == 4 and r.legitimate_rows == 36
    assert r.count_by_severity("error") == 0


def test_types_are_converted(clean_csv):
    df = process_dataset(clean_csv, MAPPING).data
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["amount"].dtype == "float64"
    assert df["is_fraud"].dtype == "Int8"
    assert isinstance(df["category"].dtype, pd.CategoricalDtype)
    # Card numbers must stay exact text, never become floats like 2.7e15
    assert df["card_id"].astype(str).isin(["2703186189652095", "4613314721966"]).all()


def test_personal_columns_are_not_loaded(clean_csv):
    result = process_dataset(clean_csv, MAPPING)
    assert not {"first", "last", "street"} & set(result.data.columns)
    unused = issue_codes(result.report)[("unused_columns", None)]
    assert "personal data" in unused.message


def test_messy_file_every_problem_is_caught(messy_csv):
    result = process_dataset(messy_csv, MAPPING, chunksize=7)  # small chunks on purpose
    r = result.report
    codes = issue_codes(r)

    # Removed rows: invalid amount, negative amount, bad date, missing card, duplicate, reused ID
    assert r.rows_analysed == 42
    assert r.rows_clean == 36
    assert r.quality_score == quality_score(36, 42)
    assert codes[("invalid_format", "amount")].examples == ["abc"]
    assert codes[("out_of_range", "amount")].severity == "error"
    assert codes[("invalid_format", "timestamp")].count == 1
    assert codes[("missing_value", "card_id")].severity == "error"
    assert codes[("duplicate_row", None)].count == 1
    assert codes[("duplicate_id", "transaction_id")].examples == ["t00011"]

    # Kept rows with a cleared value or a warning
    assert codes[("out_of_range", "lat")].severity == "warning"
    assert codes[("invalid_format", "is_fraud")].examples == ["maybe"]
    assert ("implausible_dob", "dob") in codes
    assert ("zero_amount", "amount") in codes

    df = result.data
    assert df.loc[df["transaction_id"] == "t00004", "lat"].isna().all()
    assert df.loc[df["transaction_id"] == "t00005", "is_fraud"].isna().all()
    assert df.loc[df["transaction_id"] == "t00011", "amount"].item() == pytest.approx(48.5)
    assert df["transaction_id"].is_unique
    assert df["timestamp"].is_monotonic_increasing


def test_errors_are_listed_before_warnings(messy_csv):
    severities = [i.severity for i in process_dataset(messy_csv, MAPPING).report.issues]
    order = {"error": 0, "warning": 1, "info": 2}
    assert severities == sorted(severities, key=order.get)


def test_chunk_size_does_not_change_result(messy_csv):
    small = process_dataset(messy_csv, MAPPING, chunksize=3)
    large = process_dataset(messy_csv, MAPPING, chunksize=1000)
    pd.testing.assert_frame_equal(small.data, large.data)
    assert small.report.rows_clean == large.report.rows_clean


def test_sampling_is_reproducible_and_reduces_rows(tmp_path):
    path = tmp_path / "big.csv"
    pd.DataFrame([make_row(i) for i in range(2000)], columns=SPARKOV_COLUMNS).to_csv(
        path, index=False
    )
    a = process_dataset(path, MAPPING, sample_fraction=0.25, seed=1, chunksize=500)
    b = process_dataset(path, MAPPING, sample_fraction=0.25, seed=1, chunksize=500)
    assert a.report.rows_in_file == 2000
    assert 400 < a.report.rows_analysed < 600
    pd.testing.assert_frame_equal(a.data, b.data)
    assert 0.06 < a.report.fraud_rate < 0.14  # true rate is 0.10


def test_future_dates_are_flagged(tmp_path):
    path = tmp_path / "f.csv"
    rows = [make_row(i) for i in range(5)]
    rows[0]["trans_date_trans_time"] = "2099-01-01 00:00:00"
    pd.DataFrame(rows, columns=SPARKOV_COLUMNS).to_csv(path, index=False)
    report = process_dataset(path, MAPPING, now=datetime(2026, 1, 1)).report
    assert ("future_timestamp", "timestamp") in issue_codes(report)


def test_unlabelled_dataset_is_supported(tmp_path, clean_rows):
    path = tmp_path / "nolabel.csv"
    clean_rows.drop(columns="is_fraud").to_csv(path, index=False)
    mapping = suggest_mapping([c for c in SPARKOV_COLUMNS if c != "is_fraud"])
    report = process_dataset(path, mapping).report
    assert report.has_label is False
    assert report.fraud_rate is None


def test_single_class_labels_warn(tmp_path, clean_rows):
    path = tmp_path / "oneclass.csv"
    clean_rows.assign(is_fraud="0").to_csv(path, index=False)
    assert ("single_class", "is_fraud") in issue_codes(process_dataset(path, MAPPING).report)


def test_invalid_mapping_is_rejected(clean_csv):
    with pytest.raises(DatasetValidationError, match="Required field"):
        process_dataset(clean_csv, {**MAPPING, "amount": None})


def test_mapping_to_missing_column_is_rejected(clean_csv):
    with pytest.raises(DatasetValidationError, match="not found"):
        process_dataset(clean_csv, {**MAPPING, "amount": "no_such_column"})


def test_header_only_file_is_rejected(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text(",".join(SPARKOV_COLUMNS) + "\n")
    with pytest.raises(DatasetValidationError, match="no data rows"):
        process_dataset(path, MAPPING)


def test_report_survives_json_round_trip(messy_csv):
    import json

    report = process_dataset(messy_csv, MAPPING).report
    restored = DataReport.from_dict(json.loads(json.dumps(report.to_dict())))
    assert restored == report


def test_quality_score_edge_cases():
    assert quality_score(0, 0) == 0.0
    assert quality_score(1, 3) == 33.33
