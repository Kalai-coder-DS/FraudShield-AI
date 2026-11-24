import sqlite3

import pandas as pd
import pytest
from conftest import SPARKOV_COLUMNS

from fraudshield.data.schema import suggest_mapping
from fraudshield.data.validation import process_dataset
from fraudshield.storage import db


@pytest.fixture
def processed(messy_csv):
    return process_dataset(messy_csv, suggest_mapping(SPARKOV_COLUMNS))


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


def test_save_and_load_round_trip_is_exact(processed, db_path):
    ds_id = db.save_dataset("messy", "test", processed.data, processed.report, db_path=db_path)
    loaded = db.load_transactions(ds_id, db_path=db_path)
    pd.testing.assert_frame_equal(loaded, processed.data, check_categorical=False)
    assert db.load_report(ds_id, db_path=db_path) == processed.report


def test_load_selected_columns(processed, db_path):
    ds_id = db.save_dataset("m", "test", processed.data, processed.report, db_path=db_path)
    cols = db.load_transactions(ds_id, ["amount", "is_fraud"], db_path=db_path).columns
    assert list(cols) == ["amount", "is_fraud"]


def test_pagination(processed, db_path):
    ds_id = db.save_dataset("m", "test", processed.data, processed.report, db_path=db_path)
    pages = [db.fetch_page(ds_id, p, 10, db_path=db_path) for p in (1, 2, 3, 4)]
    assert [len(p) for p in pages] == [10, 10, 10, 6]
    ids = pd.concat(pages)["transaction_id"].tolist()
    assert ids == processed.data["transaction_id"].tolist()
    with pytest.raises(ValueError):
        db.fetch_page(ds_id, 0, 10, db_path=db_path)


def test_only_one_dataset_is_active(processed, db_path):
    first = db.save_dataset("a", "test", processed.data, processed.report, db_path=db_path)
    second = db.save_dataset("b", "test", processed.data, processed.report, db_path=db_path)
    assert db.get_active_dataset(db_path).id == second
    db.set_active(first, db_path)
    active = [d for d in db.list_datasets(db_path) if d.is_active]
    assert [d.id for d in active] == [first]


def test_delete_removes_dataset_and_transactions(processed, db_path):
    ds_id = db.save_dataset("a", "test", processed.data, processed.report, db_path=db_path)
    db.delete_dataset(ds_id, db_path)
    assert db.list_datasets(db_path) == []
    with db.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
    with pytest.raises(KeyError):
        db.delete_dataset(ds_id, db_path)


def test_empty_name_is_rejected(processed, db_path):
    with pytest.raises(ValueError, match="name"):
        db.save_dataset("   ", "test", processed.data, processed.report, db_path=db_path)


def test_failed_save_leaves_no_partial_dataset(processed, db_path):
    broken = processed.data.copy()
    broken["amount"] = broken["amount"].astype(object)
    broken.loc[0, "amount"] = object()  # SQLite cannot store this value
    with pytest.raises(sqlite3.Error):
        db.save_dataset("bad", "test", broken, processed.report, db_path=db_path)
    assert db.list_datasets(db_path) == []
