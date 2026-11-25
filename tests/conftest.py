"""Shared test fixtures.

The rows below are hand-made TEST DATA in the Sparkov column format. They exist
only to check that validation catches each kind of problem. They are never
shown in the app or used to train a model.
"""

from __future__ import annotations

import pandas as pd
import pytest

SPARKOV_COLUMNS = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]


def make_row(i: int, **overrides) -> dict:
    row = {
        "Unnamed: 0": i,
        "trans_date_trans_time": f"2020-06-{1 + i % 28:02d} {i % 24:02d}:15:00",
        "cc_num": "2703186189652095" if i % 2 else "4613314721966",
        "merchant": f"fraud_Merchant {i % 5}",
        "category": ["grocery_pos", "shopping_net", "gas_transport", "misc_net"][i % 4],
        "amt": f"{10 + i * 3.5:.2f}",
        "first": "Test",
        "last": "Person",
        "gender": "F" if i % 2 else "M",
        "street": "1 Test Street",
        "city": "Testville",
        "state": "NC",
        "zip": "28654",
        "lat": "36.0788",
        "long": "-81.1781",
        "city_pop": "3495",
        "job": "Engineer",
        "dob": "1988-03-09",
        "trans_num": f"t{i:05d}",
        "unix_time": "1371816865",
        "merch_lat": "36.011293",
        "merch_long": "-82.048315",
        "is_fraud": "1" if i % 10 == 0 else "0",
    }
    row.update(overrides)
    return row


@pytest.fixture
def clean_rows() -> pd.DataFrame:
    return pd.DataFrame([make_row(i) for i in range(40)], columns=SPARKOV_COLUMNS)


@pytest.fixture
def clean_csv(tmp_path, clean_rows):
    path = tmp_path / "clean.csv"
    clean_rows.to_csv(path, index=False)
    return path


@pytest.fixture
def messy_csv(tmp_path):
    """40 rows with one known problem each in rows 0-9."""
    rows = [make_row(i) for i in range(40)]
    rows[0]["amt"] = "abc"  # invalid amount -> removed
    rows[1]["amt"] = "-5.00"  # negative amount -> removed
    rows[2]["trans_date_trans_time"] = "not a date"  # invalid timestamp -> removed
    rows[3]["cc_num"] = ""  # missing card -> removed
    rows[4]["lat"] = "123.0"  # latitude out of range -> cleared, row kept
    rows[5]["is_fraud"] = "maybe"  # invalid label -> cleared, row kept
    rows[6]["dob"] = "2030-01-01"  # born after transaction -> cleared, row kept
    rows[7]["amt"] = "0"  # zero amount -> warning, kept
    rows.append(dict(rows[10]))  # exact duplicate of row 10 -> removed
    rows.append(make_row(11, amt="999.99"))  # reuses ID t00011 -> removed
    df = pd.DataFrame(rows, columns=SPARKOV_COLUMNS)
    path = tmp_path / "messy.csv"
    df.to_csv(path, index=False)
    return path


def make_signal_rows(n: int, start: str, seed: int) -> pd.DataFrame:
    """TEST DATA in Sparkov format with a planted fraud pattern.

    Fraud is made more likely late at night and in two categories, and fraud
    amounts are several times the card's usual amount. Used only to check that
    the model pipeline can find a known pattern.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    ts = pd.Timestamp(start) + pd.to_timedelta(np.sort(rng.integers(0, 120 * 86400, n)), unit="s")
    cards = rng.integers(0, 60, n)
    scale = np.exp(np.random.default_rng(0).normal(3.5, 0.4, 60))
    cats = np.array(["grocery_pos", "shopping_net", "gas_transport", "misc_net", "home"])
    cat = cats[rng.integers(0, len(cats), n)]
    amt = scale[cards] * rng.lognormal(0, 0.5, n)
    logit = -5.0 + 2.0 * np.isin(ts.hour, [22, 23, 0, 1, 2, 3]) + 1.0 * np.isin(cat, cats[[1, 3]])
    fraud = rng.random(n) < 1 / (1 + np.exp(-logit))
    amt = np.where(fraud, amt * rng.uniform(3, 8, n), amt)
    rows = pd.DataFrame([make_row(i) for i in range(n)], columns=SPARKOV_COLUMNS)
    rows["trans_date_trans_time"] = ts.strftime("%Y-%m-%d %H:%M:%S")
    rows["cc_num"] = (4_000_000_000_000 + cards).astype(str)
    rows["category"] = cat
    rows["amt"] = np.round(amt, 2).astype(str)
    rows["trans_num"] = [f"s{seed}-{i}" for i in range(n)]
    rows["is_fraud"] = fraud.astype(int).astype(str)
    return rows
