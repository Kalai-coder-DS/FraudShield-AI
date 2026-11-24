import sys

import pandas as pd
import pytest
from conftest import SPARKOV_COLUMNS, make_row

from fraudshield.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from make_sample import make_sample  # noqa: E402


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "fraudTest.csv"
    pd.DataFrame([make_row(i) for i in range(3000)], columns=SPARKOV_COLUMNS).to_csv(
        path, index=False
    )
    return path


def test_sample_keeps_fraud_rate_and_drops_personal_columns(source, tmp_path):
    out = tmp_path / "sample.csv"
    sample = make_sample(source, out, rows=600, chunksize=700)
    assert out.is_file()
    assert 550 <= len(sample) <= 650
    assert (sample["is_fraud"] == "1").mean() == pytest.approx(0.10, abs=0.01)
    assert not {"first", "last", "street", "Unnamed: 0"} & set(sample.columns)
    assert sample["trans_date_trans_time"].is_monotonic_increasing


def test_sample_is_reproducible(source, tmp_path):
    a = make_sample(source, tmp_path / "a.csv", rows=500, seed=7)
    b = make_sample(source, tmp_path / "b.csv", rows=500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_missing_input_gives_download_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        make_sample(tmp_path / "missing.csv", tmp_path / "out.csv", rows=10)
