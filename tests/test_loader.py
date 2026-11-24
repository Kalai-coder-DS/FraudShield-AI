import pytest

from fraudshield.data.loader import DatasetLoadError, detect_format, iter_chunks, read_columns


def test_detects_semicolon_delimiter():
    data = b"transaction_id;amount;timestamp\nt1;10.5;2020-01-01\nt2;3;2020-01-02\n"
    assert detect_format(data).delimiter == ";"
    assert read_columns(data) == ["transaction_id", "amount", "timestamp"]


def test_detects_non_utf8_encoding():
    data = "id,merchant\n1,Café Zoë\n".encode("cp1252")
    fmt = detect_format(data)
    assert fmt.encoding != "utf-8-sig"
    chunk = next(iter_chunks(data, ["merchant"], fmt, 10))
    assert chunk["merchant"].iloc[0] == "Café Zoë"


def test_utf8_bom_is_removed_from_header():
    data = "﻿id,amount\n1,2\n".encode()
    assert read_columns(data) == ["id", "amount"]


@pytest.mark.parametrize(
    "data, message",
    [
        (b"", "empty"),
        (b"   \n\n", "empty"),
        (b"PK\x03\x04 rest of a zip", "Excel"),
        (b"abc\x00\x01\x02", "binary"),
    ],
)
def test_bad_files_get_clear_errors(data, message):
    with pytest.raises(DatasetLoadError, match=message):
        read_columns(data)


def test_single_column_file_is_rejected():
    with pytest.raises(DatasetLoadError, match="one column"):
        read_columns(b"just_one_column\n1\n2\n")


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(DatasetLoadError, match="not found"):
        read_columns(tmp_path / "nope.csv")


def test_reads_in_chunks_and_only_selected_columns():
    rows = "\n".join(f"t{i},{i},secret" for i in range(25))
    data = f"id,amount,private\n{rows}\n".encode()
    chunks = list(iter_chunks(data, ["id", "amount"], detect_format(data), 10))
    assert [len(c) for c in chunks] == [10, 10, 5]
    assert list(chunks[0].columns) == ["id", "amount"]


def test_common_missing_markers_become_missing():
    data = b"id,amount\n1,null\n2,N/A\n3,\n4,5\n"
    chunk = next(iter_chunks(data, ["id", "amount"], detect_format(data), 10))
    assert chunk["amount"].isna().sum() == 3
