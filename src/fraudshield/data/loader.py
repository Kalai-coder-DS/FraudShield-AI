"""Safe, memory-efficient CSV reading.

Uploaded files can be in the wrong format, use a different delimiter or text
encoding, or be too large to read in one go. This module:

* detects the text encoding and delimiter from the first 64 KB,
* rejects files that are clearly not CSV (Excel, binary) with a clear message,
* reads the file in chunks, and only the columns that are actually needed.

All values are read as text. Converting text to numbers and dates happens in
``validation.py``, where every value that fails to convert can be counted and
reported instead of silently disappearing.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

Source = str | Path | bytes

_SNIFF_BYTES = 64 * 1024
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
_DELIMITERS = ",;\t|"
# Extra text values treated as missing, on top of pandas' defaults ("", "NA", "NaN", ...)
NA_VALUES = ["null", "NULL", "None", "none", "N/A", "n/a", "-", "?"]


class DatasetLoadError(ValueError):
    """A problem with the file itself. The message is shown to the user as-is."""


@dataclass(frozen=True)
class CsvFormat:
    encoding: str
    delimiter: str


def _head(source: Source, size: int = _SNIFF_BYTES) -> bytes:
    if isinstance(source, bytes):
        return source[:size]
    path = Path(source)
    if not path.is_file():
        raise DatasetLoadError(f"File not found: {path}")
    with path.open("rb") as f:
        return f.read(size)


def _buffer(source: Source) -> str | io.BytesIO:
    """pandas reads paths directly; bytes are wrapped in a fresh buffer each time."""
    return io.BytesIO(source) if isinstance(source, bytes) else str(source)


def _decode(raw: bytes) -> tuple[str, str]:
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as err:
            # The 64 KB sample may cut a multi-byte character in half at the end.
            if encoding.startswith("utf-8") and err.start >= len(raw) - 3:
                return raw[: err.start].decode(encoding), encoding
    raise DatasetLoadError("The file's text encoding could not be recognised.")  # pragma: no cover


def detect_format(source: Source) -> CsvFormat:
    raw = _head(source)
    if not raw.strip():
        raise DatasetLoadError("The file is empty.")
    if raw.startswith(b"PK"):
        raise DatasetLoadError(
            "This looks like an Excel (.xlsx) or zip file. Open it in Excel and use "
            "'Save As → CSV (Comma delimited)', then upload the .csv file."
        )
    if b"\x00" in raw:
        raise DatasetLoadError("This is not a text CSV file (it contains binary data).")

    text, encoding = _decode(raw)
    sample = "\n".join(text.splitlines()[:50])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=_DELIMITERS).delimiter
    except csv.Error:
        delimiter = ","
    return CsvFormat(encoding=encoding, delimiter=delimiter)


def read_columns(source: Source, fmt: CsvFormat | None = None) -> list[str]:
    """Return the header row (column names) without reading the data."""
    fmt = fmt or detect_format(source)
    try:
        header = pd.read_csv(_buffer(source), nrows=0, sep=fmt.delimiter, encoding=fmt.encoding)
    except pd.errors.EmptyDataError as err:
        raise DatasetLoadError("The file has no header row.") from err
    except (pd.errors.ParserError, UnicodeDecodeError) as err:
        raise DatasetLoadError(f"The header row could not be read: {err}") from err

    columns = [str(c) for c in header.columns]
    if len(columns) < 2:
        raise DatasetLoadError(
            f"Only one column was found ('{columns[0] if columns else ''}'). The file may use "
            "an unusual separator, or it is not a table of transactions."
        )
    return columns


def iter_chunks(
    source: Source,
    usecols: list[str],
    fmt: CsvFormat,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    """Yield the selected columns as text, ``chunksize`` rows at a time."""
    try:
        reader = pd.read_csv(
            _buffer(source),
            sep=fmt.delimiter,
            encoding=fmt.encoding,
            usecols=usecols,
            dtype=str,
            na_values=NA_VALUES,
            skipinitialspace=True,
            chunksize=chunksize,
        )
        with reader:
            yield from reader
    except pd.errors.ParserError as err:
        raise DatasetLoadError(
            f"The file has a badly formatted row and could not be read completely: {err}"
        ) from err
    except UnicodeDecodeError as err:
        raise DatasetLoadError(
            "Part of the file uses a different text encoding and could not be read."
        ) from err
