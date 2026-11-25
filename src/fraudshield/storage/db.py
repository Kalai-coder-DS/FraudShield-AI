"""SQLite storage for datasets.

Why SQLite: it is built into Python (nothing to install, no server running in
the background) and stores the cleaned transactions on disk. Pages of a large
table can then be read with ``LIMIT``/``OFFSET`` instead of keeping every row in
memory, and saved datasets survive an app restart.

Tables
------
datasets      one row per saved dataset: name, source, quality report, active flag
transactions  cleaned transactions of every dataset, linked by dataset_id
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from fraudshield import config
from fraudshield.data.schema import FIELDS, FIELDS_BY_NAME
from fraudshield.data.validation import DataReport

_SQL_TYPES = {
    "id": "TEXT",
    "category": "TEXT",
    "datetime": "TEXT",  # ISO-8601 text; sorts correctly as text
    "date": "TEXT",
    "numeric": "REAL",
    "label": "INTEGER",
}
_COLUMNS = [f.name for f in FIELDS]

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS datasets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    source        TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    rows_clean    INTEGER NOT NULL,
    has_label     INTEGER NOT NULL,
    quality_score REAL    NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 0,
    report_json   TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    row_number INTEGER NOT NULL,
    {", ".join(f'"{f.name}" {_SQL_TYPES[f.kind]}' for f in FIELDS)},
    PRIMARY KEY (dataset_id, row_number)
);
CREATE INDEX IF NOT EXISTS idx_tx_id   ON transactions(dataset_id, transaction_id);
CREATE INDEX IF NOT EXISTS idx_tx_time ON transactions(dataset_id, timestamp);
CREATE TABLE IF NOT EXISTS scores (
    dataset_id        INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    row_number        INTEGER NOT NULL,
    model_sha256      TEXT    NOT NULL,
    fraud_probability REAL    NOT NULL,
    predicted_fraud   INTEGER NOT NULL,
    risk_tier         TEXT    NOT NULL,
    PRIMARY KEY (dataset_id, row_number)
);
CREATE INDEX IF NOT EXISTS idx_scores_prob ON scores(dataset_id, fraud_probability);
"""


@dataclass(frozen=True)
class DatasetInfo:
    id: int
    name: str
    source: str
    created_at: str
    rows_clean: int
    has_label: bool
    quality_score: float
    is_active: bool


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the database, create tables if needed, commit on success.

    The default location is read from config at call time, so tests can point
    the whole app at a temporary database.
    """
    db_path = Path(db_path or config.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _to_sql_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert typed columns into values SQLite can store."""
    out = pd.DataFrame(index=df.index)
    for name in _COLUMNS:
        if name not in df:
            out[name] = None
            continue
        col = df[name]
        kind = FIELDS_BY_NAME[name].kind
        if kind == "datetime":
            out[name] = col.dt.strftime("%Y-%m-%d %H:%M:%S").astype(object)
        elif kind == "date":
            out[name] = col.dt.strftime("%Y-%m-%d").astype(object)
        else:
            out[name] = col.astype(object)
        out[name] = out[name].where(col.notna(), None)
    return out


def _from_sql_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Restore the same dtypes that ``validation.process_dataset`` produces."""
    for name in list(df.columns):
        if name not in FIELDS_BY_NAME:
            continue
        kind = FIELDS_BY_NAME[name].kind
        if kind in ("datetime", "date"):
            df[name] = pd.to_datetime(df[name]).astype("datetime64[ns]")
        elif kind == "numeric":
            df[name] = pd.to_numeric(df[name]).astype("float64")
        elif kind == "label":
            df[name] = df[name].astype("Int8")
        elif kind == "category" or name == "card_id":
            df[name] = df[name].astype("category")
        else:
            df[name] = df[name].astype(object)
    return df


def save_dataset(
    name: str,
    source: str,
    data: pd.DataFrame,
    report: DataReport,
    make_active: bool = True,
    db_path: Path | None = None,
    chunksize: int = 50_000,
) -> int:
    """Store a cleaned dataset and its report. Returns the new dataset id."""
    name = name.strip()
    if not name:
        raise ValueError("Please give the dataset a name.")
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO datasets (name, source, created_at, rows_clean, has_label, "
            "quality_score, report_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                source,
                datetime.now().isoformat(timespec="seconds"),
                len(data),
                int(report.has_label),
                report.quality_score,
                json.dumps(report.to_dict()),
            ),
        )
        dataset_id = int(cur.lastrowid)
        placeholders = ", ".join("?" * (len(_COLUMNS) + 2))
        columns = ", ".join(["dataset_id", "row_number"] + [f'"{c}"' for c in _COLUMNS])
        sql = f"INSERT INTO transactions ({columns}) VALUES ({placeholders})"
        for start in range(0, len(data), chunksize):
            part = _to_sql_frame(data.iloc[start : start + chunksize])
            rows = (
                (dataset_id, start + i, *values)
                for i, values in enumerate(part.itertuples(index=False, name=None))
            )
            conn.executemany(sql, rows)
        if make_active:
            _set_active(conn, dataset_id)
    return dataset_id


def _set_active(conn: sqlite3.Connection, dataset_id: int) -> None:
    conn.execute("UPDATE datasets SET is_active = (id = ?)", (dataset_id,))


def set_active(dataset_id: int, db_path: Path | None = None) -> None:
    with connect(db_path) as conn:
        if conn.execute("SELECT 1 FROM datasets WHERE id = ?", (dataset_id,)).fetchone() is None:
            raise KeyError(f"Dataset {dataset_id} does not exist.")
        _set_active(conn, dataset_id)


def list_datasets(db_path: Path | None = None) -> list[DatasetInfo]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, source, created_at, rows_clean, has_label, quality_score, "
            "is_active FROM datasets ORDER BY id DESC"
        ).fetchall()
    return [DatasetInfo(r[0], r[1], r[2], r[3], r[4], bool(r[5]), r[6], bool(r[7])) for r in rows]


def get_active_dataset(db_path: Path | None = None) -> DatasetInfo | None:
    return next((d for d in list_datasets(db_path) if d.is_active), None)


def load_report(dataset_id: int, db_path: Path | None = None) -> DataReport:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT report_json FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"Dataset {dataset_id} does not exist.")
    return DataReport.from_dict(json.loads(row[0]))


def _dataset_columns(dataset_id: int, db_path: Path) -> list[str]:
    """Canonical columns this dataset actually has (from its saved mapping)."""
    mapped = load_report(dataset_id, db_path).mapping
    return [c for c in _COLUMNS if c in mapped]


def load_transactions(
    dataset_id: int,
    columns: list[str] | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load a whole dataset (optionally only some columns) with correct dtypes."""
    available = _dataset_columns(dataset_id, db_path)
    selected = [c for c in (columns or available) if c in available]
    sql_cols = ", ".join(f'"{c}"' for c in selected)
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            f"SELECT {sql_cols} FROM transactions WHERE dataset_id = ? ORDER BY row_number",
            conn,
            params=(dataset_id,),
        )
    return _from_sql_frame(df)


def fetch_page(
    dataset_id: int,
    page: int,
    page_size: int,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Return one page of transactions (page numbers start at 1)."""
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be at least 1")
    sql_cols = ", ".join(f'"{c}"' for c in _dataset_columns(dataset_id, db_path))
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            f"SELECT {sql_cols} FROM transactions WHERE dataset_id = ? ORDER BY row_number "
            "LIMIT ? OFFSET ?",
            conn,
            params=(dataset_id, page_size, (page - 1) * page_size),
        )
    return _from_sql_frame(df)


def delete_dataset(dataset_id: int, db_path: Path | None = None) -> None:
    """Delete a dataset and all its transactions."""
    with connect(db_path) as conn:
        cur = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        if cur.rowcount == 0:
            raise KeyError(f"Dataset {dataset_id} does not exist.")
        # ON DELETE CASCADE removes the transactions and scores; this is a safety net.
        conn.execute("DELETE FROM transactions WHERE dataset_id = ?", (dataset_id,))
        conn.execute("DELETE FROM scores WHERE dataset_id = ?", (dataset_id,))


# ---------------------------------------------------------------------------
# Scores (model output per transaction)
# ---------------------------------------------------------------------------


def scores_model(dataset_id: int, db_path: Path | None = None) -> str | None:
    """Fingerprint of the model that produced the stored scores (None if not scored)."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT model_sha256 FROM scores WHERE dataset_id = ? LIMIT 1", (dataset_id,)
        ).fetchone()
    return row[0] if row else None


def save_scores(
    dataset_id: int, scores: pd.DataFrame, model_sha256: str, db_path: Path | None = None
) -> None:
    """Replace the stored scores of a dataset. ``scores`` is indexed by row number."""
    rows = (
        (dataset_id, int(i), model_sha256, float(p), int(y), str(t))
        for i, p, y, t in zip(
            scores.index,
            scores["fraud_probability"],
            scores["predicted_fraud"],
            scores["risk_tier"],
            strict=True,
        )
    )
    with connect(db_path) as conn:
        conn.execute("DELETE FROM scores WHERE dataset_id = ?", (dataset_id,))
        conn.executemany("INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?)", rows)


def load_scores(dataset_id: int, db_path: Path | None = None) -> pd.DataFrame:
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT row_number, fraud_probability, predicted_fraud, risk_tier FROM scores "
            "WHERE dataset_id = ? ORDER BY row_number",
            conn,
            params=(dataset_id,),
        )
    return df.set_index("row_number")


@dataclass
class TransactionFilter:
    """Search criteria for the investigation table. Empty fields mean "any"."""

    text: str = ""  # part of a transaction ID
    tiers: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    min_amount: float | None = None
    max_amount: float | None = None
    date_from: str | None = None  # "YYYY-MM-DD"
    date_to: str | None = None
    label: int | None = None  # 1 = known fraud, 0 = known legitimate
    flagged_only: bool = False


def _where(dataset_id: int, f: TransactionFilter) -> tuple[str, list]:
    # Only placeholders (?) carry user input, never string formatting: no SQL injection.
    clauses, params = ["t.dataset_id = ?"], [dataset_id]
    if f.text.strip():
        clauses.append("t.transaction_id LIKE ? ESCAPE '\\'")
        escaped = f.text.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.append(f"%{escaped}%")
    if f.tiers:
        clauses.append(f"s.risk_tier IN ({', '.join('?' * len(f.tiers))})")
        params += list(f.tiers)
    if f.categories:
        clauses.append(f"t.category IN ({', '.join('?' * len(f.categories))})")
        params += list(f.categories)
    if f.min_amount is not None:
        clauses.append("t.amount >= ?")
        params.append(f.min_amount)
    if f.max_amount is not None:
        clauses.append("t.amount <= ?")
        params.append(f.max_amount)
    if f.date_from:
        clauses.append("t.timestamp >= ?")
        params.append(f.date_from)
    if f.date_to:
        clauses.append("t.timestamp < date(?, '+1 day')")
        params.append(f.date_to)
    if f.label is not None:
        clauses.append("t.is_fraud = ?")
        params.append(int(f.label))
    if f.flagged_only:
        clauses.append("s.predicted_fraud = 1")
    return " AND ".join(clauses), params


_JOIN = (
    "FROM transactions t JOIN scores s "
    "ON s.dataset_id = t.dataset_id AND s.row_number = t.row_number"
)


def count_transactions(dataset_id: int, f: TransactionFilter, db_path: Path | None = None) -> int:
    where, params = _where(dataset_id, f)
    with connect(db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) {_JOIN} WHERE {where}", params).fetchone()[0])


def search_transactions(
    dataset_id: int,
    f: TransactionFilter,
    page: int = 1,
    page_size: int = 50,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """One page of scored transactions matching ``f``, riskiest first."""
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be at least 1")
    where, params = _where(dataset_id, f)
    cols = ", ".join(f't."{c}"' for c in _dataset_columns(dataset_id, db_path))
    sql = (
        f"SELECT t.row_number, {cols}, s.fraud_probability, s.predicted_fraud, s.risk_tier "
        f"{_JOIN} WHERE {where} ORDER BY s.fraud_probability DESC, t.row_number "
        "LIMIT ? OFFSET ?"
    )
    with connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=[*params, page_size, (page - 1) * page_size])
    return _from_sql_frame(df.set_index("row_number"))


def get_transaction(
    dataset_id: int, transaction_id: str, db_path: Path | None = None
) -> pd.DataFrame:
    """A single transaction (exact ID match) with its row number as index."""
    cols = ", ".join(f'"{c}"' for c in _dataset_columns(dataset_id, db_path))
    with connect(db_path) as conn:
        df = pd.read_sql_query(
            f"SELECT row_number, {cols} FROM transactions "
            "WHERE dataset_id = ? AND transaction_id = ?",
            conn,
            params=(dataset_id, transaction_id),
        )
    return _from_sql_frame(df.set_index("row_number"))
