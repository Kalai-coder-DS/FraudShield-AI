"""Validation, type conversion and cleaning of transaction data.

``process_dataset`` turns a raw CSV into a clean, correctly typed table plus a
``DataReport`` that explains every problem found. It works in chunks so large
files never have to sit in memory as text.

What happens to each kind of problem:

=================================  ==========  ==================================
Problem                            Severity    Action
=================================  ==========  ==================================
Required value missing / invalid   error       row removed
Negative amount                    error       row removed
Duplicate transaction ID           error       later copies removed
Exact duplicate row                warning     later copies removed
Optional value invalid / out of    warning     value cleared (set to missing)
range
Optional value missing             warning     kept as missing
Zero amount, future timestamp      warning     kept
=================================  ==========  ==================================

Missing optional values are deliberately *not* filled in here. Filling them
(imputation) uses statistics such as the median, and those must be learned from
training data only, otherwise information from the test data leaks into the
model. That happens inside the model pipeline in Phase 4.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from fraudshield.config import CSV_CHUNK_SIZE, RANDOM_SEED
from fraudshield.data import loader
from fraudshield.data.schema import (
    FIELDS_BY_NAME,
    LABEL_FIELD,
    PERSONAL_COLUMNS,
    REQUIRED_FIELDS,
    Mapping,
    check_mapping,
    mapped_fields,
)

Severity = Literal["error", "warning", "info"]

# Fields stored with pandas' memory-saving "category" dtype (few distinct values).
_CATEGORY_STORAGE = {"card_id"}
_TRUE_VALUES = {"1", "1.0", "true", "yes", "y", "fraud"}
_FALSE_VALUES = {"0", "0.0", "false", "no", "n", "legit", "legitimate"}
_CURRENCY_CHARS = "$€£₹¥ "
_MAX_EXAMPLES = 3


class DatasetValidationError(ValueError):
    """The dataset cannot be used at all. The message is shown to the user."""


@dataclass
class Issue:
    code: str
    severity: Severity
    message: str  # plain-language explanation and what was done about it
    column: str | None = None
    count: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass
class ColumnProfile:
    column: str
    source_column: str
    kind: str
    dtype: str
    missing: int
    missing_pct: float
    invalid: int
    unique: int
    example: str


@dataclass
class DataReport:
    rows_in_file: int
    rows_analysed: int
    sample_fraction: float
    rows_clean: int
    quality_score: float  # % of analysed rows that are usable (see quality_score docs)
    mapping: dict[str, str]
    unused_columns: list[str]
    has_label: bool
    fraud_rows: int | None
    legitimate_rows: int | None
    date_min: str | None
    date_max: str | None
    issues: list[Issue]
    columns: list[ColumnProfile]

    @property
    def fraud_rate(self) -> float | None:
        if not self.has_label or not self.fraud_rows and not self.legitimate_rows:
            return None
        return self.fraud_rows / (self.fraud_rows + self.legitimate_rows)

    def count_by_severity(self, severity: Severity) -> int:
        return sum(1 for issue in self.issues if issue.severity == severity)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataReport:
        data = dict(data)
        data["issues"] = [Issue(**i) for i in data["issues"]]
        data["columns"] = [ColumnProfile(**c) for c in data["columns"]]
        return cls(**data)


@dataclass
class ProcessedDataset:
    data: pd.DataFrame  # clean, typed rows using canonical column names
    report: DataReport


# ---------------------------------------------------------------------------
# Type conversion (one column of text -> typed values)
# ---------------------------------------------------------------------------


def _to_datetime(text: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        # pandas warns when it has to guess the date format; failures are counted instead.
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(text, errors="coerce")
        retry = parsed.isna() & text.notna()
        if retry.any():  # mixed formats in one file: parse the leftovers one by one
            parsed[retry] = pd.to_datetime(text[retry], errors="coerce", format="mixed")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert(None)  # store everything as naive UTC
    return parsed.astype("datetime64[ns]")


def _to_numeric(text: pd.Series) -> pd.Series:
    cleaned = text.str.strip(_CURRENCY_CHARS)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _to_label(text: pd.Series) -> pd.Series:
    lowered = text.str.strip().str.lower()
    out = pd.Series(pd.NA, index=text.index, dtype="Int8")
    out[lowered.isin(_TRUE_VALUES)] = 1
    out[lowered.isin(_FALSE_VALUES)] = 0
    return out


def _convert(name: str, text: pd.Series) -> pd.Series:
    kind = FIELDS_BY_NAME[name].kind
    if kind in ("datetime", "date"):
        return _to_datetime(text)
    if kind == "numeric":
        return _to_numeric(text)
    if kind == "label":
        return _to_label(text)
    if kind == "category" or name in _CATEGORY_STORAGE:
        return text.astype("category")
    return text.astype(object)  # identifiers stay as exact text (card numbers are not numbers)


# ---------------------------------------------------------------------------
# Counters collected across chunks
# ---------------------------------------------------------------------------


class _Tally:
    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = defaultdict(int)
        self.examples: dict[tuple[str, str], list[str]] = defaultdict(list)

    def add(self, code: str, column: str, mask: pd.Series, values: pd.Series) -> None:
        n = int(mask.sum())
        if n == 0:
            return
        key = (code, column)
        self.counts[key] += n
        room = _MAX_EXAMPLES - len(self.examples[key])
        if room > 0:
            self.examples[key] += [str(v) for v in values[mask].head(room)]

    def get(self, code: str, column: str) -> tuple[int, list[str]]:
        return self.counts.get((code, column), 0), self.examples.get((code, column), [])


def _convert_chunk(chunk: pd.DataFrame, tally: _Tally) -> pd.DataFrame:
    typed = {}
    for name in chunk.columns:
        text = chunk[name].str.strip()
        text = text.mask(text == "")
        tally.add("missing", name, text.isna(), text)
        values = _convert(name, text)
        tally.add("invalid", name, text.notna() & values.isna(), text)

        valid_range = FIELDS_BY_NAME[name].valid_range
        if valid_range is not None:
            low, high = valid_range
            out_of_range = values.notna() & ((values < low) | (values > high))
            tally.add("out_of_range", name, out_of_range, values)
            values = values.mask(out_of_range)
        typed[name] = values
    return pd.DataFrame(typed, index=chunk.index)


def _concat_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate while keeping category columns as categories (saves memory)."""
    for col in chunks[0].columns:
        if isinstance(chunks[0][col].dtype, pd.CategoricalDtype):
            categories = pd.Index(sorted(set().union(*(c[col].cat.categories for c in chunks))))
            for c in chunks:
                c[col] = c[col].cat.set_categories(categories)
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def process_dataset(
    source: loader.Source,
    mapping: Mapping,
    sample_fraction: float = 1.0,
    seed: int = RANDOM_SEED,
    chunksize: int = CSV_CHUNK_SIZE,
    now: datetime | None = None,
) -> ProcessedDataset:
    """Read, validate, type and clean a transaction CSV.

    ``sample_fraction`` < 1 keeps a random share of rows (each row is kept with
    that probability, using a fixed seed). Simple random sampling keeps the
    fraud rate the same on average, which matters for honest analysis.
    """
    problems = check_mapping(mapping)
    if problems:
        raise DatasetValidationError(" ".join(problems))
    if not 0 < sample_fraction <= 1:
        raise DatasetValidationError("Sample fraction must be between 0 and 1.")

    fields = mapped_fields(mapping)
    source_to_canonical = {src: name for name, src in fields.items()}
    fmt = loader.detect_format(source)
    all_columns = loader.read_columns(source, fmt)
    missing_cols = [src for src in fields.values() if src not in all_columns]
    if missing_cols:
        raise DatasetValidationError(f"Columns not found in the file: {', '.join(missing_cols)}")

    rng = np.random.default_rng(seed)
    tally = _Tally()
    chunks: list[pd.DataFrame] = []
    rows_in_file = 0
    for chunk in loader.iter_chunks(source, list(fields.values()), fmt, chunksize):
        rows_in_file += len(chunk)
        if sample_fraction < 1:
            chunk = chunk[rng.random(len(chunk)) < sample_fraction]
        if chunk.empty:
            continue
        chunk = chunk.rename(columns=source_to_canonical)[list(fields)]
        chunks.append(_convert_chunk(chunk, tally))

    if rows_in_file == 0:
        raise DatasetValidationError("The file has a header row but no data rows.")
    if not chunks:
        raise DatasetValidationError(
            "No rows were selected. Use a larger sample fraction for this small file."
        )

    df = _concat_chunks(chunks)
    del chunks
    issues = _row_checks(df, tally, now or datetime.now())
    clean, removal_issues = _clean(df)
    issues = removal_issues + issues
    issues += _label_issues(clean) if LABEL_FIELD in clean else []
    issues += _unused_column_issues(all_columns, set(fields.values()))

    report = _build_report(
        df, clean, fields, all_columns, rows_in_file, sample_fraction, issues, tally
    )
    return ProcessedDataset(data=clean, report=report)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _row_checks(df: pd.DataFrame, tally: _Tally, now: datetime) -> list[Issue]:
    issues: list[Issue] = []
    for name in df.columns:
        spec = FIELDS_BY_NAME[name]
        req = spec.required
        n, ex = tally.get("missing", name)
        if n:
            issues.append(
                Issue(
                    "missing_value",
                    "error" if req else "warning",
                    f"{n:,} rows have no '{name}'. " + _missing_action(name, req),
                    name,
                    n,
                )
            )
        n, ex = tally.get("invalid", name)
        if n:
            issues.append(
                Issue(
                    "invalid_format",
                    "error" if req else "warning",
                    f"{n:,} values in '{name}' could not be read as {_kind_text(spec.kind)}. "
                    + ("These rows are removed." if req else "These values are cleared."),
                    name,
                    n,
                    ex,
                )
            )
        n, ex = tally.get("out_of_range", name)
        if n:
            low, high = spec.valid_range or (None, None)
            issues.append(
                Issue(
                    "out_of_range",
                    "error" if req else "warning",
                    f"{n:,} values in '{name}' are outside the valid range "
                    f"({_range_text(low, high)}). "
                    + ("These rows are removed." if req else "These values are cleared."),
                    name,
                    n,
                    ex,
                )
            )

    if "amount" in df:
        zero = df["amount"] == 0
        if zero.any():
            issues.append(
                Issue(
                    "zero_amount",
                    "warning",
                    f"{int(zero.sum()):,} transactions have an amount of 0. They are kept, "
                    "but may be test or verification transactions.",
                    "amount",
                    int(zero.sum()),
                )
            )

    future = df["timestamp"] > pd.Timestamp(now)
    if future.any():
        issues.append(
            Issue(
                "future_timestamp",
                "warning",
                f"{int(future.sum()):,} transactions are dated in the future. They are kept; "
                "check the date column.",
                "timestamp",
                int(future.sum()),
                [str(v) for v in df.loc[future, "timestamp"].head(_MAX_EXAMPLES)],
            )
        )

    if "dob" in df:
        age_years = (df["timestamp"] - df["dob"]).dt.days / 365.25
        bad_dob = (
            df["dob"].notna() & df["timestamp"].notna() & ((age_years < 0) | (age_years > 120))
        )
        if bad_dob.any():
            issues.append(
                Issue(
                    "implausible_dob",
                    "warning",
                    f"{int(bad_dob.sum()):,} dates of birth give an age below 0 or above 120 at "
                    "the time of the transaction. These values are cleared.",
                    "dob",
                    int(bad_dob.sum()),
                    [str(v.date()) for v in df.loc[bad_dob, "dob"].head(_MAX_EXAMPLES)],
                )
            )
            df.loc[bad_dob, "dob"] = pd.NaT
    return issues


def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Issue]]:
    issues: list[Issue] = []
    exact_dup = df.duplicated(keep="first")
    if exact_dup.any():
        n = int(exact_dup.sum())
        issues.append(
            Issue(
                "duplicate_row",
                "warning",
                f"{n:,} rows are exact copies of an earlier row. The copies are removed.",
                None,
                n,
                [str(v) for v in df.loc[exact_dup, "transaction_id"].head(_MAX_EXAMPLES)],
            )
        )
    df = df[~exact_dup]

    dup_id = df["transaction_id"].notna() & df["transaction_id"].duplicated(keep="first")
    if dup_id.any():
        n = int(dup_id.sum())
        issues.append(
            Issue(
                "duplicate_id",
                "error",
                f"{n:,} rows reuse a transaction ID with different details. Only the first row "
                "for each ID is kept.",
                "transaction_id",
                n,
                [str(v) for v in df.loc[dup_id, "transaction_id"].head(_MAX_EXAMPLES)],
            )
        )
    df = df[~dup_id]

    usable = df[list(REQUIRED_FIELDS)].notna().all(axis=1)
    clean = df[usable].sort_values("timestamp", kind="stable").reset_index(drop=True)
    for col in clean.select_dtypes("category").columns:
        clean[col] = clean[col].cat.remove_unused_categories()
    return clean, issues


def _label_issues(clean: pd.DataFrame) -> list[Issue]:
    labels = clean[LABEL_FIELD]
    known = labels.dropna()
    issues = []
    if len(known) == 0:
        return [
            Issue(
                "no_labels",
                "warning",
                "The fraud label column has no usable values, so model accuracy cannot be "
                "measured on this dataset.",
                LABEL_FIELD,
            )
        ]
    fraud = int((known == 1).sum())
    if fraud == 0 or fraud == len(known):
        issues.append(
            Issue(
                "single_class",
                "warning",
                "All labelled rows belong to one class, so a model cannot be trained or "
                "evaluated on this dataset.",
                LABEL_FIELD,
            )
        )
    else:
        issues.append(
            Issue(
                "class_balance",
                "info",
                f"{fraud:,} of {len(known):,} labelled transactions are fraud "
                f"({fraud / len(known):.2%}). Fraud is rare, so accuracy alone would be "
                "misleading; precision and recall are used instead.",
                LABEL_FIELD,
                fraud,
            )
        )
    return issues


def _unused_column_issues(all_columns: list[str], used: set[str]) -> list[Issue]:
    unused = [c for c in all_columns if c not in used]
    if not unused:
        return []
    personal = [c for c in unused if c.lower() in PERSONAL_COLUMNS]
    msg = f"{len(unused)} columns are not used by FraudShield and were not loaded."
    if personal:
        msg += f" This includes personal data ({', '.join(personal)}), which is never stored."
    return [Issue("unused_columns", "info", msg, None, len(unused), unused[:10])]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def quality_score(rows_clean: int, rows_analysed: int) -> float:
    """Share of analysed rows that are usable, as a percentage.

    A row is usable when it is not a duplicate and all required fields are
    present and valid. Optional-field problems do not lower the score, because
    those rows can still be analysed.
    """
    return round(100.0 * rows_clean / rows_analysed, 2) if rows_analysed else 0.0


def _build_report(
    df: pd.DataFrame,
    clean: pd.DataFrame,
    fields: dict[str, str],
    all_columns: list[str],
    rows_in_file: int,
    sample_fraction: float,
    issues: list[Issue],
    tally: _Tally,
) -> DataReport:
    rows = len(df)
    profiles = []
    for name, src in fields.items():
        missing, _ = tally.get("missing", name)
        invalid = tally.get("invalid", name)[0] + tally.get("out_of_range", name)[0]
        col = clean[name]
        non_null = col.dropna()
        profiles.append(
            ColumnProfile(
                column=name,
                source_column=src,
                kind=FIELDS_BY_NAME[name].kind,
                dtype=str(col.dtype),
                missing=missing,
                missing_pct=round(100.0 * missing / rows, 2) if rows else 0.0,
                invalid=invalid,
                unique=int(non_null.nunique()),
                example=str(non_null.iloc[0]) if len(non_null) else "",
            )
        )

    has_label = LABEL_FIELD in clean and clean[LABEL_FIELD].notna().any()
    labels = clean[LABEL_FIELD].dropna() if has_label else None
    order = {"error": 0, "warning": 1, "info": 2}
    issues = sorted(issues, key=lambda i: order[i.severity])
    return DataReport(
        rows_in_file=rows_in_file,
        rows_analysed=rows,
        sample_fraction=sample_fraction,
        rows_clean=len(clean),
        quality_score=quality_score(len(clean), rows),
        mapping=dict(fields),
        unused_columns=[c for c in all_columns if c not in set(fields.values())],
        has_label=bool(has_label),
        fraud_rows=int((labels == 1).sum()) if has_label else None,
        legitimate_rows=int((labels == 0).sum()) if has_label else None,
        date_min=str(clean["timestamp"].min()) if len(clean) else None,
        date_max=str(clean["timestamp"].max()) if len(clean) else None,
        issues=issues,
        columns=profiles,
    )


def _missing_action(name: str, required: bool) -> str:
    if required:
        return "These rows cannot be used and are removed."
    if name == LABEL_FIELD:
        return "Kept, but unlabelled rows are left out of model training and evaluation."
    return "Kept as missing; handled during modelling."


def _kind_text(kind: str) -> str:
    return {
        "datetime": "a date and time",
        "date": "a date",
        "numeric": "a number",
        "label": "a fraud label (0/1, true/false, yes/no)",
    }.get(kind, "text")


def _range_text(low: float | None, high: float | None) -> str:
    if high is None or high == float("inf"):
        return f"must be {low:g} or more"
    return f"{low:g} to {high:g}"
