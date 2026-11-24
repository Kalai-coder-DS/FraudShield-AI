"""Module 2 - Dataset Management.

Flow: choose a source -> match columns -> check quality -> save as the active
dataset. Saved datasets can be inspected page by page, switched or deleted.
All checks are done by fraudshield.data.validation; this page only displays them.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from fraudshield.config import RAW_DATA_DIR, SAMPLE_DATASET_FILE, SAMPLE_FRACTIONS, SETTINGS
from fraudshield.data.loader import DatasetLoadError, detect_format, read_columns
from fraudshield.data.schema import FIELDS, check_mapping, suggest_mapping
from fraudshield.data.validation import DataReport, DatasetValidationError, process_dataset
from fraudshield.storage import db
from fraudshield.ui.components import (
    escape_markdown,
    mask_card,
    render_empty_state,
    render_header,
    render_kpi,
)

NOT_AVAILABLE = "— not in this file —"
SEVERITY_LABEL = {"error": "🔴 Error", "warning": "🟠 Warning", "info": "🔵 Info"}
SOURCES = ("Upload a CSV file", "Bundled sample dataset", "Large file in data/raw folder")

render_header(
    "Dataset Management",
    "Load transaction data and check its quality. Every later module works on the "
    "active dataset chosen here.",
    eyebrow="Module 2 · Data",
)

flash = st.session_state.pop("dataset_flash", None)
if flash:
    st.success(escape_markdown(flash), icon=":material/check_circle:")


# ---------------------------------------------------------------------------
# Shared display helpers
# ---------------------------------------------------------------------------


def show_report(report: DataReport, preview: pd.DataFrame | None) -> None:
    """Display a data-quality report. Every number comes from ``report``."""
    kpis = [
        ("Rows in file", f"{report.rows_in_file:,}", "Data rows found in the CSV file."),
        (
            "Usable rows",
            f"{report.rows_clean:,}",
            "Rows kept after removing duplicates and rows with invalid required values.",
        ),
        (
            "Quality score",
            f"{report.quality_score:.1f}%",
            "Share of analysed rows that are usable.",
        ),
    ]
    if report.fraud_rate is not None:
        kpis.append(
            (
                "Known fraud rate",
                f"{report.fraud_rate:.2%}",
                f"{report.fraud_rows:,} labelled fraud transactions.",
            )
        )
    else:
        kpis.append(("Fraud labels", "None", "Without labels, accuracy cannot be measured."))
    for column, (label, value, caption) in zip(st.columns(len(kpis)), kpis, strict=True):
        with column:
            render_kpi(label, value, caption)

    notes = []
    if report.sample_fraction < 1:
        notes.append(
            f"A random {report.sample_fraction:.0%} of rows was analysed "
            f"({report.rows_analysed:,} rows)."
        )
    if report.date_min:
        notes.append(f"Transactions from {report.date_min[:10]} to {report.date_max[:10]}.")
    notes.append(f"{len(report.mapping)} columns used.")
    st.caption(" ".join(notes))

    st.markdown("##### Data quality issues")
    if not report.issues:
        st.success("No problems found.", icon=":material/verified:")
    else:
        st.dataframe(
            pd.DataFrame(
                {
                    "Severity": [SEVERITY_LABEL[i.severity] for i in report.issues],
                    "What was found and what was done": [i.message for i in report.issues],
                    "Column": [i.column or "—" for i in report.issues],
                    "Examples": [", ".join(i.examples) for i in report.issues],
                }
            ),
            hide_index=True,
        )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Columns")
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": [c.column for c in report.columns],
                    "From file column": [c.source_column for c in report.columns],
                    "Type": [c.dtype for c in report.columns],
                    "Missing": [c.missing for c in report.columns],
                    "Invalid": [c.invalid for c in report.columns],
                    "Distinct values": [c.unique for c in report.columns],
                }
            ),
            hide_index=True,
        )
    with right:
        st.markdown("##### Missing values by column")
        missing = pd.DataFrame(
            {
                "column": [c.column for c in report.columns if c.missing],
                "% missing": [c.missing_pct for c in report.columns if c.missing],
            }
        )
        if missing.empty:
            st.success("No column has missing values.", icon=":material/verified:")
        else:
            st.bar_chart(missing, x="column", y="% missing", color="#E63946")
            st.caption("Percentage of analysed rows with no value in each column.")

    if preview is not None:
        st.markdown(f"##### Preview (first {len(preview)} usable rows)")
        shown = preview.copy()
        if "card_id" in shown:
            shown["card_id"] = shown["card_id"].astype(object).map(mask_card)
        st.dataframe(shown, hide_index=True)
        st.caption("Card numbers are masked; only the last 4 digits are shown.")


# ---------------------------------------------------------------------------
# Add a dataset
# ---------------------------------------------------------------------------


def choose_source() -> tuple[bytes | Path, str, str] | None:
    """Return (data source, description, suggested name), or None if nothing chosen."""
    choice = st.radio("Where is the data?", SOURCES, horizontal=True)

    if choice == SOURCES[0]:
        uploaded = st.file_uploader(
            "CSV file of transactions",
            type=["csv"],
            help=f"Up to {SETTINGS.max_upload_mb} MB. For bigger files use the data/raw option.",
        )
        if uploaded is None:
            return None
        return uploaded.getvalue(), f"Upload: {uploaded.name}", Path(uploaded.name).stem

    if choice == SOURCES[1]:
        if not SAMPLE_DATASET_FILE.is_file():
            render_empty_state(
                "Sample dataset not created yet",
                "Download the Sparkov dataset from Kaggle, put fraudTest.csv in data/raw/, "
                "then run:  python scripts/make_sample.py  (see docs/dataset.md).",
            )
            return None
        st.caption(
            "A stratified sample of the real Sparkov test data, built by scripts/make_sample.py."
        )
        return SAMPLE_DATASET_FILE, "Bundled sample (Sparkov)", "Sparkov sample"

    files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not files:
        render_empty_state(
            "No CSV files in data/raw",
            f"Copy large CSV files into {RAW_DATA_DIR} to load them without uploading.",
        )
        return None
    path = st.selectbox(
        "File",
        files,
        format_func=lambda p: f"{p.name}  ({p.stat().st_size / 1e6:,.0f} MB)",
    )
    return path, f"File: data/raw/{path.name}", path.stem


def mapping_form(columns: list[str], key: str, size_mb: float):
    suggested = suggest_mapping(columns)
    with st.form(f"mapping-{key}"):
        st.markdown("##### Match your columns")
        st.caption(
            "FraudShield uses standard column names. Columns were matched automatically; "
            "check them and fix any that are wrong. Fields marked * are required."
        )
        options = [NOT_AVAILABLE, *columns]
        mapping = {}
        grid = st.columns(3)
        for i, field in enumerate(FIELDS):
            default = suggested.get(field.name)
            with grid[i % 3]:
                choice = st.selectbox(
                    f"{field.name}{' *' if field.required else ''}",
                    options,
                    index=options.index(default) if default else 0,
                    help=field.description,
                    key=f"map-{key}-{field.name}",
                )
            mapping[field.name] = None if choice == NOT_AVAILABLE else choice

        fractions = sorted(SAMPLE_FRACTIONS)
        fraction = st.select_slider(
            "Share of rows to load",
            options=fractions,
            value=1.0 if size_mb < 100 else 0.25,
            format_func=lambda f: f"{f:.0%}",
            help="Loading a random share of a large file saves memory and time. "
            "The fraud rate stays the same on average.",
        )
        submitted = st.form_submit_button("Check data quality", type="primary")
    return submitted, mapping, fraction


def add_dataset() -> None:
    chosen = choose_source()
    if chosen is None:
        return
    source, description, default_name = chosen
    size_mb = (len(source) if isinstance(source, bytes) else source.stat().st_size) / 1e6
    key = hashlib.md5(f"{description}-{size_mb}".encode()).hexdigest()[:10]

    try:
        columns = read_columns(source, detect_format(source))
    except DatasetLoadError as err:
        st.error(escape_markdown(str(err)), icon=":material/error:")
        return
    st.caption(f"{len(columns)} columns found · {size_mb:,.1f} MB")

    submitted, mapping, fraction = mapping_form(columns, key, size_mb)
    if submitted:
        problems = check_mapping(mapping)
        if problems:
            for problem in problems:
                st.error(escape_markdown(problem), icon=":material/error:")
            return
        try:
            with st.spinner("Reading and checking the data..."):
                result = process_dataset(source, mapping, sample_fraction=fraction)
        except (DatasetLoadError, DatasetValidationError) as err:
            st.error(escape_markdown(str(err)), icon=":material/error:")
            return
        st.session_state["pending_dataset"] = {
            "key": key,
            "result": result,
            "source": description,
            "name": default_name,
        }

    pending = st.session_state.get("pending_dataset")
    if not pending or pending["key"] != key:
        return

    result = pending["result"]
    st.divider()
    show_report(result.report, result.data.head(SETTINGS.preview_rows))
    st.divider()
    if result.report.rows_clean == 0:
        st.error("No usable rows remain, so this dataset cannot be saved.", icon=":material/error:")
        return
    with st.form(f"save-{key}"):
        name = st.text_input("Dataset name", value=pending["name"], max_chars=80)
        save = st.form_submit_button("Save and use this dataset", type="primary")
    if save:
        try:
            with st.spinner(f"Saving {result.report.rows_clean:,} rows..."):
                db.save_dataset(name, pending["source"], result.data, result.report)
        except ValueError as err:
            st.error(escape_markdown(str(err)), icon=":material/error:")
            return
        del st.session_state["pending_dataset"]
        st.session_state["dataset_flash"] = (
            f"Saved '{name.strip()}' and made it the active dataset."
        )
        st.rerun()


# ---------------------------------------------------------------------------
# Saved datasets
# ---------------------------------------------------------------------------


@st.dialog("Delete dataset?")
def confirm_delete(info: db.DatasetInfo) -> None:
    st.markdown(
        f"This permanently deletes **{escape_markdown(info.name)}** and its "
        f"{info.rows_clean:,} stored transactions. This cannot be undone."
    )
    cancel, delete = st.columns(2)
    if cancel.button("Cancel"):
        st.rerun()
    if delete.button("Delete permanently", type="primary"):
        db.delete_dataset(info.id)
        st.session_state["dataset_flash"] = f"Deleted '{info.name}'."
        st.rerun()


def saved_datasets(saved: list[db.DatasetInfo]) -> None:
    if not saved:
        render_empty_state(
            "No saved datasets",
            "Add a dataset in the 'Add dataset' tab. It will appear here after saving.",
        )
        return

    st.dataframe(
        pd.DataFrame(
            {
                "Active": ["✓" if d.is_active else "" for d in saved],
                "Name": [d.name for d in saved],
                "Usable rows": [d.rows_clean for d in saved],
                "Fraud labels": ["Yes" if d.has_label else "No" for d in saved],
                "Quality score": [f"{d.quality_score:.1f}%" for d in saved],
                "Source": [d.source for d in saved],
                "Added": [d.created_at.replace("T", " ") for d in saved],
            }
        ),
        hide_index=True,
    )

    info = st.selectbox(
        "Select a dataset",
        saved,
        format_func=lambda d: f"{d.name}  (#{d.id}){'  · active' if d.is_active else ''}",
    )
    use, delete, _ = st.columns([1, 1, 3])
    if use.button("Use this dataset", disabled=info.is_active):
        db.set_active(info.id)
        st.session_state["dataset_flash"] = f"'{info.name}' is now the active dataset."
        st.rerun()
    if delete.button("Delete"):
        confirm_delete(info)

    st.divider()
    show_report(db.load_report(info.id), preview=None)

    st.markdown("##### Stored transactions")
    page_size = SETTINGS.table_page_size
    pages = max(1, math.ceil(info.rows_clean / page_size))
    page = st.number_input(f"Page (of {pages:,})", 1, pages, 1, key=f"page-{info.id}")
    rows = db.fetch_page(info.id, int(page), page_size)
    if "card_id" in rows:
        rows["card_id"] = rows["card_id"].astype(object).map(mask_card)
    st.dataframe(rows, hide_index=True)
    first = (int(page) - 1) * page_size + 1
    st.caption(
        f"Rows {first:,}–{first + len(rows) - 1:,} of {info.rows_clean:,}. "
        "Only this page is read from the database."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

saved = db.list_datasets()
active = next((d for d in saved if d.is_active), None)
if active:
    st.info(
        f"Active dataset: {escape_markdown(active.name)} · {active.rows_clean:,} usable rows · "
        f"quality score {active.quality_score:.1f}%",
        icon=":material/database:",
    )
else:
    st.info("No active dataset yet. Add one below to get started.", icon=":material/info:")

add_tab, saved_tab = st.tabs(["Add dataset", f"Saved datasets ({len(saved)})"])
with add_tab:
    add_dataset()
with saved_tab:
    saved_datasets(saved)
