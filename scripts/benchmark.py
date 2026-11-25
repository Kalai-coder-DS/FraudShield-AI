"""Measure how fast and how memory-hungry each step is on this computer.

Usage (from the project folder, virtual environment active):

    python scripts/benchmark.py

Uses the bundled sample, the full test file if it is in data/raw, and the trained
model. Nothing is saved to the app's database (a temporary one is used).
Prints a table you can paste into docs/testing.md.
"""

from __future__ import annotations

import gc
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def peak_memory_mb() -> float:
    """Peak memory of this process so far, in MB (Windows, macOS and Linux)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.PeakWorkingSetSize / 1e6
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # bytes on macOS, KB on Linux


RESULTS: list[tuple[str, str, float, float]] = []


def measure(step: str, size: str, fn):
    gc.collect()
    start = time.perf_counter()
    value = fn()
    seconds = time.perf_counter() - start
    RESULTS.append((step, size, seconds, peak_memory_mb()))
    print(f"  {step:<42} {size:>16} {seconds:>8.2f} s")
    return value


def bench_file(label: str, path: Path, model, meta: dict, parts, db_path: Path) -> None:
    from fraudshield.data.loader import read_columns
    from fraudshield.data.schema import suggest_mapping
    from fraudshield.data.validation import process_dataset
    from fraudshield.eda import analysis as a
    from fraudshield.features.basic import add_basic_features
    from fraudshield.models import explain, scoring
    from fraudshield.reports.pdf_report import build_pdf
    from fraudshield.reports.summary import build_summary
    from fraudshield.storage import db

    print(f"\n{label}: {path.stat().st_size / 1e6:,.0f} MB")
    result = measure(
        "Read + validate + clean",
        label,
        lambda: process_dataset(path, suggest_mapping(read_columns(path))),
    )
    rows = f"{len(result.data):,} rows"
    dataset_id = measure(
        "Save to SQLite",
        rows,
        lambda: db.save_dataset(label, "benchmark", result.data, result.report, db_path=db_path),
    )
    measure(
        "Read one table page (50 rows)",
        rows,
        lambda: db.fetch_page(dataset_id, 100, 50, db_path=db_path),
    )
    stored = db.load_transactions(dataset_id, db_path=db_path)
    scores = measure(
        "Score all transactions",
        rows,
        lambda: scoring.score_frame(model, stored, meta["decision_threshold"]),
    )
    measure("Store scores", rows, lambda: db.save_scores(dataset_id, scores, "bench", db_path))

    def eda():
        df = add_basic_features(stored)
        return [
            a.overview(df),
            a.amount_histogram(df),
            a.weekly_activity(df),
            a.correlation_matrix(df),
            a.key_findings(df),
            a.amount_box_stats(df),
        ]

    measure("EDA summaries", rows, eda)
    background = explain.background_matrix(parts, stored)
    measure(
        "Explain one transaction",
        "1 row",
        lambda: explain.explain(parts, stored.iloc[[0]], background),
    )
    summary = measure(
        "Dashboard summary", rows, lambda: build_summary(stored, scores, meta, label, 100.0)
    )
    measure("Build PDF report", rows, lambda: build_pdf(summary))


def main() -> None:
    print("FraudShield AI benchmark")
    print(
        f"Python {platform.python_version()} on {platform.system()} {platform.release()}, "
        f"{os.cpu_count()} logical CPUs\n"
    )

    start = time.perf_counter()
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401

    from fraudshield import config
    from fraudshield.models import explain, registry

    RESULTS.append(("Import libraries", "-", time.perf_counter() - start, peak_memory_mb()))

    if not registry.metadata_exists():
        raise SystemExit("No trained model. Run: python scripts/train_models.py")
    model = measure("Load model (with fingerprint check)", "-", registry.load_model)
    meta = registry.load_metadata()
    parts = explain.model_parts(model)

    files = [
        ("sample", config.SAMPLE_DATASET_FILE),
        ("fraudTest.csv", config.RAW_DATA_DIR / "fraudTest.csv"),
    ]
    tmp = Path(tempfile.mkdtemp())
    for label, path in files:
        if path.is_file():
            bench_file(label, path, model, meta, parts, tmp / f"{label}.db")
        else:
            print(f"  (skipped {label}: {path} not found)")

    print("\n| Step | Data | Time (s) | Peak memory so far (MB) |")
    print("|---|---|---|---|")
    for step, size, seconds, mem in RESULTS:
        print(f"| {step} | {size} | {seconds:.2f} | {mem:,.0f} |")


if __name__ == "__main__":
    main()
