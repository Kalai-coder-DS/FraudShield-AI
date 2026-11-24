"""Build the bundled sample dataset from the real Sparkov data.

Usage (from the project folder):

    python scripts/make_sample.py                       # default: data/raw/fraudTest.csv
    python scripts/make_sample.py --input data/raw/fraudTest.csv --rows 50000

Why the TEST file: the model is trained on fraudTrain.csv (earlier dates) in
Phase 4. Building the sample from fraudTest.csv (later dates) means the sample
the app scores was never seen during training, so results shown in the app are
honest out-of-sample results.

How: the file is read in chunks (never fully in memory). From every chunk the
same share of fraud rows and of legitimate rows is kept (stratified sampling),
so the sample has the same fraud rate as the full file. Personal columns
(first name, last name, street) are removed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudshield.config import RANDOM_SEED, RAW_DATA_DIR, SAMPLE_DATASET_FILE  # noqa: E402
from fraudshield.data.schema import PERSONAL_COLUMNS  # noqa: E402

DROP_COLUMNS = set(PERSONAL_COLUMNS) | {"Unnamed: 0"}
LABEL = "is_fraud"


def count_rows(path: Path) -> int:
    with path.open("rb") as f:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b"")) - 1


def make_sample(
    input_path: Path,
    output_path: Path,
    rows: int,
    seed: int = RANDOM_SEED,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"{input_path} not found. Download 'Credit Card Transactions Fraud Detection "
            "Dataset' from Kaggle and put fraudTest.csv in data/raw/ (see docs/dataset.md)."
        )
    columns = pd.read_csv(input_path, nrows=0).columns
    if LABEL not in columns:
        raise ValueError(f"{input_path.name} has no '{LABEL}' column; is it the Sparkov dataset?")
    usecols = [c for c in columns if c not in DROP_COLUMNS]

    total = count_rows(input_path)
    fraction = min(1.0, rows / total)
    parts = []
    for i, chunk in enumerate(
        pd.read_csv(input_path, usecols=usecols, dtype=str, chunksize=chunksize)
    ):
        parts.append(
            chunk.groupby(LABEL, group_keys=False).sample(frac=fraction, random_state=seed + i)
        )
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sort_values("trans_date_trans_time", kind="stable").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR / "fraudTest.csv")
    parser.add_argument("--output", type=Path, default=SAMPLE_DATASET_FILE)
    parser.add_argument("--rows", type=int, default=50_000, help="approximate sample size")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    sample = make_sample(args.input, args.output, args.rows, args.seed)
    fraud = (sample[LABEL] == "1").sum()
    size_mb = args.output.stat().st_size / 1e6
    print(f"Saved {len(sample):,} rows ({fraud:,} fraud, {fraud / len(sample):.2%}) ")
    print(f"to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
