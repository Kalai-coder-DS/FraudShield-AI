"""Train, compare and save the fraud models.

Usage (from the project folder, with the virtual environment active):

    python scripts/train_models.py                        # full training file
    python scripts/train_models.py --train-fraction 0.3   # 30% of rows: faster, less memory

Inputs:  data/raw/fraudTrain.csv (earlier dates) and data/raw/fraudTest.csv (later dates)
Outputs: models/fraudshield_model.joblib (git-ignored) and models/metadata.json

Both files are read and validated with the same code as the app's Dataset
Management page, so the model is trained on exactly the data the app accepts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudshield.config import MODELS_DIR, RAW_DATA_DIR  # noqa: E402
from fraudshield.data.loader import read_columns  # noqa: E402
from fraudshield.data.schema import suggest_mapping  # noqa: E402
from fraudshield.data.validation import process_dataset  # noqa: E402
from fraudshield.models import registry  # noqa: E402
from fraudshield.models.training import train_and_evaluate  # noqa: E402


def load(path: Path, fraction: float):
    if not path.is_file():
        raise SystemExit(
            f"{path} not found. Put fraudTrain.csv and fraudTest.csv from the Kaggle "
            "'Credit Card Transactions Fraud Detection Dataset' in data/raw/ (see docs/dataset.md)."
        )
    print(f"Reading {path.name} ({path.stat().st_size / 1e6:,.0f} MB)...")
    result = process_dataset(path, suggest_mapping(read_columns(path)), sample_fraction=fraction)
    r = result.report
    print(f"  {r.rows_clean:,} usable rows, {r.fraud_rows:,} fraud, quality {r.quality_score:.1f}%")
    return result.data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare FraudShield models.")
    parser.add_argument("--train", type=Path, default=RAW_DATA_DIR / "fraudTrain.csv")
    parser.add_argument("--test", type=Path, default=RAW_DATA_DIR / "fraudTest.csv")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="share of training rows to use (lower it on a laptop with little RAM)",
    )
    parser.add_argument("--rf-trees", type=int, default=150, help="trees in the Random Forest")
    parser.add_argument("--jobs", type=int, default=-1, help="CPU cores to use (-1 = all)")
    parser.add_argument("--output", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    start = time.perf_counter()
    train = load(args.train, args.train_fraction)
    test = load(args.test, 1.0)  # the test set is always used in full

    result = train_and_evaluate(train, test, n_jobs=args.jobs, rf_trees=args.rf_trees)
    metadata = registry.save(
        result,
        args.output,
        data_files={
            "train": args.train.name,
            "test": args.test.name,
            "train_fraction": args.train_fraction,
        },
    )

    print("\nResults on the TEST period (never used for any training decision):")
    print(f"{'Model':<22}{'PR-AUC':>8}{'ROC-AUC':>9}{'Precision':>11}{'Recall':>8}{'F1':>7}")
    for name, m in metadata["models"].items():
        t = m["test"]
        p, rc, f1 = (f"{t[k]:.3f}" if k in t else "  -  " for k in ("precision", "recall", "f1"))
        flag = "  <- selected" if name == metadata["selected_model"] else ""
        print(f"{name:<22}{t['pr_auc']:>8.3f}{t['roc_auc']:>9.3f}{p:>11}{rc:>8}{f1:>7}{flag}")
    best = metadata["models"][metadata["selected_model"]]["test"]
    print(
        f"\nSelected: {metadata['selected_model']} (highest validation PR-AUC). "
        f"At threshold {metadata['decision_threshold']:.2f} it catches "
        f"{best['true_positives']:,} of {best['true_positives'] + best['false_negatives']:,} "
        f"fraud cases with {best['false_positives']:,} false alarms."
    )
    seconds = time.perf_counter() - start
    print(f"Saved to {args.output} ({metadata['model_size_mb']} MB) in {seconds:.0f}s")


if __name__ == "__main__":
    main()
