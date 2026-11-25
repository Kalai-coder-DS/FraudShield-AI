"""The bundled sample dataset: check it and save it as the active dataset."""

from __future__ import annotations

from fraudshield import config
from fraudshield.data.loader import read_columns
from fraudshield.data.schema import suggest_mapping
from fraudshield.data.validation import process_dataset
from fraudshield.storage import db


def sample_available() -> bool:
    return config.SAMPLE_DATASET_FILE.is_file()


def load_sample() -> int:
    """Validate the sample with automatic column matching, save it and make it active."""
    path = config.SAMPLE_DATASET_FILE
    result = process_dataset(path, suggest_mapping(read_columns(path)))
    return db.save_dataset("Sparkov sample", "Bundled sample (Sparkov)", result.data, result.report)
