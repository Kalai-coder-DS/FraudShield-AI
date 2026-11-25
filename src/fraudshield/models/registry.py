"""Saving and loading the trained model.

Two files are written to ``models/``:

* ``fraudshield_model.joblib`` - the selected, calibrated pipeline. The app
  loads it instead of retraining, so start-up stays fast.
* ``metadata.json`` - everything needed to explain the model: when and on what
  data it was trained, all models' metrics, curves, the decision threshold,
  risk tiers and feature importance. The Model Performance page reads only
  this file.

Security: joblib files can run code when loaded, so only files this project
produced should be loaded. The SHA-256 fingerprint of the model file is
stored in the metadata and checked before loading; a changed or swapped file
is refused.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import sklearn

from fraudshield import __version__, config
from fraudshield.config import SETTINGS
from fraudshield.features.engineering import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    NUMERIC_FEATURES,
)
from fraudshield.models.training import TrainingResult

MODEL_FILE = "fraudshield_model.joblib"
METADATA_FILE = "metadata.json"


class ModelNotFoundError(FileNotFoundError):
    pass


class ModelIntegrityError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dir(models_dir: Path | None) -> Path:
    """The models folder: read from config at call time so tests can redirect it."""
    return Path(models_dir or config.MODELS_DIR)


def save(
    result: TrainingResult, models_dir: Path | None = None, data_files: dict | None = None
) -> dict:
    """Save the selected model and the metadata of all models. Returns the metadata."""
    models_dir = _dir(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    selected = result.results[result.selected]
    model_path = models_dir / MODEL_FILE
    joblib.dump(selected.model, model_path, compress=3)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fraudshield_version": __version__,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "selected_model": result.selected,
        "selection_rule": "Highest PR-AUC on the validation period",
        "decision_threshold": selected.test["threshold"],
        "risk_thresholds": asdict(SETTINGS.risk_thresholds),
        "features": {"numeric": list(NUMERIC_FEATURES), "categorical": list(CATEGORICAL_FEATURES)},
        "excluded_features": EXCLUDED_FEATURES,
        "data": {**result.data_info, "files": data_files or {}},
        "models": {
            name: {k: v for k, v in asdict(r).items() if k != "model"}
            for name, r in result.results.items()
        },
        "tiers": result.tiers,
        "feature_importance": result.importance,
        "model_file": MODEL_FILE,
        "model_sha256": file_sha256(model_path),
        "model_size_mb": round(model_path.stat().st_size / 1e6, 2),
    }
    (models_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def metadata_path(models_dir: Path | None = None) -> Path:
    return _dir(models_dir) / METADATA_FILE


def metadata_exists(models_dir: Path | None = None) -> bool:
    return metadata_path(models_dir).is_file()


def load_metadata(models_dir: Path | None = None) -> dict:
    path = metadata_path(models_dir)
    if not path.is_file():
        raise ModelNotFoundError("No trained model yet. Run: python scripts/train_models.py")
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(models_dir: Path | None = None):
    """Load the saved model after checking its fingerprint."""
    metadata = load_metadata(models_dir)
    path = _dir(models_dir) / metadata["model_file"]
    if not path.is_file():
        raise ModelNotFoundError(
            f"{path.name} is missing. Model files are not stored in Git; "
            "run: python scripts/train_models.py"
        )
    if file_sha256(path) != metadata["model_sha256"]:
        raise ModelIntegrityError(
            "The model file does not match its recorded fingerprint, so it was not loaded. "
            "Retrain with: python scripts/train_models.py"
        )
    return joblib.load(path)
