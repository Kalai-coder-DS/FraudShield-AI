"""Central configuration for FraudShield AI.

Every path, random seed and business threshold lives here, so that nothing is
hard-coded inside the UI or the ML code. Changing a value in this file changes
it everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# config.py lives at <root>/src/fraudshield/config.py, so the root is 3 levels up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"  # small, committed sample of the real dataset
RAW_DATA_DIR: Path = DATA_DIR / "raw"  # full dataset downloaded by the user (git-ignored)
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"  # caches and SQLite database (git-ignored)
MODELS_DIR: Path = PROJECT_ROOT / "models"  # saved models + metadata
DATABASE_PATH: Path = PROCESSED_DATA_DIR / "fraudshield.db"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskThresholds:
    """Probability cut-offs that turn a fraud probability into a risk tier.

    A transaction's tier is decided by the highest cut-off it reaches:

        probability <  medium            -> Low
        medium   <= probability < high    -> Medium
        high     <= probability < critical-> High
        critical <= probability           -> Critical

    These defaults are a documented starting point. Phase 4 reports precision
    and recall at each cut-off so the final values can be justified with data.
    """

    medium: float = 0.30
    high: float = 0.60
    critical: float = 0.85

    def __post_init__(self) -> None:
        values = (self.medium, self.high, self.critical)
        if not all(0.0 < v < 1.0 for v in values):
            raise ValueError(f"Risk thresholds must be between 0 and 1 (exclusive): {values}")
        if not self.medium < self.high < self.critical:
            raise ValueError(
                "Risk thresholds must increase: medium < high < critical, "
                f"got {self.medium}, {self.high}, {self.critical}"
            )


RISK_TIERS: tuple[str, ...] = ("Low", "Medium", "High", "Critical")


@dataclass(frozen=True)
class AppSettings:
    """Settings that affect performance and the user interface."""

    table_page_size: int = 50  # rows shown per page in transaction tables
    preview_rows: int = 20  # rows shown in dataset preview
    max_upload_mb: int = 200  # must match .streamlit/config.toml
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)


SETTINGS = AppSettings()
