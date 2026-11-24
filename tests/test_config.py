import pytest

from fraudshield import config
from fraudshield.config import RiskThresholds


def test_project_root_contains_app():
    assert (config.PROJECT_ROOT / "app.py").exists()


def test_data_paths_are_inside_project():
    for path in (config.SAMPLE_DATA_DIR, config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR):
        assert config.PROJECT_ROOT in path.parents


def test_default_thresholds_are_valid_and_ordered():
    t = config.SETTINGS.risk_thresholds
    assert 0 < t.medium < t.high < t.critical < 1


@pytest.mark.parametrize(
    "medium, high, critical",
    [
        (0.6, 0.3, 0.85),  # not increasing
        (0.3, 0.3, 0.85),  # equal values
        (0.0, 0.5, 0.9),  # boundary 0 not allowed
        (0.3, 0.6, 1.0),  # boundary 1 not allowed
    ],
)
def test_invalid_thresholds_are_rejected(medium, high, critical):
    with pytest.raises(ValueError):
        RiskThresholds(medium=medium, high=high, critical=critical)


def test_four_risk_tiers():
    assert config.RISK_TIERS == ("Low", "Medium", "High", "Critical")
