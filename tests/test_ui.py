from pathlib import Path

import pytest

from fraudshield.config import PROJECT_ROOT, RISK_TIERS
from fraudshield.ui import components, theme
from fraudshield.ui.modules import MODULES, get_module


def test_kpi_html_escapes_user_text():
    html = components.build_kpi_html("<script>x</script>", "1,000", "a & b")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "a &amp; b" in html


def test_header_and_empty_state_escape_text():
    assert "<b>" not in components.build_header_html("<b>t</b>", "d")
    assert "<img" not in components.build_empty_state_html("t", "<img src=x>")


def test_module_keys_are_unique():
    keys = [m.key for m in MODULES]
    assert len(keys) == len(set(keys))


def test_every_module_has_a_view_file():
    for module in MODULES:
        assert (PROJECT_ROOT / "views" / f"{module.key}.py").exists(), module.key


def test_module_phases_are_valid():
    assert all(1 <= m.phase <= 8 for m in MODULES)


def test_get_module_unknown_key_raises():
    with pytest.raises(KeyError):
        get_module("nope")


def test_every_risk_tier_has_a_colour():
    assert set(theme.RISK_COLORS) == set(RISK_TIERS)


def test_all_python_files_compile():
    for path in list(PROJECT_ROOT.glob("views/*.py")) + [PROJECT_ROOT / "app.py"]:
        compile(Path(path).read_text(encoding="utf-8"), str(path), "exec")


# --- Streamlit smoke tests (run where Streamlit is installed) ----------------


@pytest.mark.parametrize("view", ["home"] + [m.key for m in MODULES])
def test_each_view_renders_without_errors(view):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(PROJECT_ROOT / "views" / f"{view}.py"))
    at.run(timeout=30)
    assert not at.exception, at.exception
