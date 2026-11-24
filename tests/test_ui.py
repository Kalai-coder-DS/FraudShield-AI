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


# --- Theme consistency and readability ----------------------------------------


def test_streamlit_config_matches_theme():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    with open(PROJECT_ROOT / ".streamlit" / "config.toml", "rb") as f:
        cfg = tomllib.load(f)["theme"]
    assert cfg["primaryColor"].upper() == theme.PRIMARY.upper()
    assert cfg["backgroundColor"].upper() == theme.BACKGROUND.upper()
    assert cfg["secondaryBackgroundColor"].upper() == theme.SIDEBAR.upper()
    assert cfg["textColor"].upper() == theme.TEXT.upper()


def _contrast(a: str, b: str) -> float:
    """WCAG 2 contrast ratio between two hex colours."""

    def luminance(hex_colour: str) -> float:
        rgb = [int(hex_colour.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_text_is_readable_on_cards():
    # WCAG AA: 4.5:1 for normal text
    assert _contrast(theme.TEXT, theme.CARD) >= 4.5
    assert _contrast(theme.TEXT_MUTED, theme.CARD) >= 4.5
    assert _contrast(theme.TEXT_MUTED, theme.SIDEBAR) >= 4.5


def test_risk_colours_are_visible_on_cards():
    # WCAG AA: 3:1 for large text and graphics such as chart bars and badges
    for tier, colour in theme.RISK_COLORS.items():
        assert _contrast(colour, theme.CARD) >= 3, tier


def test_mask_card_shows_only_last_four_digits():
    assert components.mask_card("2703186189652095") == "•••• 2095"
    assert components.mask_card(None) == ""
    assert components.mask_card("12") == "••••"


def test_escape_markdown_neutralises_formatting():
    assert components.escape_markdown("**bold** [x](y)") == r"\*\*bold\*\* \[x\]\(y\)"
