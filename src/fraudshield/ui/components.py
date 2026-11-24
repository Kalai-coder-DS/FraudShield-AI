"""Reusable UI building blocks.

Each component has two parts:
  * ``build_*_html`` - a pure function that returns HTML. It escapes all text,
    so user-supplied values (e.g. column names from an uploaded CSV) can never
    inject HTML or scripts. These are unit-tested without Streamlit.
  * ``render_*`` - a thin wrapper that displays the HTML in Streamlit.
"""

from __future__ import annotations

from html import escape


def build_header_html(title: str, description: str, eyebrow: str = "") -> str:
    eyebrow_html = f'<div class="fs-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    return (
        f'<div class="fs-header">{eyebrow_html}'
        f"<h1>{escape(title)}</h1><p>{escape(description)}</p></div>"
    )


def build_kpi_html(label: str, value: str, caption: str = "") -> str:
    caption_html = f'<div class="fs-kpi-caption">{escape(caption)}</div>' if caption else ""
    return (
        f'<div class="fs-kpi"><div class="fs-kpi-label">{escape(label)}</div>'
        f'<div class="fs-kpi-value">{escape(value)}</div>{caption_html}</div>'
    )


def build_empty_state_html(title: str, message: str) -> str:
    return f'<div class="fs-empty"><h3>{escape(title)}</h3><p>{escape(message)}</p></div>'


def build_badge_html(text: str, ready: bool) -> str:
    css_class = "fs-badge-ready" if ready else "fs-badge-planned"
    return f'<span class="fs-badge {css_class}">{escape(text)}</span>'


# --- Streamlit wrappers -----------------------------------------------------
# Streamlit is imported inside the functions so the pure builders above can be
# imported and tested without Streamlit installed.


def render_header(title: str, description: str, eyebrow: str = "") -> None:
    import streamlit as st

    st.markdown(build_header_html(title, description, eyebrow), unsafe_allow_html=True)


def render_kpi(label: str, value: str, caption: str = "") -> None:
    import streamlit as st

    st.markdown(build_kpi_html(label, value, caption), unsafe_allow_html=True)


def render_empty_state(title: str, message: str) -> None:
    import streamlit as st

    st.markdown(build_empty_state_html(title, message), unsafe_allow_html=True)


# --- Text helpers -------------------------------------------------------------

_MARKDOWN_SPECIAL = set("\\`*_{}[]()#+-.!|<>~$:")


def escape_markdown(text: str) -> str:
    """Escape user text so Streamlit markdown shows it literally."""
    return "".join(f"\\{c}" if c in _MARKDOWN_SPECIAL else c for c in str(text))


def mask_card(value: object) -> str:
    """Show only the last 4 digits of a card number: '•••• 2095'."""
    text = "" if value is None else str(value)
    if text in ("", "nan", "<NA>", "None"):
        return ""
    return f"•••• {text[-4:]}" if len(text) > 4 else "••••"
