"""Colour system and global CSS for the FraudShield AI interface.

All colours are defined once here and reused by the CSS, the Plotly charts
(Phase 3+) and the PDF report (Phase 6), so the product looks consistent.
"""

from __future__ import annotations

# Base surfaces: neutral graphite, so colour is kept for meaning
BACKGROUND = "#111318"
SIDEBAR = "#171A21"
CARD = "#1E222B"
CARD_BORDER = "#2B313C"

# Brand accent: teal (trust, security). Deep enough for white button text.
PRIMARY = "#0D9488"
PRIMARY_SOFT = "#0F766E"
SECONDARY = "#F5A524"  # amber: labels and warnings
AMBER = "#F5A524"

# Red is reserved for fraud and critical risk, so it always means "danger".
FRAUD = "#E5484D"

# Text
TEXT = "#E8EAED"  # soft white
TEXT_MUTED = "#9AA3AE"  # muted grey

# Risk tiers go from green to bright red; the brightest red always means Critical.
RISK_COLORS: dict[str, str] = {
    "Low": "#4ADE80",
    "Medium": AMBER,
    "High": "#FB923C",
    "Critical": "#FF4D6D",
}

# Ordered palette for categorical charts
CHART_SEQUENCE: list[str] = [PRIMARY, SECONDARY, "#A78BFA", "#8D99AE", "#4ADE80", FRAUD]

GLOBAL_CSS = f"""
<style>
:root {{
    --fs-bg: {BACKGROUND};
    --fs-sidebar: {SIDEBAR};
    --fs-card: {CARD};
    --fs-border: {CARD_BORDER};
    --fs-primary: {PRIMARY};
    --fs-secondary: {SECONDARY};
    --fs-amber: {AMBER};
    --fs-text: {TEXT};
    --fs-muted: {TEXT_MUTED};
}}

section[data-testid="stSidebar"] {{
    background-color: var(--fs-sidebar);
    border-right: 1px solid var(--fs-border);
}}

.block-container {{ padding-top: 4.5rem; max-width: 1200px; }}  /* clears the Streamlit top bar */

/* Page header */
.fs-header h1 {{ font-size: 1.9rem; font-weight: 700; margin: 0 0 .25rem 0;
                 color: var(--fs-text); }}
.fs-header p  {{ color: var(--fs-muted); margin: 0 0 1.25rem 0; font-size: 1rem; }}
.fs-eyebrow  {{ color: var(--fs-secondary); font-size: .75rem; font-weight: 600;
                letter-spacing: .08em; text-transform: uppercase; }}

/* KPI card */
.fs-kpi {{
    background: var(--fs-card);
    border: 1px solid var(--fs-border);
    border-left: 3px solid var(--fs-primary);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    height: 100%;
}}
.fs-kpi-label   {{ color: var(--fs-muted); font-size: .82rem; font-weight: 500; }}
.fs-kpi-value   {{ color: var(--fs-text); font-size: 1.7rem; font-weight: 700; margin: .2rem 0; }}
.fs-kpi-caption {{ color: var(--fs-muted); font-size: .78rem; line-height: 1.35; }}

/* Generic card */
.fs-card {{
    background: var(--fs-card);
    border: 1px solid var(--fs-border);
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
}}
.fs-card h3 {{ font-size: 1.05rem; margin: 0 0 .4rem 0; color: var(--fs-text); }}
.fs-card p, .fs-card li {{ color: var(--fs-muted); font-size: .92rem; }}

/* Empty state */
.fs-empty {{
    background: var(--fs-card);
    border: 1px dashed var(--fs-border);
    border-radius: 10px;
    padding: 2rem 1.5rem;
    text-align: center;
}}
.fs-empty h3 {{ color: var(--fs-text); margin-bottom: .4rem; font-size: 1.1rem; }}
.fs-empty p  {{ color: var(--fs-muted); margin: 0 auto; max-width: 560px; }}

/* Status badge */
.fs-badge {{
    display: inline-block; padding: .1rem .55rem; border-radius: 999px;
    font-size: .72rem; font-weight: 600;
}}
.fs-badge-ready   {{ background: rgba(76, 201, 160, .15); color: #4CC9A0; }}
.fs-badge-planned {{ background: rgba(245, 165, 36, .15); color: {AMBER}; }}
</style>
"""
