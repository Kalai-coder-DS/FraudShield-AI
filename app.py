"""FraudShield AI - application entry point.

Run with:  streamlit run app.py

This file only sets up the page, the theme and the sidebar navigation.
Each module lives in its own file under views/, and all data/ML logic lives
in the fraudshield package under src/.
"""

from __future__ import annotations

import streamlit as st

from fraudshield import __version__
from fraudshield.ui.modules import MODULES
from fraudshield.ui.theme import GLOBAL_CSS

st.set_page_config(
    page_title="FraudShield AI",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# Build sidebar navigation from the module registry, grouped by section.
home = st.Page("views/home.py", title="Home", icon=":material/home:", default=True)
sections: dict[str, list] = {"": [home]}
for module in MODULES:
    page = st.Page(f"views/{module.key}.py", title=module.title, icon=module.icon)
    sections.setdefault(module.section, []).append(page)

navigation = st.navigation(sections)

with st.sidebar:
    st.markdown(
        f"<div style='color:#A99BB3;font-size:.75rem;margin-top:1rem'>"
        f"FraudShield AI · v{__version__}</div>",
        unsafe_allow_html=True,
    )

navigation.run()
