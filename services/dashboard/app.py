"""
Streamlit entry point for the Predictive Maintenance Dashboard.

This file is intentionally small:
- Sets up Streamlit page config + global styling
- Provides a sidebar "router" to switch between pages
- Calls the selected page render function

Keeping this file minimal reduces complexity and makes the UI easier to maintain.
"""

import streamlit as st

from ui.pages import (
    render_predict,
    render_logs,
    render_batch_predict,
    render_train,
    render_status,
)
from ui.components import apply_global_styles

# Map page names shown in the sidebar -> functions that render each page.
# This makes it easy to add new pages: just create a render_* function and add it here.
PAGES = {
    "Predict": render_predict,
    "Logs": render_logs,
    "Batch Predict": render_batch_predict,
    "Train": render_train,
    "Status": render_status,
}


def main() -> None:
    """
    Main Streamlit app function.

    Streamlit behavior:
    - Streamlit reruns the entire script frequently (on every user interaction).
    So we keep this function:
    - fast (no heavy computation here)
    - clean (delegates actual content rendering to ui/pages.py)
    - predictable (minimal side effects at import time)
    """
    # -----------------------------
    # Page configuration
    # -----------------------------
    # - page_title: browser tab title
    # - layout="wide": use more horizontal space
    # - initial_sidebar_state: show the sidebar by default
    st.set_page_config(
        page_title="Predictive Maintenance Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -----------------------------
    # Global styling
    # -----------------------------
    # Adds CSS used by ui/components.py (cards, status badges, etc.)
    apply_global_styles()

    # -----------------------------
    # Sidebar navigation ("router")
    # -----------------------------
    # This works like a simple router:
    # user chooses a page name -> we call that page's render function.
    st.sidebar.title("Navigation")
    page_name = st.sidebar.radio("Go to", list(PAGES.keys()), index=0)

    # -----------------------------
    # Render selected page
    # -----------------------------
    # Each page function handles its own UI + API calls.
    PAGES[page_name]()


# Run main() only when this file is executed directly:
# - In Docker/Streamlit: streamlit run app.py -> __name__ == "__main__"
# - If imported by tests/tools, main() will NOT auto-run.
if __name__ == "__main__":
    main()
