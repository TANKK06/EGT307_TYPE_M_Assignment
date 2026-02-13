from __future__ import annotations

import streamlit as st


def apply_global_styles() -> None:
    """
    Inject global CSS styles used across the dashboard.

    Why this function exists:
    - Streamlit has basic styling by default.
    - We add a small custom "design system" (cards + badges) so all pages look consistent.

    How it works:
    - st.markdown(..., unsafe_allow_html=True) allows us to render a <style> block.
    - The CSS class names defined here are used by helper functions like:
      - card()
      - status_badge()
    """
    st.markdown(
        """
        <style>
          /* -----------------------------
             Card container used for small info panels
             ----------------------------- */
          .pm-card {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 12px;
          }

          /* Subtle/secondary text color used inside cards */
          .pm-subtle {
            color: rgba(49, 51, 63, 0.65);
          }

          /* -----------------------------
             Generic badge style (pill shape)
             ----------------------------- */
          .pm-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid rgba(49, 51, 63, 0.2);
            margin-left: 6px;
          }

          /* Badge variants: success vs failure */
          .pm-badge-ok  { background: rgba(0, 200, 0, 0.07); }
          .pm-badge-bad { background: rgba(200, 0, 0, 0.07); }
        </style>
        """,
        # Required so Streamlit will render HTML/CSS content.
        unsafe_allow_html=True,
    )


def card(title: str, body_md: str) -> None:
    """
    Render a simple "card" UI block with a title and a body.

    Args:
        title:
            Displayed at the top in bold.
        body_md:
            Card content area (supports basic HTML/Markdown).

    Note:
    - This uses HTML <div> blocks, so unsafe_allow_html=True is required.
    - body_md is inserted directly into HTML; avoid untrusted user input here.
    """
    st.markdown(
        f"""
        <div class="pm-card">
          <!-- Card title -->
          <div style="font-weight:700; font-size:16px;">{title}</div>

          <!-- Card body -->
          <div class="pm-subtle" style="margin-top:6px;">
            {body_md}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(ok: bool, label_ok: str = "OK", label_bad: str = "Down") -> str:
    """
    Return an HTML string representing a status badge.

    Used for:
    - Showing service status (UP/DOWN) in the dashboard status page

    Args:
        ok:
            True  -> green-ish badge
            False -> red-ish badge
        label_ok:
            Text shown when ok=True (default "OK")
        label_bad:
            Text shown when ok=False (default "Down")

    Returns:
        An HTML <span> string. You still need to render it using:
            st.markdown(badge_html, unsafe_allow_html=True)
    """
    klass = "pm-badge-ok" if ok else "pm-badge-bad"
    label = label_ok if ok else label_bad
    return f'<span class="pm-badge {klass}">{label}</span>'


def section_title(title: str, subtitle: str | None = None) -> None:
    """
    Standard section header used on each page for consistency.

    Args:
        title:
            Main header text (rendered as "## title")
        subtitle:
            Optional helper text shown below the title using st.caption()
    """
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
