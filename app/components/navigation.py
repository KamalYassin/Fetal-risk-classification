"""Shared layout, styling, and source/disclaimer components."""

from html import escape

import streamlit as st

APP_CSS = """
<style>
    :root {
        --ink: #10243e;
        --muted: #5e7086;
        --navy: #0b1f35;
        --teal: #0f8b8d;
        --aqua: #dff5f2;
        --coral: #ef6f61;
        --amber: #e9a23b;
        --paper: #f7f9fc;
        --line: #dce5ee;
    }
    .stApp {
        background:
            radial-gradient(circle at 92% 2%, rgba(15,139,141,.10), transparent 24rem),
            linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2.3rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3 {
        color: var(--navy);
        letter-spacing: -0.025em;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f35 0%, #102d49 100%);
    }
    [data-testid="stSidebar"] * {
        color: #eef7f7;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(117, 221, 210, .16);
        border-left: 3px solid #75ddd2;
    }
    .ctg-brand {
        padding: .9rem .25rem 1.2rem;
        border-bottom: 1px solid rgba(255,255,255,.14);
        margin-bottom: 1rem;
    }
    .ctg-brand-mark {
        color: #75ddd2;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .ctg-brand-title {
        color: #fff;
        font-size: 1.12rem;
        font-weight: 760;
        line-height: 1.2;
        margin-top: .35rem;
    }
    .ctg-hero {
        position: relative;
        overflow: hidden;
        padding: 1.75rem 2rem;
        margin: 0 0 1.6rem;
        border: 1px solid rgba(15,139,141,.18);
        border-radius: 22px;
        background:
            linear-gradient(120deg, rgba(11,31,53,.98), rgba(15,72,92,.96));
        box-shadow: 0 20px 55px rgba(11,31,53,.14);
    }
    .ctg-hero:after {
        content: "";
        position: absolute;
        width: 18rem;
        height: 18rem;
        right: -5rem;
        top: -8rem;
        border-radius: 50%;
        border: 2.5rem solid rgba(117,221,210,.10);
    }
    .ctg-kicker {
        color: #75ddd2;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .17em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .ctg-hero h1 {
        color: #fff;
        font-size: clamp(2rem, 4vw, 3.3rem);
        line-height: 1.03;
        max-width: 900px;
        margin: 0;
    }
    .ctg-hero p {
        color: #d6e6ec;
        max-width: 820px;
        font-size: 1.02rem;
        line-height: 1.65;
        margin: .9rem 0 0;
    }
    .ctg-section-label {
        color: var(--teal);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin: 2rem 0 .25rem;
    }
    .ctg-callout {
        padding: 1rem 1.15rem;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,.82);
        color: var(--ink);
        line-height: 1.55;
    }
    .ctg-callout strong { color: var(--navy); }
    .ctg-callout.teal {
        border-left: 4px solid var(--teal);
        background: #f0fbf9;
    }
    .ctg-callout.amber {
        border-left: 4px solid var(--amber);
        background: #fff9ee;
    }
    .ctg-callout.coral {
        border-left: 4px solid var(--coral);
        background: #fff5f3;
    }
    .ctg-source {
        font-size: .78rem;
        color: var(--muted);
        border-top: 1px solid var(--line);
        margin-top: 1.2rem;
        padding-top: .65rem;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.88);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: .85rem 1rem;
        box-shadow: 0 8px 24px rgba(20,48,76,.05);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--navy); }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: .65rem 1rem;
    }
    .ctg-footer {
        color: var(--muted);
        font-size: .76rem;
        text-align: center;
        padding-top: 2.4rem;
    }
    .ctg-choice-card {
        min-height: 142px;
        padding: 1.15rem 1.2rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255,255,255,.9);
        box-shadow: 0 9px 28px rgba(20,48,76,.06);
    }
    .ctg-choice-card h3 {
        font-size: 1.08rem;
        margin: 0 0 .45rem;
    }
    .ctg-choice-card p {
        color: var(--muted);
        line-height: 1.45;
        margin: 0;
    }
</style>
"""


def configure_app() -> None:
    """Set global Streamlit metadata and the visual system."""
    st.set_page_config(
        page_title="CTG Fetal Risk | Honours Project",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "Interactive research dashboard for the CTG fetal-risk "
                "classification honours project. Research use only."
            )
        },
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_sidebar_context() -> None:
    """Show compact project identity and safety context below navigation."""
    st.sidebar.markdown(
        """
        <div class="ctg-brand">
          <div class="ctg-brand-mark">Honours research</div>
          <div class="ctg-brand-title">CTG fetal-risk classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### How to use this site")
    st.sidebar.write(
        "Start at page 1 and follow the numbers, or jump straight to "
        "**6 · View a CTG scan**."
    )
    st.sidebar.caption("552 scans · 3 risk groups · 5 models")
    st.sidebar.warning(
        "Research prototype only — not for clinical decision making.",
        icon="⚠️",
    )


def page_header(kicker: str, title: str, description: str) -> None:
    """Render the shared high-contrast page masthead."""
    st.markdown(
        (
            '<section class="ctg-hero">'
            f'<div class="ctg-kicker">{escape(kicker)}</div>'
            f"<h1>{escape(title)}</h1>"
            f"<p>{escape(description)}</p>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    """Render a small editorial section marker."""
    st.markdown(
        f'<div class="ctg-section-label">{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def callout(body: str, tone: str = "teal") -> None:
    """Render trusted local HTML inside a styled callout."""
    st.markdown(
        f'<div class="ctg-callout {tone}">{body}</div>',
        unsafe_allow_html=True,
    )


def choice_card(title: str, body: str) -> None:
    """Render a large, easy-to-scan home-page choice."""
    st.markdown(
        (
            '<div class="ctg-choice-card">'
            f"<h3>{escape(title)}</h3>"
            f"<p>{escape(body)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def source_note(*paths: str) -> None:
    """Show which committed project artifacts support a view."""
    joined = " · ".join(f"<code>{escape(path)}</code>" for path in paths)
    st.markdown(
        f'<div class="ctg-source"><strong>Source:</strong> {joined}</div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    """Render the shared research disclaimer."""
    st.markdown(
        """
        <div class="ctg-footer">
          Research prototype · Outcome-derived labels · No clinical validation ·
          Not intended for diagnosis or patient management
        </div>
        """,
        unsafe_allow_html=True,
    )
