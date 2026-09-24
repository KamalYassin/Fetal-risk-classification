"""Interactive portfolio dashboard for the completed CTG honours project.

Run from the project root:

    streamlit run app/app.py
"""

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st
from components.navigation import (
    configure_app,
    render_sidebar_context,
)
from pages import (
    ctg_visualization,
    dataset_explorer,
    experiment_journey,
    feature_engineering,
    model_comparison,
    prediction_demo,
    project_overview,
)

configure_app()

navigation = st.navigation(
    [
        st.Page(
            project_overview.render,
            title="1 · Start here",
            icon=":material/home:",
            url_path="overview",
            default=True,
        ),
        st.Page(
            dataset_explorer.render,
            title="2 · About the data",
            icon=":material/database:",
            url_path="dataset",
        ),
        st.Page(
            feature_engineering.render,
            title="3 · How it works",
            icon=":material/monitoring:",
            url_path="features",
        ),
        st.Page(
            experiment_journey.render,
            title="4 · What we tested",
            icon=":material/route:",
            url_path="experiments",
        ),
        st.Page(
            model_comparison.render,
            title="5 · Results",
            icon=":material/leaderboard:",
            url_path="models",
        ),
        st.Page(
            ctg_visualization.render,
            title="6 · View a CTG scan",
            icon=":material/ecg_heart:",
            url_path="signals",
        ),
        st.Page(
            prediction_demo.render,
            title="7 · Upload data (advanced)",
            icon=":material/upload_file:",
            url_path="prediction",
        ),
    ],
    position="sidebar",
    expanded=True,
)

render_sidebar_context()
navigation.run()
