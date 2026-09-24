"""Project overview page."""

import streamlit as st
from components.metric_cards import metric_row
from components.navigation import (
    callout,
    choice_card,
    footer,
    page_header,
    section_label,
    source_note,
)
from utils.data_loader import final_dataset


def render() -> None:
    dataset = final_dataset()
    outcome_columns = {"record_id", "pH", "BE", "BDecf", "Apgar5", "label"}
    model_features = [
        column for column in dataset.columns if column not in outcome_columns
    ]

    page_header(
        "Welcome",
        "A simple guide to the CTG fetal-risk project",
        "You do not need a medical or technical background. Follow the numbered "
        "pages, or jump straight to the scan viewer to choose a recording and "
        "see its model result.",
    )

    metric_row(
        [
            ("Patients", f"{len(dataset):,}", "Unique CTU-UHB recordings"),
            ("Final features", str(len(model_features)), "Leakage-aware model inputs"),
            ("Risk classes", str(dataset["label"].nunique()), "Outcome-derived targets"),
            ("Models compared", "5", "Identical final evaluation protocol"),
        ]
    )

    section_label("Choose where to begin")
    choice_columns = st.columns(3, gap="large")
    choices = [
        (
            "I want the short story",
            "Continue down this page for the problem, the process, and the main idea.",
            "2 · About the data",
            "./dataset",
        ),
        (
            "I want to see the results",
            "See which machine-learning model ranked highest and what the trade-offs were.",
            "5 · Open results",
            "./models",
        ),
        (
            "I want to view a scan",
            "Choose any CTG recording, see the trace, its risk group, and a simple explanation.",
            "6 · Open scan viewer",
            "./signals",
        ),
    ]
    for column, (title, body, button, url) in zip(choice_columns, choices):
        with column:
            choice_card(title, body)
            st.link_button(button, url, width="stretch")

    section_label("The problem in plain English")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader("What is a CTG scan?")
        st.write(
            "A CTG scan records a baby's heart rate and the mother's contractions "
            "during labour. Doctors look at how these signals change over time "
            "to help understand how the baby is coping."
        )
        st.write(
            "This project asks whether a computer model can use patterns in a "
            "CTG recording to place it into one of three research groups: "
            "Normal, Suspicious, or Pathological."
        )
    with right:
        callout(
            "<strong>The important idea</strong><br>"
            "The model is a research assistant, not a doctor. It finds patterns "
            "in numbers extracted from a scan, but its answer still has limits "
            "and must never replace clinical judgement.",
            "teal",
        )

    section_label("Research pipeline")
    stages = [
        ("01", "Raw CTG", "FHR + uterine contractions"),
        ("02", "Preprocessing", "Missingness + artifact filtering"),
        ("03", "Clinical features", "Events, variability, baseline, UC"),
        ("04", "Temporal context", "Four selected window summaries"),
        ("05", "Model evidence", "Five classical ML approaches"),
        ("06", "Risk output", "Three-class research prediction"),
    ]
    columns = st.columns(3)
    for index, (number, title, detail) in enumerate(stages):
        with columns[index % 3]:
            st.markdown(
                f"**{number} · {title}**  \n"
                f"<span style='color:#5e7086'>{detail}</span>",
                unsafe_allow_html=True,
            )
            st.progress((index + 1) / len(stages))

    section_label("What this dashboard preserves")
    preserve_columns = st.columns(3)
    preserve_items = [
        (
            "Traceable claims",
            "Metrics and conclusions are loaded from committed reports and CSVs.",
        ),
        (
            "Unchanged science",
            "No labels, features, models, reports, or experiment outputs are rewritten.",
        ),
        (
            "Visible limitations",
            "The class imbalance, signal quality, model trade-offs, and artifact mismatch are explicit.",
        ),
    ]
    for column, (title, body) in zip(preserve_columns, preserve_items):
        with column:
            st.subheader(title)
            st.write(body)

    callout(
        "<strong>Important:</strong> this dashboard presents completed research. "
        "The labels are derived from outcome metadata, not original CTU-UHB risk "
        "annotations, and the interface is not a medical device.",
        "amber",
    )
    source_note(
        "data/processed/ml_dataset_final_temporal.csv",
        "reports/final_model_comparison/",
        "README.md",
    )
    footer()
