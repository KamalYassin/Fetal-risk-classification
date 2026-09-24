"""Dataset and label-methodology page."""

import pandas as pd
import streamlit as st
from components.charts import class_distribution_figure
from components.metric_cards import metric_row
from components.navigation import (
    callout,
    footer,
    page_header,
    section_label,
    source_note,
)
from utils.data_loader import final_dataset, labels

LABEL_RULES = pd.DataFrame(
    [
        {
            "Assigned class": "Pathological",
            "Rule": "(pH < 7.05 AND severe metabolic criterion) OR Apgar5 ≤ 3",
            "Metabolic criterion": "BE ≤ −12 OR BDecf ≥ 12",
        },
        {
            "Assigned class": "Suspicious",
            "Rule": "pH < 7.20 OR moderate metabolic criterion OR Apgar5 ≤ 6",
            "Metabolic criterion": "BE ≤ −8 OR BDecf ≥ 8",
        },
        {
            "Assigned class": "Normal",
            "Rule": "No Pathological or Suspicious rule is met",
            "Metabolic criterion": "—",
        },
    ]
)


def render() -> None:
    label_data = labels()
    dataset = final_dataset()
    counts = label_data["label"].value_counts()

    page_header(
        "Page 2 of 7 · About the data",
        "The project uses 552 CTG scans",
        "Each scan contains a baby's heart-rate signal and contraction activity. "
        "The records were placed into three research groups using information "
        "about the baby's condition after birth.",
    )
    metric_row(
        [
            ("Records", f"{len(label_data):,}", "One row per patient"),
            ("Normal", f"{counts.get('Normal', 0):,}", "Outcome-derived label"),
            ("Suspicious", f"{counts.get('Suspicious', 0):,}", "Outcome-derived label"),
            ("Pathological", f"{counts.get('Pathological', 0):,}", "Outcome-derived label"),
        ]
    )

    chart_column, context_column = st.columns([1.45, 1], gap="large")
    with chart_column:
        st.pyplot(class_distribution_figure(label_data["label"]), clear_figure=True)
    with context_column:
        st.subheader("Signals and metadata")
        st.markdown(
            """
            - **FHR:** fetal heart-rate signal
            - **UC:** uterine-contraction activity
            - **pH:** umbilical artery outcome measurement
            - **BE / BDecf:** acid–base outcome measures
            - **Apgar5:** five-minute Apgar score
            """
        )
        callout(
            "<strong>Class imbalance:</strong> only 27 of 552 records are "
            "Pathological. Accuracy alone is therefore insufficient; the final "
            "evaluation emphasizes macro metrics and per-class performance.",
            "coral",
        )

    section_label("How the groups were created")
    st.subheader("The labels came from birth outcomes")
    st.write(
        "The original dataset did not call scans Normal, Suspicious, or "
        "Pathological. This project created those groups from four measurements "
        "recorded after birth. This is useful for research, but it is not the "
        "same as a doctor's CTG diagnosis."
    )
    callout(
        "<strong>Simple meaning:</strong> Normal is the least concerning research "
        "group, Suspicious is the middle group, and Pathological is the most "
        "concerning group. These are project labels, not medical diagnoses.",
        "amber",
    )
    with st.expander("See the exact technical label rules"):
        st.dataframe(LABEL_RULES, hide_index=True, width="stretch")

    with st.expander("See detailed outcome measurements"):
        summary = (
            label_data.groupby("label")[["pH", "BE", "BDecf", "Apgar5"]]
            .agg(["count", "median", "min", "max"])
            .round(2)
        )
        st.dataframe(summary, width="stretch")

    with st.expander("Explore one patient's outcome measurements"):
        record_ids = dataset["record_id"].astype(str).tolist()
        selected_id = st.selectbox(
            "Choose a record",
            record_ids,
            index=record_ids.index("1158") if "1158" in record_ids else 0,
        )
        selected = dataset.loc[
            dataset["record_id"].astype(str) == selected_id
        ].iloc[0]
        record_columns = st.columns(5)
        values = [
            ("Research group", str(selected["label"])),
            ("pH", f"{selected['pH']:.2f}" if pd.notna(selected["pH"]) else "Missing"),
            ("BE", f"{selected['BE']:.2f}" if pd.notna(selected["BE"]) else "Missing"),
            ("BDecf", f"{selected['BDecf']:.2f}" if pd.notna(selected["BDecf"]) else "Missing"),
            ("Apgar at 5 min", f"{selected['Apgar5']:.0f}" if pd.notna(selected["Apgar5"]) else "Missing"),
        ]
        for column, (label, value) in zip(record_columns, values):
            with column:
                st.metric(label, value)

    source_note(
        "data/processed/labels.csv",
        "data/processed/ml_dataset_final_temporal.csv",
        "src/data/generate_labels.py",
    )
    footer()
