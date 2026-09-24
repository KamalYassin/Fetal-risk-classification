"""Feature-engineering and signal-quality page."""

import pandas as pd
import streamlit as st
from components.metric_cards import metric_row
from components.navigation import (
    callout,
    footer,
    page_header,
    section_label,
    source_note,
)
from utils.data_loader import PROJECT_ROOT, clinical_features, final_dataset

FEATURE_GROUPS = {
    "Recording & quality": [
        "recording_duration_minutes",
        "percent_missing_fhr",
        "longest_missing_fhr_gap_seconds",
        "valid_fhr_percentage",
        "possible_fhr_artifact_percentage",
    ],
    "FHR summary & baseline": [
        "mean_fhr",
        "median_fhr",
        "min_fhr",
        "max_fhr",
        "std_fhr",
        "baseline_fhr",
        "baseline_drift_std",
        "baseline_drift_range",
    ],
    "Variability": [
        "short_term_variability",
        "long_term_variability",
        "segment_stv_mean",
        "segment_stv_std",
        "segment_ltv_mean",
        "segment_ltv_std",
    ],
    "Accelerations & decelerations": [
        "acceleration_count",
        "max_acceleration_amplitude",
        "deceleration_count",
        "max_deceleration_depth",
        "bradycardia_percentage",
        "tachycardia_percentage",
        "percentage_time_below_baseline",
    ],
    "Contractions & interactions": [
        "contraction_count",
        "contraction_frequency_per_hour",
        "decelerations_per_contraction",
        "contractions_followed_by_deceleration_percentage",
        "mean_delay_contraction_to_deceleration_seconds",
    ],
    "Dynamics": [
        "baseline_crossing_count",
        "mean_positive_fhr_slope",
        "mean_negative_fhr_slope",
        "maximum_negative_fhr_slope",
    ],
}

TEMPORAL_FEATURES = [
    "maximum_negative_fhr_slope__window_final",
    "valid_fhr_percentage__window_min",
    "poor_quality_window_percentage",
    "final_window_quality",
]


def render() -> None:
    features = clinical_features()
    final = final_dataset()
    quality_counts = features["feature_quality_flag"].value_counts()

    page_header(
        "Page 3 of 7 · How it works",
        "The computer turns each scan into a list of useful measurements",
        "Instead of reading thousands of points one by one, the project summarizes "
        "the heart rate, contractions, important events, and signal quality.",
    )
    metric_row(
        [
            ("Whole-record inputs", "51", "Expected by the saved demo model"),
            ("Selected temporal inputs", "4", "Final controlled representation"),
            ("Final model inputs", "55", "Used in the five-model comparison"),
            ("Patients represented", f"{len(final):,}", "No patient aggregation loss"),
        ]
    )

    section_label("Four simple steps")
    process_columns = st.columns(4)
    steps = [
        ("1. Keep the original", "The unedited heart-rate signal is always preserved."),
        ("2. Clean obvious errors", "Missing data and unlikely spikes are marked as gaps."),
        ("3. Find the usual level", "The system estimates a changing heart-rate baseline."),
        ("4. Count patterns", "It measures rises, drops, variation, and contractions."),
    ]
    for column, (title, body) in zip(process_columns, steps):
        with column:
            st.subheader(title)
            st.write(body)

    callout(
        "<strong>Computational approximations:</strong> event, STV, LTV, and "
        "contraction definitions support reproducible modelling and visual "
        "validation; they are not asserted to be clinical gold standards.",
        "amber",
    )

    with st.expander("Technical details: explore the feature list"):
        selected_group = st.selectbox("Feature family", list(FEATURE_GROUPS))
        available = [name for name in FEATURE_GROUPS[selected_group] if name in final]
        catalogue = pd.DataFrame(
            {
                "feature": available,
                "dataset mean": [final[name].mean() for name in available],
                "dataset median": [final[name].median() for name in available],
                "minimum": [final[name].min() for name in available],
                "maximum": [final[name].max() for name in available],
            }
        ).round(3)
        st.dataframe(catalogue, hide_index=True, width="stretch")

    with st.expander("Technical details: the four time-based features"):
        temporal_descriptions = pd.DataFrame(
            [
                (
                    "maximum_negative_fhr_slope__window_final",
                    "Strongest downward FHR slope in the final temporal window.",
                ),
                (
                    "valid_fhr_percentage__window_min",
                    "Lowest window-level valid-FHR percentage.",
                ),
                (
                    "poor_quality_window_percentage",
                    "Share of temporal windows classified as poor quality.",
                ),
                (
                    "final_window_quality",
                    "Valid-FHR percentage in the final window.",
                ),
            ],
            columns=["feature", "interpretation"],
        )
        temporal_descriptions["mean"] = [
            final[name].mean() for name in TEMPORAL_FEATURES
        ]
        temporal_descriptions["standard deviation"] = [
            final[name].std() for name in TEMPORAL_FEATURES
        ]
        st.dataframe(
            temporal_descriptions.round(3),
            hide_index=True,
            width="stretch",
        )

    section_label("Signal quality")
    quality_columns = st.columns(3)
    for column, label in zip(quality_columns, ["good", "review", "poor"]):
        with column:
            st.metric(
                f"{label.title()} quality",
                f"{int(quality_counts.get(label, 0))} records",
            )
    st.bar_chart(
        quality_counts.rename_axis("quality").rename("records"),
        color="#0f8b8d",
    )
    st.write(
        "The feature-quality flag supports quality analysis and traceability. "
        "It is excluded from the model feature matrix."
    )

    section_label("Representative records")
    tabs = st.tabs(["Normal · 1142", "Suspicious · 1363", "Pathological · 1158"])
    images = [
        "reports/figures/representative_records/normal_1142.png",
        "reports/figures/representative_records/suspicious_1363.png",
        "reports/figures/representative_records/pathological_1158.png",
    ]
    for tab, relative_path in zip(tabs, images):
        with tab:
            image_path = PROJECT_ROOT / relative_path
            if image_path.exists():
                st.image(str(image_path), width="stretch")
            else:
                st.info(f"Figure not available: {relative_path}")

    source_note(
        "src/features/extract_clinical_features.py",
        "data/processed/clinical_features.csv",
        "data/processed/ml_dataset_final_temporal.csv",
        "reports/figures/representative_records/",
    )
    footer()
