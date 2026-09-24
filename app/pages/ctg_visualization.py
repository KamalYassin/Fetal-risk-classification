"""Interactive raw CTG signal explorer."""

import pandas as pd
import streamlit as st
from components.charts import (
    contribution_figure,
    ctg_figure,
    probability_figure,
)
from components.metric_cards import metric_row
from components.navigation import (
    callout,
    footer,
    page_header,
    section_label,
    source_note,
)
from utils.data_loader import ctg_record, final_dataset, labels
from utils.model_loader import load_artifacts, model_metadata
from utils.prediction import (
    explain_prediction,
    predict_frame,
    probability_mapping,
)

FRIENDLY_FEATURE_NAMES = {
    "recording_duration_minutes": "Scan length",
    "percent_missing_fhr": "Missing heart-rate data",
    "longest_missing_fhr_gap_seconds": "Longest missing-data gap",
    "valid_fhr_percentage": "Usable heart-rate data",
    "possible_fhr_artifact_percentage": "Possible signal errors",
    "mean_fhr": "Average heart rate",
    "median_fhr": "Typical heart rate",
    "std_fhr": "Heart-rate variation",
    "baseline_fhr": "Usual heart-rate level",
    "short_term_variability": "Short-term heart-rate variation",
    "long_term_variability": "Long-term heart-rate variation",
    "acceleration_count": "Number of heart-rate rises",
    "deceleration_count": "Number of heart-rate drops",
    "max_deceleration_depth": "Largest heart-rate drop",
    "contraction_count": "Number of contractions",
    "bradycardia_percentage": "Time with a low heart rate",
    "tachycardia_percentage": "Time with a high heart rate",
    "maximum_negative_fhr_slope": "Fastest heart-rate fall",
}


def _event_table(events: list[dict], event_type: str) -> pd.DataFrame:
    rows = []
    for event in events:
        rows.append(
            {
                "type": event_type,
                "start (min)": event["start_seconds"] / 60,
                "end (min)": event["end_seconds"] / 60,
                "duration (s)": event["duration_seconds"],
                "amplitude / depth (bpm)": event["amplitude_or_depth"],
            }
        )
    return pd.DataFrame(rows)


def _plain_feature_name(name: str) -> str:
    return FRIENDLY_FEATURE_NAMES.get(
        name,
        name.replace("_", " ").replace("  ", " ").strip().title(),
    )


def _driver_text(
    contribution_row: pd.Series,
    patient_row: pd.Series,
    dataset: pd.DataFrame,
    predicted_label: str,
) -> str:
    """Describe one model contribution without implying clinical causality."""
    feature = str(contribution_row["feature"])
    value = float(patient_row[feature])
    median = float(dataset[feature].median())
    tolerance = max(abs(median) * 0.01, 1e-9)
    if value > median + tolerance:
        comparison = "above"
    elif value < median - tolerance:
        comparison = "below"
    else:
        comparison = "close to"
    return (
        f"**{_plain_feature_name(feature)}** was {comparison} the project-cohort "
        f"typical value ({value:.2f} compared with {median:.2f}). This helped the "
        f"computer model choose **{predicted_label}**."
    )


def render() -> None:
    label_data = labels()
    dataset = final_dataset()
    model, encoder = load_artifacts()
    metadata = model_metadata(model, encoder)
    label_map = dict(
        zip(label_data["record_id"].astype(str), label_data["label"])
    )
    record_ids = sorted(label_map)

    page_header(
        "Page 6 of 7 · View a CTG scan",
        "Choose a scan, see the model result, and understand the main reasons",
        "Pick any of the 552 recordings below. The page will show the scan, the "
        "computer model's risk group, and a short explanation in plain English.",
    )

    section_label("Step 1 · Choose a scan")
    selected_id = st.selectbox(
        "CTG scan number",
        record_ids,
        index=record_ids.index("1158") if "1158" in record_ids else 0,
        format_func=lambda value: f"Scan {value}",
        help="Choose any scan number from the dataset.",
    )

    try:
        record = ctg_record(selected_id)
    # Raw waveforms are optional in the submitted repository.
    except Exception as error:  # noqa: BLE001
        st.info(
            "Raw CTG waveform files are not included in this submission, so the "
            "scan plot is unavailable. Add the matching WFDB .hea and .dat files "
            "under data/raw to enable this page."
        )
        with st.expander("Technical details"):
            st.caption(str(error))
        source_note(
            "src/data/load_ctg.py",
            "src/features/extract_clinical_features.py",
            "data/processed/labels.csv",
        )
        footer()
        return

    features = record["features"]
    patient_row = dataset.loc[
        dataset["record_id"].astype(str) == selected_id
    ]
    prediction = predict_frame(
        patient_row,
        model,
        encoder,
        metadata["feature_names"],
    ).iloc[0]
    probabilities = probability_mapping(prediction)
    explanation = explain_prediction(
        patient_row,
        model,
        encoder,
        metadata["feature_names"],
    )
    predicted_label = str(prediction["predicted_label"])
    actual_label = label_map[selected_id]

    section_label("Step 2 · See the result")
    metric_row(
        [
            ("Computer model result", predicted_label, "Saved Logistic Regression model"),
            ("Research label", actual_label, "Created from birth outcomes"),
            (
                "Model chance of Pathological",
                f"{probabilities['Pathological']:.1%}",
                "A model score, not a medical probability",
            ),
            (
                "Model's highest score",
                f"{max(probabilities.values()):.1%}",
                "Not a measure of clinical certainty",
            ),
        ]
    )

    probability_column, meaning_column = st.columns([1, 1], gap="large")
    with probability_column:
        st.pyplot(
            probability_figure(probabilities),
            clear_figure=True,
            width="stretch",
        )
        if predicted_label == actual_label:
            st.success(
                "For this retrospective record, the demo prediction matches "
                "the outcome-derived label."
            )
        else:
            st.warning(
                "For this retrospective record, the demo prediction does not "
                "match the outcome-derived label."
            )
    with meaning_column:
        st.subheader(f"What does “{predicted_label}” mean here?")
        meanings = {
            "Normal": (
                "The model found this scan most similar to the project's least "
                "concerning research group."
            ),
            "Suspicious": (
                "The model placed this scan in the middle research group. This "
                "does not mean that a doctor made a Suspicious diagnosis."
            ),
            "Pathological": (
                "The model found this scan most similar to the project's most "
                "concerning research group. This is not a medical diagnosis."
            ),
        }
        callout(meanings[predicted_label], "teal")
        st.write(
            "The **research label** shown above comes from birth outcomes. The "
            "**computer model result** comes only from patterns in the scan. "
            "They can be different."
        )

    section_label("Step 3 · Look at the scan")
    duration = float(features["recording_duration_minutes"])
    default_end = min(duration, 30.0)
    window = st.slider(
        "Visible time window (minutes)",
        min_value=0.0,
        max_value=float(round(duration, 2)),
        value=(0.0, float(round(default_end, 2))),
        step=0.5,
    )

    metric_row(
        [
            ("Duration", f"{duration:.1f} min", "Full recording"),
            ("Valid FHR", f"{features['valid_fhr_percentage']:.1f}%", "After cleaning"),
            (
                "Longest missing gap",
                f"{features['longest_missing_fhr_gap_seconds']:.1f} s",
                "Source missingness",
            ),
            (
                "Possible artifact",
                f"{features['possible_fhr_artifact_percentage']:.2f}%",
                "Range/spike filter",
            ),
            ("Baseline FHR", f"{features['baseline_fhr']:.1f} bpm", "Median valid moving baseline"),
        ]
    )

    st.pyplot(
        ctg_figure(record, window[0], window[1]),
        clear_figure=True,
        width="stretch",
    )
    with st.expander("How to read the colours on this chart"):
        st.markdown(
            """
            - **Grey line:** original heart-rate signal
            - **Dark blue line:** cleaned heart-rate signal
            - **Teal line:** estimated usual heart-rate level
            - **Green areas:** detected heart-rate rises
            - **Red areas:** detected heart-rate drops
            - **Orange lower line:** smoothed contraction signal
            """
        )

    section_label("Step 4 · Understand the main reasons")
    st.subheader("What influenced the computer model?")
    st.write(
        "The model looked at 51 measurements from the scan. These three "
        "measurements had the strongest effect in favour of its answer:"
    )
    supporting = explanation["contributions"].loc[
        explanation["contributions"]["contribution"] > 0
    ].head(3)
    for _, driver in supporting.iterrows():
        st.markdown(
            "- "
            + _driver_text(
                driver,
                patient_row.iloc[0],
                dataset,
                predicted_label,
            )
        )
    with st.expander("Technical details: see the model contribution chart and table"):
        st.pyplot(
            contribution_figure(
                explanation["contributions"],
                explanation["predicted_label"],
                explanation["runner_up_label"],
            ),
            clear_figure=True,
            width="stretch",
        )
        st.write(
            f"The chart compares **{explanation['predicted_label']}** with the "
            f"second-highest model result, **{explanation['runner_up_label']}** "
            f"({explanation['runner_up_probability']:.1%})."
        )
        st.dataframe(
            explanation["contributions"][
                [
                    "feature",
                    "value",
                    "standardized_value",
                    "contribution",
                    "direction",
                ]
            ].style.format(
                {
                    "value": "{:.3f}",
                    "standardized_value": "{:.3f}",
                    "contribution": "{:+.3f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        callout(
            "<strong>Technical note:</strong> these are exact fitted Logistic "
            "Regression contributions for the selected class versus the runner-up. "
            "They explain model mathematics—not physiology or causality.",
            "amber",
        )

    section_label("Detected events")
    analysis = record["analysis"]
    event_metrics = st.columns(3)
    with event_metrics[0]:
        st.metric(
            "Accelerations",
            analysis["acceleration_features"]["count"],
            f"Max amplitude {analysis['acceleration_features']['max_amplitude_or_depth']:.1f} bpm",
        )
    with event_metrics[1]:
        st.metric(
            "Decelerations",
            analysis["deceleration_features"]["count"],
            f"Max depth {analysis['deceleration_features']['max_amplitude_or_depth']:.1f} bpm",
        )
    with event_metrics[2]:
        st.metric(
            "Contractions",
            features["contraction_count"],
            f"{features['contraction_frequency_per_hour']:.1f} per hour",
        )

    acceleration_table = _event_table(
        analysis["acceleration_features"]["events"], "Acceleration"
    )
    deceleration_table = _event_table(
        analysis["deceleration_features"]["events"], "Deceleration"
    )
    event_table = pd.concat(
        [acceleration_table, deceleration_table],
        ignore_index=True,
    )
    if event_table.empty:
        st.info("No acceleration or deceleration met the configured criteria.")
    else:
        st.dataframe(
            event_table.round(2),
            hide_index=True,
            width="stretch",
        )

    with st.expander("View extracted patient features"):
        feature_frame = pd.DataFrame(
            {
                "feature": list(features),
                "value": [str(features[name]) for name in features],
            }
        )
        st.dataframe(
            feature_frame,
            hide_index=True,
            width="stretch",
        )

    callout(
        "Event and variability measurements are computational approximations "
        "for this research pipeline—not clinical CTG annotations.",
        "amber",
    )
    source_note(
        "data/raw/ctu-chb-intrapartum-cardiotocography-database-1.0.0/",
        "src/data/load_ctg.py",
        "src/features/extract_clinical_features.py",
        "data/processed/labels.csv",
    )
    footer()
