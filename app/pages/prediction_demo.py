"""Saved-model inference demonstration."""

import pandas as pd
import streamlit as st
from components.charts import probability_figure
from components.metric_cards import metric_row
from components.navigation import (
    callout,
    footer,
    page_header,
    section_label,
    source_note,
)
from utils.data_loader import final_dataset
from utils.model_loader import load_artifacts, model_metadata
from utils.prediction import (
    predict_frame,
    probability_mapping,
)


def _single_record_demo(
    dataset: pd.DataFrame,
    model,
    encoder,
    metadata: dict,
) -> None:
    record_ids = dataset["record_id"].astype(str).tolist()
    selected_id = st.selectbox(
        "Select a patient record",
        record_ids,
        index=record_ids.index("1158") if "1158" in record_ids else 0,
    )
    row = dataset.loc[dataset["record_id"].astype(str) == selected_id]
    result = predict_frame(
        row,
        model,
        encoder,
        metadata["feature_names"],
    ).iloc[0]
    probabilities = probability_mapping(result)
    actual_label = str(row.iloc[0]["label"])
    predicted_label = str(result["predicted_label"])

    metric_row(
        [
            ("Record", selected_id, "CTU-UHB record identifier"),
            ("Outcome-derived label", actual_label, "Retrospective research target"),
            ("Demo prediction", predicted_label, "Saved Logistic Regression artifact"),
            (
                "Highest probability",
                f"{max(probabilities.values()):.1%}",
                "Not a calibrated clinical confidence",
            ),
        ]
    )
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.pyplot(
            probability_figure(probabilities),
            clear_figure=True,
            width="stretch",
        )
    with right:
        st.subheader("Feature snapshot")
        snapshot_features = [
            "baseline_fhr",
            "short_term_variability",
            "long_term_variability",
            "deceleration_count",
            "max_deceleration_depth",
            "bradycardia_percentage",
            "valid_fhr_percentage",
            "maximum_negative_fhr_slope",
        ]
        snapshot = row[snapshot_features].T.reset_index()
        snapshot.columns = ["Feature", "Value"]
        st.dataframe(
            snapshot.style.format({"Value": "{:.3f}"}),
            hide_index=True,
            width="stretch",
        )


def _upload_demo(model, encoder, metadata: dict) -> None:
    st.write(
        "Upload an in-memory CSV containing all 51 expected feature columns. "
        "Extra traceability columns such as `record_id` are retained in the "
        "download; no uploaded data are written by the application."
    )
    uploaded = st.file_uploader("Feature CSV", type=["csv"])
    if uploaded is None:
        with st.expander("View required columns"):
            st.code("\n".join(metadata["feature_names"]))
        return
    try:
        frame = pd.read_csv(uploaded)
        result = predict_frame(
            frame,
            model,
            encoder,
            metadata["feature_names"],
        )
    # Uploaded files can fail at CSV parsing, schema validation, or inference.
    except Exception as error:  # noqa: BLE001
        st.error(str(error))
        return
    st.success(f"Generated {len(result)} prediction(s) without fitting the model.")
    st.dataframe(
        result.style.format(
            {
                "probability_normal": "{:.4f}",
                "probability_suspicious": "{:.4f}",
                "probability_pathological": "{:.4f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download predictions",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="ctg_demo_predictions.csv",
        mime="text/csv",
    )


def render() -> None:
    dataset = final_dataset()
    model, encoder = load_artifacts()
    metadata = model_metadata(model, encoder)

    page_header(
        "Page 7 of 7 · Advanced tool",
        "Upload prepared feature data",
        "Most visitors do not need this page. It is for technical users who "
        "already have a CSV containing the exact measurements expected by the "
        "saved model.",
    )

    callout(
        "<strong>Artifact distinction:</strong> the saved inference artifact is "
        f"a {metadata['classifier']} pipeline using {metadata['feature_count']} "
        "whole-record features. The later five-model experiment recommended SVM "
        "on 55 finalized features but explicitly saved no trained artifact. This "
        "page does not claim that the demo artifact is the final recommended model.",
        "amber",
    )

    mode = st.radio(
        "Demonstration mode",
        ["Upload feature CSV", "Existing project record"],
        horizontal=True,
    )
    if mode == "Existing project record":
        _single_record_demo(dataset, model, encoder, metadata)
    else:
        _upload_demo(model, encoder, metadata)

    section_label("Inference contract")
    contract_columns = st.columns(3)
    details = [
        (
            "Model",
            metadata["classifier"],
            f"Fitted inside a {metadata['artifact_type']}",
        ),
        (
            "Schema",
            f"{metadata['feature_count']} features",
            "Exact fitted feature order is enforced",
        ),
        (
            "Classes",
            " · ".join(metadata["classes"]),
            "Decoded by the saved LabelEncoder",
        ),
    ]
    for column, (label, value, caption) in zip(contract_columns, details):
        with column:
            st.metric(label, value)
            st.caption(caption)

    with st.expander("Inspect the exact saved-model feature list"):
        st.dataframe(
            pd.DataFrame(
                {
                    "position": range(1, len(metadata["feature_names"]) + 1),
                    "feature": metadata["feature_names"],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    callout(
        "<strong>Limitations:</strong> selecting a committed record is a "
        "retrospective demonstration, not an unbiased test. Probabilities are "
        "not clinically calibrated, the labels are outcome-derived, and the "
        "model must not be used for diagnosis or patient management.",
        "coral",
    )
    source_note(
        "models/best_model.pkl",
        "models/label_encoder.pkl",
        "data/processed/ml_dataset_final_temporal.csv",
        "src/models/train_models.py",
        "reports/final_model_comparison/final_model_comparison_report.md",
    )
    footer()
