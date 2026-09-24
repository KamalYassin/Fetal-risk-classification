"""Research experiment journey page."""

import streamlit as st
from components.navigation import (
    callout,
    footer,
    page_header,
    section_label,
    source_note,
)
from utils.data_loader import (
    PROJECT_ROOT,
    extract_markdown_section,
    read_markdown,
    temporal_cv_summary,
)

JOURNEY = [
    (
        "01",
        "Clinical feature foundation",
        (
            "Artifact-aware preprocessing, moving baseline, continuous event rules, "
            "contraction filtering, quality controls, and expanded CTG summaries."
        ),
        "src/features/extract_clinical_features.py",
    ),
    (
        "02",
        "Pathological-case diagnosis",
        (
            "Class balancing, feature separation, split stability, threshold behaviour, "
            "and patient-level error analysis."
        ),
        "reports/pathological_diagnostic_report.md",
    ),
    (
        "03",
        "Signal-quality audit",
        "Measured how missingness, fragmentation, and masking influence feature reliability.",
        "reports/signal_quality_audit/signal_quality_audit_report.md",
    ),
    (
        "04",
        "Temporal information audit",
        "Tested whether whole-record aggregation concealed local extremes and evolving patterns.",
        "reports/temporal_information_audit/temporal_information_audit_report.md",
    ),
    (
        "05",
        "Quality-aware aggregation",
        "Compared the unchanged baseline against an alternative quality-aware representation.",
        "reports/quality_aware_aggregation/quality_aware_aggregation_report.md",
    ),
    (
        "06",
        "Limited temporal experiment",
        "Evaluated a constrained group of temporal candidates without broad feature expansion.",
        "reports/limited_temporal_experiment/limited_temporal_experiment_report.md",
    ),
    (
        "07",
        "Final simplification",
        "Removed one temporal transition feature after a controlled equivalence comparison.",
        "reports/final_temporal_simplification/final_temporal_simplification_report.md",
    ),
    (
        "08",
        "Five-model comparison",
        "Compared Logistic Regression, Random Forest, XGBoost, LightGBM, and RBF SVM.",
        "reports/final_model_comparison/final_model_comparison_report.md",
    ),
]


def render() -> None:
    page_header(
        "Page 4 of 7 · What we tested",
        "The final answer came after several careful experiments",
        "The project did not jump straight to one model. It checked data quality, "
        "tested whether timing information helped, simplified the feature set, "
        "and then compared five models fairly.",
    )

    section_label("Research timeline")
    for number, title, body, source in JOURNEY:
        left, right = st.columns([0.12, 0.88])
        with left:
            st.markdown(f"### {number}")
        with right:
            st.subheader(title)
            st.write(body)
            st.caption(source)
        st.divider()

    with st.expander("See the technical temporal-feature comparison"):
        temporal_summary_path = (
            PROJECT_ROOT
            / "reports/final_temporal_simplification/repeated_cv_summary.csv"
        )
        temporal_report_path = (
            PROJECT_ROOT
            / "reports/final_temporal_simplification/"
            "final_temporal_simplification_report.md"
        )
        if not temporal_summary_path.exists() or not temporal_report_path.exists():
            st.info(
                "The generated report bundle is not included in this submission. "
                "The experiment design remains available in "
                "src/experiments/final_temporal_simplification_experiment.py."
            )
        else:
            temporal = temporal_cv_summary()
            macro = temporal[
                (temporal["row_type"] == "representation")
                & (temporal["metric"] == "macro_f1")
            ][
                ["representation_or_comparison", "mean", "standard_deviation"]
            ].copy()
            macro["representation_or_comparison"] = macro[
                "representation_or_comparison"
            ].replace(
                {
                    "A_baseline": "A · Whole record",
                    "B_full_combined_temporal": "B · Five temporal",
                    "C_simplified_temporal": "C · Four temporal",
                }
            )
            st.bar_chart(
                macro.set_index("representation_or_comparison")["mean"],
                color="#0f8b8d",
            )
            st.dataframe(
                macro.rename(
                    columns={
                        "representation_or_comparison": "Representation",
                        "mean": "Repeated-CV Macro F1",
                        "standard_deviation": "SD",
                    }
                ).round(4),
                hide_index=True,
                width="stretch",
            )
            temporal_report = read_markdown(
                "reports/final_temporal_simplification/"
                "final_temporal_simplification_report.md"
            )
            conclusion = extract_markdown_section(temporal_report, "Conclusion")
            callout(
                "<strong>Recorded conclusion:</strong> " + conclusion,
                "teal",
            )

    section_label("Evidence gallery")
    gallery = [
        (
            "Signal quality",
            "reports/signal_quality_audit/figures/03_feature_reliability_heatmap.png",
        ),
        (
            "Temporal candidates",
            "reports/temporal_information_audit/figures/07_temporal_candidates_auc.png",
        ),
        (
            "Limited temporal comparison",
            "reports/limited_temporal_experiment/figures/confusion_matrix_D_combined_limited_temporal.png",
        ),
    ]
    tabs = st.tabs([title for title, _ in gallery])
    for tab, (_, relative_path) in zip(tabs, gallery):
        with tab:
            path = PROJECT_ROOT / relative_path
            if path.exists():
                st.image(str(path), width="stretch")
            else:
                st.info(f"Figure not available: {relative_path}")

    callout(
        "<strong>Why retain every stage?</strong> Intermediate and negative "
        "experiments support the thesis reasoning: they show which hypotheses "
        "were tested, what changed, and why the final representation stayed limited.",
        "amber",
    )
    source_note(
        "reports/signal_quality_audit/",
        "reports/temporal_information_audit/",
        "reports/quality_aware_aggregation/",
        "reports/limited_temporal_experiment/",
        "reports/final_temporal_simplification/",
    )
    footer()
