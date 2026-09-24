"""Final five-model comparison page."""

import pandas as pd
import streamlit as st
from components.charts import (
    confusion_matrix_figure,
    model_metric_figure,
    repeated_distribution_figure,
)
from components.metric_cards import decimal, metric_row, percentage
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
    fixed_model_results,
    read_markdown,
    repeated_cv_results,
    repeated_cv_summary,
    repeated_split_summary,
)

METRICS = {
    "Macro F1": "macro_f1",
    "Balanced accuracy": "balanced_accuracy",
    "Pathological recall": "pathological_recall",
    "Pathological precision": "pathological_precision",
    "Accuracy": "accuracy",
    "Weighted F1": "weighted_f1",
}


def _model_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary[summary["row_type"] == "model_summary"]
    selected = rows[rows["metric"].isin(METRICS.values())]
    table = selected.pivot(
        index="model_or_comparison",
        columns="metric",
        values="mean",
    )
    ranking_columns = [
        "macro_f1",
        "pathological_recall",
        "pathological_precision",
        "balanced_accuracy",
    ]
    return (
        table.sort_values(ranking_columns, ascending=False)
        .reset_index()
        .rename(columns={"model_or_comparison": "Model"})
    )


def render() -> None:
    page_header(
        "Page 5 of 7 · Results",
        "The SVM ranked first overall, but every model had trade-offs",
        "Five machine-learning methods were tested on the same data in the same "
        "way. This page starts with the simple result; detailed statistics are "
        "available further down for anyone who wants them.",
    )

    required_reports = [
        "reports/final_model_comparison/repeated_cv_summary.csv",
        "reports/final_model_comparison/repeated_cv_results.csv",
        "reports/final_model_comparison/repeated_split_summary.csv",
        "reports/final_model_comparison/fixed_test_results.csv",
        "reports/final_model_comparison/final_model_comparison_report.md",
    ]
    missing_reports = [
        relative_path
        for relative_path in required_reports
        if not (PROJECT_ROOT / relative_path).exists()
    ]
    if missing_reports:
        callout(
            "<strong>Detailed result files are not included in this submission.</strong> "
            "The comparison code and fixed model settings remain available in "
            "<code>src/experiments/final_model_comparison.py</code>.",
            "amber",
        )
        st.caption("No metrics are reconstructed or estimated by the dashboard.")
        source_note("src/experiments/final_model_comparison.py")
        footer()
        return

    cv_summary = repeated_cv_summary()
    cv_results = repeated_cv_results()
    split_summary = repeated_split_summary()
    fixed = fixed_model_results()
    table = _model_table(cv_summary)
    winner = table.iloc[0]

    false_pathological = cv_summary[
        (cv_summary["row_type"] == "model_summary")
        & (cv_summary["metric"] == "false_pathological_predictions")
        & (cv_summary["model_or_comparison"] == winner["Model"])
    ]["mean"].iloc[0]

    metric_row(
        [
            (
                "Top-ranked model",
                "SVM" if winner["Model"] == "Support Vector Machine" else str(winner["Model"]),
                "Support Vector Machine — final ranking",
            ),
            ("Repeated-CV Macro F1", decimal(winner["macro_f1"], 4), "Primary ranking metric"),
            (
                "Pathological recall",
                percentage(winner["pathological_recall"]),
                "Mean repeated-CV recall",
            ),
            (
                "Pathological precision",
                percentage(winner["pathological_precision"]),
                "Mean repeated-CV precision",
            ),
            (
                "False Pathological / fold",
                decimal(false_pathological, 2),
                "Mean repeated-CV count",
            ),
        ]
    )

    section_label("The result in plain English")
    summary_columns = st.columns(3, gap="large")
    with summary_columns[0]:
        st.subheader("Best overall score")
        st.write(
            "The **Support Vector Machine** ranked first using the project's "
            "main measure, called Macro F1."
        )
    with summary_columns[1]:
        st.subheader("Important trade-off")
        st.write(
            "**Logistic Regression** found a larger share of the Pathological "
            "cases, but it also produced more false Pathological warnings."
        )
    with summary_columns[2]:
        st.subheader("What that means")
        st.write(
            "No model was best at everything. The preferred model depends on "
            "whether catching more cases or avoiding false alarms matters most."
        )
    callout(
        "<strong>Bottom line:</strong> SVM was the recorded final recommendation. "
        "This is a research result, not proof that the model is safe for clinical use.",
        "teal",
    )

    section_label("Optional detailed results")
    st.caption(
        "You can stop here if you only wanted the main result. The sections "
        "below preserve the detailed evidence used in the honours project."
    )
    st.subheader("Scores across repeated tests")
    metric_label = st.selectbox("Metric", list(METRICS))
    metric = METRICS[metric_label]
    chart_column, distribution_column = st.columns(2, gap="large")
    with chart_column:
        st.pyplot(
            model_metric_figure(
                cv_summary,
                metric,
                f"Mean {metric_label.lower()} ± SD",
            ),
            clear_figure=True,
        )
    with distribution_column:
        st.pyplot(
            repeated_distribution_figure(cv_results, metric),
            clear_figure=True,
        )

    display_columns = [
        "Model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "pathological_recall",
        "pathological_precision",
    ]
    st.dataframe(
        table[display_columns].rename(
            columns={
                "accuracy": "Accuracy",
                "balanced_accuracy": "Balanced accuracy",
                "macro_f1": "Macro F1",
                "weighted_f1": "Weighted F1",
                "pathological_recall": "Pathological recall",
                "pathological_precision": "Pathological precision",
            }
        ).style.format(precision=4),
        hide_index=True,
        width="stretch",
    )

    section_label("One example test split")
    selected_model = st.selectbox(
        "Confusion matrix model",
        fixed["model"].tolist(),
        index=fixed["model"].tolist().index("Support Vector Machine"),
    )
    fixed_row = fixed.loc[fixed["model"] == selected_model].iloc[0]
    matrix_column, metrics_column = st.columns([1, 1.15], gap="large")
    with matrix_column:
        st.pyplot(
            confusion_matrix_figure(
                fixed_row["confusion_matrix"],
                ["Normal", "Pathological", "Suspicious"],
                f"{selected_model} · fixed test",
            ),
            clear_figure=True,
        )
    with metrics_column:
        st.subheader("Fixed-test snapshot")
        fixed_metrics = pd.DataFrame(
            [
                ("Accuracy", fixed_row["accuracy"]),
                ("Balanced accuracy", fixed_row["balanced_accuracy"]),
                ("Macro F1", fixed_row["macro_f1"]),
                ("Normal recall", fixed_row["normal_recall"]),
                ("Suspicious recall", fixed_row["suspicious_recall"]),
                ("Pathological recall", fixed_row["pathological_recall"]),
                ("Pathological precision", fixed_row["pathological_precision"]),
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(
            fixed_metrics.style.format({"Value": "{:.4f}"}),
            hide_index=True,
            width="stretch",
        )
        callout(
            "The fixed test was evaluated once and was <strong>not used</strong> "
            "to choose the recommendation.",
            "amber",
        )

    section_label("Technical comparison with Logistic Regression")
    pairwise_metric = st.selectbox(
        "Pairwise metric",
        [
            "macro_f1",
            "balanced_accuracy",
            "pathological_recall",
            "pathological_precision",
            "false_pathological_predictions",
        ],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    pairwise = cv_summary[
        (cv_summary["row_type"] == "paired_comparison")
        & (cv_summary["metric"] == pairwise_metric)
    ][
        [
            "model_or_comparison",
            "paired_mean_difference",
            "paired_mean_bootstrap_95ci_lower",
            "paired_mean_bootstrap_95ci_upper",
            "improved_count",
            "worsened_count",
            "equal_count",
        ]
    ].copy()
    st.dataframe(
        pairwise.rename(
            columns={
                "model_or_comparison": "Comparison",
                "paired_mean_difference": "Mean difference",
                "paired_mean_bootstrap_95ci_lower": "95% CI lower",
                "paired_mean_bootstrap_95ci_upper": "95% CI upper",
                "improved_count": "Improved folds",
                "worsened_count": "Worsened folds",
                "equal_count": "Equal folds",
            }
        ).style.format(precision=4),
        hide_index=True,
        width="stretch",
    )

    section_label("Full recorded recommendation")
    report = read_markdown(
        "reports/final_model_comparison/final_model_comparison_report.md"
    )
    recommendation = extract_markdown_section(report, "Recommendation")
    with st.expander("Read the full recommendation from the final report"):
        st.markdown(recommendation)
    callout(
        "<strong>Interpret carefully:</strong> the SVM leads the primary "
        "Macro-F1 ranking, while Logistic Regression has higher Pathological "
        "recall. The report does not claim clinical significance.",
        "coral",
    )

    with st.expander("Compare 30-split stability"):
        split_table = _model_table(split_summary)
        st.dataframe(
            split_table.style.format(precision=4),
            hide_index=True,
            width="stretch",
        )

    source_note(
        "reports/final_model_comparison/repeated_cv_results.csv",
        "reports/final_model_comparison/repeated_cv_summary.csv",
        "reports/final_model_comparison/repeated_split_summary.csv",
        "reports/final_model_comparison/fixed_test_results.csv",
        "reports/final_model_comparison/final_model_comparison_report.md",
    )
    footer()
