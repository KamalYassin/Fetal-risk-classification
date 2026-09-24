"""Matplotlib charts used across dashboard pages."""

from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#15324b"
TEAL = "#0f8b8d"
MINT = "#75c9bd"
CORAL = "#ef6f61"
AMBER = "#e9a23b"
SLATE = "#718096"
CLASS_COLORS = {
    "Normal": TEAL,
    "Suspicious": AMBER,
    "Pathological": CORAL,
}


def _style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#ccd8e3")
    axis.tick_params(colors="#52667a")
    axis.grid(axis="y", alpha=0.16, color=SLATE)
    axis.set_axisbelow(True)


def class_distribution_figure(labels: pd.Series):
    """Plot the outcome-derived class balance."""
    order = ["Normal", "Suspicious", "Pathological"]
    counts = labels.value_counts().reindex(order, fill_value=0)
    figure, axis = plt.subplots(figsize=(8.4, 4.3))
    bars = axis.bar(
        counts.index,
        counts.values,
        color=[CLASS_COLORS[label] for label in counts.index],
        width=0.62,
    )
    axis.bar_label(bars, padding=5, color=NAVY, fontweight="bold")
    axis.set_ylabel("Patients")
    axis.set_title("Outcome-derived class distribution", loc="left", weight="bold")
    axis.set_ylim(0, max(counts.values) * 1.18)
    _style_axis(axis)
    figure.tight_layout()
    return figure


def model_metric_figure(summary: pd.DataFrame, metric: str, title: str):
    """Plot one repeated-evaluation metric for all five models."""
    rows = summary[
        (summary["row_type"] == "model_summary")
        & (summary["metric"] == metric)
    ].copy()
    rows = rows.sort_values("mean", ascending=True)
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    colors = [TEAL if name == "Support Vector Machine" else NAVY for name in rows["model_or_comparison"]]
    axis.barh(rows["model_or_comparison"], rows["mean"], color=colors, alpha=0.92)
    if "standard_deviation" in rows:
        axis.errorbar(
            rows["mean"],
            rows["model_or_comparison"],
            xerr=rows["standard_deviation"],
            fmt="none",
            ecolor="#8fa2b4",
            capsize=3,
            linewidth=1,
        )
    for y, value in enumerate(rows["mean"]):
        axis.text(value + 0.006, y, f"{value:.3f}", va="center", color=NAVY, fontsize=9)
    axis.set_xlabel("Mean across repeated CV folds")
    axis.set_title(title, loc="left", weight="bold")
    axis.set_xlim(0, max(0.60, float(rows["mean"].max() + 0.08)))
    _style_axis(axis)
    axis.grid(axis="x", alpha=0.16)
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    return figure


def repeated_distribution_figure(results: pd.DataFrame, metric: str):
    """Plot fold-level distributions for one model metric."""
    model_order = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Support Vector Machine",
    ]
    groups = [
        results.loc[results["model"] == model, metric].dropna().to_numpy()
        for model in model_order
    ]
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    box = axis.boxplot(groups, patch_artist=True, tick_labels=model_order)
    for patch, model in zip(box["boxes"], model_order):
        patch.set_facecolor(TEAL if model == "Support Vector Machine" else "#cbd8e5")
        patch.set_alpha(0.9)
    for median in box["medians"]:
        median.set_color(CORAL)
        median.set_linewidth(2)
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title("Fold-level distribution", loc="left", weight="bold")
    axis.tick_params(axis="x", rotation=18)
    _style_axis(axis)
    figure.tight_layout()
    return figure


def confusion_matrix_figure(raw_matrix: str, labels: list[str], title: str):
    """Render a confusion matrix stored as a CSV string."""
    matrix = np.asarray(literal_eval(raw_matrix), dtype=int)
    figure, axis = plt.subplots(figsize=(5.8, 5.0))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > matrix.max() * 0.52 else NAVY
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color, weight="bold")
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title, loc="left", weight="bold")
    figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    figure.tight_layout()
    return figure


def probability_figure(probabilities: dict[str, float]):
    """Plot prediction probabilities in clinical class order."""
    order = ["Normal", "Suspicious", "Pathological"]
    values = [probabilities[label] for label in order]
    figure, axis = plt.subplots(figsize=(8.2, 3.3))
    bars = axis.barh(
        order[::-1],
        values[::-1],
        color=[CLASS_COLORS[label] for label in order[::-1]],
        height=0.58,
    )
    for bar, value in zip(bars, values[::-1]):
        axis.text(
            min(value + 0.018, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            weight="bold",
            color=NAVY,
        )
    axis.set_xlim(0, 1)
    axis.set_xlabel("Model probability")
    axis.set_title("Prediction profile", loc="left", weight="bold")
    _style_axis(axis)
    axis.grid(axis="x", alpha=0.16)
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    return figure


def contribution_figure(
    contributions: pd.DataFrame,
    predicted_label: str,
    runner_up_label: str,
    top_n: int = 10,
):
    """Plot the largest feature contributions to the logit difference."""
    selected = (
        contributions.nlargest(top_n, "absolute_contribution")
        .sort_values("contribution")
        .copy()
    )
    selected["display_feature"] = (
        selected["feature"].str.replace("_", " ").str.replace("  ", " ").str.title()
    )
    colors = np.where(selected["contribution"] >= 0, TEAL, CORAL)
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.barh(
        selected["display_feature"],
        selected["contribution"],
        color=colors,
        alpha=0.92,
    )
    axis.axvline(0, color=NAVY, linewidth=0.9)
    axis.set_xlabel(
        f"Contribution to {predicted_label} vs {runner_up_label} logit"
    )
    axis.set_title("Largest model contributions", loc="left", weight="bold")
    axis.grid(axis="x", alpha=0.16)
    axis.grid(axis="y", visible=False)
    _style_axis(axis)
    figure.tight_layout()
    return figure


def ctg_figure(record_data: dict, start_minute: float, end_minute: float):
    """Plot raw/cleaned FHR, baseline, events, and smoothed UC."""
    frequency = record_data["sampling_frequency"]
    analysis = record_data["analysis"]
    raw_uc = np.asarray(record_data["raw_uc"], dtype=float)
    time_minutes = np.arange(len(raw_uc)) / frequency / 60.0
    visible = (time_minutes >= start_minute) & (time_minutes <= end_minute)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1]},
    )
    fhr_axis, uc_axis = axes
    fhr_axis.plot(
        time_minutes[visible],
        np.asarray(analysis["raw_fhr"])[visible],
        color="#b7c3ce",
        linewidth=0.65,
        label="Raw FHR",
    )
    fhr_axis.plot(
        time_minutes[visible],
        np.asarray(analysis["cleaned_fhr"])[visible],
        color=NAVY,
        linewidth=0.85,
        label="Cleaned FHR",
    )
    fhr_axis.plot(
        time_minutes[visible],
        np.asarray(analysis["baseline_series"])[visible],
        color=TEAL,
        linewidth=1.7,
        label="Moving baseline",
    )

    first_acceleration = True
    for event in analysis["acceleration_features"]["events"]:
        start, end = event["start_seconds"] / 60, event["end_seconds"] / 60
        if end >= start_minute and start <= end_minute:
            fhr_axis.axvspan(
                start,
                end,
                color=MINT,
                alpha=0.34,
                label="Acceleration" if first_acceleration else None,
            )
            first_acceleration = False
    first_deceleration = True
    for event in analysis["deceleration_features"]["events"]:
        start, end = event["start_seconds"] / 60, event["end_seconds"] / 60
        if end >= start_minute and start <= end_minute:
            fhr_axis.axvspan(
                start,
                end,
                color=CORAL,
                alpha=0.25,
                label="Deceleration" if first_deceleration else None,
            )
            first_deceleration = False

    smoothed_uc = np.asarray(
        analysis["contraction_features"]["smoothed_uc"], dtype=float
    )
    uc_axis.plot(time_minutes[visible], raw_uc[visible], color="#d7b982", linewidth=0.6, label="Raw UC")
    uc_axis.plot(time_minutes[visible], smoothed_uc[visible], color=AMBER, linewidth=1.15, label="Smoothed UC")
    contraction_peaks = [
        event["peak_index"]
        for event in analysis["contraction_features"]["events"]
        if start_minute <= event["peak_index"] / frequency / 60 <= end_minute
    ]
    if contraction_peaks:
        uc_axis.scatter(
            time_minutes[contraction_peaks],
            smoothed_uc[contraction_peaks],
            color=CORAL,
            edgecolor="white",
            linewidth=0.6,
            s=30,
            zorder=4,
            label="Detected contraction",
        )

    fhr_axis.set_ylabel("FHR (bpm)")
    uc_axis.set_ylabel("UC (dataset units)")
    uc_axis.set_xlabel("Time (minutes)")
    fhr_axis.set_title(
        f"CTG record {record_data['record_id']}", loc="left", weight="bold"
    )
    for axis in axes:
        axis.set_xlim(start_minute, end_minute)
        _style_axis(axis)
        axis.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    figure.tight_layout()
    return figure
