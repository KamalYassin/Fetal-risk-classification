"""Plot the fixed representative CTG records using shared detector metadata.

Run from the project root::

    python src/features/plot_representative_records.py \
        --before-features /path/to/previous/clinical_features.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extract_clinical_features import (
    FEATURE_COLUMNS,
    RAW_DIR,
    extract_features_for_record,
    find_header_files,
)

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "representative_records"

# Keep these IDs fixed so revisions are compared on the same records.
REPRESENTATIVE_RECORDS = {
    "Normal": "1142",
    "Suspicious": "1363",
    "Pathological": "1158",
}

COMPARISON_COLUMNS = [
    "acceleration_count",
    "deceleration_count",
    "baseline_fhr",
    "contraction_count",
    "max_acceleration_amplitude",
    "max_deceleration_depth",
]


def load_csv_rows(path):
    """Load a CSV keyed by record_id."""
    with Path(path).open("r", newline="", encoding="utf-8") as csv_file:
        return {row["record_id"]: row for row in csv.DictReader(csv_file)}


def find_header_path(record_id):
    """Find the .hea path for one record ID."""
    for header_path in find_header_files(RAW_DIR):
        if header_path.stem == str(record_id):
            return header_path
    raise FileNotFoundError(f"Could not find header file for record {record_id}")


def add_event_spans(axis, time_minutes, events, color, label):
    """Shade detected events using metadata returned by the extractor."""
    for event_number, event in enumerate(events):
        start = event["start_index"]
        end = event["end_index"]
        axis.axvspan(
            time_minutes[start],
            time_minutes[end - 1],
            color=color,
            alpha=0.22,
            label=label if event_number == 0 else None,
        )


def plot_record(class_name, record_id):
    """Create one raw/cleaned FHR and filtered-UC diagnostic plot."""
    features, analysis = extract_features_for_record(
        find_header_path(record_id), return_analysis=True
    )
    sampling_frequency = len(analysis["raw_fhr"]) / (
        features["recording_duration_minutes"] * 60.0
    )
    time_minutes = np.arange(len(analysis["raw_fhr"])) / sampling_frequency / 60.0
    contractions = analysis["contraction_features"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(
        f"{class_name} record {record_id} — FHR quality: "
        f"{features['feature_quality_flag']}",
        fontsize=14,
    )

    axes[0].plot(
        time_minutes,
        analysis["raw_fhr"],
        color="0.75",
        linewidth=0.6,
        label="Raw FHR",
    )
    axes[0].plot(
        time_minutes,
        analysis["cleaned_fhr"],
        color="black",
        linewidth=0.8,
        label="Cleaned FHR",
    )
    axes[0].plot(
        time_minutes,
        analysis["baseline_series"],
        color="tab:blue",
        linewidth=1.4,
        label=f"5-minute moving baseline (median {features['baseline_fhr']:.1f} bpm)",
    )
    artifact_indices = np.flatnonzero(analysis["artifact_mask"])
    if len(artifact_indices):
        axes[0].scatter(
            time_minutes[artifact_indices],
            analysis["raw_fhr"][artifact_indices],
            color="tab:orange",
            marker="x",
            s=14,
            linewidths=0.8,
            label="Possible artifact",
            zorder=4,
        )

    add_event_spans(
        axes[0],
        time_minutes,
        analysis["acceleration_features"]["events"],
        "tab:green",
        "Acceleration",
    )
    add_event_spans(
        axes[0],
        time_minutes,
        analysis["deceleration_features"]["events"],
        "tab:red",
        "Deceleration",
    )
    axes[0].set_ylabel("FHR (bpm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(
        time_minutes,
        contractions["smoothed_uc"],
        color="tab:brown",
        linewidth=1.1,
        label="10-second median-smoothed UC",
    )
    if np.isfinite(contractions["height_threshold"]):
        axes[1].axhline(
            contractions["height_threshold"],
            color="tab:purple",
            linestyle="--",
            linewidth=1.0,
            label="Minimum peak height",
        )
    peaks = np.array(
        [event["peak_index"] for event in contractions["events"]], dtype=int
    )
    if len(peaks):
        axes[1].scatter(
            time_minutes[peaks],
            contractions["smoothed_uc"][peaks],
            color="tab:purple",
            s=28,
            label="Width/prominence-filtered contractions",
            zorder=3,
        )
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylabel("UC")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{class_name.lower()}_{record_id}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path, features


def format_comparison_value(value):
    """Format a CSV/extracted value for a compact comparison table."""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def print_before_after(before_rows, after_rows):
    """Print changes for the three representative records."""
    print("\nBefore-versus-after representative features")
    print("============================================")
    for class_name, record_id in REPRESENTATIVE_RECORDS.items():
        before = before_rows.get(record_id, {})
        after = after_rows[record_id]
        print(f"\n{class_name} record {record_id}")
        print(f"{'feature':36} {'before':>12} {'after':>12}")
        print(f"{'-' * 36} {'-' * 12} {'-' * 12}")
        for column in COMPARISON_COLUMNS:
            print(
                f"{column:36} "
                f"{format_comparison_value(before.get(column, 'n/a')):>12} "
                f"{format_comparison_value(after[column]):>12}"
            )


def print_feature_values(class_name, features):
    """Print the full feature row for validation."""
    print(f"\nCurrent features — {class_name} record {features['record_id']}")
    for column in FEATURE_COLUMNS:
        value = features[column]
        formatted = str(value) if column == "record_id" else format_comparison_value(value)
        print(f"{column}: {formatted}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--before-features",
        type=Path,
        help="Optional previous clinical_features.csv used for before/after output.",
    )
    return parser.parse_args()


def main():
    """Regenerate the same three plots and optionally compare old features."""
    args = parse_args()
    after_rows = {}
    for class_name, record_id in REPRESENTATIVE_RECORDS.items():
        output_path, features = plot_record(class_name, record_id)
        after_rows[record_id] = features
        print(f"Saved plot: {output_path}")

    if args.before_features:
        print_before_after(load_csv_rows(args.before_features), after_rows)

    for class_name, record_id in REPRESENTATIVE_RECORDS.items():
        print_feature_values(class_name, after_rows[record_id])


if __name__ == "__main__":
    main()
