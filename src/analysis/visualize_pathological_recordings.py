"""Create visual, descriptive diagnostics for all Pathological CTG records.

This script does not train a model, change labels, alter the feature extractor,
or create a window-level machine-learning dataset.  It reuses the current FHR,
baseline, event, variability, and contraction helpers to inspect non-overlapping
20-minute windows.  Window flags are exploratory data-distribution comparisons,
not clinical diagnoses.

Run from the project root::

    python src/analysis/visualize_pathological_recordings.py
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import extract_clinical_features as extractor
from src.models import train_models
from src.utils.record_ids import normalize_record_id

# ---------------------------------------------------------------------------
# Input paths and diagnostic settings
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.csv"
CLINICAL_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"
ML_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
PATHOLOGICAL_AUDIT_PATH = PROJECT_ROOT / "reports" / "pathological_case_audit.csv"

OUTPUT_ROOT = PROJECT_ROOT / "reports" / "pathological_visual_review"
RECORD_FIGURE_DIR = OUTPUT_ROOT / "records"
SUMMARY_FIGURE_DIR = OUTPUT_ROOT / "summary"
WINDOW_CSV_PATH = OUTPUT_ROOT / "pathological_window_diagnostics.csv"
REPORT_PATH = OUTPUT_ROOT / "pathological_visual_review.md"

# Non-overlapping window length used in the plots.
WINDOW_MINUTES = 20.0

# Exploratory tails are defined relative to the distribution of all valid
# 20-minute windows from all 552 records.  These percentiles are not medical
# cutoffs and must not be interpreted as diagnostic thresholds.
EXPLORATORY_LOW_PERCENTILE = 10.0
EXPLORATORY_HIGH_PERCENTILE = 90.0

# A window needs at least this much valid cleaned FHR for variability and event
WINDOW_MIN_VALID_FHR_PERCENTAGE = extractor.SEGMENT_MIN_VALID_FHR_PERCENTAGE

# To decide whether whole-record averaging looks milder, compare the record's
MILD_FULL_RECORD_COMPARISON_FEATURES = (
    "short_term_variability",
    "bradycardia_percentage",
)

# Plotting settings affect presentation only.
FIGURE_DPI = 220
SIGNAL_LINE_WIDTH = 0.7
BASELINE_LINE_WIDTH = 1.3

WINDOW_COLUMNS = [
    "record_id",
    "split",
    "window_number",
    "start_minute",
    "end_minute",
    "is_final_window",
    "window_position",
    "valid_fhr_percentage",
    "missing_fhr_percentage",
    "artifact_percentage",
    "baseline_fhr",
    "short_term_variability",
    "long_term_variability",
    "acceleration_count",
    "deceleration_count",
    "max_deceleration_depth",
    "longest_deceleration_duration_seconds",
    "bradycardia_percentage",
    "tachycardia_percentage",
    "contraction_count",
    "stored_predicted_class",
    "stored_pathological_probability",
    "low_stv_exploratory_flag",
    "deep_deceleration_exploratory_flag",
    "high_bradycardia_exploratory_flag",
    "long_deceleration_exploratory_flag",
    "poor_signal_exploratory_flag",
    "local_abnormality_exploratory_flag",
    "any_exploratory_flag",
]


@dataclass
class RecordBundle:
    """Raw record plus the shared extractor analysis and window summaries."""

    record_id: str
    sampling_frequency: float
    raw_fhr: np.ndarray
    uc: np.ndarray
    analysis: dict
    windows: list[dict]


def load_and_validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load project tables and verify one-to-one record alignment."""
    required_paths = [LABELS_PATH, CLINICAL_FEATURES_PATH, ML_DATASET_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))

    labels = pd.read_csv(LABELS_PATH)
    features = pd.read_csv(CLINICAL_FEATURES_PATH)
    dataset = pd.read_csv(ML_DATASET_PATH)
    audit = (
        pd.read_csv(PATHOLOGICAL_AUDIT_PATH)
        if PATHOLOGICAL_AUDIT_PATH.exists()
        else pd.DataFrame()
    )

    for frame, name in (
        (labels, "labels"),
        (features, "clinical_features"),
        (dataset, "ml_dataset"),
    ):
        if "record_id" not in frame:
            raise ValueError(f"{name} is missing record_id")
        frame["record_id"] = normalize_record_id(frame["record_id"])
        if frame["record_id"].duplicated().any():
            duplicates = frame.loc[frame["record_id"].duplicated(), "record_id"].tolist()
            raise ValueError(f"{name} contains duplicate record IDs: {duplicates}")

    if not audit.empty:
        audit["record_id"] = normalize_record_id(audit["record_id"])
        if audit["record_id"].duplicated().any():
            raise ValueError("pathological_case_audit.csv contains duplicate record IDs")

    label_ids = set(labels["record_id"])
    for frame, name in ((features, "clinical_features"), (dataset, "ml_dataset")):
        if set(frame["record_id"]) != label_ids:
            missing_ids = sorted(label_ids - set(frame["record_id"]))
            extra_ids = sorted(set(frame["record_id"]) - label_ids)
            raise ValueError(
                f"{name} record alignment failed; missing={missing_ids}, extra={extra_ids}"
            )

    dataset_labels = dataset.set_index("record_id")["label"]
    stored_labels = labels.set_index("record_id")["label"]
    disagreements = dataset_labels[dataset_labels != stored_labels]
    if len(disagreements):
        raise ValueError(
            "Stored labels disagree between labels.csv and ml_dataset.csv for: "
            + ", ".join(disagreements.index.tolist())
        )
    return labels, features, dataset, audit


def reproduce_fixed_split(dataset: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Reproduce the row order, encoding, and 80/20 split from training."""
    model_features, encoded_target, _ = train_models.prepare_features_and_target(
        dataset.copy()
    )
    train_features, test_features, _, _ = train_models.split_dataset(
        model_features, encoded_target
    )
    train_ids = set(dataset.loc[train_features.index, "record_id"])
    test_ids = set(dataset.loc[test_features.index, "record_id"])
    if train_ids & test_ids or len(train_ids | test_ids) != len(dataset):
        raise RuntimeError("The reproduced fixed split is not a complete partition")
    return train_ids, test_ids


def find_waveform_directory(expected_record_ids: Iterable[str]) -> Path:
    """Locate the existing CTU-UHB WFDB directory without copying its files."""
    candidates = [
        extractor.RAW_DIR,
        WORKSPACE_ROOT / "dataset" / "raw",
        WORKSPACE_ROOT
        / "dataset"
        / "raw"
        / "ctu-chb-intrapartum-cardiotocography-database-1.0.0",
    ]
    expected = set(expected_record_ids)
    for candidate in candidates:
        if not candidate.exists():
            continue
        direct_ids = {path.stem for path in candidate.glob("*.hea")}
        if expected.issubset(direct_ids):
            return candidate
        for child in candidate.iterdir():
            if child.is_dir():
                child_ids = {path.stem for path in child.glob("*.hea")}
                if expected.issubset(child_ids):
                    return child
    raise FileNotFoundError(
        "Could not find a WFDB directory containing every required record. "
        "Checked data/raw and the surrounding honours-project dataset/raw paths."
    )


def ensure_new_output_location() -> None:
    """Refuse to overwrite any existing report file."""
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(
            f"{OUTPUT_ROOT} already contains files. Move or rename the existing "
            "report before rerunning; this script will not overwrite it."
        )
    RECORD_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def event_slice(events: list[dict], start: int, end: int) -> list[dict]:
    """Assign full-record events to the window containing their start sample."""
    return [event for event in events if start <= event["start_index"] < end]


def contraction_slice(events: list[dict], start: int, end: int) -> list[dict]:
    """Assign contractions to the window containing their peak sample."""
    return [event for event in events if start <= event["peak_index"] < end]


def calculate_window_rows(
    record_id: str,
    analysis: dict,
    sampling_frequency: float,
    split: str,
    predicted_class: str | None,
    pathological_probability: float | None,
) -> list[dict]:
    """Calculate descriptive summaries over non-overlapping 20-minute windows."""
    fhr = analysis["cleaned_fhr"]
    baseline = analysis["baseline_series"]
    raw_missing = analysis["raw_missing_mask"]
    artifact = analysis["artifact_mask"]
    acceleration_events = analysis["acceleration_features"]["events"]
    deceleration_events = analysis["deceleration_features"]["events"]
    contraction_events = analysis["contraction_features"]["events"]

    window_samples = max(1, round(WINDOW_MINUTES * 60 * sampling_frequency))
    window_bounds = [
        (start, min(start + window_samples, len(fhr)))
        for start in range(0, len(fhr), window_samples)
    ]
    rows = []
    for position, (start, end) in enumerate(window_bounds):
        fhr_window = fhr[start:end]
        baseline_window = baseline[start:end]
        valid_percentage = 100.0 * float(np.mean(np.isfinite(fhr_window)))
        missing_percentage = 100.0 * float(np.mean(raw_missing[start:end]))
        artifact_percentage = 100.0 * float(np.mean(artifact[start:end]))
        accelerations = event_slice(acceleration_events, start, end)
        decelerations = event_slice(deceleration_events, start, end)
        contractions = contraction_slice(contraction_events, start, end)
        threshold_exposure = extractor.calculate_fhr_threshold_exposure(
            fhr_window, sampling_frequency
        )

        if valid_percentage >= WINDOW_MIN_VALID_FHR_PERCENTAGE:
            stv = extractor.calculate_short_term_variability(fhr_window)
            ltv = extractor.calculate_long_term_variability(
                fhr_window, sampling_frequency
            )
            baseline_fhr = extractor.safe_nan_stat(baseline_window, np.median)
            max_depth = (
                max(event["amplitude_or_depth"] for event in decelerations)
                if decelerations
                else 0.0
            )
            longest_deceleration = (
                max(event["duration_seconds"] for event in decelerations)
                if decelerations
                else 0.0
            )
            acceleration_count = len(accelerations)
            deceleration_count = len(decelerations)
        else:
            # Events may have starts in a poor-quality window, but values are
            # withheld so such windows do not masquerade as normal observations.
            stv = np.nan
            ltv = np.nan
            baseline_fhr = np.nan
            max_depth = np.nan
            longest_deceleration = np.nan
            acceleration_count = np.nan
            deceleration_count = np.nan

        if position == 0:
            window_position = "early"
        elif position == len(window_bounds) - 1:
            window_position = "final"
        else:
            window_position = "middle"

        rows.append(
            {
                "record_id": record_id,
                "split": split,
                "window_number": position + 1,
                "start_minute": start / sampling_frequency / 60.0,
                "end_minute": end / sampling_frequency / 60.0,
                "is_final_window": position == len(window_bounds) - 1,
                "window_position": window_position,
                "valid_fhr_percentage": valid_percentage,
                "missing_fhr_percentage": missing_percentage,
                "artifact_percentage": artifact_percentage,
                "baseline_fhr": baseline_fhr,
                "short_term_variability": stv,
                "long_term_variability": ltv,
                "acceleration_count": acceleration_count,
                "deceleration_count": deceleration_count,
                "max_deceleration_depth": max_depth,
                "longest_deceleration_duration_seconds": longest_deceleration,
                "bradycardia_percentage": threshold_exposure[
                    "bradycardia_percentage"
                ],
                "tachycardia_percentage": threshold_exposure[
                    "tachycardia_percentage"
                ],
                "contraction_count": len(contractions),
                "stored_predicted_class": predicted_class,
                "stored_pathological_probability": pathological_probability,
            }
        )
    return rows


def load_record_bundle(
    record_id: str,
    waveform_directory: Path,
    split: str,
    audit_lookup: pd.DataFrame,
) -> RecordBundle:
    """Load one WFDB record and run the shared extractor analysis once."""
    record = extractor.load_record(waveform_directory / f"{record_id}.hea")
    sampling_frequency = float(record.fs)
    raw_fhr = extractor.get_signal_by_name(record, "FHR")
    uc = extractor.get_signal_by_name(record, "UC")
    analysis = extractor.analyze_signals(raw_fhr, uc, sampling_frequency)

    predicted_class = None
    pathological_probability = None
    if record_id in audit_lookup.index:
        audit_row = audit_lookup.loc[record_id]
        predicted_class = audit_row.get("predicted_class")
        pathological_probability = audit_row.get(
            "predicted_probability_pathological"
        )
        if pd.isna(predicted_class):
            predicted_class = None
        if pd.isna(pathological_probability):
            pathological_probability = None

    windows = calculate_window_rows(
        record_id,
        analysis,
        sampling_frequency,
        split,
        predicted_class,
        pathological_probability,
    )
    return RecordBundle(
        record_id=record_id,
        sampling_frequency=sampling_frequency,
        raw_fhr=raw_fhr,
        uc=uc,
        analysis=analysis,
        windows=windows,
    )


def build_reference_windows(
    record_ids: list[str], waveform_directory: Path
) -> pd.DataFrame:
    """Build the all-record reference distribution used only for tail flags."""
    reference_rows = []
    for index, record_id in enumerate(record_ids, start=1):
        record = extractor.load_record(waveform_directory / f"{record_id}.hea")
        sampling_frequency = float(record.fs)
        raw_fhr = extractor.get_signal_by_name(record, "FHR")
        uc = extractor.get_signal_by_name(record, "UC")
        analysis = extractor.analyze_signals(raw_fhr, uc, sampling_frequency)
        reference_rows.extend(
            calculate_window_rows(
                record_id,
                analysis,
                sampling_frequency,
                split="reference_only",
                predicted_class=None,
                pathological_probability=None,
            )
        )
        if index % 100 == 0 or index == len(record_ids):
            print(f"  Reference waveforms processed: {index}/{len(record_ids)}")
    return pd.DataFrame(reference_rows)


def finite_percentile(values: pd.Series, percentile: float) -> float:
    """Return a finite percentile or raise when a reference metric is empty."""
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        raise ValueError(f"No finite values available for percentile {percentile}")
    return float(np.percentile(numeric, percentile))


def calculate_exploratory_thresholds(reference: pd.DataFrame) -> dict[str, float]:
    """Calculate non-clinical tail thresholds from all valid reference windows."""
    eligible = reference[
        reference["valid_fhr_percentage"] >= WINDOW_MIN_VALID_FHR_PERCENTAGE
    ].copy()
    invalid_percentage = (
        reference["missing_fhr_percentage"] + reference["artifact_percentage"]
    )
    return {
        "low_stv": finite_percentile(
            eligible["short_term_variability"], EXPLORATORY_LOW_PERCENTILE
        ),
        "deep_deceleration": finite_percentile(
            eligible["max_deceleration_depth"], EXPLORATORY_HIGH_PERCENTILE
        ),
        "high_bradycardia": finite_percentile(
            eligible["bradycardia_percentage"], EXPLORATORY_HIGH_PERCENTILE
        ),
        "long_deceleration": finite_percentile(
            eligible["longest_deceleration_duration_seconds"],
            EXPLORATORY_HIGH_PERCENTILE,
        ),
        "poor_signal": finite_percentile(
            invalid_percentage, EXPLORATORY_HIGH_PERCENTILE
        ),
    }


def apply_exploratory_flags(
    windows: pd.DataFrame, thresholds: dict[str, float]
) -> pd.DataFrame:
    """Add exploratory flags; these are not window labels."""
    flagged = windows.copy()
    eligible = flagged["valid_fhr_percentage"] >= WINDOW_MIN_VALID_FHR_PERCENTAGE
    invalid_percentage = (
        flagged["missing_fhr_percentage"] + flagged["artifact_percentage"]
    )
    flagged["low_stv_exploratory_flag"] = (
        eligible
        & flagged["short_term_variability"].notna()
        & (flagged["short_term_variability"] <= thresholds["low_stv"])
    )
    flagged["deep_deceleration_exploratory_flag"] = (
        eligible
        & flagged["max_deceleration_depth"].notna()
        & (
            flagged["max_deceleration_depth"]
            >= thresholds["deep_deceleration"]
        )
    )
    flagged["high_bradycardia_exploratory_flag"] = (
        eligible
        & flagged["bradycardia_percentage"].notna()
        & (flagged["bradycardia_percentage"] >= thresholds["high_bradycardia"])
    )
    flagged["long_deceleration_exploratory_flag"] = (
        eligible
        & flagged["longest_deceleration_duration_seconds"].notna()
        & (
            flagged["longest_deceleration_duration_seconds"]
            >= thresholds["long_deceleration"]
        )
    )
    flagged["poor_signal_exploratory_flag"] = (
        invalid_percentage >= thresholds["poor_signal"]
    )
    local_abnormality_columns = [
        "low_stv_exploratory_flag",
        "deep_deceleration_exploratory_flag",
        "high_bradycardia_exploratory_flag",
        "long_deceleration_exploratory_flag",
    ]
    flagged["local_abnormality_exploratory_flag"] = flagged[
        local_abnormality_columns
    ].any(axis=1)
    flagged["any_exploratory_flag"] = (
        flagged["local_abnormality_exploratory_flag"]
        | flagged["poor_signal_exploratory_flag"]
    )
    return flagged


def contiguous_true_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open bounds for contiguous True runs."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    transitions = np.diff(padded.astype(np.int8))
    return list(
        zip(
            np.flatnonzero(transitions == 1),
            np.flatnonzero(transitions == -1),
        )
    )


def shade_regions(
    axis: plt.Axes,
    mask: np.ndarray,
    sampling_frequency: float,
    color: str,
    alpha: float,
) -> None:
    """Shade missing/artifact runs without connecting signal across gaps."""
    for start, end in contiguous_true_regions(mask):
        axis.axvspan(
            start / sampling_frequency / 60.0,
            end / sampling_frequency / 60.0,
            color=color,
            alpha=alpha,
            linewidth=0,
        )


def plot_event_regions(
    axis: plt.Axes,
    events: list[dict],
    sampling_frequency: float,
    color: str,
) -> None:
    """Highlight retained continuous events using their stored sample bounds."""
    for event in events:
        axis.axvspan(
            event["start_index"] / sampling_frequency / 60.0,
            event["end_index"] / sampling_frequency / 60.0,
            color=color,
            alpha=0.22,
            linewidth=0,
        )


def add_window_boundaries(
    axes: Iterable[plt.Axes], duration_minutes: float
) -> None:
    """Add common 20-minute boundaries to signal panels."""
    for boundary in np.arange(WINDOW_MINUTES, duration_minutes, WINDOW_MINUTES):
        for axis in axes:
            axis.axvline(boundary, color="#666666", linestyle="--", linewidth=0.8)


def format_number(value: object, digits: int = 1) -> str:
    """Format table/title values, preserving explicit unavailable values."""
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, (float, np.floating, int, np.integer)):
        return f"{float(value):.{digits}f}"
    return str(value)


def create_record_figure(
    bundle: RecordBundle,
    window_rows: pd.DataFrame,
    clinical_row: pd.Series,
    audit_row: pd.Series | None,
) -> Path:
    """Create the four-panel high-resolution visual review."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fs = bundle.sampling_frequency
    analysis = bundle.analysis
    cleaned_fhr = analysis["cleaned_fhr"]
    duration_minutes = len(cleaned_fhr) / fs / 60.0
    time_minutes = np.arange(len(cleaned_fhr)) / fs / 60.0

    figure = plt.figure(figsize=(17, 12))
    grid = figure.add_gridspec(
        4, 1, height_ratios=[3.4, 2.0, 0.65, 2.45], hspace=0.32
    )
    fhr_axis = figure.add_subplot(grid[0])
    uc_axis = figure.add_subplot(grid[1], sharex=fhr_axis)
    quality_axis = figure.add_subplot(grid[2], sharex=fhr_axis)
    table_axis = figure.add_subplot(grid[3])

    # Raw is retained in light gray for context; cleaned FHR is the signal used
    # by every baseline, variability, and event calculation.
    raw_for_plot = np.asarray(bundle.raw_fhr, dtype=float).copy()
    raw_for_plot[analysis["raw_missing_mask"]] = np.nan
    fhr_axis.plot(
        time_minutes,
        raw_for_plot,
        color="#b8b8b8",
        linewidth=0.45,
        alpha=0.55,
        label="Raw FHR",
    )
    fhr_axis.plot(
        time_minutes,
        cleaned_fhr,
        color="#1f4e79",
        linewidth=SIGNAL_LINE_WIDTH,
        label="Cleaned FHR",
    )
    fhr_axis.plot(
        time_minutes,
        analysis["baseline_series"],
        color="#111111",
        linewidth=BASELINE_LINE_WIDTH,
        label="Moving baseline",
    )
    plot_event_regions(
        fhr_axis,
        analysis["acceleration_features"]["events"],
        fs,
        "#2ca02c",
    )
    plot_event_regions(
        fhr_axis,
        analysis["deceleration_features"]["events"],
        fs,
        "#d62728",
    )
    shade_regions(
        fhr_axis, analysis["raw_missing_mask"], fs, color="#7f7f7f", alpha=0.18
    )
    shade_regions(
        fhr_axis, analysis["artifact_mask"], fs, color="#ffbf00", alpha=0.23
    )
    fhr_axis.set_ylabel("FHR (bpm)")
    fhr_axis.set_title("Fetal heart rate, moving baseline, and retained events")
    fhr_axis.set_xlim(0, duration_minutes)
    fhr_axis.legend(
        handles=[
            Line2D([0], [0], color="#b8b8b8", lw=1.2, label="Raw FHR"),
            Line2D([0], [0], color="#1f4e79", lw=1.5, label="Cleaned FHR"),
            Line2D([0], [0], color="#111111", lw=1.5, label="Moving baseline"),
            Patch(facecolor="#2ca02c", alpha=0.25, label="Acceleration"),
            Patch(facecolor="#d62728", alpha=0.25, label="Deceleration"),
            Patch(facecolor="#7f7f7f", alpha=0.2, label="Missing"),
            Patch(facecolor="#ffbf00", alpha=0.25, label="Artifact/invalid"),
        ],
        ncol=7,
        loc="upper center",
        fontsize=8,
    )

    smoothed_uc = analysis["contraction_features"]["smoothed_uc"]
    uc_axis.plot(
        time_minutes,
        bundle.uc,
        color="#b9a2cf",
        linewidth=0.45,
        alpha=0.55,
        label="Raw UC",
    )
    uc_axis.plot(
        time_minutes,
        smoothed_uc,
        color="#6a3d9a",
        linewidth=0.9,
        label="Smoothed UC",
    )
    contraction_events = analysis["contraction_features"]["events"]
    if contraction_events:
        peak_indices = np.array(
            [event["peak_index"] for event in contraction_events], dtype=int
        )
        uc_axis.scatter(
            peak_indices / fs / 60.0,
            smoothed_uc[peak_indices],
            color="#e31a1c",
            marker="v",
            s=30,
            label="Contraction peak",
            zorder=4,
        )
    uc_axis.set_ylabel("UC (dataset units)")
    uc_axis.set_title("Smoothed uterine activity and detected contraction peaks")
    uc_axis.legend(loc="upper right", ncol=3, fontsize=8)

    status = np.full(len(cleaned_fhr), 2, dtype=int)
    status[analysis["artifact_mask"]] = 1
    status[analysis["raw_missing_mask"]] = 0
    quality_axis.imshow(
        status[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        extent=[0, duration_minutes, 0, 1],
        cmap=matplotlib.colors.ListedColormap(["#777777", "#ffbf00", "#2ca02c"]),
        vmin=0,
        vmax=2,
    )
    quality_axis.set_yticks([])
    quality_axis.set_ylabel("Quality")
    quality_axis.set_xlabel("Time (minutes)")
    quality_axis.legend(
        handles=[
            Patch(facecolor="#2ca02c", label="Valid"),
            Patch(facecolor="#777777", label="Missing"),
            Patch(facecolor="#ffbf00", label="Artifact/invalid"),
        ],
        ncol=3,
        loc="center right",
        fontsize=8,
    )

    add_window_boundaries((fhr_axis, uc_axis, quality_axis), duration_minutes)

    table_axis.axis("off")
    display_columns = [
        "window_number",
        "valid_fhr_percentage",
        "short_term_variability",
        "long_term_variability",
        "deceleration_count",
        "max_deceleration_depth",
        "bradycardia_percentage",
    ]
    display_headers = [
        "Window",
        "Valid FHR %",
        "STV*",
        "LTV*",
        "Decel count",
        "Max decel depth",
        "Bradycardia %",
    ]
    cell_text = []
    row_colors = []
    for _, row in window_rows.iterrows():
        cell_text.append(
            [
                int(row["window_number"]),
                format_number(row["valid_fhr_percentage"]),
                format_number(row["short_term_variability"], 2),
                format_number(row["long_term_variability"], 2),
                format_number(row["deceleration_count"], 0),
                format_number(row["max_deceleration_depth"], 1),
                format_number(row["bradycardia_percentage"], 1),
            ]
        )
        if row["any_exploratory_flag"]:
            row_colors.append(["#fff1cc"] * len(display_columns))
        else:
            row_colors.append(["#ffffff"] * len(display_columns))
    table = table_axis.table(
        cellText=cell_text,
        colLabels=display_headers,
        cellColours=row_colors,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.03, 0.15, 0.94, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for column in range(len(display_headers)):
        table[(0, column)].set_facecolor("#d9eaf7")
        table[(0, column)].set_text_props(weight="bold")
    table_axis.set_title(
        "20-minute window diagnostics "
        "(yellow = at least one exploratory tail/quality flag)",
        fontsize=12,
        pad=8,
    )
    table_axis.text(
        0.5,
        0.03,
        "* STV and LTV are the extractor's computational approximations, "
        "not clinical gold-standard measurements.",
        transform=table_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

    predicted_class = (
        audit_row.get("predicted_class")
        if audit_row is not None
        else None
    )
    pathological_probability = (
        audit_row.get("predicted_probability_pathological")
        if audit_row is not None
        else None
    )
    split = window_rows.iloc[0]["split"]
    fixed_test_label = ""
    if split == "test":
        if predicted_class is not None and predicted_class != "Pathological":
            fixed_test_label = "FIXED TEST – MISSED PATHOLOGICAL CASE"
        elif predicted_class == "Pathological":
            fixed_test_label = "FIXED TEST – CORRECTLY CLASSIFIED PATHOLOGICAL CASE"

    title = (
        f"Record {bundle.record_id} | {split.upper()}"
        f"{chr(10) + fixed_test_label if fixed_test_label else ''}\n"
        f"pH {format_number(clinical_row['pH'], 2)} | "
        f"BE {format_number(clinical_row['BE'], 2)} | "
        f"BDecf {format_number(clinical_row['BDecf'], 2)} | "
        f"Apgar5 {format_number(clinical_row['Apgar5'], 0)} | "
        f"Predicted {format_number(predicted_class)} | "
        f"P(Pathological) {format_number(pathological_probability, 3)} | "
        f"Quality {format_number(clinical_row['feature_quality_flag'])}"
    )
    figure.suptitle(title, fontsize=13, fontweight="bold", y=0.997)
    output_path = RECORD_FIGURE_DIR / (
        f"record_{bundle.record_id}_pathological_review.png"
    )
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return output_path


def record_level_summary(
    windows: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Aggregate the Pathological window table for plots and reporting."""
    grouped = windows.groupby("record_id", sort=True)
    summary = grouped.agg(
        split=("split", "first"),
        valid_fhr_percentage=("valid_fhr_percentage", "mean"),
        missing_fhr_percentage=("missing_fhr_percentage", "mean"),
        artifact_percentage=("artifact_percentage", "mean"),
        maximum_window_deceleration_depth=("max_deceleration_depth", "max"),
        minimum_window_stv=("short_term_variability", "min"),
        maximum_window_bradycardia_percentage=("bradycardia_percentage", "max"),
        flagged_window_count=("local_abnormality_exploratory_flag", "sum"),
        poor_quality_window_count=("poor_signal_exploratory_flag", "sum"),
        final_window_flagged=(
            "local_abnormality_exploratory_flag",
            lambda values: bool(values.iloc[-1]),
        ),
    ).reset_index()
    full = features.copy()
    full["record_id"] = normalize_record_id(full["record_id"])
    summary = summary.merge(
        full[
            [
                "record_id",
                "short_term_variability",
                "bradycardia_percentage",
                "feature_quality_flag",
            ]
        ],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    summary["full_record_average_appears_milder"] = (
        (
            (summary["minimum_window_stv"] <= thresholds["low_stv"])
            & (summary["short_term_variability"] > thresholds["low_stv"])
        )
        | (
            (
                summary["maximum_window_bradycardia_percentage"]
                >= thresholds["high_bradycardia"]
            )
            & (
                summary["bradycardia_percentage"]
                < thresholds["high_bradycardia"]
            )
        )
    )
    return summary


def save_bar_plot(
    summary: pd.DataFrame,
    value_columns: list[str],
    title: str,
    ylabel: str,
    filename: str,
    stacked: bool = False,
) -> Path:
    """Save a consistent record-level bar chart with fixed-test records marked."""
    plot_data = summary.sort_values("record_id").set_index("record_id")
    colors = ["#4c78a8", "#f2a541"][: len(value_columns)]
    axis = plot_data[value_columns].plot(
        kind="bar",
        stacked=stacked,
        figsize=(15, 6),
        color=colors,
        width=0.82,
    )
    axis.set_title(title)
    axis.set_xlabel("Pathological record ID (* = fixed test)")
    axis.set_ylabel(ylabel)
    labels = [
        f"{record_id}*" if split == "test" else record_id
        for record_id, split in zip(plot_data.index, plot_data["split"])
    ]
    axis.set_xticklabels(labels, rotation=55, ha="right")
    axis.grid(axis="x", visible=False)
    if len(value_columns) == 1:
        axis.get_legend().remove()
    else:
        axis.legend(
            [column.replace("_", " ").title() for column in value_columns],
            fontsize=9,
        )
    figure = axis.get_figure()
    figure.tight_layout()
    output = SUMMARY_FIGURE_DIR / filename
    figure.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return output


def create_summary_figures(
    windows: pd.DataFrame, summary: pd.DataFrame
) -> list[Path]:
    """Create the seven cross-record comparison figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    paths = [
        save_bar_plot(
            summary,
            ["valid_fhr_percentage"],
            "Mean valid FHR percentage across 20-minute windows",
            "Valid FHR (%)",
            "01_valid_fhr_by_record.png",
        ),
        save_bar_plot(
            summary,
            ["missing_fhr_percentage", "artifact_percentage"],
            "Mean missing and artifact percentages across windows",
            "Samples (%)",
            "02_missing_artifact_by_record.png",
            stacked=True,
        ),
        save_bar_plot(
            summary,
            ["maximum_window_deceleration_depth"],
            "Maximum window-level deceleration depth",
            "Depth (bpm)",
            "03_maximum_window_deceleration_depth.png",
        ),
        save_bar_plot(
            summary,
            ["minimum_window_stv"],
            "Minimum window-level STV approximation",
            "Mean absolute adjacent FHR change (bpm)",
            "04_minimum_window_stv.png",
        ),
        save_bar_plot(
            summary,
            ["maximum_window_bradycardia_percentage"],
            "Maximum window-level bradycardia percentage",
            "Valid FHR samples below 110 bpm (%)",
            "05_maximum_window_bradycardia.png",
        ),
    ]

    comparison_metrics = [
        ("valid_fhr_percentage", "Valid FHR (%)"),
        ("short_term_variability", "STV approximation"),
        ("max_deceleration_depth", "Max deceleration depth"),
        ("bradycardia_percentage", "Bradycardia (%)"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (metric, label) in zip(axes.flat, comparison_metrics):
        group_values = [
            windows.loc[windows["split"] == split, metric].dropna().to_numpy()
            for split in ("training", "test")
        ]
        axis.boxplot(
            group_values,
            tick_labels=["training", "test"],
            patch_artist=True,
            boxprops={"facecolor": "#8fb9dd", "edgecolor": "#333333"},
            medianprops={"color": "#d95f02", "linewidth": 1.4},
        )
        # Deterministic horizontal offsets keep individual windows visible
        # without introducing randomness into the generated figures.
        for position, values in enumerate(group_values, start=1):
            offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) else []
            axis.scatter(
                np.full(len(values), position) + offsets,
                values,
                color="#333333",
                alpha=0.35,
                s=12,
                zorder=3,
            )
        axis.set_xlabel("")
        axis.set_ylabel(label)
    figure.suptitle(
        "Training versus fixed-test Pathological windows (descriptive only)",
        fontsize=14,
    )
    figure.tight_layout()
    split_path = SUMMARY_FIGURE_DIR / "06_training_vs_fixed_test.png"
    figure.savefig(split_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(split_path)

    position_summary = (
        windows.groupby("window_position", observed=False)
        .agg(
            window_count=("record_id", "size"),
            local_abnormality_rate=(
                "local_abnormality_exploratory_flag",
                "mean",
            ),
            poor_signal_rate=("poor_signal_exploratory_flag", "mean"),
            median_stv=("short_term_variability", "median"),
            median_bradycardia=("bradycardia_percentage", "median"),
        )
        .reindex(["early", "middle", "final"])
        .reset_index()
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    position_summary.plot(
        x="window_position",
        y=["local_abnormality_rate", "poor_signal_rate"],
        kind="bar",
        color=["#d95f02", "#7570b3"],
        ax=axes[0],
    )
    axes[0].set_title("Exploratory and poor-signal flag rates")
    axes[0].set_xlabel("Window position")
    axes[0].set_ylabel("Proportion of windows")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(
        ["Local FHR/event abnormality", "Poor signal"],
        fontsize=9,
    )
    position_summary.plot(
        x="window_position",
        y=["median_stv", "median_bradycardia"],
        kind="bar",
        color=["#1b9e77", "#e6ab02"],
        ax=axes[1],
    )
    axes[1].set_title("Median local signal summaries")
    axes[1].set_xlabel("Window position")
    axes[1].set_ylabel("Metric value (different units)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(
        ["Median STV approximation", "Median bradycardia (%)"],
        fontsize=9,
    )
    figure.suptitle("Early, middle, and final Pathological windows")
    figure.tight_layout()
    position_path = SUMMARY_FIGURE_DIR / "07_early_middle_final_comparison.png"
    figure.savefig(position_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(position_path)
    return paths


def rank_manual_review_records(summary: pd.DataFrame, limit: int = 8) -> list[str]:
    """Rank records by exploratory flags, quality flags, and fixed-test status."""
    ranked = summary.copy()
    ranked["review_score"] = (
        ranked["flagged_window_count"] * 2
        + ranked["poor_quality_window_count"] * 2
        + ranked["final_window_flagged"].astype(int)
        + (ranked["split"] == "test").astype(int)
        + ranked["full_record_average_appears_milder"].astype(int)
    )
    return (
        ranked.sort_values(
            ["review_score", "poor_quality_window_count", "record_id"],
            ascending=[False, False, True],
        )
        .head(limit)["record_id"]
        .tolist()
    )


def determine_conclusion(
    summary: pd.DataFrame, windows: pd.DataFrame
) -> tuple[str, str]:
    """Select one required conclusion from explicit descriptive evidence."""
    records_with_flags = int((summary["flagged_window_count"] > 0).sum())
    records_with_poor_quality = int(
        (summary["poor_quality_window_count"] > 0).sum()
    )
    records_with_milder_average = int(
        summary["full_record_average_appears_milder"].sum()
    )
    final_rate = float(
        windows.loc[
            windows["is_final_window"], "local_abnormality_exploratory_flag"
        ].mean()
    )
    nonfinal_rate = float(
        windows.loc[
            ~windows["is_final_window"], "local_abnormality_exploratory_flag"
        ].mean()
    )
    if records_with_poor_quality >= len(summary) * 0.60:
        return (
            "poor signal quality is the dominant visible issue",
            (
                f"{records_with_poor_quality}/{len(summary)} records contain at least "
                "one worst-decile signal-quality window."
            ),
        )
    if records_with_milder_average >= len(summary) * 0.50:
        return (
            "whole-record averaging likely hides important local abnormalities",
            (
                f"{records_with_milder_average}/{len(summary)} records have a full-record "
                "STV or bradycardia summary that is milder than an extreme local window."
            ),
        )
    if (
        records_with_flags >= len(summary) * 0.40
        or abs(final_rate - nonfinal_rate) >= 0.10
    ):
        return (
            "evidence is mixed and window-based modelling should be tested",
            (
                f"{records_with_flags}/{len(summary)} records contain an exploratory "
                f"flagged window; final/non-final flag rates are {final_rate:.1%} "
                f"and {nonfinal_rate:.1%}."
            ),
        )
    return (
        "whole-record averaging does not appear to be the main issue",
        (
            f"Only {records_with_flags}/{len(summary)} records contain an exploratory "
            "flagged window and local-versus-full summaries show limited contrast."
        ),
    )


def split_comparison_text(windows: pd.DataFrame) -> str:
    """Describe fixed-test versus training differences without inference claims."""
    metrics = [
        "valid_fhr_percentage",
        "short_term_variability",
        "max_deceleration_depth",
        "bradycardia_percentage",
        "local_abnormality_exploratory_flag",
    ]
    means = windows.groupby("split")[metrics].mean(numeric_only=True)
    if not {"training", "test"}.issubset(means.index):
        return "A training/test comparison could not be calculated."
    pieces = []
    labels = {
        "valid_fhr_percentage": "valid FHR",
        "short_term_variability": "STV",
        "max_deceleration_depth": "maximum deceleration depth",
        "bradycardia_percentage": "bradycardia exposure",
        "local_abnormality_exploratory_flag": "local abnormality flag rate",
    }
    for metric in metrics:
        pieces.append(
            f"{labels[metric]} {means.loc['test', metric]:.2f} versus "
            f"{means.loc['training', metric]:.2f}"
        )
    return (
        "At window level, fixed-test versus training Pathological means were: "
        + "; ".join(pieces)
        + ". These are descriptive comparisons from only six test records."
    )


def write_markdown_report(
    pathological_ids: list[str],
    test_ids: list[str],
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    thresholds: dict[str, float],
    conclusion: str,
    conclusion_support: str,
) -> None:
    """Write the visual review in plain language."""
    records_with_flags = int((summary["flagged_window_count"] > 0).sum())
    records_with_poor_quality = int(
        (summary["poor_quality_window_count"] > 0).sum()
    )
    final_window_records = int(summary["final_window_flagged"].sum())
    milder_average_records = int(
        summary["full_record_average_appears_milder"].sum()
    )
    final_rate = float(
        windows.loc[
            windows["is_final_window"], "local_abnormality_exploratory_flag"
        ].mean()
    )
    nonfinal_rate = float(
        windows.loc[
            ~windows["is_final_window"], "local_abnormality_exploratory_flag"
        ].mean()
    )
    final_poor_signal_rate = float(
        windows.loc[
            windows["is_final_window"], "poor_signal_exploratory_flag"
        ].mean()
    )
    nonfinal_poor_signal_rate = float(
        windows.loc[
            ~windows["is_final_window"], "poor_signal_exploratory_flag"
        ].mean()
    )
    more_common_near_end = final_rate > nonfinal_rate
    priority_ids = rank_manual_review_records(summary)

    report = f"""# Pathological CTG Visual Review

## Scope and safeguards

This descriptive review inspected all **{len(pathological_ids)}** records with
the stored label `Pathological`. It did not retrain a model, change a label,
alter the fixed split, modify the feature extractor, or create a window-level
machine-learning dataset. The 20-minute rows remain diagnostics for records
whose label is already known; no labels were assigned to windows.

The six Pathological records in the fixed test set are:
**{", ".join(test_ids)}**.

## What was plotted

Each record figure shows raw and cleaned FHR, the current moving baseline,
retained acceleration and deceleration events, missing/artifact regions,
smoothed UC and retained contraction peaks, and a 20-minute summary table.
The same current extractor helpers and thresholds were reused. Long gaps were
not interpolated.

STV is the mean absolute difference between adjacent continuously valid
samples. LTV is the standard deviation of valid 60-second epoch means. Both
are computational approximations and are not clinical gold-standard measures.

## Exploratory window flags

Flags use the distribution of 20-minute windows from all {len(pd.read_csv(ML_DATASET_PATH))}
records, not newly invented medical cutoffs:

- Low STV: at or below the {EXPLORATORY_LOW_PERCENTILE:.0f}th percentile
  ({thresholds['low_stv']:.4f}).
- Deep deceleration: at or above the {EXPLORATORY_HIGH_PERCENTILE:.0f}th
  percentile ({thresholds['deep_deceleration']:.2f} bpm).
- High bradycardia exposure: at or above the
  {EXPLORATORY_HIGH_PERCENTILE:.0f}th percentile
  ({thresholds['high_bradycardia']:.2f}%).
- Long deceleration: at or above the {EXPLORATORY_HIGH_PERCENTILE:.0f}th
  percentile ({thresholds['long_deceleration']:.2f} seconds).
- Poor signal: combined missing/artifact percentage at or above the
  {EXPLORATORY_HIGH_PERCENTILE:.0f}th percentile
  ({thresholds['poor_signal']:.2f}%).

These are exploratory review flags, not clinical diagnoses.

## Main results

- Pathological records inspected: **{len(pathological_ids)}**.
- Pathological windows analysed: **{len(windows)}**.
- Records with at least one local FHR/event abnormality flag:
  **{records_with_flags}/{len(pathological_ids)}**.
- Records with at least one worst-decile poor-quality window:
  **{records_with_poor_quality}/{len(pathological_ids)}**.
- Records whose final 20-minute window was flagged:
  **{final_window_records}/{len(pathological_ids)}**.
- Local-abnormality flag rate in final windows: **{final_rate:.1%}**.
- Local-abnormality flag rate in non-final windows: **{nonfinal_rate:.1%}**.
- Local FHR/event abnormalities were
  {"more" if more_common_near_end else "not more"} common near the end by this
  descriptive comparison. Poor-signal flags are reported separately because
  missing data are not physiological abnormalities.
- Poor-signal flag rate in final windows: **{final_poor_signal_rate:.1%}**,
  compared with **{nonfinal_poor_signal_rate:.1%}** in non-final windows. Thus,
  any visible end-of-record concentration should primarily be interpreted as
  signal loss unless the individual trace shows a retained FHR/event flag.
- Records where a full-record STV or bradycardia average appears milder than
  the worst local window: **{milder_average_records}/{len(pathological_ids)}**.

## Fixed-test versus training Pathological records

{split_comparison_text(windows)}

The figures mark a fixed-test record as a missed Pathological case only when
the stored audit prediction is not `Pathological`. No new prediction was made
by this script.

## Does whole-record averaging hide local abnormalities?

The comparison is strongest for measurements that are genuinely averaged over
the full record, particularly STV and bradycardia percentage. For
**{milder_average_records}** records, at least one local window entered an
exploratory extreme tail while the corresponding full-record average did not.
Maximum-based full-record features, such as maximum deceleration depth, are
not described as averaging losses.

## Manual review priorities

The first records to inspect manually are:
**{", ".join(priority_ids)}**.

They rank highest on a transparent combination of flagged windows,
poor-quality windows, final-window flags, fixed-test membership, and evidence
that a full-record average looks milder than the worst window. This ordering is
for review efficiency only.

## Limitations

- A 20-minute boundary can divide a physiological episode; retained events are
  assigned to the window containing their start or peak.
- The final window can be shorter than 20 minutes.
- Tail thresholds are dataset-relative and can flag noise, artifacts, or
  benign variation.
- The six-record test comparison is too small for a reliable causal claim.
- Visual inspection can motivate a later experiment but cannot show that
  window-based modelling improves prediction.

## Conclusion

**{conclusion}.**

{conclusion_support} This result supports visual/manual follow-up and, where
appropriate, a separately controlled future windowing experiment. It does not
claim a performance improvement.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def validate_outputs(
    windows: pd.DataFrame,
    pathological_ids: list[str],
    test_ids: list[str],
    record_figure_paths: list[Path],
    summary_figure_paths: list[Path],
) -> None:
    """Perform compact schema, range, count, and artifact validation."""
    if list(windows.columns) != WINDOW_COLUMNS:
        raise AssertionError("Window CSV columns do not match the required schema")
    if set(windows["record_id"]) != set(pathological_ids):
        raise AssertionError("Window CSV does not contain exactly the Pathological IDs")
    if windows[["record_id", "window_number"]].duplicated().any():
        raise AssertionError("Duplicate record/window rows were generated")
    if set(windows.loc[windows["split"] == "test", "record_id"]) != set(test_ids):
        raise AssertionError("Window split membership does not match the fixed test IDs")
    percentage_columns = [
        "valid_fhr_percentage",
        "missing_fhr_percentage",
        "artifact_percentage",
        "bradycardia_percentage",
        "tachycardia_percentage",
    ]
    for column in percentage_columns:
        if not windows[column].dropna().between(0, 100).all():
            raise AssertionError(f"{column} contains a value outside 0--100")
    if len(record_figure_paths) != 27 or not all(path.exists() for path in record_figure_paths):
        raise AssertionError("Expected 27 record-level figure files")
    if len(summary_figure_paths) != 7 or not all(path.exists() for path in summary_figure_paths):
        raise AssertionError("Expected seven summary figure files")
    if not WINDOW_CSV_PATH.exists() or not REPORT_PATH.exists():
        raise AssertionError("CSV or Markdown report is missing")
    if any(path.stat().st_size < 20_000 for path in record_figure_paths):
        raise AssertionError("At least one record figure is unexpectedly small")
    if any(path.stat().st_size < 15_000 for path in summary_figure_paths):
        raise AssertionError("At least one summary figure is unexpectedly small")


def main() -> None:
    """Run the read-only Pathological visual diagnostic workflow."""
    labels, features, dataset, audit = load_and_validate_inputs()
    pathological_ids = sorted(
        labels.loc[labels["label"] == "Pathological", "record_id"].tolist()
    )
    if len(pathological_ids) != 27:
        raise ValueError(
            f"Expected exactly 27 Pathological records, found {len(pathological_ids)}"
        )

    _train_ids, test_ids_all = reproduce_fixed_split(dataset)
    pathological_test_ids = sorted(set(pathological_ids) & test_ids_all)
    if len(pathological_test_ids) != 6:
        raise ValueError(
            "Expected six Pathological records in the fixed test set, found "
            f"{len(pathological_test_ids)}"
        )

    print("Pathological records (27):")
    print(", ".join(pathological_ids))
    print("\nFixed-test Pathological records (6):")
    print(", ".join(pathological_test_ids))

    waveform_directory = find_waveform_directory(dataset["record_id"].tolist())
    print(f"\nUsing read-only waveform directory: {waveform_directory}")
    ensure_new_output_location()

    audit_lookup = (
        audit.set_index("record_id", drop=False)
        if not audit.empty
        else pd.DataFrame()
    )
    clinical_lookup = (
        dataset.set_index("record_id", drop=False)
        .join(
            features.set_index("record_id")[["feature_quality_flag"]],
            rsuffix="_feature",
        )
    )

    print("\nCalculating all-record reference windows for exploratory tails...")
    reference_windows = build_reference_windows(
        dataset["record_id"].tolist(), waveform_directory
    )
    thresholds = calculate_exploratory_thresholds(reference_windows)

    print("\nLoading and plotting the 27 Pathological records...")
    bundles: dict[str, RecordBundle] = {}
    pathological_rows = []
    for record_id in pathological_ids:
        split = "test" if record_id in test_ids_all else "training"
        bundle = load_record_bundle(
            record_id, waveform_directory, split, audit_lookup
        )
        bundles[record_id] = bundle
        pathological_rows.extend(bundle.windows)

    windows = apply_exploratory_flags(
        pd.DataFrame(pathological_rows), thresholds
    )
    windows = windows[WINDOW_COLUMNS].sort_values(
        ["record_id", "window_number"]
    )
    windows.to_csv(WINDOW_CSV_PATH, index=False)

    record_figure_paths = []
    for index, record_id in enumerate(pathological_ids, start=1):
        audit_row = (
            audit_lookup.loc[record_id]
            if not audit_lookup.empty and record_id in audit_lookup.index
            else None
        )
        record_figure_paths.append(
            create_record_figure(
                bundles[record_id],
                windows[windows["record_id"] == record_id],
                clinical_lookup.loc[record_id],
                audit_row,
            )
        )
        print(f"  Record figures generated: {index}/{len(pathological_ids)}")

    summary = record_level_summary(windows, features, thresholds)
    summary_figure_paths = create_summary_figures(windows, summary)
    conclusion, conclusion_support = determine_conclusion(summary, windows)
    write_markdown_report(
        pathological_ids,
        pathological_test_ids,
        windows,
        summary,
        thresholds,
        conclusion,
        conclusion_support,
    )
    validate_outputs(
        windows,
        pathological_ids,
        pathological_test_ids,
        record_figure_paths,
        summary_figure_paths,
    )

    records_with_flags = int((summary["flagged_window_count"] > 0).sum())
    records_with_poor_quality = int(
        (summary["poor_quality_window_count"] > 0).sum()
    )
    final_rate = float(
        windows.loc[
            windows["is_final_window"], "local_abnormality_exploratory_flag"
        ].mean()
    )
    nonfinal_rate = float(
        windows.loc[
            ~windows["is_final_window"], "local_abnormality_exploratory_flag"
        ].mean()
    )

    print("\nPathological visual review complete")
    print(f"Pathological records inspected: {len(pathological_ids)}")
    print(
        "Fixed-test Pathological record IDs: "
        + ", ".join(pathological_test_ids)
    )
    print(f"Generated record figures: {len(record_figure_paths)}")
    print(f"20-minute windows analysed: {len(windows)}")
    print(
        "Records with at least one flagged abnormal window: "
        f"{records_with_flags}"
    )
    print(f"Records with poor-quality windows: {records_with_poor_quality}")
    print(
        "Abnormal windows more common near the end: "
        f"{'yes' if final_rate > nonfinal_rate else 'no'} "
        f"(final {final_rate:.1%}, non-final {nonfinal_rate:.1%})"
    )
    print(f"Main visual conclusion: {conclusion}")
    print(f"Window CSV: {WINDOW_CSV_PATH}")
    print(f"Record figures: {RECORD_FIGURE_DIR}")
    print(f"Summary figures: {SUMMARY_FIGURE_DIR}")
    print(f"Markdown report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
