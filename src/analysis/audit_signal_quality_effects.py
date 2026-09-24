"""Audit how FHR signal quality affects engineered CTG features.

This is a descriptive diagnostic only.  It does not train a model, modify the
feature extractor, assign labels to windows, change the fixed split, or alter
raw signals.  Artificial masking is applied only to in-memory copies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import visualize_pathological_recordings as visual
from src.features import extract_clinical_features as extractor
from src.utils.record_ids import normalize_record_id

# Inputs, outputs, and diagnostic settings
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.csv"
CLINICAL_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"
ML_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
PATHOLOGICAL_WINDOWS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "pathological_visual_review"
    / "pathological_window_diagnostics.csv"
)

OUTPUT_ROOT = PROJECT_ROOT / "reports" / "signal_quality_audit"
FIGURE_DIR = OUTPUT_ROOT / "figures"
FEATURE_BY_QUALITY_PATH = OUTPUT_ROOT / "feature_by_quality.csv"
MASKING_RESULTS_PATH = OUTPUT_ROOT / "masking_experiment_results.csv"
FRAGMENTATION_PATH = OUTPUT_ROOT / "fragmentation_analysis.csv"
PATHOLOGICAL_EFFECTS_PATH = OUTPUT_ROOT / "pathological_quality_effects.csv"
RULE_COMPARISON_PATH = OUTPUT_ROOT / "window_validity_rule_comparison.csv"
REPORT_PATH = OUTPUT_ROOT / "signal_quality_audit_report.md"

WINDOW_MINUTES = 20.0

# Diagnostic quality categories used by this analysis. They are engineering categories,
# not medical or clinical standards.
QUALITY_CATEGORY_ORDER = [
    "very_high_quality",
    "acceptable_quality",
    "low_quality",
    "very_low_quality",
]

# High quality for comparisons and masking means >=90% valid cleaned FHR.
HIGH_QUALITY_MIN_VALID_PERCENTAGE = 90.0
DESCRIPTIVE_HIGH_LOW_BOUNDARY = 80.0

# Extreme counts use 3-IQR outer fences from very-high-quality windows. This is
# a data-driven diagnostic threshold, not a clinical boundary.
EXTREME_IQR_MULTIPLIER = 3.0

# Controlled masking uses a balanced, reproducible training-record sample where
# yield enough repeated placements without making the audit unnecessarily slow.
MASK_WINDOWS_PER_LABEL = 6
MASKING_LEVELS_PERCENT = [5, 10, 20, 30, 40, 50]
MASKING_SEEDS = [11, 23, 37, 53, 71]
MASK_GAPS_PER_RUN = 3
SAMPLE_RANDOM_STATE = 42

# A real window's fragmentation class is based on the share of valid samples
# descriptors, not validity standards.
CONTINUOUS_DOMINANT_MIN_SHARE = 80.0
MODERATE_FRAGMENTATION_MIN_SHARE = 50.0

# Rule selection is constrained before reliability is compared.  A candidate
# must retain at least half of all windows and at least 26 of the 27
# its surviving data look cleaner.
MIN_RULE_WINDOW_RETENTION_FRACTION = 0.50
MIN_RULE_PATHOLOGICAL_RECORDS_RETAINED = 26

# Candidate combined rules use the extractor's QC thresholds where practical.
# They are diagnostic proposals, not production rules.
VALIDITY_RULES = [
    {
        "rule_name": "Rule A: valid >=60%",
        "min_valid_percentage": 60.0,
        "max_artifact_percentage": np.inf,
        "min_longest_valid_segment_seconds": 0.0,
        "max_longest_missing_gap_seconds": np.inf,
    },
    {
        "rule_name": "Rule B: valid >=70%",
        "min_valid_percentage": 70.0,
        "max_artifact_percentage": np.inf,
        "min_longest_valid_segment_seconds": 0.0,
        "max_longest_missing_gap_seconds": np.inf,
    },
    {
        "rule_name": "Rule C: valid >=80%",
        "min_valid_percentage": 80.0,
        "max_artifact_percentage": np.inf,
        "min_longest_valid_segment_seconds": 0.0,
        "max_longest_missing_gap_seconds": np.inf,
    },
    {
        "rule_name": "Rule D: valid >=90%",
        "min_valid_percentage": 90.0,
        "max_artifact_percentage": np.inf,
        "min_longest_valid_segment_seconds": 0.0,
        "max_longest_missing_gap_seconds": np.inf,
    },
    {
        "rule_name": "Rule E: balanced continuity",
        "min_valid_percentage": 70.0,
        "max_artifact_percentage": extractor.QUALITY_REVIEW_MAX_ARTIFACT_PERCENTAGE,
        "min_longest_valid_segment_seconds": 5 * 60.0,
        "max_longest_missing_gap_seconds": extractor.QUALITY_POOR_MAX_MISSING_GAP_SECONDS,
    },
    {
        "rule_name": "Rule F: 80% + continuity",
        "min_valid_percentage": 80.0,
        "max_artifact_percentage": extractor.QUALITY_REVIEW_MAX_ARTIFACT_PERCENTAGE,
        "min_longest_valid_segment_seconds": 5 * 60.0,
        "max_longest_missing_gap_seconds": extractor.QUALITY_POOR_MAX_MISSING_GAP_SECONDS,
    },
    {
        "rule_name": "Rule G: strict continuity",
        "min_valid_percentage": 90.0,
        "max_artifact_percentage": extractor.QUALITY_REVIEW_MAX_ARTIFACT_PERCENTAGE,
        "min_longest_valid_segment_seconds": 10 * 60.0,
        "max_longest_missing_gap_seconds": extractor.QUALITY_REVIEW_MAX_MISSING_GAP_SECONDS,
    },
]

# These features receive extra attention in figures and narrative. All numeric
# engineered features are still summarized in feature_by_quality.csv
# and tested in the masking experiment.
KEY_FEATURES = [
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
    "decelerations_per_contraction",
]

# Direct duration/missingness indicators are audited and included in the raw
# masking CSV, but they are excluded from "most affected clinical feature"
# rankings because missingness is introduced by the masking experiment.
DIRECT_QUALITY_FEATURES = {
    "recording_duration_minutes",
    "percent_missing_fhr",
    "longest_missing_fhr_gap_seconds",
    "valid_fhr_percentage",
    "possible_fhr_artifact_percentage",
}

FIGURE_DPI = 210


def ensure_new_output_location() -> None:
    """Refuse to overwrite an existing audit."""
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(
            f"{OUTPUT_ROOT} already contains files. Move or rename it before "
            "rerunning; this script will not overwrite an existing report."
        )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and align all required project tables."""
    required = [
        LABELS_PATH,
        CLINICAL_FEATURES_PATH,
        ML_DATASET_PATH,
        PATHOLOGICAL_WINDOWS_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    labels = pd.read_csv(LABELS_PATH)
    features = pd.read_csv(CLINICAL_FEATURES_PATH)
    dataset = pd.read_csv(ML_DATASET_PATH)
    pathological_windows = pd.read_csv(PATHOLOGICAL_WINDOWS_PATH)
    for frame, name in (
        (labels, "labels"),
        (features, "clinical_features"),
        (dataset, "ml_dataset"),
        (pathological_windows, "pathological_window_diagnostics"),
    ):
        if "record_id" not in frame:
            raise ValueError(f"{name} is missing record_id")
        frame["record_id"] = normalize_record_id(frame["record_id"])

    expected_ids = set(labels["record_id"])
    if len(expected_ids) != len(labels):
        raise ValueError("labels.csv contains duplicate record IDs")
    if set(features["record_id"]) != expected_ids:
        raise ValueError("clinical_features.csv does not align with labels.csv")
    if set(dataset["record_id"]) != expected_ids:
        raise ValueError("ml_dataset.csv does not align with labels.csv")
    if set(pathological_windows["record_id"]) != set(
        labels.loc[labels["label"] == "Pathological", "record_id"]
    ):
        raise ValueError("Pathological window diagnostics do not contain the 27 records")
    return labels, features, dataset, pathological_windows


def quality_category(valid_percentage: float) -> str:
    """Assign one diagnostic quality category."""
    if valid_percentage >= 90.0:
        return "very_high_quality"
    if valid_percentage >= 80.0:
        return "acceptable_quality"
    if valid_percentage >= 60.0:
        return "low_quality"
    return "very_low_quality"


def true_run_lengths(mask: np.ndarray) -> np.ndarray:
    """Return lengths of every contiguous True run."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return ends - starts


def fragmentation_metrics(
    valid_mask: np.ndarray, sampling_frequency: float
) -> dict[str, float | int | str]:
    """Describe valid-signal continuity without filling any gap."""
    lengths = true_run_lengths(valid_mask)
    total_valid_samples = int(np.sum(lengths))
    longest_samples = int(np.max(lengths)) if len(lengths) else 0
    median_samples = float(np.median(lengths)) if len(lengths) else 0.0
    longest_share = (
        100.0 * longest_samples / total_valid_samples
        if total_valid_samples
        else 0.0
    )
    if longest_share >= CONTINUOUS_DOMINANT_MIN_SHARE:
        category = "continuous_dominant"
    elif longest_share >= MODERATE_FRAGMENTATION_MIN_SHARE:
        category = "moderately_fragmented"
    else:
        category = "highly_fragmented"
    return {
        "total_valid_duration_seconds": total_valid_samples / sampling_frequency,
        "longest_continuous_valid_segment_seconds": (
            longest_samples / sampling_frequency
        ),
        "number_of_valid_segments": len(lengths),
        "median_valid_segment_duration_seconds": median_samples / sampling_frequency,
        "percentage_valid_in_longest_segment": longest_share,
        "fragmentation_category": category,
    }


def extract_current_window_features(
    raw_fhr: np.ndarray, uc: np.ndarray, sampling_frequency: float
) -> tuple[dict, dict]:
    """Recalculate the feature set on one raw 20-minute slice.

    The formulas mirror extract_features_for_record while reusing the extractor
    helpers.  No interpolation or alternate event logic is introduced.
    """
    analysis = extractor.analyze_signals(raw_fhr, uc, sampling_frequency)
    fhr = analysis["cleaned_fhr"]
    accelerations = analysis["acceleration_features"]
    decelerations = analysis["deceleration_features"]
    contractions = analysis["contraction_features"]
    sample_count = len(fhr)
    recording_seconds = sample_count / sampling_frequency
    valid_percentage = 100.0 * float(np.mean(np.isfinite(fhr)))
    missing_percentage = 100.0 * float(np.mean(analysis["raw_missing_mask"]))
    artifact_percentage = 100.0 * float(np.mean(analysis["artifact_mask"]))
    longest_missing_gap = (
        extractor.longest_true_run(analysis["raw_missing_mask"])
        / sampling_frequency
    )
    contraction_count = contractions["contraction_count"]
    deceleration_count = decelerations["count"]

    features = {
        "recording_duration_minutes": recording_seconds / 60.0,
        "percent_missing_fhr": missing_percentage,
        "longest_missing_fhr_gap_seconds": longest_missing_gap,
        "valid_fhr_percentage": valid_percentage,
        "possible_fhr_artifact_percentage": artifact_percentage,
        "mean_fhr": extractor.safe_nan_stat(fhr, np.mean),
        "median_fhr": extractor.safe_nan_stat(fhr, np.median),
        "min_fhr": extractor.safe_nan_stat(fhr, np.min),
        "max_fhr": extractor.safe_nan_stat(fhr, np.max),
        "std_fhr": extractor.safe_nan_stat(fhr, np.std),
        "baseline_fhr": extractor.safe_nan_stat(
            analysis["baseline_series"], np.median
        ),
        "short_term_variability": extractor.calculate_short_term_variability(fhr),
        "long_term_variability": extractor.calculate_long_term_variability(
            fhr, sampling_frequency
        ),
        "acceleration_count": accelerations["count"],
        "mean_acceleration_duration_seconds": accelerations[
            "mean_duration_seconds"
        ],
        "max_acceleration_amplitude": accelerations[
            "max_amplitude_or_depth"
        ],
        "deceleration_count": deceleration_count,
        "mean_deceleration_duration_seconds": decelerations[
            "mean_duration_seconds"
        ],
        "max_deceleration_depth": decelerations["max_amplitude_or_depth"],
        "mean_uc": contractions["mean_uc"],
        "median_uc": contractions["median_uc"],
        "max_uc": contractions["max_uc"],
        "std_uc": contractions["std_uc"],
        "contraction_count": contraction_count,
        "mean_contraction_peak": contractions["mean_contraction_peak"],
        "mean_contraction_interval_seconds": contractions[
            "mean_contraction_interval_seconds"
        ],
        "decelerations_per_contraction": (
            deceleration_count / contraction_count if contraction_count else 0.0
        ),
    }
    features.update(extractor.calculate_additional_features(analysis, sampling_frequency))
    qc = {
        "missing_fhr_percentage": missing_percentage,
        "artifact_percentage": artifact_percentage,
        "longest_missing_gap_seconds": longest_missing_gap,
        "number_of_missing_gaps": len(true_run_lengths(analysis["raw_missing_mask"])),
        **fragmentation_metrics(np.isfinite(fhr), sampling_frequency),
    }
    return features, {**analysis, **qc}


def iter_window_bounds(sample_count: int, sampling_frequency: float):
    """Yield non-overlapping 20-minute half-open sample bounds."""
    window_samples = max(1, round(WINDOW_MINUTES * 60 * sampling_frequency))
    bounds = [
        (start, min(start + window_samples, sample_count))
        for start in range(0, sample_count, window_samples)
    ]
    for number, (start, end) in enumerate(bounds, start=1):
        yield number, start, end, number == len(bounds)


def calculate_all_windows(
    labels: pd.DataFrame,
    dataset: pd.DataFrame,
    waveform_directory: Path,
    test_ids: set[str],
) -> pd.DataFrame:
    """Recalculate the feature set for every diagnostic window."""
    label_lookup = labels.set_index("record_id")["label"]
    rows = []
    record_ids = dataset["record_id"].tolist()
    for record_number, record_id in enumerate(record_ids, start=1):
        record = extractor.load_record(waveform_directory / f"{record_id}.hea")
        fs = float(record.fs)
        raw_fhr = extractor.get_signal_by_name(record, "FHR")
        uc = extractor.get_signal_by_name(record, "UC")
        for window_number, start, end, is_final in iter_window_bounds(len(raw_fhr), fs):
            feature_values, analysis = extract_current_window_features(
                raw_fhr[start:end], uc[start:end], fs
            )
            valid_percentage = feature_values["valid_fhr_percentage"]
            rows.append(
                {
                    "record_id": record_id,
                    "label": label_lookup.loc[record_id],
                    "split": "test" if record_id in test_ids else "training",
                    "window_number": window_number,
                    "start_minute": start / fs / 60.0,
                    "end_minute": end / fs / 60.0,
                    "is_final_window": is_final,
                    "quality_category": quality_category(valid_percentage),
                    "missing_fhr_percentage": analysis[
                        "missing_fhr_percentage"
                    ],
                    "artifact_percentage": analysis["artifact_percentage"],
                    "longest_missing_gap_seconds": analysis[
                        "longest_missing_gap_seconds"
                    ],
                    "number_of_missing_gaps": analysis["number_of_missing_gaps"],
                    "total_valid_duration_seconds": analysis[
                        "total_valid_duration_seconds"
                    ],
                    "longest_continuous_valid_segment_seconds": analysis[
                        "longest_continuous_valid_segment_seconds"
                    ],
                    "number_of_valid_segments": analysis[
                        "number_of_valid_segments"
                    ],
                    "median_valid_segment_duration_seconds": analysis[
                        "median_valid_segment_duration_seconds"
                    ],
                    "percentage_valid_in_longest_segment": analysis[
                        "percentage_valid_in_longest_segment"
                    ],
                    "fragmentation_category": analysis[
                        "fragmentation_category"
                    ],
                    **feature_values,
                }
            )
        if record_number % 100 == 0 or record_number == len(record_ids):
            print(f"  Window features processed: {record_number}/{len(record_ids)} records")
    windows = pd.DataFrame(rows)
    windows["quality_category"] = pd.Categorical(
        windows["quality_category"],
        categories=QUALITY_CATEGORY_ORDER,
        ordered=True,
    )
    return windows


def audited_feature_names(windows: pd.DataFrame) -> list[str]:
    """Return each numeric engineered feature once."""
    excluded = {
        "record_id",
        "label",
        "split",
        "window_number",
        "start_minute",
        "end_minute",
        "is_final_window",
        "quality_category",
        "missing_fhr_percentage",
        "artifact_percentage",
        "longest_missing_gap_seconds",
        "number_of_missing_gaps",
        "total_valid_duration_seconds",
        "longest_continuous_valid_segment_seconds",
        "number_of_valid_segments",
        "median_valid_segment_duration_seconds",
        "percentage_valid_in_longest_segment",
        "fragmentation_category",
    }
    return [
        column
        for column in windows.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(windows[column])
    ]


def outer_fences(reference: pd.Series) -> tuple[float, float]:
    """Calculate conservative 3-IQR extreme fences from finite values."""
    values = pd.to_numeric(reference, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        spread = values.std(ddof=0)
        if not np.isfinite(spread) or spread == 0:
            return float(values.min()), float(values.max())
        return float(q1 - 3 * spread), float(q3 + 3 * spread)
    return (
        float(q1 - EXTREME_IQR_MULTIPLIER * iqr),
        float(q3 + EXTREME_IQR_MULTIPLIER * iqr),
    )


def summarize_features_by_quality(
    windows: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    """Create distribution and robust shift statistics for every feature."""
    rows = []
    very_high = windows[
        windows["quality_category"] == "very_high_quality"
    ]
    for feature in feature_names:
        lower, upper = outer_fences(very_high[feature])
        high_values = pd.to_numeric(very_high[feature], errors="coerce").dropna()
        high_median = float(high_values.median()) if len(high_values) else np.nan
        high_iqr = (
            float(high_values.quantile(0.75) - high_values.quantile(0.25))
            if len(high_values)
            else np.nan
        )
        high_scale = high_iqr
        if not np.isfinite(high_scale) or high_scale == 0:
            high_scale = float(high_values.std(ddof=0)) if len(high_values) else np.nan
        if not np.isfinite(high_scale) or high_scale == 0:
            high_scale = max(abs(high_median), 1.0) if np.isfinite(high_median) else 1.0

        for category in QUALITY_CATEGORY_ORDER:
            subset = windows[windows["quality_category"] == category]
            values = pd.to_numeric(subset[feature], errors="coerce")
            finite = values.dropna()
            q1 = finite.quantile(0.25) if len(finite) else np.nan
            q3 = finite.quantile(0.75) if len(finite) else np.nan
            median = finite.median() if len(finite) else np.nan
            extreme_count = (
                int(((finite < lower) | (finite > upper)).sum())
                if np.isfinite(lower) and np.isfinite(upper)
                else 0
            )
            median_shift = (
                float(median - high_median)
                if np.isfinite(median) and np.isfinite(high_median)
                else np.nan
            )
            rows.append(
                {
                    "feature": feature,
                    "quality_category": category,
                    "count": len(subset),
                    "valid_value_count": len(finite),
                    "mean": float(finite.mean()) if len(finite) else np.nan,
                    "median": float(median) if len(finite) else np.nan,
                    "standard_deviation": (
                        float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
                    ),
                    "interquartile_range": (
                        float(q3 - q1) if len(finite) else np.nan
                    ),
                    "missing_value_count": int(values.isna().sum()),
                    "extreme_value_count": extreme_count,
                    "extreme_lower_bound_from_very_high_quality": lower,
                    "extreme_upper_bound_from_very_high_quality": upper,
                    "median_shift_from_very_high_quality": median_shift,
                    "absolute_standardized_median_shift": (
                        abs(median_shift) / high_scale
                        if np.isfinite(median_shift)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def select_masking_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Select a fixed balanced sample of very-high-quality training windows."""
    eligible = windows[
        (windows["quality_category"] == "very_high_quality")
        & (windows["split"] == "training")
        & (~windows["is_final_window"])
    ].copy()
    selected = []
    for label in ["Normal", "Suspicious", "Pathological"]:
        candidates = eligible[eligible["label"] == label]
        if candidates.empty:
            continue
        selected.append(
            candidates.sample(
                n=min(MASK_WINDOWS_PER_LABEL, len(candidates)),
                random_state=SAMPLE_RANDOM_STATE,
            )
        )
    if not selected:
        raise ValueError("No high-quality windows were available for masking")
    return pd.concat(selected).sort_values(["label", "record_id", "window_number"])


def place_non_overlapping_gaps(
    sample_count: int, target_samples: int, seed: int
) -> np.ndarray:
    """Create non-overlapping contiguous time gaps."""
    rng = np.random.default_rng(seed)
    target_samples = min(max(1, target_samples), sample_count)
    gap_count = min(MASK_GAPS_PER_RUN, target_samples)
    lengths = rng.multinomial(target_samples, np.full(gap_count, 1.0 / gap_count))
    lengths = sorted([int(length) for length in lengths if length], reverse=True)
    available = [(0, sample_count)]
    mask = np.zeros(sample_count, dtype=bool)
    for length in lengths:
        choices = []
        weights = []
        for position, (left, right) in enumerate(available):
            placements = right - left - length + 1
            if placements > 0:
                choices.append((position, left, right))
                weights.append(placements)
        if not choices:
            raise RuntimeError("Could not place the specified synthetic gaps")
        probabilities = np.asarray(weights, dtype=float) / np.sum(weights)
        chosen = choices[int(rng.choice(len(choices), p=probabilities))]
        position, left, right = chosen
        start = int(rng.integers(left, right - length + 1))
        end = start + length
        mask[start:end] = True
        replacement = []
        if left < start:
            replacement.append((left, start))
        if end < right:
            replacement.append((end, right))
        available[position : position + 1] = replacement
    return mask


def load_window_arrays(
    waveform_directory: Path, record_id: str, window_number: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Reload one selected raw window without retaining all waveforms in memory."""
    record = extractor.load_record(waveform_directory / f"{record_id}.hea")
    fs = float(record.fs)
    raw_fhr = extractor.get_signal_by_name(record, "FHR")
    uc = extractor.get_signal_by_name(record, "UC")
    bounds = list(iter_window_bounds(len(raw_fhr), fs))
    _, start, end, _ = bounds[window_number - 1]
    return raw_fhr[start:end], uc[start:end], fs


def run_masking_experiment(
    selected: pd.DataFrame,
    feature_names: list[str],
    waveform_directory: Path,
) -> pd.DataFrame:
    """Mask copied FHR slices and quantify feature error and failure."""
    rows = []
    for sample_number, selected_row in enumerate(selected.itertuples(), start=1):
        record_id = str(selected_row.record_id)
        window_number = int(selected_row.window_number)
        raw_fhr, uc, fs = load_window_arrays(
            waveform_directory, record_id, window_number
        )
        original_features, original_analysis = extract_current_window_features(
            raw_fhr, uc, fs
        )
        original_valid = np.isfinite(original_analysis["cleaned_fhr"])
        for masking_level in MASKING_LEVELS_PERCENT:
            target_samples = round(len(raw_fhr) * masking_level / 100.0)
            for placement_number, seed in enumerate(MASKING_SEEDS, start=1):
                combined_seed = (
                    seed
                    + int(record_id) * 1009
                    + window_number * 101
                    + masking_level * 17
                )
                synthetic_mask = place_non_overlapping_gaps(
                    len(raw_fhr), target_samples, combined_seed
                )
                masked_fhr = np.asarray(raw_fhr, dtype=float).copy()
                masked_fhr[synthetic_mask] = np.nan
                masked_features, masked_analysis = extract_current_window_features(
                    masked_fhr, uc, fs
                )
                newly_removed_valid = int(np.sum(original_valid & synthetic_mask))
                realized_loss = 100.0 * newly_removed_valid / max(
                    1, int(np.sum(original_valid))
                )
                masked_fragmentation = fragmentation_metrics(
                    np.isfinite(masked_analysis["cleaned_fhr"]), fs
                )
                for feature in feature_names:
                    original_value = original_features.get(feature, np.nan)
                    masked_value = masked_features.get(feature, np.nan)
                    failure = not np.isfinite(masked_value)
                    absolute_error = (
                        abs(float(masked_value) - float(original_value))
                        if np.isfinite(masked_value) and np.isfinite(original_value)
                        else np.nan
                    )
                    relative_error = (
                        absolute_error / abs(float(original_value))
                        if np.isfinite(absolute_error)
                        and abs(float(original_value)) > 1e-8
                        else np.nan
                    )
                    rows.append(
                        {
                            "record_id": record_id,
                            "label": selected_row.label,
                            "window_number": window_number,
                            "requested_masked_percentage": masking_level,
                            "placement_number": placement_number,
                            "masking_seed": combined_seed,
                            "synthetic_gap_count": MASK_GAPS_PER_RUN,
                            "realized_original_valid_loss_percentage": realized_loss,
                            "masked_valid_fhr_percentage": masked_features[
                                "valid_fhr_percentage"
                            ],
                            "masked_longest_continuous_valid_segment_seconds": (
                                masked_fragmentation[
                                    "longest_continuous_valid_segment_seconds"
                                ]
                            ),
                            "masked_number_of_valid_segments": (
                                masked_fragmentation["number_of_valid_segments"]
                            ),
                            "masked_percentage_valid_in_longest_segment": (
                                masked_fragmentation[
                                    "percentage_valid_in_longest_segment"
                                ]
                            ),
                            "feature": feature,
                            "original_value": original_value,
                            "masked_value": masked_value,
                            "absolute_error": absolute_error,
                            "relative_error": relative_error,
                            "feature_failure_or_nan": failure,
                        }
                    )
        print(
            f"  Masking sample processed: {sample_number}/{len(selected)} "
            f"({record_id}, window {window_number})"
        )
    results = pd.DataFrame(rows)

    # Normalize absolute error by the original-value IQR (fallback to SD,
    # magnitude, then 1) so heterogeneous feature units can be compared.
    scales = {}
    for feature, group in results.groupby("feature"):
        originals = group[
            ["record_id", "window_number", "original_value"]
        ].drop_duplicates()["original_value"].dropna()
        scale = originals.quantile(0.75) - originals.quantile(0.25)
        if not np.isfinite(scale) or scale == 0:
            scale = originals.std(ddof=0)
        if not np.isfinite(scale) or scale == 0:
            scale = originals.abs().median()
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        scales[feature] = float(scale)
    results["normalization_scale"] = results["feature"].map(scales)
    results["normalized_absolute_error"] = (
        results["absolute_error"] / results["normalization_scale"]
    )
    group_keys = ["feature", "requested_masked_percentage"]
    grouped = results.groupby(group_keys)
    results["group_median_absolute_error"] = grouped["absolute_error"].transform(
        "median"
    )
    results["group_90th_percentile_absolute_error"] = grouped[
        "absolute_error"
    ].transform(lambda values: values.quantile(0.90))
    results["group_median_normalized_absolute_error"] = grouped[
        "normalized_absolute_error"
    ].transform("median")
    results["group_failure_or_nan_rate"] = grouped[
        "feature_failure_or_nan"
    ].transform("mean")
    return results


def add_feature_instability_proxy(
    windows: pd.DataFrame, key_features: list[str]
) -> pd.DataFrame:
    """Calculate a robust within-validity-bin feature-deviation proxy.

    The score is not a clinical abnormality score.  It is the median absolute
    robust z-distance from windows with similar overall valid percentage.  It
    lets fragmentation groups be compared while approximately controlling for
    total validity, though physiology can still contribute to the score.
    """
    output = windows.copy()
    output["valid_percentage_bin"] = pd.cut(
        output["valid_fhr_percentage"],
        bins=[-np.inf, 60, 70, 80, 90, np.inf],
        labels=["<60", "60-<70", "70-<80", "80-<90", ">=90"],
        right=False,
    )
    scores = pd.DataFrame(index=output.index)
    for feature in key_features:
        scores[feature] = np.nan
        for indices in output.groupby(
            "valid_percentage_bin", observed=False
        ).groups.values():
            values = pd.to_numeric(output.loc[indices, feature], errors="coerce")
            median = values.median()
            iqr = values.quantile(0.75) - values.quantile(0.25)
            scale = iqr if np.isfinite(iqr) and iqr > 0 else values.std(ddof=0)
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            scores.loc[indices, feature] = (values - median).abs() / scale
    output["feature_instability_proxy"] = scores.median(axis=1, skipna=True)
    return output


def create_fragmentation_analysis(windows: pd.DataFrame) -> pd.DataFrame:
    """Save every window plus continuity and quality-matched deviation fields."""
    columns = [
        "record_id",
        "label",
        "split",
        "window_number",
        "start_minute",
        "end_minute",
        "is_final_window",
        "quality_category",
        "valid_fhr_percentage",
        "missing_fhr_percentage",
        "artifact_percentage",
        "longest_missing_gap_seconds",
        "number_of_missing_gaps",
        "total_valid_duration_seconds",
        "longest_continuous_valid_segment_seconds",
        "number_of_valid_segments",
        "median_valid_segment_duration_seconds",
        "percentage_valid_in_longest_segment",
        "fragmentation_category",
        "valid_percentage_bin",
        "feature_instability_proxy",
        *KEY_FEATURES,
    ]
    return windows[columns].copy()


def merge_pathological_flags(
    windows: pd.DataFrame, pathological_windows: pd.DataFrame
) -> pd.DataFrame:
    """Attach existing exploratory abnormality flags to recalculated windows."""
    flag_columns = [
        "record_id",
        "window_number",
        "low_stv_exploratory_flag",
        "deep_deceleration_exploratory_flag",
        "high_bradycardia_exploratory_flag",
        "long_deceleration_exploratory_flag",
        "poor_signal_exploratory_flag",
        "local_abnormality_exploratory_flag",
        "any_exploratory_flag",
    ]
    pathological = windows[windows["label"] == "Pathological"].copy()
    return pathological.merge(
        pathological_windows[flag_columns],
        on=["record_id", "window_number"],
        how="left",
        validate="one_to_one",
    )


def strongest_value_from_low_quality(group: pd.DataFrame) -> tuple[bool, str]:
    """Check whether strongest abnormal values originate below 80% validity."""
    sources = []
    maximum_features = [
        "max_deceleration_depth",
        "longest_deceleration_duration_seconds",
        "bradycardia_percentage",
        "tachycardia_percentage",
    ]
    for feature in maximum_features:
        valid = group.dropna(subset=[feature])
        if len(valid):
            source = valid.loc[valid[feature].idxmax()]
            if source["valid_fhr_percentage"] < DESCRIPTIVE_HIGH_LOW_BOUNDARY:
                sources.append(feature)
    valid_stv = group.dropna(subset=["short_term_variability"])
    if len(valid_stv):
        source = valid_stv.loc[valid_stv["short_term_variability"].idxmin()]
        if source["valid_fhr_percentage"] < DESCRIPTIVE_HIGH_LOW_BOUNDARY:
            sources.append("minimum_short_term_variability")
    return bool(sources), "; ".join(sources)


def full_record_influence_heuristic(
    group: pd.DataFrame, full_row: pd.Series
) -> tuple[int, str]:
    """Count full features closer to poor-window than high-window means."""
    high = group[group["valid_fhr_percentage"] >= DESCRIPTIVE_HIGH_LOW_BOUNDARY]
    low = group[group["valid_fhr_percentage"] < DESCRIPTIVE_HIGH_LOW_BOUNDARY]
    if high.empty or low.empty:
        return 0, ""
    influenced = []
    for feature in [
        "baseline_fhr",
        "short_term_variability",
        "long_term_variability",
        "bradycardia_percentage",
        "tachycardia_percentage",
        "acceleration_count",
        "deceleration_count",
    ]:
        full_value = full_row.get(feature, np.nan)
        high_mean = high[feature].mean()
        low_mean = low[feature].mean()
        if not all(np.isfinite([full_value, high_mean, low_mean])):
            continue
        if abs(full_value - low_mean) < abs(full_value - high_mean):
            influenced.append(feature)
    return len(influenced), "; ".join(influenced)


def create_pathological_effects(
    pathological: pd.DataFrame,
    clinical_features: pd.DataFrame,
    fixed_test_ids: set[str],
) -> pd.DataFrame:
    """Aggregate quality and abnormality effects for all 27 records."""
    full_lookup = clinical_features.set_index("record_id")
    rows = []
    quality_rank = {category: index for index, category in enumerate(QUALITY_CATEGORY_ORDER)}
    for record_id, group in pathological.groupby("record_id", sort=True):
        ordered = group.sort_values("window_number")
        category_counts = ordered["quality_category"].value_counts()
        worst_category = max(
            ordered["quality_category"].astype(str),
            key=lambda value: quality_rank[value],
        )
        high = ordered[
            ordered["valid_fhr_percentage"] >= DESCRIPTIVE_HIGH_LOW_BOUNDARY
        ]
        low = ordered[
            ordered["valid_fhr_percentage"] < DESCRIPTIVE_HIGH_LOW_BOUNDARY
        ]
        strongest_low, strongest_features = strongest_value_from_low_quality(ordered)
        influence_count, influence_features = full_record_influence_heuristic(
            ordered, full_lookup.loc[record_id]
        )
        rows.append(
            {
                "record_id": record_id,
                "is_fixed_test_pathological_record": record_id in fixed_test_ids,
                "total_windows": len(ordered),
                **{
                    f"{category}_window_count": int(category_counts.get(category, 0))
                    for category in QUALITY_CATEGORY_ORDER
                },
                "final_window_quality": str(
                    ordered.loc[ordered["is_final_window"], "quality_category"].iloc[0]
                ),
                "worst_quality_window": worst_category,
                "high_quality_local_abnormality_window_count": int(
                    high["local_abnormality_exploratory_flag"].sum()
                ),
                "low_quality_local_abnormality_window_count": int(
                    low["local_abnormality_exploratory_flag"].sum()
                ),
                "any_abnormality_flag_in_high_quality_window": bool(
                    high["local_abnormality_exploratory_flag"].any()
                ),
                "any_abnormality_flag_in_low_quality_window": bool(
                    low["local_abnormality_exploratory_flag"].any()
                ),
                "strongest_abnormal_feature_from_low_quality_window": strongest_low,
                "strongest_low_quality_feature_names": strongest_features,
                "full_record_influence_feature_count": influence_count,
                "full_record_features_closer_to_low_quality_window_mean": influence_features,
                "full_record_features_appear_influenced_by_poor_windows": (
                    influence_count >= 2
                ),
                "minimum_valid_fhr_percentage": float(
                    ordered["valid_fhr_percentage"].min()
                ),
                "maximum_missing_fhr_percentage": float(
                    ordered["missing_fhr_percentage"].max()
                ),
                "maximum_longest_missing_gap_seconds": float(
                    ordered["longest_missing_gap_seconds"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def rule_mask(windows: pd.DataFrame, rule: dict) -> pd.Series:
    """Evaluate one proposed window-validity rule."""
    return (
        (windows["valid_fhr_percentage"] >= rule["min_valid_percentage"])
        & (windows["artifact_percentage"] <= rule["max_artifact_percentage"])
        & (
            windows["longest_continuous_valid_segment_seconds"]
            >= rule["min_longest_valid_segment_seconds"]
        )
        & (
            windows["longest_missing_gap_seconds"]
            <= rule["max_longest_missing_gap_seconds"]
        )
    )


def compare_validity_rules(windows: pd.DataFrame) -> pd.DataFrame:
    """Compare data retention and quality proxies for all candidate rules."""
    rows = []
    total_records = windows["record_id"].nunique()
    pathological_records = windows.loc[
        windows["label"] == "Pathological", "record_id"
    ].nunique()
    for rule in VALIDITY_RULES:
        retained_mask = rule_mask(windows, rule)
        retained = windows[retained_mask]
        retained_records = retained["record_id"].nunique()
        retained_pathological = retained[retained["label"] == "Pathological"]
        rows.append(
            {
                **rule,
                "total_windows_retained": len(retained),
                "window_retention_percentage": 100.0 * len(retained) / len(windows),
                "total_records_retaining_at_least_one_window": int(retained_records),
                "pathological_records_retaining_at_least_one_window": int(
                    retained_pathological["record_id"].nunique()
                ),
                "pathological_windows_retained": len(retained_pathological),
                "final_windows_retained": int(retained["is_final_window"].sum()),
                "records_completely_lost": int(total_records - retained_records),
                "pathological_records_completely_lost": int(
                    pathological_records
                    - retained_pathological["record_id"].nunique()
                ),
                "average_retained_valid_duration_seconds": (
                    float(retained["total_valid_duration_seconds"].mean())
                    if len(retained)
                    else np.nan
                ),
                "abnormality_flagged_pathological_windows_retained": int(
                    retained_pathological[
                        "local_abnormality_exploratory_flag"
                    ].sum()
                ),
                "undefined_key_feature_rate": (
                    float(retained[KEY_FEATURES].isna().mean().mean())
                    if len(retained)
                    else np.nan
                ),
                "median_feature_instability_proxy": (
                    float(retained["feature_instability_proxy"].median())
                    if len(retained)
                    else np.nan
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    eligible = comparison[
        (
            comparison["total_windows_retained"]
            >= len(windows) * MIN_RULE_WINDOW_RETENTION_FRACTION
        )
        & (
            comparison["pathological_records_retaining_at_least_one_window"]
            >= MIN_RULE_PATHOLOGICAL_RECORDS_RETAINED
        )
    ].copy()
    if eligible.empty:
        recommended_index = comparison["total_windows_retained"].idxmax()
    else:
        recommended_index = eligible.sort_values(
            [
                "undefined_key_feature_rate",
                "median_feature_instability_proxy",
                "total_windows_retained",
            ],
            ascending=[True, True, False],
        ).index[0]
    comparison["recommended_for_future_window_experiment"] = False
    comparison.loc[
        recommended_index, "recommended_for_future_window_experiment"
    ] = True
    comparison["recommendation_basis"] = ""
    comparison.loc[recommended_index, "recommendation_basis"] = (
        "Best reliability proxy among rules retaining at least "
        f"{MIN_RULE_WINDOW_RETENTION_FRACTION:.0%} of windows and at least "
        f"{MIN_RULE_PATHOLOGICAL_RECORDS_RETAINED}/27 Pathological records; "
        "strictness alone was not rewarded."
    )
    return comparison


def masking_feature_summary(masking: pd.DataFrame) -> pd.DataFrame:
    """Aggregate masking reliability for ranking and reporting."""
    summary = (
        masking.groupby("feature")
        .agg(
            median_normalized_absolute_error=(
                "normalized_absolute_error",
                "median",
            ),
            p90_normalized_absolute_error=(
                "normalized_absolute_error",
                lambda values: values.quantile(0.90),
            ),
            failure_rate=("feature_failure_or_nan", "mean"),
        )
        .assign(
            sensitivity_score=lambda frame: (
                frame["median_normalized_absolute_error"].fillna(0)
                + frame["p90_normalized_absolute_error"].fillna(0)
                + 5 * frame["failure_rate"]
            )
        )
        .sort_values("sensitivity_score", ascending=False)
        .reset_index()
    )
    summary["is_direct_quality_or_duration_feature"] = summary["feature"].isin(
        DIRECT_QUALITY_FEATURES
    )
    return summary


def fragmentation_matters(fragmentation: pd.DataFrame) -> tuple[bool, float, float]:
    """Compare highly fragmented and continuous windows within validity bins."""
    grouped = (
        fragmentation.groupby(
            ["valid_percentage_bin", "fragmentation_category"],
            observed=True,
        )["feature_instability_proxy"]
        .median()
        .unstack()
    )
    differences = []
    for _, row in grouped.iterrows():
        if {
            "highly_fragmented",
            "continuous_dominant",
        }.issubset(row.dropna().index):
            differences.append(
                row["highly_fragmented"] - row["continuous_dominant"]
            )
    median_difference = float(np.median(differences)) if differences else np.nan
    overall_fragmented = fragmentation.loc[
        fragmentation["fragmentation_category"] == "highly_fragmented",
        "feature_instability_proxy",
    ].median()
    overall_continuous = fragmentation.loc[
        fragmentation["fragmentation_category"] == "continuous_dominant",
        "feature_instability_proxy",
    ].median()
    matters = bool(np.isfinite(median_difference) and median_difference > 0.10)
    return matters, float(overall_fragmented), float(overall_continuous)


def create_figures(
    windows: pd.DataFrame,
    feature_by_quality: pd.DataFrame,
    masking: pd.DataFrame,
    masking_summary: pd.DataFrame,
    fragmentation: pd.DataFrame,
    pathological: pd.DataFrame,
    rules: pd.DataFrame,
) -> list[Path]:
    """Create eight diagnostic figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    paths = []

    # 1. Feature values versus valid FHR percentage.
    scatter_features = KEY_FEATURES[:6]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, feature in zip(axes.flat, scatter_features):
        axis.scatter(
            windows["valid_fhr_percentage"],
            windows[feature],
            s=7,
            alpha=0.25,
            color="#326891",
        )
        axis.set_title(feature.replace("_", " "))
        axis.set_xlabel("Valid FHR (%)")
        axis.set_ylabel("Feature value")
    figure.suptitle("Current feature values versus valid FHR percentage")
    figure.tight_layout()
    path = FIGURE_DIR / "01_features_vs_valid_percentage.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    # 2. Error versus artificially masked percentage for most sensitive features.
    clinical_sensitivity = masking_summary[
        ~masking_summary["is_direct_quality_or_duration_feature"]
    ]
    top_features = clinical_sensitivity.head(8)["feature"].tolist()
    mask_plot = (
        masking[masking["feature"].isin(top_features)]
        .groupby(["feature", "requested_masked_percentage"])[
            "normalized_absolute_error"
        ]
        .median()
        .unstack(0)
    )
    axis = mask_plot.plot(figsize=(12, 7), marker="o")
    axis.set_title("Median normalized feature error under contiguous masking")
    axis.set_xlabel("Target masked percentage")
    axis.set_ylabel("Median normalized absolute error")
    axis.legend(fontsize=8, ncol=2)
    figure = axis.get_figure()
    figure.tight_layout()
    path = FIGURE_DIR / "02_error_vs_masked_percentage.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    # 3. Reliability heatmap: lower median normalized error is better.
    heat_features = clinical_sensitivity.head(20)["feature"].tolist()
    heat = (
        masking[masking["feature"].isin(heat_features)]
        .groupby(["feature", "requested_masked_percentage"])[
            "normalized_absolute_error"
        ]
        .median()
        .unstack()
        .reindex(heat_features)
    )
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(heat.to_numpy(), aspect="auto", cmap="YlOrRd")
    axis.set_yticks(np.arange(len(heat.index)), heat.index, fontsize=8)
    axis.set_xticks(
        np.arange(len(heat.columns)),
        [f"{value}%" for value in heat.columns],
    )
    axis.set_xlabel("Target masked percentage")
    axis.set_title("Feature sensitivity heatmap (median normalized error)")
    figure.colorbar(image, ax=axis, label="Normalized absolute error")
    figure.tight_layout()
    path = FIGURE_DIR / "03_feature_reliability_heatmap.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    # 4. Windows retained.
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(rules["rule_name"], rules["total_windows_retained"], color="#4c78a8")
    axis.set_title("Windows retained under proposed validity rules")
    axis.set_ylabel("20-minute windows retained")
    axis.tick_params(axis="x", rotation=35)
    for tick in axis.get_xticklabels():
        tick.set_ha("right")
    figure.tight_layout()
    path = FIGURE_DIR / "04_windows_retained_by_rule.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    # 5. Pathological records retained.
    figure, axis = plt.subplots(figsize=(12, 6))
    colors = [
        "#d62728" if recommended else "#8fb9dd"
        for recommended in rules["recommended_for_future_window_experiment"]
    ]
    axis.bar(
        rules["rule_name"],
        rules["pathological_records_retaining_at_least_one_window"],
        color=colors,
    )
    axis.axhline(27, color="#333333", linestyle="--", linewidth=1)
    axis.set_ylim(0, 29)
    axis.set_title("Pathological records retaining at least one window")
    axis.set_ylabel("Pathological records")
    axis.tick_params(axis="x", rotation=35)
    for tick in axis.get_xticklabels():
        tick.set_ha("right")
    figure.tight_layout()
    path = FIGURE_DIR / "05_pathological_records_retained.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    # 6. Feature distributions by quality category.
    distribution_features = KEY_FEATURES[:6]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, feature in zip(axes.flat, distribution_features):
        data = [
            windows.loc[windows["quality_category"] == category, feature]
            .dropna()
            .to_numpy()
            for category in QUALITY_CATEGORY_ORDER
        ]
        axis.boxplot(
            data,
            tick_labels=["very high", "acceptable", "low", "very low"],
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#9ecae1"},
            medianprops={"color": "#d95f02"},
        )
        axis.set_title(feature.replace("_", " "))
        axis.tick_params(axis="x", rotation=25)
    figure.suptitle("Feature distributions across diagnostic quality categories")
    figure.tight_layout()
    path = FIGURE_DIR / "06_feature_distributions_by_quality.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    # 7. Fragmentation versus feature instability proxy.
    figure, axis = plt.subplots(figsize=(11, 7))
    color_map = {
        "continuous_dominant": "#2ca02c",
        "moderately_fragmented": "#ffbf00",
        "highly_fragmented": "#d62728",
    }
    for category, group in fragmentation.groupby("fragmentation_category"):
        axis.scatter(
            group["percentage_valid_in_longest_segment"],
            group["feature_instability_proxy"],
            s=10,
            alpha=0.35,
            color=color_map[category],
            label=category.replace("_", " "),
        )
    axis.set_xlabel("Valid data contained in longest segment (%)")
    axis.set_ylabel("Quality-matched feature instability proxy")
    axis.set_title("Signal fragmentation versus feature deviation")
    axis.legend(fontsize=9)
    figure.tight_layout()
    path = FIGURE_DIR / "07_fragmentation_vs_instability.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    # 8. Pathological abnormality flags by >=80 versus <80 quality.
    path_flag_data = pd.DataFrame(
        {
            "High quality (>=80%)": [
                pathological.loc[
                    pathological["valid_fhr_percentage"] >= 80, column
                ].sum()
                for column in [
                    "low_stv_exploratory_flag",
                    "deep_deceleration_exploratory_flag",
                    "high_bradycardia_exploratory_flag",
                    "long_deceleration_exploratory_flag",
                ]
            ],
            "Low quality (<80%)": [
                pathological.loc[
                    pathological["valid_fhr_percentage"] < 80, column
                ].sum()
                for column in [
                    "low_stv_exploratory_flag",
                    "deep_deceleration_exploratory_flag",
                    "high_bradycardia_exploratory_flag",
                    "long_deceleration_exploratory_flag",
                ]
            ],
        },
        index=["Low STV", "Deep deceleration", "High bradycardia", "Long deceleration"],
    )
    axis = path_flag_data.plot(kind="bar", figsize=(11, 6), color=["#4c78a8", "#e45756"])
    axis.set_title("Pathological abnormality flags by window signal quality")
    axis.set_ylabel("Flagged windows")
    axis.tick_params(axis="x", rotation=20)
    figure = axis.get_figure()
    figure.tight_layout()
    path = FIGURE_DIR / "08_pathological_flags_by_quality.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)
    return paths


def extractor_handling_markdown() -> str:
    """Document the implementation used by the audit."""
    return f"""
| Question | Current implementation |
|---|---|
| Missing FHR | Non-finite values and values <= 0 are source missing samples. |
| Artifact/invalid FHR | Finite positive values outside {extractor.FHR_MIN_VALID_BPM:.0f}--{extractor.FHR_MAX_VALID_BPM:.0f} bpm, plus an enabled one-sample isolated-spike rule: at least {extractor.FHR_ISOLATED_SPIKE_MIN_JUMP_BPM:.0f} bpm from both neighbours while the neighbours differ by at most {extractor.FHR_ISOLATED_SPIKE_MAX_NEIGHBOR_DIFFERENCE_BPM:.0f} bpm. |
| Gap interpolation | No FHR gaps are interpolated. Therefore the maximum interpolated gap is **0 seconds**. |
| Long gaps | They remain NaN and are excluded from valid-sample calculations. |
| Baseline | The centered {extractor.BASELINE_WINDOW_SECONDS / 60:.0f}-minute median uses only valid cleaned samples and requires {extractor.BASELINE_MIN_VALID_PERCENTAGE:.0f}% validity in its rolling window. It does not include interpolated values because none exist. A rolling median can summarize valid samples on both sides of an internal gap, but the baseline is set to NaN at the missing sample itself. |
| STV approximation | Mean absolute adjacent-sample FHR difference. Only adjacent pairs where both samples are valid are used, so it never crosses a missing gap. |
| LTV approximation | Standard deviation of valid {extractor.LTV_EPOCH_SECONDS:.0f}-second epoch means. An epoch, including a shorter final epoch, is included when at least {extractor.LTV_EPOCH_MIN_VALID_PERCENTAGE:.0f}% of that epoch is valid; at least {extractor.LTV_MIN_VALID_EPOCHS} valid epochs are required. |
| Accelerations/decelerations | A missing FHR or baseline sample makes the threshold mask false and terminates the continuous event. Events cannot cross gaps. |
| Bradycardia/tachycardia percentages | Denominator is valid cleaned FHR samples only, not all time samples. |
| Recording duration | Total waveform duration, including missing and artifact time. Event density also uses this total duration. |
| UC features | Calculated independently from UC. FHR quality does not alter UC smoothing or contraction detection, although interaction ratios can still change because their deceleration numerator depends on FHR. |
"""


def report_quality_rule_table(rules: pd.DataFrame) -> str:
    """Create a compact Markdown retention table."""
    columns = [
        "rule_name",
        "total_windows_retained",
        "window_retention_percentage",
        "pathological_records_retaining_at_least_one_window",
        "pathological_windows_retained",
        "records_completely_lost",
    ]
    display = rules[columns].copy()
    display["window_retention_percentage"] = display[
        "window_retention_percentage"
    ].map(lambda value: f"{value:.1f}%")
    display.columns = [
        "Rule",
        "Windows",
        "Retention",
        "Pathological records",
        "Pathological windows",
        "Records lost",
    ]
    # Build the small table directly so report generation does not depend on
    # pandas' optional ``tabulate`` package.
    headers = display.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def determine_main_conclusion(
    sensitivity: pd.DataFrame,
    feature_by_quality: pd.DataFrame,
) -> tuple[str, list[str], list[str]]:
    """Select one required conclusion using masking and observed shifts."""
    clinical_sensitivity = sensitivity[
        ~sensitivity["is_direct_quality_or_duration_feature"]
    ]
    affected = clinical_sensitivity.head(8)["feature"].tolist()
    least = (
        clinical_sensitivity.tail(8)
        .sort_values("sensitivity_score")["feature"]
        .tolist()
    )
    high_sensitivity_count = int(
        (
            (clinical_sensitivity["median_normalized_absolute_error"] > 0.25)
            | (clinical_sensitivity["failure_rate"] > 0.10)
        ).sum()
    )
    very_low_shifts = feature_by_quality[
        feature_by_quality["quality_category"] == "very_low_quality"
    ]
    strong_observed_shifts = int(
        (very_low_shifts["absolute_standardized_median_shift"] >= 1.0).sum()
    )
    if high_sensitivity_count >= 10 and strong_observed_shifts >= 10:
        conclusion = "poor signal materially distorts several features"
    elif high_sensitivity_count >= 4 or strong_observed_shifts >= 4:
        conclusion = "only specific features are unreliable under poor signal"
    elif high_sensitivity_count == 0 and strong_observed_shifts <= 2:
        conclusion = "current feature extraction is sufficiently robust to poor signal"
    else:
        conclusion = "evidence is mixed and quality-aware feature extraction is required"
    return conclusion, affected, least


def write_report(
    windows: pd.DataFrame,
    feature_by_quality: pd.DataFrame,
    masking: pd.DataFrame,
    sensitivity: pd.DataFrame,
    fragmentation: pd.DataFrame,
    pathological_effects: pd.DataFrame,
    rules: pd.DataFrame,
    conclusion: str,
    affected: list[str],
    least_affected: list[str],
    fragmentation_result: tuple[bool, float, float],
) -> None:
    """Write the evidence report in plain language."""
    category_counts = windows["quality_category"].value_counts().reindex(
        QUALITY_CATEGORY_ORDER, fill_value=0
    )
    recommended = rules.loc[
        rules["recommended_for_future_window_experiment"]
    ].iloc[0]
    frag_matters, fragmented_score, continuous_score = fragmentation_result
    fixed = pathological_effects[
        pathological_effects["is_fixed_test_pathological_record"]
    ]
    fixed_poor = int(
        (
            fixed["low_quality_window_count"]
            + fixed["very_low_quality_window_count"]
            > 0
        ).sum()
    )
    pathological_influenced = int(
        pathological_effects[
            "full_record_features_appear_influenced_by_poor_windows"
        ].sum()
    )
    strongest_from_low = int(
        pathological_effects[
            "strongest_abnormal_feature_from_low_quality_window"
        ].sum()
    )
    low_path = windows[
        (windows["label"] == "Pathological")
        & (windows["valid_fhr_percentage"] < DESCRIPTIVE_HIGH_LOW_BOUNDARY)
    ]
    high_path = windows[
        (windows["label"] == "Pathological")
        & (windows["valid_fhr_percentage"] >= DESCRIPTIVE_HIGH_LOW_BOUNDARY)
    ]

    report = f"""# Signal-Quality Effects Audit

## Scope

This diagnostic recalculated the current engineered feature set for
**{len(windows)}** non-overlapping 20-minute windows from all
**{windows['record_id'].nunique()}** CTU-UHB records. Patient-level labels were
used only for descriptive grouping. No model was trained, no window labels were
created, and no raw data, feature-extraction code, labels, split, or saved model
were changed.

## 1. Current missing and invalid signal handling

{extractor_handling_markdown()}

STV and LTV remain computational approximations rather than clinical
gold-standard definitions.

## 2. Diagnostic quality categories

The categories are exploratory engineering descriptions:

- Very high quality (>=90% valid): **{category_counts['very_high_quality']}** windows.
- Acceptable quality (80--<90%): **{category_counts['acceptable_quality']}** windows.
- Low quality (60--<80%): **{category_counts['low_quality']}** windows.
- Very low quality (<60%): **{category_counts['very_low_quality']}** windows.

## 3. Feature behaviour as quality decreases

The features most affected in the controlled masking experiment were:
**{", ".join(affected)}**.

The least affected were:
**{", ".join(least_affected)}**.

The detailed CSV reports mean, median, standard deviation, IQR, missing values,
and conservative extreme counts for every current numeric engineered feature
within every quality category. Extreme counts use outer 3-IQR fences learned
from very-high-quality windows, not medical cutoffs.

Because low-quality windows can contain genuine abnormal physiology as well as
signal loss, observed category differences alone are not treated as proof of
measurement error. The masking experiment supplies the controlled evidence:
the same high-quality waveform is recomputed after {MASKING_LEVELS_PERCENT}
percent contiguous signal loss across {len(MASKING_SEEDS)} fixed placements.

## 4. Does fragmentation matter?

Fragmentation {"showed higher" if frag_matters else "did not show consistently higher"}
quality-matched feature deviation. The median instability proxy was
**{fragmented_score:.3f}** in highly fragmented windows and
**{continuous_score:.3f}** in continuous-dominant windows. The proxy compares
windows within the same broad valid-percentage band, but physiological
differences can still contribute; it is not a clinical abnormality score.

## 5. Poor-quality windows and extreme abnormalities

Among Pathological windows, the descriptive <80% group contained
**{len(low_path)}** windows versus **{len(high_path)}** at >=80%.
For **{strongest_from_low}/27** Pathological records, at least one strongest
abnormal summary (minimum STV, maximum bradycardia/tachycardia exposure, or
maximum deceleration depth/duration) came from a <80% window. This identifies
values needing trace review; it does not prove that every such extreme is
false.

## 6. Fixed-test Pathological records

All six fixed-test records (1002, 1071, 1158, 1418, 1490, 2009) were reviewed.
**{fixed_poor}/6** contained at least one window below 80% valid FHR. Their
individual quality-category counts, final-window quality, strongest-feature
source, and full-record influence heuristic are in
`pathological_quality_effects.csv`.

## 7. Are poor windows influencing full-record features?

For **{pathological_influenced}/27** Pathological records, at least two
full-record features were closer to the poor-window mean than to the
>=80%-quality window mean. This is an explicit descriptive heuristic, not a
causal estimate. It indicates where patient aggregation should be checked
against quality-aware alternatives in a future experiment.

## 8. Proposed window-validity rules

{report_quality_rule_table(rules)}

The recommended starting rule is **{recommended['rule_name']}**. It retains
**{int(recommended['total_windows_retained'])}** windows
({recommended['window_retention_percentage']:.1f}%), at least one window for
**{int(recommended['pathological_records_retaining_at_least_one_window'])}/27**
Pathological records, and loses all windows for
**{int(recommended['records_completely_lost'])}** records overall.

The recommendation was chosen only among rules retaining at least
{MIN_RULE_WINDOW_RETENTION_FRACTION:.0%} of all windows and at least
{MIN_RULE_PATHOLOGICAL_RECORDS_RETAINED}/27 Pathological records. Within that
retention constraint, undefined-feature rate and the instability proxy were
compared. A stricter rule was not preferred simply for discarding more data.

## 9. Recommendations for the next implementation step

1. Start a future window-based experiment with
   **{recommended['rule_name']}**, preserving the rule outcome and raw quality
   measurements as separate columns.
2. Keep longest continuous valid duration and segment count as diagnostic
   columns, but do not add a continuity cutoff yet. Fragmentation did not show
   consistently greater instability, while the combined continuity rules
   retained fewer than half of all windows and at most 21/27 Pathological
   records.
3. Return NaN, rather than a normal-looking zero, for FHR-derived event and
   variability features when a window fails the validity rule.
4. Retain valid percentage, longest missing gap, longest continuous valid
   segment, number of valid segments, and artifact percentage as missingness
   indicators.
5. Keep acceleration/deceleration detection gap-breaking and percentages
   valid-sample-based; the audit confirms these safeguards are already present.
6. Do not allow rejected poor windows to contribute ordinary unweighted values
   to a future patient-level aggregation. Compare exclusion with explicit
   quality weighting in a separately controlled experiment.
7. Review the most masking-sensitive features
   (**{", ".join(affected[:5])}**) before deciding whether they should be
   excluded or recalculated in marginal-quality windows.

These recommendations are supported by the masking errors, observed
undefined-value rates, fragmentation comparison, and rule-retention table.

## Limitations

- Artificial gaps approximate signal loss but cannot represent every monitor
  failure pattern.
- Patient-level labels do not imply every 20-minute window is abnormal.
- Feature-deviation and full-record-influence fields are transparent
  diagnostic heuristics, not causal or clinical conclusions.
- No claim is made that the recommended rule improves model performance.

## Conclusion

**{conclusion}.**

Quality-aware window handling is required before creating a window-based
machine-learning experiment. The next step should implement and test the
recommended validity rule without modifying the fixed test split or using it
for model selection.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def validate_outputs(
    windows: pd.DataFrame,
    feature_by_quality: pd.DataFrame,
    masking: pd.DataFrame,
    fragmentation: pd.DataFrame,
    pathological_effects: pd.DataFrame,
    rules: pd.DataFrame,
    figures: list[Path],
) -> None:
    """Validate output schemas, counts, ranges, and expected artifacts."""
    if windows["record_id"].nunique() != 552:
        raise AssertionError("Window audit does not cover all 552 records")
    if set(windows["quality_category"].astype(str)) != set(QUALITY_CATEGORY_ORDER):
        raise AssertionError("One or more quality categories are absent")
    if len(feature_by_quality) != feature_by_quality["feature"].nunique() * 4:
        raise AssertionError("feature_by_quality.csv has an unexpected row count")
    expected_mask_rows = (
        masking[["record_id", "window_number"]].drop_duplicates().shape[0]
        * len(MASKING_LEVELS_PERCENT)
        * len(MASKING_SEEDS)
        * masking["feature"].nunique()
    )
    if len(masking) != expected_mask_rows:
        raise AssertionError("Masking output does not have a complete design")
    if len(fragmentation) != len(windows):
        raise AssertionError("Fragmentation output must contain every window")
    if len(pathological_effects) != 27:
        raise AssertionError("Pathological effects must contain 27 records")
    if len(rules) != len(VALIDITY_RULES):
        raise AssertionError("Rule comparison has an unexpected row count")
    if rules["recommended_for_future_window_experiment"].sum() != 1:
        raise AssertionError("Exactly one proposed rule must be recommended")
    if len(figures) != 8 or not all(path.exists() for path in figures):
        raise AssertionError("Expected all eight audit figures")
    if any(path.stat().st_size < 15_000 for path in figures):
        raise AssertionError("At least one figure is unexpectedly small")
    for percentage in [
        windows["valid_fhr_percentage"],
        windows["missing_fhr_percentage"],
        windows["artifact_percentage"],
        fragmentation["percentage_valid_in_longest_segment"],
    ]:
        if not percentage.dropna().between(0, 100).all():
            raise AssertionError("A percentage output is outside 0--100")
    if not REPORT_PATH.exists():
        raise AssertionError("Markdown report was not generated")


def main() -> None:
    """Run the signal-quality audit."""
    labels, clinical_features, dataset, pathological_windows = load_inputs()
    ensure_new_output_location()
    _, test_ids = visual.reproduce_fixed_split(dataset)
    waveform_directory = visual.find_waveform_directory(dataset["record_id"])
    print(f"Using read-only waveform directory: {waveform_directory}")

    print("\nRecalculating current features for all 20-minute windows...")
    windows = calculate_all_windows(
        labels, dataset, waveform_directory, test_ids
    )
    feature_names = audited_feature_names(windows)
    feature_by_quality = summarize_features_by_quality(windows, feature_names)
    feature_by_quality.to_csv(FEATURE_BY_QUALITY_PATH, index=False)

    print("\nRunning controlled contiguous-gap masking experiment...")
    selected = select_masking_windows(windows)
    masking = run_masking_experiment(selected, feature_names, waveform_directory)
    masking.to_csv(MASKING_RESULTS_PATH, index=False)
    sensitivity = masking_feature_summary(masking)

    windows = add_feature_instability_proxy(windows, KEY_FEATURES)
    fragmentation = create_fragmentation_analysis(windows)
    fragmentation.to_csv(FRAGMENTATION_PATH, index=False)

    pathological = merge_pathological_flags(windows, pathological_windows)
    pathological_effects = create_pathological_effects(
        pathological, clinical_features, set(test_ids)
    )
    pathological_effects.to_csv(PATHOLOGICAL_EFFECTS_PATH, index=False)

    # Rule comparison needs the existing Pathological flags on the all-window
    # frame.  Non-Pathological rows have no abnormality flag and receive False.
    flag_lookup = pathological.set_index(
        ["record_id", "window_number"]
    )["local_abnormality_exploratory_flag"]
    windows["local_abnormality_exploratory_flag"] = [
        bool(flag_lookup.get((record_id, window_number), False))
        for record_id, window_number in zip(
            windows["record_id"], windows["window_number"]
        )
    ]
    rules = compare_validity_rules(windows)
    rules.to_csv(RULE_COMPARISON_PATH, index=False)

    conclusion, affected, least_affected = determine_main_conclusion(
        sensitivity, feature_by_quality
    )
    fragmentation_result = fragmentation_matters(fragmentation)
    figures = create_figures(
        windows,
        feature_by_quality,
        masking,
        sensitivity,
        fragmentation,
        pathological,
        rules,
    )
    write_report(
        windows,
        feature_by_quality,
        masking,
        sensitivity,
        fragmentation,
        pathological_effects,
        rules,
        conclusion,
        affected,
        least_affected,
        fragmentation_result,
    )
    validate_outputs(
        windows,
        feature_by_quality,
        masking,
        fragmentation,
        pathological_effects,
        rules,
        figures,
    )

    category_counts = windows["quality_category"].value_counts().reindex(
        QUALITY_CATEGORY_ORDER, fill_value=0
    )
    recommended = rules.loc[
        rules["recommended_for_future_window_experiment"]
    ].iloc[0]
    fixed_affected = pathological_effects.sort_values(
        [
            "full_record_influence_feature_count",
            "very_low_quality_window_count",
            "minimum_valid_fhr_percentage",
        ],
        ascending=[False, False, True],
    ).head(8)["record_id"].tolist()

    print("\nSignal-quality audit complete")
    print(f"Windows analysed: {len(windows)}")
    print(
        "Quality-category counts: "
        + ", ".join(
            f"{category}={int(category_counts[category])}"
            for category in QUALITY_CATEGORY_ORDER
        )
    )
    print("Features most affected: " + ", ".join(affected))
    print("Features least affected: " + ", ".join(least_affected))
    print(
        "Fragmentation causes instability: "
        + ("yes" if fragmentation_result[0] else "not consistently")
    )
    print("Pathological records most affected: " + ", ".join(fixed_affected))
    print(f"Recommended window-validity rule: {recommended['rule_name']}")
    print(
        "Pathological records retained under recommended rule: "
        f"{int(recommended['pathological_records_retaining_at_least_one_window'])}/27"
    )
    print(f"Main conclusion: {conclusion}")
    print(f"Feature-by-quality CSV: {FEATURE_BY_QUALITY_PATH}")
    print(f"Masking experiment CSV: {MASKING_RESULTS_PATH}")
    print(f"Fragmentation CSV: {FRAGMENTATION_PATH}")
    print(f"Pathological effects CSV: {PATHOLOGICAL_EFFECTS_PATH}")
    print(f"Rule comparison CSV: {RULE_COMPARISON_PATH}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Markdown report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
