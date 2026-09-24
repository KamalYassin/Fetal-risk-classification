"""Extract reproducible, computational CTG features from raw FHR and UC.

This is feature extraction only; it does not train a model.  The event and
variability measurements below are pragmatic signal-processing approximations,
not clinical gold-standard interpretations.  In particular, clinical review
must not be replaced by the values produced here.

Run from the project root::

    python src/features/extract_clinical_features.py
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"

# ---------------------------------------------------------------------------
# Configurable preprocessing and feature thresholds
# ---------------------------------------------------------------------------
# FHR outside this inclusive range is treated as a possible artifact. The broad
# 50--210 bpm range catches implausible measurements while retaining
# unusual but potentially physiological values. Removed values are
# retained in ``raw_fhr`` and counted in possible_fhr_artifact_percentage.
FHR_MIN_VALID_BPM = 50.0
FHR_MAX_VALID_BPM = 210.0

# Remove a one-sample isolated spike when it differs from both immediately
# adjacent valid samples by at least 25 bpm, while those two neighbors differ
# by no more than 10 bpm.  This targets abrupt impulses without erasing a
# sustained physiological change.  Set ENABLE_ISOLATED_SPIKE_FILTER to False
# to retain these samples.
ENABLE_ISOLATED_SPIKE_FILTER = True
FHR_ISOLATED_SPIKE_MIN_JUMP_BPM = 25.0
FHR_ISOLATED_SPIKE_MAX_NEIGHBOR_DIFFERENCE_BPM = 10.0

# The pointwise baseline is a centered 5-minute moving median.  A baseline is
# emitted only where that full window contains at least 50% valid cleaned FHR;
# therefore events cannot be classified in data-poor windows.  The CSV summary
# baseline_fhr is the median of all valid pointwise baseline values.
BASELINE_WINDOW_SECONDS = 5 * 60
BASELINE_MIN_VALID_PERCENTAGE = 50.0

# Acceleration/deceleration defaults implement a 15 bpm amplitude/depth and
# 15-second continuous-duration rule.  NaN FHR or NaN baseline samples always
# terminate an event; gaps are never interpolated for event detection.
ACCELERATION_THRESHOLD_BPM = 15.0
DECELERATION_THRESHOLD_BPM = 15.0
MIN_EVENT_DURATION_SECONDS = 15.0

# Short-term variability (STV) is the mean absolute difference between
# adjacent, continuously valid samples.  Long-term variability (LTV) is the
# standard deviation of valid non-overlapping 60-second epoch mean FHR values.
# An epoch needs 50% valid cleaned FHR and LTV needs at least two valid epochs.
# Both are computational approximations, not clinical gold-standard STV/LTV.
LTV_EPOCH_SECONDS = 60.0
LTV_EPOCH_MIN_VALID_PERCENTAGE = 50.0
LTV_MIN_VALID_EPOCHS = 2

# UC is median-smoothed over 10 seconds.  A contraction peak must be at least
# 10 UC units above the recording median, have 10 units of prominence, be at
# least 60 seconds from another retained peak, and have a half-prominence width
# of at least 20 seconds.  These requirements prevent narrow noise spikes from
# being counted.  UC units are dataset-specific, not calibrated pressure.
UC_SMOOTHING_SECONDS = 10.0
UC_MIN_HEIGHT_ABOVE_MEDIAN = 10.0
UC_MIN_PROMINENCE = 10.0
MIN_CONTRACTION_DISTANCE_SECONDS = 60.0
MIN_CONTRACTION_WIDTH_SECONDS = 20.0

# Quality flags summarize cleaned-FHR usability.  ``poor`` takes precedence;
# otherwise any review threshold produces ``review`` and the remainder are
# ``good``. These are engineering QC flags, not clinical labels.
QUALITY_REVIEW_MIN_VALID_FHR_PERCENTAGE = 80.0
QUALITY_POOR_MIN_VALID_FHR_PERCENTAGE = 50.0
QUALITY_REVIEW_MAX_MISSING_GAP_SECONDS = 60.0
QUALITY_POOR_MAX_MISSING_GAP_SECONDS = 5 * 60.0
QUALITY_REVIEW_MAX_ARTIFACT_PERCENTAGE = 2.0
QUALITY_POOR_MAX_ARTIFACT_PERCENTAGE = 10.0

# Additional feature thresholds.  These features are computational summaries,
# not clinical diagnoses or replacements for expert CTG interpretation.
# Bradycardia/tachycardia exposure uses cleaned valid FHR only.  Duration is
# valid sample count divided by sampling frequency; percentages use valid FHR
# time as their denominator so missing data are not treated as normal FHR.
BRADYCARDIA_THRESHOLD_BPM = 110.0
TACHYCARDIA_THRESHOLD_BPM = 160.0

# Above/below-baseline percentages use paired valid FHR/baseline samples.  A
# configurable zero-bpm margin means any strict departure is counted by
# default; increase it to ignore small departures around the baseline.
BASELINE_COMPARISON_MARGIN_BPM = 0.0

# A contraction is associated with the earliest deceleration whose start is
# between the contraction peak and 120 seconds afterward.  Each contraction is
# assessed independently; one deceleration can therefore follow two very close
# contraction peaks if both satisfy this explicit temporal rule.
CONTRACTION_DECELERATION_WINDOW_SECONDS = 120.0

# Baseline crossings use a 1-bpm hysteresis band to reduce counts caused by
# tiny oscillations.  Slopes smaller than the configurable threshold are
# ignored when positive/negative slope summaries are formed.
BASELINE_CROSSING_HYSTERESIS_BPM = 1.0
FHR_SLOPE_MIN_ABSOLUTE_BPM_PER_SECOND = 0.0

# Segment variability uses complete, non-overlapping 5-minute windows.  A
# window must contain at least 50% valid cleaned FHR.  Segment STV is the mean
# absolute adjacent-sample change; segment LTV is the SD of valid 60-second
# epoch means within the 5-minute window.  These remain computational
# approximations rather than clinical gold-standard STV/LTV definitions.
SEGMENT_VARIABILITY_WINDOW_SECONDS = 5 * 60.0
SEGMENT_MIN_VALID_FHR_PERCENTAGE = 50.0

ORIGINAL_FEATURE_COLUMNS = [
    "record_id",
    "recording_duration_minutes",
    "percent_missing_fhr",
    "longest_missing_fhr_gap_seconds",
    "valid_fhr_percentage",
    "possible_fhr_artifact_percentage",
    "feature_quality_flag",
    "mean_fhr",
    "median_fhr",
    "min_fhr",
    "max_fhr",
    "std_fhr",
    "baseline_fhr",
    "short_term_variability",
    "long_term_variability",
    "acceleration_count",
    "mean_acceleration_duration_seconds",
    "max_acceleration_amplitude",
    "deceleration_count",
    "mean_deceleration_duration_seconds",
    "max_deceleration_depth",
    "mean_uc",
    "median_uc",
    "max_uc",
    "std_uc",
    "contraction_count",
    "mean_contraction_peak",
    "mean_contraction_interval_seconds",
    "decelerations_per_contraction",
]

NEW_FEATURE_COLUMNS = [
    "bradycardia_duration_seconds",
    "bradycardia_percentage",
    "tachycardia_duration_seconds",
    "tachycardia_percentage",
    "acceleration_density_per_hour",
    "longest_acceleration_duration_seconds",
    "deceleration_density_per_hour",
    "longest_deceleration_duration_seconds",
    "percentage_time_below_baseline",
    "percentage_time_above_baseline",
    "baseline_drift_std",
    "baseline_drift_range",
    "contraction_interval_std_seconds",
    "contraction_frequency_per_hour",
    "contractions_followed_by_deceleration_percentage",
    "mean_delay_contraction_to_deceleration_seconds",
    "baseline_crossing_count",
    "mean_positive_fhr_slope",
    "mean_negative_fhr_slope",
    "maximum_negative_fhr_slope",
    "segment_stv_mean",
    "segment_stv_std",
    "segment_ltv_mean",
    "segment_ltv_std",
]

FEATURE_COLUMNS = ORIGINAL_FEATURE_COLUMNS + NEW_FEATURE_COLUMNS


def find_header_files(raw_dir):
    """Find .hea files directly in data/raw or one folder below it."""
    return sorted(list(raw_dir.glob("*.hea")) + list(raw_dir.glob("*/*.hea")))


def load_record(header_path):
    """Load one WFDB record from a .hea path."""
    return wfdb.rdrecord(str(header_path.with_suffix("")))


def get_signal_by_name(record, signal_name):
    """Return a signal by channel name instead of assuming a fixed index."""
    if signal_name not in record.sig_name:
        raise ValueError(f"Signal {signal_name} was not found in {record.sig_name}")
    return record.p_signal[:, record.sig_name.index(signal_name)].astype(float)


def preprocess_fhr(fhr):
    """Return raw/cleaned FHR and explicit missing/artifact masks.

    Raw missingness is non-finite FHR or FHR <= 0.  Implausible-range and
    isolated-spike samples are possible artifacts.  All of these become NaN in
    cleaned_fhr, while raw_fhr is never modified.
    """
    raw_fhr = np.asarray(fhr, dtype=float).copy()
    raw_missing_mask = ~np.isfinite(raw_fhr) | (raw_fhr <= 0)
    range_artifact_mask = (
        ~raw_missing_mask
        & ((raw_fhr < FHR_MIN_VALID_BPM) | (raw_fhr > FHR_MAX_VALID_BPM))
    )

    cleaned_fhr = raw_fhr.copy()
    cleaned_fhr[raw_missing_mask | range_artifact_mask] = np.nan

    isolated_spike_mask = np.zeros(len(cleaned_fhr), dtype=bool)
    if ENABLE_ISOLATED_SPIKE_FILTER and len(cleaned_fhr) >= 3:
        previous_values = cleaned_fhr[:-2]
        current_values = cleaned_fhr[1:-1]
        next_values = cleaned_fhr[2:]
        valid_triplets = (
            np.isfinite(previous_values)
            & np.isfinite(current_values)
            & np.isfinite(next_values)
        )
        isolated_spike_mask[1:-1] = (
            valid_triplets
            & (np.abs(current_values - previous_values) >= FHR_ISOLATED_SPIKE_MIN_JUMP_BPM)
            & (np.abs(current_values - next_values) >= FHR_ISOLATED_SPIKE_MIN_JUMP_BPM)
            & (
                np.abs(previous_values - next_values)
                <= FHR_ISOLATED_SPIKE_MAX_NEIGHBOR_DIFFERENCE_BPM
            )
        )
        cleaned_fhr[isolated_spike_mask] = np.nan

    return {
        "raw_fhr": raw_fhr,
        "cleaned_fhr": cleaned_fhr,
        "raw_missing_mask": raw_missing_mask,
        "artifact_mask": range_artifact_mask | isolated_spike_mask,
    }


def clean_fhr(fhr):
    """Compatibility helper returning only the artifact-filtered FHR."""
    return preprocess_fhr(fhr)["cleaned_fhr"]


def safe_nan_stat(values, stat_function):
    """Calculate a finite-value statistic, returning NaN when none exist."""
    valid_values = np.asarray(values, dtype=float)
    valid_values = valid_values[np.isfinite(valid_values)]
    return np.nan if len(valid_values) == 0 else stat_function(valid_values)


def make_odd_window_size(window_samples):
    """Return a positive odd integer window size."""
    window_samples = max(1, round(window_samples))
    return window_samples if window_samples % 2 else window_samples + 1


def centered_rolling_median(values, window_samples, min_valid_samples=None):
    """Centered NaN-aware rolling median without interpolating missing data."""
    values = np.asarray(values, dtype=float)
    window_samples = make_odd_window_size(window_samples)
    if min_valid_samples is None:
        min_valid_samples = 1
    return (
        pd.Series(values)
        .rolling(window_samples, center=True, min_periods=max(1, int(min_valid_samples)))
        .median()
        .to_numpy()
    )


def rolling_median_approximation(values, window_samples, min_valid_percentage=0.0):
    """Return a centered median with an optional full-window validity rule."""
    window_samples = make_odd_window_size(window_samples)
    min_valid_samples = max(
        1, int(np.ceil(window_samples * float(min_valid_percentage) / 100.0))
    )
    return centered_rolling_median(values, window_samples, min_valid_samples)


def estimate_baseline_series(fhr, sampling_frequency):
    """Estimate a pointwise moving baseline, leaving data-poor windows NaN."""
    window_samples = BASELINE_WINDOW_SECONDS * sampling_frequency
    baseline = rolling_median_approximation(
        fhr, window_samples, BASELINE_MIN_VALID_PERCENTAGE
    )
    baseline[~np.isfinite(fhr)] = np.nan
    return baseline


def estimate_baseline(fhr, sampling_frequency):
    """Return the median valid moving baseline for CSV compatibility."""
    return safe_nan_stat(estimate_baseline_series(fhr, sampling_frequency), np.median)


def calculate_short_term_variability(fhr):
    """Mean absolute adjacent-sample change (computational STV approximation)."""
    fhr = np.asarray(fhr, dtype=float)
    if len(fhr) < 2:
        return np.nan
    valid_pairs = np.isfinite(fhr[:-1]) & np.isfinite(fhr[1:])
    if not np.any(valid_pairs):
        return np.nan
    return float(np.mean(np.abs(np.diff(fhr)[valid_pairs])))


def calculate_long_term_variability(fhr, sampling_frequency):
    """SD of valid 60-second epoch means (computational LTV approximation)."""
    fhr = np.asarray(fhr, dtype=float)
    epoch_samples = max(1, round(LTV_EPOCH_SECONDS * sampling_frequency))
    epoch_means = []
    for start in range(0, len(fhr), epoch_samples):
        epoch = fhr[start : start + epoch_samples]
        required = int(np.ceil(len(epoch) * LTV_EPOCH_MIN_VALID_PERCENTAGE / 100.0))
        valid = epoch[np.isfinite(epoch)]
        if len(valid) >= max(1, required):
            epoch_means.append(float(np.mean(valid)))
    if len(epoch_means) < LTV_MIN_VALID_EPOCHS:
        return np.nan
    return float(np.std(epoch_means))


def _empty_event_features():
    return {
        "count": 0,
        "mean_duration_seconds": 0.0,
        "max_amplitude_or_depth": 0.0,
        "events": [],
    }


def detect_threshold_events(signal, baseline, sampling_frequency, mode):
    """Detect continuous threshold events and retain plotting metadata."""
    signal = np.asarray(signal, dtype=float)
    if np.isscalar(baseline):
        baseline = np.full(len(signal), float(baseline), dtype=float)
    else:
        baseline = np.asarray(baseline, dtype=float)
    if len(signal) != len(baseline):
        raise ValueError("signal and baseline must have equal length")

    valid = np.isfinite(signal) & np.isfinite(baseline)
    if mode == "acceleration":
        departure = signal - baseline
        threshold_mask = valid & (departure >= ACCELERATION_THRESHOLD_BPM)
    elif mode == "deceleration":
        departure = baseline - signal
        threshold_mask = valid & (departure >= DECELERATION_THRESHOLD_BPM)
    else:
        raise ValueError(f"Unknown event mode: {mode}")

    min_samples = max(1, int(np.ceil(MIN_EVENT_DURATION_SECONDS * sampling_frequency)))
    padded = np.r_[False, threshold_mask, False]
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    events = []

    for start, end in zip(starts, ends):
        if end - start < min_samples:
            continue
        amplitude = float(np.max(departure[start:end]))
        events.append(
            {
                "start_index": int(start),
                "end_index": int(end),  # exclusive
                "start_seconds": float(start / sampling_frequency),
                "end_seconds": float(end / sampling_frequency),
                "duration_seconds": float((end - start) / sampling_frequency),
                "amplitude_or_depth": amplitude,
            }
        )

    if not events:
        return _empty_event_features()
    return {
        "count": len(events),
        "mean_duration_seconds": float(np.mean([event["duration_seconds"] for event in events])),
        "max_amplitude_or_depth": float(max(event["amplitude_or_depth"] for event in events)),
        "events": events,
    }


def detect_accelerations(fhr, baseline, sampling_frequency):
    """Detect >=15 bpm, >=15 s continuous accelerations by default."""
    return detect_threshold_events(fhr, baseline, sampling_frequency, "acceleration")


def detect_decelerations(fhr, baseline, sampling_frequency):
    """Detect >=15 bpm, >=15 s continuous decelerations by default."""
    return detect_threshold_events(fhr, baseline, sampling_frequency, "deceleration")


def smooth_uc(uc, sampling_frequency):
    """Median-smooth UC without filling missing samples."""
    window_samples = UC_SMOOTHING_SECONDS * sampling_frequency
    smoothed = rolling_median_approximation(uc, window_samples)
    smoothed[~np.isfinite(np.asarray(uc, dtype=float))] = np.nan
    return smoothed


def detect_contractions(uc, sampling_frequency):
    """Detect UC peaks satisfying height, prominence, distance, and width."""
    uc = np.asarray(uc, dtype=float)
    smoothed_uc = smooth_uc(uc, sampling_frequency)
    median_smoothed_uc = safe_nan_stat(smoothed_uc, np.median)
    if np.isnan(median_smoothed_uc):
        return {"smoothed_uc": smoothed_uc, "height_threshold": np.nan, "events": []}

    height_threshold = float(median_smoothed_uc + UC_MIN_HEIGHT_ABOVE_MEDIAN)
    # scipy.find_peaks cannot reason across NaNs.  -inf preserves gaps and
    # prevents a missing sample itself from becoming a candidate peak.
    peak_input = np.where(np.isfinite(smoothed_uc), smoothed_uc, -np.inf)
    peaks, properties = find_peaks(
        peak_input,
        height=height_threshold,
        prominence=UC_MIN_PROMINENCE,
        distance=max(1, int(np.ceil(MIN_CONTRACTION_DISTANCE_SECONDS * sampling_frequency))),
        width=max(1, int(np.ceil(MIN_CONTRACTION_WIDTH_SECONDS * sampling_frequency))),
    )

    events = []
    for position, peak in enumerate(peaks):
        left = max(0, int(np.floor(properties["left_ips"][position])))
        right = min(len(uc), int(np.ceil(properties["right_ips"][position])))
        events.append(
            {
                "peak_index": int(peak),
                "start_index": left,
                "end_index": right,
                "duration_seconds": float(properties["widths"][position] / sampling_frequency),
                "peak_height": float(smoothed_uc[peak]),
                "prominence": float(properties["prominences"][position]),
            }
        )
    return {
        "smoothed_uc": smoothed_uc,
        "height_threshold": height_threshold,
        "events": events,
    }


def extract_contraction_features(uc, sampling_frequency):
    """Extract raw UC summaries and filtered contraction peak features."""
    uc = np.asarray(uc, dtype=float)
    detection = detect_contractions(uc, sampling_frequency)
    events = detection["events"]
    peaks = np.array([event["peak_index"] for event in events], dtype=int)
    intervals = np.diff(peaks) / sampling_frequency if len(peaks) > 1 else []
    return {
        "mean_uc": safe_nan_stat(uc, np.mean),
        "median_uc": safe_nan_stat(uc, np.median),
        "max_uc": safe_nan_stat(uc, np.max),
        "std_uc": safe_nan_stat(uc, np.std),
        "contraction_count": len(events),
        "mean_contraction_peak": (
            float(np.mean([event["peak_height"] for event in events])) if events else 0.0
        ),
        "mean_contraction_interval_seconds": (
            float(np.mean(intervals)) if len(intervals) else 0.0
        ),
        "events": events,
        "smoothed_uc": detection["smoothed_uc"],
        "height_threshold": detection["height_threshold"],
    }


def longest_true_run(mask):
    """Return the length of the longest contiguous True run."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False]
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def determine_feature_quality_flag(
    valid_percentage, longest_missing_gap_seconds, artifact_percentage
):
    """Map signal-quality thresholds to good, review, or poor."""
    if (
        valid_percentage < QUALITY_POOR_MIN_VALID_FHR_PERCENTAGE
        or longest_missing_gap_seconds > QUALITY_POOR_MAX_MISSING_GAP_SECONDS
        or artifact_percentage > QUALITY_POOR_MAX_ARTIFACT_PERCENTAGE
    ):
        return "poor"
    if (
        valid_percentage < QUALITY_REVIEW_MIN_VALID_FHR_PERCENTAGE
        or longest_missing_gap_seconds > QUALITY_REVIEW_MAX_MISSING_GAP_SECONDS
        or artifact_percentage > QUALITY_REVIEW_MAX_ARTIFACT_PERCENTAGE
    ):
        return "review"
    return "good"


def analyze_signals(raw_fhr, uc, sampling_frequency):
    """Run the shared preprocessing/detection path used by CSV and plots."""
    fhr_processing = preprocess_fhr(raw_fhr)
    cleaned_fhr = fhr_processing["cleaned_fhr"]
    baseline_series = estimate_baseline_series(cleaned_fhr, sampling_frequency)
    acceleration_features = detect_accelerations(
        cleaned_fhr, baseline_series, sampling_frequency
    )
    deceleration_features = detect_decelerations(
        cleaned_fhr, baseline_series, sampling_frequency
    )
    contraction_features = extract_contraction_features(uc, sampling_frequency)
    return {
        **fhr_processing,
        "baseline_series": baseline_series,
        "acceleration_features": acceleration_features,
        "deceleration_features": deceleration_features,
        "contraction_features": contraction_features,
    }


def percentage(numerator, denominator):
    """Return a percentage, using 0 when the denominator is zero."""
    return 100.0 * numerator / denominator if denominator else 0.0


def calculate_fhr_threshold_exposure(fhr, sampling_frequency):
    """Measure valid time below 110 bpm and above 160 bpm by default.

    These simple threshold exposures are computational summaries.  They do not
    enforce a minimum episode duration and therefore are not diagnoses of
    clinical bradycardia or tachycardia.
    """
    fhr = np.asarray(fhr, dtype=float)
    valid = np.isfinite(fhr)
    valid_count = int(np.sum(valid))
    bradycardia_count = int(np.sum(valid & (fhr < BRADYCARDIA_THRESHOLD_BPM)))
    tachycardia_count = int(np.sum(valid & (fhr > TACHYCARDIA_THRESHOLD_BPM)))
    return {
        "bradycardia_duration_seconds": bradycardia_count / sampling_frequency,
        "bradycardia_percentage": percentage(bradycardia_count, valid_count),
        "tachycardia_duration_seconds": tachycardia_count / sampling_frequency,
        "tachycardia_percentage": percentage(tachycardia_count, valid_count),
    }


def calculate_event_extension_features(
    accelerations, decelerations, recording_duration_seconds
):
    """Calculate event densities and longest continuous event durations.

    Density is event count per total recording hour.  The duration values reuse
    the continuous, valid-sample event metadata from the existing detector.
    """
    recording_hours = recording_duration_seconds / 3600.0
    acceleration_durations = [
        event["duration_seconds"] for event in accelerations["events"]
    ]
    deceleration_durations = [
        event["duration_seconds"] for event in decelerations["events"]
    ]
    return {
        "acceleration_density_per_hour": (
            accelerations["count"] / recording_hours if recording_hours else 0.0
        ),
        "longest_acceleration_duration_seconds": (
            float(max(acceleration_durations)) if acceleration_durations else 0.0
        ),
        "deceleration_density_per_hour": (
            decelerations["count"] / recording_hours if recording_hours else 0.0
        ),
        "longest_deceleration_duration_seconds": (
            float(max(deceleration_durations)) if deceleration_durations else 0.0
        ),
    }


def calculate_baseline_extension_features(fhr, baseline):
    """Summarize time around and slow drift of the moving FHR baseline.

    Above/below percentages consider only samples where cleaned FHR and the
    moving baseline are both valid.  Baseline drift is the standard deviation
    and range of all valid pointwise baseline estimates.
    """
    fhr = np.asarray(fhr, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    paired_valid = np.isfinite(fhr) & np.isfinite(baseline)
    paired_count = int(np.sum(paired_valid))
    below_count = int(
        np.sum(
            paired_valid
            & (fhr < baseline - BASELINE_COMPARISON_MARGIN_BPM)
        )
    )
    above_count = int(
        np.sum(
            paired_valid
            & (fhr > baseline + BASELINE_COMPARISON_MARGIN_BPM)
        )
    )
    valid_baseline = baseline[np.isfinite(baseline)]
    baseline_range = (
        float(np.max(valid_baseline) - np.min(valid_baseline))
        if len(valid_baseline)
        else np.nan
    )
    return {
        "percentage_time_below_baseline": percentage(below_count, paired_count),
        "percentage_time_above_baseline": percentage(above_count, paired_count),
        "baseline_drift_std": (
            float(np.std(valid_baseline)) if len(valid_baseline) else np.nan
        ),
        "baseline_drift_range": baseline_range,
    }


def calculate_contraction_extension_features(
    contractions, recording_duration_seconds, sampling_frequency
):
    """Calculate contraction interval dispersion and hourly frequency."""
    peak_indices = np.asarray(
        [event["peak_index"] for event in contractions["events"]], dtype=int
    )
    intervals = (
        np.diff(peak_indices) / sampling_frequency if len(peak_indices) > 1 else []
    )
    recording_hours = recording_duration_seconds / 3600.0
    return {
        "contraction_interval_std_seconds": (
            float(np.std(intervals)) if len(intervals) else 0.0
        ),
        "contraction_frequency_per_hour": (
            contractions["contraction_count"] / recording_hours
            if recording_hours
            else 0.0
        ),
    }


def calculate_contraction_deceleration_interactions(
    contraction_events, deceleration_events, sampling_frequency
):
    """Associate contractions with decelerations starting shortly afterward.

    For each contraction peak, the earliest deceleration start at or after the
    peak and no later than CONTRACTION_DECELERATION_WINDOW_SECONDS is matched.
    The mean delay is calculated over matched contractions only.
    """
    if not contraction_events:
        return {
            "contractions_followed_by_deceleration_percentage": 0.0,
            "mean_delay_contraction_to_deceleration_seconds": 0.0,
        }

    deceleration_starts = np.asarray(
        sorted(event["start_index"] for event in deceleration_events), dtype=int
    )
    maximum_delay_samples = int(
        np.floor(CONTRACTION_DECELERATION_WINDOW_SECONDS * sampling_frequency)
    )
    delays = []
    for contraction in contraction_events:
        peak = contraction["peak_index"]
        candidate_position = np.searchsorted(deceleration_starts, peak)
        if candidate_position >= len(deceleration_starts):
            continue
        delay_samples = int(deceleration_starts[candidate_position] - peak)
        if delay_samples <= maximum_delay_samples:
            delays.append(delay_samples / sampling_frequency)

    return {
        "contractions_followed_by_deceleration_percentage": percentage(
            len(delays), len(contraction_events)
        ),
        "mean_delay_contraction_to_deceleration_seconds": (
            float(np.mean(delays)) if delays else 0.0
        ),
    }


def calculate_baseline_crossing_count(fhr, baseline):
    """Count continuous transitions across a configurable baseline band.

    Missing FHR/baseline resets the crossing state, so a gap can never create a
    crossing.  Samples inside the hysteresis band preserve the last side until
    the signal clearly reaches the opposite side.
    """
    fhr = np.asarray(fhr, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    difference = fhr - baseline
    valid = np.isfinite(difference)
    crossing_count = 0
    previous_side = 0
    for is_valid, departure in zip(valid, difference):
        if not is_valid:
            previous_side = 0
            continue
        if departure > BASELINE_CROSSING_HYSTERESIS_BPM:
            current_side = 1
        elif departure < -BASELINE_CROSSING_HYSTERESIS_BPM:
            current_side = -1
        else:
            continue
        if previous_side and current_side != previous_side:
            crossing_count += 1
        previous_side = current_side
    return crossing_count


def calculate_fhr_slope_features(fhr, sampling_frequency):
    """Summarize adjacent valid-sample slopes in bpm per second.

    These derivatives are signal-dynamics approximations and are sensitive to
    sampling rate and residual noise; they are not clinical gold standards.
    """
    fhr = np.asarray(fhr, dtype=float)
    if len(fhr) < 2:
        slopes = np.array([], dtype=float)
    else:
        valid_pairs = np.isfinite(fhr[:-1]) & np.isfinite(fhr[1:])
        slopes = np.diff(fhr)[valid_pairs] * sampling_frequency

    positive_slopes = slopes[
        slopes > FHR_SLOPE_MIN_ABSOLUTE_BPM_PER_SECOND
    ]
    negative_slopes = slopes[
        slopes < -FHR_SLOPE_MIN_ABSOLUTE_BPM_PER_SECOND
    ]
    return {
        "mean_positive_fhr_slope": (
            float(np.mean(positive_slopes)) if len(positive_slopes) else 0.0
        ),
        "mean_negative_fhr_slope": (
            float(np.mean(negative_slopes)) if len(negative_slopes) else 0.0
        ),
        "maximum_negative_fhr_slope": (
            float(np.min(negative_slopes)) if len(negative_slopes) else 0.0
        ),
    }


def calculate_segment_variability_features(fhr, sampling_frequency):
    """Aggregate STV/LTV approximations across complete 5-minute windows."""
    fhr = np.asarray(fhr, dtype=float)
    segment_samples = max(
        1, round(SEGMENT_VARIABILITY_WINDOW_SECONDS * sampling_frequency)
    )
    segment_stv_values = []
    segment_ltv_values = []
    for start in range(0, len(fhr) - segment_samples + 1, segment_samples):
        segment = fhr[start : start + segment_samples]
        valid_percentage = 100.0 * np.mean(np.isfinite(segment))
        if valid_percentage < SEGMENT_MIN_VALID_FHR_PERCENTAGE:
            continue
        segment_stv = calculate_short_term_variability(segment)
        segment_ltv = calculate_long_term_variability(segment, sampling_frequency)
        if np.isfinite(segment_stv):
            segment_stv_values.append(segment_stv)
        if np.isfinite(segment_ltv):
            segment_ltv_values.append(segment_ltv)

    def summarize(values):
        if not values:
            return np.nan, np.nan
        return float(np.mean(values)), float(np.std(values))

    segment_stv_mean, segment_stv_std = summarize(segment_stv_values)
    segment_ltv_mean, segment_ltv_std = summarize(segment_ltv_values)
    return {
        "segment_stv_mean": segment_stv_mean,
        "segment_stv_std": segment_stv_std,
        "segment_ltv_mean": segment_ltv_mean,
        "segment_ltv_std": segment_ltv_std,
    }


def calculate_additional_features(analysis, sampling_frequency):
    """Calculate all additive Phase 3 clinical-engineering features."""
    fhr = analysis["cleaned_fhr"]
    baseline = analysis["baseline_series"]
    accelerations = analysis["acceleration_features"]
    decelerations = analysis["deceleration_features"]
    contractions = analysis["contraction_features"]
    recording_duration_seconds = len(fhr) / sampling_frequency

    return {
        **calculate_fhr_threshold_exposure(fhr, sampling_frequency),
        **calculate_event_extension_features(
            accelerations, decelerations, recording_duration_seconds
        ),
        **calculate_baseline_extension_features(fhr, baseline),
        **calculate_contraction_extension_features(
            contractions, recording_duration_seconds, sampling_frequency
        ),
        **calculate_contraction_deceleration_interactions(
            contractions["events"], decelerations["events"], sampling_frequency
        ),
        "baseline_crossing_count": calculate_baseline_crossing_count(fhr, baseline),
        **calculate_fhr_slope_features(fhr, sampling_frequency),
        **calculate_segment_variability_features(fhr, sampling_frequency),
    }


def extract_features_for_record(header_path, return_analysis=False):
    """Extract all clinical features for one WFDB record."""
    record = load_record(header_path)
    sampling_frequency = float(record.fs)
    raw_fhr = get_signal_by_name(record, "FHR")
    uc = get_signal_by_name(record, "UC")
    analysis = analyze_signals(raw_fhr, uc, sampling_frequency)
    fhr = analysis["cleaned_fhr"]
    accelerations = analysis["acceleration_features"]
    decelerations = analysis["deceleration_features"]
    contractions = analysis["contraction_features"]

    sample_count = len(fhr)
    raw_missing_percentage = 100.0 * np.mean(analysis["raw_missing_mask"])
    valid_percentage = 100.0 * np.mean(np.isfinite(fhr))
    artifact_percentage = 100.0 * np.mean(analysis["artifact_mask"])
    # Missing-gap QC describes source missingness only.  Artifact removals are
    # reported separately and still reduce valid_fhr_percentage.
    longest_gap_seconds = (
        longest_true_run(analysis["raw_missing_mask"]) / sampling_frequency
    )
    contraction_count = contractions["contraction_count"]
    deceleration_count = decelerations["count"]

    features = {
        "record_id": record.record_name,
        "recording_duration_minutes": sample_count / sampling_frequency / 60.0,
        "percent_missing_fhr": raw_missing_percentage,
        "longest_missing_fhr_gap_seconds": longest_gap_seconds,
        "valid_fhr_percentage": valid_percentage,
        "possible_fhr_artifact_percentage": artifact_percentage,
        "feature_quality_flag": determine_feature_quality_flag(
            valid_percentage, longest_gap_seconds, artifact_percentage
        ),
        "mean_fhr": safe_nan_stat(fhr, np.mean),
        "median_fhr": safe_nan_stat(fhr, np.median),
        "min_fhr": safe_nan_stat(fhr, np.min),
        "max_fhr": safe_nan_stat(fhr, np.max),
        "std_fhr": safe_nan_stat(fhr, np.std),
        "baseline_fhr": safe_nan_stat(analysis["baseline_series"], np.median),
        "short_term_variability": calculate_short_term_variability(fhr),
        "long_term_variability": calculate_long_term_variability(fhr, sampling_frequency),
        "acceleration_count": accelerations["count"],
        "mean_acceleration_duration_seconds": accelerations["mean_duration_seconds"],
        "max_acceleration_amplitude": accelerations["max_amplitude_or_depth"],
        "deceleration_count": decelerations["count"],
        "mean_deceleration_duration_seconds": decelerations["mean_duration_seconds"],
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
    # Add the extension features after calculating the original feature block.
    features.update(calculate_additional_features(analysis, sampling_frequency))
    return (features, analysis) if return_analysis else features


def print_feature_summary(
    feature_rows,
    columns=ORIGINAL_FEATURE_COLUMNS,
    title="Original feature summary statistics",
):
    """Print compact summary statistics for a selected feature group."""
    numeric_columns = [
        column
        for column in columns
        if column not in {"record_id", "feature_quality_flag"}
    ]
    print(f"\n{title}")
    print("-" * len(title))
    print(
        f"{'feature':35} {'count':>6} {'mean':>10} {'std':>10} "
        f"{'min':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'max':>10}"
    )
    for column in numeric_columns:
        values = np.asarray([row[column] for row in feature_rows], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            print(f"{column:35} {0:6d} {'NaN':>10}")
            continue
        percentiles = np.percentile(values, [25, 50, 75])
        print(
            f"{column:35} {len(values):6d} {np.mean(values):10.2f} "
            f"{np.std(values):10.2f} {np.min(values):10.2f} "
            f"{percentiles[0]:10.2f} {percentiles[1]:10.2f} "
            f"{percentiles[2]:10.2f} {np.max(values):10.2f}"
        )


def save_features_csv(feature_rows):
    """Save extracted features to data/processed/clinical_features.csv."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(feature_rows)


def main():
    """Extract every record and save the feature CSV; never train a model."""
    header_files = find_header_files(RAW_DIR)
    if not header_files:
        raise FileNotFoundError(f"No .hea files were found under {RAW_DIR}")
    feature_rows = []
    failed_records = []
    print(f"Found {len(header_files)} records to process.", flush=True)
    for record_number, header_path in enumerate(header_files, start=1):
        try:
            feature_rows.append(extract_features_for_record(header_path))
        # Continue processing when an individual WFDB record is unreadable.
        except Exception as error:  # noqa: BLE001
            failed_records.append((header_path.stem, str(error)))
        if record_number == 1 or record_number % 25 == 0 or record_number == len(header_files):
            print(f"Processed {record_number}/{len(header_files)} records...", flush=True)
    save_features_csv(feature_rows)
    print(f"Saved clinical features to: {OUTPUT_PATH}")
    print(f"Total records processed: {len(feature_rows)}")
    if failed_records:
        print("\nFailed records")
        print("--------------")
        for record_id, error_message in failed_records:
            print(f"{record_id}: {error_message}")
    else:
        print("Failed records: none")
    print_feature_summary(feature_rows)
    print_feature_summary(
        feature_rows,
        columns=NEW_FEATURE_COLUMNS,
        title="New feature summary statistics",
    )


if __name__ == "__main__":
    main()
