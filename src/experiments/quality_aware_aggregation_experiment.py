"""Compare whole-record and quality-aware patient-level CTG representations.

The only experimental change is the patient-level feature representation.
Both datasets use the same patient labels, leakage exclusions, model family,
model parameters, balancing method, fixed split, repeated split seeds, and
repeated cross-validation folds.  No window receives a target label and no
window-level classifier is trained.

Run from the project root::

    python src/experiments/quality_aware_aggregation_experiment.py

The script refuses to overwrite an existing experiment directory or
quality-aware dataset.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import audit_signal_quality_effects as quality_audit
from src.analysis import diagnose_pathological_cases as prior_diagnostic
from src.analysis import visualize_pathological_recordings as visual
from src.features import extract_clinical_features as extractor
from src.models import train_models
from src.utils.record_ids import normalize_record_id

# ---------------------------------------------------------------------------
# Inputs, outputs, and fixed experiment design
# ---------------------------------------------------------------------------
BASELINE_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.csv"
CLINICAL_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"
PATHOLOGICAL_WINDOWS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "pathological_visual_review"
    / "pathological_window_diagnostics.csv"
)
QUALITY_AUDIT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "signal_quality_audit"
    / "signal_quality_audit_report.md"
)
PATHOLOGICAL_QUALITY_EFFECTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "signal_quality_audit"
    / "pathological_quality_effects.csv"
)
SAVED_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

QUALITY_AWARE_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ml_dataset_quality_aware.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "quality_aware_aggregation"
FIGURE_DIR = OUTPUT_ROOT / "figures"
ALL_WINDOWS_PATH = OUTPUT_ROOT / "all_window_features.csv"
FEATURE_CHANGES_PATH = OUTPUT_ROOT / "feature_representation_changes.csv"
FIXED_COMPARISON_PATH = OUTPUT_ROOT / "fixed_split_comparison.csv"
REPEATED_SPLIT_PATH = OUTPUT_ROOT / "repeated_split_comparison.csv"
REPEATED_SPLIT_SUMMARY_PATH = OUTPUT_ROOT / "repeated_split_summary.csv"
REPEATED_CV_PATH = OUTPUT_ROOT / "repeated_cv_comparison.csv"
PATHOLOGICAL_CASE_PATH = OUTPUT_ROOT / "pathological_case_comparison.csv"
UC_DIAGNOSTIC_PATH = OUTPUT_ROOT / "uc_aggregation_diagnostic.csv"
REPORT_PATH = OUTPUT_ROOT / "quality_aware_aggregation_report.md"

# Window and quality-rule settings. A shorter final window is included only
# when it contains at least 10 minutes; no label is assigned to any window.
WINDOW_MINUTES = 20.0
MINIMUM_FINAL_WINDOW_MINUTES = 10.0
MINIMUM_VALID_FHR_PERCENTAGE = 80.0
# The earlier audit counted every residual end slice, including slices shorter
# than 10 minutes. Keep that count available for comparison with this rule.
PRIOR_AUDIT_WINDOW_COUNT = 2_297

# Use the same partitions as the earlier Pathological diagnostic.
SPLIT_SEEDS = list(prior_diagnostic.SPLIT_SEEDS)  # predefined 0..29
CV_SPLITS = prior_diagnostic.CV_SPLITS
CV_REPEATS = prior_diagnostic.CV_REPEATS
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_RANDOM_STATE = 42

# The selected saved classifier is Logistic Regression.  The previous
# 30-split diagnostic evaluated it with median imputation, scaling, and
# training-only SMOTE.  This same pipeline is used for both representations;
# there is no class_weight and no separate tuning.
PIPELINE_BALANCING = "SMOTE on training data only"

# These columns depend only on UC. Every other engineered
# clinical feature is FHR-derived, FHR-dependent, an FHR/UC interaction, or a
# source quality/duration field.
UC_ONLY_FEATURES = [
    "mean_uc",
    "median_uc",
    "max_uc",
    "std_uc",
    "contraction_count",
    "mean_contraction_peak",
    "mean_contraction_interval_seconds",
    "contraction_interval_std_seconds",
    "contraction_frequency_per_hour",
]

SOURCE_QUALITY_FEATURES = [
    "recording_duration_minutes",
    "percent_missing_fhr",
    "longest_missing_fhr_gap_seconds",
    "valid_fhr_percentage",
    "possible_fhr_artifact_percentage",
]

COUNT_FEATURES = {
    "acceleration_count",
    "deceleration_count",
    "baseline_crossing_count",
}
DENSITY_FEATURES = {
    "acceleration_density_per_hour",
    "deceleration_density_per_hour",
}
PERCENTAGE_FEATURES = {
    "bradycardia_percentage",
    "tachycardia_percentage",
    "percentage_time_below_baseline",
    "percentage_time_above_baseline",
}
EXPOSURE_DURATION_FEATURES = {
    "bradycardia_duration_seconds",
    "tachycardia_duration_seconds",
}
EVENT_MEAN_DURATION_WEIGHTS = {
    "mean_acceleration_duration_seconds": "acceleration_count",
    "mean_deceleration_duration_seconds": "deceleration_count",
}
EVENT_MAX_FEATURES = {
    "max_acceleration_amplitude",
    "max_deceleration_depth",
    "longest_acceleration_duration_seconds",
    "longest_deceleration_duration_seconds",
}
VARIABILITY_FEATURES = {
    "short_term_variability",
    "long_term_variability",
    "segment_stv_mean",
    "segment_stv_std",
    "segment_ltv_mean",
    "segment_ltv_std",
}
BASELINE_DRIFT_FEATURES = {"baseline_drift_std", "baseline_drift_range"}
FHR_WEIGHTED_SUMMARIES = {
    "mean_fhr",
    "median_fhr",
    "std_fhr",
}
FHR_MIN_FEATURES = {"min_fhr", "maximum_negative_fhr_slope"}
FHR_MAX_FEATURES = {"max_fhr"}
SLOPE_MEAN_FEATURES = {"mean_positive_fhr_slope", "mean_negative_fhr_slope"}
INTERACTION_FEATURES = {
    "decelerations_per_contraction",
    "contractions_followed_by_deceleration_percentage",
    "mean_delay_contraction_to_deceleration_seconds",
}

IMPORTANT_CHANGE_CONCEPTS = [
    "short_term_variability",
    "long_term_variability",
    "acceleration_count",
    "acceleration_density_per_hour",
    "baseline_crossing_count",
    "percentage_time_above_baseline",
    "percentage_time_below_baseline",
    "bradycardia_percentage",
    "tachycardia_percentage",
    "deceleration_count",
    "max_deceleration_depth",
    "mean_deceleration_duration_seconds",
    "longest_deceleration_duration_seconds",
    "baseline_drift_std",
    "baseline_drift_range",
    "segment_stv_std",
    "segment_ltv_std",
]

FIXED_TEST_PATHOLOGICAL_IDS = ["1002", "1071", "1158", "1418", "1490", "2009"]
FIGURE_DPI = 210


def ensure_new_outputs() -> None:
    """Refuse to overwrite protected datasets, reports, or saved models."""
    if QUALITY_AWARE_DATASET_PATH.exists():
        raise FileExistsError(
            f"{QUALITY_AWARE_DATASET_PATH} already exists; this experiment "
            "will not overwrite it."
        )
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(
            f"{OUTPUT_ROOT} already contains files; move or rename the prior "
            "experiment before rerunning."
        )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    """Load the experiment inputs and selected model artifacts."""
    paths = [
        BASELINE_DATASET_PATH,
        LABELS_PATH,
        CLINICAL_FEATURES_PATH,
        PATHOLOGICAL_WINDOWS_PATH,
        QUALITY_AUDIT_REPORT_PATH,
        PATHOLOGICAL_QUALITY_EFFECTS_PATH,
        SAVED_MODEL_PATH,
        LABEL_ENCODER_PATH,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    baseline = pd.read_csv(BASELINE_DATASET_PATH)
    labels = pd.read_csv(LABELS_PATH)
    clinical = pd.read_csv(CLINICAL_FEATURES_PATH)
    pathological_windows = pd.read_csv(PATHOLOGICAL_WINDOWS_PATH)
    pathological_quality = pd.read_csv(PATHOLOGICAL_QUALITY_EFFECTS_PATH)
    for frame in [
        baseline,
        labels,
        clinical,
        pathological_windows,
        pathological_quality,
    ]:
        frame["record_id"] = normalize_record_id(frame["record_id"])
    with SAVED_MODEL_PATH.open("rb") as handle:
        selected_model = pickle.load(handle)
    with LABEL_ENCODER_PATH.open("rb") as handle:
        encoder = pickle.load(handle)

    expected_ids = set(labels["record_id"])
    for frame, name in [
        (baseline, "ml_dataset"),
        (clinical, "clinical_features"),
    ]:
        if len(frame) != 552 or frame["record_id"].duplicated().any():
            raise ValueError(f"{name} must have 552 unique record rows")
        if set(frame["record_id"]) != expected_ids:
            raise ValueError(f"{name} does not align with labels.csv")
    if not np.array_equal(
        baseline["label"].to_numpy(),
        labels.set_index("record_id").loc[baseline["record_id"], "label"].to_numpy(),
    ):
        raise ValueError("Baseline labels disagree with labels.csv")
    return (
        baseline,
        labels,
        clinical,
        pathological_windows,
        pathological_quality,
        selected_model,
        encoder,
    )


def selected_classifier(model):
    """Clone the terminal estimator from the saved selected pipeline."""
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        return clone(model.named_steps["classifier"])
    return clone(model)


def build_comparison_pipeline(model, seed: int):
    """Build the identical fold-safe pipeline for either representation."""
    classifier = selected_classifier(model)
    if hasattr(classifier, "random_state"):
        classifier.set_params(random_state=seed)
    if hasattr(classifier, "class_weight"):
        classifier.set_params(class_weight=None)
    return ImbalancedPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=seed)),
            ("classifier", classifier),
        ]
    )


def current_feature_groups() -> tuple[list[str], list[str]]:
    """Return the documented FHR-dependent and UC-only columns."""
    current = [
        feature
        for feature in extractor.FEATURE_COLUMNS
        if feature
        not in {
            "record_id",
            "feature_quality_flag",
            *SOURCE_QUALITY_FEATURES,
        }
    ]
    missing_uc = sorted(set(UC_ONLY_FEATURES) - set(current))
    if missing_uc:
        raise ValueError(f"Configured UC-only features are absent: {missing_uc}")
    fhr_dependent = [feature for feature in current if feature not in UC_ONLY_FEATURES]
    if set(fhr_dependent) & set(UC_ONLY_FEATURES):
        raise AssertionError("Feature groups overlap")
    return fhr_dependent, list(UC_ONLY_FEATURES)


def diagnostic_window_bounds(sample_count: int, sampling_frequency: float):
    """Return windows, excluding a final slice shorter than 10 minutes."""
    window_samples = max(1, round(WINDOW_MINUTES * 60 * sampling_frequency))
    minimum_final_samples = round(
        MINIMUM_FINAL_WINDOW_MINUTES * 60 * sampling_frequency
    )
    bounds = []
    for start in range(0, sample_count, window_samples):
        end = min(start + window_samples, sample_count)
        is_source_final = end == sample_count
        if is_source_final and end - start < minimum_final_samples:
            continue
        bounds.append((start, end, is_source_final))
    return bounds


def generate_all_window_features(
    record_ids: list[str], waveform_directory: Path
) -> pd.DataFrame:
    """Recalculate the feature set on diagnostic raw-waveform slices."""
    rows = []
    for record_number, record_id in enumerate(record_ids, start=1):
        record = extractor.load_record(waveform_directory / f"{record_id}.hea")
        fs = float(record.fs)
        raw_fhr = extractor.get_signal_by_name(record, "FHR")
        uc = extractor.get_signal_by_name(record, "UC")
        bounds = diagnostic_window_bounds(len(raw_fhr), fs)
        for window_number, (start, end, is_final) in enumerate(bounds, start=1):
            features, analysis = quality_audit.extract_current_window_features(
                raw_fhr[start:end],
                uc[start:end],
                fs,
            )
            duration_minutes = (end - start) / fs / 60.0
            accepted = (
                features["valid_fhr_percentage"]
                >= MINIMUM_VALID_FHR_PERCENTAGE
            )
            rows.append(
                {
                    "record_id": record_id,
                    "window_number": window_number,
                    "start_minute": start / fs / 60.0,
                    "end_minute": end / fs / 60.0,
                    "window_duration_minutes": duration_minutes,
                    "is_final_window": is_final,
                    "accepted_by_quality_rule": bool(accepted),
                    "missing_fhr_percentage": analysis[
                        "missing_fhr_percentage"
                    ],
                    "artifact_percentage": analysis["artifact_percentage"],
                    "longest_missing_gap_seconds": analysis[
                        "longest_missing_gap_seconds"
                    ],
                    "longest_continuous_valid_segment_seconds": analysis[
                        "longest_continuous_valid_segment_seconds"
                    ],
                    "number_of_valid_segments": analysis[
                        "number_of_valid_segments"
                    ],
                    "valid_fhr_sample_count": round(
                        analysis["total_valid_duration_seconds"] * fs
                    ),
                    "total_window_sample_count": int(end - start),
                    "sampling_frequency": fs,
                    **features,
                }
            )
        if record_number % 100 == 0 or record_number == len(record_ids):
            print(f"  Window generation: {record_number}/{len(record_ids)} records")
    windows = pd.DataFrame(rows)
    # Window rows have no patient labels because they are diagnostic slices,
    # not training examples.
    return windows


def finite_weighted_mean(values, weights) -> float:
    """Return a finite weighted mean without converting missing values to zero."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def add_stat(output: dict, key: str, values: pd.Series, statistic: str) -> str:
    """Calculate one named aggregation and return the generated column."""
    column = f"{key}__qa_{statistic}"
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        output[column] = np.nan
    elif statistic == "mean":
        output[column] = float(finite.mean())
    elif statistic == "median":
        output[column] = float(finite.median())
    elif statistic == "min":
        output[column] = float(finite.min())
    elif statistic == "max":
        output[column] = float(finite.max())
    elif statistic == "std":
        output[column] = float(finite.std(ddof=0))
    elif statistic == "sum":
        output[column] = float(finite.sum())
    else:
        raise ValueError(f"Unknown statistic {statistic}")
    return column


def add_weighted(
    output: dict,
    key: str,
    values: pd.Series,
    weights: pd.Series,
    suffix: str = "weighted_mean",
) -> str:
    """Add a weighted-mean aggregation."""
    column = f"{key}__qa_{suffix}"
    output[column] = finite_weighted_mean(values, weights)
    return column


def aggregate_fhr_features(
    accepted: pd.DataFrame,
    fhr_features: list[str],
) -> tuple[dict, dict[str, str]]:
    """Aggregate accepted-window FHR-dependent features by documented meaning."""
    output: dict[str, float] = {}
    primary: dict[str, str] = {}
    valid_weights = accepted.get("valid_fhr_sample_count", pd.Series(dtype=float))
    duration_weights = accepted.get(
        "window_duration_minutes", pd.Series(dtype=float)
    )

    for feature in fhr_features:
        values = (
            accepted[feature]
            if feature in accepted
            else pd.Series(dtype=float)
        )
        if feature in COUNT_FEATURES:
            primary[feature] = add_stat(output, feature, values, "sum")
            add_stat(output, feature, values, "mean")
            add_stat(output, feature, values, "max")
        elif feature in DENSITY_FEATURES:
            primary[feature] = add_weighted(
                output, feature, values, duration_weights, "duration_weighted_mean"
            )
            add_stat(output, feature, values, "max")
        elif feature in PERCENTAGE_FEATURES:
            primary[feature] = add_weighted(
                output, feature, values, valid_weights, "valid_weighted_mean"
            )
            add_stat(output, feature, values, "max")
        elif feature in EXPOSURE_DURATION_FEATURES:
            primary[feature] = add_stat(output, feature, values, "sum")
            add_stat(output, feature, values, "max")
        elif feature == "baseline_fhr":
            primary[feature] = add_weighted(
                output, feature, values, valid_weights, "valid_weighted_mean"
            )
            minimum = add_stat(output, feature, values, "min")
            maximum = add_stat(output, feature, values, "max")
            output[f"{feature}__qa_range"] = (
                output[maximum] - output[minimum]
                if np.isfinite(output[maximum]) and np.isfinite(output[minimum])
                else np.nan
            )
        elif feature in EVENT_MEAN_DURATION_WEIGHTS:
            weight_feature = EVENT_MEAN_DURATION_WEIGHTS[feature]
            weights = (
                accepted[weight_feature]
                if weight_feature in accepted
                else pd.Series(dtype=float)
            )
            primary[feature] = add_weighted(
                output, feature, values, weights, "event_weighted_mean"
            )
            add_stat(output, feature, values, "max")
        elif feature in EVENT_MAX_FEATURES:
            primary[feature] = add_stat(output, feature, values, "max")
            add_stat(output, feature, values, "mean")
            add_stat(output, feature, values, "median")
        elif feature in VARIABILITY_FEATURES:
            primary[feature] = add_stat(output, feature, values, "mean")
            add_stat(output, feature, values, "min")
            add_stat(output, feature, values, "std")
        elif feature in BASELINE_DRIFT_FEATURES:
            statistic = "max" if feature.endswith("range") else "mean"
            primary[feature] = add_stat(output, feature, values, statistic)
            if statistic != "max":
                add_stat(output, feature, values, "max")
        elif feature in FHR_WEIGHTED_SUMMARIES:
            primary[feature] = add_weighted(
                output, feature, values, valid_weights, "valid_weighted_mean"
            )
            add_stat(output, feature, values, "min")
            add_stat(output, feature, values, "max")
        elif feature in FHR_MIN_FEATURES:
            primary[feature] = add_stat(output, feature, values, "min")
            add_stat(output, feature, values, "mean")
        elif feature in FHR_MAX_FEATURES:
            primary[feature] = add_stat(output, feature, values, "max")
            add_stat(output, feature, values, "mean")
        elif feature in SLOPE_MEAN_FEATURES:
            primary[feature] = add_weighted(
                output, feature, values, valid_weights, "valid_weighted_mean"
            )
            add_stat(
                output,
                feature,
                values,
                "min" if "negative" in feature else "max",
            )
        elif feature == "decelerations_per_contraction":
            decelerations = (
                accepted["deceleration_count"].sum()
                if "deceleration_count" in accepted
                else 0
            )
            contractions = (
                accepted["contraction_count"].sum()
                if "contraction_count" in accepted
                else 0
            )
            column = f"{feature}__qa_ratio_of_sums"
            output[column] = (
                float(decelerations / contractions) if contractions else 0.0
            ) if len(accepted) else np.nan
            primary[feature] = column
            add_stat(output, feature, values, "max")
        elif feature in {
            "contractions_followed_by_deceleration_percentage",
            "mean_delay_contraction_to_deceleration_seconds",
        }:
            weights = (
                accepted["contraction_count"]
                if "contraction_count" in accepted
                else pd.Series(dtype=float)
            )
            primary[feature] = add_weighted(
                output, feature, values, weights, "contraction_weighted_mean"
            )
            add_stat(output, feature, values, "max")
        else:
            # Reaching this branch means a feature has no documented grouping.
            raise ValueError(f"No documented FHR aggregation for {feature}")
    return output, primary


def aggregate_uc_features(
    selected: pd.DataFrame,
    prefix: str = "qa",
) -> tuple[dict, dict[str, str]]:
    """Aggregate UC-only features using all or accepted windows."""
    output: dict[str, float] = {}
    primary: dict[str, str] = {}
    duration = selected.get("window_duration_minutes", pd.Series(dtype=float))
    contractions = selected.get("contraction_count", pd.Series(dtype=float))

    def uc_column(feature: str, suffix: str) -> str:
        return f"{feature}__{prefix}_{suffix}"

    def store(feature: str, suffix: str, value: float) -> str:
        column = uc_column(feature, suffix)
        output[column] = value
        return column

    for feature in UC_ONLY_FEATURES:
        values = (
            pd.to_numeric(selected[feature], errors="coerce")
            if feature in selected
            else pd.Series(dtype=float)
        )
        finite = values.dropna()
        if feature in {"mean_uc", "median_uc", "std_uc"}:
            primary[feature] = store(
                feature,
                "duration_weighted_mean",
                finite_weighted_mean(values, duration),
            )
            store(feature, "max", float(finite.max()) if len(finite) else np.nan)
        elif feature == "max_uc":
            primary[feature] = store(
                feature, "max", float(finite.max()) if len(finite) else np.nan
            )
            store(feature, "mean", float(finite.mean()) if len(finite) else np.nan)
        elif feature == "contraction_count":
            primary[feature] = store(
                feature, "sum", float(finite.sum()) if len(finite) else np.nan
            )
            store(feature, "mean", float(finite.mean()) if len(finite) else np.nan)
            store(feature, "max", float(finite.max()) if len(finite) else np.nan)
        elif feature == "mean_contraction_peak":
            primary[feature] = store(
                feature,
                "contraction_weighted_mean",
                finite_weighted_mean(values, contractions),
            )
            store(feature, "max", float(finite.max()) if len(finite) else np.nan)
        elif feature == "mean_contraction_interval_seconds":
            interval_weights = np.maximum(
                np.asarray(contractions, dtype=float) - 1.0, 0.0
            )
            primary[feature] = store(
                feature,
                "interval_weighted_mean",
                finite_weighted_mean(values, interval_weights),
            )
            store(feature, "max", float(finite.max()) if len(finite) else np.nan)
        elif feature == "contraction_interval_std_seconds":
            primary[feature] = store(
                feature,
                "mean",
                float(finite.mean()) if len(finite) else np.nan,
            )
            store(feature, "max", float(finite.max()) if len(finite) else np.nan)
        elif feature == "contraction_frequency_per_hour":
            primary[feature] = store(
                feature,
                "duration_weighted_mean",
                finite_weighted_mean(values, duration),
            )
            store(feature, "max", float(finite.max()) if len(finite) else np.nan)
        else:
            raise ValueError(f"No documented UC aggregation for {feature}")
    return output, primary


def patient_quality_indicators(group: pd.DataFrame) -> dict:
    """Create patient-level signal-quality indicators."""
    accepted = group[group["accepted_by_quality_rule"]]
    total_windows = len(group)
    accepted_count = len(accepted)
    window_sample_weights = group["total_window_sample_count"]
    accepted_duration_minutes = (
        accepted["valid_fhr_sample_count"] / accepted["sampling_frequency"] / 60.0
        if accepted_count
        else pd.Series(dtype=float)
    )
    final_rows = group[group["is_final_window"]]
    final_accepted = bool(
        len(final_rows) and final_rows["accepted_by_quality_rule"].iloc[0]
    )
    full_final_available = bool(
        len(final_rows)
        and final_rows["window_duration_minutes"].iloc[0] >= WINDOW_MINUTES - 1e-6
        and final_accepted
    )
    return {
        "total_window_count": total_windows,
        "accepted_window_count": accepted_count,
        "rejected_window_count": total_windows - accepted_count,
        "accepted_window_percentage": (
            100.0 * accepted_count / total_windows if total_windows else 0.0
        ),
        "mean_valid_fhr_percentage": float(
            group["valid_fhr_percentage"].mean()
        ),
        "overall_valid_fhr_percentage": finite_weighted_mean(
            group["valid_fhr_percentage"], window_sample_weights
        ),
        "minimum_valid_fhr_percentage": float(
            group["valid_fhr_percentage"].min()
        ),
        "maximum_missing_fhr_percentage": float(
            group["missing_fhr_percentage"].max()
        ),
        "overall_missing_fhr_percentage": finite_weighted_mean(
            group["missing_fhr_percentage"], window_sample_weights
        ),
        "maximum_artifact_percentage": float(
            group["artifact_percentage"].max()
        ),
        "overall_artifact_percentage": finite_weighted_mean(
            group["artifact_percentage"], window_sample_weights
        ),
        "maximum_longest_missing_gap_seconds": float(
            group["longest_missing_gap_seconds"].max()
        ),
        "mean_longest_continuous_valid_segment_seconds": float(
            group["longest_continuous_valid_segment_seconds"].mean()
        ),
        "total_valid_fhr_duration_minutes": float(
            accepted_duration_minutes.sum()
        ) if accepted_count else 0.0,
        "has_no_accepted_fhr_window": int(accepted_count == 0),
        "final_window_accepted": int(final_accepted),
        "accepted_final_20_minutes_available": int(full_final_available),
        "analyzed_window_duration_minutes": float(
            group["window_duration_minutes"].sum()
        ),
    }


def aggregate_patient_rows(
    windows: pd.DataFrame,
    record_ids: list[str],
    fhr_features: list[str],
    uc_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Create main and secondary patient representations plus primary mapping."""
    main_rows = []
    secondary_rows = []
    primary_mapping: dict[str, str] = {
        "recording_duration_minutes": "analyzed_window_duration_minutes",
        "percent_missing_fhr": "overall_missing_fhr_percentage",
        "longest_missing_fhr_gap_seconds": "maximum_longest_missing_gap_seconds",
        "valid_fhr_percentage": "overall_valid_fhr_percentage",
        "possible_fhr_artifact_percentage": "overall_artifact_percentage",
    }
    for record_number, record_id in enumerate(record_ids, start=1):
        group = windows[windows["record_id"] == record_id].sort_values(
            "window_number"
        )
        accepted = group[group["accepted_by_quality_rule"]]
        quality = patient_quality_indicators(group)
        fhr_values, fhr_mapping = aggregate_fhr_features(
            accepted, fhr_features
        )
        uc_all_values, uc_all_mapping = aggregate_uc_features(
            group, prefix="qa"
        )
        uc_accepted_values, _ = aggregate_uc_features(
            accepted, prefix="qa"
        )
        main_rows.append(
            {
                "record_id": record_id,
                **fhr_values,
                **uc_all_values,
                **quality,
            }
        )
        secondary_rows.append(
            {
                "record_id": record_id,
                **fhr_values,
                **uc_accepted_values,
                **quality,
            }
        )
        primary_mapping.update(fhr_mapping)
        primary_mapping.update(uc_all_mapping)
        if record_number % 100 == 0 or record_number == len(record_ids):
            print(
                f"  Patient aggregation: {record_number}/{len(record_ids)} records"
            )
    return (
        pd.DataFrame(main_rows),
        pd.DataFrame(secondary_rows),
        primary_mapping,
    )


def attach_outcomes(
    aggregated: pd.DataFrame, baseline: pd.DataFrame
) -> pd.DataFrame:
    """Attach outcome and label fields plus the compatibility quality flag."""
    outcome_columns = [
        "record_id",
        "feature_quality_flag",
        "pH",
        "BE",
        "BDecf",
        "Apgar5",
        "label",
    ]
    return aggregated.merge(
        baseline[outcome_columns],
        on="record_id",
        how="left",
        validate="one_to_one",
    )


def training_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    """Apply the project leakage exclusions while allowing missing values."""
    required = {train_models.TARGET_COLUMN, *train_models.LEAKAGE_COLUMNS}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset is missing leakage/target columns: {missing}")
    matrix = dataset.drop(
        columns=[train_models.TARGET_COLUMN, *train_models.LEAKAGE_COLUMNS]
    )
    non_numeric = matrix.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric model columns remain: {non_numeric}")
    return matrix


def class_metric_name(class_name: str) -> str:
    """Return a stable lowercase metric prefix."""
    return class_name.lower().replace(" ", "_")


def evaluate_predictions(
    target: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    encoder,
    record_ids: np.ndarray,
) -> dict:
    """Calculate the fixed-split and resampled metrics."""
    labels = np.arange(len(encoder.classes_))
    precision, recall, f1, _ = precision_recall_fscore_support(
        target,
        predictions,
        labels=labels,
        zero_division=0,
    )
    path_code = int(encoder.transform(["Pathological"])[0])
    path_mask = target == path_code
    predicted_path = predictions == path_code
    result = {
        "accuracy": accuracy_score(target, predictions),
        "balanced_accuracy": balanced_accuracy_score(target, predictions),
        "macro_f1": f1_score(target, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            target, predictions, average="weighted", zero_division=0
        ),
        "confusion_matrix": json.dumps(
            confusion_matrix(target, predictions, labels=labels).tolist()
        ),
        "pathological_cases_correctly_detected": int(
            np.sum(predictions[path_mask] == path_code)
        ),
        "false_pathological_predictions": int(
            np.sum(predicted_path & ~path_mask)
        ),
        "correct_pathological_record_ids": ";".join(
            sorted(record_ids[path_mask & (predictions == path_code)].tolist())
        ),
        "false_pathological_record_ids": ";".join(
            sorted(record_ids[~path_mask & predicted_path].tolist())
        ),
    }
    for code, class_name in enumerate(encoder.classes_):
        prefix = class_metric_name(class_name)
        result[f"{prefix}_precision"] = float(precision[code])
        result[f"{prefix}_recall"] = float(recall[code])
        result[f"{prefix}_f1"] = float(f1[code])
    return result


def fixed_indices(target: np.ndarray):
    """Reproduce the project stratified 80/20 patient split."""
    indices = np.arange(len(target))
    return train_test_split(
        indices,
        test_size=train_models.TEST_SIZE,
        random_state=train_models.RANDOM_STATE,
        stratify=target,
    )


def fit_predict_split(
    matrix: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    model,
    seed: int,
):
    """Fit only on training patients and return test predictions/probabilities."""
    pipeline = build_comparison_pipeline(model, seed)
    pipeline.fit(matrix.iloc[train_indices], target[train_indices])
    return (
        pipeline,
        pipeline.predict(matrix.iloc[test_indices]),
        pipeline.predict_proba(matrix.iloc[test_indices]),
    )


def fixed_split_comparison(
    baseline_matrix: pd.DataFrame,
    quality_matrix: pd.DataFrame,
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
):
    """Evaluate both representations on the untouched identical fixed split."""
    train_indices, test_indices = fixed_indices(target)
    rows = []
    fitted = {}
    outputs = {}
    for representation, matrix in [
        ("baseline_whole_record", baseline_matrix),
        ("quality_aware", quality_matrix),
    ]:
        pipeline, predictions, probabilities = fit_predict_split(
            matrix,
            target,
            train_indices,
            test_indices,
            model,
            train_models.RANDOM_STATE,
        )
        metrics = evaluate_predictions(
            target[test_indices],
            predictions,
            probabilities,
            encoder,
            record_ids[test_indices],
        )
        row = {
            "representation": representation,
            "model_pipeline": (
                "SimpleImputer(median) -> StandardScaler -> "
                "SMOTE(training only) -> saved LogisticRegression"
            ),
            "train_patient_count": len(train_indices),
            "test_patient_count": len(test_indices),
            **metrics,
        }
        path_code = int(encoder.transform(["Pathological"])[0])
        for record_id in FIXED_TEST_PATHOLOGICAL_IDS:
            positions = np.flatnonzero(record_ids[test_indices] == record_id)
            if len(positions) == 1:
                row[f"pathological_probability_record_{record_id}"] = float(
                    probabilities[positions[0], path_code]
                )
                row[f"predicted_class_record_{record_id}"] = encoder.inverse_transform(
                    [predictions[positions[0]]]
                )[0]
        rows.append(row)
        fitted[representation] = pipeline
        outputs[representation] = {
            "predictions": predictions,
            "probabilities": probabilities,
        }
    return (
        pd.DataFrame(rows),
        train_indices,
        test_indices,
        fitted,
        outputs,
    )


PAIRED_METRICS = [
    "macro_f1",
    "balanced_accuracy",
    "pathological_recall",
    "pathological_precision",
    "suspicious_recall",
    "normal_recall",
    "pathological_cases_correctly_detected",
    "false_pathological_predictions",
]


def repeated_split_comparison(
    baseline_matrix: pd.DataFrame,
    quality_matrix: pd.DataFrame,
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
) -> pd.DataFrame:
    """Evaluate predefined seeds with identical patient IDs for each pair."""
    rows = []
    indices = np.arange(len(target))
    for position, seed in enumerate(SPLIT_SEEDS, start=1):
        train_indices, test_indices = train_test_split(
            indices,
            test_size=train_models.TEST_SIZE,
            random_state=seed,
            stratify=target,
        )
        metrics_by_representation = {}
        for name, matrix in [
            ("baseline", baseline_matrix),
            ("quality_aware", quality_matrix),
        ]:
            _, predictions, probabilities = fit_predict_split(
                matrix,
                target,
                train_indices,
                test_indices,
                model,
                seed,
            )
            metrics_by_representation[name] = evaluate_predictions(
                target[test_indices],
                predictions,
                probabilities,
                encoder,
                record_ids[test_indices],
            )
        row = {
            "random_seed": seed,
            "train_patient_count": len(train_indices),
            "test_patient_count": len(test_indices),
        }
        for metric in PAIRED_METRICS:
            baseline_value = metrics_by_representation["baseline"][metric]
            quality_value = metrics_by_representation["quality_aware"][metric]
            row[f"baseline_{metric}"] = baseline_value
            row[f"quality_aware_{metric}"] = quality_value
            row[f"delta_{metric}"] = quality_value - baseline_value
        rows.append(row)
        print(f"  Repeated split: {position}/{len(SPLIT_SEEDS)}")
    return pd.DataFrame(rows)


def bootstrap_mean_interval(values: np.ndarray) -> tuple[float, float]:
    """Return a paired 95% bootstrap interval for the mean difference."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_STATE)
    samples = rng.choice(
        values,
        size=(BOOTSTRAP_RESAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return float(lower), float(upper)


def summarize_repeated_splits(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize both representations and paired differences without filtering."""
    rows = []
    for metric in PAIRED_METRICS:
        baseline = comparison[f"baseline_{metric}"]
        quality = comparison[f"quality_aware_{metric}"]
        delta = comparison[f"delta_{metric}"]
        lower, upper = bootstrap_mean_interval(delta.to_numpy())
        rows.append(
            {
                "metric": metric,
                "baseline_mean": baseline.mean(),
                "baseline_standard_deviation": baseline.std(ddof=1),
                "quality_aware_mean": quality.mean(),
                "quality_aware_standard_deviation": quality.std(ddof=1),
                "paired_difference_mean": delta.mean(),
                "paired_difference_median": delta.median(),
                "paired_difference_standard_deviation": delta.std(ddof=1),
                "paired_difference_minimum": delta.min(),
                "paired_difference_maximum": delta.max(),
                "paired_mean_bootstrap_95ci_lower": lower,
                "paired_mean_bootstrap_95ci_upper": upper,
                "quality_aware_better_split_count": int((delta > 1e-12).sum()),
                "quality_aware_worse_split_count": int((delta < -1e-12).sum()),
                "equal_split_count": int((np.abs(delta) <= 1e-12).sum()),
            }
        )
    return pd.DataFrame(rows)


def repeated_cv_comparison(
    baseline_matrix: pd.DataFrame,
    quality_matrix: pd.DataFrame,
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
) -> pd.DataFrame:
    """Run paired repeated 5x5 patient-level CV using identical folds."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=train_models.RANDOM_STATE,
    )
    rows = []
    for fold_number, (train_indices, test_indices) in enumerate(
        cv.split(np.zeros(len(target)), target), start=1
    ):
        metrics_by_representation = {}
        seed = train_models.RANDOM_STATE + fold_number
        for name, matrix in [
            ("baseline", baseline_matrix),
            ("quality_aware", quality_matrix),
        ]:
            _, predictions, probabilities = fit_predict_split(
                matrix,
                target,
                train_indices,
                test_indices,
                model,
                seed,
            )
            metrics_by_representation[name] = evaluate_predictions(
                target[test_indices],
                predictions,
                probabilities,
                encoder,
                record_ids[test_indices],
            )
        row = {
            "fold_number": fold_number,
            "repeat_number": (fold_number - 1) // CV_SPLITS + 1,
            "fold_within_repeat": (fold_number - 1) % CV_SPLITS + 1,
        }
        for metric in [
            "macro_f1",
            "balanced_accuracy",
            "pathological_recall",
            "pathological_precision",
            "pathological_f1",
            "suspicious_recall",
            "normal_recall",
            "false_pathological_predictions",
        ]:
            baseline_value = metrics_by_representation["baseline"][metric]
            quality_value = metrics_by_representation["quality_aware"][metric]
            row[f"baseline_{metric}"] = baseline_value
            row[f"quality_aware_{metric}"] = quality_value
            row[f"delta_{metric}"] = quality_value - baseline_value
        rows.append(row)
        print(f"  Repeated CV fold: {fold_number}/{CV_SPLITS * CV_REPEATS}")
    return pd.DataFrame(rows)


def secondary_uc_diagnostic(
    main_matrix: pd.DataFrame,
    secondary_matrix: pd.DataFrame,
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
) -> pd.DataFrame:
    """Compare all-window versus accepted-only UC aggregation on paired CV."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=train_models.RANDOM_STATE,
    )
    rows = []
    metrics = [
        "macro_f1",
        "balanced_accuracy",
        "pathological_recall",
        "pathological_precision",
        "pathological_f1",
        "false_pathological_predictions",
    ]
    for fold_number, (train_indices, test_indices) in enumerate(
        cv.split(np.zeros(len(target)), target), start=1
    ):
        fold_metrics = {}
        seed = train_models.RANDOM_STATE + fold_number
        for name, matrix in [
            ("uc_all_windows", main_matrix),
            ("uc_accepted_windows_only", secondary_matrix),
        ]:
            _, predictions, probabilities = fit_predict_split(
                matrix,
                target,
                train_indices,
                test_indices,
                model,
                seed,
            )
            fold_metrics[name] = evaluate_predictions(
                target[test_indices],
                predictions,
                probabilities,
                encoder,
                record_ids[test_indices],
            )
        row = {
            "fold_number": fold_number,
            "repeat_number": (fold_number - 1) // CV_SPLITS + 1,
            "fold_within_repeat": (fold_number - 1) % CV_SPLITS + 1,
        }
        for metric in metrics:
            main = fold_metrics["uc_all_windows"][metric]
            secondary = fold_metrics["uc_accepted_windows_only"][metric]
            row[f"main_uc_all_{metric}"] = main
            row[f"secondary_uc_accepted_only_{metric}"] = secondary
            row[f"delta_secondary_minus_main_{metric}"] = secondary - main
        rows.append(row)
    result = pd.DataFrame(rows)
    for metric in metrics:
        result[f"main_mean_{metric}"] = result[
            f"main_uc_all_{metric}"
        ].mean()
        result[f"secondary_mean_{metric}"] = result[
            f"secondary_uc_accepted_only_{metric}"
        ].mean()
        result[f"mean_delta_{metric}"] = result[
            f"delta_secondary_minus_main_{metric}"
        ].mean()
    return result


def feature_representation_changes(
    baseline: pd.DataFrame,
    quality_dataset: pd.DataFrame,
    primary_mapping: dict[str, str],
    fixed_test_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Compare each old concept with its documented primary QA aggregation."""
    old = baseline.set_index("record_id")
    new = quality_dataset.set_index("record_id")
    labels = old["label"]
    rows = []
    per_record_scores: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for feature in [
        column
        for column in clinical_numeric_feature_columns(baseline)
        if column in primary_mapping
    ]:
        new_feature = primary_mapping[feature]
        old_values = pd.to_numeric(old[feature], errors="coerce")
        new_values = pd.to_numeric(new[new_feature], errors="coerce")
        paired = old_values.notna() & new_values.notna()
        absolute = (new_values - old_values).abs()
        threshold = absolute.median() + 1.5 * (
            absolute.quantile(0.75) - absolute.quantile(0.25)
        )
        correlation = (
            old_values[paired].corr(new_values[paired])
            if paired.sum() >= 3
            else np.nan
        )
        old_scale = old_values.quantile(0.75) - old_values.quantile(0.25)
        if not np.isfinite(old_scale) or old_scale == 0:
            old_scale = old_values.std(ddof=0)
        if not np.isfinite(old_scale) or old_scale == 0:
            old_scale = 1.0
        for record_id, value in absolute.items():
            if np.isfinite(value):
                per_record_scores[record_id].append(
                    (feature, float(value / old_scale))
                )
        row = {
            "original_feature": feature,
            "quality_aware_feature": new_feature,
            "paired_record_count": int(paired.sum()),
            "records_missing_quality_aware_value": int(new_values.isna().sum()),
            "mean_absolute_patient_change": float(absolute.mean()),
            "median_absolute_patient_change": float(absolute.median()),
            "old_new_correlation": correlation,
            "substantial_change_threshold": float(threshold),
            "records_changing_substantially": int((absolute > threshold).sum()),
            "mean_absolute_change_normal": float(
                absolute[labels == "Normal"].mean()
            ),
            "mean_absolute_change_suspicious": float(
                absolute[labels == "Suspicious"].mean()
            ),
            "mean_absolute_change_pathological": float(
                absolute[labels == "Pathological"].mean()
            ),
            "mean_absolute_change_fixed_test_pathological": float(
                absolute[absolute.index.isin(fixed_test_ids)].mean()
            ),
            "is_priority_feature_concept": feature in IMPORTANT_CHANGE_CONCEPTS,
        }
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        [
            "is_priority_feature_concept",
            "mean_absolute_change_pathological",
        ],
        ascending=[False, False],
    )
    most_changed = {}
    for record_id, values in per_record_scores.items():
        most_changed[record_id] = ";".join(
            feature
            for feature, _ in sorted(values, key=lambda item: item[1], reverse=True)[
                :5
            ]
        )
    return table, most_changed


def clinical_numeric_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return retained model features in dataset order."""
    return [
        column
        for column in dataset.columns
        if column
        not in {
            train_models.TARGET_COLUMN,
            *train_models.LEAKAGE_COLUMNS,
        }
    ]


def pathological_case_comparison(
    baseline_matrix: pd.DataFrame,
    quality_matrix: pd.DataFrame,
    quality_dataset: pd.DataFrame,
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    fixed_outputs: dict,
    most_changed: dict[str, str],
) -> pd.DataFrame:
    """Use fixed-test and fixed-training OOF predictions for all 27 cases."""
    prediction_store = {}
    probability_store = {}
    for representation, matrix in [
        ("baseline", baseline_matrix),
        ("quality_aware", quality_matrix),
    ]:
        predictions = np.full(len(target), -1, dtype=int)
        probabilities = np.full((len(target), len(encoder.classes_)), np.nan)
        fixed_name = (
            "baseline_whole_record"
            if representation == "baseline"
            else "quality_aware"
        )
        predictions[test_indices] = fixed_outputs[fixed_name]["predictions"]
        probabilities[test_indices] = fixed_outputs[fixed_name]["probabilities"]

        fixed_training_target = target[train_indices]
        cv = StratifiedKFold(
            n_splits=CV_SPLITS,
            shuffle=True,
            random_state=train_models.RANDOM_STATE,
        )
        for fold_number, (inner_train, inner_validation) in enumerate(
            cv.split(np.zeros(len(train_indices)), fixed_training_target),
            start=1,
        ):
            fit_indices = train_indices[inner_train]
            validation_indices = train_indices[inner_validation]
            pipeline = build_comparison_pipeline(
                model, train_models.RANDOM_STATE + fold_number
            )
            pipeline.fit(matrix.iloc[fit_indices], target[fit_indices])
            predictions[validation_indices] = pipeline.predict(
                matrix.iloc[validation_indices]
            )
            probabilities[validation_indices] = pipeline.predict_proba(
                matrix.iloc[validation_indices]
            )
        if np.any(predictions < 0) or np.isnan(probabilities).any():
            raise AssertionError("Pathological OOF/fixed prediction coverage failed")
        prediction_store[representation] = predictions
        probability_store[representation] = probabilities

    path_code = int(encoder.transform(["Pathological"])[0])
    qa_lookup = quality_dataset.set_index("record_id")
    train_set = set(record_ids[train_indices])
    rows = []
    for position in np.flatnonzero(target == path_code):
        record_id = record_ids[position]
        baseline_prediction = prediction_store["baseline"][position]
        quality_prediction = prediction_store["quality_aware"][position]
        baseline_probability = probability_store["baseline"][position, path_code]
        quality_probability = probability_store["quality_aware"][
            position, path_code
        ]
        quality_row = qa_lookup.loc[record_id]
        rows.append(
            {
                "record_id": record_id,
                "fixed_split_membership": (
                    "training_oof" if record_id in train_set else "fixed_test"
                ),
                "accepted_window_count": int(
                    quality_row["accepted_window_count"]
                ),
                "rejected_window_count": int(
                    quality_row["rejected_window_count"]
                ),
                "accepted_window_percentage": quality_row[
                    "accepted_window_percentage"
                ],
                "has_no_accepted_fhr_window": int(
                    quality_row["has_no_accepted_fhr_window"]
                ),
                "baseline_model_prediction": encoder.inverse_transform(
                    [baseline_prediction]
                )[0],
                "quality_aware_model_prediction": encoder.inverse_transform(
                    [quality_prediction]
                )[0],
                "baseline_pathological_probability": baseline_probability,
                "quality_aware_pathological_probability": quality_probability,
                "pathological_probability_difference": (
                    quality_probability - baseline_probability
                ),
                "baseline_correct": bool(baseline_prediction == path_code),
                "quality_aware_correct": bool(quality_prediction == path_code),
                "most_changed_important_features": most_changed.get(
                    record_id, ""
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("record_id")


def confusion_matrix_figures(
    fixed_table: pd.DataFrame, encoder
) -> list[Path]:
    """Save one readable fixed-test confusion-matrix figure per representation."""
    paths = []
    for _, row in fixed_table.iterrows():
        matrix = np.asarray(json.loads(row["confusion_matrix"]), dtype=int)
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        image = axis.imshow(matrix, cmap="Blues")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    fontsize=13,
                    color=(
                        "white"
                        if matrix[row_index, column_index] > matrix.max() / 2
                        else "black"
                    ),
                )
        axis.set_xticks(np.arange(len(encoder.classes_)), encoder.classes_)
        axis.set_yticks(np.arange(len(encoder.classes_)), encoder.classes_)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(
            "Fixed-test confusion matrix\n"
            + row["representation"].replace("_", " ").title()
        )
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        path = FIGURE_DIR / f"confusion_matrix_{row['representation']}.png"
        figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def additional_figures(
    changes: pd.DataFrame,
    repeated_splits: pd.DataFrame,
) -> list[Path]:
    """Save compact feature-change and paired-split diagnostics."""
    priority = (
        changes[changes["is_priority_feature_concept"]]
        .sort_values("mean_absolute_change_pathological", ascending=False)
        .head(15)
        .sort_values("mean_absolute_change_pathological")
    )
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        priority["original_feature"],
        priority["mean_absolute_change_pathological"],
        color="#4c78a8",
    )
    axis.set_xlabel("Mean absolute change among Pathological records")
    axis.set_title("Priority feature concepts changed by quality-aware aggregation")
    figure.tight_layout()
    path_changes = FIGURE_DIR / "priority_feature_representation_changes.png"
    figure.savefig(path_changes, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].plot(
        repeated_splits["random_seed"],
        repeated_splits["delta_pathological_recall"],
        marker="o",
        color="#d62728",
    )
    axes[0].set_title("Paired Pathological recall change")
    axes[0].set_xlabel("Predefined split seed")
    axes[0].set_ylabel("Quality-aware minus baseline")
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].plot(
        repeated_splits["random_seed"],
        repeated_splits["delta_macro_f1"],
        marker="o",
        color="#1f77b4",
    )
    axes[1].set_title("Paired Macro F1 change")
    axes[1].set_xlabel("Predefined split seed")
    axes[1].set_ylabel("Quality-aware minus baseline")
    figure.tight_layout()
    path_splits = FIGURE_DIR / "paired_repeated_split_changes.png"
    figure.savefig(path_splits, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return [path_changes, path_splits]


def choose_conclusion(
    repeated_summary: pd.DataFrame,
    repeated_cv: pd.DataFrame,
) -> str:
    """Select exactly one required conclusion using paired repeated evidence."""
    split = repeated_summary.set_index("metric")
    recall_delta = split.loc["pathological_recall", "paired_difference_mean"]
    macro_delta = split.loc["macro_f1", "paired_difference_mean"]
    false_delta = split.loc[
        "false_pathological_predictions", "paired_difference_mean"
    ]
    recall_better = split.loc[
        "pathological_recall", "quality_aware_better_split_count"
    ]
    cv_recall_delta = repeated_cv["delta_pathological_recall"].mean()
    cv_macro_delta = repeated_cv["delta_macro_f1"].mean()

    if (
        recall_delta >= 0.10
        and macro_delta >= 0
        and cv_recall_delta >= 0.08
        and cv_macro_delta >= 0
        and recall_better > len(SPLIT_SEEDS) / 2
        and false_delta <= 3
    ):
        return "quality-aware aggregation provides a clear improvement"
    if (
        recall_delta >= 0.05
        and cv_recall_delta >= 0.04
        and macro_delta >= -0.03
        and cv_macro_delta >= -0.03
        and false_delta <= 4
    ):
        return "quality-aware aggregation provides a useful recall trade-off"
    if (
        abs(recall_delta) < 0.05
        and abs(cv_recall_delta) < 0.05
        and abs(macro_delta) < 0.02
        and abs(cv_macro_delta) < 0.02
    ):
        return "quality-aware aggregation produces no meaningful improvement"
    if recall_delta < -0.05 and cv_recall_delta < -0.05 and macro_delta < 0:
        return "quality-aware aggregation reduces performance"
    return "evidence is mixed and further temporal modelling is justified"


def group_mapping_markdown(
    fhr_features: list[str], uc_features: list[str]
) -> str:
    """Document the actual feature-column grouping."""
    def wrapped(values):
        return ", ".join(f"`{value}`" for value in values)

    return (
        "**FHR-derived/dependent or FHR–UC interaction columns:** "
        + wrapped(fhr_features)
        + "\n\n**UC-only columns:** "
        + wrapped(uc_features)
    )


def repeated_metric_line(
    summary: pd.DataFrame, metric: str, label: str
) -> str:
    """Format one paired repeated-split result for the report."""
    row = summary.set_index("metric").loc[metric]
    return (
        f"- {label}: baseline {row['baseline_mean']:.4f} ± "
        f"{row['baseline_standard_deviation']:.4f}; quality-aware "
        f"{row['quality_aware_mean']:.4f} ± "
        f"{row['quality_aware_standard_deviation']:.4f}; paired mean "
        f"difference {row['paired_difference_mean']:+.4f} "
        f"(95% bootstrap CI {row['paired_mean_bootstrap_95ci_lower']:+.4f} to "
        f"{row['paired_mean_bootstrap_95ci_upper']:+.4f}); better/worse/equal "
        f"splits {int(row['quality_aware_better_split_count'])}/"
        f"{int(row['quality_aware_worse_split_count'])}/"
        f"{int(row['equal_split_count'])}."
    )


def write_report(
    windows: pd.DataFrame,
    quality_dataset: pd.DataFrame,
    no_accepted_ids: list[str],
    fhr_features: list[str],
    uc_features: list[str],
    changes: pd.DataFrame,
    fixed: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    repeated_cv: pd.DataFrame,
    uc_diagnostic: pd.DataFrame,
    pathological_cases: pd.DataFrame,
    conclusion: str,
) -> None:
    """Write the controlled-experiment interpretation."""
    accepted = int(windows["accepted_by_quality_rule"].sum())
    baseline_fixed = fixed.set_index("representation").loc[
        "baseline_whole_record"
    ]
    quality_fixed = fixed.set_index("representation").loc["quality_aware"]
    cv_metrics = {}
    for metric in [
        "macro_f1",
        "balanced_accuracy",
        "pathological_recall",
        "pathological_precision",
        "pathological_f1",
    ]:
        cv_metrics[metric] = {
            "baseline_mean": repeated_cv[f"baseline_{metric}"].mean(),
            "baseline_std": repeated_cv[f"baseline_{metric}"].std(ddof=1),
            "quality_mean": repeated_cv[f"quality_aware_{metric}"].mean(),
            "quality_std": repeated_cv[f"quality_aware_{metric}"].std(ddof=1),
            "delta_mean": repeated_cv[f"delta_{metric}"].mean(),
            "better_folds": int((repeated_cv[f"delta_{metric}"] > 1e-12).sum()),
        }
    top_changes = (
        changes[changes["is_priority_feature_concept"]]
        .sort_values("mean_absolute_change_pathological", ascending=False)
        .head(10)["original_feature"]
        .tolist()
    )
    main_uc_macro = uc_diagnostic["main_uc_all_macro_f1"].mean()
    secondary_uc_macro = uc_diagnostic[
        "secondary_uc_accepted_only_macro_f1"
    ].mean()
    main_uc_recall = uc_diagnostic["main_uc_all_pathological_recall"].mean()
    secondary_uc_recall = uc_diagnostic[
        "secondary_uc_accepted_only_pathological_recall"
    ].mean()
    split_false = repeated_summary.set_index("metric").loc[
        "false_pathological_predictions"
    ]
    path_retained = int(
        (
            pathological_cases["accepted_window_count"] > 0
        ).sum()
    )
    report = f"""# Quality-Aware Patient-Level Aggregation Experiment

## 1. Controlled hypothesis

This experiment changed only the patient-level feature representation. The
baseline remained the existing `ml_dataset.csv`; it was not recalculated.
The quality-aware dataset recomputed current features in 20-minute waveform
windows, rejected FHR contributions below {MINIMUM_VALID_FHR_PERCENTAGE:.0f}%
valid FHR, and aggregated the remaining values back to exactly one row per
patient. No target was assigned to a window and no window classifier was
trained.

Unchanged elements were the 552 patient IDs, labels and outcomes, leakage
exclusions, LabelEncoder, selected Logistic Regression family and parameters,
training-only SMOTE approach, fixed 80/20 split, random state 42, predefined
30 split seeds, and repeated 5x5 patient-level cross-validation design.

## 2. Exact comparison pipeline

Both representations used:

`SimpleImputer(strategy="median") -> StandardScaler() -> SMOTE(training fold only) -> LogisticRegression(C=1, solver="lbfgs", max_iter=2000, class_weight=None)`.

Imputation was learned inside each training fold. SMOTE followed imputation
and scaling and never saw a validation or test patient. No hyperparameter,
threshold, seed, or quality rule was selected from the fixed test results.

## 3. Window generation and quality rule

The experiment created **{len(windows)}** diagnostic windows. A shorter final
window was retained only when it contained at least
{MINIMUM_FINAL_WINDOW_MINUTES:.0f} minutes. **{accepted}**
({100 * accepted / len(windows):.1f}%) met valid FHR >=80%. Long gaps were not
interpolated, and current event detection continued to terminate events at
missing samples.

The earlier audit's **{PRIOR_AUDIT_WINDOW_COUNT}** count included every
residual end slice. Applying this experiment's required minimum final-window
duration excluded **{PRIOR_AUDIT_WINDOW_COUNT - len(windows)}** sub-10-minute
end slices, so its window and acceptance counts are deliberately lower.

Rejected windows did not contribute ordinary values to FHR-derived or
FHR-dependent aggregation. UC-only features used all retained diagnostic
windows in the main experiment.

## 4. Actual feature grouping

{group_mapping_markdown(fhr_features, uc_features)}

## 5. Patient aggregation

- Counts were summed, with mean and maximum window counts retained where useful.
- Densities used duration-weighted means and maxima.
- Percentages used valid-FHR-sample-weighted means and local maxima.
- Baseline used a valid-sample-weighted mean, minimum, maximum, and range.
- Variability used window mean, minimum, and between-window standard deviation.
- Severity and longest-duration features used maxima plus descriptive mean or
  median values.
- Event mean durations were weighted by their corresponding event counts.
- UC summaries used duration or contraction weighting, while UC counts were
  summed.

The aggregation map is encoded explicitly in the experiment script rather
than applying all five statistics blindly to every feature.

## 6. Records without accepted FHR windows

**{len(no_accepted_ids)}** records had no accepted FHR window:
**{", ".join(no_accepted_ids)}**.

Their rows were preserved. FHR-derived aggregates remained NaN,
`has_no_accepted_fhr_window` was set to 1, and UC-only all-window aggregates
were retained. The fold-local median imputer handled these NaNs. No patient was
silently removed. **{path_retained}/27** Pathological patients retained at
least one accepted FHR window; record 2013 did not.

## 7. Feature representation changes

The priority concepts with the largest mean absolute changes among
Pathological records were: **{", ".join(top_changes)}**.

`feature_representation_changes.csv` reports the closest documented mapping,
absolute changes, old/new correlation, conservative substantial-change counts,
and separate Normal, Suspicious, Pathological, and fixed-test Pathological
summaries.

## 8. Fixed-test comparison

- Baseline Pathological recall:
  **{baseline_fixed['pathological_recall']:.4f}**
  ({int(baseline_fixed['pathological_cases_correctly_detected'])}/6).
- Quality-aware Pathological recall:
  **{quality_fixed['pathological_recall']:.4f}**
  ({int(quality_fixed['pathological_cases_correctly_detected'])}/6).
- Baseline Macro F1 / balanced accuracy:
  **{baseline_fixed['macro_f1']:.4f} / {baseline_fixed['balanced_accuracy']:.4f}**.
- Quality-aware Macro F1 / balanced accuracy:
  **{quality_fixed['macro_f1']:.4f} / {quality_fixed['balanced_accuracy']:.4f}**.
- False Pathological predictions:
  baseline **{int(baseline_fixed['false_pathological_predictions'])}**,
  quality-aware **{int(quality_fixed['false_pathological_predictions'])}**.

The fixed test set was evaluated once and was not used for selection.

## 9. Thirty predefined split comparison

{repeated_metric_line(repeated_summary, 'pathological_recall', 'Pathological recall')}

{repeated_metric_line(repeated_summary, 'pathological_precision', 'Pathological precision')}

{repeated_metric_line(repeated_summary, 'macro_f1', 'Macro F1')}

{repeated_metric_line(repeated_summary, 'balanced_accuracy', 'Balanced accuracy')}

{repeated_metric_line(repeated_summary, 'false_pathological_predictions', 'False Pathological predictions')}

## 10. Repeated patient-level cross-validation

- Macro F1: baseline **{cv_metrics['macro_f1']['baseline_mean']:.4f} ± {cv_metrics['macro_f1']['baseline_std']:.4f}**;
  quality-aware **{cv_metrics['macro_f1']['quality_mean']:.4f} ± {cv_metrics['macro_f1']['quality_std']:.4f}**;
  paired mean change **{cv_metrics['macro_f1']['delta_mean']:+.4f}**.
- Balanced accuracy: baseline **{cv_metrics['balanced_accuracy']['baseline_mean']:.4f}**;
  quality-aware **{cv_metrics['balanced_accuracy']['quality_mean']:.4f}**;
  change **{cv_metrics['balanced_accuracy']['delta_mean']:+.4f}**.
- Pathological recall: baseline **{cv_metrics['pathological_recall']['baseline_mean']:.4f} ± {cv_metrics['pathological_recall']['baseline_std']:.4f}**;
  quality-aware **{cv_metrics['pathological_recall']['quality_mean']:.4f} ± {cv_metrics['pathological_recall']['quality_std']:.4f}**;
  change **{cv_metrics['pathological_recall']['delta_mean']:+.4f}**.
- Pathological precision: baseline **{cv_metrics['pathological_precision']['baseline_mean']:.4f}**;
  quality-aware **{cv_metrics['pathological_precision']['quality_mean']:.4f}**;
  change **{cv_metrics['pathological_precision']['delta_mean']:+.4f}**.

The table contains all 25 paired folds; no favourable fold was selected.

## 11. Pathological cases

`pathological_case_comparison.csv` includes all 27 records. Fixed-test cases
use the fixed models; fixed-training cases use five-fold out-of-fold
predictions, avoiding optimistic predictions from a model trained on the same
patient.

## 12. UC aggregation diagnostic

Using UC from all windows produced repeated-CV Macro F1
**{main_uc_macro:.4f}** and Pathological recall **{main_uc_recall:.4f}**.
Restricting UC to accepted FHR windows produced Macro F1
**{secondary_uc_macro:.4f}** and Pathological recall
**{secondary_uc_recall:.4f}**.

This secondary result is diagnostic. It does not replace the main version
unless it shows a clear, stable paired advantage.

## 13. Statistical interpretation

The interpretation uses paired split/fold differences, recall and precision,
Macro F1, balanced accuracy, false alarms, variability, and patient coverage.
A numerical increase on one fixed split is not called an improvement by
itself. The mean change in false Pathological predictions across predefined
splits was **{split_false['paired_difference_mean']:+.2f}**.

This conclusion is based on all paired repeated-split and repeated-CV results,
not on the fixed test set alone.

## Conclusion

{conclusion}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def validate_dataset(
    quality_dataset: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:
    """Validate one row per patient and matching outcome and label fields."""
    if len(quality_dataset) != 552:
        raise AssertionError("Quality-aware dataset must have 552 rows")
    if quality_dataset["record_id"].duplicated().any():
        raise AssertionError("Quality-aware dataset has duplicate record IDs")
    if set(quality_dataset["record_id"]) != set(baseline["record_id"]):
        raise AssertionError("Quality-aware patient IDs differ from baseline")
    aligned_new = quality_dataset.set_index("record_id")
    aligned_old = baseline.set_index("record_id")
    for column in ["pH", "BE", "BDecf", "Apgar5", "label"]:
        old = aligned_old.loc[aligned_new.index, column]
        new = aligned_new[column]
        if column == "label":
            equal = old.equals(new)
        else:
            equal = np.allclose(old, new, equal_nan=True)
        if not equal:
            raise AssertionError(f"Outcome/label changed: {column}")
    if not quality_dataset["label"].value_counts().equals(
        baseline["label"].value_counts()
    ):
        raise AssertionError("Class distribution changed")


def validate_outputs(
    windows: pd.DataFrame,
    quality_dataset: pd.DataFrame,
    changes: pd.DataFrame,
    fixed: pd.DataFrame,
    repeated_splits: pd.DataFrame,
    repeated_summary: pd.DataFrame,
    repeated_cv: pd.DataFrame,
    pathological_cases: pd.DataFrame,
    uc_diagnostic: pd.DataFrame,
    figures: list[Path],
) -> None:
    """Run compact coverage, pairing, range, and artifact validation."""
    validate_dataset(quality_dataset, pd.read_csv(BASELINE_DATASET_PATH).assign(
        record_id=lambda frame: normalize_record_id(frame["record_id"])
    ))
    if windows["record_id"].nunique() != 552:
        raise AssertionError("Window table does not cover all records")
    if windows[["record_id", "window_number"]].duplicated().any():
        raise AssertionError("Duplicate diagnostic windows")
    if (windows["window_duration_minutes"] < MINIMUM_FINAL_WINDOW_MINUTES).any():
        raise AssertionError("A final window shorter than 10 minutes was included")
    if len(fixed) != 2:
        raise AssertionError("Fixed comparison must contain two representations")
    if len(repeated_splits) != len(SPLIT_SEEDS):
        raise AssertionError("Repeated split output must contain all 30 seeds")
    if set(repeated_splits["random_seed"]) != set(SPLIT_SEEDS):
        raise AssertionError("Repeated split seeds changed")
    if len(repeated_summary) != len(PAIRED_METRICS):
        raise AssertionError("Repeated split summary is incomplete")
    if len(repeated_cv) != CV_SPLITS * CV_REPEATS:
        raise AssertionError("Repeated CV must contain 25 paired folds")
    if len(pathological_cases) != 27:
        raise AssertionError("Pathological case table must have 27 records")
    if len(uc_diagnostic) != CV_SPLITS * CV_REPEATS:
        raise AssertionError("UC diagnostic must contain 25 paired folds")
    if changes["original_feature"].duplicated().any():
        raise AssertionError("Feature-change mappings are duplicated")
    if len(figures) != 4 or not all(path.exists() for path in figures):
        raise AssertionError("Expected four experiment figures")
    if any(path.stat().st_size < 15_000 for path in figures):
        raise AssertionError("At least one experiment figure is unexpectedly small")
    if not REPORT_PATH.exists() or not QUALITY_AWARE_DATASET_PATH.exists():
        raise AssertionError("Final report or quality-aware dataset is missing")


def main() -> None:
    """Run the controlled quality-aware aggregation experiment."""
    ensure_new_outputs()
    (
        baseline,
        _labels,
        _clinical,
        _pathological_windows,
        _pathological_quality,
        selected_model,
        encoder,
    ) = load_inputs()
    record_ids = baseline["record_id"].to_numpy()
    target = encoder.transform(baseline["label"])
    fhr_features, uc_features = current_feature_groups()
    waveform_directory = visual.find_waveform_directory(record_ids)

    print("Selected comparison classifier:")
    print(selected_classifier(selected_model))
    print(
        "Pipeline: median imputation -> scaling -> training-only SMOTE -> "
        "selected Logistic Regression"
    )
    print(f"Read-only waveform directory: {waveform_directory}")

    print("\nGenerating diagnostic windows...")
    windows = generate_all_window_features(
        record_ids.tolist(), waveform_directory
    )
    windows.to_csv(ALL_WINDOWS_PATH, index=False)

    print("\nAggregating one row per patient...")
    main_aggregated, secondary_aggregated, primary_mapping = aggregate_patient_rows(
        windows,
        record_ids.tolist(),
        fhr_features,
        uc_features,
    )
    quality_dataset = attach_outcomes(main_aggregated, baseline)
    secondary_dataset = attach_outcomes(secondary_aggregated, baseline)
    validate_dataset(quality_dataset, baseline)
    quality_dataset.to_csv(QUALITY_AWARE_DATASET_PATH, index=False)

    no_accepted_ids = quality_dataset.loc[
        quality_dataset["has_no_accepted_fhr_window"] == 1, "record_id"
    ].tolist()
    baseline_matrix = training_matrix(baseline)
    quality_matrix = training_matrix(quality_dataset)
    secondary_matrix = training_matrix(secondary_dataset)

    changes, most_changed = feature_representation_changes(
        baseline,
        quality_dataset,
        primary_mapping,
        set(FIXED_TEST_PATHOLOGICAL_IDS),
    )
    changes.to_csv(FEATURE_CHANGES_PATH, index=False)

    print("\nRunning fixed-split comparison...")
    (
        fixed,
        train_indices,
        test_indices,
        _fitted,
        fixed_outputs,
    ) = fixed_split_comparison(
        baseline_matrix,
        quality_matrix,
        target,
        record_ids,
        encoder,
        selected_model,
    )
    fixed.to_csv(FIXED_COMPARISON_PATH, index=False)

    print("\nRunning 30 predefined paired splits...")
    repeated_splits = repeated_split_comparison(
        baseline_matrix,
        quality_matrix,
        target,
        record_ids,
        encoder,
        selected_model,
    )
    repeated_splits.to_csv(REPEATED_SPLIT_PATH, index=False)
    repeated_summary = summarize_repeated_splits(repeated_splits)
    repeated_summary.to_csv(REPEATED_SPLIT_SUMMARY_PATH, index=False)

    print("\nRunning repeated 5x5 paired cross-validation...")
    repeated_cv = repeated_cv_comparison(
        baseline_matrix,
        quality_matrix,
        target,
        record_ids,
        encoder,
        selected_model,
    )
    repeated_cv.to_csv(REPEATED_CV_PATH, index=False)

    print("\nGenerating Pathological fixed/OOF comparison...")
    pathological_cases = pathological_case_comparison(
        baseline_matrix,
        quality_matrix,
        quality_dataset,
        target,
        record_ids,
        encoder,
        selected_model,
        train_indices,
        test_indices,
        fixed_outputs,
        most_changed,
    )
    pathological_cases.to_csv(PATHOLOGICAL_CASE_PATH, index=False)

    print("\nRunning secondary UC aggregation diagnostic...")
    uc_diagnostic = secondary_uc_diagnostic(
        quality_matrix,
        secondary_matrix,
        target,
        record_ids,
        encoder,
        selected_model,
    )
    uc_diagnostic.to_csv(UC_DIAGNOSTIC_PATH, index=False)

    conclusion = choose_conclusion(repeated_summary, repeated_cv)
    figures = confusion_matrix_figures(fixed, encoder)
    figures.extend(additional_figures(changes, repeated_splits))
    write_report(
        windows,
        quality_dataset,
        no_accepted_ids,
        fhr_features,
        uc_features,
        changes,
        fixed,
        repeated_summary,
        repeated_cv,
        uc_diagnostic,
        pathological_cases,
        conclusion,
    )
    validate_outputs(
        windows,
        quality_dataset,
        changes,
        fixed,
        repeated_splits,
        repeated_summary,
        repeated_cv,
        pathological_cases,
        uc_diagnostic,
        figures,
    )

    fixed_lookup = fixed.set_index("representation")
    split_summary = repeated_summary.set_index("metric")
    accepted = int(windows["accepted_by_quality_rule"].sum())
    pathological_retained = int(
        (
            pathological_cases["accepted_window_count"] > 0
        ).sum()
    )
    false_change = split_summary.loc[
        "false_pathological_predictions", "paired_difference_mean"
    ]
    print("\nQuality-aware aggregation experiment complete")
    print(f"Records processed: {len(quality_dataset)}")
    print(f"Windows created: {len(windows)}")
    print(
        f"Windows accepted: {accepted}/{len(windows)} "
        f"({100 * accepted / len(windows):.1f}%)"
    )
    print(
        "Records with at least one accepted window: "
        f"{len(quality_dataset) - len(no_accepted_ids)}/552"
    )
    print("Records with no accepted window: " + ", ".join(no_accepted_ids))
    print(f"Pathological records retained: {pathological_retained}/27")
    print(
        "Fixed-test baseline Pathological recall: "
        f"{fixed_lookup.loc['baseline_whole_record', 'pathological_recall']:.4f}"
    )
    print(
        "Fixed-test quality-aware Pathological recall: "
        f"{fixed_lookup.loc['quality_aware', 'pathological_recall']:.4f}"
    )
    print(
        "Repeated-split baseline Pathological recall: "
        f"{split_summary.loc['pathological_recall', 'baseline_mean']:.4f} ± "
        f"{split_summary.loc['pathological_recall', 'baseline_standard_deviation']:.4f}"
    )
    print(
        "Repeated-split quality-aware Pathological recall: "
        f"{split_summary.loc['pathological_recall', 'quality_aware_mean']:.4f} ± "
        f"{split_summary.loc['pathological_recall', 'quality_aware_standard_deviation']:.4f}"
    )
    print(
        "Repeated-CV baseline Macro F1: "
        f"{repeated_cv['baseline_macro_f1'].mean():.4f}"
    )
    print(
        "Repeated-CV quality-aware Macro F1: "
        f"{repeated_cv['quality_aware_macro_f1'].mean():.4f}"
    )
    print(
        "Mean change in false Pathological predictions: "
        f"{false_change:+.2f}"
    )
    print(f"Main conclusion: {conclusion}")
    print(f"Quality-aware dataset: {QUALITY_AWARE_DATASET_PATH}")
    print(f"Diagnostic windows: {ALL_WINDOWS_PATH}")
    print(f"Feature changes: {FEATURE_CHANGES_PATH}")
    print(f"Fixed comparison: {FIXED_COMPARISON_PATH}")
    print(f"Repeated splits: {REPEATED_SPLIT_PATH}")
    print(f"Repeated split summary: {REPEATED_SPLIT_SUMMARY_PATH}")
    print(f"Repeated CV: {REPEATED_CV_PATH}")
    print(f"Pathological cases: {PATHOLOGICAL_CASE_PATH}")
    print(f"UC diagnostic: {UC_DIAGNOSTIC_PATH}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
