"""Audit temporal information lost by whole-record CTG feature aggregation.

This is a diagnostic, patient-level experiment.  Existing 20-minute window
features are summarized chronologically, but windows never receive independent
labels and no window-level classifier is trained.

Run from the project root:

    python src/analysis/audit_temporal_information_loss.py

"""

from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import diagnose_pathological_cases as prior_diagnostic
from src.experiments import quality_aware_aggregation_experiment as qa
from src.features import extract_clinical_features as extractor
from src.models import train_models
from src.utils.record_ids import normalize_record_id

# ---------------------------------------------------------------------------
# Read-only inputs and new diagnostic outputs
# ---------------------------------------------------------------------------
WINDOW_PATH = (
    PROJECT_ROOT
    / "reports"
    / "quality_aware_aggregation"
    / "all_window_features.csv"
)
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.csv"
CLINICAL_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"
BASELINE_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
PATHOLOGICAL_WINDOW_PATH = (
    PROJECT_ROOT
    / "reports"
    / "pathological_visual_review"
    / "pathological_window_diagnostics.csv"
)
FEATURE_QUALITY_PATH = (
    PROJECT_ROOT / "reports" / "signal_quality_audit" / "feature_by_quality.csv"
)
REPRESENTATION_CHANGE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "quality_aware_aggregation"
    / "feature_representation_changes.csv"
)
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

OUTPUT_ROOT = PROJECT_ROOT / "reports" / "temporal_information_audit"
FIGURE_DIR = OUTPUT_ROOT / "figures"
SUMMARY_PATH = OUTPUT_ROOT / "temporal_patient_summaries.csv"
WHOLE_VS_TEMPORAL_PATH = OUTPUT_ROOT / "whole_record_vs_temporal.csv"
UNIVARIATE_PATH = OUTPUT_ROOT / "univariate_feature_discrimination.csv"
PAIRED_RANKING_PATH = OUTPUT_ROOT / "paired_feature_ranking.csv"
PATTERN_PATH = OUTPUT_ROOT / "temporal_pattern_categories.csv"
PATHOLOGICAL_CASE_PATH = OUTPUT_ROOT / "pathological_temporal_case_audit.csv"
SINGLE_SCREEN_PATH = OUTPUT_ROOT / "single_feature_addition_screen.csv"
GROUP_SCREEN_PATH = OUTPUT_ROOT / "temporal_group_screen.csv"
REPORT_PATH = OUTPUT_ROOT / "temporal_information_audit_report.md"

# Fixed, documented audit settings.  These are engineering/descriptive rules,
# not clinical cutoffs.
MIN_VALID_FHR_PERCENTAGE = 80.0
LOW_QUANTILE = 0.10
HIGH_QUANTILE = 0.90
MAX_SINGLE_FEATURE_CANDIDATES = 20
NEAR_DUPLICATE_CORRELATION = 0.95
SUBSTANTIAL_CHANGE_FRACTION = 0.25
MIN_ABSOLUTE_SUBSTANTIAL_CHANGE = 1e-9
PATTERN_POOR_QUALITY_FRACTION = 0.50
PATTERN_CONSISTENT_ABNORMAL_FRACTION = 0.50
PATTERN_TREND_SPEARMAN = 0.50
PATTERN_MIN_TREND_WINDOWS = 3
BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_RANDOM_STATE = 42
FIXED_TEST_PATHOLOGICAL_IDS = {"1002", "1071", "1158", "1418", "1490", "2009"}

# Feature groups are resolved against columns produced by the extractor. Any
# unavailable columns are reported by the audit.
FEATURE_GROUP_REQUESTS = {
    "FHR summary": [
        "mean_fhr", "median_fhr", "min_fhr", "max_fhr", "std_fhr"
    ],
    "Baseline and baseline dynamics": [
        "baseline_fhr", "baseline_drift_std", "baseline_drift_range",
        "baseline_crossing_count", "percentage_time_above_baseline",
        "percentage_time_below_baseline",
    ],
    "Variability": [
        "short_term_variability", "long_term_variability", "segment_stv_mean",
        "segment_stv_std", "segment_ltv_mean", "segment_ltv_std",
    ],
    "Accelerations": [
        "acceleration_count", "acceleration_density_per_hour",
        "mean_acceleration_duration_seconds", "longest_acceleration_duration_seconds",
        "max_acceleration_amplitude",
    ],
    "Decelerations": [
        "deceleration_count", "deceleration_density_per_hour",
        "max_deceleration_depth", "mean_deceleration_duration_seconds",
        "longest_deceleration_duration_seconds",
    ],
    "Bradycardia and tachycardia": [
        "bradycardia_duration_seconds", "bradycardia_percentage",
        "tachycardia_duration_seconds", "tachycardia_percentage",
    ],
    "UC and contractions": list(qa.UC_ONLY_FEATURES),
    "FHR-UC interactions": [
        "decelerations_per_contraction",
        "contractions_followed_by_deceleration_percentage",
        "mean_delay_contraction_to_deceleration_seconds",
    ],
    "Signal quality": [
        "valid_fhr_percentage", "missing_fhr_percentage", "artifact_percentage",
        "longest_missing_gap_seconds",
        "longest_continuous_valid_segment_seconds", "number_of_valid_segments",
    ],
}

COUNT_FEATURES = {
    "acceleration_count", "deceleration_count", "contraction_count",
    "baseline_crossing_count", "number_of_valid_segments",
}
DENSITY_FEATURES = {
    "acceleration_density_per_hour", "deceleration_density_per_hour",
    "contraction_frequency_per_hour",
}
DURATION_FEATURES = {
    feature for feature in extractor.FEATURE_COLUMNS if "duration_seconds" in feature
} | {"longest_missing_gap_seconds", "longest_continuous_valid_segment_seconds"}
SEVERITY_FEATURES = {
    "max_acceleration_amplitude", "max_deceleration_depth",
    "longest_acceleration_duration_seconds",
    "longest_deceleration_duration_seconds", "baseline_drift_range",
    "maximum_negative_fhr_slope", "max_fhr", "max_uc",
}
VARIABILITY_FEATURES = {
    "short_term_variability", "long_term_variability", "std_fhr",
    "segment_stv_mean", "segment_stv_std", "segment_ltv_mean",
    "segment_ltv_std", "baseline_drift_std", "std_uc",
}
PERCENTAGE_FEATURES = {
    feature for feature in extractor.FEATURE_COLUMNS if "percentage" in feature
} | {"valid_fhr_percentage", "missing_fhr_percentage", "artifact_percentage"}
BASELINE_FEATURES = {
    "baseline_fhr", "baseline_drift_std", "baseline_drift_range",
    "percentage_time_above_baseline", "percentage_time_below_baseline",
}

# Data-relative abnormality indicators.  ``high`` and ``low`` are directions
# used only to describe unusual local windows, not to create clinical labels.
ABNORMAL_DIRECTIONS = {
    "short_term_variability": "low",
    "long_term_variability": "low",
    "bradycardia_percentage": "high",
    "tachycardia_percentage": "high",
    "deceleration_density_per_hour": "high",
    "max_deceleration_depth": "high",
    "longest_deceleration_duration_seconds": "high",
    "baseline_drift_range": "high",
}


def baseline_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return the leakage-free model columns."""
    excluded = {train_models.TARGET_COLUMN, *train_models.LEAKAGE_COLUMNS}
    return [column for column in dataset.columns if column not in excluded]


def refuse_overwrite() -> None:
    """Keep this audit separate and prevent accidental replacement."""
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(
            f"{OUTPUT_ROOT} already contains files; move it before rerunning."
        )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    """Load and align all required read-only inputs and saved configuration."""
    required = [
        WINDOW_PATH, LABELS_PATH, CLINICAL_PATH, BASELINE_PATH,
        PATHOLOGICAL_WINDOW_PATH, FEATURE_QUALITY_PATH,
        REPRESENTATION_CHANGE_PATH, MODEL_PATH, ENCODER_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    windows = pd.read_csv(WINDOW_PATH)
    labels = pd.read_csv(LABELS_PATH)
    clinical = pd.read_csv(CLINICAL_PATH)
    baseline = pd.read_csv(BASELINE_PATH)
    for frame in [windows, labels, clinical, baseline]:
        frame["record_id"] = normalize_record_id(frame["record_id"])
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    with ENCODER_PATH.open("rb") as handle:
        encoder = pickle.load(handle)
    return windows, labels, clinical, baseline, model, encoder


def validate_windows(
    windows: pd.DataFrame,
    labels: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate patient/window integrity without assigning window targets."""
    required = {
        "record_id", "window_number", "start_minute", "end_minute",
        "window_duration_minutes", "is_final_window",
    }
    missing = sorted(required - set(windows.columns))
    if missing:
        raise ValueError(f"Window table lacks required fields: {missing}")
    forbidden = {"label", "pH", "BE", "BDecf", "Apgar5"}
    present_forbidden = sorted(forbidden & set(windows.columns))
    if present_forbidden:
        raise ValueError(
            "Window rows contain target/outcome fields: "
            + ", ".join(present_forbidden)
        )
    if windows[["record_id", "window_number"]].duplicated().any():
        raise ValueError("Duplicate record/window pairs found")
    if windows["record_id"].nunique() != 552:
        raise ValueError("Window table must represent all 552 patients")
    if set(windows["record_id"]) != set(labels["record_id"]):
        raise ValueError("Window record IDs do not match labels.csv")
    expected_labels = labels.set_index("record_id").loc[
        baseline["record_id"], "label"
    ].to_numpy()
    if not np.array_equal(expected_labels, baseline["label"].to_numpy()):
        raise ValueError("Baseline patient labels do not match labels.csv")
    ordered = windows.sort_values(["record_id", "window_number"]).copy()
    if (ordered["end_minute"] <= ordered["start_minute"]).any():
        raise ValueError("Invalid window timing found")
    if (ordered["window_duration_minutes"] < 10.0 - 1e-6).any():
        raise ValueError("A final diagnostic window shorter than 10 minutes exists")
    counts = ordered.groupby("record_id").size().rename("window_count")
    patient_classes = labels["label"].value_counts().rename("patient_count")
    joined = ordered[["record_id"]].merge(
        labels[["record_id", "label"]], on="record_id", validate="many_to_one"
    )
    window_classes = joined["label"].value_counts().rename("window_count")
    class_counts = pd.concat([patient_classes, window_classes], axis=1).fillna(0)
    distribution = counts.describe(percentiles=[0.25, 0.5, 0.75]).to_frame().T
    return distribution, class_counts


def actual_feature_groups(windows: pd.DataFrame) -> dict[str, list[str]]:
    """Resolve feature groups against available numeric columns."""
    groups = {}
    for name, feature_names in FEATURE_GROUP_REQUESTS.items():
        groups[name] = [
            feature
            for feature in feature_names
            if feature in windows.columns
            and pd.api.types.is_numeric_dtype(windows[feature])
        ]
    grouped = {feature for values in groups.values() for feature in values}
    current = {
        feature
        for feature in extractor.FEATURE_COLUMNS
        if feature in windows.columns
        and feature not in qa.SOURCE_QUALITY_FEATURES
        and pd.api.types.is_numeric_dtype(windows[feature])
    }
    # Keep ungrouped extractor columns under signal dynamics.
    remainder = sorted(current - grouped)
    if remainder:
        groups["Other signal dynamics"] = remainder
    return groups


def summary_statistics_for(feature: str) -> list[str]:
    """Return a semantically constrained summary map for one feature."""
    common_temporal = [
        "mean", "median", "min", "max", "range", "std", "iqr",
        "first", "final", "delta", "relative_delta", "slope", "spearman",
        "largest_increase", "largest_decrease",
    ]
    if feature in COUNT_FEATURES:
        return [
            "sum", "mean", "min", "max", "std", "first", "final", "delta",
            "slope", "largest_increase", "largest_decrease", "top_two_mean",
            "percent_high", "longest_extreme_run",
        ]
    if (
        feature in SEVERITY_FEATURES
        or feature in DURATION_FEATURES
        or feature in DENSITY_FEATURES
    ):
        return [
            "mean", "median", "max", "std", "first", "final", "delta",
            "slope", "top_two_mean", "percent_high", "longest_extreme_run",
        ]
    if feature in VARIABILITY_FEATURES:
        return [
            "mean", "median", "min", "max", "range", "std", "first", "final",
            "delta", "relative_delta", "slope", "spearman",
            "largest_increase", "largest_decrease", "bottom_two_mean",
            "percent_low", "longest_extreme_run",
        ]
    if feature in PERCENTAGE_FEATURES:
        return [
            "weighted_mean", "median", "min", "max", "range", "std", "first",
            "final", "delta", "relative_delta", "slope", "spearman",
            "largest_increase", "largest_decrease", "top_two_mean",
            "bottom_two_mean", "percent_high", "percent_low",
            "longest_extreme_run",
        ]
    if feature in BASELINE_FEATURES:
        return common_temporal + [
            "top_two_mean", "bottom_two_mean", "percent_high", "percent_low",
            "longest_extreme_run",
        ]
    return common_temporal + ["top_two_mean", "bottom_two_mean"]


def longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive True run."""
    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return int(best)


def calculate_summary(
    values: np.ndarray,
    times: np.ndarray,
    weights: np.ndarray,
    statistic: str,
    low_threshold: float,
    high_threshold: float,
) -> float:
    """Calculate one temporal statistic using finite windows only."""
    valid = np.isfinite(values) & np.isfinite(times)
    x = values[valid]
    t = times[valid]
    w = weights[valid]
    if not len(x):
        return np.nan
    if statistic == "sum":
        return float(np.sum(x))
    if statistic == "mean":
        return float(np.mean(x))
    if statistic == "weighted_mean":
        usable = np.isfinite(w) & (w > 0)
        return float(np.average(x[usable], weights=w[usable])) if usable.any() else float(np.mean(x))
    if statistic == "median":
        return float(np.median(x))
    if statistic == "min":
        return float(np.min(x))
    if statistic == "max":
        return float(np.max(x))
    if statistic == "range":
        return float(np.max(x) - np.min(x))
    if statistic == "std":
        return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    if statistic == "iqr":
        return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
    if statistic == "first":
        return float(x[0])
    if statistic == "final":
        return float(x[-1])
    if statistic == "top_two_mean":
        return float(np.mean(np.sort(x)[-min(2, len(x)):]))
    if statistic == "bottom_two_mean":
        return float(np.mean(np.sort(x)[:min(2, len(x))]))
    if statistic in {
        "delta", "relative_delta", "slope", "spearman",
        "largest_increase", "largest_decrease",
    } and len(x) < 2:
        return np.nan
    if statistic == "delta":
        return float(x[-1] - x[0])
    if statistic == "relative_delta":
        scale = abs(x[0])
        return float((x[-1] - x[0]) / scale) if scale > 1e-9 else np.nan
    if statistic == "slope":
        return float(np.polyfit(t, x, 1)[0]) if np.ptp(t) > 0 else np.nan
    if statistic == "spearman":
        if np.std(x) == 0:
            return np.nan
        value = spearmanr(t, x).statistic
        return float(value) if np.isfinite(value) else np.nan
    differences = np.diff(x)
    if statistic == "largest_increase":
        return float(np.max(differences))
    if statistic == "largest_decrease":
        return float(np.min(differences))
    if statistic == "percent_high":
        return float(100.0 * np.mean(x >= high_threshold))
    if statistic == "percent_low":
        return float(100.0 * np.mean(x <= low_threshold))
    if statistic == "longest_extreme_run":
        return float(
            max(
                longest_true_run(x >= high_threshold),
                longest_true_run(x <= low_threshold),
            )
        )
    raise ValueError(f"Unsupported statistic: {statistic}")


def build_temporal_summaries(
    windows: pd.DataFrame,
    labels: pd.DataFrame,
    groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, tuple[float, float]]]:
    """Create chronological one-row-per-patient temporal summaries."""
    features = []
    feature_to_group = {}
    for group_name, group_features in groups.items():
        for feature in group_features:
            if feature not in features:
                features.append(feature)
                feature_to_group[feature] = group_name
    thresholds = {}
    for feature in features:
        finite = pd.to_numeric(windows[feature], errors="coerce").dropna()
        thresholds[feature] = (
            float(finite.quantile(LOW_QUANTILE)),
            float(finite.quantile(HIGH_QUANTILE)),
        )
    summary_map = {
        feature: summary_statistics_for(feature) for feature in features
    }
    rows = []
    for record_number, (record_id, group) in enumerate(
        windows.groupby("record_id", sort=True), start=1
    ):
        group = group.sort_values("window_number")
        row = {
            "record_id": record_id,
            "total_window_count": len(group),
            "valid_window_count": int(
                (group["valid_fhr_percentage"] >= MIN_VALID_FHR_PERCENTAGE).sum()
            ),
            "poor_quality_window_percentage": float(
                100.0
                * (group["valid_fhr_percentage"] < MIN_VALID_FHR_PERCENTAGE).mean()
            ),
            "final_window_quality": float(group.iloc[-1]["valid_fhr_percentage"]),
            "quality_transition_count": int(
                np.sum(
                    np.diff(
                        (
                            group["valid_fhr_percentage"].to_numpy()
                            >= MIN_VALID_FHR_PERCENTAGE
                        ).astype(int)
                    )
                    != 0
                )
            ),
        }
        times = (
            group["start_minute"].to_numpy(float)
            + group["end_minute"].to_numpy(float)
        ) / 2.0
        weights = group["valid_fhr_sample_count"].to_numpy(float)
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce").to_numpy(float)
            low, high = thresholds[feature]
            for statistic in summary_map[feature]:
                row[f"{feature}__window_{statistic}"] = calculate_summary(
                    values, times, weights, statistic, low, high
                )
        rows.append(row)
        if record_number % 100 == 0 or record_number == 552:
            print(f"  Temporal summaries: {record_number}/552 patients")
    result = pd.DataFrame(rows).merge(
        labels[["record_id", "label"]],
        on="record_id",
        how="left",
        validate="one_to_one",
    )
    if len(result) != 552 or result["record_id"].duplicated().any():
        raise AssertionError("Temporal summaries must contain 552 unique patients")
    return result, summary_map, thresholds


def finite_pair(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return paired finite arrays."""
    a = pd.to_numeric(left, errors="coerce").to_numpy(float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def closest_candidates(feature: str, summary_map: dict[str, list[str]]) -> list[str]:
    """Return semantically comparable temporal variants for a whole feature."""
    available = summary_map.get(feature, [])
    preferred = []
    if feature in COUNT_FEATURES:
        preferred = ["sum", "mean", "max", "min", "delta", "slope"]
    elif feature in SEVERITY_FEATURES or feature in DURATION_FEATURES:
        preferred = ["max", "top_two_mean", "final", "mean", "delta", "slope"]
    elif feature in VARIABILITY_FEATURES:
        preferred = [
            "mean", "min", "final", "range", "std", "delta", "slope",
            "largest_decrease",
        ]
    elif feature in PERCENTAGE_FEATURES:
        preferred = [
            "weighted_mean", "max", "final", "range", "delta", "slope",
            "percent_high",
        ]
    elif feature == "baseline_fhr":
        preferred = [
            "weighted_mean", "min", "max", "range", "final", "delta", "slope"
        ]
    else:
        preferred = ["mean", "min", "max", "range", "final", "delta", "slope"]
    return [
        f"{feature}__window_{statistic}"
        for statistic in preferred
        if statistic in available
    ]


def whole_record_comparison(
    baseline: pd.DataFrame,
    summaries: pd.DataFrame,
    summary_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Compare recording-level values with patient temporal summaries."""
    merged = baseline.merge(
        summaries, on=["record_id", "label"], validate="one_to_one"
    )
    rows = []
    for original in baseline_feature_columns(baseline):
        if original not in summary_map:
            continue
        for candidate in closest_candidates(original, summary_map):
            if candidate not in merged:
                continue
            old, new = finite_pair(merged[original], merged[candidate])
            correlation = (
                float(np.corrcoef(old, new)[0, 1])
                if len(old) > 2 and np.std(old) > 0 and np.std(new) > 0
                else np.nan
            )
            comparable = candidate.endswith(
                ("_mean", "_weighted_mean", "_min", "_max", "_final", "_first", "_sum")
            )
            scale = max(
                float(np.nanmedian(np.abs(old))) if len(old) else 0.0,
                float(np.nanstd(old)) if len(old) else 0.0,
                MIN_ABSOLUTE_SUBSTANTIAL_CHANGE,
            )
            differences = np.abs(new - old) if comparable else np.full(len(old), np.nan)
            row = {
                "feature_concept": original,
                "whole_record_feature": original,
                "temporal_candidate": candidate,
                "units_comparable": comparable,
                "paired_patient_count": len(old),
                "correlation": correlation,
                "mean_absolute_difference": float(np.nanmean(differences))
                if comparable and len(differences)
                else np.nan,
                "substantial_change_threshold": scale * SUBSTANTIAL_CHANGE_FRACTION
                if comparable else np.nan,
                "substantially_more_extreme_patient_count": int(
                    np.sum(np.abs(new) > np.abs(old) + scale * SUBSTANTIAL_CHANGE_FRACTION)
                ) if comparable else np.nan,
            }
            for label in ["Normal", "Suspicious", "Pathological"]:
                subset = merged["label"] == label
                a, b = finite_pair(merged.loc[subset, original], merged.loc[subset, candidate])
                row[f"{label.lower()}_candidate_mean"] = float(np.mean(b)) if len(b) else np.nan
                row[f"{label.lower()}_mean_absolute_difference"] = (
                    float(np.mean(np.abs(b - a))) if comparable and len(a) else np.nan
                )
            path = merged["label"] == "Pathological"
            fixed = merged["record_id"].isin(FIXED_TEST_PATHOLOGICAL_IDS)
            fixed_old, fixed_new = finite_pair(
                merged.loc[fixed, original], merged.loc[fixed, candidate]
            )
            row["pathological_patient_count"] = int(
                merged.loc[path, [original, candidate]].dropna().shape[0]
            )
            row["fixed_test_pathological_whole_record_mean"] = (
                float(np.mean(fixed_old)) if len(fixed_old) else np.nan
            )
            row["fixed_test_pathological_candidate_mean"] = float(
                pd.to_numeric(merged.loc[fixed, candidate], errors="coerce").mean()
            )
            row["fixed_test_pathological_mean_absolute_difference"] = (
                float(np.mean(np.abs(fixed_new - fixed_old)))
                if comparable and len(fixed_old) else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def safe_mutual_information(values: np.ndarray, labels: np.ndarray) -> float:
    """Calculate reproducible one-feature MI after median imputation."""
    finite = np.isfinite(values)
    if finite.sum() < 10 or np.unique(values[finite]).size < 2:
        return np.nan
    filled = values.copy()
    filled[~finite] = np.median(values[finite])
    return float(
        mutual_info_classif(
            filled.reshape(-1, 1),
            labels,
            random_state=42,
            discrete_features=False,
        )[0]
    )


def discrimination_row(
    feature: str,
    values: pd.Series,
    labels: pd.Series,
    source: str,
    concept: str,
) -> dict:
    """Calculate exploratory univariate discrimination statistics."""
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    label_array = labels.to_numpy()
    valid = np.isfinite(x)
    xv = x[valid]
    lv = label_array[valid]
    binary = (lv == "Pathological").astype(int)
    row = {
        "feature": feature,
        "feature_concept": concept,
        "source": source,
        "valid_patient_count": int(valid.sum()),
        "missing_percentage": float(100.0 * (~valid).mean()),
    }
    for label in ["Normal", "Suspicious", "Pathological"]:
        group = xv[lv == label]
        row[f"{label.lower()}_mean"] = float(np.mean(group)) if len(group) else np.nan
        row[f"{label.lower()}_median"] = float(np.median(group)) if len(group) else np.nan
        row[f"{label.lower()}_iqr"] = (
            float(np.quantile(group, 0.75) - np.quantile(group, 0.25))
            if len(group) else np.nan
        )
    pathological = xv[binary == 1]
    nonpathological = xv[binary == 0]
    pooled = np.sqrt(
        (
            (len(pathological) - 1) * np.var(pathological, ddof=1)
            + (len(nonpathological) - 1) * np.var(nonpathological, ddof=1)
        )
        / max(len(pathological) + len(nonpathological) - 2, 1)
    ) if len(pathological) > 1 and len(nonpathological) > 1 else np.nan
    row["pathological_standardized_effect_size"] = (
        float((np.mean(pathological) - np.mean(nonpathological)) / pooled)
        if np.isfinite(pooled) and pooled > 0 else np.nan
    )
    try:
        raw_auc = roc_auc_score(binary, xv)
        direction = 1.0 if raw_auc >= 0.5 else -1.0
        row["pathological_roc_auc_raw"] = float(raw_auc)
        row["pathological_roc_auc_discrimination"] = float(max(raw_auc, 1 - raw_auc))
        row["pathological_average_precision_oriented"] = float(
            average_precision_score(binary, direction * xv)
        )
        row["higher_values_associated_with_pathological"] = bool(direction > 0)
    except ValueError:
        row["pathological_roc_auc_raw"] = np.nan
        row["pathological_roc_auc_discrimination"] = np.nan
        row["pathological_average_precision_oriented"] = np.nan
        row["higher_values_associated_with_pathological"] = np.nan
    encoded = pd.Categorical(lv, categories=["Normal", "Suspicious", "Pathological"]).codes
    row["mutual_information_three_class"] = safe_mutual_information(xv, encoded)
    groups = [xv[lv == label] for label in ["Normal", "Suspicious", "Pathological"]]
    try:
        row["kruskal_wallis_statistic"] = float(kruskal(*groups).statistic)
        row["kruskal_wallis_p_value_exploratory"] = float(kruskal(*groups).pvalue)
    except ValueError:
        row["kruskal_wallis_statistic"] = np.nan
        row["kruskal_wallis_p_value_exploratory"] = np.nan
    return row


def univariate_discrimination(
    baseline: pd.DataFrame,
    summaries: pd.DataFrame,
    summary_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Rank whole-record and temporal patient features descriptively."""
    rows = []
    for feature in baseline_feature_columns(baseline):
        rows.append(
            discrimination_row(
                feature, baseline[feature], baseline["label"],
                "whole_record", feature,
            )
        )
    for concept, statistics in summary_map.items():
        for statistic in statistics:
            feature = f"{concept}__window_{statistic}"
            rows.append(
                discrimination_row(
                    feature, summaries[feature], summaries["label"],
                    "temporal_summary", concept,
                )
            )
    for feature in [
        "total_window_count", "valid_window_count",
        "poor_quality_window_percentage", "final_window_quality",
        "quality_transition_count",
    ]:
        rows.append(
            discrimination_row(
                feature, summaries[feature], summaries["label"],
                "temporal_quality_summary", "signal_quality",
            )
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["pathological_roc_auc_discrimination", "mutual_information_three_class"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def residual_information(
    original: pd.Series,
    candidate: pd.Series,
    labels: pd.Series,
) -> tuple[float, float]:
    """Estimate candidate information left after a linear original adjustment."""
    a = pd.to_numeric(original, errors="coerce").to_numpy(float)
    b = pd.to_numeric(candidate, errors="coerce").to_numpy(float)
    y = (labels.to_numpy() == "Pathological").astype(int)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 20 or np.std(a[valid]) == 0:
        return np.nan, np.nan
    fit = np.polyfit(a[valid], b[valid], 1)
    residual = b[valid] - np.polyval(fit, a[valid])
    effect = (
        (np.mean(residual[y[valid] == 1]) - np.mean(residual[y[valid] == 0]))
        / np.std(residual, ddof=1)
        if np.std(residual, ddof=1) > 0 else np.nan
    )
    mi = safe_mutual_information(residual, y[valid])
    return float(effect), float(mi)


def paired_feature_ranking(
    baseline: pd.DataFrame,
    summaries: pd.DataFrame,
    discrimination: pd.DataFrame,
    summary_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Compare the strongest nontrivial temporal candidate per concept."""
    merged = baseline.merge(
        summaries, on=["record_id", "label"], validate="one_to_one"
    )
    lookup = discrimination.set_index("feature")
    rows = []
    for original in baseline_feature_columns(baseline):
        candidates = [
            value for value in closest_candidates(original, summary_map)
            if value in lookup.index
        ]
        if not candidates or original not in lookup.index:
            continue
        ranked = sorted(
            candidates,
            key=lambda name: (
                lookup.loc[name, "pathological_roc_auc_discrimination"]
                if np.isfinite(lookup.loc[name, "pathological_roc_auc_discrimination"])
                else -np.inf
            ),
            reverse=True,
        )
        candidate = ranked[0]
        old, new = finite_pair(merged[original], merged[candidate])
        correlation = (
            float(np.corrcoef(old, new)[0, 1])
            if len(old) > 2 and np.std(old) and np.std(new) else np.nan
        )
        residual_effect, residual_mi = residual_information(
            merged[original], merged[candidate], merged["label"]
        )
        old_row = lookup.loc[original]
        new_row = lookup.loc[candidate]
        auc_change = (
            new_row["pathological_roc_auc_discrimination"]
            - old_row["pathological_roc_auc_discrimination"]
        )
        ap_change = (
            new_row["pathological_average_precision_oriented"]
            - old_row["pathological_average_precision_oriented"]
        )
        effect_change = (
            abs(new_row["pathological_standardized_effect_size"])
            - abs(old_row["pathological_standardized_effect_size"])
        )
        adds = bool(
            (not np.isfinite(correlation) or abs(correlation) < NEAR_DUPLICATE_CORRELATION)
            and (
                auc_change >= 0.02
                or ap_change >= 0.02
                or (np.isfinite(residual_mi) and residual_mi >= 0.005)
            )
        )
        rows.append(
            {
                "feature_concept": original,
                "whole_record_feature": original,
                "best_temporal_candidate": candidate,
                "whole_record_pathological_auc": old_row["pathological_roc_auc_discrimination"],
                "temporal_pathological_auc": new_row["pathological_roc_auc_discrimination"],
                "pathological_auc_difference": auc_change,
                "whole_record_average_precision": old_row["pathological_average_precision_oriented"],
                "temporal_average_precision": new_row["pathological_average_precision_oriented"],
                "average_precision_difference": ap_change,
                "absolute_effect_size_difference": effect_change,
                "missingness_difference_percentage_points": (
                    new_row["missing_percentage"] - old_row["missing_percentage"]
                ),
                "original_candidate_correlation": correlation,
                "residual_pathological_effect_size": residual_effect,
                "residual_mutual_information": residual_mi,
                "adds_information_beyond_whole_record": adds,
                "exploratory_full_dataset_ranking": True,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["adds_information_beyond_whole_record", "pathological_auc_difference"],
        ascending=False,
    )


def add_window_abnormality(
    windows: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Add data-relative abnormality indicators."""
    result = windows.copy()
    indicators = []
    for feature, direction in ABNORMAL_DIRECTIONS.items():
        if feature not in result:
            continue
        low, high = thresholds[feature]
        indicator = (
            result[feature] <= low if direction == "low" else result[feature] >= high
        )
        name = f"_extreme_{feature}"
        result[name] = indicator.fillna(False).astype(int)
        indicators.append(name)
    result["_extreme_indicator_count"] = result[indicators].sum(axis=1)
    result["_is_extreme_window"] = result["_extreme_indicator_count"] >= 2
    result["_poor_signal"] = result["valid_fhr_percentage"] < MIN_VALID_FHR_PERCENTAGE
    return result


def classify_pattern(group: pd.DataFrame) -> tuple[str, dict]:
    """Classify an exploratory temporal pattern using explicit quantile rules."""
    group = group.sort_values("window_number")
    extreme = group["_is_extreme_window"].to_numpy(bool)
    poor = group["_poor_signal"].to_numpy(bool)
    scores = group["_extreme_indicator_count"].to_numpy(float)
    extreme_count = int(extreme.sum())
    extreme_fraction = extreme.mean()
    poor_fraction = poor.mean()
    run = longest_true_run(extreme)
    trend = (
        float(spearmanr(group["window_number"], scores).statistic)
        if len(group) >= PATTERN_MIN_TREND_WINDOWS and np.std(scores) > 0
        else np.nan
    )
    transitions = int(np.sum(np.diff(extreme.astype(int)) != 0))
    final_strongest = bool(scores[-1] == np.max(scores) and np.max(scores) > 0)
    if poor_fraction >= PATTERN_POOR_QUALITY_FRACTION:
        category = "dominated by poor signal"
    elif extreme_fraction >= PATTERN_CONSISTENT_ABNORMAL_FRACTION:
        category = "consistently abnormal"
    elif np.isfinite(trend) and trend >= PATTERN_TREND_SPEARMAN:
        category = "progressive deterioration"
    elif np.isfinite(trend) and trend <= -PATTERN_TREND_SPEARMAN:
        category = "progressive improvement"
    elif extreme_count == 1:
        category = "one isolated extreme window"
    elif extreme_count >= 2 and run >= 2:
        category = "several repeated abnormal windows"
    elif extreme_count >= 2 or transitions >= 2:
        category = "unstable or alternating"
    else:
        category = "consistently normal or mild"
    return category, {
        "extreme_window_count": extreme_count,
        "extreme_window_percentage": float(100.0 * extreme_fraction),
        "longest_consecutive_extreme_run": run,
        "poor_signal_window_percentage": float(100.0 * poor_fraction),
        "abnormality_score_spearman_trend": trend,
        "extreme_state_transition_count": transitions,
        "strongest_abnormality_in_final_window": int(final_strongest),
    }


def temporal_patterns(
    windows: pd.DataFrame,
    labels: pd.DataFrame,
    thresholds: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one temporal-pattern category per patient."""
    scored = add_window_abnormality(windows, thresholds)
    rows = []
    for record_id, group in scored.groupby("record_id"):
        category, details = classify_pattern(group)
        rows.append(
            {"record_id": record_id, "temporal_pattern_category": category, **details}
        )
    result = pd.DataFrame(rows).merge(
        labels[["record_id", "label"]], on="record_id", validate="one_to_one"
    )
    class_summary = (
        result.groupby(["label", "temporal_pattern_category"])
        .size()
        .rename("patient_count")
        .reset_index()
    )
    class_totals = result["label"].value_counts()
    class_summary["class_percentage"] = class_summary.apply(
        lambda row: 100.0 * row["patient_count"] / class_totals[row["label"]],
        axis=1,
    )
    summary_lookup = {
        (row["label"], row["temporal_pattern_category"]): (
            row["patient_count"], row["class_percentage"]
        )
        for _, row in class_summary.iterrows()
    }
    result["category_class_patient_count"] = result.apply(
        lambda row: summary_lookup[(row["label"], row["temporal_pattern_category"])][0],
        axis=1,
    )
    result["category_class_percentage"] = result.apply(
        lambda row: summary_lookup[(row["label"], row["temporal_pattern_category"])][1],
        axis=1,
    )
    return result, scored


def strongest_abnormality(group: pd.DataFrame) -> tuple[str, int, pd.Series]:
    """Return strongest extreme feature, window number, and row."""
    group = group.sort_values("window_number")
    best = group.loc[group["_extreme_indicator_count"].idxmax()]
    active = []
    for feature in ABNORMAL_DIRECTIONS:
        if best.get(f"_extreme_{feature}", 0):
            active.append(feature)
    text = ", ".join(active) if active else "no quantile-defined local extreme"
    return text, int(best["window_number"]), best


def pathological_case_audit(
    summaries: pd.DataFrame,
    patterns: pd.DataFrame,
    scored_windows: pd.DataFrame,
    baseline: pd.DataFrame,
    target: np.ndarray,
) -> pd.DataFrame:
    """Create the 27-record Pathological temporal audit."""
    _, fixed_test = train_test_split(
        np.arange(len(baseline)),
        test_size=train_models.TEST_SIZE,
        random_state=train_models.RANDOM_STATE,
        stratify=target,
    )
    fixed_test_ids = set(baseline.iloc[fixed_test]["record_id"])
    merged = summaries.merge(
        patterns.drop(columns="label"), on="record_id", validate="one_to_one"
    )
    merged = merged.merge(
        baseline[["record_id", "label", "short_term_variability",
                  "bradycardia_percentage", "max_deceleration_depth"]],
        on=["record_id", "label"],
        validate="one_to_one",
        suffixes=("", "__whole"),
    )
    rows = []
    for _, patient in merged[merged["label"] == "Pathological"].iterrows():
        record_id = patient["record_id"]
        group = scored_windows[scored_windows["record_id"] == record_id]
        strongest, window_number, best = strongest_abnormality(group)
        local_candidates = [
            patient.get("short_term_variability__window_min", np.nan),
            patient.get("bradycardia_percentage__window_max", np.nan),
            patient.get("max_deceleration_depth__window_max", np.nan),
        ]
        whole_candidates = [
            patient.get("short_term_variability", np.nan),
            patient.get("bradycardia_percentage", np.nan),
            patient.get("max_deceleration_depth", np.nan),
        ]
        hides = bool(
            patient["extreme_window_count"] > 0
            and any(
                np.isfinite(local) and np.isfinite(whole)
                and abs(local - whole) > 0.25 * max(abs(whole), 1e-9)
                for local, whole in zip(local_candidates, whole_candidates)
            )
        )
        rows.append(
            {
                "record_id": record_id,
                "fixed_train_test_membership": (
                    "fixed_test" if record_id in fixed_test_ids else "fixed_train"
                ),
                "number_of_windows": int(patient["total_window_count"]),
                "number_of_valid_windows": int(patient["valid_window_count"]),
                "strongest_local_abnormality": strongest,
                "strongest_abnormality_window_number": window_number,
                "minimum_stv": patient.get("short_term_variability__window_min"),
                "final_stv": patient.get("short_term_variability__window_final"),
                "stv_slope_per_minute": patient.get("short_term_variability__window_slope"),
                "largest_stv_decrease": patient.get(
                    "short_term_variability__window_largest_decrease"
                ),
                "maximum_bradycardia_percentage": patient.get(
                    "bradycardia_percentage__window_max"
                ),
                "maximum_tachycardia_percentage": patient.get(
                    "tachycardia_percentage__window_max"
                ),
                "maximum_deceleration_depth": patient.get(
                    "max_deceleration_depth__window_max"
                ),
                "longest_deceleration_duration": patient.get(
                    "longest_deceleration_duration_seconds__window_max"
                ),
                "maximum_deceleration_density": patient.get(
                    "deceleration_density_per_hour__window_max"
                ),
                "baseline_range": patient.get("baseline_fhr__window_range"),
                "baseline_slope_per_minute": patient.get(
                    "baseline_fhr__window_slope"
                ),
                "number_of_extreme_windows": int(patient["extreme_window_count"]),
                "longest_consecutive_extreme_run": int(
                    patient["longest_consecutive_extreme_run"]
                ),
                "temporal_pattern_category": patient["temporal_pattern_category"],
                "whole_record_averaging_appears_to_hide_local_extreme": hides,
                "extreme_window_valid_fhr_percentage": best["valid_fhr_percentage"],
                "extreme_window_missing_fhr_percentage": best["missing_fhr_percentage"],
                "extreme_window_artifact_percentage": best["artifact_percentage"],
                "extreme_window_accepted_by_quality_rule": bool(
                    best["accepted_by_quality_rule"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fixed_train_test_membership", "record_id"]
    )


def select_candidate_shortlist(
    paired: pd.DataFrame,
    discrimination: pd.DataFrame,
    baseline: pd.DataFrame,
    summaries: pd.DataFrame,
) -> list[str]:
    """Select at most 20 understandable, nonredundant temporal candidates."""
    temporal = discrimination[
        discrimination["source"].isin(["temporal_summary", "temporal_quality_summary"])
    ].copy()
    paired_candidates = set(paired["best_temporal_candidate"])
    temporal["paired_concept_candidate"] = temporal["feature"].isin(paired_candidates)
    temporal["screen_score"] = (
        temporal["pathological_roc_auc_discrimination"].fillna(0.5) - 0.5
        + 0.5 * temporal["pathological_average_precision_oriented"].fillna(0)
        + 0.1 * temporal["mutual_information_three_class"].fillna(0)
        - 0.002 * temporal["missing_percentage"]
    )
    temporal = temporal.sort_values(
        ["paired_concept_candidate", "screen_score"], ascending=False
    )
    selected = []
    selected_concepts = defaultdict(int)
    aligned = summaries.set_index("record_id").loc[baseline["record_id"]]
    for _, row in temporal.iterrows():
        feature = row["feature"]
        concept = row["feature_concept"]
        # Dataset-quantile summaries are descriptive only.  Excluding them
        # from modelling prevents test-fold distribution information entering
        # a candidate before the fold-local pipeline is fit.
        if feature.endswith(
            ("__window_percent_high", "__window_percent_low",
             "__window_longest_extreme_run")
        ):
            continue
        if row["missing_percentage"] > 40:
            continue
        if concept in baseline.columns:
            old, new = finite_pair(baseline[concept], aligned[feature])
            correlation = (
                float(np.corrcoef(old, new)[0, 1])
                if len(old) > 2 and np.std(old) > 0 and np.std(new) > 0
                else np.nan
            )
            if np.isfinite(correlation) and abs(correlation) >= NEAR_DUPLICATE_CORRELATION:
                continue
        if selected_concepts[concept] >= 2:
            continue
        selected.append(feature)
        selected_concepts[concept] += 1
        if len(selected) == MAX_SINGLE_FEATURE_CANDIDATES:
            break
    return selected


def model_metrics(y_true, predictions, encoder) -> dict:
    """Return screening metrics, including false Pathological predictions."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        zero_division=0,
    )
    pathological = int(encoder.transform(["Pathological"])[0])
    return {
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "pathological_recall": float(recall[pathological]),
        "pathological_precision": float(precision[pathological]),
        "pathological_f1": float(f1[pathological]),
        "false_pathological_predictions": int(
            np.sum((predictions == pathological) & (y_true != pathological))
        ),
    }


def repeated_cv_screen(
    baseline: pd.DataFrame,
    summaries: pd.DataFrame,
    target: np.ndarray,
    encoder,
    model,
    candidates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baseline and each one-feature addition on identical folds."""
    base_columns = baseline_feature_columns(baseline)
    base_matrix = baseline[base_columns]
    summary_index = summaries.set_index("record_id")
    additions = {
        candidate: summary_index.loc[baseline["record_id"], candidate].to_numpy()
        for candidate in candidates
    }
    cv = RepeatedStratifiedKFold(
        n_splits=prior_diagnostic.CV_SPLITS,
        n_repeats=prior_diagnostic.CV_REPEATS,
        random_state=train_models.RANDOM_STATE,
    )
    fold_rows = []
    metrics = [
        "macro_f1", "balanced_accuracy", "pathological_recall",
        "pathological_precision", "pathological_f1",
        "false_pathological_predictions",
    ]
    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(base_matrix, target), start=1
    ):
        seed = train_models.RANDOM_STATE + fold_number
        baseline_pipeline = qa.build_comparison_pipeline(model, seed)
        baseline_pipeline.fit(base_matrix.iloc[train_idx], target[train_idx])
        baseline_prediction = baseline_pipeline.predict(base_matrix.iloc[test_idx])
        baseline_metrics = model_metrics(
            target[test_idx], baseline_prediction, encoder
        )
        for candidate, values in additions.items():
            augmented = base_matrix.copy()
            augmented[candidate] = values
            pipeline = qa.build_comparison_pipeline(model, seed)
            pipeline.fit(augmented.iloc[train_idx], target[train_idx])
            predictions = pipeline.predict(augmented.iloc[test_idx])
            candidate_metrics = model_metrics(target[test_idx], predictions, encoder)
            row = {
                "candidate": candidate,
                "fold_number": fold_number,
                "repeat_number": (fold_number - 1) // prior_diagnostic.CV_SPLITS + 1,
                "fold_within_repeat": (fold_number - 1) % prior_diagnostic.CV_SPLITS + 1,
            }
            for metric in metrics:
                row[f"baseline_{metric}"] = baseline_metrics[metric]
                row[f"candidate_{metric}"] = candidate_metrics[metric]
                row[f"delta_{metric}"] = (
                    candidate_metrics[metric] - baseline_metrics[metric]
                )
            fold_rows.append(row)
        print(
            f"  Single-feature screen fold: {fold_number}/"
            f"{prior_diagnostic.CV_SPLITS * prior_diagnostic.CV_REPEATS}"
        )
    folds = pd.DataFrame(fold_rows)
    rows = []
    for candidate, group in folds.groupby("candidate"):
        row = {"candidate": candidate, "fold_count": len(group)}
        for metric in metrics:
            row[f"baseline_{metric}_mean"] = group[f"baseline_{metric}"].mean()
            row[f"baseline_{metric}_std"] = group[f"baseline_{metric}"].std(ddof=1)
            row[f"candidate_{metric}_mean"] = group[f"candidate_{metric}"].mean()
            row[f"candidate_{metric}_std"] = group[f"candidate_{metric}"].std(ddof=1)
            row[f"paired_delta_{metric}_mean"] = group[f"delta_{metric}"].mean()
            row[f"paired_delta_{metric}_std"] = group[f"delta_{metric}"].std(ddof=1)
        delta = group["delta_macro_f1"]
        row["macro_f1_improved_fold_count"] = int((delta > 1e-12).sum())
        row["macro_f1_worsened_fold_count"] = int((delta < -1e-12).sum())
        row["macro_f1_equal_fold_count"] = int((abs(delta) <= 1e-12).sum())
        recall_delta = group["delta_pathological_recall"]
        row["pathological_recall_improved_fold_count"] = int(
            (recall_delta > 1e-12).sum()
        )
        row["pathological_recall_worsened_fold_count"] = int(
            (recall_delta < -1e-12).sum()
        )
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["paired_delta_macro_f1_mean", "paired_delta_pathological_recall_mean"],
        ascending=False,
    )
    return summary, folds


def predefined_groups(summaries: pd.DataFrame) -> dict[str, list[str]]:
    """Resolve a small number of predefined, interpretable temporal groups."""
    feature_sets = {
        "variability_deterioration": [
            "short_term_variability__window_min",
            "short_term_variability__window_final",
            "short_term_variability__window_slope",
            "short_term_variability__window_largest_decrease",
            "long_term_variability__window_min",
            "long_term_variability__window_slope",
        ],
        "local_deceleration_burden": [
            "deceleration_density_per_hour__window_max",
            "deceleration_density_per_hour__window_top_two_mean",
            "max_deceleration_depth__window_max",
            "longest_deceleration_duration_seconds__window_max",
            "deceleration_density_per_hour__window_longest_extreme_run",
        ],
        "bradycardia_tachycardia_burden": [
            "bradycardia_percentage__window_max",
            "tachycardia_percentage__window_max",
            "bradycardia_percentage__window_final",
            "tachycardia_percentage__window_final",
        ],
        "baseline_dynamics": [
            "baseline_fhr__window_range", "baseline_fhr__window_slope",
            "baseline_fhr__window_delta", "baseline_drift_range__window_max",
        ],
        "temporal_quality_context": [
            "valid_fhr_percentage__window_min",
            "poor_quality_window_percentage", "final_window_quality",
            "quality_transition_count",
        ],
    }
    return {
        name: [feature for feature in features if feature in summaries.columns]
        for name, features in feature_sets.items()
    }


def group_cv_screen(
    baseline: pd.DataFrame,
    summaries: pd.DataFrame,
    target: np.ndarray,
    encoder,
    model,
    groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Evaluate each predefined temporal group on identical repeated-CV folds."""
    base = baseline[baseline_feature_columns(baseline)]
    indexed = summaries.set_index("record_id").loc[baseline["record_id"]]
    cv = RepeatedStratifiedKFold(
        n_splits=prior_diagnostic.CV_SPLITS,
        n_repeats=prior_diagnostic.CV_REPEATS,
        random_state=train_models.RANDOM_STATE,
    )
    metrics = [
        "macro_f1", "balanced_accuracy", "pathological_recall",
        "pathological_precision", "pathological_f1",
        "false_pathological_predictions",
    ]
    fold_rows = []
    for fold_number, (train_idx, test_idx) in enumerate(cv.split(base, target), start=1):
        seed = train_models.RANDOM_STATE + fold_number
        baseline_pipeline = qa.build_comparison_pipeline(model, seed)
        baseline_pipeline.fit(base.iloc[train_idx], target[train_idx])
        baseline_metrics = model_metrics(
            target[test_idx], baseline_pipeline.predict(base.iloc[test_idx]), encoder
        )
        for group_name, features in groups.items():
            augmented = pd.concat(
                [base.reset_index(drop=True), indexed[features].reset_index(drop=True)],
                axis=1,
            )
            pipeline = qa.build_comparison_pipeline(model, seed)
            pipeline.fit(augmented.iloc[train_idx], target[train_idx])
            candidate_metrics = model_metrics(
                target[test_idx], pipeline.predict(augmented.iloc[test_idx]), encoder
            )
            row = {
                "temporal_group": group_name,
                "features": " | ".join(features),
                "feature_count": len(features),
                "fold_number": fold_number,
            }
            for metric in metrics:
                row[f"baseline_{metric}"] = baseline_metrics[metric]
                row[f"group_{metric}"] = candidate_metrics[metric]
                row[f"delta_{metric}"] = candidate_metrics[metric] - baseline_metrics[metric]
            fold_rows.append(row)
        print(
            f"  Group screen fold: {fold_number}/"
            f"{prior_diagnostic.CV_SPLITS * prior_diagnostic.CV_REPEATS}"
        )
    folds = pd.DataFrame(fold_rows)
    rows = []
    for (name, features), group in folds.groupby(["temporal_group", "features"]):
        row = {
            "temporal_group": name,
            "features": features,
            "feature_count": int(group["feature_count"].iloc[0]),
            "fold_count": len(group),
        }
        for metric in metrics:
            row[f"baseline_{metric}_mean"] = group[f"baseline_{metric}"].mean()
            row[f"baseline_{metric}_std"] = group[f"baseline_{metric}"].std(ddof=1)
            row[f"group_{metric}_mean"] = group[f"group_{metric}"].mean()
            row[f"group_{metric}_std"] = group[f"group_{metric}"].std(ddof=1)
            row[f"paired_delta_{metric}_mean"] = group[f"delta_{metric}"].mean()
            row[f"paired_delta_{metric}_std"] = group[f"delta_{metric}"].std(ddof=1)
        delta = group["delta_macro_f1"]
        row["macro_f1_improved_fold_count"] = int((delta > 1e-12).sum())
        row["macro_f1_worsened_fold_count"] = int((delta < -1e-12).sum())
        row["macro_f1_equal_fold_count"] = int((abs(delta) <= 1e-12).sum())
        recall = group["delta_pathological_recall"]
        row["pathological_recall_improved_fold_count"] = int((recall > 1e-12).sum())
        row["pathological_recall_worsened_fold_count"] = int((recall < -1e-12).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["paired_delta_macro_f1_mean", "paired_delta_pathological_recall_mean"],
        ascending=False,
    )


def choose_conclusion(single: pd.DataFrame, groups: pd.DataFrame) -> str:
    """Choose exactly one permitted conclusion from paired-CV evidence."""
    best_single = single.iloc[0]
    best_group = groups.iloc[0]
    best_macro = max(
        best_single["paired_delta_macro_f1_mean"],
        best_group["paired_delta_macro_f1_mean"],
    )
    recall_candidates = pd.concat(
        [
            single[["paired_delta_macro_f1_mean", "paired_delta_pathological_recall_mean"]],
            groups[["paired_delta_macro_f1_mean", "paired_delta_pathological_recall_mean"]],
        ]
    )
    useful_tradeoff = recall_candidates[
        (recall_candidates["paired_delta_pathological_recall_mean"] >= 0.04)
        & (recall_candidates["paired_delta_macro_f1_mean"] >= -0.015)
    ]
    if best_macro >= 0.02:
        return "temporal summaries provide clear additional predictive information"
    if not useful_tradeoff.empty:
        return "a small temporal feature set provides a useful recall trade-off"
    if best_macro <= 0.005 and (
        recall_candidates["paired_delta_pathological_recall_mean"].max() <= 0.02
    ):
        return "temporal summaries are descriptive but do not improve prediction"
    if best_macro > -0.01:
        return "evidence is mixed and a limited temporal feature experiment is justified"
    return "direct sequence modelling is justified by the observed temporal structure"


def save_figures(
    baseline: pd.DataFrame,
    summaries: pd.DataFrame,
    windows: pd.DataFrame,
    patterns: pd.DataFrame,
    discrimination: pd.DataFrame,
    single: pd.DataFrame,
    groups: pd.DataFrame,
) -> list[Path]:
    """Create ten compact diagnostic figures."""
    merged = baseline.merge(summaries, on=["record_id", "label"], validate="one_to_one")
    colors = {"Normal": "#2a9d8f", "Suspicious": "#e9c46a", "Pathological": "#e76f51"}
    paths = []

    def scatter_pair(x, y, title, filename):
        fig, ax = plt.subplots(figsize=(7.2, 5.5))
        for label, group in merged.groupby("label"):
            ax.scatter(group[x], group[y], s=22, alpha=0.65, label=label, color=colors[label])
        ax.set(xlabel=x.replace("__window_", " window "), ylabel=y.replace("__window_", " window "), title=title)
        ax.legend()
        fig.tight_layout()
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    scatter_pair(
        "short_term_variability", "short_term_variability__window_min",
        "Whole-record STV versus minimum-window STV", "01_stv_whole_vs_minimum.png",
    )
    scatter_pair(
        "bradycardia_percentage", "bradycardia_percentage__window_max",
        "Whole-record versus maximum-window bradycardia", "02_bradycardia_whole_vs_maximum.png",
    )
    scatter_pair(
        "max_deceleration_depth", "max_deceleration_depth__window_max",
        "Whole-record versus local maximum deceleration depth",
        "03_deceleration_whole_vs_local_extreme.png",
    )

    def class_box(feature, title, filename):
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        data = [
            merged.loc[merged["label"] == label, feature].dropna()
            for label in ["Normal", "Suspicious", "Pathological"]
        ]
        boxes = ax.boxplot(data, tick_labels=["Normal", "Suspicious", "Pathological"], patch_artist=True)
        for patch, label in zip(boxes["boxes"], ["Normal", "Suspicious", "Pathological"]):
            patch.set_facecolor(colors[label])
        ax.set(title=title, ylabel=feature.replace("__window_", " window "))
        fig.tight_layout()
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    class_box(
        "short_term_variability__window_delta",
        "Early-to-late STV change by class", "04_variability_early_to_late.png",
    )
    class_box(
        "baseline_fhr__window_slope",
        "Baseline temporal slope by class", "05_temporal_slopes_by_class.png",
    )

    counts = (
        patterns.groupby(["temporal_pattern_category", "label"]).size().unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    counts[["Normal", "Suspicious", "Pathological"]].plot(
        kind="bar", ax=ax, color=[colors["Normal"], colors["Suspicious"], colors["Pathological"]]
    )
    ax.set(title="Exploratory temporal-pattern counts", xlabel="", ylabel="Patients")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = FIGURE_DIR / "06_temporal_pattern_counts.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    temporal_top = discrimination[discrimination["source"] != "whole_record"].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        temporal_top["feature"].str.replace("__window_", " / ", regex=False),
        temporal_top["pathological_roc_auc_discrimination"],
        color="#457b9d",
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Direction-agnostic Pathological ROC AUC", title="Top exploratory temporal candidates")
    fig.tight_layout()
    path = FIGURE_DIR / "07_temporal_candidates_auc.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    def delta_bar(frame, name_col, delta_col, title, filename):
        plot = frame.sort_values(delta_col).copy()
        fig, ax = plt.subplots(figsize=(10, max(4.5, 0.32 * len(plot))))
        ax.barh(
            plot[name_col].str.replace("__window_", " / ", regex=False),
            plot[delta_col],
            color=np.where(plot[delta_col] >= 0, "#2a9d8f", "#e76f51"),
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set(xlabel="Paired mean Macro F1 change", title=title)
        fig.tight_layout()
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    delta_bar(
        single, "candidate", "paired_delta_macro_f1_mean",
        "Single temporal-feature paired repeated-CV changes",
        "08_single_feature_cv_changes.png",
    )
    delta_bar(
        groups, "temporal_group", "paired_delta_macro_f1_mean",
        "Temporal-group paired repeated-CV changes",
        "09_temporal_group_cv_changes.png",
    )

    selected = [
        "short_term_variability", "bradycardia_percentage",
        "deceleration_density_per_hour", "max_deceleration_depth",
        "valid_fhr_percentage",
    ]
    pathological_ids = baseline.loc[baseline["label"] == "Pathological", "record_id"]
    path_windows = windows[windows["record_id"].isin(pathological_ids)].copy()
    max_windows = int(path_windows.groupby("record_id").size().max())
    columns = [(feature, window) for window in range(1, max_windows + 1) for feature in selected]
    heat = np.full((len(pathological_ids), len(columns)), np.nan)
    for row_index, record_id in enumerate(pathological_ids):
        group = path_windows[path_windows["record_id"] == record_id].set_index("window_number")
        for column_index, (feature, window) in enumerate(columns):
            if window in group.index:
                heat[row_index, column_index] = group.loc[window, feature]
    # Standardize each feature across its window-position columns for visibility.
    for feature in selected:
        idx = [i for i, (name, _) in enumerate(columns) if name == feature]
        values = heat[:, idx]
        mean, std = np.nanmean(values), np.nanstd(values)
        if std > 0:
            heat[:, idx] = (values - mean) / std
    fig, ax = plt.subplots(figsize=(15, 8))
    image = ax.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_yticks(np.arange(len(pathological_ids)), pathological_ids)
    ax.set_xticks(
        np.arange(len(columns)),
        [f"W{window} {feature[:8]}" for feature, window in columns],
        rotation=90,
        fontsize=6,
    )
    ax.set(title="Pathological patients: standardized selected features by ordered window")
    fig.colorbar(image, ax=ax, label="Feature-specific standardized value")
    fig.tight_layout()
    path = FIGURE_DIR / "10_pathological_ordered_window_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)
    return paths


def group_mapping_markdown(groups: dict[str, list[str]]) -> str:
    """Render actual feature groups for the report."""
    return "\n".join(
        f"- **{name}:** " + ", ".join(f"`{feature}`" for feature in features)
        for name, features in groups.items()
    )


def write_report(
    windows: pd.DataFrame,
    summaries: pd.DataFrame,
    groups: dict[str, list[str]],
    whole_comparison: pd.DataFrame,
    discrimination: pd.DataFrame,
    paired: pd.DataFrame,
    patterns: pd.DataFrame,
    pathological: pd.DataFrame,
    single: pd.DataFrame,
    group_screen: pd.DataFrame,
    conclusion: str,
) -> None:
    """Write a simple-language audit report grounded in paired-CV results."""
    top_loss = (
        paired.sort_values("pathological_auc_difference", ascending=False)
        .head(8)["feature_concept"].tolist()
    )
    top_temporal = (
        discrimination[discrimination["source"] != "whole_record"]
        .head(8)["feature"].tolist()
    )
    best_single = single.iloc[0]
    best_group = group_screen.iloc[0]
    baseline_macro = best_single["baseline_macro_f1_mean"]
    baseline_recall = best_single["baseline_pathological_recall_mean"]
    baseline_precision = best_single["baseline_pathological_precision_mean"]
    isolated = int(
        (
            pathological["temporal_pattern_category"]
            == "one isolated extreme window"
        ).sum()
    )
    repeated = int(
        pathological["temporal_pattern_category"].isin(
            ["several repeated abnormal windows", "consistently abnormal"]
        ).sum()
    )
    deteriorating = int(
        (pathological["temporal_pattern_category"] == "progressive deterioration").sum()
    )
    final_strongest = int(pathological.merge(
        patterns[["record_id", "strongest_abnormality_in_final_window"]],
        on="record_id",
    )["strongest_abnormality_in_final_window"].sum())
    dominated = int(
        (pathological["temporal_pattern_category"] == "dominated by poor signal").sum()
    )
    no_clear = int(
        (
            pathological["temporal_pattern_category"]
            == "consistently normal or mild"
        ).sum()
    )
    report = f"""# Temporal Information-Loss Audit

## Why this audit was performed

The previous quality-aware experiment showed that rejecting windows below 80%
valid FHR removed useful predictive information. This audit therefore keeps
all diagnostic windows and asks a different question: whether whole-record
aggregation hides local extremes or chronological change.

This is exploratory. Window rows have no independent labels, no window
classifier was trained, and no clinical phenotype is claimed.

## Window representation and validation

The existing quality-aware diagnostic table was reused without recomputation.
It contains **{len(windows)}** non-overlapping windows for **552** patients.
The final slice is present only when it is at least 10 minutes. Window order,
start, end, and duration are explicit. Patient labels were joined only after
patient-level summaries were constructed; outcomes never entered window
features.

## Actual feature mapping

{group_mapping_markdown(groups)}

## Temporal summaries

The audit created **{len(summaries.columns) - 2}** patient-level numeric
summaries. Counts, severity, percentages, baseline, variability, and quality
features used separate documented summary maps. A one-window record receives
NaN—not a false zero—for changes and trends.

The whole-record concepts with the largest exploratory gains in Pathological
univariate AUC were: **{", ".join(top_loss)}**. The highest-ranked temporal
summaries were: **{", ".join(top_temporal)}**.

These rankings use all patients descriptively and are not estimates of model
performance. Near-duplicates and missingness are shown in
`paired_feature_ranking.csv`.

## Localized abnormalities

Patterns use dataset 10th/90th-percentile window thresholds. A window is called
locally extreme only when at least two transparent abnormality indicators are
active. These are exploratory engineering categories, not clinical phenotypes.

Among 27 Pathological patients:

- **{isolated}** had one isolated extreme window.
- **{repeated}** had repeated or consistently abnormal windows.
- **{deteriorating}** were categorized as progressive deterioration.
- **{final_strongest}** had their strongest quantile-defined abnormality in the final window.
- **{dominated}** were dominated by poor signal.
- **{no_clear}** had no clear local temporal abnormality under these definitions.

The detailed audit includes the six fixed-test Pathological records, but the
fixed test set was not used for feature selection or modelling conclusions.

## Single-feature repeated-CV screen

The unchanged baseline pipeline was median imputation, scaling, training-only
SMOTE, and the saved Logistic Regression parameters. All candidates used the
same repeated 5x5 patient folds. No tuning occurred.

Baseline Macro F1 was **{baseline_macro:.4f}**. The highest Macro-F1-ranked
single addition was **{best_single['candidate']}**, with paired mean Macro F1
change **{best_single['paired_delta_macro_f1_mean']:+.4f}**, Pathological
recall change **{best_single['paired_delta_pathological_recall_mean']:+.4f}**,
Pathological precision change
**{best_single['paired_delta_pathological_precision_mean']:+.4f}**, and false
Pathological prediction change
**{best_single['paired_delta_false_pathological_predictions_mean']:+.2f}**.
It improved Macro F1 in
**{int(best_single['macro_f1_improved_fold_count'])}/25** folds.

Because 20 candidates were screened, small favourable values are treated as
hypothesis-generating and not as confirmatory discoveries.

## Small predefined temporal-group screen

The strongest group by paired Macro F1 was
**{best_group['temporal_group']}**, changing Macro F1 by
**{best_group['paired_delta_macro_f1_mean']:+.4f}**, Pathological recall by
**{best_group['paired_delta_pathological_recall_mean']:+.4f}**, Pathological
precision by **{best_group['paired_delta_pathological_precision_mean']:+.4f}**,
and false Pathological predictions by
**{best_group['paired_delta_false_pathological_predictions_mean']:+.2f}**.

The common baseline Pathological recall and precision were
**{baseline_recall:.4f}** and **{baseline_precision:.4f}**.

## Recommendation

Candidates should move forward only when they are understandable, not nearly
duplicate, sufficiently complete, and directionally stable across paired
folds. Descriptive AUC alone is insufficient. Features with high missingness,
near-perfect whole-record correlation, or unstable recall/precision trade-offs
should not be promoted.

The leading candidate for a limited follow-up is
**{best_single['candidate']}**, and the leading predefined group is
**{best_group['temporal_group']}**. Dataset-quantile extreme percentages and
runs are useful descriptions but were deliberately excluded from the modelling
shortlist because their thresholds were estimated across the full dataset.

The next step should remain a small patient-level temporal feature experiment
unless paired results clearly justify a larger architecture. A direct sequence
model is not automatically justified by localized descriptive differences.

## Conclusion

{conclusion}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def validate_outputs(
    summaries, whole, discrimination, paired, patterns, pathological,
    single, groups, figures,
) -> None:
    """Validate required coverage and artifact integrity."""
    if len(summaries) != 552 or summaries["record_id"].nunique() != 552:
        raise AssertionError("Temporal patient summary coverage failed")
    if len(patterns) != 552 or patterns["record_id"].nunique() != 552:
        raise AssertionError("Temporal pattern coverage failed")
    if len(pathological) != 27:
        raise AssertionError("Pathological audit must contain 27 records")
    if len(single) > MAX_SINGLE_FEATURE_CANDIDATES or len(single) < 10:
        raise AssertionError("Single-feature shortlist size is unreasonable")
    if len(groups) != 5:
        raise AssertionError("All five predefined temporal groups are required")
    if not all(len(frame) for frame in [whole, discrimination, paired]):
        raise AssertionError("A required diagnostic table is empty")
    if len(figures) != 10 or not all(path.exists() for path in figures):
        raise AssertionError("All ten expected figures must exist")
    if not REPORT_PATH.exists():
        raise AssertionError("Final report missing")


def main() -> None:
    """Run the temporal information-loss audit."""
    refuse_overwrite()
    windows, labels, _clinical, baseline, model, encoder = load_inputs()
    _distribution, class_counts = validate_windows(windows, labels, baseline)
    groups = actual_feature_groups(windows)

    print("Window dataset validation")
    print(f"Total windows: {len(windows)}")
    counts = windows.groupby("record_id").size()
    print(
        "Windows per patient: "
        f"min={counts.min()}, median={counts.median():.1f}, max={counts.max()}"
    )
    print(class_counts.to_string())
    print("Final-window rule: residual windows shorter than 10 minutes excluded")
    print("No labels or outcomes are present in window feature rows")

    print("\nCreating temporal patient summaries...")
    summaries, summary_map, thresholds = build_temporal_summaries(
        windows, labels, groups
    )
    summaries.to_csv(SUMMARY_PATH, index=False)

    print("\nComparing whole-record and temporal representations...")
    whole = whole_record_comparison(baseline, summaries, summary_map)
    whole.to_csv(WHOLE_VS_TEMPORAL_PATH, index=False)

    print("\nCalculating exploratory univariate discrimination...")
    discrimination = univariate_discrimination(baseline, summaries, summary_map)
    discrimination.to_csv(UNIVARIATE_PATH, index=False)
    paired = paired_feature_ranking(
        baseline, summaries, discrimination, summary_map
    )
    paired.to_csv(PAIRED_RANKING_PATH, index=False)

    print("\nClassifying transparent temporal patterns...")
    patterns, scored_windows = temporal_patterns(windows, labels, thresholds)
    patterns.to_csv(PATTERN_PATH, index=False)
    target = encoder.transform(baseline["label"])
    pathological = pathological_case_audit(
        summaries, patterns, scored_windows, baseline, target
    )
    pathological.to_csv(PATHOLOGICAL_CASE_PATH, index=False)

    shortlist = select_candidate_shortlist(
        paired, discrimination, baseline, summaries
    )
    print(f"\nScreening {len(shortlist)} single temporal additions...")
    single, _single_folds = repeated_cv_screen(
        baseline, summaries, target, encoder, model, shortlist
    )
    single.to_csv(SINGLE_SCREEN_PATH, index=False)

    temporal_groups = predefined_groups(summaries)
    print("\nScreening five predefined temporal groups...")
    group_screen = group_cv_screen(
        baseline, summaries, target, encoder, model, temporal_groups
    )
    group_screen.to_csv(GROUP_SCREEN_PATH, index=False)

    conclusion = choose_conclusion(single, group_screen)
    figures = save_figures(
        baseline, summaries, windows, patterns, discrimination, single,
        group_screen,
    )
    write_report(
        windows, summaries, groups, whole, discrimination, paired, patterns,
        pathological, single, group_screen, conclusion,
    )
    validate_outputs(
        summaries, whole, discrimination, paired, patterns, pathological,
        single, group_screen, figures,
    )

    top_loss = paired.head(5)["feature_concept"].tolist()
    top_univariate = (
        discrimination[discrimination["source"] != "whole_record"]
        .head(5)["feature"].tolist()
    )
    best_single = single.iloc[0]
    best_group = group_screen.iloc[0]
    isolated = int(
        (pathological["temporal_pattern_category"] == "one isolated extreme window").sum()
    )
    repeated = int(
        pathological["temporal_pattern_category"].isin(
            ["several repeated abnormal windows", "consistently abnormal"]
        ).sum()
    )
    print("\nTemporal information-loss audit complete")
    print("Total patients: 552")
    print(f"Total windows: {len(windows)}")
    print(
        f"Windows per patient: min={counts.min()}, median={counts.median():.1f}, "
        f"max={counts.max()}"
    )
    print(f"Candidate temporal summaries created: {len(summaries.columns) - 2}")
    print("Greatest exploratory information loss: " + ", ".join(top_loss))
    print("Top five univariate temporal candidates: " + ", ".join(top_univariate))
    print(
        "Top five single additions: "
        + ", ".join(single.head(5)["candidate"].tolist())
    )
    print(f"Top temporal group: {best_group['temporal_group']}")
    print(
        "Repeated-CV baseline Macro F1 / Pathological recall / precision: "
        f"{best_single['baseline_macro_f1_mean']:.4f} / "
        f"{best_single['baseline_pathological_recall_mean']:.4f} / "
        f"{best_single['baseline_pathological_precision_mean']:.4f}"
    )
    print(
        f"Best screened single result: {best_single['candidate']}; "
        f"Macro F1 change={best_single['paired_delta_macro_f1_mean']:+.4f}"
    )
    print(
        "Pathological recall / precision change: "
        f"{best_single['paired_delta_pathological_recall_mean']:+.4f} / "
        f"{best_single['paired_delta_pathological_precision_mean']:+.4f}"
    )
    print(
        "False Pathological prediction change: "
        f"{best_single['paired_delta_false_pathological_predictions_mean']:+.2f}"
    )
    print(
        f"Pathological isolated / repeated extreme patterns: {isolated} / {repeated}"
    )
    print(f"Main conclusion: {conclusion}")
    for path in [
        SUMMARY_PATH, WHOLE_VS_TEMPORAL_PATH, UNIVARIATE_PATH,
        PAIRED_RANKING_PATH, PATTERN_PATH, PATHOLOGICAL_CASE_PATH,
        SINGLE_SCREEN_PATH, GROUP_SCREEN_PATH, REPORT_PATH, FIGURE_DIR,
    ]:
        print(path)


if __name__ == "__main__":
    main()
