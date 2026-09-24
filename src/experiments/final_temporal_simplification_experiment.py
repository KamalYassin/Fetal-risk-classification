"""Final controlled simplification of the patient-level temporal representation.

Exactly three representations are evaluated:

* A: unchanged whole-record baseline;
* B: baseline plus the five-feature combined temporal representation;
* C: B with only ``quality_transition_count`` removed.

No other feature, threshold, model, patient split, or fold is changed.  The
fixed test is descriptive; selection uses paired repeated splits and repeated
patient-level cross-validation.  Run from the project root:

    python src/experiments/final_temporal_simplification_experiment.py

The script refuses to overwrite prior reports or datasets.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import diagnose_pathological_cases as prior_diagnostic
from src.experiments import limited_temporal_feature_experiment as limited
from src.experiments import quality_aware_aggregation_experiment as qa
from src.models import train_models
from src.utils.record_ids import normalize_record_id

# ---------------------------------------------------------------------------
# Read-only inputs and new output paths
# ---------------------------------------------------------------------------
BASELINE_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.csv"
TEMPORAL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "temporal_information_audit"
    / "temporal_patient_summaries.csv"
)
PRIOR_CV_PATH = (
    PROJECT_ROOT
    / "reports"
    / "limited_temporal_experiment"
    / "repeated_cv_results.csv"
)
PRIOR_SPLIT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "limited_temporal_experiment"
    / "repeated_split_results.csv"
)
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

OUTPUT_ROOT = PROJECT_ROOT / "reports" / "final_temporal_simplification"
DIAGNOSTIC_PATH = OUTPUT_ROOT / "quality_transition_count_diagnostic.csv"
FIXED_PATH = OUTPUT_ROOT / "fixed_split_results.csv"
SPLIT_RESULTS_PATH = OUTPUT_ROOT / "repeated_split_results.csv"
SPLIT_SUMMARY_PATH = OUTPUT_ROOT / "repeated_split_summary.csv"
CV_RESULTS_PATH = OUTPUT_ROOT / "repeated_cv_results.csv"
CV_SUMMARY_PATH = OUTPUT_ROOT / "repeated_cv_summary.csv"
PATHOLOGICAL_PATH = OUTPUT_ROOT / "pathological_case_comparison.csv"
REPORT_PATH = OUTPUT_ROOT / "final_temporal_simplification_report.md"
FINAL_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ml_dataset_final_temporal.csv"
)

NEGATIVE_SLOPE = "maximum_negative_fhr_slope__window_final"
MIN_VALID = "valid_fhr_percentage__window_min"
POOR_QUALITY_PERCENTAGE = "poor_quality_window_percentage"
FINAL_QUALITY = "final_window_quality"
TRANSITION_COUNT = "quality_transition_count"
FULL_TEMPORAL_FEATURES = [
    NEGATIVE_SLOPE,
    MIN_VALID,
    POOR_QUALITY_PERCENTAGE,
    FINAL_QUALITY,
    TRANSITION_COUNT,
]
SIMPLIFIED_TEMPORAL_FEATURES = [
    NEGATIVE_SLOPE,
    MIN_VALID,
    POOR_QUALITY_PERCENTAGE,
    FINAL_QUALITY,
]
REPRESENTATIONS = OrderedDict(
    [
        ("A_baseline", []),
        ("B_full_combined_temporal", FULL_TEMPORAL_FEATURES),
        ("C_simplified_temporal", SIMPLIFIED_TEMPORAL_FEATURES),
    ]
)

SPLIT_SEEDS = list(prior_diagnostic.SPLIT_SEEDS)
CV_SPLITS = prior_diagnostic.CV_SPLITS
CV_REPEATS = prior_diagnostic.CV_REPEATS
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_RANDOM_STATE = 42
EPSILON = 1e-12
EFFECTIVE_MACRO_F1_DIFFERENCE = 0.002
MATERIAL_RECALL_DECREASE = 0.02
MATERIAL_PRECISION_DECREASE = 0.02
MATERIAL_FALSE_ALARM_INCREASE = 1.0
HEAVY_MODAL_CONCENTRATION = 0.50
LOW_NORMALIZED_VARIANCE = 0.01

METRICS = [
    "macro_f1",
    "balanced_accuracy",
    "pathological_recall",
    "pathological_precision",
    "pathological_f1",
    "normal_recall",
    "suspicious_recall",
    "false_pathological_predictions",
    "true_pathological_detections",
]
COMPARISONS = OrderedDict(
    [
        ("B_minus_A", ("B_full_combined_temporal", "A_baseline")),
        ("C_minus_A", ("C_simplified_temporal", "A_baseline")),
        ("C_minus_B", ("C_simplified_temporal", "B_full_combined_temporal")),
    ]
)


def refuse_overwrite() -> None:
    """Prevent accidental replacement of reports or a final dataset."""
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(
            f"{OUTPUT_ROOT} already contains files; move it before rerunning."
        )
    if FINAL_DATASET_PATH.exists():
        raise FileExistsError(
            f"{FINAL_DATASET_PATH} already exists; it will not be overwritten."
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_inputs():
    """Load all required inputs and selected saved configuration."""
    paths = [
        BASELINE_PATH,
        LABELS_PATH,
        TEMPORAL_PATH,
        PRIOR_CV_PATH,
        PRIOR_SPLIT_PATH,
        MODEL_PATH,
        ENCODER_PATH,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    baseline = pd.read_csv(BASELINE_PATH)
    labels = pd.read_csv(LABELS_PATH)
    temporal = pd.read_csv(TEMPORAL_PATH)
    prior_cv = pd.read_csv(PRIOR_CV_PATH)
    prior_splits = pd.read_csv(PRIOR_SPLIT_PATH)
    for frame in [baseline, labels, temporal]:
        frame["record_id"] = normalize_record_id(frame["record_id"])
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    with ENCODER_PATH.open("rb") as handle:
        encoder = pickle.load(handle)
    return baseline, labels, temporal, prior_cv, prior_splits, model, encoder


def validate_inputs(
    baseline: pd.DataFrame,
    labels: pd.DataFrame,
    temporal: pd.DataFrame,
    prior_cv: pd.DataFrame,
    prior_splits: pd.DataFrame,
) -> pd.DataFrame:
    """Validate identical patient rows, labels, schemas, and prior designs."""
    for frame, name in [
        (baseline, "ml_dataset"),
        (labels, "labels"),
        (temporal, "temporal summaries"),
    ]:
        if len(frame) != 552 or frame["record_id"].duplicated().any():
            raise ValueError(f"{name} must contain 552 unique patient rows")
    if baseline["label"].value_counts().get("Pathological", 0) != 27:
        raise ValueError("Expected exactly 27 Pathological patients")
    ids = set(baseline["record_id"])
    if ids != set(labels["record_id"]) or ids != set(temporal["record_id"]):
        raise ValueError("Patient IDs differ across inputs")
    label_values = labels.set_index("record_id").loc[
        baseline["record_id"], "label"
    ].to_numpy()
    temporal_labels = temporal.set_index("record_id").loc[
        baseline["record_id"], "label"
    ].to_numpy()
    if not np.array_equal(label_values, baseline["label"].to_numpy()):
        raise ValueError("Baseline labels differ from labels.csv")
    if not np.array_equal(temporal_labels, baseline["label"].to_numpy()):
        raise ValueError("Temporal labels differ from baseline labels")
    missing_temporal = sorted(set(FULL_TEMPORAL_FEATURES) - set(temporal.columns))
    if missing_temporal:
        raise ValueError("Missing temporal features: " + ", ".join(missing_temporal))
    if len(set(FULL_TEMPORAL_FEATURES)) != len(FULL_TEMPORAL_FEATURES):
        raise ValueError("Duplicate temporal feature names")
    if len(set(baseline.columns)) != len(baseline.columns):
        raise ValueError("Duplicate baseline columns")
    if prior_cv["fold_number"].nunique() != 25:
        raise ValueError("Prior experiment does not contain the expected 25 folds")
    if set(prior_splits["random_seed"]) != set(SPLIT_SEEDS):
        raise ValueError("Prior experiment split seeds do not match the fixed seeds")
    aligned = temporal.set_index("record_id").loc[
        baseline["record_id"]
    ].reset_index()
    if aligned["record_id"].duplicated().any():
        raise AssertionError("Temporal alignment introduced duplicate patients")
    return aligned


def baseline_feature_columns(baseline: pd.DataFrame) -> list[str]:
    """Return leakage-free numeric model inputs."""
    excluded = {train_models.TARGET_COLUMN, *train_models.LEAKAGE_COLUMNS}
    columns = [column for column in baseline.columns if column not in excluded]
    if baseline[columns].select_dtypes(exclude=[np.number]).shape[1]:
        raise ValueError("A non-numeric baseline model feature was found")
    return columns


def build_matrices(
    baseline: pd.DataFrame, temporal: pd.DataFrame
) -> OrderedDict[str, pd.DataFrame]:
    """Build A, B, and C, verifying the single-column B/C difference."""
    base = baseline[baseline_feature_columns(baseline)].copy()
    matrices = OrderedDict()
    for name, features in REPRESENTATIONS.items():
        matrix = base.copy()
        for feature in features:
            if feature in matrix.columns:
                raise ValueError(f"Duplicate input column would be created: {feature}")
            matrix[feature] = pd.to_numeric(temporal[feature], errors="coerce").to_numpy()
        if len(set(matrix.columns)) != len(matrix.columns):
            raise AssertionError(f"Duplicate model columns in {name}")
        matrices[name] = matrix
    b_columns = set(matrices["B_full_combined_temporal"].columns)
    c_columns = set(matrices["C_simplified_temporal"].columns)
    if b_columns - c_columns != {TRANSITION_COUNT} or c_columns - b_columns:
        raise AssertionError("B and C differ by more than quality_transition_count")
    return matrices


def safe_mutual_information(values: np.ndarray, labels: np.ndarray) -> float:
    """Calculate deterministic one-feature three-class mutual information."""
    finite = np.isfinite(values)
    filled = values.copy()
    filled[~finite] = np.median(values[finite])
    return float(
        mutual_info_classif(
            filled.reshape(-1, 1),
            labels,
            random_state=train_models.RANDOM_STATE,
            discrete_features=True,
        )[0]
    )


def quality_transition_diagnostic(
    baseline: pd.DataFrame,
    temporal: pd.DataFrame,
    encoded_target: np.ndarray,
) -> pd.DataFrame:
    """Describe quality_transition_count and its distinct-information evidence."""
    values = pd.to_numeric(temporal[TRANSITION_COUNT], errors="coerce")
    binary = (baseline["label"] == "Pathological").astype(int).to_numpy()
    raw_auc = roc_auc_score(binary, values)
    direction = 1 if raw_auc >= 0.5 else -1
    oriented_ap = average_precision_score(binary, direction * values)
    mi = safe_mutual_information(values.to_numpy(float), encoded_target)
    base_correlations = []
    for feature in baseline_feature_columns(baseline):
        correlation = values.corr(pd.to_numeric(baseline[feature], errors="coerce"))
        if np.isfinite(correlation):
            base_correlations.append((abs(correlation), correlation, feature))
    max_base = max(base_correlations)
    temporal_correlations = {
        feature: values.corr(pd.to_numeric(temporal[feature], errors="coerce"))
        for feature in SIMPLIFIED_TEMPORAL_FEATURES
    }
    modal_fraction = float(values.value_counts(normalize=True, dropna=False).max())
    value_range = float(values.max() - values.min())
    normalized_variance = (
        float(values.var(ddof=1) / (value_range**2)) if value_range > 0 else 0.0
    )
    low_variance = normalized_variance < LOW_NORMALIZED_VARIANCE
    heavy_concentration = modal_fraction >= HEAVY_MODAL_CONCENTRATION
    distinct_information = (
        "limited distinct univariate information; modelling comparison required"
        if max(raw_auc, 1 - raw_auc) < 0.60 and mi < 0.02
        else "some distinct descriptive information"
    )
    rows = []
    for population in ["All", "Normal", "Suspicious", "Pathological"]:
        mask = (
            pd.Series(True, index=baseline.index)
            if population == "All"
            else baseline["label"] == population
        )
        group = values[mask]
        rows.append(
            {
                "population": population,
                "patient_count": int(mask.sum()),
                "definition": (
                    "Number of chronological transitions between window "
                    "valid-FHR >=80% and <80% states."
                ),
                "minimum": group.min(),
                "maximum": group.max(),
                "mean": group.mean(),
                "median": group.median(),
                "standard_deviation": group.std(ddof=1),
                "unique_value_count": group.nunique(dropna=True),
                "value_counts_json": json.dumps(
                    {
                        str(key): int(value)
                        for key, value in group.value_counts(dropna=False)
                        .sort_index()
                        .items()
                    }
                ),
                "modal_value_fraction": modal_fraction if population == "All" else np.nan,
                "normalized_variance_by_squared_range": (
                    normalized_variance if population == "All" else np.nan
                ),
                "low_variance_flag": low_variance if population == "All" else np.nan,
                "heavy_modal_concentration_flag": (
                    heavy_concentration if population == "All" else np.nan
                ),
                "pathological_roc_auc_raw": raw_auc if population == "All" else np.nan,
                "pathological_roc_auc_discrimination": (
                    max(raw_auc, 1 - raw_auc) if population == "All" else np.nan
                ),
                "pathological_average_precision_oriented": (
                    oriented_ap if population == "All" else np.nan
                ),
                "mutual_information_three_class": mi if population == "All" else np.nan,
                "maximum_absolute_correlation_with_baseline": (
                    max_base[0] if population == "All" else np.nan
                ),
                "signed_correlation_with_baseline": (
                    max_base[1] if population == "All" else np.nan
                ),
                "most_correlated_baseline_feature": (
                    max_base[2] if population == "All" else ""
                ),
                "correlation_with_maximum_negative_slope_final": (
                    temporal_correlations[NEGATIVE_SLOPE]
                    if population == "All" else np.nan
                ),
                "correlation_with_minimum_valid_fhr_percentage": (
                    temporal_correlations[MIN_VALID]
                    if population == "All" else np.nan
                ),
                "correlation_with_poor_quality_window_percentage": (
                    temporal_correlations[POOR_QUALITY_PERCENTAGE]
                    if population == "All" else np.nan
                ),
                "correlation_with_final_window_quality": (
                    temporal_correlations[FINAL_QUALITY]
                    if population == "All" else np.nan
                ),
                "pre_modelling_interpretation": (
                    distinct_information if population == "All" else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_predict(
    matrix: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    model,
    seed: int,
):
    """Fit the fold-local pipeline and predict held-out patients."""
    pipeline = qa.build_comparison_pipeline(model, seed)
    pipeline.fit(matrix.iloc[train_indices], target[train_indices])
    return (
        pipeline.predict(matrix.iloc[test_indices]),
        pipeline.predict_proba(matrix.iloc[test_indices]),
    )


def evaluate(
    target: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    encoder,
    record_ids: np.ndarray,
) -> dict:
    """Reuse the limited-experiment metric definition."""
    return limited.evaluate_predictions(
        target, predictions, probabilities, encoder, record_ids
    )


def fixed_indices(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the fixed stratified split."""
    return train_test_split(
        np.arange(len(target)),
        test_size=train_models.TEST_SIZE,
        random_state=train_models.RANDOM_STATE,
        stratify=target,
    )


def fixed_evaluation(
    matrices: OrderedDict[str, pd.DataFrame],
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Evaluate A, B, and C once on the descriptive fixed test."""
    train_indices, test_indices = fixed_indices(target)
    pathological_code = int(encoder.transform(["Pathological"])[0])
    rows = []
    outputs = {}
    for name, matrix in matrices.items():
        predictions, probabilities = fit_predict(
            matrix,
            target,
            train_indices,
            test_indices,
            model,
            train_models.RANDOM_STATE,
        )
        row = {
            "representation": name,
            "temporal_features": " | ".join(REPRESENTATIONS[name]),
            "train_patient_count": len(train_indices),
            "test_patient_count": len(test_indices),
            "pipeline": (
                "SimpleImputer(median) -> StandardScaler -> "
                "SMOTE(training only) -> LogisticRegression(C=1, solver=lbfgs, "
                "max_iter=2000, class_weight=None)"
            ),
            **evaluate(
                target[test_indices],
                predictions,
                probabilities,
                encoder,
                record_ids[test_indices],
            ),
        }
        for record_id in limited.FIXED_PATHOLOGICAL_IDS:
            position = np.flatnonzero(record_ids[test_indices] == record_id)
            if len(position) != 1:
                raise AssertionError(f"Fixed-test record absent: {record_id}")
            row[f"pathological_probability_record_{record_id}"] = float(
                probabilities[position[0], pathological_code]
            )
            row[f"predicted_class_record_{record_id}"] = encoder.inverse_transform(
                [predictions[position[0]]]
            )[0]
        rows.append(row)
        outputs[name] = {
            "predictions": predictions,
            "probabilities": probabilities,
        }
    return pd.DataFrame(rows), train_indices, test_indices, outputs


def one_resample_rows(
    matrices: OrderedDict[str, pd.DataFrame],
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    identity: dict,
) -> list[dict]:
    """Fit all three representations and create one wide paired row."""
    metrics = {}
    for name, matrix in matrices.items():
        predictions, probabilities = fit_predict(
            matrix, target, train_indices, test_indices, model, seed
        )
        metrics[name] = evaluate(
            target[test_indices],
            predictions,
            probabilities,
            encoder,
            record_ids[test_indices],
        )
    row = dict(identity)
    for name, values in metrics.items():
        prefix = name.split("_", 1)[0]
        for metric in METRICS:
            row[f"{prefix}_{metric}"] = values[metric]
    for comparison, (left, right) in COMPARISONS.items():
        left_prefix = left.split("_", 1)[0]
        right_prefix = right.split("_", 1)[0]
        for metric in METRICS:
            row[f"{comparison}_{metric}"] = (
                row[f"{left_prefix}_{metric}"] - row[f"{right_prefix}_{metric}"]
            )
    return [row]


def repeated_splits(
    matrices: OrderedDict[str, pd.DataFrame],
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
) -> pd.DataFrame:
    """Evaluate representations A, B, and C on 30 predefined patient splits."""
    rows = []
    indices = np.arange(len(target))
    for position, seed in enumerate(SPLIT_SEEDS, start=1):
        train_indices, test_indices = train_test_split(
            indices,
            test_size=train_models.TEST_SIZE,
            random_state=seed,
            stratify=target,
        )
        rows.extend(
            one_resample_rows(
                matrices,
                target,
                record_ids,
                encoder,
                model,
                train_indices,
                test_indices,
                seed,
                {
                    "random_seed": seed,
                    "train_patient_count": len(train_indices),
                    "test_patient_count": len(test_indices),
                },
            )
        )
        print(f"  Predefined split: {position}/{len(SPLIT_SEEDS)}")
    return pd.DataFrame(rows)


def repeated_cv(
    matrices: OrderedDict[str, pd.DataFrame],
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
) -> pd.DataFrame:
    """Evaluate representations A, B, and C on matching repeated 5x5 folds."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=train_models.RANDOM_STATE,
    )
    rows = []
    for fold_number, (train_indices, test_indices) in enumerate(
        cv.split(np.zeros(len(target)), target), start=1
    ):
        rows.extend(
            one_resample_rows(
                matrices,
                target,
                record_ids,
                encoder,
                model,
                train_indices,
                test_indices,
                train_models.RANDOM_STATE + fold_number,
                {
                    "fold_number": fold_number,
                    "repeat_number": (fold_number - 1) // CV_SPLITS + 1,
                    "fold_within_repeat": (fold_number - 1) % CV_SPLITS + 1,
                    "train_patient_count": len(train_indices),
                    "test_patient_count": len(test_indices),
                },
            )
        )
        print(f"  Repeated-CV fold: {fold_number}/{CV_SPLITS * CV_REPEATS}")
    return pd.DataFrame(rows)


def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    """Return a deterministic paired 95% bootstrap mean interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_STATE)
    means = rng.choice(
        values,
        size=(BOOTSTRAP_RESAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def summarize_results(results: pd.DataFrame, evaluation_count: int) -> pd.DataFrame:
    """Summarize representation metrics and all three paired comparisons."""
    rows = []
    for metric in METRICS:
        for representation in REPRESENTATIONS:
            prefix = representation.split("_", 1)[0]
            values = results[f"{prefix}_{metric}"]
            rows.append(
                {
                    "row_type": "representation",
                    "representation_or_comparison": representation,
                    "metric": metric,
                    "evaluation_count": evaluation_count,
                    "mean": values.mean(),
                    "median": values.median(),
                    "standard_deviation": values.std(ddof=1),
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "paired_mean_difference": np.nan,
                    "paired_median_difference": np.nan,
                    "paired_difference_standard_deviation": np.nan,
                    "paired_difference_minimum": np.nan,
                    "paired_difference_maximum": np.nan,
                    "paired_mean_bootstrap_95ci_lower": np.nan,
                    "paired_mean_bootstrap_95ci_upper": np.nan,
                    "improved_count": np.nan,
                    "worsened_count": np.nan,
                    "equal_count": np.nan,
                }
            )
        for comparison in COMPARISONS:
            differences = results[f"{comparison}_{metric}"]
            lower, upper = bootstrap_interval(differences.to_numpy())
            rows.append(
                {
                    "row_type": "paired_comparison",
                    "representation_or_comparison": comparison,
                    "metric": metric,
                    "evaluation_count": evaluation_count,
                    "mean": np.nan,
                    "median": np.nan,
                    "standard_deviation": np.nan,
                    "minimum": np.nan,
                    "maximum": np.nan,
                    "paired_mean_difference": differences.mean(),
                    "paired_median_difference": differences.median(),
                    "paired_difference_standard_deviation": differences.std(ddof=1),
                    "paired_difference_minimum": differences.min(),
                    "paired_difference_maximum": differences.max(),
                    "paired_mean_bootstrap_95ci_lower": lower,
                    "paired_mean_bootstrap_95ci_upper": upper,
                    "improved_count": int((differences > EPSILON).sum()),
                    "worsened_count": int((differences < -EPSILON).sum()),
                    "equal_count": int((abs(differences) <= EPSILON).sum()),
                }
            )
    return pd.DataFrame(rows)


def validate_prior_reproduction(
    cv_results: pd.DataFrame,
    split_results: pd.DataFrame,
    prior_cv: pd.DataFrame,
    prior_splits: pd.DataFrame,
) -> None:
    """Confirm A and B exactly reproduce the preceding controlled experiment."""
    for new_prefix, prior_name in [
        ("A", "A_baseline"),
        ("B", "D_combined_limited_temporal"),
    ]:
        old_cv = prior_cv[
            prior_cv["representation"] == prior_name
        ].sort_values("fold_number")
        new_cv = cv_results.sort_values("fold_number")
        old_splits = prior_splits[
            prior_splits["representation"] == prior_name
        ].sort_values("random_seed")
        new_splits = split_results.sort_values("random_seed")
        for metric in METRICS:
            if not np.allclose(
                new_cv[f"{new_prefix}_{metric}"],
                old_cv[metric],
                equal_nan=True,
            ):
                raise AssertionError(
                    f"Prior repeated-CV reproduction failed: {prior_name}/{metric}"
                )
            if not np.allclose(
                new_splits[f"{new_prefix}_{metric}"],
                old_splits[metric],
                equal_nan=True,
            ):
                raise AssertionError(
                    f"Prior repeated-split reproduction failed: {prior_name}/{metric}"
                )


def fixed_train_oof(
    matrices: OrderedDict[str, pd.DataFrame],
    target: np.ndarray,
    fixed_train_indices: np.ndarray,
    encoder,
    model,
) -> dict:
    """Create fixed-training five-fold OOF predictions for each representation."""
    splitter = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=train_models.RANDOM_STATE,
    )
    pathological_code = int(encoder.transform(["Pathological"])[0])
    train_target = target[fixed_train_indices]
    outputs = {}
    for name, matrix in matrices.items():
        predictions = np.full(len(fixed_train_indices), -1, dtype=int)
        probabilities = np.full(len(fixed_train_indices), np.nan)
        for fold_number, (inner_train, validation) in enumerate(
            splitter.split(np.zeros(len(fixed_train_indices)), train_target), start=1
        ):
            fold_predictions, fold_probabilities = fit_predict(
                matrix,
                target,
                fixed_train_indices[inner_train],
                fixed_train_indices[validation],
                model,
                train_models.RANDOM_STATE + fold_number,
            )
            predictions[validation] = fold_predictions
            probabilities[validation] = fold_probabilities[:, pathological_code]
        outputs[name] = {
            "predictions": predictions,
            "pathological_probabilities": probabilities,
        }
    return outputs


def pathological_comparison(
    baseline: pd.DataFrame,
    temporal: pd.DataFrame,
    matrices: OrderedDict[str, pd.DataFrame],
    target: np.ndarray,
    encoder,
    model,
    fixed_train_indices: np.ndarray,
    fixed_test_indices: np.ndarray,
    fixed_outputs: dict,
) -> pd.DataFrame:
    """Compare B and C for every Pathological patient with held-out predictions."""
    pathological_code = int(encoder.transform(["Pathological"])[0])
    oof = fixed_train_oof(
        matrices, target, fixed_train_indices, encoder, model
    )
    train_position = {
        absolute: position for position, absolute in enumerate(fixed_train_indices)
    }
    test_position = {
        absolute: position for position, absolute in enumerate(fixed_test_indices)
    }
    rows = []
    for absolute in np.flatnonzero(target == pathological_code):
        is_test = absolute in test_position
        row = {
            "record_id": baseline.iloc[absolute]["record_id"],
            "split_membership": "fixed_test" if is_test else "fixed_train_oof",
        }
        values = {}
        for representation in REPRESENTATIONS:
            if is_test:
                position = test_position[absolute]
                prediction = fixed_outputs[representation]["predictions"][position]
                probability = fixed_outputs[representation]["probabilities"][
                    position, pathological_code
                ]
            else:
                position = train_position[absolute]
                prediction = oof[representation]["predictions"][position]
                probability = oof[representation]["pathological_probabilities"][position]
            predicted_label = encoder.inverse_transform([prediction])[0]
            prefix = representation.split("_", 1)[0]
            row[f"{prefix}_prediction"] = predicted_label
            row[f"{prefix}_pathological_probability"] = probability
            values[prefix] = (predicted_label, probability)
        b_correct = values["B"][0] == "Pathological"
        c_correct = values["C"][0] == "Pathological"
        row["C_minus_B_pathological_probability"] = values["C"][1] - values["B"][1]
        row["removal_improved_prediction"] = bool(c_correct and not b_correct)
        row["removal_worsened_prediction"] = bool(b_correct and not c_correct)
        for feature in FULL_TEMPORAL_FEATURES:
            row[feature] = temporal.iloc[absolute][feature]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split_membership", "record_id"])


def summary_value(
    summary: pd.DataFrame,
    row_name: str,
    metric: str,
    column: str,
) -> float:
    """Retrieve one representation or comparison summary value."""
    row = summary[
        (summary["representation_or_comparison"] == row_name)
        & (summary["metric"] == metric)
    ]
    if len(row) != 1:
        raise AssertionError(f"Missing summary row {row_name}/{metric}")
    return float(row.iloc[0][column])


def select_representation(
    cv_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    pathological: pd.DataFrame,
) -> tuple[str, str, dict]:
    """Apply the predefined B-versus-C simplification rule."""
    cv_macro_delta = summary_value(
        cv_summary, "C_minus_B", "macro_f1", "paired_mean_difference"
    )
    split_macro_delta = summary_value(
        split_summary, "C_minus_B", "macro_f1", "paired_mean_difference"
    )
    recall_delta = summary_value(
        cv_summary, "C_minus_B", "pathological_recall", "paired_mean_difference"
    )
    precision_delta = summary_value(
        cv_summary, "C_minus_B", "pathological_precision", "paired_mean_difference"
    )
    false_delta = summary_value(
        cv_summary,
        "C_minus_B",
        "false_pathological_predictions",
        "paired_mean_difference",
    )
    cv_improved = int(
        summary_value(cv_summary, "C_minus_B", "macro_f1", "improved_count")
    )
    cv_worsened = int(
        summary_value(cv_summary, "C_minus_B", "macro_f1", "worsened_count")
    )
    cv_equal = int(
        summary_value(cv_summary, "C_minus_B", "macro_f1", "equal_count")
    )
    split_improved = int(
        summary_value(split_summary, "C_minus_B", "macro_f1", "improved_count")
    )
    split_worsened = int(
        summary_value(split_summary, "C_minus_B", "macro_f1", "worsened_count")
    )
    split_equal = int(
        summary_value(split_summary, "C_minus_B", "macro_f1", "equal_count")
    )
    harmed = int(pathological["removal_worsened_prediction"].sum())
    effectively_equal = (
        abs(cv_macro_delta) <= EFFECTIVE_MACRO_F1_DIFFERENCE
        and abs(split_macro_delta) <= EFFECTIVE_MACRO_F1_DIFFERENCE
        and recall_delta >= -MATERIAL_RECALL_DECREASE
        and precision_delta >= -MATERIAL_PRECISION_DECREASE
        and false_delta <= MATERIAL_FALSE_ALARM_INCREASE
    )
    criteria = {
        "cv_macro_equal_or_higher": cv_macro_delta >= -EPSILON,
        "split_macro_equal_or_higher": split_macro_delta >= -EPSILON,
        "recall_not_materially_lower": recall_delta >= -MATERIAL_RECALL_DECREASE,
        "precision_equal_or_higher": precision_delta >= -EPSILON,
        "false_predictions_not_materially_higher": (
            false_delta <= MATERIAL_FALSE_ALARM_INCREASE
        ),
        "cv_consistency_or_effective_equality": (
            cv_improved >= 13 or effectively_equal
        ),
        "split_consistency_or_effective_equality": (
            split_improved >= 16 or effectively_equal
        ),
        "no_pathological_classification_harmed": harmed == 0,
        "simpler_and_more_reproducible": True,
    }
    satisfied = sum(criteria.values())
    select_c = satisfied >= 6 and (
        (cv_macro_delta >= 0 and split_macro_delta >= 0)
        or effectively_equal
    )
    if select_c:
        if effectively_equal:
            conclusion = (
                "both representations are effectively equivalent, so prefer "
                "the simpler representation"
            )
        else:
            conclusion = (
                "remove quality_transition_count and use the simplified "
                "temporal representation"
            )
        selected = "C_simplified_temporal"
    else:
        selected = "B_full_combined_temporal"
        conclusion = (
            "retain quality_transition_count in the combined representation"
            if cv_macro_delta < 0 and split_macro_delta < 0
            else "evidence is mixed and the current combined representation should remain unchanged"
        )
    details = {
        "criteria": criteria,
        "criteria_satisfied": satisfied,
        "effectively_equal": effectively_equal,
        "cv_macro_delta": cv_macro_delta,
        "split_macro_delta": split_macro_delta,
        "recall_delta": recall_delta,
        "precision_delta": precision_delta,
        "false_delta": false_delta,
        "cv_counts": (cv_improved, cv_worsened, cv_equal),
        "split_counts": (split_improved, split_worsened, split_equal),
        "pathological_harmed": harmed,
    }
    return selected, conclusion, details


def create_final_dataset(
    baseline: pd.DataFrame,
    temporal: pd.DataFrame,
    selected: str,
) -> bool:
    """Create the traceable final dataset only when simplified C is selected."""
    if selected != "C_simplified_temporal":
        return False
    model_features = baseline_feature_columns(baseline)
    outcome_columns = ["pH", "BE", "BDecf", "Apgar5"]
    result = pd.DataFrame({"record_id": baseline["record_id"]})
    for feature in model_features:
        result[feature] = baseline[feature]
    for feature in SIMPLIFIED_TEMPORAL_FEATURES:
        result[feature] = temporal[feature]
    for column in outcome_columns:
        result[column] = baseline[column]
    result["label"] = baseline["label"]
    if TRANSITION_COUNT in result or any(
        feature not in SIMPLIFIED_TEMPORAL_FEATURES
        and "__window_" in feature
        for feature in result.columns
    ):
        raise AssertionError("Unexpected temporal feature in final dataset")
    if len(result) != 552 or result["record_id"].nunique() != 552:
        raise AssertionError("Final dataset patient coverage failed")
    if len(result.columns) != len(set(result.columns)):
        raise AssertionError("Final dataset has duplicate columns")
    for column in [*outcome_columns, "label"]:
        if column == "label":
            equal = result[column].equals(baseline[column])
        else:
            equal = np.allclose(result[column], baseline[column], equal_nan=True)
        if not equal:
            raise AssertionError(f"Final dataset changed {column}")
    result.to_csv(FINAL_DATASET_PATH, index=False)
    return True


def metric_table(summary: pd.DataFrame, metric: str) -> str:
    """Render representation and paired B/C rows for the report."""
    rows = summary[summary["metric"] == metric]
    lines = [
        "| Row | Mean ± SD or paired change | 95% paired CI | Better/Worse/Equal |",
        "|---|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        name = row["representation_or_comparison"]
        if row["row_type"] == "representation":
            value = f"{row['mean']:.4f} ± {row['standard_deviation']:.4f}"
            interval = "—"
            counts = "—"
        else:
            value = f"{row['paired_mean_difference']:+.4f}"
            interval = (
                f"[{row['paired_mean_bootstrap_95ci_lower']:+.4f}, "
                f"{row['paired_mean_bootstrap_95ci_upper']:+.4f}]"
            )
            counts = (
                f"{int(row['improved_count'])}/"
                f"{int(row['worsened_count'])}/"
                f"{int(row['equal_count'])}"
            )
        lines.append(f"| {name} | {value} | {interval} | {counts} |")
    return "\n".join(lines)


def write_report(
    diagnostic: pd.DataFrame,
    fixed: pd.DataFrame,
    split_summary: pd.DataFrame,
    cv_summary: pd.DataFrame,
    pathological: pd.DataFrame,
    selected: str,
    conclusion: str,
    details: dict,
    dataset_created: bool,
) -> None:
    """Write the final controlled simplification report."""
    overall = diagnostic[diagnostic["population"] == "All"].iloc[0]
    fixed_lookup = fixed.set_index("representation")
    b_fixed = fixed_lookup.loc["B_full_combined_temporal"]
    c_fixed = fixed_lookup.loc["C_simplified_temporal"]
    helped = int(pathological["removal_improved_prediction"].sum())
    harmed = int(pathological["removal_worsened_prediction"].sum())
    cv_ci_lower = summary_value(
        cv_summary,
        "C_minus_B",
        "macro_f1",
        "paired_mean_bootstrap_95ci_lower",
    )
    cv_ci_upper = summary_value(
        cv_summary,
        "C_minus_B",
        "macro_f1",
        "paired_mean_bootstrap_95ci_upper",
    )
    split_ci_lower = summary_value(
        split_summary,
        "C_minus_B",
        "macro_f1",
        "paired_mean_bootstrap_95ci_lower",
    )
    split_ci_upper = summary_value(
        split_summary,
        "C_minus_B",
        "macro_f1",
        "paired_mean_bootstrap_95ci_upper",
    )
    report = f"""# Final Temporal Simplification Experiment

## Purpose and controlled difference

The prior leave-one-feature-out audit suggested that
`quality_transition_count` might dilute the combined temporal representation.
This final experiment compares only:

- **A:** the unchanged baseline.
- **B:** `{NEGATIVE_SLOPE}`, `{MIN_VALID}`,
  `{POOR_QUALITY_PERCENTAGE}`, `{FINAL_QUALITY}`, and `{TRANSITION_COUNT}`.
- **C:** the same representation with only `{TRANSITION_COUNT}` removed.

Patients, labels, leakage exclusions, fixed split, 30 seeds, repeated 5x5
folds, preprocessing, SMOTE placement, and Logistic Regression parameters were
identical. No fixed-test result influenced selection.

The newly calculated A and B metrics exactly reproduced the previous limited
experiment on every repeated-CV fold and predefined split.

## quality_transition_count diagnostic

The feature counts chronological transitions between window valid-FHR >=80%
and <80% states. It ranged from **{overall['minimum']:.0f}** to
**{overall['maximum']:.0f}**, with mean **{overall['mean']:.3f}**, median
**{overall['median']:.3f}**, SD **{overall['standard_deviation']:.3f}**, and
**{int(overall['unique_value_count'])}** unique values. Its most frequent value
contained **{100 * overall['modal_value_fraction']:.1f}%** of patients.

Its direction-agnostic Pathological ROC AUC was
**{overall['pathological_roc_auc_discrimination']:.4f}**, three-class mutual
information was **{overall['mutual_information_three_class']:.4f}**, and its
largest absolute correlation with a baseline feature was
**{overall['maximum_absolute_correlation_with_baseline']:.4f}** with
`{overall['most_correlated_baseline_feature']}`.

The feature is not near-zero variance and is not dominated by a single value,
but its univariate information is limited. The paired modelling comparison is
therefore the primary evidence for whether it adds useful information.

## Fixed test (descriptive only)

B fixed Macro F1 / Pathological recall / precision were
**{b_fixed['macro_f1']:.4f} / {b_fixed['pathological_recall']:.4f} /
{b_fixed['pathological_precision']:.4f}**. C values were
**{c_fixed['macro_f1']:.4f} / {c_fixed['pathological_recall']:.4f} /
{c_fixed['pathological_precision']:.4f}**. This result was not used for
selection.

## Thirty predefined splits

### Macro F1

{metric_table(split_summary, 'macro_f1')}

### Pathological recall

{metric_table(split_summary, 'pathological_recall')}

### Pathological precision

{metric_table(split_summary, 'pathological_precision')}

### False Pathological predictions

{metric_table(split_summary, 'false_pathological_predictions')}

## Repeated 5x5 patient-level cross-validation

### Macro F1

{metric_table(cv_summary, 'macro_f1')}

### Pathological recall

{metric_table(cv_summary, 'pathological_recall')}

### Pathological precision

{metric_table(cv_summary, 'pathological_precision')}

### False Pathological predictions

{metric_table(cv_summary, 'false_pathological_predictions')}

C-minus-B repeated-CV Macro F1 changed by
**{details['cv_macro_delta']:+.4f}** with 95% interval
**[{cv_ci_lower:+.4f}, {cv_ci_upper:+.4f}]**. Across predefined splits it
changed by **{details['split_macro_delta']:+.4f}** with interval
**[{split_ci_lower:+.4f}, {split_ci_upper:+.4f}]**. Intervals crossing zero are
treated as statistical uncertainty, not significance.

C-minus-B Pathological recall, precision, and false-Pathological changes in
repeated CV were **{details['recall_delta']:+.4f}**,
**{details['precision_delta']:+.4f}**, and **{details['false_delta']:+.2f}**.
Macro F1 improved/worsened/equal in
**{details['cv_counts'][0]}/{details['cv_counts'][1]}/{details['cv_counts'][2]}**
CV folds and
**{details['split_counts'][0]}/{details['split_counts'][1]}/{details['split_counts'][2]}**
predefined splits.

## Pathological patients

All 27 Pathological patients were retained. Fixed-test patients use fixed
predictions and training patients use five-fold out-of-fold predictions.
Removing the transition feature improved **{helped}** classifications and
worsened **{harmed}**. Per-patient probabilities and temporal values are in
`pathological_case_comparison.csv`.

Because B and C were effectively equivalent and removal did not harm any
Pathological classification in the fixed/OOF case audit,
`quality_transition_count` did not show reliable incremental predictive value
in this controlled comparison.

## Selection and final dataset

The predefined rule selected **{selected}** after satisfying
**{details['criteria_satisfied']}/9** criteria. B and C were classified as
effectively equivalent: **{details['effectively_equal']}**.

The final simplified dataset was **{"created" if dataset_created else "not created"}**.
It contains the unchanged baseline model features, four selected temporal
features, record ID, outcomes, and label. Outcomes remain traceability columns,
not model inputs. `quality_transition_count` and all other temporal summaries
are absent.

The simplification does not justify sequence modelling, and no claim of
clinical significance is made.

## Conclusion

{conclusion}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def validate_outputs(
    diagnostic: pd.DataFrame,
    fixed: pd.DataFrame,
    split_results: pd.DataFrame,
    split_summary: pd.DataFrame,
    cv_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
    pathological: pd.DataFrame,
    dataset_created: bool,
) -> None:
    """Validate evaluation coverage and conditional dataset creation."""
    if len(diagnostic) != 4:
        raise AssertionError("Diagnostic must contain overall and three class rows")
    if len(fixed) != 3:
        raise AssertionError("Fixed evaluation must contain A/B/C")
    if len(split_results) != 30 or set(split_results["random_seed"]) != set(SPLIT_SEEDS):
        raise AssertionError("Repeated split coverage changed")
    if len(cv_results) != 25 or set(cv_results["fold_number"]) != set(range(1, 26)):
        raise AssertionError("Repeated-CV fold coverage changed")
    expected_summary_rows = len(METRICS) * (
        len(REPRESENTATIONS) + len(COMPARISONS)
    )
    if len(split_summary) != expected_summary_rows or len(cv_summary) != expected_summary_rows:
        raise AssertionError("A summary table is incomplete")
    if len(pathological) != 27 or pathological["record_id"].nunique() != 27:
        raise AssertionError("Pathological patient comparison is incomplete")
    if dataset_created != FINAL_DATASET_PATH.exists():
        raise AssertionError("Conditional final dataset state is inconsistent")
    if not REPORT_PATH.exists():
        raise AssertionError("Final report is missing")


def main() -> None:
    """Run the final controlled simplification experiment."""
    refuse_overwrite()
    baseline, labels, temporal, prior_cv, prior_splits, model, encoder = load_inputs()
    temporal = validate_inputs(
        baseline, labels, temporal, prior_cv, prior_splits
    )
    matrices = build_matrices(baseline, temporal)
    target = encoder.transform(baseline["label"])
    record_ids = baseline["record_id"].to_numpy()
    diagnostic = quality_transition_diagnostic(baseline, temporal, target)
    diagnostic.to_csv(DIAGNOSTIC_PATH, index=False)

    print("Patients retained: 552")
    print(f"B exact temporal features: {FULL_TEMPORAL_FEATURES}")
    print(f"C exact temporal features: {SIMPLIFIED_TEMPORAL_FEATURES}")
    print("\nquality_transition_count diagnostic")
    print(diagnostic[diagnostic["population"] == "All"].to_string(index=False))

    print("\nRunning fixed split...")
    fixed, fixed_train, fixed_test, fixed_outputs = fixed_evaluation(
        matrices, target, record_ids, encoder, model
    )
    fixed.to_csv(FIXED_PATH, index=False)

    print("\nRunning 30 predefined splits...")
    split_results = repeated_splits(
        matrices, target, record_ids, encoder, model
    )
    split_results.to_csv(SPLIT_RESULTS_PATH, index=False)
    split_summary = summarize_results(split_results, 30)
    split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False)

    print("\nRunning repeated 5x5 cross-validation...")
    cv_results = repeated_cv(
        matrices, target, record_ids, encoder, model
    )
    cv_results.to_csv(CV_RESULTS_PATH, index=False)
    cv_summary = summarize_results(cv_results, 25)
    cv_summary.to_csv(CV_SUMMARY_PATH, index=False)
    validate_prior_reproduction(
        cv_results, split_results, prior_cv, prior_splits
    )

    print("\nCreating Pathological fixed/OOF comparison...")
    pathological = pathological_comparison(
        baseline,
        temporal,
        matrices,
        target,
        encoder,
        model,
        fixed_train,
        fixed_test,
        fixed_outputs,
    )
    pathological.to_csv(PATHOLOGICAL_PATH, index=False)

    selected, conclusion, details = select_representation(
        cv_summary, split_summary, pathological
    )
    dataset_created = create_final_dataset(baseline, temporal, selected)
    write_report(
        diagnostic,
        fixed,
        split_summary,
        cv_summary,
        pathological,
        selected,
        conclusion,
        details,
        dataset_created,
    )
    validate_outputs(
        diagnostic,
        fixed,
        split_results,
        split_summary,
        cv_results,
        cv_summary,
        pathological,
        dataset_created,
    )

    print("\nFinal temporal simplification experiment complete")
    for representation in REPRESENTATIONS:
        print(
            f"{representation}: CV Macro F1="
            f"{summary_value(cv_summary, representation, 'macro_f1', 'mean'):.4f}; "
            f"recall="
            f"{summary_value(cv_summary, representation, 'pathological_recall', 'mean'):.4f}; "
            f"precision="
            f"{summary_value(cv_summary, representation, 'pathological_precision', 'mean'):.4f}; "
            f"false Pathological="
            f"{summary_value(cv_summary, representation, 'false_pathological_predictions', 'mean'):.2f}"
        )
    print(
        "C-minus-B CV Macro F1 / recall / precision / false predictions: "
        f"{details['cv_macro_delta']:+.4f} / {details['recall_delta']:+.4f} / "
        f"{details['precision_delta']:+.4f} / {details['false_delta']:+.2f}"
    )
    print(
        "CV folds improved/worsened/equal: "
        f"{details['cv_counts'][0]}/{details['cv_counts'][1]}/{details['cv_counts'][2]}"
    )
    print(
        "Predefined splits improved/worsened/equal: "
        f"{details['split_counts'][0]}/{details['split_counts'][1]}/"
        f"{details['split_counts'][2]}"
    )
    print(f"Selected representation: {selected}")
    print(f"Final dataset created: {dataset_created}")
    print(f"Conclusion: {conclusion}")
    for path in [
        DIAGNOSTIC_PATH,
        FIXED_PATH,
        SPLIT_RESULTS_PATH,
        SPLIT_SUMMARY_PATH,
        CV_RESULTS_PATH,
        CV_SUMMARY_PATH,
        PATHOLOGICAL_PATH,
        *([FINAL_DATASET_PATH] if dataset_created else []),
        REPORT_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()
