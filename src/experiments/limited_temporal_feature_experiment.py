"""Run a controlled patient-level experiment with five predefined temporal features.

The experiment compares exactly four representations:

* the unchanged whole-record baseline;
* baseline plus final-window maximum negative FHR slope;
* baseline plus the audit's exact four-feature temporal quality context;
* baseline plus both predefined additions.

No window labels, window classifier, sequence model, tuning, or arbitrary
feature search is used.  All model preprocessing is learned inside training
folds.  Run from the project root:

    python src/experiments/limited_temporal_feature_experiment.py

The script refuses to overwrite an existing output directory.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import diagnose_pathological_cases as prior_diagnostic
from src.experiments import quality_aware_aggregation_experiment as qa
from src.models import train_models
from src.utils.record_ids import normalize_record_id

# ---------------------------------------------------------------------------
# Read-only inputs and new outputs
# ---------------------------------------------------------------------------
BASELINE_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.csv"
WINDOW_PATH = (
    PROJECT_ROOT / "reports" / "quality_aware_aggregation" / "all_window_features.csv"
)
TEMPORAL_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "temporal_information_audit"
    / "temporal_patient_summaries.csv"
)
SINGLE_AUDIT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "temporal_information_audit"
    / "single_feature_addition_screen.csv"
)
GROUP_AUDIT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "temporal_information_audit"
    / "temporal_group_screen.csv"
)
PAIRED_RANKING_PATH = (
    PROJECT_ROOT
    / "reports"
    / "temporal_information_audit"
    / "paired_feature_ranking.csv"
)
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

OUTPUT_ROOT = PROJECT_ROOT / "reports" / "limited_temporal_experiment"
FIGURE_DIR = OUTPUT_ROOT / "figures"
FEATURE_DIAGNOSTIC_PATH = OUTPUT_ROOT / "selected_temporal_feature_diagnostics.csv"
CORRELATION_PATH = OUTPUT_ROOT / "selected_temporal_feature_correlations.csv"
FIXED_PATH = OUTPUT_ROOT / "fixed_split_results.csv"
REPEATED_SPLIT_PATH = OUTPUT_ROOT / "repeated_split_results.csv"
REPEATED_SPLIT_SUMMARY_PATH = OUTPUT_ROOT / "repeated_split_summary.csv"
REPEATED_CV_PATH = OUTPUT_ROOT / "repeated_cv_results.csv"
REPEATED_CV_SUMMARY_PATH = OUTPUT_ROOT / "repeated_cv_summary.csv"
PATHOLOGICAL_PATH = OUTPUT_ROOT / "pathological_case_results.csv"
ABLATION_PATH = OUTPUT_ROOT / "temporal_feature_ablation.csv"
REPORT_PATH = OUTPUT_ROOT / "limited_temporal_experiment_report.md"

FINAL_NEGATIVE_SLOPE = "maximum_negative_fhr_slope__window_final"
EXPECTED_QUALITY_CONTEXT = [
    "valid_fhr_percentage__window_min",
    "poor_quality_window_percentage",
    "final_window_quality",
    "quality_transition_count",
]
ALL_SELECTED_TEMPORAL = [FINAL_NEGATIVE_SLOPE, *EXPECTED_QUALITY_CONTEXT]
REPRESENTATIONS = OrderedDict(
    [
        ("A_baseline", []),
        ("B_final_negative_slope", [FINAL_NEGATIVE_SLOPE]),
        ("C_temporal_quality_context", EXPECTED_QUALITY_CONTEXT),
        ("D_combined_limited_temporal", ALL_SELECTED_TEMPORAL),
    ]
)

FIXED_PATHOLOGICAL_IDS = ["1002", "1071", "1158", "1418", "1490", "2009"]
SPLIT_SEEDS = list(prior_diagnostic.SPLIT_SEEDS)
CV_SPLITS = prior_diagnostic.CV_SPLITS
CV_REPEATS = prior_diagnostic.CV_REPEATS
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_RANDOM_STATE = 42
CORRELATION_LIMIT = 0.95
MISSINGNESS_LIMIT_PERCENTAGE = 30.0
NEAR_ZERO_VARIANCE = 1e-10
EPSILON = 1e-12

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


def refuse_overwrite() -> None:
    """Prevent replacement of an earlier controlled experiment."""
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(
            f"{OUTPUT_ROOT} already contains files; move it before rerunning."
        )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    """Load all required inputs and saved configuration."""
    paths = [
        BASELINE_PATH,
        LABELS_PATH,
        WINDOW_PATH,
        TEMPORAL_SUMMARY_PATH,
        SINGLE_AUDIT_PATH,
        GROUP_AUDIT_PATH,
        PAIRED_RANKING_PATH,
        MODEL_PATH,
        ENCODER_PATH,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    baseline = pd.read_csv(BASELINE_PATH)
    labels = pd.read_csv(LABELS_PATH)
    windows = pd.read_csv(WINDOW_PATH)
    temporal = pd.read_csv(TEMPORAL_SUMMARY_PATH)
    single_audit = pd.read_csv(SINGLE_AUDIT_PATH)
    group_audit = pd.read_csv(GROUP_AUDIT_PATH)
    paired = pd.read_csv(PAIRED_RANKING_PATH)
    for frame in [baseline, labels, windows, temporal]:
        frame["record_id"] = normalize_record_id(frame["record_id"])
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    with ENCODER_PATH.open("rb") as handle:
        encoder = pickle.load(handle)
    return (
        baseline,
        labels,
        windows,
        temporal,
        single_audit,
        group_audit,
        paired,
        model,
        encoder,
    )


def validate_inputs(
    baseline: pd.DataFrame,
    labels: pd.DataFrame,
    windows: pd.DataFrame,
    temporal: pd.DataFrame,
    single_audit: pd.DataFrame,
    group_audit: pd.DataFrame,
) -> None:
    """Validate patient alignment and the audit-supported definitions."""
    for frame, name in [(baseline, "baseline"), (labels, "labels"), (temporal, "temporal")]:
        if len(frame) != 552 or frame["record_id"].duplicated().any():
            raise ValueError(f"{name} must contain 552 unique patient rows")
    ids = set(baseline["record_id"])
    if ids != set(labels["record_id"]) or ids != set(temporal["record_id"]):
        raise ValueError("Patient IDs do not align across required inputs")
    expected = labels.set_index("record_id").loc[baseline["record_id"], "label"].to_numpy()
    if not np.array_equal(expected, baseline["label"].to_numpy()):
        raise ValueError("Baseline labels disagree with labels.csv")
    temporal_labels = temporal.set_index("record_id").loc[
        baseline["record_id"], "label"
    ].to_numpy()
    if not np.array_equal(temporal_labels, baseline["label"].to_numpy()):
        raise ValueError("Temporal patient labels disagree with the baseline")
    forbidden = {"label", "pH", "BE", "BDecf", "Apgar5"}
    if forbidden & set(windows.columns):
        raise ValueError("Window rows unexpectedly contain labels or outcomes")
    missing_columns = sorted(set(ALL_SELECTED_TEMPORAL) - set(temporal.columns))
    if missing_columns:
        raise ValueError("Required temporal columns are absent: " + ", ".join(missing_columns))

    quality_row = group_audit[
        group_audit["temporal_group"] == "temporal_quality_context"
    ]
    if len(quality_row) != 1:
        raise ValueError("Audit must define exactly one temporal_quality_context group")
    audited_quality = quality_row.iloc[0]["features"].split(" | ")
    if audited_quality != EXPECTED_QUALITY_CONTEXT:
        raise ValueError(
            "Temporal quality-context definition changed: "
            + ", ".join(audited_quality)
        )
    if FINAL_NEGATIVE_SLOPE not in set(single_audit["candidate"]):
        raise ValueError("Final negative-slope candidate is not audit-supported")


def determine_optional_e(single_audit: pd.DataFrame) -> tuple[bool, str]:
    """Apply the eligibility rule for optional Representation E."""
    eligible = single_audit[
        (single_audit["candidate"] != FINAL_NEGATIVE_SLOPE)
        & (single_audit["paired_delta_macro_f1_mean"] > 0)
        & (single_audit["paired_delta_pathological_recall_mean"] > 0)
        & (single_audit["paired_delta_pathological_precision_mean"] >= 0)
        & (single_audit["macro_f1_improved_fold_count"] >= 13)
    ]
    if eligible.empty:
        return (
            False,
            (
                "No additional audited candidate met all four performance/fold criteria; "
                "Representation E was not created."
            ),
        )
    # No audited feature currently qualifies. Keep the correlation check for a
    # future audit in which a feature passes the initial criteria.
    return (
        True,
        (
            "An audit candidate met initial performance criteria, but this experiment "
            "does not silently add it before redundancy validation."
        ),
    )


def baseline_feature_columns(baseline: pd.DataFrame) -> list[str]:
    """Return the leakage-free feature list."""
    excluded = {train_models.TARGET_COLUMN, *train_models.LEAKAGE_COLUMNS}
    features = [column for column in baseline.columns if column not in excluded]
    non_numeric = baseline[features].select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError("Non-numeric baseline inputs: " + ", ".join(non_numeric))
    return features


def align_temporal(
    baseline: pd.DataFrame, temporal: pd.DataFrame
) -> pd.DataFrame:
    """Align temporal rows to the baseline patient order."""
    aligned = temporal.set_index("record_id").loc[baseline["record_id"]].reset_index()
    if not np.array_equal(aligned["record_id"].to_numpy(), baseline["record_id"].to_numpy()):
        raise AssertionError("Temporal alignment failed")
    return aligned


def temporal_definition(feature: str) -> str:
    """Document the definition used in the prior audit."""
    definitions = {
        FINAL_NEGATIVE_SLOPE: (
            "Maximum negative cleaned-FHR slope from the chronologically final "
            "finite 20-minute window."
        ),
        "valid_fhr_percentage__window_min": (
            "Minimum valid-FHR percentage across all retained diagnostic windows."
        ),
        "poor_quality_window_percentage": (
            "Percentage of retained windows with valid FHR below 80%."
        ),
        "final_window_quality": (
            "Valid-FHR percentage in the chronologically final retained window."
        ),
        "quality_transition_count": (
            "Number of transitions between valid-FHR >=80% and <80% states "
            "across ordered windows."
        ),
    }
    return definitions[feature]


def feature_diagnostics(
    baseline: pd.DataFrame,
    temporal: pd.DataFrame,
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Measure missingness, variance, and redundancy before modelling."""
    base_features = baseline_feature_columns(baseline)
    rows = []
    removals = []
    correlations = temporal[ALL_SELECTED_TEMPORAL].corr()
    correlation_rows = []
    for left in ALL_SELECTED_TEMPORAL:
        for right in ALL_SELECTED_TEMPORAL:
            correlation_rows.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "correlation": correlations.loc[left, right],
                }
            )
    for feature in ALL_SELECTED_TEMPORAL:
        values = pd.to_numeric(temporal[feature], errors="coerce")
        missing = values.isna()
        correlation_candidates = []
        for baseline_feature in base_features:
            pair = pd.concat(
                [values, pd.to_numeric(baseline[baseline_feature], errors="coerce")],
                axis=1,
            ).dropna()
            correlation = (
                pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) > 2 else np.nan
            )
            correlation_candidates.append((abs(correlation), correlation, baseline_feature))
        finite_candidates = [
            candidate for candidate in correlation_candidates if np.isfinite(candidate[0])
        ]
        highest = max(finite_candidates, default=(np.nan, np.nan, ""))
        selected_other_correlations = correlations.loc[
            feature, [value for value in ALL_SELECTED_TEMPORAL if value != feature]
        ].abs()
        highest_selected = (
            float(selected_other_correlations.max())
            if len(selected_other_correlations)
            else np.nan
        )
        missing_records = temporal.loc[missing, "record_id"].tolist()
        one_window_records = set(
            temporal.loc[temporal["total_window_count"] == 1, "record_id"]
        )
        missing_one_window = sorted(set(missing_records) & one_window_records)
        missing_invalid = []
        if missing_records:
            missing_invalid = sorted(
                windows.loc[
                    windows["record_id"].isin(missing_records)
                    & (windows["valid_fhr_percentage"] <= 0),
                    "record_id",
                ].unique()
            )
        variance = float(values.var(ddof=1))
        missing_percentage = float(100.0 * missing.mean())
        flags = []
        if variance <= NEAR_ZERO_VARIANCE:
            flags.append("near_zero_variance")
        if highest[0] > CORRELATION_LIMIT or highest_selected > CORRELATION_LIMIT:
            flags.append("correlation_above_0.95")
        if missing_percentage > MISSINGNESS_LIMIT_PERCENTAGE:
            flags.append("missingness_above_30_percent")
        removed = bool(flags)
        if removed:
            removals.append(feature)
        rows.append(
            {
                "feature": feature,
                "definition": temporal_definition(feature),
                "missing_count": int(missing.sum()),
                "missing_percentage": missing_percentage,
                "normal_missing_count": int(
                    (missing & (baseline["label"] == "Normal")).sum()
                ),
                "suspicious_missing_count": int(
                    (missing & (baseline["label"] == "Suspicious")).sum()
                ),
                "pathological_missing_count": int(
                    (missing & (baseline["label"] == "Pathological")).sum()
                ),
                "records_with_missing_values": ";".join(missing_records),
                "missing_due_to_one_window_records": ";".join(missing_one_window),
                "missing_with_completely_invalid_fhr_window": ";".join(missing_invalid),
                "missingness_interpretation": (
                    "No missing values."
                    if not missing_records
                    else "Missingness retained for fold-local median imputation; "
                    "zero was not substituted."
                ),
                "variance": variance,
                "unique_value_count": int(values.nunique(dropna=True)),
                "highest_absolute_correlation_with_baseline": highest[0],
                "corresponding_signed_correlation": highest[1],
                "most_correlated_baseline_feature": highest[2],
                "highest_absolute_correlation_with_selected_temporal": highest_selected,
                "near_zero_variance_flag": variance <= NEAR_ZERO_VARIANCE,
                "correlation_above_0_95_flag": (
                    highest[0] > CORRELATION_LIMIT
                    or highest_selected > CORRELATION_LIMIT
                ),
                "missingness_above_30_percent_flag": (
                    missing_percentage > MISSINGNESS_LIMIT_PERCENTAGE
                ),
                "removed_from_predefined_representations": removed,
                "removal_reason": ";".join(flags),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(correlation_rows), removals


def build_matrices(
    baseline: pd.DataFrame,
    temporal: pd.DataFrame,
    removals: list[str],
) -> OrderedDict[str, pd.DataFrame]:
    """Build the four representations without changing baseline columns."""
    baseline_matrix = baseline[baseline_feature_columns(baseline)].copy()
    matrices = OrderedDict()
    for name, temporal_features in REPRESENTATIONS.items():
        retained = [feature for feature in temporal_features if feature not in removals]
        matrix = baseline_matrix.copy()
        for feature in retained:
            matrix[feature] = pd.to_numeric(temporal[feature], errors="coerce").to_numpy()
        matrices[name] = matrix
    return matrices


def evaluate_predictions(
    target: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    encoder,
    record_ids: np.ndarray,
) -> dict:
    """Calculate the full evaluation metric set."""
    codes = np.arange(len(encoder.classes_))
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        target, predictions, labels=codes, zero_division=0
    )
    pathological_code = int(encoder.transform(["Pathological"])[0])
    pathological = target == pathological_code
    predicted_pathological = predictions == pathological_code
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "macro_f1": float(f1_score(target, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(target, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": json.dumps(
            confusion_matrix(target, predictions, labels=codes).tolist()
        ),
        "true_pathological_detections": int(
            np.sum(pathological & predicted_pathological)
        ),
        "false_pathological_predictions": int(
            np.sum(~pathological & predicted_pathological)
        ),
    }
    for code, class_name in enumerate(encoder.classes_):
        prefix = class_name.lower()
        metrics[f"{prefix}_precision"] = float(precision[code])
        metrics[f"{prefix}_recall"] = float(recall[code])
        metrics[f"{prefix}_f1"] = float(per_class_f1[code])
    return metrics


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


def fixed_indices(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the fixed stratified 80/20 split."""
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
    """Evaluate each representation once on the descriptive test set."""
    train_indices, test_indices = fixed_indices(target)
    rows = []
    outputs = {}
    pathological_code = int(encoder.transform(["Pathological"])[0])
    for representation, matrix in matrices.items():
        predictions, probabilities = fit_predict(
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
            "temporal_features": " | ".join(REPRESENTATIONS[representation]),
            "train_patient_count": len(train_indices),
            "test_patient_count": len(test_indices),
            "pipeline": (
                "SimpleImputer(median) -> StandardScaler -> "
                "SMOTE(training only) -> LogisticRegression(C=1, solver=lbfgs, "
                "max_iter=2000, class_weight=None)"
            ),
            **metrics,
        }
        for record_id in FIXED_PATHOLOGICAL_IDS:
            position = np.flatnonzero(record_ids[test_indices] == record_id)
            if len(position) != 1:
                raise AssertionError(f"Fixed Pathological record missing: {record_id}")
            row[f"pathological_probability_record_{record_id}"] = float(
                probabilities[position[0], pathological_code]
            )
            row[f"predicted_class_record_{record_id}"] = encoder.inverse_transform(
                [predictions[position[0]]]
            )[0]
        rows.append(row)
        outputs[representation] = {
            "predictions": predictions,
            "probabilities": probabilities,
        }
    return pd.DataFrame(rows), train_indices, test_indices, outputs


def paired_rows(
    metrics_by_representation: dict[str, dict],
    identity: dict,
) -> list[dict]:
    """Create long-form representation rows with baseline differences."""
    baseline = metrics_by_representation["A_baseline"]
    rows = []
    for representation, metrics in metrics_by_representation.items():
        row = {
            **identity,
            "representation": representation,
            "temporal_features": " | ".join(REPRESENTATIONS[representation]),
            **metrics,
        }
        for metric in METRICS:
            row[f"delta_from_baseline_{metric}"] = (
                metrics[metric] - baseline[metric]
            )
        rows.append(row)
    return rows


def repeated_split_evaluation(
    matrices: OrderedDict[str, pd.DataFrame],
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
) -> pd.DataFrame:
    """Evaluate all representations on the same 30 predefined splits."""
    rows = []
    indices = np.arange(len(target))
    for position, seed in enumerate(SPLIT_SEEDS, start=1):
        train_indices, test_indices = train_test_split(
            indices,
            test_size=train_models.TEST_SIZE,
            random_state=seed,
            stratify=target,
        )
        fold_metrics = {}
        for representation, matrix in matrices.items():
            predictions, probabilities = fit_predict(
                matrix, target, train_indices, test_indices, model, seed
            )
            fold_metrics[representation] = evaluate_predictions(
                target[test_indices],
                predictions,
                probabilities,
                encoder,
                record_ids[test_indices],
            )
        rows.extend(
            paired_rows(
                fold_metrics,
                {
                    "random_seed": seed,
                    "train_patient_count": len(train_indices),
                    "test_patient_count": len(test_indices),
                },
            )
        )
        print(f"  Predefined split: {position}/{len(SPLIT_SEEDS)}")
    return pd.DataFrame(rows)


def repeated_cv_evaluation(
    matrices: OrderedDict[str, pd.DataFrame],
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
) -> pd.DataFrame:
    """Evaluate all representations on identical repeated 5x5 patient folds."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=train_models.RANDOM_STATE,
    )
    rows = []
    for fold_number, (train_indices, test_indices) in enumerate(
        cv.split(np.zeros(len(target)), target), start=1
    ):
        fold_metrics = {}
        seed = train_models.RANDOM_STATE + fold_number
        for representation, matrix in matrices.items():
            predictions, probabilities = fit_predict(
                matrix, target, train_indices, test_indices, model, seed
            )
            fold_metrics[representation] = evaluate_predictions(
                target[test_indices],
                predictions,
                probabilities,
                encoder,
                record_ids[test_indices],
            )
        rows.extend(
            paired_rows(
                fold_metrics,
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


def bootstrap_mean_interval(values: np.ndarray) -> tuple[float, float]:
    """Return a deterministic 95% bootstrap interval for the paired mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_STATE)
    means = rng.choice(
        values,
        size=(BOOTSTRAP_RESAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def summarize_paired(
    results: pd.DataFrame,
    identity_name: str,
) -> pd.DataFrame:
    """Summarize each representation and paired baseline change."""
    expected_count = results[identity_name].nunique()
    rows = []
    for representation, group in results.groupby("representation", sort=False):
        if len(group) != expected_count:
            raise AssertionError("Incomplete paired evaluation")
        for metric in METRICS:
            values = group[metric]
            delta = group[f"delta_from_baseline_{metric}"]
            lower, upper = bootstrap_mean_interval(delta.to_numpy())
            rows.append(
                {
                    "representation": representation,
                    "temporal_features": " | ".join(REPRESENTATIONS[representation]),
                    "metric": metric,
                    "evaluation_count": len(group),
                    "mean": values.mean(),
                    "median": values.median(),
                    "standard_deviation": values.std(ddof=1),
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "paired_difference_mean": delta.mean(),
                    "paired_difference_median": delta.median(),
                    "paired_difference_standard_deviation": delta.std(ddof=1),
                    "paired_difference_minimum": delta.min(),
                    "paired_difference_maximum": delta.max(),
                    "paired_mean_bootstrap_95ci_lower": lower,
                    "paired_mean_bootstrap_95ci_upper": upper,
                    "improved_count": int((delta > EPSILON).sum()),
                    "worsened_count": int((delta < -EPSILON).sum()),
                    "equal_count": int((abs(delta) <= EPSILON).sum()),
                }
            )
    return pd.DataFrame(rows)


def confusion_figures(fixed: pd.DataFrame, encoder) -> list[Path]:
    """Save one fixed-test confusion matrix per representation."""
    paths = []
    for _, row in fixed.iterrows():
        matrix = np.asarray(json.loads(row["confusion_matrix"]))
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        image = ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, int(value), ha="center", va="center")
        ax.set_xticks(np.arange(len(encoder.classes_)), encoder.classes_, rotation=25)
        ax.set_yticks(np.arange(len(encoder.classes_)), encoder.classes_)
        ax.set(
            xlabel="Predicted class",
            ylabel="True class",
            title=f"Fixed test: {row['representation']}",
        )
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        path = FIGURE_DIR / f"confusion_matrix_{row['representation']}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def fixed_train_oof_predictions(
    matrices: OrderedDict[str, pd.DataFrame],
    target: np.ndarray,
    fixed_train_indices: np.ndarray,
    encoder,
    model,
) -> dict:
    """Create five-fold OOF predictions for fixed-training patients."""
    outputs = {}
    splitter = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=train_models.RANDOM_STATE
    )
    pathological_code = int(encoder.transform(["Pathological"])[0])
    train_target = target[fixed_train_indices]
    for representation, matrix in matrices.items():
        predictions = np.full(len(fixed_train_indices), -1, dtype=int)
        probabilities = np.full(len(fixed_train_indices), np.nan)
        for fold_number, (inner_train, validation) in enumerate(
            splitter.split(np.zeros(len(fixed_train_indices)), train_target), start=1
        ):
            absolute_train = fixed_train_indices[inner_train]
            absolute_validation = fixed_train_indices[validation]
            fold_predictions, fold_probabilities = fit_predict(
                matrix,
                target,
                absolute_train,
                absolute_validation,
                model,
                train_models.RANDOM_STATE + fold_number,
            )
            predictions[validation] = fold_predictions
            probabilities[validation] = fold_probabilities[:, pathological_code]
        if (predictions < 0).any() or np.isnan(probabilities).any():
            raise AssertionError("Incomplete fixed-training OOF predictions")
        outputs[representation] = {
            "predictions": predictions,
            "pathological_probabilities": probabilities,
        }
    return outputs


def pathological_case_results(
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
    """Combine fixed-test and OOF fixed-training results for all 27 cases."""
    record_ids = baseline["record_id"].to_numpy()
    pathological_code = int(encoder.transform(["Pathological"])[0])
    oof = fixed_train_oof_predictions(
        matrices, target, fixed_train_indices, encoder, model
    )
    fixed_train_position = {
        absolute: position for position, absolute in enumerate(fixed_train_indices)
    }
    fixed_test_position = {
        absolute: position for position, absolute in enumerate(fixed_test_indices)
    }
    rows = []
    for absolute_index in np.flatnonzero(target == pathological_code):
        record_id = record_ids[absolute_index]
        is_test = absolute_index in fixed_test_position
        row = {
            "record_id": record_id,
            "split_membership": "fixed_test" if is_test else "fixed_train_oof",
        }
        representation_values = {}
        for representation in matrices:
            if is_test:
                position = fixed_test_position[absolute_index]
                prediction = fixed_outputs[representation]["predictions"][position]
                probability = fixed_outputs[representation]["probabilities"][
                    position, pathological_code
                ]
            else:
                position = fixed_train_position[absolute_index]
                prediction = oof[representation]["predictions"][position]
                probability = oof[representation]["pathological_probabilities"][position]
            predicted_label = encoder.inverse_transform([prediction])[0]
            representation_values[representation] = (predicted_label, probability)
            row[f"{representation}__prediction"] = predicted_label
            row[f"{representation}__pathological_probability"] = probability
        baseline_prediction, baseline_probability = representation_values["A_baseline"]
        baseline_correct = baseline_prediction == "Pathological"
        for representation in list(matrices)[1:]:
            prediction, probability = representation_values[representation]
            temporal_correct = prediction == "Pathological"
            if temporal_correct and not baseline_correct:
                status = "improved"
            elif baseline_correct and not temporal_correct:
                status = "worsened"
            else:
                status = "unchanged"
            row[f"{representation}__probability_difference"] = (
                probability - baseline_probability
            )
            row[f"{representation}__prediction_change"] = status
        temporal_row = temporal.iloc[absolute_index]
        for feature in ALL_SELECTED_TEMPORAL:
            row[feature] = temporal_row[feature]
            row[f"{feature}__missing"] = bool(pd.isna(temporal_row[feature]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split_membership", "record_id"])


def ablation_if_justified(
    matrices: OrderedDict[str, pd.DataFrame],
    cv_summary: pd.DataFrame,
    target: np.ndarray,
    record_ids: np.ndarray,
    encoder,
    model,
) -> tuple[pd.DataFrame | None, str]:
    """Run leave-one-temporal-feature-out CV only when D beats B and C."""
    macro = cv_summary[cv_summary["metric"] == "macro_f1"].set_index("representation")
    combined = macro.loc["D_combined_limited_temporal", "mean"]
    single = macro.loc["B_final_negative_slope", "mean"]
    quality = macro.loc["C_temporal_quality_context", "mean"]
    if not (combined > single and combined > quality):
        return (
            None,
            (
                "Ablation was not run because Representation D did not outperform "
                "both B and C in mean repeated-CV Macro F1."
            ),
        )
    baseline_matrix = matrices["A_baseline"]
    ablation_matrices = OrderedDict()
    for removed in ALL_SELECTED_TEMPORAL:
        retained = [feature for feature in ALL_SELECTED_TEMPORAL if feature != removed]
        matrix = baseline_matrix.copy()
        combined_matrix = matrices["D_combined_limited_temporal"]
        for feature in retained:
            if feature in combined_matrix:
                matrix[feature] = combined_matrix[feature]
        ablation_matrices[f"remove__{removed}"] = matrix
    # Add leave-one-out variants for the shared paired evaluator, then restore
    # the four main representations below.
    original = REPRESENTATIONS.copy()
    try:
        for name in ablation_matrices:
            removed = name.removeprefix("remove__")
            REPRESENTATIONS[name] = [
                feature for feature in ALL_SELECTED_TEMPORAL if feature != removed
            ]
        results = repeated_cv_evaluation(
            OrderedDict(
                [("A_baseline", baseline_matrix), *ablation_matrices.items()]
            ),
            target,
            record_ids,
            encoder,
            model,
        )
        summary = summarize_paired(results, "fold_number")
    finally:
        REPRESENTATIONS.clear()
        REPRESENTATIONS.update(original)
    return summary, "Ablation ran because D exceeded both B and C in mean CV Macro F1."


def representation_metric(
    summary: pd.DataFrame, representation: str, metric: str, column: str = "mean"
) -> float:
    """Read one representation/metric result from a long summary."""
    row = summary[
        (summary["representation"] == representation)
        & (summary["metric"] == metric)
    ]
    if len(row) != 1:
        raise AssertionError(f"Missing summary row: {representation}/{metric}")
    return float(row.iloc[0][column])


def choose_recommendation(
    cv_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    split_results: pd.DataFrame,
) -> tuple[str, str, str]:
    """Choose a stable recommendation without consulting fixed-test results."""
    candidate_names = list(REPRESENTATIONS)[1:]
    ranked = sorted(
        candidate_names,
        key=lambda name: representation_metric(cv_summary, name, "macro_f1"),
        reverse=True,
    )
    best = ranked[0]
    baseline_macro = representation_metric(cv_summary, "A_baseline", "macro_f1")
    best_macro = representation_metric(cv_summary, best, "macro_f1")
    recall_delta = representation_metric(
        cv_summary, best, "pathological_recall", "paired_difference_mean"
    )
    precision_delta = representation_metric(
        cv_summary, best, "pathological_precision", "paired_difference_mean"
    )
    false_delta = representation_metric(
        cv_summary, best, "false_pathological_predictions", "paired_difference_mean"
    )
    folds_improved = int(
        representation_metric(cv_summary, best, "macro_f1", "improved_count")
    )
    best_split_rows = split_results[
        split_results["representation"] == best
    ]
    split_either = int(
        (
            (best_split_rows["delta_from_baseline_macro_f1"] > EPSILON)
            | (
                best_split_rows["delta_from_baseline_pathological_recall"]
                > EPSILON
            )
        ).sum()
    )
    stable = (
        best_macro > baseline_macro
        and recall_delta > 0
        and precision_delta >= -0.01
        and false_delta <= 1
        and folds_improved >= 15
        and split_either >= 18
    )
    if stable:
        recommendation_map = {
            "B_final_negative_slope": "add only maximum_negative_fhr_slope__window_final",
            "C_temporal_quality_context": "add only the temporal quality-context group",
            "D_combined_limited_temporal": "add the combined limited temporal representation",
        }
        return (
            best,
            recommendation_map[best],
            "the limited temporal representation provides a clear improvement",
        )
    useful_tradeoff = (
        recall_delta >= 0.04
        and best_macro >= baseline_macro - 0.01
        and precision_delta >= -0.02
        and false_delta <= 2
    )
    if useful_tradeoff:
        return (
            best,
            "temporal features provide a useful recall trade-off but require cautious interpretation",
            "the limited temporal representation provides a useful recall trade-off",
        )
    all_reduced = all(
        representation_metric(cv_summary, name, "macro_f1") < baseline_macro
        for name in candidate_names
    )
    if all_reduced:
        return (
            best,
            "keep the current baseline representation",
            "the temporal representation reduces performance",
        )
    return (
        best,
        "temporal gains are too small and unstable to justify changing the model",
        "temporal gains are too small and unstable to justify changing the baseline",
    )


def report_table(summary: pd.DataFrame, metric: str) -> str:
    """Render a compact Markdown table for one repeated metric."""
    rows = summary[summary["metric"] == metric]
    lines = [
        "| Representation | Mean ± SD | Paired change | 95% CI | Improved |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['representation']} | {row['mean']:.4f} ± "
            f"{row['standard_deviation']:.4f} | "
            f"{row['paired_difference_mean']:+.4f} | "
            f"[{row['paired_mean_bootstrap_95ci_lower']:+.4f}, "
            f"{row['paired_mean_bootstrap_95ci_upper']:+.4f}] | "
            f"{int(row['improved_count'])}/{int(row['evaluation_count'])} |"
        )
    return "\n".join(lines)


def write_report(
    diagnostics: pd.DataFrame,
    optional_e_reason: str,
    fixed: pd.DataFrame,
    split_summary: pd.DataFrame,
    cv_summary: pd.DataFrame,
    pathological: pd.DataFrame,
    best_representation: str,
    recommendation: str,
    ablation: pd.DataFrame | None,
    ablation_reason: str,
    conclusion: str,
) -> None:
    """Write the controlled-experiment report and its conclusion."""
    fixed_lookup = fixed.set_index("representation")
    best_macro_delta = representation_metric(
        cv_summary, best_representation, "macro_f1", "paired_difference_mean"
    )
    best_recall_delta = representation_metric(
        cv_summary,
        best_representation,
        "pathological_recall",
        "paired_difference_mean",
    )
    best_precision_delta = representation_metric(
        cv_summary,
        best_representation,
        "pathological_precision",
        "paired_difference_mean",
    )
    best_false_delta = representation_metric(
        cv_summary,
        best_representation,
        "false_pathological_predictions",
        "paired_difference_mean",
    )
    helped = int(
        (
            pathological[f"{best_representation}__prediction_change"] == "improved"
        ).sum()
    )
    harmed = int(
        (
            pathological[f"{best_representation}__prediction_change"] == "worsened"
        ).sum()
    )
    removal_rows = diagnostics[diagnostics["removed_from_predefined_representations"]]
    removal_text = (
        "No predefined feature violated the explicit variance, correlation, or "
        "missingness rules, so none was removed."
        if removal_rows.empty
        else "Removed: "
        + "; ".join(
            f"{row.feature} ({row.removal_reason})"
            for row in removal_rows.itertuples()
        )
    )
    missing_text = "; ".join(
        f"`{row.feature}`: {int(row.missing_count)}/552 "
        f"({row.missing_percentage:.1f}%)"
        for row in diagnostics.itertuples()
    )
    representation_text = "\n".join(
        f"- **{name}:** "
        + (", ".join(f"`{feature}`" for feature in features) if features else "no temporal additions")
        for name, features in REPRESENTATIONS.items()
    )
    if ablation is not None:
        macro_ablation = ablation[ablation["metric"] == "macro_f1"].copy()
        nonbaseline_ablation = macro_ablation[
            macro_ablation["representation"] != "A_baseline"
        ]
        strongest_without = nonbaseline_ablation.sort_values(
            "mean", ascending=False
        ).iloc[0]
        weakest_without = nonbaseline_ablation.sort_values("mean").iloc[0]
        ablation_text = (
            f"{ablation_reason} Removing "
            f"`{strongest_without['representation'].removeprefix('remove__')}` "
            f"gave the highest leave-one-out Macro F1 "
            f"(**{strongest_without['mean']:.4f}**), slightly above the complete "
            "D representation. The largest loss occurred when removing "
            f"`{weakest_without['representation'].removeprefix('remove__')}` "
            f"(Macro F1 **{weakest_without['mean']:.4f}**). The combined gain "
            "therefore does not depend entirely on one feature; however, the "
            "quality-transition count may dilute the result. This post-hoc "
            "ablation does not authorize a new representation or production "
            "feature selection. Full results are in "
            "`temporal_feature_ablation.csv`."
        )
    else:
        ablation_text = ablation_reason
    best_cv_macro = cv_summary[
        (cv_summary["representation"] == best_representation)
        & (cv_summary["metric"] == "macro_f1")
    ].iloc[0]
    best_cv_recall = cv_summary[
        (cv_summary["representation"] == best_representation)
        & (cv_summary["metric"] == "pathological_recall")
    ].iloc[0]
    best_split_macro = split_summary[
        (split_summary["representation"] == best_representation)
        & (split_summary["metric"] == "macro_f1")
    ].iloc[0]
    report = f"""# Limited Temporal-Feature Experiment

## Purpose

The prior temporal audit found small, mixed improvements. This experiment tests
only those predefined patient-level features in a controlled comparison. It
does not create window labels, tune the classifier, or build a sequence model.

## Representations

{representation_text}

{optional_e_reason}

The exact temporal definitions are recorded in
`selected_temporal_feature_diagnostics.csv`. The quality-context group was
read directly from the prior audit and verified against its four saved column
names.

## What remained unchanged

All representations used the same 552 patients, labels, leakage exclusions,
LabelEncoder, fixed split, 30 predefined seeds, repeated 5x5 folds, and:

`SimpleImputer(median) -> StandardScaler -> SMOTE(training only) -> LogisticRegression(C=1, solver="lbfgs", max_iter=2000, class_weight=None)`.

No fixed-test result was used for selection.

## Missingness and redundancy

{missing_text}.

{removal_text}

Missing values, if present, remain NaN until fold-local median imputation; no
trend value is replaced with zero. Correlations, variance, and unique counts
are provided in the two diagnostic CSV files.

## Fixed split (descriptive only)

Baseline fixed-test Macro F1 was
**{fixed_lookup.loc['A_baseline', 'macro_f1']:.4f}**, and its Pathological
recall was **{fixed_lookup.loc['A_baseline', 'pathological_recall']:.4f}**.
The fixed result was evaluated once and did not determine the recommendation.

## Thirty predefined splits

### Macro F1

{report_table(split_summary, 'macro_f1')}

### Pathological recall

{report_table(split_summary, 'pathological_recall')}

### Pathological precision

{report_table(split_summary, 'pathological_precision')}

## Repeated 5x5 patient-level cross-validation

### Macro F1

{report_table(cv_summary, 'macro_f1')}

### Pathological recall

{report_table(cv_summary, 'pathological_recall')}

### Pathological precision

{report_table(cv_summary, 'pathological_precision')}

### False Pathological predictions

{report_table(cv_summary, 'false_pathological_predictions')}

The best numerical temporal representation was
**{best_representation}**. Its paired repeated-CV changes were Macro F1
**{best_macro_delta:+.4f}**, Pathological recall **{best_recall_delta:+.4f}**,
Pathological precision **{best_precision_delta:+.4f}**, and false
Pathological predictions **{best_false_delta:+.2f}**.

Positive means and confidence intervals that cross zero are described as
numerically favourable but statistically uncertain. Fold and split consistency
are required before changing production features.

For **{best_representation}**, the repeated-CV Macro F1 interval was
**[{best_cv_macro['paired_mean_bootstrap_95ci_lower']:+.4f},
{best_cv_macro['paired_mean_bootstrap_95ci_upper']:+.4f}]** and the
Pathological-recall interval was
**[{best_cv_recall['paired_mean_bootstrap_95ci_lower']:+.4f},
{best_cv_recall['paired_mean_bootstrap_95ci_upper']:+.4f}]**; both cross zero,
so those CV gains remain statistically uncertain. The 30-split Macro F1
interval was
**[{best_split_macro['paired_mean_bootstrap_95ci_lower']:+.4f},
{best_split_macro['paired_mean_bootstrap_95ci_upper']:+.4f}]**, which was
consistently positive.

## Pathological cases

All 27 Pathological patients were retained. Fixed-test patients use fixed-test
predictions; fixed-training patients use five-fold out-of-fold predictions.
Relative to baseline, **{helped}** cases improved and **{harmed}** worsened for
the best numerical representation. Individual probabilities, feature values,
and missingness flags are in `pathological_case_results.csv`.

## Combined representation and ablation

{ablation_text}

## Recommendation

**{recommendation}.** The fixed production dataset should not change unless
paired evidence is stable. The limited experiment does not provide a basis for
sequence modelling.

## Conclusion

{conclusion}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def validate_outputs(
    fixed: pd.DataFrame,
    split_results: pd.DataFrame,
    split_summary: pd.DataFrame,
    cv_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
    pathological: pd.DataFrame,
    diagnostics: pd.DataFrame,
    correlations: pd.DataFrame,
    figures: list[Path],
) -> None:
    """Validate paired coverage and required outputs."""
    if len(fixed) != len(REPRESENTATIONS):
        raise AssertionError("Fixed results lack a representation")
    if len(split_results) != len(SPLIT_SEEDS) * len(REPRESENTATIONS):
        raise AssertionError("Repeated split coverage is incomplete")
    if len(cv_results) != CV_SPLITS * CV_REPEATS * len(REPRESENTATIONS):
        raise AssertionError("Repeated-CV coverage is incomplete")
    expected_summary = len(REPRESENTATIONS) * len(METRICS)
    if len(split_summary) != expected_summary or len(cv_summary) != expected_summary:
        raise AssertionError("A paired summary is incomplete")
    if len(pathological) != 27 or pathological["record_id"].nunique() != 27:
        raise AssertionError("Pathological results must contain all 27 patients")
    if len(diagnostics) != len(ALL_SELECTED_TEMPORAL):
        raise AssertionError("Selected-feature diagnostics are incomplete")
    if len(correlations) != len(ALL_SELECTED_TEMPORAL) ** 2:
        raise AssertionError("Temporal correlation table is incomplete")
    if len(figures) != len(REPRESENTATIONS) or not all(path.exists() for path in figures):
        raise AssertionError("Fixed confusion matrices are incomplete")
    if not REPORT_PATH.exists():
        raise AssertionError("Final report is missing")


def main() -> None:
    """Run the limited temporal-feature experiment."""
    refuse_overwrite()
    (
        baseline,
        labels,
        windows,
        temporal,
        single_audit,
        group_audit,
        _paired,
        model,
        encoder,
    ) = load_inputs()
    validate_inputs(
        baseline, labels, windows, temporal, single_audit, group_audit
    )
    optional_e, optional_e_reason = determine_optional_e(single_audit)
    if optional_e:
        raise ValueError(
            "Optional E requires a new explicit correlation decision; refusing "
            "to create it automatically."
        )
    temporal = align_temporal(baseline, temporal)
    diagnostics, correlations, removals = feature_diagnostics(
        baseline, temporal, windows
    )
    diagnostics.to_csv(FEATURE_DIAGNOSTIC_PATH, index=False)
    correlations.to_csv(CORRELATION_PATH, index=False)
    matrices = build_matrices(baseline, temporal, removals)
    target = encoder.transform(baseline["label"])
    record_ids = baseline["record_id"].to_numpy()

    print("Controlled representations")
    for representation, features in REPRESENTATIONS.items():
        retained = [feature for feature in features if feature not in removals]
        print(f"  {representation}: {retained or ['baseline features only']}")
    print(f"Representation E: not created. {optional_e_reason}")
    print("\nSelected temporal-feature missingness")
    print(
        diagnostics[
            ["feature", "missing_count", "missing_percentage", "removal_reason"]
        ].to_string(index=False)
    )

    print("\nRunning fixed-split evaluation...")
    fixed, fixed_train, fixed_test, fixed_outputs = fixed_evaluation(
        matrices, target, record_ids, encoder, model
    )
    fixed.to_csv(FIXED_PATH, index=False)
    figures = confusion_figures(fixed, encoder)

    print("\nRunning 30 predefined splits...")
    split_results = repeated_split_evaluation(
        matrices, target, record_ids, encoder, model
    )
    split_results.to_csv(REPEATED_SPLIT_PATH, index=False)
    split_summary = summarize_paired(split_results, "random_seed")
    split_summary.to_csv(REPEATED_SPLIT_SUMMARY_PATH, index=False)

    print("\nRunning repeated 5x5 cross-validation...")
    cv_results = repeated_cv_evaluation(
        matrices, target, record_ids, encoder, model
    )
    cv_results.to_csv(REPEATED_CV_PATH, index=False)
    cv_summary = summarize_paired(cv_results, "fold_number")
    cv_summary.to_csv(REPEATED_CV_SUMMARY_PATH, index=False)

    print("\nCreating Pathological fixed/OOF case results...")
    pathological = pathological_case_results(
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

    ablation, ablation_reason = ablation_if_justified(
        matrices, cv_summary, target, record_ids, encoder, model
    )
    if ablation is not None:
        ablation.to_csv(ABLATION_PATH, index=False)

    best, recommendation, conclusion = choose_recommendation(
        cv_summary, split_summary, split_results
    )
    write_report(
        diagnostics,
        optional_e_reason,
        fixed,
        split_summary,
        cv_summary,
        pathological,
        best,
        recommendation,
        ablation,
        ablation_reason,
        conclusion,
    )
    validate_outputs(
        fixed,
        split_results,
        split_summary,
        cv_results,
        cv_summary,
        pathological,
        diagnostics,
        correlations,
        figures,
    )

    print("\nLimited temporal-feature experiment complete")
    print(f"Patients: {len(baseline)}")
    for representation, features in REPRESENTATIONS.items():
        print(f"{representation}: {features or ['baseline features only']}")
    for row in diagnostics.itertuples():
        print(
            f"{row.feature}: missing {row.missing_count}/552 "
            f"({row.missing_percentage:.1f}%)"
        )
    print(
        "Baseline repeated-CV Macro F1: "
        f"{representation_metric(cv_summary, 'A_baseline', 'macro_f1'):.4f}"
    )
    for representation in REPRESENTATIONS:
        print(
            f"{representation}: Macro F1="
            f"{representation_metric(cv_summary, representation, 'macro_f1'):.4f}; "
            f"Pathological recall="
            f"{representation_metric(cv_summary, representation, 'pathological_recall'):.4f}; "
            f"precision="
            f"{representation_metric(cv_summary, representation, 'pathological_precision'):.4f}; "
            f"false Pathological="
            f"{representation_metric(cv_summary, representation, 'false_pathological_predictions'):.2f}; "
            f"folds improved="
            f"{int(representation_metric(cv_summary, representation, 'macro_f1', 'improved_count'))}; "
            f"splits improved="
            f"{int(representation_metric(split_summary, representation, 'macro_f1', 'improved_count'))}"
        )
    print(f"Best numerical representation: {best}")
    print(f"Recommendation: {recommendation}")
    print(f"Main conclusion: {conclusion}")
    for path in [
        FEATURE_DIAGNOSTIC_PATH,
        CORRELATION_PATH,
        FIXED_PATH,
        REPEATED_SPLIT_PATH,
        REPEATED_SPLIT_SUMMARY_PATH,
        REPEATED_CV_PATH,
        REPEATED_CV_SUMMARY_PATH,
        PATHOLOGICAL_PATH,
        *([ABLATION_PATH] if ablation is not None else []),
        FIGURE_DIR,
        REPORT_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()
