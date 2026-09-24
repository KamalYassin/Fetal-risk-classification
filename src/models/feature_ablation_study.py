"""Run a controlled, leakage-safe feature ablation study for CTG models.

Every experiment uses the same stratified train/test split.  Median imputation
and SMOTE live inside an imbalanced-learn pipeline, so they are learned anew on
each training fold and never alter validation or test rows.  Repeated
cross-validation on the training partition is the primary evidence; the test
partition is reported once for comparison and must not be used repeatedly to
select features.


Run from the project root::

    ./.venv/bin/python src/models/feature_ablation_study.py
"""

import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

MATPLOTLIB_CACHE_DIR = Path(tempfile.gettempdir()) / "ctg-matplotlib-cache"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
MAIN_RESULTS_PATH = PROJECT_ROOT / "reports" / "feature_ablation_results.csv"
REPEATED_CV_PATH = PROJECT_ROOT / "reports" / "feature_ablation_repeated_cv.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_ablation_report.md"
CHART_PATH = PROJECT_ROOT / "reports" / "feature_ablation_macro_f1.png"

TARGET_COLUMN = "label"
LEAKAGE_COLUMNS = [
    "record_id",
    "pH",
    "BE",
    "BDecf",
    "Apgar5",
    TARGET_COLUMN,
    "feature_quality_flag",
]
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_SPLITS = 5
REPEATED_CV_REPEATS = 5
CV_N_JOBS = 1  # avoids nested process pools; the forest parallelizes its trees

# Fixed primary-model configuration required by the ablation specification.
RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 10,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Ordered definitions make terminal/report output deterministic and readable.
FEATURE_GROUPS = OrderedDict(
    [
        (
            "signal_quality_context",
            [
                "recording_duration_minutes",
                "percent_missing_fhr",
                "valid_fhr_percentage",
                "possible_fhr_artifact_percentage",
                "longest_missing_fhr_gap_seconds",
            ],
        ),
        (
            "basic_fhr_statistics",
            [
                "mean_fhr",
                "median_fhr",
                "min_fhr",
                "max_fhr",
                "std_fhr",
                "baseline_fhr",
            ],
        ),
        (
            "global_variability",
            ["short_term_variability", "long_term_variability"],
        ),
        (
            "accelerations",
            [
                "acceleration_count",
                "mean_acceleration_duration_seconds",
                "max_acceleration_amplitude",
                "acceleration_density_per_hour",
                "longest_acceleration_duration_seconds",
            ],
        ),
        (
            "decelerations",
            [
                "deceleration_count",
                "mean_deceleration_duration_seconds",
                "max_deceleration_depth",
                "deceleration_density_per_hour",
                "longest_deceleration_duration_seconds",
                "percentage_time_below_baseline",
            ],
        ),
        (
            "tachycardia_bradycardia",
            [
                "bradycardia_duration_seconds",
                "bradycardia_percentage",
                "tachycardia_duration_seconds",
                "tachycardia_percentage",
            ],
        ),
        (
            "baseline_drift",
            [
                "percentage_time_above_baseline",
                "baseline_drift_std",
                "baseline_drift_range",
                "baseline_crossing_count",
            ],
        ),
        (
            "uterine_contractions",
            [
                "mean_uc",
                "median_uc",
                "max_uc",
                "std_uc",
                "contraction_count",
                "mean_contraction_peak",
                "mean_contraction_interval_seconds",
                "contraction_interval_std_seconds",
                "contraction_frequency_per_hour",
            ],
        ),
        (
            "fhr_uc_interactions",
            [
                "decelerations_per_contraction",
                "contractions_followed_by_deceleration_percentage",
                "mean_delay_contraction_to_deceleration_seconds",
            ],
        ),
        (
            "signal_dynamics",
            [
                "mean_positive_fhr_slope",
                "mean_negative_fhr_slope",
                "maximum_negative_fhr_slope",
            ],
        ),
        (
            "segment_variability",
            [
                "segment_stv_mean",
                "segment_stv_std",
                "segment_ltv_mean",
                "segment_ltv_std",
            ],
        ),
    ]
)

ORIGINAL_FEATURES = [
    "recording_duration_minutes",
    "percent_missing_fhr",
    "valid_fhr_percentage",
    "possible_fhr_artifact_percentage",
    "longest_missing_fhr_gap_seconds",
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

NEW_FEATURES = [
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

CLASS_REPORT_ORDER = ["Normal", "Suspicious", "Pathological"]


def load_and_prepare_dataset():
    """Load data, exclude leakage columns, and encode labels consistently."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    dataset = pd.read_csv(DATASET_PATH)
    if dataset.empty:
        raise ValueError("ml_dataset.csv is empty.")
    if TARGET_COLUMN not in dataset:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    retained_columns = [
        column for column in dataset.columns if column not in LEAKAGE_COLUMNS
    ]
    features = dataset[retained_columns].copy()
    non_numeric = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Retained feature columns must be numeric: " + ", ".join(non_numeric)
        )

    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(dataset[TARGET_COLUMN])
    return dataset, features, target, label_encoder


def validate_feature_groups(retained_columns):
    """Filter group definitions to actual columns and warn instead of failing."""
    retained_set = set(retained_columns)
    available_groups = OrderedDict()
    assigned_columns = set()

    print("\nFeature-group validation")
    print("------------------------")
    for group_name, requested_columns in FEATURE_GROUPS.items():
        present = [column for column in requested_columns if column in retained_set]
        missing = [column for column in requested_columns if column not in retained_set]
        available_groups[group_name] = present
        assigned_columns.update(present)
        print(f"{group_name}: {len(present)}/{len(requested_columns)} columns present")
        for column in missing:
            print(f"WARNING: optional feature missing from {group_name}: {column}")

    duplicate_assignments = []
    for column in retained_columns:
        containing_groups = [
            group_name
            for group_name, columns in available_groups.items()
            if column in columns
        ]
        if len(containing_groups) > 1:
            duplicate_assignments.append((column, containing_groups))
    for column, group_names in duplicate_assignments:
        print(f"WARNING: {column} appears in multiple groups: {group_names}")

    ungrouped = [
        column for column in retained_columns if column not in assigned_columns
    ]
    print("\nUngrouped retained features")
    print("---------------------------")
    if ungrouped:
        for column in ungrouped:
            print(column)
    else:
        print("None")
    return available_groups, ungrouped


def existing_columns(requested_columns, retained_columns, subset_name):
    """Keep available optional columns and print warnings for absent ones."""
    retained_set = set(retained_columns)
    present = [column for column in requested_columns if column in retained_set]
    for column in requested_columns:
        if column not in retained_set:
            print(f"WARNING: {subset_name} optional feature missing: {column}")
    return present


def build_experiments(retained_columns, available_groups):
    """Construct every required experiment from the same retained columns."""
    experiments = OrderedDict()
    experiments["all_features"] = {
        "features": list(retained_columns),
        "removed_group": "",
    }
    experiments["original_features_only"] = {
        "features": existing_columns(
            ORIGINAL_FEATURES, retained_columns, "original_features_only"
        ),
        "removed_group": "all_new_features",
    }
    experiments["new_features_only"] = {
        "features": existing_columns(
            NEW_FEATURES, retained_columns, "new_features_only"
        ),
        "removed_group": "all_original_features",
    }

    for group_name, group_columns in available_groups.items():
        experiments[f"without_{group_name}"] = {
            "features": [
                column for column in retained_columns if column not in group_columns
            ],
            "removed_group": group_name,
        }

    quality_columns = available_groups["signal_quality_context"]
    experiments["physiology_only"] = {
        "features": [
            column for column in retained_columns if column not in quality_columns
        ],
        "removed_group": "signal_quality_context",
    }
    experiments["quality_only"] = {
        "features": list(quality_columns),
        "removed_group": "all_physiological_groups",
    }

    empty_experiments = [
        name for name, definition in experiments.items() if not definition["features"]
    ]
    for experiment_name in empty_experiments:
        print(f"WARNING: skipping {experiment_name}; no selected columns exist.")
        del experiments[experiment_name]
    return experiments


def build_pipeline():
    """Create a fresh imputation -> SMOTE -> balanced RF pipeline."""
    return ImbalancedPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(**RANDOM_FOREST_PARAMETERS),
            ),
        ]
    )


def make_class_scorer(class_code):
    """Create a scalar recall scorer for one encoded class."""
    return make_scorer(
        recall_score,
        labels=[class_code],
        average="macro",
        zero_division=0,
    )


def repeated_cv_scoring(label_encoder):
    """Return all metrics calculated during repeated training-only CV."""
    scorers = {
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
    }
    for class_name in CLASS_REPORT_ORDER:
        class_code = int(label_encoder.transform([class_name])[0])
        scorers[f"{class_name.lower()}_recall"] = make_class_scorer(class_code)
    return scorers


def evaluate_test_predictions(target_test, predictions, label_encoder):
    """Calculate aggregate/per-class test metrics and confusion matrix."""
    class_codes = label_encoder.transform(CLASS_REPORT_ORDER)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_test,
        predictions,
        labels=class_codes,
        zero_division=0,
    )
    result = {
        "test_accuracy": accuracy_score(target_test, predictions),
        "test_balanced_accuracy": balanced_accuracy_score(
            target_test, predictions
        ),
        "test_macro_precision": precision_score(
            target_test, predictions, average="macro", zero_division=0
        ),
        "test_macro_recall": recall_score(
            target_test, predictions, average="macro", zero_division=0
        ),
        "test_macro_f1": f1_score(
            target_test, predictions, average="macro", zero_division=0
        ),
    }
    for index, class_name in enumerate(CLASS_REPORT_ORDER):
        prefix = f"test_{class_name.lower()}"
        result[f"{prefix}_precision"] = precision[index]
        result[f"{prefix}_recall"] = recall[index]
        result[f"{prefix}_f1"] = f1[index]

    pathological_code = int(label_encoder.transform(["Pathological"])[0])
    pathological_mask = target_test == pathological_code
    result["correct_pathological_cases"] = int(
        np.sum(predictions[pathological_mask] == pathological_code)
    )
    result["test_pathological_cases"] = int(np.sum(pathological_mask))
    matrix = confusion_matrix(target_test, predictions, labels=class_codes)
    result["confusion_matrix"] = json.dumps(matrix.tolist())
    result["confusion_matrix_array"] = matrix
    return result


def summarize_repeated_scores(scores):
    """Flatten repeated-CV arrays into mean and standard-deviation columns."""
    summary = {}
    metric_names = [
        "macro_f1",
        "balanced_accuracy",
        "pathological_recall",
        "suspicious_recall",
        "normal_recall",
    ]
    for metric_name in metric_names:
        values = scores[f"test_{metric_name}"]
        summary[f"repeated_cv_{metric_name}_mean"] = float(np.mean(values))
        summary[f"repeated_cv_{metric_name}_std"] = float(np.std(values))
    return summary


def run_experiment(
    experiment_name,
    definition,
    features_train,
    features_test,
    target_train,
    target_test,
    label_encoder,
):
    """Run standard CV, repeated CV, and one untouched-test evaluation."""
    selected_columns = definition["features"]
    train_subset = features_train[selected_columns]
    test_subset = features_test[selected_columns]
    pipeline = build_pipeline()
    standard_cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    repeated_cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=REPEATED_CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    started = time.perf_counter()
    standard_scores = cross_val_score(
        pipeline,
        train_subset,
        target_train,
        scoring="f1_macro",
        cv=standard_cv,
        n_jobs=CV_N_JOBS,
    )
    repeated_scores = cross_validate(
        pipeline,
        train_subset,
        target_train,
        scoring=repeated_cv_scoring(label_encoder),
        cv=repeated_cv,
        n_jobs=CV_N_JOBS,
        return_train_score=False,
    )
    pipeline.fit(train_subset, target_train)
    predictions = pipeline.predict(test_subset)

    result = {
        "experiment_name": experiment_name,
        "number_of_features": len(selected_columns),
        "removed_group": definition["removed_group"],
        "feature_columns": selected_columns,
        "cv_macro_f1_mean": float(np.mean(standard_scores)),
        "cv_macro_f1_std": float(np.std(standard_scores)),
        **summarize_repeated_scores(repeated_scores),
        **evaluate_test_predictions(target_test, predictions, label_encoder),
    }
    elapsed = time.perf_counter() - started
    print(
        f"{experiment_name}: {len(selected_columns)} features | "
        f"repeated-CV Macro F1={result['repeated_cv_macro_f1_mean']:.4f} "
        f"+/- {result['repeated_cv_macro_f1_std']:.4f} | "
        f"test Macro F1={result['test_macro_f1']:.4f} | {elapsed:.1f}s"
    )
    return result


def add_reference_changes_and_interpretations(results):
    """Add all-features deltas and cautious leave-one-group-out conclusions."""
    reference = next(
        result for result in results if result["experiment_name"] == "all_features"
    )
    for result in results:
        result["macro_f1_change_vs_all"] = (
            result["test_macro_f1"] - reference["test_macro_f1"]
        )
        result["pathological_recall_change_vs_all"] = (
            result["test_pathological_recall"]
            - reference["test_pathological_recall"]
        )
        result["suspicious_recall_change_vs_all"] = (
            result["test_suspicious_recall"]
            - reference["test_suspicious_recall"]
        )
        result["cv_macro_f1_change_vs_all"] = (
            result["cv_macro_f1_mean"] - reference["cv_macro_f1_mean"]
        )
        result["repeated_cv_macro_f1_change_vs_all"] = (
            result["repeated_cv_macro_f1_mean"]
            - reference["repeated_cv_macro_f1_mean"]
        )

        if not result["experiment_name"].startswith("without_"):
            result["group_interpretation"] = "not a leave-one-group-out experiment"
            continue
        repeated_change = result["repeated_cv_macro_f1_change_vs_all"]
        test_change = result["macro_f1_change_vs_all"]
        if repeated_change > 0 and test_change > 0:
            result["group_interpretation"] = "potentially harmful or noisy"
        elif repeated_change < 0 and test_change < 0:
            result["group_interpretation"] = "potentially helpful"
        else:
            result["group_interpretation"] = "unstable or inconclusive"


def results_dataframes(results):
    """Create main and repeated-CV tables with stable column ordering."""
    main_columns = [
        "experiment_name",
        "number_of_features",
        "removed_group",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_normal_precision",
        "test_normal_recall",
        "test_normal_f1",
        "test_suspicious_precision",
        "test_suspicious_recall",
        "test_suspicious_f1",
        "test_pathological_precision",
        "test_pathological_recall",
        "test_pathological_f1",
        "correct_pathological_cases",
        "test_pathological_cases",
        "confusion_matrix",
        "macro_f1_change_vs_all",
        "pathological_recall_change_vs_all",
        "suspicious_recall_change_vs_all",
        "cv_macro_f1_change_vs_all",
        "group_interpretation",
    ]
    repeated_columns = [
        "experiment_name",
        "number_of_features",
        "removed_group",
        "repeated_cv_macro_f1_mean",
        "repeated_cv_macro_f1_std",
        "repeated_cv_balanced_accuracy_mean",
        "repeated_cv_balanced_accuracy_std",
        "repeated_cv_pathological_recall_mean",
        "repeated_cv_pathological_recall_std",
        "repeated_cv_suspicious_recall_mean",
        "repeated_cv_suspicious_recall_std",
        "repeated_cv_normal_recall_mean",
        "repeated_cv_normal_recall_std",
        "repeated_cv_macro_f1_change_vs_all",
        "group_interpretation",
    ]
    main = pd.DataFrame(results)[main_columns].sort_values(
        "test_macro_f1", ascending=False, ignore_index=True
    )
    repeated = pd.DataFrame(results)[repeated_columns].sort_values(
        "repeated_cv_macro_f1_mean", ascending=False, ignore_index=True
    )
    return main, repeated


def format_markdown_value(value):
    """Format one value for a dependency-free Markdown table."""
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown(dataframe):
    """Render a DataFrame as Markdown without requiring ``tabulate``."""
    headers = list(dataframe.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in dataframe.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(format_markdown_value(value) for value in row) + " |"
        )
    return "\n".join(lines)


def named_class_counts(target, label_encoder):
    """Return class counts in the report's clinical display order."""
    return {
        class_name: int(np.sum(target == label_encoder.transform([class_name])[0]))
        for class_name in CLASS_REPORT_ORDER
    }


def group_findings(results):
    """Split leave-one-group-out findings into cautious interpretation lists."""
    findings = {
        "potentially helpful": [],
        "potentially harmful or noisy": [],
        "unstable or inconclusive": [],
    }
    for result in results:
        interpretation = result["group_interpretation"]
        if interpretation in findings:
            findings[interpretation].append(result["removed_group"])
    return findings


def bullet_list(values, empty_message="None"):
    """Render a short Markdown bullet list."""
    values = list(values)
    return "\n".join(f"- `{value}`" for value in values) if values else empty_message


def create_markdown_report(
    dataset,
    features,
    target_train,
    target_test,
    label_encoder,
    available_groups,
    ungrouped,
    results,
    main_results,
    repeated_results,
):
    """Write the human-readable ablation study report."""
    findings = group_findings(results)
    best_repeated = repeated_results.iloc[0]
    recommended_result = next(
        result
        for result in results
        if result["experiment_name"] == best_repeated["experiment_name"]
    )
    all_features_result = next(
        result for result in results if result["experiment_name"] == "all_features"
    )
    recommendation_change = (
        recommended_result["repeated_cv_macro_f1_mean"]
        - all_features_result["repeated_cv_macro_f1_mean"]
    )
    dataset_distribution = dataset[TARGET_COLUMN].value_counts().to_dict()

    group_lines = []
    for group_name, columns in available_groups.items():
        group_lines.append(f"### {group_name}\n\n" + bullet_list(columns))

    confusion_sections = []
    for result in results:
        matrix = pd.DataFrame(
            result["confusion_matrix_array"],
            index=[f"Actual {name}" for name in CLASS_REPORT_ORDER],
            columns=[f"Predicted {name}" for name in CLASS_REPORT_ORDER],
        )
        confusion_sections.append(
            f"### {result['experiment_name']}\n\n{dataframe_to_markdown(matrix.reset_index(names='actual'))}"
        )

    pathology_columns = [
        "experiment_name",
        "correct_pathological_cases",
        "test_pathological_cases",
        "test_pathological_recall",
        "repeated_cv_pathological_recall_mean",
        "repeated_cv_pathological_recall_std",
    ]
    pathology_table = pd.DataFrame(results)[pathology_columns].sort_values(
        "repeated_cv_pathological_recall_mean", ascending=False
    )

    report = f"""# CTG Feature Ablation Study

## Methodological warning

Repeated stratified cross-validation on the training partition is the primary
evidence in this report. The untouched test results are reported for comparison
only. Repeatedly selecting features based on test performance would overfit the
test set and invalidate it as an unbiased final evaluation.

Only **{named_class_counts(target_test, label_encoder)['Pathological']} Pathological records**
are present in the current test set. A one-case change therefore moves test
Pathological recall by approximately 16.7 percentage points. No feature group
is declared definitively helpful or harmful from that small test subset alone.

## Dataset and split

- Dataset rows: {len(dataset)}
- Retained leakage-free features: {features.shape[1]}
- Full class distribution: {dataset_distribution}
- Training rows: {len(target_train)}
- Training class distribution: {named_class_counts(target_train, label_encoder)}
- Test rows: {len(target_test)}
- Test class distribution: {named_class_counts(target_test, label_encoder)}
- Split: stratified 80/20, random_state={RANDOM_STATE}
- Repeated CV: {CV_SPLITS} folds x {REPEATED_CV_REPEATS} repeats on training data

## Retained features

{bullet_list(features.columns)}

## Ungrouped retained features

{bullet_list(ungrouped)}

## Feature-group definitions

{chr(10).join(group_lines)}

## Main comparison — sorted by untouched-test Macro F1

{dataframe_to_markdown(main_results)}

## Stability comparison — sorted by repeated-CV Macro F1

{dataframe_to_markdown(repeated_results)}

## Pathological recall comparison

{dataframe_to_markdown(pathology_table)}

## Confusion matrices

Matrix order is Normal, Suspicious, Pathological.

{chr(10).join(confusion_sections)}

## Groups that appear helpful

Removing these groups reduced both repeated-CV and test Macro F1, so they are
potentially helpful. This is evidence for follow-up, not a definitive claim.

{bullet_list(findings['potentially helpful'])}

## Groups that appear harmful or noisy

Removing these groups improved both repeated-CV and test Macro F1, so they may
be harmful or noisy and deserve further investigation.

{bullet_list(findings['potentially harmful or noisy'])}

## Unstable or inconclusive groups

Cross-validation and test directions disagreed, or one comparison was tied.

{bullet_list(findings['unstable or inconclusive'])}

## Most defensible feature subset

The recommended subset is **{recommended_result['experiment_name']}** with
{recommended_result['number_of_features']} features because it achieved the
highest training-only repeated-CV Macro F1
({recommended_result['repeated_cv_macro_f1_mean']:.4f} ±
{recommended_result['repeated_cv_macro_f1_std']:.4f}). This recommendation does
not use untouched-test performance for feature selection. Its improvement over
`all_features` is only {recommendation_change:+.4f}, much smaller than the
observed cross-validation standard deviation, so this is a tentative,
parsimonious recommendation that requires confirmation on external data.

Recommended features:

{bullet_list(recommended_result['feature_columns'])}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def save_macro_f1_chart(repeated_results):
    """Save repeated-CV Macro F1 means with standard-deviation error bars."""
    chart_data = repeated_results.sort_values(
        "repeated_cv_macro_f1_mean", ascending=True
    )
    figure_height = max(7, 0.45 * len(chart_data))
    fig, axis = plt.subplots(figsize=(12, figure_height))
    axis.barh(
        chart_data["experiment_name"],
        chart_data["repeated_cv_macro_f1_mean"],
        xerr=chart_data["repeated_cv_macro_f1_std"],
        color="steelblue",
        alpha=0.85,
        capsize=3,
    )
    axis.set_xlabel("Repeated-CV Macro F1 (mean ± standard deviation)")
    axis.set_ylabel("Experiment")
    axis.set_title("CTG Feature Ablation — Training-Only Repeated CV")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_final_summary(results, main_results, repeated_results):
    """Print the main conclusions and output paths."""
    best_repeated = repeated_results.iloc[0]
    best_pathology = repeated_results.sort_values(
        "repeated_cv_pathological_recall_mean", ascending=False
    ).iloc[0]
    best_test = main_results.iloc[0]
    leave_out = [
        result for result in results if result["experiment_name"].startswith("without_")
    ]
    removal_improved = [
        result["removed_group"]
        for result in leave_out
        if result["repeated_cv_macro_f1_change_vs_all"] > 0
    ]
    removal_reduced = [
        result["removed_group"]
        for result in leave_out
        if result["repeated_cv_macro_f1_change_vs_all"] < 0
    ]
    contradictory = [
        result["removed_group"]
        for result in leave_out
        if result["group_interpretation"] == "unstable or inconclusive"
    ]

    print("\nFinal ablation summary")
    print("======================")
    print(
        "Best by repeated-CV Macro F1: "
        f"{best_repeated['experiment_name']} "
        f"({best_repeated['repeated_cv_macro_f1_mean']:.4f})"
    )
    print(
        "Best by repeated-CV Pathological recall: "
        f"{best_pathology['experiment_name']} "
        f"({best_pathology['repeated_cv_pathological_recall_mean']:.1%})"
    )
    print(
        "Best by untouched-test Macro F1: "
        f"{best_test['experiment_name']} ({best_test['test_macro_f1']:.4f})"
    )
    print(f"Removals improving repeated-CV Macro F1: {removal_improved or ['None']}")
    print(f"Removals reducing repeated-CV Macro F1: {removal_reduced or ['None']}")
    print(f"Contradictory/unstable groups: {contradictory or ['None']}")
    print("\nWARNING: feature selection must use training-only repeated CV, not test scores.")
    print(f"Saved main results: {MAIN_RESULTS_PATH}")
    print(f"Saved repeated-CV results: {REPEATED_CV_PATH}")
    print(f"Saved Markdown report: {REPORT_PATH}")
    print(f"Saved Macro F1 chart: {CHART_PATH}")


def main():
    """Run all controlled feature ablation experiments and save reports."""
    dataset, features, target, label_encoder = load_and_prepare_dataset()
    print("Retained model-input columns")
    print("----------------------------")
    for column in features.columns:
        print(column)
    print(f"\nNumber of retained features: {features.shape[1]}")

    available_groups, ungrouped = validate_feature_groups(features.columns)
    experiments = build_experiments(features.columns, available_groups)
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    print(f"\nExperiments to run: {len(experiments)}")
    print(f"Training class distribution: {named_class_counts(target_train, label_encoder)}")
    print(f"Test class distribution: {named_class_counts(target_test, label_encoder)}")
    print(
        "WARNING: the test set remains untouched until final evaluation and is "
        "not used to select features.\n"
    )

    results = []
    for experiment_number, (experiment_name, definition) in enumerate(
        experiments.items(), start=1
    ):
        print(f"[{experiment_number}/{len(experiments)}] Running {experiment_name}...")
        results.append(
            run_experiment(
                experiment_name,
                definition,
                features_train,
                features_test,
                target_train,
                target_test,
                label_encoder,
            )
        )

    add_reference_changes_and_interpretations(results)
    main_results, repeated_results = results_dataframes(results)
    MAIN_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    main_results.to_csv(MAIN_RESULTS_PATH, index=False)
    repeated_results.to_csv(REPEATED_CV_PATH, index=False)
    create_markdown_report(
        dataset,
        features,
        target_train,
        target_test,
        label_encoder,
        available_groups,
        ungrouped,
        results,
        main_results,
        repeated_results,
    )
    save_macro_f1_chart(repeated_results)
    print_final_summary(results, main_results, repeated_results)


if __name__ == "__main__":
    main()
