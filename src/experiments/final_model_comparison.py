"""Final five-model comparison on the finalized temporal patient dataset.

No feature engineering, tuning, threshold selection, or model persistence is
performed.  Every model uses identical patient partitions and training-only
SMOTE. Linear/SVM models receive fold-local median imputation and scaling;
tree models receive fold-local median imputation without scaling.

Run from the project root:

    python src/experiments/final_model_comparison.py

The script refuses to overwrite an existing report directory.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

# Avoid joblib's unreliable macOS physical-core probe. This only controls the
# reported worker limit; model settings and evaluation partitions are unchanged.
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import diagnose_pathological_cases as prior_diagnostic
from src.models import train_models
from src.utils.record_ids import normalize_record_id

DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ml_dataset_final_temporal.csv"
)
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "final_model_comparison"
FIXED_PATH = OUTPUT_ROOT / "fixed_test_results.csv"
SPLIT_RESULTS_PATH = OUTPUT_ROOT / "repeated_split_results.csv"
SPLIT_SUMMARY_PATH = OUTPUT_ROOT / "repeated_split_summary.csv"
CV_RESULTS_PATH = OUTPUT_ROOT / "repeated_cv_results.csv"
CV_SUMMARY_PATH = OUTPUT_ROOT / "repeated_cv_summary.csv"
REPORT_PATH = OUTPUT_ROOT / "final_model_comparison_report.md"

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "Support Vector Machine",
]
LINEAR_MODELS = {"Logistic Regression", "Support Vector Machine"}
SPLIT_SEEDS = list(prior_diagnostic.SPLIT_SEEDS)
CV_SPLITS = prior_diagnostic.CV_SPLITS
CV_REPEATS = prior_diagnostic.CV_REPEATS
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_RANDOM_STATE = 42
EPSILON = 1e-12
EFFECTIVE_EQUIVALENCE_MACRO_F1 = 0.005

ALL_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "normal_recall",
    "suspicious_recall",
    "pathological_recall",
    "normal_precision",
    "suspicious_precision",
    "pathological_precision",
    "pathological_f1",
    "false_pathological_predictions",
    "true_pathological_detections",
]
PAIRWISE_METRICS = [
    "macro_f1",
    "balanced_accuracy",
    "pathological_recall",
    "pathological_precision",
    "false_pathological_predictions",
]
LOWER_IS_BETTER = {"false_pathological_predictions"}


def refuse_overwrite() -> None:
    """Keep this comparison separate from every previous experiment."""
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.rglob("*")):
        raise FileExistsError(
            f"{OUTPUT_ROOT} already contains files; move it before rerunning."
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_dataset_and_encoder() -> tuple[pd.DataFrame, object]:
    """Load the finalized dataset and saved label encoder."""
    if not DATASET_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError("Final dataset or saved LabelEncoder is missing")
    dataset = pd.read_csv(DATASET_PATH)
    dataset["record_id"] = normalize_record_id(dataset["record_id"])
    with ENCODER_PATH.open("rb") as handle:
        encoder = pickle.load(handle)
    return dataset, encoder


def validate_dataset(dataset: pd.DataFrame) -> list[str]:
    """Validate the finalized, leakage-aware 55-feature patient schema."""
    if len(dataset) != 552 or dataset["record_id"].nunique() != 552:
        raise ValueError("Final dataset must contain 552 unique patients")
    if dataset["label"].value_counts().to_dict() != {
        "Normal": 354,
        "Suspicious": 171,
        "Pathological": 27,
    }:
        raise ValueError("Final class distribution changed")
    excluded = {
        train_models.TARGET_COLUMN,
        *train_models.LEAKAGE_COLUMNS,
    }
    feature_columns = [
        column for column in dataset.columns if column not in excluded
    ]
    if len(feature_columns) != 55:
        raise ValueError(
            f"Expected exactly 55 finalized features, found {len(feature_columns)}"
        )
    non_numeric = dataset[feature_columns].select_dtypes(
        exclude=[np.number]
    ).columns.tolist()
    if non_numeric:
        raise ValueError("Non-numeric model features: " + ", ".join(non_numeric))
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Duplicate model feature columns")
    expected_temporal = {
        "maximum_negative_fhr_slope__window_final",
        "valid_fhr_percentage__window_min",
        "poor_quality_window_percentage",
        "final_window_quality",
    }
    actual_temporal = {
        column
        for column in feature_columns
        if "__window_" in column
        or column in {"poor_quality_window_percentage", "final_window_quality"}
    }
    if actual_temporal != expected_temporal:
        raise ValueError("Final temporal feature set changed")
    return feature_columns


def model_estimators() -> OrderedDict[str, object]:
    """Create the five untuned classifiers used in the comparison."""
    return OrderedDict(
        [
            (
                "Logistic Regression",
                LogisticRegression(
                    C=1,
                    solver="lbfgs",
                    max_iter=2000,
                ),
            ),
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                ),
            ),
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=42,
                ),
            ),
            (
                "LightGBM",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                ),
            ),
            (
                "Support Vector Machine",
                SVC(
                    kernel="rbf",
                    C=1,
                    gamma="scale",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )


def build_pipeline(model_name: str, estimator, smote_seed: int):
    """Build each model's fold-local preprocessing pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name in LINEAR_MODELS:
        steps.append(("scaler", StandardScaler()))
    steps.extend(
        [
            ("smote", SMOTE(random_state=smote_seed)),
            ("classifier", clone(estimator)),
        ]
    )
    return ImbalancedPipeline(steps=steps)


def fit_predict(
    model_name: str,
    estimator,
    features: pd.DataFrame,
    target: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    smote_seed: int,
):
    """Fit on training patients and predict only held-out patients."""
    pipeline = build_pipeline(model_name, estimator, smote_seed)
    # Suppress LightGBM training messages; this does not affect model settings.
    if model_name == "LightGBM":
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            pipeline.fit(features.iloc[train_indices], target[train_indices])
    else:
        pipeline.fit(features.iloc[train_indices], target[train_indices])
    predictions = pipeline.predict(features.iloc[test_indices])
    return predictions


def evaluate_predictions(
    target: np.ndarray,
    predictions: np.ndarray,
    encoder,
) -> dict:
    """Calculate evaluation metrics and a serializable confusion matrix."""
    codes = np.arange(len(encoder.classes_))
    precision, recall, class_f1, _ = precision_recall_fscore_support(
        target,
        predictions,
        labels=codes,
        zero_division=0,
    )
    pathological_code = int(encoder.transform(["Pathological"])[0])
    pathological = target == pathological_code
    predicted_pathological = predictions == pathological_code
    result = {
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, predictions)
        ),
        "macro_f1": float(
            f1_score(target, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(target, predictions, average="weighted", zero_division=0)
        ),
        "confusion_matrix": json.dumps(
            confusion_matrix(target, predictions, labels=codes).tolist()
        ),
        "false_pathological_predictions": int(
            np.sum(~pathological & predicted_pathological)
        ),
        "true_pathological_detections": int(
            np.sum(pathological & predicted_pathological)
        ),
    }
    for code, class_name in enumerate(encoder.classes_):
        prefix = class_name.lower()
        result[f"{prefix}_recall"] = float(recall[code])
        result[f"{prefix}_precision"] = float(precision[code])
        result[f"{prefix}_f1"] = float(class_f1[code])
    return result


def evaluate_partition(
    models: OrderedDict[str, object],
    features: pd.DataFrame,
    target: np.ndarray,
    encoder,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    smote_seed: int,
    identity: dict,
) -> list[dict]:
    """Evaluate all models on one identical held-out patient partition."""
    rows = []
    for model_name, estimator in models.items():
        predictions = fit_predict(
            model_name,
            estimator,
            features,
            target,
            train_indices,
            test_indices,
            smote_seed,
        )
        rows.append(
            {
                **identity,
                "model": model_name,
                "scaled_features": model_name in LINEAR_MODELS,
                "preprocessing": (
                    "median imputation -> StandardScaler -> training-only SMOTE"
                    if model_name in LINEAR_MODELS
                    else "median imputation -> training-only SMOTE"
                ),
                **evaluate_predictions(
                    target[test_indices], predictions, encoder
                ),
            }
        )
    return rows


def fixed_evaluation(
    models: OrderedDict[str, object],
    features: pd.DataFrame,
    target: np.ndarray,
    encoder,
) -> pd.DataFrame:
    """Evaluate all five models once on the fixed split."""
    train_indices, test_indices = train_test_split(
        np.arange(len(target)),
        test_size=train_models.TEST_SIZE,
        random_state=train_models.RANDOM_STATE,
        stratify=target,
    )
    return pd.DataFrame(
        evaluate_partition(
            models,
            features,
            target,
            encoder,
            train_indices,
            test_indices,
            train_models.RANDOM_STATE,
            {
                "train_patient_count": len(train_indices),
                "test_patient_count": len(test_indices),
                "random_state": train_models.RANDOM_STATE,
            },
        )
    )


def repeated_split_evaluation(
    models: OrderedDict[str, object],
    features: pd.DataFrame,
    target: np.ndarray,
    encoder,
) -> pd.DataFrame:
    """Evaluate every model on the same 30 predefined stratified splits."""
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
            evaluate_partition(
                models,
                features,
                target,
                encoder,
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


def repeated_cv_evaluation(
    models: OrderedDict[str, object],
    features: pd.DataFrame,
    target: np.ndarray,
    encoder,
) -> pd.DataFrame:
    """Evaluate every model on identical repeated 5x5 patient-level folds."""
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
            evaluate_partition(
                models,
                features,
                target,
                encoder,
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
    """Return a deterministic paired 95% bootstrap interval for the mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_STATE)
    means = rng.choice(
        values,
        size=(BOOTSTRAP_RESAMPLES, len(values)),
        replace=True,
    ).mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def summarize_results(
    results: pd.DataFrame,
    partition_column: str,
) -> pd.DataFrame:
    """Summarize models and paired differences from Logistic Regression."""
    rows = []
    evaluation_count = results[partition_column].nunique()
    logistic = results[
        results["model"] == "Logistic Regression"
    ].sort_values(partition_column)
    for model_name in MODEL_NAMES:
        model_rows = results[results["model"] == model_name].sort_values(
            partition_column
        )
        if len(model_rows) != evaluation_count:
            raise AssertionError(f"Incomplete results for {model_name}")
        for metric in ALL_METRICS:
            values = model_rows[metric]
            rows.append(
                {
                    "row_type": "model_summary",
                    "model_or_comparison": model_name,
                    "metric": metric,
                    "evaluation_count": evaluation_count,
                    "mean": values.mean(),
                    "median": values.median(),
                    "standard_deviation": values.std(ddof=1),
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "paired_mean_difference": np.nan,
                    "paired_mean_bootstrap_95ci_lower": np.nan,
                    "paired_mean_bootstrap_95ci_upper": np.nan,
                    "improved_count": np.nan,
                    "worsened_count": np.nan,
                    "equal_count": np.nan,
                }
            )
        if model_name == "Logistic Regression":
            continue
        for metric in PAIRWISE_METRICS:
            differences = (
                model_rows[metric].to_numpy()
                - logistic[metric].to_numpy()
            )
            lower, upper = bootstrap_interval(differences)
            if metric in LOWER_IS_BETTER:
                improved = differences < -EPSILON
                worsened = differences > EPSILON
            else:
                improved = differences > EPSILON
                worsened = differences < -EPSILON
            rows.append(
                {
                    "row_type": "pairwise_vs_logistic",
                    "model_or_comparison": (
                        f"{model_name} minus Logistic Regression"
                    ),
                    "metric": metric,
                    "evaluation_count": evaluation_count,
                    "mean": np.nan,
                    "median": np.nan,
                    "standard_deviation": np.nan,
                    "minimum": np.nan,
                    "maximum": np.nan,
                    "paired_mean_difference": differences.mean(),
                    "paired_mean_bootstrap_95ci_lower": lower,
                    "paired_mean_bootstrap_95ci_upper": upper,
                    "improved_count": int(improved.sum()),
                    "worsened_count": int(worsened.sum()),
                    "equal_count": int(
                        (np.abs(differences) <= EPSILON).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summary_value(
    summary: pd.DataFrame,
    name: str,
    metric: str,
    column: str = "mean",
) -> float:
    """Read one model/metric value from a long summary."""
    row = summary[
        (summary["model_or_comparison"] == name)
        & (summary["metric"] == metric)
    ]
    if len(row) != 1:
        raise AssertionError(f"Missing summary row: {name}/{metric}")
    return float(row.iloc[0][column])


def rank_models(cv_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank models by the repeated-CV comparison priorities."""
    rows = []
    for model_name in MODEL_NAMES:
        rows.append(
            {
                "model": model_name,
                "repeated_cv_macro_f1": summary_value(
                    cv_summary, model_name, "macro_f1"
                ),
                "repeated_cv_pathological_recall": summary_value(
                    cv_summary, model_name, "pathological_recall"
                ),
                "repeated_cv_pathological_precision": summary_value(
                    cv_summary, model_name, "pathological_precision"
                ),
                "repeated_cv_balanced_accuracy": summary_value(
                    cv_summary, model_name, "balanced_accuracy"
                ),
                "repeated_cv_false_pathological_predictions": summary_value(
                    cv_summary, model_name, "false_pathological_predictions"
                ),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        [
            "repeated_cv_macro_f1",
            "repeated_cv_pathological_recall",
            "repeated_cv_pathological_precision",
            "repeated_cv_balanced_accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def choose_recommendation(
    ranking: pd.DataFrame,
    cv_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> tuple[str, str]:
    """Recommend by performance, stability, false alarms, then simplicity."""
    top = ranking.iloc[0]["model"]
    top_macro = ranking.iloc[0]["repeated_cv_macro_f1"]
    logistic_macro = summary_value(
        cv_summary, "Logistic Regression", "macro_f1"
    )
    reason_parts = [
        f"{top} ranked first in repeated-CV Macro F1 ({top_macro:.4f})"
    ]
    if top != "Logistic Regression":
        comparison = f"{top} minus Logistic Regression"
        pair = cv_summary[
            (cv_summary["model_or_comparison"] == comparison)
            & (cv_summary["metric"] == "macro_f1")
        ].iloc[0]
        ci_crosses_zero = (
            pair["paired_mean_bootstrap_95ci_lower"] <= 0
            <= pair["paired_mean_bootstrap_95ci_upper"]
        )
        recall_gain = (
            summary_value(cv_summary, top, "pathological_recall")
            - summary_value(
                cv_summary, "Logistic Regression", "pathological_recall"
            )
        )
        if (
            top_macro - logistic_macro <= EFFECTIVE_EQUIVALENCE_MACRO_F1
            and ci_crosses_zero
            and recall_gain <= 0.02
        ):
            reason = (
                f"{top} was numerically first, but its Macro F1 advantage "
                "over Logistic Regression was effectively equivalent, its "
                "paired interval crossed zero, and it did not materially "
                "improve Pathological recall; the simpler model is preferred."
            )
            return "Logistic Regression", reason
        reason_parts.append(
            "its paired Macro F1 comparison with Logistic Regression had "
            f"95% CI [{pair['paired_mean_bootstrap_95ci_lower']:+.4f}, "
            f"{pair['paired_mean_bootstrap_95ci_upper']:+.4f}]"
        )
        recall_pair = cv_summary[
            (cv_summary["model_or_comparison"] == comparison)
            & (cv_summary["metric"] == "pathological_recall")
        ].iloc[0]
        precision_pair = cv_summary[
            (cv_summary["model_or_comparison"] == comparison)
            & (cv_summary["metric"] == "pathological_precision")
        ].iloc[0]
        false_pair = cv_summary[
            (cv_summary["model_or_comparison"] == comparison)
            & (
                cv_summary["metric"]
                == "false_pathological_predictions"
            )
        ].iloc[0]
        top_split_macro = summary_value(
            split_summary, top, "macro_f1"
        )
        logistic_split_macro = summary_value(
            split_summary, "Logistic Regression", "macro_f1"
        )
        reason_parts.append(
            f"versus Logistic Regression it changed Pathological recall by "
            f"{recall_pair['paired_mean_difference']:+.4f}, precision by "
            f"{precision_pair['paired_mean_difference']:+.4f}, and false "
            f"Pathological predictions by "
            f"{false_pair['paired_mean_difference']:+.2f}"
        )
        reason_parts.append(
            f"across the 30 predefined splits its Macro F1 was "
            f"{top_split_macro:.4f} versus {logistic_split_macro:.4f} for "
            "Logistic Regression"
        )
    split_std = summary_value(
        split_summary, top, "macro_f1", "standard_deviation"
    )
    recall = summary_value(cv_summary, top, "pathological_recall")
    precision = summary_value(cv_summary, top, "pathological_precision")
    false_predictions = summary_value(
        cv_summary, top, "false_pathological_predictions"
    )
    reason_parts.append(
        f"30-split Macro F1 SD was {split_std:.4f}, Pathological recall was "
        f"{recall:.4f}, precision was {precision:.4f}, and mean false "
        f"Pathological predictions were {false_predictions:.2f}"
    )
    return top, "; ".join(reason_parts) + "."


def markdown_metric_table(summary: pd.DataFrame, metric: str) -> str:
    """Render model means and LR-paired comparisons for the report."""
    rows = summary[summary["metric"] == metric]
    lines = [
        "| Model/comparison | Mean ± SD or paired change | 95% paired CI | Improved/Worsened/Equal |",
        "|---|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        if row["row_type"] == "model_summary":
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
        lines.append(
            f"| {row['model_or_comparison']} | {value} | {interval} | {counts} |"
        )
    return "\n".join(lines)


def write_report(
    fixed: pd.DataFrame,
    split_summary: pd.DataFrame,
    cv_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    recommendation: str,
    reason: str,
) -> None:
    """Write the final comparison report without overstating uncertainty."""
    settings = {
        "Logistic Regression": 'C=1, solver="lbfgs", max_iter=2000',
        "Random Forest": (
            "n_estimators=300, max_depth=None, min_samples_split=2, "
            "min_samples_leaf=1, random_state=42"
        ),
        "XGBoost": (
            "n_estimators=300, learning_rate=0.05, max_depth=4, "
            "subsample=0.8, colsample_bytree=0.8, "
            'objective="multi:softprob", eval_metric="mlogloss", '
            "random_state=42"
        ),
        "LightGBM": (
            "n_estimators=300, learning_rate=0.05, num_leaves=31, "
            "subsample=0.8, colsample_bytree=0.8, random_state=42"
        ),
        "Support Vector Machine": (
            'kernel="rbf", C=1, gamma="scale", probability=True, '
            "random_state=42"
        ),
    }
    model_lines = "\n".join(
        f"- **{name}:** `{settings[name]}`" for name in MODEL_NAMES
    )
    ranking_lines = [
        "| Rank | Model | Macro F1 | Path recall | Path precision | Balanced accuracy | False Pathological |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranking.itertuples():
        ranking_lines.append(
            f"| {row.rank} | {row.model} | {row.repeated_cv_macro_f1:.4f} | "
            f"{row.repeated_cv_pathological_recall:.4f} | "
            f"{row.repeated_cv_pathological_precision:.4f} | "
            f"{row.repeated_cv_balanced_accuracy:.4f} | "
            f"{row.repeated_cv_false_pathological_predictions:.2f} |"
        )
    fixed_lines = [
        "| Model | Accuracy | Balanced accuracy | Macro F1 | Path recall | Path precision | False Pathological |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed.itertuples():
        fixed_lines.append(
            f"| {row.model} | {row.accuracy:.4f} | "
            f"{row.balanced_accuracy:.4f} | {row.macro_f1:.4f} | "
            f"{row.pathological_recall:.4f} | "
            f"{row.pathological_precision:.4f} | "
            f"{row.false_pathological_predictions} |"
        )
    report = f"""# Final Model Comparison

## Scope

This is the final comparison on `ml_dataset_final_temporal.csv`. The finalized
55-feature representation, 552 patient IDs, labels, leakage exclusions, fixed
split, 30 seeds, and repeated 5x5 folds were unchanged. No feature engineering,
hyperparameter tuning, threshold tuning, model saving, or sequence modelling
was performed.

## Models

{model_lines}

Every pipeline used fold-local median imputation followed by training-only
SMOTE. Logistic Regression and the RBF SVM also used fold-local
`StandardScaler`; tree models were not scaled.

## Fixed test (descriptive only)

{chr(10).join(fixed_lines)}

The fixed test was evaluated once and was not used to choose the recommendation.

## Thirty predefined splits

### Macro F1

{markdown_metric_table(split_summary, 'macro_f1')}

### Balanced accuracy

{markdown_metric_table(split_summary, 'balanced_accuracy')}

### Pathological recall

{markdown_metric_table(split_summary, 'pathological_recall')}

### Pathological precision

{markdown_metric_table(split_summary, 'pathological_precision')}

### False Pathological predictions

{markdown_metric_table(split_summary, 'false_pathological_predictions')}

## Repeated 5x5 cross-validation

### Macro F1

{markdown_metric_table(cv_summary, 'macro_f1')}

### Balanced accuracy

{markdown_metric_table(cv_summary, 'balanced_accuracy')}

### Pathological recall

{markdown_metric_table(cv_summary, 'pathological_recall')}

### Pathological precision

{markdown_metric_table(cv_summary, 'pathological_precision')}

### False Pathological predictions

{markdown_metric_table(cv_summary, 'false_pathological_predictions')}

Pairwise differences are model minus Logistic Regression. For false
Pathological predictions, a negative difference is favourable and the
improved/worsened counts use that direction. Confidence intervals crossing zero
are treated as statistical uncertainty, not significance.

## Final ranking

{chr(10).join(ranking_lines)}

The ranking is lexicographic: repeated-CV Macro F1, then Pathological recall,
Pathological precision, and balanced accuracy. Stability across the 30 splits,
false alarms, paired uncertainty, and simplicity were then used for the final
recommendation.

## Recommendation

**{recommendation}.** {reason}

This recommendation follows the primary Macro-F1 ranking. The paired
tables must still be used to judge its recall, precision, and false-alarm
trade-offs. If the deployment objective prioritizes sensitivity over precision
and false alarms, the model with stronger Pathological recall may be the more
appropriate operational alternative.

No trained artifact was saved or replaced. The result does not establish
clinical significance.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def validate_outputs(
    fixed: pd.DataFrame,
    split_results: pd.DataFrame,
    split_summary: pd.DataFrame,
    cv_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
    ranking: pd.DataFrame,
) -> None:
    """Validate model, partition, and metric coverage."""
    if len(fixed) != 5 or set(fixed["model"]) != set(MODEL_NAMES):
        raise AssertionError("Fixed-test model coverage failed")
    if len(split_results) != 30 * 5:
        raise AssertionError("Repeated-split coverage failed")
    if len(cv_results) != 25 * 5:
        raise AssertionError("Repeated-CV coverage failed")
    expected_summary_rows = len(MODEL_NAMES) * len(ALL_METRICS) + (
        (len(MODEL_NAMES) - 1) * len(PAIRWISE_METRICS)
    )
    if len(split_summary) != expected_summary_rows:
        raise AssertionError("Repeated-split summary is incomplete")
    if len(cv_summary) != expected_summary_rows:
        raise AssertionError("Repeated-CV summary is incomplete")
    if len(ranking) != 5 or set(ranking["rank"]) != set(range(1, 6)):
        raise AssertionError("Final ranking is incomplete")
    for results, partition, expected in [
        (split_results, "random_seed", 30),
        (cv_results, "fold_number", 25),
    ]:
        if results[partition].nunique() != expected:
            raise AssertionError(f"Partition coverage failed: {partition}")
        counts = results.groupby(partition)["model"].nunique()
        if not (counts == 5).all():
            raise AssertionError("A partition does not contain all five models")
    if not REPORT_PATH.exists():
        raise AssertionError("Final model report is missing")


def main() -> None:
    """Run the final five-model comparison."""
    refuse_overwrite()
    dataset, encoder = load_dataset_and_encoder()
    feature_columns = validate_dataset(dataset)
    features = dataset[feature_columns]
    target = encoder.transform(dataset["label"])
    models = model_estimators()

    print("Finalized dataset validation")
    print(f"Patients: {len(dataset)}")
    print(f"Model features: {len(feature_columns)}")
    print(f"Class counts: {dataset['label'].value_counts().to_dict()}")
    print(f"Models: {MODEL_NAMES}")

    print("\nRunning fixed test...")
    fixed = fixed_evaluation(models, features, target, encoder)
    fixed.to_csv(FIXED_PATH, index=False)

    print("\nRunning 30 predefined splits...")
    split_results = repeated_split_evaluation(
        models, features, target, encoder
    )
    split_results.to_csv(SPLIT_RESULTS_PATH, index=False)
    split_summary = summarize_results(split_results, "random_seed")
    split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False)

    print("\nRunning repeated 5x5 cross-validation...")
    cv_results = repeated_cv_evaluation(models, features, target, encoder)
    cv_results.to_csv(CV_RESULTS_PATH, index=False)
    cv_summary = summarize_results(cv_results, "fold_number")
    cv_summary.to_csv(CV_SUMMARY_PATH, index=False)

    ranking = rank_models(cv_summary)
    recommendation, reason = choose_recommendation(
        ranking, cv_summary, split_summary
    )
    write_report(
        fixed,
        split_summary,
        cv_summary,
        ranking,
        recommendation,
        reason,
    )
    validate_outputs(
        fixed,
        split_results,
        split_summary,
        cv_results,
        cv_summary,
        ranking,
    )

    print("\nFinal model comparison complete")
    print("Final ranking")
    for row in ranking.itertuples():
        print(
            f"{row.rank}. {row.model}: Macro F1="
            f"{row.repeated_cv_macro_f1:.4f}; Pathological recall="
            f"{row.repeated_cv_pathological_recall:.4f}; Pathological precision="
            f"{row.repeated_cv_pathological_precision:.4f}; balanced accuracy="
            f"{row.repeated_cv_balanced_accuracy:.4f}; false Pathological="
            f"{row.repeated_cv_false_pathological_predictions:.2f}"
        )
    print(f"Recommendation: {recommendation}")
    print(f"Reason: {reason}")
    for path in [
        FIXED_PATH,
        SPLIT_RESULTS_PATH,
        SPLIT_SUMMARY_PATH,
        CV_RESULTS_PATH,
        CV_SUMMARY_PATH,
        REPORT_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()
