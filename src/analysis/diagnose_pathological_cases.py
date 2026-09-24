"""Diagnose why the Pathological CTG class is difficult to classify.

This script is intentionally diagnostic.  It reads the current datasets,
saved model, encoder, and split configuration, then writes new reports only.
It never changes labels, extracted features, the split definition, or saved
model artifacts.

Run from the project root::

    ./.venv/bin/python src/analysis/diagnose_pathological_cases.py
"""

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLOT_CACHE = Path(tempfile.gettempdir()) / "ctg-pathological-diagnostics-mpl"
PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

from src.models import train_models

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"
ML_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "pathological_diagnostics"
AUDIT_PATH = REPORTS_DIR / "pathological_case_audit.csv"
SUBGROUP_PATH = REPORTS_DIR / "pathological_subgroup_summary.csv"
FEATURE_COMPARISON_PATH = REPORTS_DIR / "pathological_feature_comparison.csv"
TEST_CASES_PATH = REPORTS_DIR / "pathological_test_cases.csv"
SPLIT_STABILITY_PATH = REPORTS_DIR / "pathological_split_stability.csv"
THRESHOLD_PATH = REPORTS_DIR / "pathological_threshold_analysis.csv"
BALANCING_PATH = REPORTS_DIR / "pathological_balancing_comparison.csv"
BINARY_PATH = REPORTS_DIR / "pathological_binary_diagnostic.csv"
REPORT_PATH = REPORTS_DIR / "pathological_diagnostic_report.md"

OUTCOME_COLUMNS = ["pH", "BE", "BDecf", "Apgar5"]
LEAKAGE_COLUMNS = {
    "record_id",
    "pH",
    "BE",
    "BDecf",
    "Apgar5",
    "label",
    "feature_quality_flag",
}
CLASS_ORDER = ["Normal", "Suspicious", "Pathological"]
SIGNAL_QUALITY_COLUMNS = [
    "percent_missing_fhr",
    "valid_fhr_percentage",
    "possible_fhr_artifact_percentage",
    "longest_missing_fhr_gap_seconds",
]
IMPORTANT_CTG_COLUMNS = [
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
]
SPLIT_SEEDS = list(range(30))
CV_SPLITS = 5
CV_REPEATS = 5
CV_N_JOBS = 1
THRESHOLDS = np.round(np.arange(0.05, 0.501, 0.01), 2)


def load_inputs():
    """Load all required inputs and preserve record IDs as strings."""
    for path in [
        LABELS_PATH,
        FEATURES_PATH,
        ML_DATASET_PATH,
        MODEL_PATH,
        ENCODER_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    labels = pd.read_csv(LABELS_PATH, dtype={"record_id": str})
    features = pd.read_csv(FEATURES_PATH, dtype={"record_id": str})
    ml = pd.read_csv(ML_DATASET_PATH, dtype={"record_id": str})
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)
    with ENCODER_PATH.open("rb") as file:
        encoder = pickle.load(file)
    return labels, features, ml, model, encoder


def compare_series(left, right):
    """Compare numeric or text Series while treating paired NaNs as equal."""
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return np.isclose(
            left.to_numpy(float), right.to_numpy(float), equal_nan=True
        )
    return (
        left.fillna("<MISSING>").astype(str).to_numpy()
        == right.fillna("<MISSING>").astype(str).to_numpy()
    )


def validate_alignment(labels, features, ml):
    """Audit IDs, duplicates, merge values, labels, and row order."""
    messages = []
    alignment_errors = 0
    label_ids = set(labels["record_id"])
    feature_ids = set(features["record_id"])
    missing_features = sorted(label_ids - feature_ids)
    missing_labels = sorted(feature_ids - label_ids)
    if missing_features:
        messages.append(f"Labels without features: {missing_features}")
        alignment_errors += len(missing_features)
    if missing_labels:
        messages.append(f"Features without labels: {missing_labels}")
        alignment_errors += len(missing_labels)

    for name, frame in [
        ("labels.csv", labels),
        ("clinical_features.csv", features),
        ("ml_dataset.csv", ml),
    ]:
        duplicates = frame.loc[
            frame["record_id"].duplicated(keep=False), "record_id"
        ].tolist()
        if duplicates:
            messages.append(f"{name} duplicate record IDs: {duplicates}")
            alignment_errors += len(set(duplicates))

    expected = features.merge(labels, on="record_id", how="inner")
    expected_by_id = expected.set_index("record_id").sort_index()
    ml_by_id = ml.set_index("record_id").sort_index()
    if set(expected_by_id.index) != set(ml_by_id.index):
        messages.append("ml_dataset record IDs differ from the expected merge.")
        alignment_errors += len(
            set(expected_by_id.index).symmetric_difference(ml_by_id.index)
        )

    common_ids = expected_by_id.index.intersection(ml_by_id.index)
    columns_to_check = ["label", *OUTCOME_COLUMNS]
    for column in columns_to_check:
        matches = compare_series(
            expected_by_id.loc[common_ids, column],
            ml_by_id.loc[common_ids, column],
        )
        mismatch_ids = common_ids[~matches].tolist()
        if mismatch_ids:
            messages.append(f"{column} mismatches after merge: {mismatch_ids}")
            alignment_errors += len(mismatch_ids)

    expected_order = expected["record_id"].tolist()
    ml_order = ml["record_id"].tolist()
    order_matches = expected_order == ml_order
    if not order_matches:
        messages.append(
            "ml_dataset row order differs from the deterministic features→labels merge; "
            "ID-based values remain checked separately."
        )

    return {
        "alignment_errors": alignment_errors,
        "messages": messages,
        "missing_features": missing_features,
        "missing_labels": missing_labels,
        "merge_order_matches": order_matches,
    }


def independently_label(row):
    """Reproduce the outcome-based rule without importing label code."""
    severe = row["pH"] < 7.05 and (
        row["BE"] <= -12 or row["BDecf"] >= 12
    )
    low_apgar = row["Apgar5"] <= 3
    moderate = (
        row["pH"] < 7.20
        or row["BE"] <= -8
        or row["BDecf"] >= 8
        or row["Apgar5"] <= 6
    )
    if severe or low_apgar:
        return "Pathological"
    if moderate:
        return "Suspicious"
    return "Normal"


def pathological_reason(row):
    """Assign the severe-metabolic or low-Apgar pathological mechanism."""
    severe = row["pH"] < 7.05 and (
        row["BE"] <= -12 or row["BDecf"] >= 12
    )
    low_apgar = row["Apgar5"] <= 3
    if severe and low_apgar:
        return "severe_metabolic_and_low_apgar"
    if severe:
        return "severe_metabolic_criteria"
    if low_apgar:
        return "low_apgar_only"
    return "other"


def recalculate_labels(labels):
    """Recalculate every label and return mismatch/count diagnostics."""
    audited = labels.copy()
    audited["recalculated_label"] = audited.apply(independently_label, axis=1)
    audited["label_matches_rule"] = (
        audited["label"] == audited["recalculated_label"]
    )
    audited["pathological_reason"] = audited.apply(pathological_reason, axis=1)
    disagreements = audited.loc[~audited["label_matches_rule"]].copy()
    counts = audited["recalculated_label"].value_counts().to_dict()
    return audited, disagreements, counts


def retained_feature_columns(ml, model):
    """Use the ordered features expected by the saved model."""
    if hasattr(model, "feature_names_in_"):
        columns = list(model.feature_names_in_)
    else:
        columns = [
            column for column in ml.columns if column not in LEAKAGE_COLUMNS
        ]
    missing = [column for column in columns if column not in ml]
    if missing:
        raise ValueError(f"Saved model expects missing columns: {missing}")
    return columns


def reproduce_split(ml, encoder):
    """Reproduce the train/test membership from train_models.py."""
    target = encoder.transform(ml["label"])
    indices = np.arange(len(ml))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=train_models.TEST_SIZE,
        random_state=train_models.RANDOM_STATE,
        stratify=target,
    )
    return train_indices, test_indices, target


def probability_columns(model, encoder, probabilities):
    """Map model probability columns to readable class-specific arrays."""
    model_classes = model.classes_
    mapping = {}
    for class_name in CLASS_ORDER:
        code = int(encoder.transform([class_name])[0])
        position = int(np.flatnonzero(model_classes == code)[0])
        mapping[class_name] = probabilities[:, position]
    return mapping


def create_pathological_audit(
    labels_audited,
    ml,
    model,
    encoder,
    feature_columns,
    train_indices,
    test_indices,
):
    """Create a record-level audit of all Pathological cases."""
    inputs = ml[feature_columns]
    probabilities = model.predict_proba(inputs)
    predictions = model.predict(inputs)
    probability_map = probability_columns(model, encoder, probabilities)
    prediction_names = encoder.inverse_transform(predictions)

    prediction_frame = pd.DataFrame(
        {
            "record_id": ml["record_id"],
            "predicted_class": prediction_names,
            "predicted_probability_normal": probability_map["Normal"],
            "predicted_probability_suspicious": probability_map["Suspicious"],
            "predicted_probability_pathological": probability_map["Pathological"],
        }
    )
    partition = np.full(len(ml), "training", dtype=object)
    partition[test_indices] = "test"
    prediction_frame["current_split_partition"] = partition

    reason_frame = labels_audited[
        ["record_id", "recalculated_label", "pathological_reason"]
    ]
    audit = (
        ml.merge(reason_frame, on="record_id", how="left")
        .merge(prediction_frame, on="record_id", how="left")
    )
    audit["stored_label"] = audit["label"]
    audit["correctly_classified"] = (
        audit["predicted_class"] == audit["stored_label"]
    )
    requested_columns = [
        "record_id",
        "pH",
        "BE",
        "BDecf",
        "Apgar5",
        "stored_label",
        "recalculated_label",
        "pathological_reason",
        "feature_quality_flag",
        "recording_duration_minutes",
        "percent_missing_fhr",
        "valid_fhr_percentage",
        "possible_fhr_artifact_percentage",
        "longest_missing_fhr_gap_seconds",
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
        "predicted_class",
        "predicted_probability_normal",
        "predicted_probability_suspicious",
        "predicted_probability_pathological",
        "current_split_partition",
        "correctly_classified",
    ]
    pathological = audit.loc[audit["stored_label"] == "Pathological", requested_columns]
    return pathological.sort_values("record_id").reset_index(drop=True)


def most_common_incorrect(group):
    """Return the most common wrong class for a Pathological subgroup."""
    incorrect = group.loc[~group["correctly_classified"], "predicted_class"]
    return incorrect.mode().iloc[0] if not incorrect.empty else "none"


def summarize_pathological_subgroups(audit):
    """Summarize clinical outcomes, quality, CTG values, and predictions."""
    numeric_columns = [
        *OUTCOME_COLUMNS,
        *SIGNAL_QUALITY_COLUMNS,
        *IMPORTANT_CTG_COLUMNS,
        "predicted_probability_pathological",
    ]
    rows = []
    for reason, group in audit.groupby("pathological_reason", dropna=False):
        row = {
            "pathological_reason": reason,
            "number_of_records": len(group),
            "classification_accuracy": group["correctly_classified"].mean(),
            "pathological_recall": group["correctly_classified"].mean(),
            "most_common_incorrect_predicted_class": most_common_incorrect(group),
        }
        for column in numeric_columns:
            row[f"average_{column}"] = group[column].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pathological_reason").reset_index(drop=True)


def cliffs_delta(first, second):
    """Compute robust Cliff's delta: P(first>second) - P(first<second)."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if not len(first) or not len(second):
        return np.nan
    differences = first[:, None] - second[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def effect_magnitude(value):
    """Categorize absolute Cliff's delta using common robust thresholds."""
    if not np.isfinite(value):
        return "unavailable"
    absolute = abs(value)
    if absolute < 0.147:
        return "negligible"
    if absolute < 0.33:
        return "small"
    if absolute < 0.474:
        return "medium"
    return "large"


def feature_distribution_comparison(ml, feature_columns):
    """Create per-class summaries, robust effect sizes, and redundancy flags."""
    numeric = ml[feature_columns]
    correlation = numeric.corr(method="spearman")
    rows = []
    class_values = {
        class_name: ml.loc[ml["label"] == class_name]
        for class_name in CLASS_ORDER
    }
    pathological = class_values["Pathological"]
    non_pathological = ml.loc[ml["label"] != "Pathological"]

    for feature in feature_columns:
        row = {
            "feature": feature,
            "missing_count": int(ml[feature].isna().sum()),
            "missing_percentage": 100.0 * ml[feature].isna().mean(),
            "near_zero_variance": bool(ml[feature].std(skipna=True) <= 1e-12),
        }
        for class_name in CLASS_ORDER:
            values = class_values[class_name][feature].dropna()
            prefix = class_name.lower()
            row.update(
                {
                    f"{prefix}_count": len(values),
                    f"{prefix}_mean": values.mean(),
                    f"{prefix}_median": values.median(),
                    f"{prefix}_std": values.std(),
                    f"{prefix}_minimum": values.min(),
                    f"{prefix}_maximum": values.max(),
                    f"{prefix}_iqr": values.quantile(0.75) - values.quantile(0.25),
                }
            )

        comparisons = {
            "pathological_vs_normal": cliffs_delta(
                pathological[feature], class_values["Normal"][feature]
            ),
            "pathological_vs_suspicious": cliffs_delta(
                pathological[feature], class_values["Suspicious"][feature]
            ),
            "pathological_vs_non_pathological": cliffs_delta(
                pathological[feature], non_pathological[feature]
            ),
        }
        for comparison_name, delta in comparisons.items():
            row[f"cliffs_delta_{comparison_name}"] = delta
            row[f"effect_magnitude_{comparison_name}"] = effect_magnitude(delta)
        row["clearly_separating_pathological"] = (
            abs(comparisons["pathological_vs_non_pathological"]) >= 0.474
        )
        row["heavy_overlap_with_normal_and_suspicious"] = (
            abs(comparisons["pathological_vs_normal"]) < 0.147
            and abs(comparisons["pathological_vs_suspicious"]) < 0.147
        )
        correlated = [
            other
            for other in feature_columns
            if other != feature and abs(correlation.loc[feature, other]) >= 0.95
        ]
        duplicates = [
            other
            for other in feature_columns
            if other != feature
            and ml[feature].fillna(np.inf).equals(ml[other].fillna(np.inf))
        ]
        row["high_correlation_partners_abs_spearman_ge_0_95"] = ";".join(correlated)
        row["exact_duplicate_partners"] = ";".join(duplicates)
        rows.append(row)
    return pd.DataFrame(rows), correlation


def create_visualizations(ml, audit, comparison, correlation, model, encoder, feature_columns):
    """Create class, probability, PCA, reason, and quality plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"Normal": "#3B82F6", "Suspicious": "#F59E0B", "Pathological": "#DC2626"}
    top_features = (
        comparison.assign(
            absolute_effect=comparison[
                "cliffs_delta_pathological_vs_non_pathological"
            ].abs()
        )
        .sort_values("absolute_effect", ascending=False)
        .head(15)["feature"]
        .tolist()
    )

    fig, axes = plt.subplots(5, 3, figsize=(16, 20))
    for feature, axis in zip(top_features, axes.flat):
        data = [
            ml.loc[ml["label"] == class_name, feature].dropna()
            for class_name in CLASS_ORDER
        ]
        axis.boxplot(data, tick_labels=CLASS_ORDER, showfliers=False)
        axis.set_title(feature, fontsize=9)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Top 15 robust Pathological vs Non-Pathological feature effects")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_15_feature_boxplots.png", dpi=170)
    plt.close(fig)

    probabilities = probability_columns(
        model, encoder, model.predict_proba(ml[feature_columns])
    )["Pathological"]
    fig, axis = plt.subplots(figsize=(10, 6))
    for class_name in CLASS_ORDER:
        mask = ml["label"] == class_name
        axis.hist(
            probabilities[mask],
            bins=np.linspace(0, 1, 21),
            alpha=0.5,
            label=class_name,
            color=colors[class_name],
        )
    axis.set_xlabel("Saved-model Pathological probability")
    axis.set_ylabel("Records")
    axis.legend()
    axis.set_title("Predicted Pathological probability by stored class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pathological_probability_histogram.png", dpi=170)
    plt.close(fig)

    imputed = SimpleImputer(strategy="median").fit_transform(ml[feature_columns])
    standardized = StandardScaler().fit_transform(imputed)
    coordinates = PCA(n_components=2, random_state=train_models.RANDOM_STATE).fit_transform(
        standardized
    )
    fig, axis = plt.subplots(figsize=(10, 7))
    for class_name in CLASS_ORDER:
        mask = ml["label"] == class_name
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=28,
            alpha=0.7,
            label=class_name,
            color=colors[class_name],
        )
    axis.set_title("PCA visual diagnostic (not proof of separability)")
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_all_classes.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 7))
    path_mask = ml["label"] == "Pathological"
    axis.scatter(
        coordinates[~path_mask, 0],
        coordinates[~path_mask, 1],
        s=20,
        alpha=0.2,
        color="gray",
        label="Non-Pathological",
    )
    axis.scatter(
        coordinates[path_mask, 0],
        coordinates[path_mask, 1],
        s=55,
        alpha=0.9,
        color=colors["Pathological"],
        label="Pathological",
    )
    axis.set_title("PCA highlighting the 27 Pathological records (visual only)")
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_pathological_highlight.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(feature_columns)))
    axis.set_yticks(range(len(feature_columns)))
    axis.set_xticklabels(feature_columns, rotation=90, fontsize=5)
    axis.set_yticklabels(feature_columns, fontsize=5)
    axis.set_title("Spearman correlation of retained engineered features")
    fig.colorbar(image, ax=axis, fraction=0.025)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "retained_feature_correlation_heatmap.png", dpi=180)
    plt.close(fig)

    reason_counts = audit["pathological_reason"].value_counts()
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(reason_counts.index, reason_counts.values, color="#B91C1C")
    axis.set_ylabel("Pathological records")
    axis.set_title("Clinical reason for Pathological label")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pathological_reason_counts.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for feature, axis in zip(SIGNAL_QUALITY_COLUMNS, axes.flat):
        data = [
            ml.loc[ml["label"] == class_name, feature].dropna()
            for class_name in CLASS_ORDER
        ]
        axis.boxplot(data, tick_labels=CLASS_ORDER, showfliers=False)
        axis.set_title(feature)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Signal quality across stored classes")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "signal_quality_by_class.png", dpi=170)
    plt.close(fig)
    return top_features


def nearest_class_distances(
    ml,
    feature_columns,
    train_indices,
    test_indices,
    target,
    encoder,
    audit,
):
    """Measure each Pathological test case's nearest training case by class."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = imputer.fit_transform(ml.iloc[train_indices][feature_columns])
    test_values = imputer.transform(ml.iloc[test_indices][feature_columns])
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)
    train_target = target[train_indices]
    test_target = target[test_indices]
    path_code = int(encoder.transform(["Pathological"])[0])

    path_train_positions = np.flatnonzero(train_target == path_code)
    within_nearest = []
    for position in path_train_positions:
        other = path_train_positions[path_train_positions != position]
        distances = np.linalg.norm(
            train_scaled[other] - train_scaled[position], axis=1
        )
        within_nearest.append(float(np.min(distances)))
    outlier_threshold = float(np.quantile(within_nearest, 0.95))

    rows = []
    path_test_positions = np.flatnonzero(test_target == path_code)
    audit_by_id = audit.set_index("record_id")
    for test_position in path_test_positions:
        global_index = test_indices[test_position]
        record_id = ml.iloc[global_index]["record_id"]
        row = audit_by_id.loc[record_id].to_dict()
        row["record_id"] = record_id
        for class_name in CLASS_ORDER:
            class_code = int(encoder.transform([class_name])[0])
            candidate_positions = np.flatnonzero(train_target == class_code)
            distances = np.linalg.norm(
                train_scaled[candidate_positions] - test_scaled[test_position],
                axis=1,
            )
            nearest_local = candidate_positions[int(np.argmin(distances))]
            nearest_global = train_indices[nearest_local]
            prefix = class_name.lower()
            row[f"nearest_{prefix}_training_record_id"] = ml.iloc[nearest_global][
                "record_id"
            ]
            row[f"distance_to_nearest_{prefix}_training_record"] = float(
                np.min(distances)
            )
        path_distance = row["distance_to_nearest_pathological_training_record"]
        row["pathological_training_outlier_threshold_95pct"] = outlier_threshold
        row["appears_outlier_vs_pathological_training"] = (
            path_distance > outlier_threshold
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("record_id").reset_index(drop=True)


def current_classifier(model):
    """Return the saved model's terminal estimator."""
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        return model.named_steps["classifier"]
    return model


def uses_scaling(model):
    """Detect whether the saved pipeline standardizes its feature inputs."""
    return hasattr(model, "named_steps") and "scaler" in model.named_steps


def build_diagnostic_pipeline(model, balancing):
    """Clone the model and apply one balancing strategy."""
    classifier = clone(current_classifier(model))
    if hasattr(classifier, "class_weight"):
        classifier.set_params(
            class_weight="balanced"
            if balancing in {"class_weights_only", "class_weights_plus_smote"}
            else None
        )
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if uses_scaling(model):
        steps.append(("scaler", StandardScaler()))
    if balancing == "random_oversampling_only":
        steps.append(("random_oversampler", RandomOverSampler(random_state=42)))
    elif balancing in {"smote_only", "class_weights_plus_smote"}:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("classifier", classifier))
    return ImbalancedPipeline(steps=steps)


def binary_counts(target, predictions):
    """Return TN, FP, FN, TP for binary Pathological detection."""
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def binary_metrics(target, predictions, probabilities=None):
    """Calculate rare-class binary metrics with visible false alarms."""
    tn, fp, fn, tp = binary_counts(target, predictions)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    result = {
        "pathological_recall": recall,
        "pathological_precision": precision,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "f1": f1_score(target, predictions, zero_division=0),
        "f2": fbeta_score(target, predictions, beta=2, zero_division=0),
        "false_alarms": fp,
        "true_positives": tp,
    }
    if probabilities is not None:
        result["roc_auc"] = roc_auc_score(target, probabilities)
        result["pr_auc"] = average_precision_score(target, probabilities)
    return result


def split_stability_analysis(ml, feature_columns, encoder, model):
    """Measure performance variation over 30 predefined stratified splits."""
    encoded = encoder.transform(ml["label"])
    path_code = int(encoder.transform(["Pathological"])[0])
    rows = []
    for seed in SPLIT_SEEDS:
        train_indices, test_indices = train_test_split(
            np.arange(len(ml)),
            test_size=train_models.TEST_SIZE,
            random_state=seed,
            stratify=encoded,
        )
        pipeline = build_diagnostic_pipeline(model, "smote_only")
        pipeline.fit(ml.iloc[train_indices][feature_columns], encoded[train_indices])
        predictions = pipeline.predict(ml.iloc[test_indices][feature_columns])
        test_target = encoded[test_indices]
        path_mask = test_target == path_code
        rows.append(
            {
                "random_seed": seed,
                "pathological_training_samples": int(
                    np.sum(encoded[train_indices] == path_code)
                ),
                "pathological_test_samples": int(np.sum(path_mask)),
                "test_macro_f1": f1_score(
                    test_target, predictions, average="macro", zero_division=0
                ),
                "pathological_recall": recall_score(
                    test_target,
                    predictions,
                    labels=[path_code],
                    average="macro",
                    zero_division=0,
                ),
                "pathological_precision": precision_score(
                    test_target,
                    predictions,
                    labels=[path_code],
                    average="macro",
                    zero_division=0,
                ),
                "correct_pathological_cases": int(
                    np.sum(predictions[path_mask] == path_code)
                ),
                "false_pathological_predictions": int(
                    np.sum((predictions == path_code) & ~path_mask)
                ),
            }
        )
        print(f"Split stability seed {seed + 1}/{len(SPLIT_SEEDS)} complete")
    return pd.DataFrame(rows)


def threshold_analysis(
    ml,
    feature_columns,
    target,
    train_indices,
    test_indices,
    encoder,
    model,
):
    """Select a recall-weighted threshold using training OOF probabilities only."""
    path_code = int(encoder.transform(["Pathological"])[0])
    training_target = target[train_indices]
    binary_training = (training_target == path_code).astype(int)
    cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=train_models.RANDOM_STATE
    )
    oof_probabilities_all = cross_val_predict(
        clone(model),
        ml.iloc[train_indices][feature_columns],
        training_target,
        cv=cv,
        method="predict_proba",
        n_jobs=CV_N_JOBS,
    )
    model_classes = model.classes_
    path_position = int(np.flatnonzero(model_classes == path_code)[0])
    oof_path_probability = oof_probabilities_all[:, path_position]

    rows = []
    for threshold in THRESHOLDS:
        predictions = (oof_path_probability >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                **binary_metrics(binary_training, predictions),
            }
        )
    table = pd.DataFrame(rows)
    selected_index = table.sort_values(
        ["f2", "pathological_recall", "pathological_precision", "threshold"],
        ascending=[False, False, False, False],
    ).index[0]
    table["selected_from_training_oof"] = False
    table.loc[selected_index, "selected_from_training_oof"] = True

    selected_threshold = float(table.loc[selected_index, "threshold"])
    test_target = target[test_indices]
    binary_test = (test_target == path_code).astype(int)
    test_probabilities = model.predict_proba(
        ml.iloc[test_indices][feature_columns]
    )[:, path_position]
    threshold_predictions = (test_probabilities >= selected_threshold).astype(int)
    test_metrics = binary_metrics(binary_test, threshold_predictions)
    for metric_name, value in test_metrics.items():
        table[f"selected_threshold_test_{metric_name}"] = np.nan
        table.loc[selected_index, f"selected_threshold_test_{metric_name}"] = value
    return table, selected_threshold, test_metrics


def multiclass_cv_scorers(encoder):
    """Create repeated-CV scorers required for balancing comparisons."""
    path_code = int(encoder.transform(["Pathological"])[0])
    suspicious_code = int(encoder.transform(["Suspicious"])[0])
    normal_code = int(encoder.transform(["Normal"])[0])

    def recall_scorer(code):
        return make_scorer(
            recall_score, labels=[code], average="macro", zero_division=0
        )

    def precision_scorer(code):
        return make_scorer(
            precision_score, labels=[code], average="macro", zero_division=0
        )

    return {
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "pathological_recall": recall_scorer(path_code),
        "pathological_precision": precision_scorer(path_code),
        "suspicious_recall": recall_scorer(suspicious_code),
        "normal_recall": recall_scorer(normal_code),
    }


def summarize_cross_validation(scores, metrics):
    """Flatten repeated score arrays into mean/std columns."""
    row = {}
    for metric in metrics:
        values = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = float(np.std(values))
    return row


def balancing_comparison(ml, feature_columns, target, encoder, model):
    """Compare five balancing strategies with identical repeated CV folds."""
    strategies = [
        "no_balancing",
        "class_weights_only",
        "random_oversampling_only",
        "smote_only",
        "class_weights_plus_smote",
    ]
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=train_models.RANDOM_STATE,
    )
    scorers = multiclass_cv_scorers(encoder)
    rows = []
    for strategy in strategies:
        scores = cross_validate(
            build_diagnostic_pipeline(model, strategy),
            ml[feature_columns],
            target,
            cv=cv,
            scoring=scorers,
            n_jobs=CV_N_JOBS,
        )
        rows.append(
            {
                "balancing_method": strategy,
                **summarize_cross_validation(scores, scorers),
            }
        )
        print(f"Balancing comparison complete: {strategy}")
    return pd.DataFrame(rows).sort_values(
        "pathological_recall_mean", ascending=False, ignore_index=True
    )


def binary_diagnostic(ml, feature_columns, target, encoder, model):
    """Evaluate Pathological-vs-rest detection using repeated CV only."""
    path_code = int(encoder.transform(["Pathological"])[0])
    binary_target = (target == path_code).astype(int)
    pipeline = build_diagnostic_pipeline(model, "class_weights_plus_smote")
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=train_models.RANDOM_STATE,
    )
    scorers = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "specificity": make_scorer(
            recall_score, pos_label=0, zero_division=0
        ),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "f1": make_scorer(f1_score, zero_division=0),
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    }
    scores = cross_validate(
        pipeline,
        ml[feature_columns],
        binary_target,
        cv=cv,
        scoring=scorers,
        n_jobs=CV_N_JOBS,
    )
    row = summarize_cross_validation(scores, scorers)
    standard_cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=train_models.RANDOM_STATE
    )
    oof_predictions = cross_val_predict(
        pipeline,
        ml[feature_columns],
        binary_target,
        cv=standard_cv,
        method="predict",
        n_jobs=CV_N_JOBS,
    )
    row["standard_5fold_oof_confusion_matrix"] = json.dumps(
        confusion_matrix(binary_target, oof_predictions, labels=[0, 1]).tolist()
    )
    return pd.DataFrame([row])


def describe_split_stability(stability):
    """Return mean, standard deviation, minimum, maximum, and median summaries."""
    columns = [
        "test_macro_f1",
        "pathological_recall",
        "pathological_precision",
        "correct_pathological_cases",
        "false_pathological_predictions",
    ]
    return stability[columns].agg(["mean", "std", "min", "max", "median"]).T


def dataframe_to_markdown(dataframe):
    """Render compact Markdown tables without an optional dependency."""
    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return "NaN" if np.isnan(value) else f"{value:.4f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = list(dataframe.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in dataframe.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def determine_main_cause(
    alignment,
    disagreements,
    comparison,
    stability,
    balancing,
    binary_result,
):
    """Choose one allowed conclusion using generated diagnostic evidence."""
    if alignment["alignment_errors"] or len(disagreements):
        return "label consistency is the main problem"
    split_recall_std = stability["pathological_recall"].std()
    large_effects = int(comparison["clearly_separating_pathological"].sum())
    binary_recall = float(binary_result.iloc[0]["recall_mean"])
    best_multiclass_recall = float(balancing["pathological_recall_mean"].max())
    if split_recall_std > 0.20 and large_effects < 5:
        return "multiple factors contribute"
    if binary_recall > best_multiclass_recall + 0.15:
        return "the current multiclass decision strategy is the main problem"
    if large_effects < 3:
        return "feature separability is the main problem"
    return "data quantity is the main problem"


def make_report(
    labels_audited,
    disagreements,
    alignment,
    audit,
    subgroups,
    comparison,
    top_features,
    test_cases,
    stability,
    threshold_table,
    selected_threshold,
    selected_test_metrics,
    balancing,
    binary_result,
    main_cause,
    model,
    feature_columns,
    train_class_counts,
    test_class_counts,
):
    """Create an evidence-based Markdown report in plain language."""
    reasons = audit["pathological_reason"].value_counts().rename_axis(
        "pathological_reason"
    ).reset_index(name="records")
    quality_by_class = (
        labels_audited[["record_id", "label"]]
        .merge(pd.read_csv(FEATURES_PATH, dtype={"record_id": str}), on="record_id")
        .groupby("label")[SIGNAL_QUALITY_COLUMNS]
        .mean()
        .reset_index()
    )
    stability_summary = describe_split_stability(stability).reset_index(
        names="metric"
    )
    best_balancing = balancing.iloc[0]
    all_path_probability = audit["predicted_probability_pathological"]
    incorrect_path = audit.loc[~audit["correctly_classified"]]
    effect_sorted = comparison.assign(
        absolute_effect=comparison[
            "cliffs_delta_pathological_vs_non_pathological"
        ].abs()
    ).sort_values("absolute_effect", ascending=False)
    overlap_count = int(comparison["heavy_overlap_with_normal_and_suspicious"].sum())
    large_effect_count = int(comparison["clearly_separating_pathological"].sum())
    correlated_count = int(
        comparison["high_correlation_partners_abs_spearman_ge_0_95"]
        .fillna("")
        .ne("")
        .sum()
    )
    near_zero_count = int(comparison["near_zero_variance"].sum())
    high_missing_count = int((comparison["missing_percentage"] >= 20).sum())

    recommendations = [
        (
            "Collect or externally validate substantially more Pathological records; "
            "27 total and six in the fixed test set make recall estimates unstable."
        ),
        (
            f"Evaluate the training-selected Pathological threshold ({selected_threshold:.2f}) "
            "prospectively, recording its false alarms; do not alter production behavior yet."
        ),
        (
            f"Prioritize the `{best_balancing['balancing_method']}` balancing strategy "
            "and the binary diagnostic as controlled follow-up experiments, then clinically "
            "review errors by pathological_reason."
        ),
    ]
    model_name = type(current_classifier(model)).__name__
    report = f"""# Diagnostic Analysis of Poor Pathological Performance

## Executive conclusion

**{main_cause}.**

This conclusion is diagnostic, not a clinical claim. No labels, features,
split settings, production thresholds, or saved models were changed.

## 1. Record and label alignment

- Alignment errors found: **{alignment['alignment_errors']}**
- Merge order matches the deterministic feature/label merge: **{alignment['merge_order_matches']}**
- Stored-label disagreements with the independently reproduced rule: **{len(disagreements)}**
- Recalculated label counts: {labels_audited['recalculated_label'].value_counts().to_dict()}

Warnings:

{chr(10).join('- ' + message for message in alignment['messages']) if alignment['messages'] else '- None'}

The evidence indicates whether records, neonatal outcomes, labels, and features
are correctly keyed before considering model behavior.

## 2. Pathological cases and clinical reasons

There are **{len(audit)} Pathological records**. Their outcome-rule mechanisms are:

{dataframe_to_markdown(reasons)}

Subgroup performance:

{dataframe_to_markdown(subgroups)}

This specifically tests whether low-Apgar-only cases behave differently from
severe metabolic cases.

## 3. Signal quality

Mean quality measurements by class:

{dataframe_to_markdown(quality_by_class)}

Poor signal quality can make clinical morphology harder to summarize, but the
quality comparison must be considered alongside class overlap and sample size.

## 4. Feature overlap and redundancy

- Retained engineered features used by the saved model: **{len(feature_columns)}**
- Features with a large robust Pathological-vs-non-Pathological effect: **{large_effect_count}**
- Features with negligible effects against both Normal and Suspicious: **{overlap_count}**
- Near-zero-variance features: **{near_zero_count}**
- Features with at least 20% missing values: **{high_missing_count}**
- Features with at least one |Spearman correlation| ≥ 0.95 partner: **{correlated_count}**

Cliff's delta is used as the robust effect size:
`P(Pathological > comparison) - P(Pathological < comparison)`. It is
rank-based, handles skew/outliers better than a standardized mean difference,
and ranges from -1 to +1. Large absolute effects are clearer; values near zero
indicate heavy overlap.

Top robust effects:

{dataframe_to_markdown(effect_sorted[['feature', 'cliffs_delta_pathological_vs_non_pathological', 'effect_magnitude_pathological_vs_non_pathological']].head(15))}

PCA figures are visual diagnostics only and do not prove separability.

## 5. Current saved-model probabilities

- Saved model family: **{model_name}**
- Mean saved-model Pathological probability among all Pathological records:
  **{all_path_probability.mean():.4f}**
- Mean probability among incorrectly classified Pathological records:
  **{incorrect_path['predicted_probability_pathological'].mean():.4f}**
- Correctly classified Pathological records (in-sample training and held-out
  test records combined): **{int(audit['correctly_classified'].sum())}/{len(audit)}**

Training-record predictions are in-sample and optimistic; the six fixed test
records are the valid held-out evidence.

## 6. Fixed test split difficulty

- Training distribution: {train_class_counts}
- Test distribution: {test_class_counts}
- Pathological test IDs: {test_cases['record_id'].tolist()}
- Test Pathological outliers relative to Pathological training nearest-neighbor
  distances: **{int(test_cases['appears_outlier_vs_pathological_training'].sum())}/{len(test_cases)}**

Nearest-neighbor distances use median-imputed, training-standardized engineered
features. They are similarity diagnostics, not clinical distance measures.

{dataframe_to_markdown(test_cases)}

## 7. Instability across 30 predefined splits

No seed was selected for favorable results.

{dataframe_to_markdown(stability_summary)}

Mean Pathological recall is **{stability['pathological_recall'].mean():.4f}**
with standard deviation **{stability['pathological_recall'].std():.4f}**.
This quantifies how much the six-case test composition changes the conclusion.

## 8. Balancing and SMOTE

All strategies use the same `{model_name}` family/parameters and identical
repeated folds. Resampling occurs only inside each training fold.

{dataframe_to_markdown(balancing)}

Best repeated-CV Pathological recall: **{best_balancing['balancing_method']}**
at **{best_balancing['pathological_recall_mean']:.4f}**. Macro F1 and false
positive behavior must be considered with recall; recall alone is not a model
selection criterion.

## 9. Training-only threshold analysis

The selected threshold is **{selected_threshold:.2f}**, chosen by maximum
training out-of-fold F2 (which weights recall more than precision). The
untouched fixed test set was used once afterward.

- Selected-threshold test Pathological recall:
  **{selected_test_metrics['pathological_recall']:.4f}**
- Test Pathological precision:
  **{selected_test_metrics['pathological_precision']:.4f}**
- Test specificity: **{selected_test_metrics['specificity']:.4f}**
- Test false alarms: **{selected_test_metrics['false_alarms']}**

This can reveal moderate Pathological probabilities that lose the default
multiclass decision, but it does not authorize changing production behavior.

## 10. Binary Pathological diagnostic

{dataframe_to_markdown(binary_result)}

The repeated-CV binary Pathological recall is
**{binary_result.iloc[0]['recall_mean']:.4f}**. Comparing it with multiclass
results indicates whether competition between Normal and Suspicious is part of
the difficulty. This binary model is diagnostic only.

## 11. Most likely cause

The evidence-based conclusion is: **{main_cause}**.

Key constraints include only 27 Pathological records, six fixed-test cases,
the measured split-to-split recall variation, the robust feature-overlap
results, and the balancing/binary comparisons. A label-rule change is not
recommended merely to improve performance.

## 12. Prioritized next actions

1. {recommendations[0]}
2. {recommendations[1]}
3. {recommendations[2]}

## Generated figures

- `pathological_diagnostics/top_15_feature_boxplots.png`
- `pathological_diagnostics/pathological_probability_histogram.png`
- `pathological_diagnostics/pca_all_classes.png`
- `pathological_diagnostics/pca_pathological_highlight.png`
- `pathological_diagnostics/retained_feature_correlation_heatmap.png`
- `pathological_diagnostics/pathological_reason_counts.png`
- `pathological_diagnostics/signal_quality_by_class.png`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def print_terminal_summary(
    alignment,
    disagreements,
    audit,
    test_cases,
    stability,
    balancing,
    selected_threshold,
    binary_result,
    main_cause,
):
    """Print the diagnostic summary."""
    print("\nFinal pathological diagnostic summary")
    print("=====================================")
    print(f"Alignment errors found: {alignment['alignment_errors']}")
    print(f"Label disagreements found: {len(disagreements)}")
    print(
        "Pathological cases by reason: "
        f"{audit['pathological_reason'].value_counts().to_dict()}"
    )
    print(f"Current test Pathological IDs: {test_cases['record_id'].tolist()}")
    print(
        "Mean Pathological recall across repeated splits: "
        f"{stability['pathological_recall'].mean():.4f}"
    )
    print(
        "Best balancing method by repeated-CV Pathological recall: "
        f"{balancing.iloc[0]['balancing_method']} "
        f"({balancing.iloc[0]['pathological_recall_mean']:.4f})"
    )
    print(f"Best threshold from training cross-validation: {selected_threshold:.2f}")
    print(
        "Binary diagnostic Pathological recall: "
        f"{binary_result.iloc[0]['recall_mean']:.4f}"
    )
    print(f"Main diagnosed cause: {main_cause}")
    for path in [
        AUDIT_PATH,
        SUBGROUP_PATH,
        FEATURE_COMPARISON_PATH,
        TEST_CASES_PATH,
        SPLIT_STABILITY_PATH,
        THRESHOLD_PATH,
        BALANCING_PATH,
        BINARY_PATH,
        REPORT_PATH,
        FIGURES_DIR,
    ]:
        print(f"Saved: {path}")


def main():
    """Run the diagnostics without modifying project inputs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    labels, features, ml, model, encoder = load_inputs()
    feature_columns = retained_feature_columns(ml, model)
    alignment = validate_alignment(labels, features, ml)
    labels_audited, disagreements, recalculated_counts = recalculate_labels(labels)
    train_indices, test_indices, target = reproduce_split(ml, encoder)
    train_class_counts = pd.Series(
        encoder.inverse_transform(target[train_indices])
    ).value_counts().to_dict()
    test_class_counts = pd.Series(
        encoder.inverse_transform(target[test_indices])
    ).value_counts().to_dict()

    print(f"Alignment errors: {alignment['alignment_errors']}")
    for message in alignment["messages"]:
        print(f"WARNING: {message}")
    print(f"Label disagreements: {len(disagreements)}")
    print(f"Recalculated class counts: {recalculated_counts}")
    print(
        "Exact split class counts — train: "
        f"{train_class_counts}, "
        "test: "
        f"{test_class_counts}"
    )

    audit = create_pathological_audit(
        labels_audited,
        ml,
        model,
        encoder,
        feature_columns,
        train_indices,
        test_indices,
    )
    subgroups = summarize_pathological_subgroups(audit)
    comparison, correlation = feature_distribution_comparison(ml, feature_columns)
    top_features = create_visualizations(
        ml, audit, comparison, correlation, model, encoder, feature_columns
    )
    test_cases = nearest_class_distances(
        ml,
        feature_columns,
        train_indices,
        test_indices,
        target,
        encoder,
        audit,
    )

    stability = split_stability_analysis(ml, feature_columns, encoder, model)
    threshold_table, selected_threshold, selected_test_metrics = threshold_analysis(
        ml,
        feature_columns,
        target,
        train_indices,
        test_indices,
        encoder,
        model,
    )
    balancing = balancing_comparison(
        ml, feature_columns, target, encoder, model
    )
    binary_result = binary_diagnostic(
        ml, feature_columns, target, encoder, model
    )
    main_cause = determine_main_cause(
        alignment,
        disagreements,
        comparison,
        stability,
        balancing,
        binary_result,
    )

    audit.to_csv(AUDIT_PATH, index=False)
    subgroups.to_csv(SUBGROUP_PATH, index=False)
    comparison.to_csv(FEATURE_COMPARISON_PATH, index=False)
    test_cases.to_csv(TEST_CASES_PATH, index=False)
    stability.to_csv(SPLIT_STABILITY_PATH, index=False)
    threshold_table.to_csv(THRESHOLD_PATH, index=False)
    balancing.to_csv(BALANCING_PATH, index=False)
    binary_result.to_csv(BINARY_PATH, index=False)
    make_report(
        labels_audited,
        disagreements,
        alignment,
        audit,
        subgroups,
        comparison,
        top_features,
        test_cases,
        stability,
        threshold_table,
        selected_threshold,
        selected_test_metrics,
        balancing,
        binary_result,
        main_cause,
        model,
        feature_columns,
        train_class_counts,
        test_class_counts,
    )
    print_terminal_summary(
        alignment,
        disagreements,
        audit,
        test_cases,
        stability,
        balancing,
        selected_threshold,
        binary_result,
        main_cause,
    )


if __name__ == "__main__":
    main()
