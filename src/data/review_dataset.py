"""Review processed CTG datasets before model training.

This script checks the labels, extracted clinical features, and merged ML
dataset. It does not train a model.

Run from the project root:

    python src/data/review_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LABELS_PATH = PROCESSED_DIR / "labels.csv"
FEATURES_PATH = PROCESSED_DIR / "clinical_features.csv"
ML_DATASET_PATH = PROCESSED_DIR / "ml_dataset.csv"
SUMMARY_OUTPUT_PATH = PROCESSED_DIR / "dataset_review_summary.csv"

IMPORTANT_FEATURES = [
    "baseline_fhr",
    "short_term_variability",
    "long_term_variability",
    "acceleration_count",
    "deceleration_count",
    "contraction_count",
    "decelerations_per_contraction",
]


def load_csv(path):
    """Load a CSV file and show a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    return pd.read_csv(path)


def print_section(title):
    """Print a simple section heading."""
    print(f"\n{title}")
    print("-" * len(title))


def print_basic_file_review(name, dataframe):
    """Print shape, columns, and first rows for one dataset."""
    print_section(name)
    print(f"Shape: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns")

    print("\nColumns:")
    print(list(dataframe.columns))

    print("\nFirst 5 rows:")
    print(dataframe.head())


def print_class_distribution(name, dataframe):
    """Print class counts and percentages from a dataframe with a label column."""
    print_section(f"Class distribution: {name}")

    counts = dataframe["label"].value_counts(dropna=False)
    percentages = dataframe["label"].value_counts(normalize=True, dropna=False) * 100

    for label, count in counts.items():
        print(f"{label}: {count} ({percentages[label]:.1f}%)")


def print_duplicate_record_check(name, dataframe):
    """Print duplicate record_id information."""
    duplicate_count = dataframe["record_id"].duplicated().sum()
    print(f"{name}: {duplicate_count} duplicate record_id values")

    if duplicate_count > 0:
        duplicate_ids = dataframe.loc[
            dataframe["record_id"].duplicated(keep=False), "record_id"
        ].sort_values()
        print(duplicate_ids.to_string(index=False))


def print_record_id_alignment(labels_df, features_df):
    """Check whether labels and features contain matching record IDs."""
    label_ids = set(labels_df["record_id"])
    feature_ids = set(features_df["record_id"])

    labels_missing_features = sorted(label_ids - feature_ids)
    features_missing_labels = sorted(feature_ids - label_ids)

    print_section("Record ID alignment")
    print(
        "All record_ids in labels.csv exist in clinical_features.csv: "
        f"{len(labels_missing_features) == 0}"
    )
    print(
        "All record_ids in clinical_features.csv exist in labels.csv: "
        f"{len(features_missing_labels) == 0}"
    )

    if labels_missing_features:
        print("\nLabels missing features:")
        print(labels_missing_features)

    if features_missing_labels:
        print("\nFeatures missing labels:")
        print(features_missing_labels)


def get_numeric_feature_columns(ml_dataset_df):
    """Return numeric columns, excluding record_id and label-like metadata."""
    excluded_columns = {"record_id", "label"}
    return [
        column
        for column in ml_dataset_df.select_dtypes(include=[np.number]).columns
        if column not in excluded_columns
    ]


def print_numeric_summary(ml_dataset_df):
    """Print summary statistics for numeric columns in the ML dataset."""
    numeric_columns = get_numeric_feature_columns(ml_dataset_df)

    print_section("Numeric summary statistics")
    print(ml_dataset_df[numeric_columns].describe().transpose())


def print_important_feature_summary(ml_dataset_df):
    """Print min, max, mean, and std for selected important features."""
    print_section("Important feature summary")

    for feature in IMPORTANT_FEATURES:
        if feature not in ml_dataset_df.columns:
            print(f"{feature}: missing from dataset")
            continue

        values = ml_dataset_df[feature]
        print(
            f"{feature}: "
            f"min={values.min():.3f}, "
            f"max={values.max():.3f}, "
            f"mean={values.mean():.3f}, "
            f"std={values.std():.3f}"
        )


def print_suspicious_values(ml_dataset_df):
    """Print potentially suspicious feature values for manual review."""
    print_section("Suspicious feature values")

    checks = {
        "baseline_fhr below 80": ml_dataset_df["baseline_fhr"] < 80,
        "baseline_fhr above 200": ml_dataset_df["baseline_fhr"] > 200,
        "percent_missing_fhr above 50": ml_dataset_df["percent_missing_fhr"] > 50,
        "negative acceleration_count": ml_dataset_df["acceleration_count"] < 0,
        "negative deceleration_count": ml_dataset_df["deceleration_count"] < 0,
        "negative contraction_count": ml_dataset_df["contraction_count"] < 0,
    }

    numeric_columns = ml_dataset_df.select_dtypes(include=[np.number]).columns
    infinite_mask = np.isinf(ml_dataset_df[numeric_columns]).any(axis=1)
    checks["infinite values"] = infinite_mask

    any_suspicious = False

    for check_name, mask in checks.items():
        matching_rows = ml_dataset_df.loc[mask, "record_id"]
        count = len(matching_rows)
        print(f"{check_name}: {count}")

        if count > 0:
            any_suspicious = True
            print(matching_rows.to_string(index=False))

    if not any_suspicious:
        print("No suspicious values found by these checks.")


def create_summary_rows(labels_df, features_df, ml_dataset_df):
    """Create rows for dataset_review_summary.csv."""
    summary_rows = [
        {
            "section": "shape",
            "item": "labels.csv rows",
            "value": labels_df.shape[0],
        },
        {
            "section": "shape",
            "item": "labels.csv columns",
            "value": labels_df.shape[1],
        },
        {
            "section": "shape",
            "item": "clinical_features.csv rows",
            "value": features_df.shape[0],
        },
        {
            "section": "shape",
            "item": "clinical_features.csv columns",
            "value": features_df.shape[1],
        },
        {
            "section": "shape",
            "item": "ml_dataset.csv rows",
            "value": ml_dataset_df.shape[0],
        },
        {
            "section": "shape",
            "item": "ml_dataset.csv columns",
            "value": ml_dataset_df.shape[1],
        },
    ]

    for label, count in labels_df["label"].value_counts().items():
        summary_rows.append(
            {
                "section": "labels_class_distribution",
                "item": label,
                "value": count,
            }
        )

    for label, count in ml_dataset_df["label"].value_counts().items():
        summary_rows.append(
            {
                "section": "ml_dataset_class_distribution",
                "item": label,
                "value": count,
            }
        )

    for column, missing_count in ml_dataset_df.isna().sum().items():
        summary_rows.append(
            {
                "section": "missing_values",
                "item": column,
                "value": missing_count,
            }
        )

    for feature in IMPORTANT_FEATURES:
        if feature not in ml_dataset_df.columns:
            continue

        values = ml_dataset_df[feature]
        summary_rows.extend(
            [
                {"section": "important_feature_min", "item": feature, "value": values.min()},
                {"section": "important_feature_max", "item": feature, "value": values.max()},
                {"section": "important_feature_mean", "item": feature, "value": values.mean()},
                {"section": "important_feature_std", "item": feature, "value": values.std()},
            ]
        )

    return summary_rows


def save_review_summary(labels_df, features_df, ml_dataset_df):
    """Save a compact review summary CSV."""
    summary_rows = create_summary_rows(labels_df, features_df, ml_dataset_df)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    print(f"\nSaved review summary to: {SUMMARY_OUTPUT_PATH}")


def main():
    """Run all dataset review checks."""
    labels_df = load_csv(LABELS_PATH)
    features_df = load_csv(FEATURES_PATH)
    ml_dataset_df = load_csv(ML_DATASET_PATH)

    print_basic_file_review("labels.csv", labels_df)
    print_basic_file_review("clinical_features.csv", features_df)
    print_basic_file_review("ml_dataset.csv", ml_dataset_df)

    print_class_distribution("labels.csv", labels_df)
    print_class_distribution("ml_dataset.csv", ml_dataset_df)

    print_section("Missing values in ml_dataset.csv")
    print(ml_dataset_df.isna().sum())

    print_section("Duplicate record_id check")
    print_duplicate_record_check("labels.csv", labels_df)
    print_duplicate_record_check("clinical_features.csv", features_df)
    print_duplicate_record_check("ml_dataset.csv", ml_dataset_df)

    print_record_id_alignment(labels_df, features_df)
    print_numeric_summary(ml_dataset_df)
    print_important_feature_summary(ml_dataset_df)
    print_suspicious_values(ml_dataset_df)
    save_review_summary(labels_df, features_df, ml_dataset_df)


if __name__ == "__main__":
    main()
