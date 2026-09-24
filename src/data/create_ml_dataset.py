"""Create the machine learning dataset by merging features with labels.

This script does not train a model. It only combines:

- data/processed/clinical_features.csv
- data/processed/labels.csv

The merged dataset is saved to:

- data/processed/ml_dataset.csv

Run from the project root:

    python src/data/create_ml_dataset.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "clinical_features.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"


def load_csv(path):
    """Load a CSV file and give a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    return pd.read_csv(path)


def create_ml_dataset(features_df, labels_df):
    """Merge feature rows with their class labels using record_id."""
    return features_df.merge(labels_df, on="record_id", how="inner")


def print_dataset_summary(dataset_df):
    """Print simple checks for the merged dataset."""
    print(f"Number of rows: {dataset_df.shape[0]}")
    print(f"Number of columns: {dataset_df.shape[1]}")

    print("\nMissing values per column")
    print("-------------------------")
    print(dataset_df.isna().sum())

    print("\nClass distribution")
    print("------------------")
    print(dataset_df["label"].value_counts())


def main():
    """Load features and labels, merge them, and save the ML dataset."""
    features_df = load_csv(FEATURES_PATH)
    labels_df = load_csv(LABELS_PATH)

    ml_dataset_df = create_ml_dataset(features_df, labels_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ml_dataset_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved merged ML dataset to: {OUTPUT_PATH}")
    print_dataset_summary(ml_dataset_df)


if __name__ == "__main__":
    main()
