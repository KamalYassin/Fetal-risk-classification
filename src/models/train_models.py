"""Train and evaluate fetal-risk classification models without data leakage.

The script compares original baseline models with class-balanced, SMOTE-based,
and tuned alternatives.  SMOTE is kept inside imbalanced-learn pipelines, so it
is fitted only on training folds during cross-validation and never modifies the
held-out test set.

Run from the project root::

    python src/models/train_models.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ml_dataset.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "feature_importance.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

TARGET_COLUMN = "label"
LEAKAGE_COLUMNS = [
    "record_id",
    "pH",
    "BE",
    "BDecf",
    "Apgar5",
    "feature_quality_flag",
]
TEST_SIZE = 0.20
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5
# Run CV folds sequentially to avoid nested process pools; Random Forest itself
# still builds trees in parallel through n_jobs=-1.
CROSS_VALIDATION_N_JOBS = 1
TOP_FEATURE_COUNT = 15

# Keep the grid small enough for reproducible project runs while covering the
# main Random Forest complexity and regularization controls.
RANDOM_FOREST_PARAM_GRID = {
    "classifier__n_estimators": [300, 500],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
    "classifier__max_features": ["sqrt", 0.5],
}


def load_dataset(dataset_path=DATASET_PATH):
    """Load the prepared machine-learning dataset from disk."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Prepare ml_dataset.csv first."
        )
    dataset = pd.read_csv(dataset_path)
    if dataset.empty:
        raise ValueError(f"Dataset at {dataset_path} is empty.")
    return dataset


def prepare_features_and_target(dataset):
    """Create leakage-free numeric features and LabelEncoder target values."""
    required_columns = {TARGET_COLUMN, *LEAKAGE_COLUMNS}
    missing_columns = sorted(required_columns.difference(dataset.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )
    if dataset[TARGET_COLUMN].isna().any():
        raise ValueError("The label column contains missing values.")

    features = dataset.drop(columns=[TARGET_COLUMN, *LEAKAGE_COLUMNS])
    non_numeric_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            "All retained model features must be numeric. Non-numeric columns: "
            + ", ".join(non_numeric_columns)
        )

    missing_feature_counts = features.isna().sum()
    missing_feature_counts = missing_feature_counts[missing_feature_counts > 0]
    if not missing_feature_counts.empty:
        details = ", ".join(
            f"{column}={count}" for column, count in missing_feature_counts.items()
        )
        raise ValueError(
            "Retained features contain missing values. Handle them before training: "
            + details
        )

    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(dataset[TARGET_COLUMN])
    return features, encoded_target, label_encoder


def split_dataset(features, target):
    """Create the one shared stratified 80/20 train/test split."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def make_cross_validator():
    """Return the reproducible stratified five-fold training validator."""
    return StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def build_baseline_models():
    """Return the original, unbalanced Logistic Regression and Random Forest."""
    return {
        "Baseline Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Baseline Random Forest": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_balanced_smote_models():
    """Return balanced estimators with fold-safe SMOTE preprocessing.

    Logistic Regression is standardized before SMOTE because SMOTE uses
    nearest-neighbor distances.  Random Forest receives the original feature
    units, and both classifiers use balanced class weights.
    """
    return {
        "Balanced + SMOTE Logistic Regression": ImbalancedPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Balanced + SMOTE Random Forest": ImbalancedPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def build_tuned_random_forest_search(cross_validator):
    """Build macro-F1 GridSearchCV around a fold-safe SMOTE/RF pipeline."""
    pipeline = ImbalancedPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=RANDOM_FOREST_PARAM_GRID,
        scoring="f1_macro",
        cv=cross_validator,
        n_jobs=CROSS_VALIDATION_N_JOBS,
        refit=True,
        return_train_score=False,
    )


def evaluate_predictions(target, predictions, label_encoder):
    """Calculate aggregate metrics and a separate recall for every class."""
    encoded_labels = np.arange(len(label_encoder.classes_))
    class_recalls = recall_score(
        target,
        predictions,
        labels=encoded_labels,
        average=None,
        zero_division=0,
    )
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "Macro Precision": precision_score(
            target, predictions, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            target, predictions, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            target, predictions, average="macro", zero_division=0
        ),
    }
    for class_name, class_recall in zip(label_encoder.classes_, class_recalls):
        metrics[f"Recall - {class_name}"] = class_recall
    return metrics


def print_detailed_evaluation(model_name, target, predictions, label_encoder):
    """Print a classification report and labeled confusion matrix."""
    class_names = label_encoder.classes_
    encoded_labels = np.arange(len(class_names))
    print(f"\n{model_name}")
    print("=" * len(model_name))
    print("\nClassification Report")
    print(
        classification_report(
            target,
            predictions,
            labels=encoded_labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(target, predictions, labels=encoded_labels)
    labeled_matrix = pd.DataFrame(
        matrix,
        index=[f"Actual {name}" for name in class_names],
        columns=[f"Predicted {name}" for name in class_names],
    )
    print("Confusion Matrix")
    print(labeled_matrix.to_string())


def cross_validate_and_fit_models(
    models,
    features_train,
    features_test,
    target_train,
    target_test,
    label_encoder,
    cross_validator,
):
    """Cross-validate, fit, and test fixed-parameter model configurations."""
    metric_rows = []
    trained_models = {}
    for model_name, model in models.items():
        print(f"\nCross-validating {model_name}...")
        cv_scores = cross_val_score(
            model,
            features_train,
            target_train,
            scoring="f1_macro",
            cv=cross_validator,
            n_jobs=CROSS_VALIDATION_N_JOBS,
        )
        print(
            f"Training CV Macro F1: {cv_scores.mean():.4f} "
            f"+/- {cv_scores.std():.4f}"
        )

        model.fit(features_train, target_train)
        predictions = model.predict(features_test)
        metrics = evaluate_predictions(target_test, predictions, label_encoder)
        metric_rows.append(
            {
                "Model": model_name,
                "CV Macro F1": cv_scores.mean(),
                **metrics,
            }
        )
        trained_models[model_name] = model
        print_detailed_evaluation(
            model_name, target_test, predictions, label_encoder
        )
    return metric_rows, trained_models


def tune_and_evaluate_random_forest(
    search,
    features_train,
    features_test,
    target_train,
    target_test,
    label_encoder,
):
    """Tune Random Forest on training folds and evaluate it once on test data."""
    model_name = "Tuned Balanced + SMOTE Random Forest"
    combinations = int(
        np.prod([len(values) for values in RANDOM_FOREST_PARAM_GRID.values()])
    )
    print(
        f"\nTuning {model_name}: {combinations} parameter combinations x "
        f"{CROSS_VALIDATION_FOLDS} folds..."
    )
    search.fit(features_train, target_train)
    tuned_model = search.best_estimator_
    predictions = tuned_model.predict(features_test)
    metrics = evaluate_predictions(target_test, predictions, label_encoder)
    metric_row = {
        "Model": model_name,
        "CV Macro F1": search.best_score_,
        **metrics,
    }

    print(f"Best training CV Macro F1: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")
    print_detailed_evaluation(
        model_name, target_test, predictions, label_encoder
    )
    return metric_row, tuned_model


def save_random_forest_feature_importance(tuned_pipeline, feature_names):
    """Print and save importance from the tuned Random Forest classifier."""
    random_forest = tuned_pipeline.named_steps["classifier"]
    importance = pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance": random_forest.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False, ignore_index=True)
    print(f"\nTop {TOP_FEATURE_COUNT} Tuned Random Forest Features")
    print("=" * 45)
    print(importance.head(TOP_FEATURE_COUNT).to_string(index=False))

    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"\nSaved feature importance to: {FEATURE_IMPORTANCE_PATH}")
    return importance


def report_pathological_recall_change(summary):
    """Report rare-class recall changes against the matching baselines."""
    summary_by_model = summary.set_index("Model")
    recall_column = "Recall - Pathological"
    comparisons = [
        (
            "Logistic Regression",
            "Baseline Logistic Regression",
            "Balanced + SMOTE Logistic Regression",
        ),
        (
            "Random Forest",
            "Baseline Random Forest",
            "Tuned Balanced + SMOTE Random Forest",
        ),
    ]
    print("\nPathological Recall Improvement")
    print("===============================")
    for family_name, baseline_name, improved_name in comparisons:
        baseline_recall = summary_by_model.loc[baseline_name, recall_column]
        improved_recall = summary_by_model.loc[improved_name, recall_column]
        change = improved_recall - baseline_recall
        status = "improved" if change > 0 else "did not improve"
        print(
            f"{family_name}: {baseline_recall:.1%} -> {improved_recall:.1%} "
            f"({change:+.1%}); {status}."
        )


def save_pickle(value, output_path):
    """Save one Python object, creating its output directory when needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        pickle.dump(value, output_file)


def print_label_mapping(label_encoder):
    """Show the LabelEncoder mapping used by every classifier."""
    print("Label encoding")
    print("--------------")
    for encoded_value, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name}: {encoded_value}")


def main():
    """Run baseline/balanced training, RF tuning, testing, and saving."""
    dataset = load_dataset()
    features, target, label_encoder = prepare_features_and_target(dataset)
    features_train, features_test, target_train, target_test = split_dataset(
        features, target
    )
    cross_validator = make_cross_validator()

    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Rows: {len(dataset)}")
    print(f"Retained engineered features: {features.shape[1]}")
    print(f"Training rows: {len(features_train)}")
    print(f"Untouched test rows: {len(features_test)}")
    encoded_classes, class_counts = np.unique(target_train, return_counts=True)
    named_class_counts = {
        label_encoder.inverse_transform([encoded_class])[0]: int(count)
        for encoded_class, count in zip(encoded_classes, class_counts)
    }
    print(f"Training class counts: {named_class_counts}\n")
    print_label_mapping(label_encoder)

    fixed_models = {
        **build_baseline_models(),
        **build_balanced_smote_models(),
    }
    metric_rows, trained_models = cross_validate_and_fit_models(
        fixed_models,
        features_train,
        features_test,
        target_train,
        target_test,
        label_encoder,
        cross_validator,
    )

    search = build_tuned_random_forest_search(cross_validator)
    tuned_metric_row, tuned_random_forest = tune_and_evaluate_random_forest(
        search,
        features_train,
        features_test,
        target_train,
        target_test,
        label_encoder,
    )
    metric_rows.append(tuned_metric_row)
    trained_models[tuned_metric_row["Model"]] = tuned_random_forest

    summary = pd.DataFrame(metric_rows).sort_values(
        by="Macro F1", ascending=False, ignore_index=True
    )
    print("\nModel Comparison (untouched test set, sorted by Macro F1)")
    print("=========================================================")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    report_pathological_recall_change(summary)

    save_random_forest_feature_importance(
        tuned_random_forest, features.columns
    )

    best_model_name = summary.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    save_pickle(best_model, BEST_MODEL_PATH)
    save_pickle(label_encoder, LABEL_ENCODER_PATH)
    best_pathological_recall = summary.iloc[0]["Recall - Pathological"]

    print(f"\nBest model by untouched-test Macro F1: {best_model_name}")
    print(f"Best-model Pathological recall: {best_pathological_recall:.1%}")
    print(f"Saved best model to: {BEST_MODEL_PATH}")
    print(f"Saved label encoder to: {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
