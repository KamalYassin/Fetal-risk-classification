"""Schema validation and inference for the saved model."""

import numpy as np
import pandas as pd


def prepare_features(frame: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    """Return numeric model columns in the order used during fitting."""
    missing = [column for column in expected_features if column not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(missing)
        )
    prepared = frame.loc[:, expected_features].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid = prepared.columns[prepared.isna().any()].tolist()
    if invalid:
        raise ValueError(
            "The saved pipeline requires finite numeric values. "
            "Invalid or missing values were found in: " + ", ".join(invalid)
        )
    return prepared


def predict_frame(
    frame: pd.DataFrame,
    model,
    encoder,
    expected_features: list[str],
) -> pd.DataFrame:
    """Predict classes and probabilities without fitting or writing artifacts."""
    prepared = prepare_features(frame, expected_features)
    encoded = np.asarray(model.predict(prepared), dtype=int)
    predicted_labels = encoder.inverse_transform(encoded)
    probabilities = np.asarray(model.predict_proba(prepared), dtype=float)
    encoded_classes = np.asarray(model.classes_, dtype=int)
    probability_labels = encoder.inverse_transform(encoded_classes)

    result = pd.DataFrame(index=frame.index)
    if "record_id" in frame:
        result["record_id"] = frame["record_id"].astype(str)
    result["predicted_label"] = predicted_labels
    for column_index, label in enumerate(probability_labels):
        result[f"probability_{str(label).lower()}"] = probabilities[
            :, column_index
        ]
    return result


def probability_mapping(result_row: pd.Series) -> dict[str, float]:
    """Convert one prediction row to display labels."""
    return {
        label: float(result_row[f"probability_{label.lower()}"])
        for label in ["Normal", "Suspicious", "Pathological"]
    }


def explain_prediction(
    frame: pd.DataFrame,
    model,
    encoder,
    expected_features: list[str],
) -> dict:
    """Explain one multiclass Logistic Regression prediction exactly.

    Each feature contribution is its standardized value multiplied by the
    difference between the predicted-class and runner-up coefficients. Positive
    values support the predicted class over the runner-up; negative values
    oppose it. This decomposes the fitted logit difference and is a model
    explanation, not a clinical or causal explanation.
    """
    if len(frame) != 1:
        raise ValueError("Prediction explanations require exactly one row")
    if not hasattr(model, "named_steps"):
        raise ValueError("The saved model does not expose pipeline steps")
    scaler = model.named_steps.get("scaler")
    classifier = model.named_steps.get("classifier")
    if scaler is None or classifier is None or not hasattr(classifier, "coef_"):
        raise ValueError(
            "The saved artifact does not support coefficient explanations"
        )

    prepared = prepare_features(frame, expected_features)
    probabilities = np.asarray(model.predict_proba(prepared)[0], dtype=float)
    class_order = np.asarray(model.classes_, dtype=int)
    ranked_positions = np.argsort(probabilities)[::-1]
    predicted_position = int(ranked_positions[0])
    runner_up_position = int(ranked_positions[1])
    predicted_encoded = int(class_order[predicted_position])
    runner_up_encoded = int(class_order[runner_up_position])
    predicted_label = str(encoder.inverse_transform([predicted_encoded])[0])
    runner_up_label = str(encoder.inverse_transform([runner_up_encoded])[0])

    standardized = np.asarray(scaler.transform(prepared)[0], dtype=float)
    coefficient_difference = (
        classifier.coef_[predicted_position]
        - classifier.coef_[runner_up_position]
    )
    contributions = standardized * coefficient_difference
    intercept_difference = float(
        classifier.intercept_[predicted_position]
        - classifier.intercept_[runner_up_position]
    )

    contribution_table = pd.DataFrame(
        {
            "feature": expected_features,
            "value": prepared.iloc[0].to_numpy(float),
            "standardized_value": standardized,
            "coefficient_difference": coefficient_difference,
            "contribution": contributions,
        }
    )
    contribution_table["direction"] = np.where(
        contribution_table["contribution"] >= 0,
        f"Supports {predicted_label}",
        f"Supports {runner_up_label}",
    )
    contribution_table["absolute_contribution"] = contribution_table[
        "contribution"
    ].abs()
    contribution_table = contribution_table.sort_values(
        "absolute_contribution",
        ascending=False,
        ignore_index=True,
    )
    return {
        "predicted_label": predicted_label,
        "runner_up_label": runner_up_label,
        "predicted_probability": float(probabilities[predicted_position]),
        "runner_up_probability": float(probabilities[runner_up_position]),
        "intercept_difference": intercept_difference,
        "contributions": contribution_table,
    }
