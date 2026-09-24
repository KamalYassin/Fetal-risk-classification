"""Loading and introspection helpers for saved inference artifacts."""

import pickle
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the fitted model and label encoder without refitting."""
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError("Saved model or label encoder is missing")
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with ENCODER_PATH.open("rb") as encoder_file:
        encoder = pickle.load(encoder_file)
    return model, encoder


def model_metadata(model, encoder) -> dict:
    """Describe the artifact from fitted attributes rather than assumptions."""
    classifier = (
        model.named_steps.get("classifier")
        if hasattr(model, "named_steps")
        else model
    )
    features = [str(value) for value in model.feature_names_in_]
    return {
        "artifact_type": type(model).__name__,
        "classifier": type(classifier).__name__,
        "feature_count": len(features),
        "feature_names": features,
        "classes": [str(value) for value in encoder.classes_],
    }
