"""Read-only access to committed datasets, reports, and raw CTG records."""

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FINAL_COMPARISON_DIR = REPORTS_DIR / "final_model_comparison"
FINAL_TEMPORAL_DIR = REPORTS_DIR / "final_temporal_simplification"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


@st.cache_data(show_spinner=False)
def read_csv(relative_path: str) -> pd.DataFrame:
    """Read a project CSV without modifying it."""
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required project data is missing: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_markdown(relative_path: str) -> str:
    """Read a committed Markdown report."""
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required report is missing: {path}")
    return path.read_text(encoding="utf-8")


def final_dataset() -> pd.DataFrame:
    return read_csv("data/processed/ml_dataset_final_temporal.csv")


def clinical_features() -> pd.DataFrame:
    return read_csv("data/processed/clinical_features.csv")


def labels() -> pd.DataFrame:
    return read_csv("data/processed/labels.csv")


def fixed_model_results() -> pd.DataFrame:
    return read_csv("reports/final_model_comparison/fixed_test_results.csv")


def repeated_cv_results() -> pd.DataFrame:
    return read_csv("reports/final_model_comparison/repeated_cv_results.csv")


def repeated_cv_summary() -> pd.DataFrame:
    return read_csv("reports/final_model_comparison/repeated_cv_summary.csv")


def repeated_split_results() -> pd.DataFrame:
    return read_csv("reports/final_model_comparison/repeated_split_results.csv")


def repeated_split_summary() -> pd.DataFrame:
    return read_csv("reports/final_model_comparison/repeated_split_summary.csv")


def temporal_cv_summary() -> pd.DataFrame:
    return read_csv(
        "reports/final_temporal_simplification/repeated_cv_summary.csv"
    )


def feature_importance() -> pd.DataFrame:
    return read_csv("reports/feature_importance.csv")


def temporal_patient_summaries() -> pd.DataFrame:
    return read_csv(
        "reports/temporal_information_audit/temporal_patient_summaries.csv"
    )


def extract_markdown_section(text: str, heading: str) -> str:
    """Extract one H2 section without interpreting its claims."""
    marker = f"## {heading}"
    if marker not in text:
        return ""
    body = text.split(marker, 1)[1]
    if "\n## " in body:
        body = body.split("\n## ", 1)[0]
    return body.strip()


@st.cache_data(show_spinner="Loading and analysing the selected CTG record…")
def ctg_record(record_id: str) -> dict:
    """Load one record through the research preprocessing functions."""
    from src.data.load_ctg import (
        find_record_path,
        get_fhr_uc_signals,
        load_record,
    )
    from src.features.extract_clinical_features import (
        extract_features_for_record,
    )

    record_path = find_record_path(str(record_id), RAW_DIR)
    record = load_record(str(record_id), RAW_DIR)
    raw_fhr, raw_uc = get_fhr_uc_signals(record)
    if raw_fhr is None or raw_uc is None:
        raise ValueError(f"Record {record_id} does not contain both FHR and UC")
    features, analysis = extract_features_for_record(
        record_path.with_suffix(".hea"),
        return_analysis=True,
    )
    return {
        "record_id": str(record.record_name),
        "sampling_frequency": float(record.fs),
        "signal_length": int(record.sig_len),
        "raw_uc": raw_uc,
        "features": features,
        "analysis": analysis,
    }
