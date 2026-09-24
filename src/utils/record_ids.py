"""Shared normalization for patient record identifiers."""

import pandas as pd


def normalize_record_id(values: pd.Series) -> pd.Series:
    """Normalize numeric and string record identifiers to stable strings."""
    return values.astype(str).str.replace(r"\.0$", "", regex=True)
