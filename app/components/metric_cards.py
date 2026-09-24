"""Consistent metric-card helpers."""

from collections.abc import Iterable

import streamlit as st


def metric_row(items: Iterable[tuple[str, str, str | None]]) -> None:
    """Render a responsive row of Streamlit metrics."""
    values = list(items)
    columns = st.columns(len(values))
    for column, (label, value, help_text) in zip(columns, values):
        with column:
            st.metric(label, value, help=help_text)


def percentage(value: float, digits: int = 1) -> str:
    """Format a [0, 1] metric as a percentage."""
    return f"{100 * float(value):.{digits}f}%"


def decimal(value: float, digits: int = 3) -> str:
    """Format a numeric metric consistently."""
    return f"{float(value):.{digits}f}"
