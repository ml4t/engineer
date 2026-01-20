"""Shared utilities for labeling module."""

import warnings

import polars as pl

# Datetime types for timestamp detection
_DATETIME_TYPES = (pl.Datetime, pl.Date)


def resolve_timestamp_col(
    data: pl.DataFrame,
    timestamp_col: str | None,
) -> str | None:
    """Resolve timestamp column for chronological sorting.

    Detection priority:
    1. Explicit `timestamp_col` parameter (if provided and exists)
    2. Dtype-based detection (pl.Datetime, pl.Date columns)
    3. None if no datetime columns found

    Args:
        data: Input DataFrame
        timestamp_col: User-specified timestamp column, or None for auto-detection

    Returns:
        Column name to use for sorting, or None if not found

    Warns:
        If multiple datetime columns found and none explicitly specified
    """
    # Explicit specification takes priority
    if timestamp_col is not None:
        if timestamp_col in data.columns:
            return timestamp_col
        else:
            warnings.warn(
                f"Specified timestamp_col '{timestamp_col}' not found in data. "
                f"Available columns: {data.columns}",
                UserWarning,
                stacklevel=3,
            )
            # Fall through to auto-detection

    # Dtype-based detection (more robust than name matching)
    datetime_cols = [
        col for col in data.columns
        if data[col].dtype in _DATETIME_TYPES
    ]

    if len(datetime_cols) == 1:
        return datetime_cols[0]
    elif len(datetime_cols) > 1:
        # Ambiguous - warn and use first one
        warnings.warn(
            f"Multiple datetime columns found: {datetime_cols}. "
            f"Using '{datetime_cols[0]}' for sorting. "
            f"Specify timestamp_col explicitly to avoid ambiguity.",
            UserWarning,
            stacklevel=3,
        )
        return datetime_cols[0]

    # No datetime columns found
    return None
