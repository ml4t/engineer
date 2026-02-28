"""Shared utilities for labeling module."""

from __future__ import annotations

import re
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ml4t.engineer.core.exceptions import DataValidationError

if TYPE_CHECKING:
    from ml4t.engineer.config import DataContractConfig, LabelingConfig

# Datetime types for timestamp detection
_DATETIME_TYPES = (pl.Datetime, pl.Date)
_DEFAULT_GROUP_COLS = ("symbol", "product", "ticker", "asset", "asset_id")

# Duration string regex pattern (e.g., "1h", "30m", "1d2h30m")
_DURATION_PATTERN = re.compile(
    r"^(?:(\d+)w)?(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$",
    re.IGNORECASE,
)


def is_duration_string(value: str) -> bool:
    """Check if string is a duration format (e.g., '1h', '30m', not a column name).

    Duration strings follow Polars format: combinations of w/d/h/m/s units.
    Valid examples: "1h", "30m", "1d2h30m", "1w", "15s"
    Invalid (column names): "close", "volume", "my_column"

    Parameters
    ----------
    value : str
        String to check

    Returns
    -------
    bool
        True if value is a valid duration string

    Examples
    --------
    >>> is_duration_string("1h")
    True
    >>> is_duration_string("30m")
    True
    >>> is_duration_string("1d2h30m")
    True
    >>> is_duration_string("close")
    False
    >>> is_duration_string("max_holding_period")
    False
    """
    if not value or not isinstance(value, str):
        return False

    # Must contain at least one digit
    if not any(c.isdigit() for c in value):
        return False

    # Match duration pattern
    match = _DURATION_PATTERN.match(value.strip())
    if not match:
        return False

    # At least one component must be non-None
    return any(g is not None for g in match.groups())


def parse_duration(value: str) -> timedelta:
    """Parse a duration string to a timedelta.

    Supports Polars-style duration format: combinations of w/d/h/m/s units.

    Parameters
    ----------
    value : str
        Duration string (e.g., '1h', '30m', '1d2h30m')

    Returns
    -------
    timedelta
        Parsed duration

    Raises
    ------
    ValueError
        If the string is not a valid duration format

    Examples
    --------
    >>> parse_duration("1h")
    datetime.timedelta(seconds=3600)
    >>> parse_duration("30m")
    datetime.timedelta(seconds=1800)
    >>> parse_duration("1d2h30m")
    datetime.timedelta(days=1, seconds=9000)
    >>> parse_duration("1w")
    datetime.timedelta(days=7)
    """
    if not is_duration_string(value):
        raise ValueError(
            f"Invalid duration string: '{value}'. Expected format like '1h', '30m', '1d2h30m'."
        )

    match = _DURATION_PATTERN.match(value.strip())
    if not match:
        raise ValueError(f"Invalid duration string: '{value}'")

    weeks, days, hours, minutes, seconds = match.groups()

    return timedelta(
        weeks=int(weeks) if weeks else 0,
        days=int(days) if days else 0,
        hours=int(hours) if hours else 0,
        minutes=int(minutes) if minutes else 0,
        seconds=int(seconds) if seconds else 0,
    )


def duration_to_polars_expr(duration: str | timedelta) -> pl.Expr:
    """Convert a duration to a Polars duration expression.

    Parameters
    ----------
    duration : str | timedelta
        Duration as string ('1h', '30m') or timedelta

    Returns
    -------
    pl.Expr
        Polars duration literal expression

    Examples
    --------
    >>> expr = duration_to_polars_expr("1h")
    >>> expr = duration_to_polars_expr(timedelta(hours=1))
    """
    if isinstance(duration, str):
        td = parse_duration(duration)
    else:
        td = duration

    # Convert to microseconds for Polars duration
    total_us = int(td.total_seconds() * 1_000_000)
    return pl.duration(microseconds=total_us)


def time_horizon_to_bars(
    timestamps: np.ndarray,
    time_horizon: str | timedelta,
    event_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Convert time horizon to bar offsets using searchsorted.

    For each event index, computes how many bars forward correspond to
    the specified time horizon. Handles irregular data correctly.

    Parameters
    ----------
    timestamps : np.ndarray
        Array of timestamps as int64 nanoseconds (from datetime64[ns])
    time_horizon : str | timedelta
        Time horizon as duration string ('1h', '30m') or timedelta
    event_indices : np.ndarray | None
        Indices of events. If None, computes for all bars.

    Returns
    -------
    np.ndarray
        Bar counts for each event (int64). Minimum value is 1.

    Examples
    --------
    >>> import numpy as np
    >>> # 5-minute bars starting at midnight
    >>> ts = np.array(['2024-01-01 00:00', '2024-01-01 00:05', '2024-01-01 00:10',
    ...                '2024-01-01 00:15', '2024-01-01 00:20'], dtype='datetime64[ns]')
    >>> ts_ns = ts.astype('int64')
    >>> bars = time_horizon_to_bars(ts_ns, '15m')
    >>> # At index 0, 15min forward = index 3, so 3 bars
    >>> bars[0]
    3
    """
    # Parse duration
    if isinstance(time_horizon, str):
        td = parse_duration(time_horizon)
    else:
        td = time_horizon

    horizon_ns = int(td.total_seconds() * 1e9)

    # Get event times
    if event_indices is None:
        event_indices = np.arange(len(timestamps))
        event_times = timestamps
    else:
        event_times = timestamps[event_indices]

    # Compute target times
    target_times = event_times + horizon_ns

    # Find exit indices via searchsorted (left side = first index >= target)
    # We want the first bar at or after target time
    exit_indices = np.searchsorted(timestamps, target_times, side="left")

    # Clip to valid range
    exit_indices = np.minimum(exit_indices, len(timestamps) - 1)

    # Compute bar counts (minimum 1)
    bar_counts = exit_indices - event_indices
    bar_counts = np.maximum(bar_counts, 1).astype(np.int64)

    return bar_counts


def get_future_price_at_time(
    data: pl.DataFrame,
    time_horizon: str | timedelta,
    price_col: str = "close",
    timestamp_col: str | None = None,
    tolerance: str | None = None,
    group_cols: list[str] | None = None,
) -> tuple[pl.Series, pl.Series]:
    """Get price at future time using join_asof.

    For irregular data (e.g., trade bars), retrieves the first available price
    at or after `time_horizon` in the future.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with price and timestamp columns
    time_horizon : str | timedelta
        Time horizon ('1h', '30m', timedelta(hours=1))
    price_col : str, default "close"
        Price column name
    timestamp_col : str | None
        Timestamp column. If None, auto-detects.
    tolerance : str | None
        Maximum time gap allowed (e.g., '2m'). If None, no tolerance check.
    group_cols : list[str] | None
        Columns to partition by before joining (e.g., ['symbol'])

    Returns
    -------
    tuple[pl.Series, pl.Series]
        (future_prices, valid_mask) where valid_mask indicates successful joins

    Examples
    --------
    >>> future_prices, valid = get_future_price_at_time(
    ...     df, "15m", price_col="close", tolerance="2m"
    ... )
    """
    # Resolve timestamp column
    ts_col = resolve_timestamp_col(data, timestamp_col)
    if ts_col is None:
        raise ValueError(
            "Timestamp column not found. Provide timestamp_col parameter "
            "or ensure data has a datetime column."
        )

    # Parse duration
    if isinstance(time_horizon, str):
        td = parse_duration(time_horizon)
    else:
        td = time_horizon

    # Compute target timestamps
    total_us = int(td.total_seconds() * 1_000_000)
    target_ts = pl.col(ts_col) + pl.duration(microseconds=total_us)

    # Create future lookup table
    lookup = data.select(
        [
            pl.col(ts_col).alias("_lookup_ts"),
            pl.col(price_col).alias("_future_price"),
            *(pl.col(c) for c in (group_cols or [])),
        ]
    )

    # Add target timestamp to data
    data_with_target = data.with_columns(target_ts.alias("_target_ts"))

    # Build join arguments
    join_kwargs: dict = {
        "left_on": "_target_ts",
        "right_on": "_lookup_ts",
        "strategy": "forward",  # Get first price at or after target time
    }

    if tolerance is not None:
        join_kwargs["tolerance"] = tolerance

    if group_cols:
        join_kwargs["by"] = group_cols

    # Perform asof join
    result = data_with_target.join_asof(lookup, **join_kwargs)

    # Extract results
    future_prices = result["_future_price"]
    valid_mask = future_prices.is_not_null()

    return future_prices, valid_mask


def resolve_timestamp_col(
    data: pl.DataFrame,
    timestamp_col: str | None,
    *,
    warn_on_missing_specified: bool = True,
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
        if warn_on_missing_specified:
            warnings.warn(
                f"Specified timestamp_col '{timestamp_col}' not found in data. "
                f"Available columns: {data.columns}",
                UserWarning,
                stacklevel=3,
            )
        # Fall through to auto-detection

    # Dtype-based detection (more robust than name matching)
    datetime_cols = [col for col in data.columns if data[col].dtype in _DATETIME_TYPES]

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


def resolve_group_cols(
    data: pl.DataFrame,
    group_col: str | list[str] | None,
) -> list[str]:
    """Resolve grouping columns for panel-aware labeling."""
    if group_col is not None:
        if isinstance(group_col, str):
            if group_col not in data.columns:
                raise DataValidationError(
                    f"group_col '{group_col}' not found in data columns: {data.columns}",
                )
            return [group_col]

        missing = [col for col in group_col if col not in data.columns]
        if missing:
            raise DataValidationError(
                f"group_col values {missing} not found in data columns: {data.columns}",
            )
        return group_col

    detected_cols: list[str] = []
    for col in _DEFAULT_GROUP_COLS:
        if col in data.columns:
            detected_cols.append(col)
            break

    if "position" in data.columns and detected_cols and detected_cols[0] in ("product", "symbol"):
        detected_cols.append("position")

    return detected_cols


def resolve_labeling_columns(
    data: pl.DataFrame,
    *,
    price_col: str | None = None,
    timestamp_col: str | None = None,
    group_col: str | list[str] | None = None,
    config: LabelingConfig | None = None,
    contract: DataContractConfig | None = None,
    require_timestamp: bool = False,
) -> tuple[str, str | None, list[str]]:
    """Resolve columns using explicit args > labeling config > shared contract > defaults."""
    config_fields_set = (
        set(getattr(config, "model_fields_set", set())) if config is not None else set()
    )
    timestamp_is_explicit = timestamp_col is not None

    if contract is None and config is not None:
        nested_contract = getattr(config, "data_contract", None)
        if nested_contract is not None:
            contract = nested_contract

    if config is not None:
        if price_col is None and "price_col" in config_fields_set:
            price_col = config.price_col
        if timestamp_col is None and "timestamp_col" in config_fields_set:
            timestamp_col = config.timestamp_col
            timestamp_is_explicit = True
        if group_col is None and "group_col" in config_fields_set and config.group_col is not None:
            group_col = config.group_col
    if contract is not None:
        if price_col is None:
            price_col = contract.price_col
        if timestamp_col is None:
            timestamp_col = contract.timestamp_col
            timestamp_is_explicit = True
        if group_col is None and contract.symbol_col is not None:
            group_col = contract.symbol_col

    if price_col is None and config is not None:
        price_col = config.price_col
    if timestamp_col is None and config is not None:
        timestamp_col = config.timestamp_col

    resolved_price_col = price_col or "close"
    if resolved_price_col not in data.columns:
        raise DataValidationError(
            f"Price column '{resolved_price_col}' not found in data columns: {data.columns}",
        )

    resolved_timestamp_col = resolve_timestamp_col(
        data,
        timestamp_col,
        warn_on_missing_specified=timestamp_is_explicit,
    )
    if require_timestamp and resolved_timestamp_col is None:
        raise DataValidationError(
            "Timestamp column not found. Provide timestamp_col or ensure data has a datetime column.",
        )

    resolved_group_cols = resolve_group_cols(data, group_col)
    return resolved_price_col, resolved_timestamp_col, resolved_group_cols
