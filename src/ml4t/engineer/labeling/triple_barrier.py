# mypy: disable-error-code="no-any-return,arg-type,call-arg,return-value,assignment"
"""Triple barrier labeling implementation.

Implements the generalized triple-barrier labeling method for financial machine learning.

References
----------
.. [1] De Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
       Chapter 3: Labeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling.barriers import BarrierConfig
from ml4t.engineer.labeling.numba_ops import _apply_triple_barrier_nb
from ml4t.engineer.labeling.uniqueness import (
    calculate_label_uniqueness,
    calculate_sample_weights,
)

if TYPE_CHECKING:
    from ml4t.engineer.config import LabelingConfig

from ml4t.engineer.labeling.utils import resolve_timestamp_col


def _prepare_barrier_arrays(
    data: pl.DataFrame,
    config: BarrierConfig,
    event_indices: npt.NDArray[np.intp],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int32],
    npt.NDArray[np.float64],
]:
    """Prepare barrier arrays from config."""
    n_events = len(event_indices)

    # Upper barriers
    if config.upper_barrier is None:
        upper_barriers = np.full(n_events, np.inf)
    elif isinstance(config.upper_barrier, (int, float)):
        upper_barriers = np.full(n_events, float(config.upper_barrier))
    else:
        if config.upper_barrier not in data.columns:
            raise DataValidationError(f"Upper barrier column '{config.upper_barrier}' not found")
        upper_barriers = data[config.upper_barrier].to_numpy()[event_indices]

    # Lower barriers
    if config.lower_barrier is None:
        lower_barriers = np.full(n_events, -np.inf)
    elif isinstance(config.lower_barrier, (int, float)):
        lower_barriers = np.full(n_events, float(config.lower_barrier))
    else:
        if config.lower_barrier not in data.columns:
            raise DataValidationError(f"Lower barrier column '{config.lower_barrier}' not found")
        lower_barriers = data[config.lower_barrier].to_numpy()[event_indices]

    # Max periods
    if isinstance(config.max_holding_period, int):
        max_periods = np.full(n_events, config.max_holding_period, dtype=np.int64)
    else:
        if config.max_holding_period not in data.columns:
            raise DataValidationError(
                f"Max holding period column '{config.max_holding_period}' not found"
            )
        max_periods = data[config.max_holding_period].to_numpy()[event_indices].astype(np.int64)

    # Sides
    if config.side is None:
        sides = np.zeros(n_events, dtype=np.int32)
    elif isinstance(config.side, int):
        sides = np.full(n_events, config.side, dtype=np.int32)
    else:
        if config.side not in data.columns:
            raise DataValidationError(f"Side column '{config.side}' not found")
        sides = data[config.side].to_numpy()[event_indices].astype(np.int32)

    # Trailing stops
    if config.trailing_stop is False or config.trailing_stop is None:
        trailing_stops = np.zeros(n_events)
    elif config.trailing_stop is True:
        if config.lower_barrier is not None and isinstance(config.lower_barrier, (int, float)):
            trailing_stops = np.full(n_events, abs(float(config.lower_barrier)))
        else:
            trailing_stops = np.full(n_events, 0.01)
    elif isinstance(config.trailing_stop, (int, float)):
        trailing_stops = np.full(n_events, float(config.trailing_stop))
    else:
        if config.trailing_stop not in data.columns:
            raise DataValidationError(f"Trailing stop column '{config.trailing_stop}' not found")
        trailing_stops = data[config.trailing_stop].to_numpy()[event_indices]

    return upper_barriers, lower_barriers, max_periods, sides, trailing_stops


def _determine_barrier_hits(labels: npt.NDArray[np.int32]) -> npt.NDArray[np.object_]:
    """Determine which barrier was hit for each label."""
    n_events = len(labels)
    barrier_hit = np.empty(n_events, dtype=object)
    for i in range(n_events):
        if labels[i] == 1:
            barrier_hit[i] = "upper"
        elif labels[i] == -1:
            barrier_hit[i] = "lower"
        else:
            barrier_hit[i] = "time"
    return barrier_hit


def _calculate_time_durations(
    data: pl.DataFrame,
    event_indices: npt.NDArray[np.intp],
    label_indices: npt.NDArray[np.int64],
    timestamp_col: str | None,
) -> tuple[list[Any], npt.NDArray[Any]]:
    """Calculate label times and durations."""
    n_events = len(event_indices)

    if timestamp_col is not None:
        timestamps = data[timestamp_col].to_list()
        label_times = [timestamps[idx] if idx < len(timestamps) else None for idx in label_indices]
        timestamps_array = data[timestamp_col].to_numpy()
        entry_times = timestamps_array[event_indices]
        exit_times = timestamps_array[label_indices]
        time_durations = exit_times - entry_times
    else:
        label_times = list(label_indices)
        time_durations = np.array([None] * n_events, dtype=object)

    return label_times, time_durations


def _build_labeling_result(
    data: pl.DataFrame,
    event_indices: npt.NDArray[np.intp],
    labels: npt.NDArray[np.int32],
    label_times: list[Any],
    label_prices: npt.NDArray[np.float64],
    label_returns: npt.NDArray[np.float64],
    bar_durations: npt.NDArray[np.int64],
    time_durations: npt.NDArray[Any],
    barrier_hit: npt.NDArray[np.object_],
    uniqueness: npt.NDArray[np.float64] | None,
    sample_weights: npt.NDArray[np.float64] | None,
) -> pl.DataFrame:
    """Build the result DataFrame from labeling outputs."""
    calc_uniq = uniqueness is not None

    if "event_time" in data.columns:
        label_dict: dict[str, Any] = {
            "event_index": event_indices,
            "label": labels,
            "label_time": label_times,
            "label_price": label_prices,
            "label_return": label_returns,
            "label_bars": bar_durations,
            "label_duration": time_durations,
            "barrier_hit": barrier_hit,
        }
        if calc_uniq:
            label_dict["label_uniqueness"] = uniqueness
            label_dict["sample_weight"] = sample_weights

        label_data = pl.DataFrame(label_dict)
        result = (
            data.with_row_index("event_index")
            .join(label_data, on="event_index", how="left")
            .drop("event_index")
        )
    else:
        columns_to_add: dict[str, pl.Series] = {
            "label": pl.Series(labels, dtype=pl.Int32),
            "label_time": pl.Series(label_times),
            "label_price": pl.Series(label_prices, dtype=pl.Float64),
            "label_return": pl.Series(label_returns, dtype=pl.Float64),
            "label_bars": pl.Series(bar_durations, dtype=pl.Int64),
            "label_duration": pl.Series(time_durations),
            "barrier_hit": pl.Series(barrier_hit),
        }
        if calc_uniq:
            columns_to_add["label_uniqueness"] = pl.Series(uniqueness, dtype=pl.Float64)
            columns_to_add["sample_weight"] = pl.Series(sample_weights, dtype=pl.Float64)

        result = data.with_columns(**columns_to_add)

    return result


def triple_barrier_labels(
    data: pl.DataFrame,
    config: BarrierConfig | LabelingConfig,
    price_col: str = "close",
    high_col: str | None = None,
    low_col: str | None = None,
    timestamp_col: str | None = None,
    calculate_uniqueness: bool = False,
    uniqueness_weight_scheme: Literal[
        "returns_uniqueness", "uniqueness_only", "returns_only", "equal"
    ] = "returns_uniqueness",
) -> pl.DataFrame:
    """Apply triple-barrier labeling to data.

    Labels price movements based on which barrier (upper, lower, or time) is touched first.
    Optionally calculates label uniqueness and sample weights (De Prado's AFML Chapter 4).

    Parameters
    ----------
    data : pl.DataFrame
        Input data with price information
    config : BarrierConfig | LabelingConfig
        Barrier configuration. Accepts either:
        - BarrierConfig: Legacy dataclass configuration
        - LabelingConfig: Pydantic config with serialization (automatically converted)
    price_col : str, default "close"
        Name of the price column
    high_col : str, optional
        Name of the high price column for OHLC barrier checking
    low_col : str, optional
        Name of the low price column for OHLC barrier checking
    timestamp_col : str, optional
        Name of the timestamp column (uses row index if None)
    calculate_uniqueness : bool, default False
        If True, calculates label uniqueness scores and sample weights
    uniqueness_weight_scheme : str, default "returns_uniqueness"
        Weighting scheme: "returns_uniqueness", "uniqueness_only", "returns_only", "equal"

    Returns
    -------
    pl.DataFrame
        Original data with added columns: label, label_time, label_price, label_return,
        label_bars, label_duration, barrier_hit, and optionally label_uniqueness, sample_weight

    Notes
    -----
    **Important**: Data is automatically sorted by timestamp before labeling.
    This is required because the algorithm scans forward in row order to find
    barrier touches. The result is returned sorted chronologically.

    Examples
    --------
    >>> # Using legacy BarrierConfig
    >>> config = BarrierConfig(upper_barrier=0.02, lower_barrier=0.01, max_periods=10, side=1)
    >>> labeled = triple_barrier_labels(df, config, timestamp_col="datetime")
    >>>
    >>> # Using LabelingConfig (with serialization support)
    >>> from ml4t.engineer.config import LabelingConfig
    >>> config = LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=0.01)
    >>> labeled = triple_barrier_labels(df, config)
    """
    # Convert LabelingConfig to BarrierConfig if needed
    if hasattr(config, "to_barrier_config"):
        config = config.to_barrier_config()  # type: ignore[operator]
    if price_col not in data.columns:
        raise DataValidationError(f"Price column '{price_col}' not found in data")

    # Sort data chronologically for correct forward scanning
    # Triple barrier scans forward in row order to find barrier touches
    resolved_ts_col = resolve_timestamp_col(data, timestamp_col)
    if resolved_ts_col:
        data = data.sort(resolved_ts_col)

    # Determine events
    if "event_time" in data.columns:
        event_mask = data["event_time"].is_not_null()
        event_indices = np.where(event_mask.to_numpy())[0]
        if len(event_indices) == 0:
            return data.with_columns(
                label=pl.lit(None, dtype=pl.Int32),
                label_time=pl.lit(None, dtype=pl.Int64),
                label_price=pl.lit(None, dtype=pl.Float64),
                label_return=pl.lit(None, dtype=pl.Float64),
                weight=pl.lit(None, dtype=pl.Float64),
            )
    else:
        event_indices = np.arange(len(data))

    # Extract price data
    closes = data[price_col].to_numpy()
    if high_col is not None:
        if high_col not in data.columns:
            raise DataValidationError(f"High column '{high_col}' not found in data")
        highs = data[high_col].to_numpy()
    else:
        highs = closes

    if low_col is not None:
        if low_col not in data.columns:
            raise DataValidationError(f"Low column '{low_col}' not found in data")
        lows = data[low_col].to_numpy()
    else:
        lows = closes

    # Prepare barriers and apply labeling
    upper_barriers, lower_barriers, max_periods, sides, trailing_stops = _prepare_barrier_arrays(
        data, config, event_indices
    )
    labels, label_indices, label_prices, label_returns, bar_durations = _apply_triple_barrier_nb(
        closes,
        highs,
        lows,
        event_indices,
        upper_barriers,
        lower_barriers,
        max_periods,
        sides,
        trailing_stops,
    )

    # Calculate uniqueness if requested
    if calculate_uniqueness:
        uniqueness = calculate_label_uniqueness(
            event_indices=event_indices, label_indices=label_indices, n_bars=len(closes)
        )
        sample_weights = calculate_sample_weights(
            uniqueness=uniqueness, returns=label_returns, weight_scheme=uniqueness_weight_scheme
        )
    else:
        uniqueness = None
        sample_weights = None

    # Build result
    barrier_hit = _determine_barrier_hits(labels)
    label_times, time_durations = _calculate_time_durations(
        data, event_indices, label_indices, timestamp_col
    )

    return _build_labeling_result(
        data=data,
        event_indices=event_indices,
        labels=labels,
        label_times=label_times,
        label_prices=label_prices,
        label_returns=label_returns,
        bar_durations=bar_durations,
        time_durations=time_durations,
        barrier_hit=barrier_hit,
        uniqueness=uniqueness,
        sample_weights=sample_weights,
    )


# Re-export utility functions from barrier_utils for backward compatibility
from ml4t.engineer.labeling.barrier_utils import (  # noqa: E402
    apply_triple_barrier,
    calculate_returns,
    compute_barrier_touches,
)

__all__ = [
    "triple_barrier_labels",
    "compute_barrier_touches",
    "calculate_returns",
    "apply_triple_barrier",
]
