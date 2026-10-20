"""Causal temporal bounds for finite-window registered features."""

import inspect
from datetime import timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from ml4t.specs import FeedSpec, TimestampSemantics, WindowUnit

from ml4t.engineer.core.bar_grid import scheduled_session_indices
from ml4t.engineer.core.registry import get_registry

_MINIMUM_PERIODS = {"sma": 1, "stddev": 1, "realized_volatility": 2}


def temporal_period(feature_name: str, params: dict[str, Any], unit: WindowUnit) -> int | timedelta:
    """Validate temporal support without executing any feature."""
    if feature_name not in _MINIMUM_PERIODS:
        raise ValueError(
            f"Feature {feature_name!r} does not support temporal windows; "
            f"supported finite-window features are {sorted(_MINIMUM_PERIODS)}. "
            "Recursive and multi-window definitions are not converted to row counts."
        )
    metadata = get_registry().get(feature_name)
    assert metadata is not None
    period = params.get(
        "period",
        metadata.parameters.get(
            "period", inspect.signature(metadata.func).parameters["period"].default
        ),
    )
    if unit == WindowUnit.SESSION:
        if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
            raise ValueError("Session feature period must be a positive integer session count")
        return period
    from ml4t.engineer.labeling.utils import parse_duration

    if isinstance(period, str):
        try:
            period = parse_duration(period)
        except ValueError as exc:
            raise ValueError("Elapsed feature period must be a positive fixed duration") from exc
    if not isinstance(period, timedelta) or period <= timedelta(0):
        raise ValueError(
            "Elapsed feature period must be a positive fixed duration, not a row count"
        )
    return period


def _window_sizes(
    data: pl.DataFrame,
    period: int | timedelta,
    feed: FeedSpec,
    calendar: str,
    timestamp_col: str,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    positions = np.arange(data.height, dtype=np.int64)
    if data.is_empty():
        return positions, np.empty(0, dtype=np.bool_)
    if isinstance(period, int):
        sessions = scheduled_session_indices(
            data, feed_spec=feed, calendar=calendar, timestamp_col=timestamp_col
        )
        # Counts above observed coverage cannot produce a complete window. Bound
        # before NumPy integer arithmetic so arbitrarily large Python counts are safe.
        if period > int(sessions[-1] - sessions[0]) + 1:
            return positions + 1, np.zeros(data.height, dtype=np.bool_)
        starts = np.searchsorted(sessions, sessions - period + 1, side="left")
        complete = sessions - sessions[0] >= period - 1
    else:
        if feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL:
            raise ValueError("session_label identifiers cannot define elapsed feature windows")
        values = data[timestamp_col]
        dtype = values.dtype
        assert isinstance(dtype, pl.Datetime)
        if dtype.time_zone is None:
            assert feed.timezone is not None
            values = values.dt.replace_time_zone(
                feed.timezone, ambiguous="raise", non_existent="raise"
            )
        timestamps = values.dt.convert_time_zone("UTC").dt.epoch("ns").to_numpy()
        duration = (
            (period.days * 86400 + period.seconds) * 1_000_000_000
            + period.microseconds * 1000
            + int(getattr(period, "nanoseconds", 0))
        )
        if duration > int(timestamps[-1] - timestamps[0]):
            return positions + 1, np.zeros(data.height, dtype=np.bool_)
        starts = np.searchsorted(timestamps, timestamps - duration, side="right")
        complete = timestamps - timestamps[0] >= duration
    return np.asarray(positions - starts + 1, dtype=np.int64), complete


def execute_temporal_feature(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    feature_name: str,
    params: dict[str, Any],
    output_name: str,
    group_cols: list[str],
    timestamp_col: str,
    feed: FeedSpec,
    calendar: str,
    period: int | timedelta,
) -> pl.DataFrame | pl.LazyFrame:
    """Reuse finite row kernels once per distinct causal window size, without resampling."""
    from ml4t.engineer.api import _execute_feature

    minimum = _MINIMUM_PERIODS[feature_name]
    template = _execute_feature(
        data.head(0), feature_name, {**params, "period": minimum}, output_name, []
    )
    schema = template.collect_schema() if isinstance(template, pl.LazyFrame) else template.schema
    if isinstance(data, pl.LazyFrame):

        def apply(frame: pl.DataFrame) -> pl.DataFrame:
            result = execute_temporal_feature(
                frame,
                feature_name=feature_name,
                params=params,
                output_name=output_name,
                group_cols=group_cols,
                timestamp_col=timestamp_col,
                feed=feed,
                calendar=calendar,
                period=period,
            )
            assert isinstance(result, pl.DataFrame)
            return result

        return data.map_batches(
            apply,
            schema=schema,
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
            streamable=False,
        )
    output_columns = [name for name in schema if name not in data.columns]
    frames = data.partition_by(group_cols, maintain_order=True) if group_cols else [data]
    results = []
    for frame in frames or [data]:
        sizes, complete = _window_sizes(frame, period, feed, calendar, timestamp_col)
        outputs = {
            name: pl.Series(name, [None] * frame.height, dtype=schema[name])
            for name in output_columns
        }
        for size in np.unique(sizes[complete & (sizes >= minimum)]):
            computed = _execute_feature(
                frame, feature_name, {**params, "period": int(size)}, output_name, []
            )
            assert isinstance(computed, pl.DataFrame)
            indices = np.flatnonzero(complete & (sizes == size))
            for name in output_columns:
                outputs[name] = outputs[name].scatter(indices, computed[name].gather(indices))
        results.append(frame.with_columns(list(outputs.values())))
    return pl.concat(results, how="vertical")
