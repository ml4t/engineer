"""Internal declared-grid checks; shared FeedSpec binding is owned by Specs."""

import warnings
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast
from zoneinfo import ZoneInfo

import polars as pl
from ml4t.specs import FeedSpec, TimestampSemantics

if TYPE_CHECKING:
    from ml4t.engineer.labeling.calendar import PandasMarketCalendar


def validate_feature_grid(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    feed_spec: FeedSpec,
    calendar: str,
    timestamp_col: str | None,
    group_cols: list[str],
) -> None:
    """Bind shared v2 declarations to the fixed-intraday schedule checker."""
    required = (
        "data_frequency",
        "timestamp_semantics",
        "bar_type",
        "bar_grid_anchor",
        "bar_label",
        "partial_bar_policy",
    )
    missing = [name for name in required if getattr(feed_spec, name) is None]
    if feed_spec.time_contract_version != 2 or missing or timestamp_col is None:
        raise ValueError(
            f"A declared bar grid requires FeedSpec(time_contract_version=2) and {required}; "
            f"missing={missing}. Use calendar=None for observation-row calculations."
        )
    if feed_spec.bar_type != "time":
        raise ValueError("Fixed bar grid validation requires bar_type='time', not event-count bars")
    if feed_spec.timestamp_semantics not in {
        TimestampSemantics.BAR_CLOSE,
        TimestampSemantics.EVENT_TIME,
    }:
        raise ValueError("Fixed intraday bar grids require event_time or bar_close timestamps")
    schema = data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema
    dtype = schema[timestamp_col]
    timezone = feed_spec.timezone
    if timezone is None and isinstance(dtype, pl.Datetime):
        timezone = dtype.time_zone
    if timezone is None:
        raise ValueError("A declared bar grid requires timezone metadata for naive timestamps")

    frequency = feed_spec.data_frequency
    if isinstance(frequency, str):
        import pandas as pd

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                warnings.simplefilter("error", DeprecationWarning)
                parsed = pd.Timedelta(frequency)
        except (ValueError, TypeError, FutureWarning, DeprecationWarning) as exc:
            raise ValueError("declared grid data_frequency must be a fixed duration") from exc
        if not isinstance(parsed, pd.Timedelta) or parsed.value % 1000:
            raise ValueError("declared grid data_frequency requires finite microsecond precision")
        frequency = parsed.to_pytimedelta()
    if not isinstance(frequency, timedelta):
        raise ValueError("declared grid data_frequency must be a duration string or timedelta")
    validate_bar_grid(
        data,
        calendar=calendar,
        frequency=frequency,
        timezone=timezone,
        timestamp_col=timestamp_col,
        group_cols=group_cols,
        anchor=cast(Literal["session_open"], feed_spec.bar_grid_anchor),
        label_side=cast(Literal["left", "right"], feed_spec.bar_label),
        include_partial=feed_spec.partial_bar_policy == "include",
    )
    if "returns" in schema:
        ordered = data.sort([*group_cols, timestamp_col])
        missing_price = ~pl.col("close").cast(pl.Float64).is_finite().fill_null(False)
        previous_missing = (
            missing_price.shift(1).over(group_cols) if group_cols else missing_price.shift(1)
        ).fill_null(False)
        finite_return = pl.col("returns").cast(pl.Float64).is_finite().fill_null(False)
        invalid = ordered.filter(finite_return & (missing_price | previous_missing)).select(
            [*group_cols, timestamp_col]
        )
        invalid = invalid.collect() if isinstance(invalid, pl.LazyFrame) else invalid
        if invalid.height:
            raise ValueError(
                f"Return adjacent to a missing price at {invalid.row(0, named=True)!r}; "
                "normalize prices before deriving returns, not an already-derived return series"
            )


def validate_bar_grid(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    calendar: str,
    frequency: timedelta,
    timezone: str,
    timestamp_col: str,
    group_cols: list[str],
    anchor: Literal["session_open"],
    label_side: Literal["left", "right"],
    include_partial: bool,
) -> None:
    """Reject missing/off-grid bars within each group's observed endpoint interval.

    The policy arguments are internal calculation inputs, not a serialized contract.
    This does not infer dataset coverage outside the first and last observation.
    """
    if not isinstance(frequency, timedelta) or not timedelta(0) < frequency < timedelta(days=1):
        raise ValueError("declared grid frequency must be a fixed duration below one day")
    if anchor != "session_open" or label_side not in {"left", "right"}:
        raise ValueError("unsupported declared grid anchor or label side")
    if not isinstance(include_partial, bool):
        raise TypeError("include_partial must be explicitly boolean")
    if not timezone:
        raise ValueError("declared grid timezone is required")
    ZoneInfo(timezone)

    # This adapter already handles real sessions, breaks and interruptions without fallback.
    from ml4t.engineer.labeling.calendar import PandasMarketCalendar

    adapter = PandasMarketCalendar(calendar)
    projection = data.select([*group_cols, timestamp_col])
    frame = projection.collect() if isinstance(projection, pl.LazyFrame) else projection
    dtype = frame.schema[timestamp_col]
    if not isinstance(dtype, pl.Datetime):
        raise ValueError("declared bar grid requires datetime timestamps, not session labels")
    timestamps = pl.col(timestamp_col)
    if dtype.time_zone is None:
        timestamps = timestamps.dt.replace_time_zone(
            timezone, ambiguous="raise", non_existent="raise"
        )
    frame = frame.with_columns(timestamps.dt.convert_time_zone("UTC"))
    if frame[timestamp_col].null_count():
        raise ValueError("declared bar grid contains null timestamps")
    if frame.select((pl.col(timestamp_col).dt.nanosecond() % 1000 != 0).any()).item():
        raise ValueError("off-grid bar timestamp has finer than supported microsecond precision")

    groups = frame.partition_by(group_cols, as_dict=True) if group_cols else {(): frame}
    for group, rows in groups.items():
        actual = sorted(rows[timestamp_col].to_list())
        if not actual:
            continue
        if len(actual) != len(set(actual)):
            raise ValueError(f"duplicate bar timestamp in group {group!r}")
        observed = set(actual)
        for expected in _expected_bar_labels(
            adapter, actual[0], actual[-1], frequency, label_side, include_partial
        ):
            if expected not in observed:
                raise ValueError(
                    f"missing scheduled bar in group {group!r} at {expected.isoformat()}; "
                    "normalize prices to the declared grid before deriving returns, or explicitly "
                    "select calendar=None for observation-row calculations"
                )
            observed.remove(expected)
        off_grid = sorted(observed)
        if off_grid:
            raise ValueError(f"off-grid bar in group {group!r} at {off_grid[0].isoformat()}")


def _expected_bar_labels(
    adapter: "PandasMarketCalendar",
    start: datetime,
    end: datetime,
    frequency: timedelta,
    label_side: Literal["left", "right"],
    include_partial: bool,
) -> Iterator[datetime]:
    intervals = set()
    day = start.date()
    while day <= end.date():
        for opened, closed, _, _ in adapter._trading_intervals_around(day):
            intervals.add((opened.astimezone(UTC), closed.astimezone(UTC)))
        day += timedelta(days=7)

    for opened, closed in sorted(intervals):
        if closed < start or opened > end:
            continue
        skipped = max(0, (start - opened) // frequency - 1)
        boundary = opened + skipped * frequency
        while boundary < closed and boundary <= end:
            next_boundary = min(boundary + frequency, closed)
            if next_boundary - boundary < frequency and not include_partial:
                break
            label = boundary if label_side == "left" else next_boundary
            if start <= label <= end:
                yield label
            boundary = next_boundary
