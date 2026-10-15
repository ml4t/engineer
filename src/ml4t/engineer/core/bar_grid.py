"""Internal declared-grid checks; shared FeedSpec binding is owned by Specs."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

import polars as pl

if TYPE_CHECKING:
    from ml4t.engineer.labeling.calendar import PandasMarketCalendar


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

    groups = frame.partition_by(group_cols, as_dict=True) if group_cols else {(): frame}
    for group, rows in groups.items():
        actual = sorted(rows[timestamp_col].to_list())
        if not actual:
            continue
        if len(actual) != len(set(actual)):
            raise ValueError(f"duplicate bar timestamp in group {group!r}")
        expected = _expected_bar_labels(
            adapter, actual[0], actual[-1], frequency, label_side, include_partial
        )
        observed = set(actual)
        missing = sorted(expected - observed)
        if missing:
            raise ValueError(
                f"missing scheduled bar in group {group!r} at {missing[0].isoformat()}; "
                "normalize prices to the declared grid before deriving returns, or explicitly "
                "select calendar=None for observation-row calculations"
            )
        off_grid = sorted(observed - expected)
        if off_grid:
            raise ValueError(f"off-grid bar in group {group!r} at {off_grid[0].isoformat()}")


def _expected_bar_labels(
    adapter: "PandasMarketCalendar",
    start: datetime,
    end: datetime,
    frequency: timedelta,
    label_side: Literal["left", "right"],
    include_partial: bool,
) -> set[datetime]:
    intervals = set()
    day = start.date()
    while day <= end.date():
        for opened, closed, _, _ in adapter._trading_intervals_around(day):
            intervals.add((opened.astimezone(UTC), closed.astimezone(UTC)))
        day += timedelta(days=7)

    expected = set()
    for opened, closed in intervals:
        if closed < start or opened > end:
            continue
        boundary = opened
        while boundary < closed and boundary <= end:
            next_boundary = min(boundary + frequency, closed)
            if next_boundary - boundary < frequency and not include_partial:
                break
            label = boundary if label_side == "left" else next_boundary
            if start <= label <= end:
                expected.add(label)
            boundary = next_boundary
    return expected
