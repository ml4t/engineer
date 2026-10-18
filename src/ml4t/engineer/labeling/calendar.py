"""Calendar-aware labeling utilities.

Provides session-aware barrier labeling that respects trading calendar gaps
(maintenance windows, overnight sessions, weekends, holidays).

This prevents data leakage when labels would otherwise span non-trading periods.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from inspect import signature
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec, TimestampSemantics, WindowUnit

from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling.triple_barrier import _triple_barrier_labels_single_group
from ml4t.engineer.labeling.utils import (
    is_duration_string,
    resolve_labeling_columns,
    validate_price_no_nans,
)

if TYPE_CHECKING:  # pragma: no cover - imports used only by static analysis
    from ml4t.engineer.config import DataContractConfig, LabelingConfig


class TradingCalendar(Protocol):
    """Protocol for trading calendar implementations.

    Any calendar implementation providing these methods can be used.
    """

    def is_trading_time(self, timestamp: datetime) -> bool:
        """Check if given timestamp is during trading hours."""
        ...  # pragma: no cover - protocol declaration has no runtime implementation

    def next_session_break(self, timestamp: datetime) -> datetime | None:
        """Get the next session break at or after the given timestamp.

        Returns None if no break before end of data.
        """
        ...  # pragma: no cover - protocol declaration has no runtime implementation


class SimpleTradingCalendar:
    """Simple calendar based on time gaps in data.

    Identifies session breaks by detecting gaps larger than threshold.
    Useful when explicit calendar is unavailable.
    """

    def __init__(self, gap_threshold_minutes: int = 30):
        """
        Parameters
        ----------
        gap_threshold_minutes : int
            Gap duration in minutes to consider as session break
        """
        if isinstance(gap_threshold_minutes, bool) or not isinstance(gap_threshold_minutes, int):
            raise TypeError("gap_threshold_minutes must be an integer")
        if gap_threshold_minutes <= 0:
            raise ValueError("gap_threshold_minutes must be positive")
        self.gap_threshold = timedelta(minutes=gap_threshold_minutes)
        self._data: pl.DataFrame | None = None
        self._session_breaks: list[datetime] | None = None

    def fit(self, data: pl.DataFrame, timestamp_col: str = "timestamp") -> SimpleTradingCalendar:
        """Learn session breaks from data gaps.

        Parameters
        ----------
        data : pl.DataFrame
            Data with timestamp column
        timestamp_col : str
            Name of timestamp column

        Returns
        -------
        self : SimpleTradingCalendar
            Fitted calendar
        """
        # Find gaps in timestamps
        gaps_df = data.select(
            [pl.col(timestamp_col), pl.col(timestamp_col).diff().alias("time_diff")]
        ).filter(
            pl.col("time_diff") > pl.duration(minutes=int(self.gap_threshold.total_seconds() / 60))
        )

        self._session_breaks = gaps_df[timestamp_col].to_list()
        return self

    def is_trading_time(self, timestamp: datetime) -> bool:  # noqa: ARG002 - interface requirement
        """Always returns True for simple calendar (data defines trading times)."""
        return True

    def next_session_break(self, timestamp: datetime) -> datetime | None:
        """Get next session break after timestamp."""
        if self._session_breaks is None:
            return None

        for break_time in self._session_breaks:
            if break_time > timestamp:
                return break_time
        return None


class PandasMarketCalendar:
    """Adapter for pandas_market_calendars library.

    Supports 200+ calendars including CME, NYSE, LSE, etc.
    """

    def __init__(
        self,
        calendar_name: str | CalendarUnset = UNSET_CALENDAR,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
    ):
        """
        Parameters
        ----------
        calendar_name : str, optional
            Legacy positional/keyword selector. Conflicting selectors fail.
        calendar : str or None, optional
            Canonical named selector, default NYSE. None is rejected because a
            provider adapter cannot represent calendar-free observation order.
        """
        if (
            calendar_name is not UNSET_CALENDAR
            and calendar is not UNSET_CALENDAR
            and calendar_name != calendar
        ):
            raise ValueError("calendar and calendar_name selectors conflict")
        selected = FeedSpec(time_contract_version=2).resolve_calendar(
            calendar=calendar_name if calendar is UNSET_CALENDAR else calendar
        )
        if selected is None:
            raise ValueError("PandasMarketCalendar requires a named calendar, not calendar=None")
        try:
            import pandas_market_calendars as mcal
        except ImportError as err:
            raise ImportError(
                "pandas_market_calendars required for PandasMarketCalendar. "
                "Install with: pip install pandas-market-calendars"
            ) from err

        self.calendar = selected
        self.calendar_name = selected
        self._calendar = mcal.get_calendar(selected)
        self._supports_interruptions = (
            "interruptions" in signature(self._calendar.schedule).parameters
        )
        self._schedule_cache: dict[date, Any] = {}
        self._interval_cache: dict[
            date,
            list[tuple[datetime, datetime, datetime, bool]],
        ] = {}

    @staticmethod
    def _normalize_timestamp(timestamp: datetime) -> datetime:
        """Normalize timestamps to UTC, interpreting naive values as UTC."""
        if not isinstance(timestamp, datetime):
            raise TypeError("calendar timestamps must be datetime values")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC)

    def _schedule_around(self, calendar_day: date) -> Any:
        """Load enough sessions to cover overnight markets and the next break."""
        if calendar_day not in self._schedule_cache:
            if len(self._schedule_cache) >= 128:
                oldest_day = next(iter(self._schedule_cache))
                self._schedule_cache.pop(oldest_day)
                self._interval_cache.pop(oldest_day, None)
            schedule_kwargs = {"interruptions": True} if self._supports_interruptions else {}
            self._schedule_cache[calendar_day] = self._calendar.schedule(
                start_date=calendar_day - timedelta(days=1),
                end_date=calendar_day + timedelta(days=7),
                **schedule_kwargs,
            )
        return self._schedule_cache[calendar_day]

    @staticmethod
    def _session_breaks(session: Any) -> list[tuple[Any, Any]]:
        """Extract scheduled intraday non-trading intervals."""
        import pandas as pd

        intervals = []
        for start_col in (
            "break_start",
            *sorted(
                column for column in session.index if str(column).startswith("interruption_start_")
            ),
        ):
            if start_col not in session.index:
                continue
            end_col = (
                "break_end"
                if start_col == "break_start"
                else str(start_col).replace("start", "end", 1)
            )
            start = session[start_col]
            end = session.get(end_col)
            if not pd.isna(start) and not pd.isna(end) and start < end:
                intervals.append((start, end))
        return intervals

    def _trading_intervals_around(
        self,
        calendar_day: date,
    ) -> list[tuple[datetime, datetime, datetime, bool]]:
        """Return trading intervals and their terminating session boundary."""
        if calendar_day in self._interval_cache:
            return self._interval_cache[calendar_day]

        intervals = []
        for _, session in self._schedule_around(calendar_day).iterrows():
            market_open = session["market_open"].to_pydatetime()
            market_close = session["market_close"].to_pydatetime()
            interval_start = market_open
            for break_start, break_end in sorted(self._session_breaks(session)):
                break_start_dt = break_start.to_pydatetime()
                break_end_dt = break_end.to_pydatetime()
                if interval_start < break_start_dt:
                    intervals.append((interval_start, break_start_dt, break_start_dt, False))
                interval_start = max(interval_start, break_end_dt)
            if interval_start <= market_close:
                intervals.append((interval_start, market_close, market_close, True))

        self._interval_cache[calendar_day] = intervals
        return intervals

    @staticmethod
    def _contains_timestamp(
        timestamp: datetime,
        start: datetime,
        end: datetime,
        end_inclusive: bool,
    ) -> bool:
        """Check timestamp membership with explicit close-boundary semantics."""
        return start <= timestamp <= end if end_inclusive else start <= timestamp < end

    def _trading_boundary(
        self, timestamp: datetime, *, bar_close: bool = False
    ) -> tuple[bool, datetime | None]:
        """Return whether a timestamp is trading and its next session boundary."""
        normalized = self._normalize_timestamp(timestamp)
        for start, end, boundary, end_inclusive in self._trading_intervals_around(
            normalized.date()
        ):
            contains = (
                start < normalized <= end
                if bar_close
                else self._contains_timestamp(normalized, start, end, end_inclusive)
            )
            if contains:
                return True, boundary
        return False, None

    def is_trading_time(self, timestamp: datetime) -> bool:
        """Check if timestamp is during trading session."""
        is_trading, _ = self._trading_boundary(timestamp)
        return is_trading

    def next_session_break(self, timestamp: datetime) -> datetime | None:
        """Get the next intraday break or session close at or after timestamp."""
        normalized = self._normalize_timestamp(timestamp)
        for start, end, boundary, end_inclusive in self._trading_intervals_around(
            normalized.date()
        ):
            if self._contains_timestamp(normalized, start, end, end_inclusive):
                return boundary
            if normalized < start:
                return boundary
        return None


def _temporary_column_name(columns: list[str], base: str) -> str:
    """Return a temporary column name that cannot replace user data."""
    name = base
    suffix = 1
    while name in columns:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _validate_explicit_calendar(calendar: TradingCalendar) -> None:
    """Validate the runtime calendar protocol before processing rows."""
    for method_name in ("is_trading_time", "next_session_break"):
        if not callable(getattr(calendar, method_name, None)):
            raise TypeError(f"calendar must define callable {method_name}()")


def _calendar_session_ids(
    data: pl.DataFrame,
    calendar: TradingCalendar,
    timestamp_col: str,
    group_cols: list[str],
    timezone: str | None = None,
    bar_close: bool = False,
) -> tuple[pl.DataFrame, list[int]]:
    """Sort rows and derive sessions from an explicit calendar."""
    _validate_explicit_calendar(calendar)
    sort_cols = [*group_cols, timestamp_col]
    sorted_data = data.sort(sort_cols)
    selected_cols = [*group_cols, timestamp_col]
    projection = sorted_data.select(selected_cols)
    dtype = projection.schema[timestamp_col]
    if isinstance(dtype, pl.Datetime):
        if projection.select((pl.col(timestamp_col).dt.nanosecond() % 1000 != 0).any()).item():
            raise ValueError("calendar timestamp has finer than supported microsecond precision")
        if dtype.time_zone is None and timezone is not None:
            projection = projection.with_columns(
                pl.col(timestamp_col)
                .dt.replace_time_zone(timezone, ambiguous="raise", non_existent="raise")
                .dt.convert_time_zone("UTC")
            )

    session_ids: list[int] = []
    previous_group: tuple[Any, ...] | None = None
    previous_break: datetime | None = None
    session_id = -1

    for row in projection.iter_rows(named=False):
        group = tuple(row[: len(group_cols)])
        timestamp = row[-1]
        if not isinstance(timestamp, datetime):
            raise TypeError(f"timestamp column {timestamp_col!r} must contain datetime values")

        if isinstance(calendar, PandasMarketCalendar):
            is_trading, session_break = calendar._trading_boundary(timestamp, bar_close=bar_close)
        else:
            is_trading = calendar.is_trading_time(timestamp)
            session_break = calendar.next_session_break(timestamp) if is_trading else None
        if not is_trading:
            raise DataValidationError(
                f"timestamp {timestamp!r} is outside the selected calendar's trading time"
            )

        if session_break is not None:
            if not isinstance(session_break, datetime):
                raise TypeError("calendar next_session_break() must return datetime or None")
            comparison_timestamp = (
                calendar._normalize_timestamp(timestamp)
                if isinstance(calendar, PandasMarketCalendar)
                else timestamp
            )
            try:
                break_delta = session_break - comparison_timestamp
            except TypeError as err:
                raise TypeError(
                    "calendar break and data timestamps must use compatible timezone semantics"
                ) from err
            if break_delta < timedelta(0):
                raise ValueError("calendar next_session_break() returned a past boundary")

        if group != previous_group:
            session_id = 0
        elif session_break != previous_break:
            session_id += 1

        session_ids.append(session_id)
        previous_group = group
        previous_break = session_break

    return sorted_data, session_ids


def _calendar_group_labels(
    data: pl.DataFrame,
    config: LabelingConfig,
    price_col: str,
    timestamp_col: str | None,
    timezone: str | None = None,
    *,
    max_holding_bars: npt.NDArray[np.int64] | None = None,
) -> pl.DataFrame:
    dtype = data.schema[timestamp_col] if timestamp_col is not None else None
    local_clock = (
        isinstance(dtype, pl.Datetime) and dtype.time_zone is None and timezone is not None
    )
    if local_clock:
        assert timestamp_col is not None
        data = data.with_columns(
            pl.col(timestamp_col)
            .dt.replace_time_zone(timezone, ambiguous="raise", non_existent="raise")
            .dt.convert_time_zone("UTC")
        )
    result = _triple_barrier_labels_single_group(
        data,
        config,
        price_col,
        None,
        None,
        None,
        timestamp_col,
        False,
        "returns_uniqueness",
        max_holding_bars=max_holding_bars,
    )
    if local_clock:
        assert timestamp_col is not None
        result = result.with_columns(
            pl.col(timestamp_col, "label_time")
            .dt.convert_time_zone(timezone)
            .dt.replace_time_zone(None)
        )
    return result


def _segment_labels(
    data: pl.DataFrame,
    config: LabelingConfig,
    price_col: str,
    group_cols: list[str],
    timestamp_col: str | None = None,
    timezone: str | None = None,
) -> pl.DataFrame:
    validate_price_no_nans(data, price_col)
    row_col = _temporary_column_name(data.columns, "__ml4t_observation_row")
    indexed = data.with_row_index(row_col)
    frames = indexed.partition_by(group_cols, maintain_order=True) if group_cols else [indexed]
    frames = frames or [indexed]
    results = [
        _calendar_group_labels(frame, config, price_col, timestamp_col, timezone)
        for frame in frames
    ]
    result = pl.concat(results, how="vertical").sort(row_col).drop(row_col)
    if timestamp_col is None:
        result = result.with_columns(pl.lit(None).alias("label_duration"))
    return result


def _session_window_labels(
    data: pl.DataFrame,
    config: LabelingConfig,
    feed: FeedSpec,
    calendar: str,
    price_col: str,
    timestamp_col: str,
    group_cols: list[str],
) -> pl.DataFrame:
    from ml4t.engineer.core.bar_grid import scheduled_session_indices

    validate_price_no_nans(data, price_col)
    sorted_data = data.sort([*group_cols, timestamp_col])
    frames = (
        sorted_data.partition_by(group_cols, maintain_order=True) if group_cols else [sorted_data]
    )
    frames = frames or [sorted_data]
    results = []
    for frame in frames:
        session_ids = scheduled_session_indices(
            frame, feed_spec=feed, calendar=calendar, timestamp_col=timestamp_col
        )
        horizon = config.max_holding_period
        if isinstance(horizon, int):
            counts = np.full(frame.height, horizon, dtype=np.float64)
        elif isinstance(horizon, str) and horizon in frame.columns:
            counts = np.asarray(frame[horizon].to_numpy(), dtype=np.float64)
        else:
            raise ValueError("Session holding periods require a positive integer or count column")
        if (
            not np.isfinite(counts).all()
            or np.any(counts <= 0)
            or np.any(counts != np.floor(counts))
        ):
            raise ValueError("Session holding periods must contain finite positive integral counts")
        # A deadline beyond observed coverage is censored at the final available bar,
        # as in the existing count-horizon engine. Clip before integer conversion.
        bounded_counts = (
            np.minimum(counts, session_ids[-1] + 1).astype(np.int64)
            if frame.height
            else np.empty(0, dtype=np.int64)
        )
        exits = np.searchsorted(session_ids, session_ids + bounded_counts, side="right") - 1
        periods = np.asarray(exits - np.arange(frame.height), dtype=np.int64)
        results.append(
            _calendar_group_labels(
                frame,
                config,
                price_col,
                timestamp_col,
                None
                if feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL
                else feed.timezone,
                max_holding_bars=periods,
            )
        )
    result = pl.concat(results, how="vertical")
    if feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL:
        result = result.with_columns(pl.lit(None).alias("label_duration"))
    return result


def calendar_aware_labels(
    data: pl.DataFrame,
    config: LabelingConfig,
    calendar: str | TradingCalendar | None | CalendarUnset = UNSET_CALENDAR,
    price_col: str | None = None,
    timestamp_col: str | None = None,
    group_col: str | list[str] | None = None,
    contract: DataContractConfig | None = None,
    *,
    feed_spec: FeedSpec | Mapping[str, Any] | None = None,
    window_unit: WindowUnit | str | None = None,
) -> pl.DataFrame:
    """Apply triple-barrier labeling with declared calendar and horizon units.

    Intraday count/elapsed compatibility modes stop at breaks. Session-count
    horizons span scheduled sessions, and daily labels count whole-session rows.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with OHLCV and timestamp
    config : LabelingConfig
        Barrier configuration
    calendar : str, TradingCalendar or None, optional
        Explicit selector overrides config, then FeedSpec metadata, then NYSE.
        None selects observation-row labeling without a schedule. Protocol objects
        and explicit legacy "auto" gap mode remain available without FeedSpec.
    price_col : str | None, default None
        Price column name
    timestamp_col : str | None, default None
        Timestamp column name
    group_col : str | list[str] | None, default None
        Grouping column(s) for panel-aware session labeling.
    contract : DataContractConfig | None, default None
        Optional shared dataframe contract. Used after config and before defaults.
    feed_spec : FeedSpec or Mapping, optional
        Shared column/time declarations, otherwise config.feed_spec. Named-calendar
        feeds require timestamp semantics and a timezone for naive timestamps.
    window_unit : WindowUnit or str, optional
        Count windows default to expected_bar for shared feeds and omitted selectors;
        a complete declared grid is required. Explicit observation counts and elapsed
        duration horizons remain distinct. Legacy explicit selectors without FeedSpec
        retain observation counts. Session counts expire on the final available
        bar of the Nth following scheduled session, or at observed coverage end.

    Returns
    -------
    pl.DataFrame
        Data with barrier labels, respecting session boundaries

    Examples
    --------
    >>> # CME futures with pandas_market_calendars
    >>> labeled = calendar_aware_labels(
    ...     data,
    ...     config=LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=0.02),
    ...     calendar="CME_Equity"  # Product-specific calendar
    ... )

    >>> # NYSE equities
    >>> labeled = calendar_aware_labels(
    ...     data,
    ...     config=LabelingConfig.triple_barrier(upper_barrier=0.01, lower_barrier=0.01),
    ...     calendar="NYSE"
    ... )

    >>> # Auto-detect gaps
    >>> labeled = calendar_aware_labels(
    ...     data,
    ...     config=LabelingConfig.triple_barrier(upper_barrier=0.02, lower_barrier=0.02),
    ...     calendar="auto"
    ... )

    >>> # Custom calendar
    >>> class MyCalendar:
    ...     def is_trading_time(self, ts): return True
    ...     def next_session_break(self, ts): return None
    >>> labeled = calendar_aware_labels(data, config, calendar=MyCalendar())

    Notes
    -----
    - Uses pandas_market_calendars for all string calendar names
    - Supports 200+ global calendars + product-specific futures calendars
    - See pandas_market_calendars.get_calendar_names() for available calendars
    - Intraday observation/expected-bar and legacy duration windows stop at breaks
    - Session-count windows cross breaks and use the same numerical barrier engine
    - This may result in more timeout labels near session closes
    - Rows outside an explicit calendar's trading times are rejected
    - Without FeedSpec, legacy naive calendar timestamps are interpreted as UTC
    - FeedSpec naive timestamps use its declared timezone; ambiguous times fail
    - Instant horizons and label_duration use physical elapsed time across DST;
      session-label identifiers retain label_time but leave label_duration unset
    - Explicit None preserves per-asset observation order and global row order;
      label_time contains per-asset row offsets and label_duration is unset
    - A named 24/7 schedule still segments daily sessions; it is not row mode
    """
    from ml4t.engineer.config import LabelingConfig as _LabelingConfig

    if not isinstance(config, _LabelingConfig):
        raise TypeError(
            "calendar_aware_labels expects LabelingConfig. "
            "Legacy BarrierConfig inputs are no longer supported; "
            "use LabelingConfig.triple_barrier(...)."
        )

    declared_feed = feed_spec if feed_spec is not None else config.feed_spec
    legacy_selector = declared_feed is None and calendar is not UNSET_CALENDAR
    feed = (
        FeedSpec.from_any(declared_feed)
        if declared_feed is not None
        else FeedSpec(time_contract_version=2)
    )
    if calendar is UNSET_CALENDAR and "calendar" in config.model_fields_set:
        calendar = config.calendar
    custom_calendar = (
        calendar is not UNSET_CALENDAR and calendar is not None and not isinstance(calendar, str)
    )
    selected = calendar if custom_calendar else feed.resolve_calendar(calendar=calendar)
    duration_horizon = isinstance(config.max_holding_period, timedelta) or (
        isinstance(config.max_holding_period, str) and is_duration_string(config.max_holding_period)
    )
    declared_unit = window_unit if window_unit is not None else config.window_unit
    unit = (
        WindowUnit(declared_unit)
        if declared_unit is not None
        else (
            WindowUnit.ELAPSED
            if duration_horizon
            else WindowUnit.OBSERVATION
            if selected is None or legacy_selector
            else WindowUnit.EXPECTED_BAR
        )
    )
    if selected is None and (unit != WindowUnit.OBSERVATION or duration_horizon):
        raise ValueError(
            "calendar=None requires observation-count holding periods and window_unit='observation'"
        )
    if (unit == WindowUnit.ELAPSED) != duration_horizon:
        raise ValueError(
            "elapsed windows require duration holding periods; count windows require observation/expected_bar"
        )
    if declared_feed is not None and contract is None and config.data_contract is None:
        from ml4t.engineer.config import DataContractConfig as _DataContractConfig

        contract = _DataContractConfig(
            timestamp_col=feed.timestamp_col,
            price_col=feed.price_col,
            symbol_col=feed.entity_col,
        )
    resolved_price_col, resolved_ts_col, resolved_group_cols = resolve_labeling_columns(
        data=data,
        price_col=price_col,
        timestamp_col=timestamp_col,
        group_col=group_col,
        config=config,
        contract=contract,
        require_timestamp=selected is not None,
        allow_temporal_contract=True,
    )
    if config.method != "triple_barrier":
        raise ValueError("calendar_aware_labels requires LabelingConfig.method='triple_barrier'.")

    if selected is None:
        return _segment_labels(data, config, resolved_price_col, resolved_group_cols)
    daily_labels = feed.timestamp_semantics == TimestampSemantics.SESSION_LABEL
    if daily_labels and unit == WindowUnit.ELAPSED:
        raise ValueError("session_label identifiers cannot define an elapsed horizon")
    if declared_feed is not None and feed.timestamp_semantics is None:
        raise ValueError("Calendar-aware FeedSpec requires declared timestamp_semantics")
    if declared_feed is not None and selected == "auto":
        raise ValueError(
            "auto gap inference is a legacy mode, not an authoritative FeedSpec calendar"
        )
    assert resolved_ts_col is not None
    dtype = data.schema[resolved_ts_col]
    if (
        declared_feed is not None
        and not daily_labels
        and isinstance(dtype, pl.Datetime)
        and dtype.time_zone is None
        and feed.timezone is None
    ):
        raise ValueError("Calendar-aware FeedSpec requires timezone metadata for naive timestamps")
    if unit in {WindowUnit.EXPECTED_BAR, WindowUnit.SESSION} or daily_labels:
        if not isinstance(selected, str) or selected == "auto":
            raise ValueError("Scheduled windows require an authoritative named calendar")
        from ml4t.engineer.core.bar_grid import validate_declared_grid

        validate_declared_grid(
            data,
            feed_spec=feed,
            calendar=selected,
            timestamp_col=resolved_ts_col,
            group_cols=resolved_group_cols,
        )
    if unit == WindowUnit.SESSION:
        assert isinstance(selected, str)
        return _session_window_labels(
            data, config, feed, selected, resolved_price_col, resolved_ts_col, resolved_group_cols
        )
    if daily_labels:
        return _segment_labels(
            data.sort([*resolved_group_cols, resolved_ts_col]),
            config,
            resolved_price_col,
            resolved_group_cols,
            resolved_ts_col,
        ).with_columns(pl.lit(None).alias("label_duration"))

    # Create calendar instance if string provided
    if isinstance(selected, str):
        if selected == "auto":
            cal = SimpleTradingCalendar()
        else:
            # Always use pandas_market_calendars for string calendars
            cal = PandasMarketCalendar(selected)
    else:
        cal = selected

    session_col = _temporary_column_name(data.columns, "__ml4t_session_id")
    assert resolved_ts_col is not None

    if isinstance(cal, SimpleTradingCalendar):
        sort_cols = [*resolved_group_cols, resolved_ts_col]
        sorted_data = data.sort(sort_cols)
        time_diff_col = _temporary_column_name(sorted_data.columns, "__ml4t_time_diff")
        new_session_col = _temporary_column_name(sorted_data.columns, "__ml4t_new_session")
        time_diff_expr = pl.col(resolved_ts_col).diff()
        if resolved_group_cols:
            time_diff_expr = time_diff_expr.over(resolved_group_cols)

        data_with_session = sorted_data.with_columns(
            time_diff_expr.alias(time_diff_col)
        ).with_columns(
            (
                pl.col(time_diff_col)
                > pl.duration(minutes=int(cal.gap_threshold.total_seconds() / 60))
            )
            .fill_null(False)
            .alias(new_session_col)
        )

        session_expr = pl.col(new_session_col).cum_sum()
        if resolved_group_cols:
            session_expr = session_expr.over(resolved_group_cols)
        data_with_session = data_with_session.with_columns(session_expr.alias(session_col)).drop(
            [time_diff_col, new_session_col]
        )
    else:
        sorted_data, session_ids = _calendar_session_ids(
            data,
            cal,
            resolved_ts_col,
            resolved_group_cols,
            timezone=feed.timezone,
            bar_close=feed.timestamp_semantics == TimestampSemantics.BAR_CLOSE
            or feed.bar_label == "right",
        )
        data_with_session = sorted_data.with_columns(
            pl.Series(session_col, session_ids, dtype=pl.Int64)
        )

    session_group_cols = (
        [*resolved_group_cols, session_col] if resolved_group_cols else [session_col]
    )
    labeled = _segment_labels(
        data_with_session,
        config,
        resolved_price_col,
        session_group_cols,
        resolved_ts_col,
        timezone=feed.timezone,
    )
    return labeled.drop(session_col)


__all__ = [
    "PandasMarketCalendar",
    "SimpleTradingCalendar",
    "TradingCalendar",
    "calendar_aware_labels",
]
