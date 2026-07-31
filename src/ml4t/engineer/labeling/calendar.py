"""Calendar-aware labeling utilities.

Provides session-aware barrier labeling that respects trading calendar gaps
(maintenance windows, overnight sessions, weekends, holidays).

This prevents data leakage when labels would otherwise span non-trading periods.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from inspect import signature
from typing import TYPE_CHECKING, Any, Protocol

import polars as pl

from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling.triple_barrier import triple_barrier_labels
from ml4t.engineer.labeling.utils import resolve_labeling_columns

if TYPE_CHECKING:
    from ml4t.engineer.config import DataContractConfig, LabelingConfig


class TradingCalendar(Protocol):
    """Protocol for trading calendar implementations.

    Any calendar implementation providing these methods can be used.
    """

    def is_trading_time(self, timestamp: datetime) -> bool:
        """Check if given timestamp is during trading hours."""
        ...

    def next_session_break(self, timestamp: datetime) -> datetime | None:
        """Get the next session break at or after the given timestamp.

        Returns None if no break before end of data.
        """
        ...


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

    def __init__(self, calendar_name: str):
        """
        Parameters
        ----------
        calendar_name : str
            Calendar name (e.g., "CME_Equity", "NYSE", "LSE")
            See pandas_market_calendars.get_calendar_names()
        """
        try:
            import pandas_market_calendars as mcal
        except ImportError as err:
            raise ImportError(
                "pandas_market_calendars required for PandasMarketCalendar. "
                "Install with: pip install pandas-market-calendars"
            ) from err

        self.calendar_name = calendar_name
        self._calendar = mcal.get_calendar(calendar_name)
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
            if not pd.isna(start) and not pd.isna(end):
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

    def _trading_boundary(self, timestamp: datetime) -> tuple[bool, datetime | None]:
        """Return whether a timestamp is trading and its next session boundary."""
        normalized = self._normalize_timestamp(timestamp)
        for start, end, boundary, end_inclusive in self._trading_intervals_around(
            normalized.date()
        ):
            if self._contains_timestamp(normalized, start, end, end_inclusive):
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
) -> tuple[pl.DataFrame, list[int]]:
    """Sort rows and derive sessions from an explicit calendar."""
    _validate_explicit_calendar(calendar)
    sort_cols = [*group_cols, timestamp_col]
    sorted_data = data.sort(sort_cols)
    selected_cols = [*group_cols, timestamp_col]

    session_ids: list[int] = []
    previous_group: tuple[Any, ...] | None = None
    previous_break: datetime | None = None
    session_id = -1

    for row in sorted_data.select(selected_cols).iter_rows(named=False):
        group = tuple(row[: len(group_cols)])
        timestamp = row[-1]
        if not isinstance(timestamp, datetime):
            raise TypeError(f"timestamp column {timestamp_col!r} must contain datetime values")

        if isinstance(calendar, PandasMarketCalendar):
            is_trading, session_break = calendar._trading_boundary(timestamp)
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


# ExchangeCalendar adapter removed - use pandas_market_calendars directly
# See .claude/reference/calendar_libraries.md for rationale:
# - pandas_market_calendars includes ALL exchange_calendars features as dependency
# - Adds critical product-specific calendars (CME_Equity, CME_Agriculture, etc.)
# - Correctly handles CME futures maintenance breaks (4-5 PM CT)
# - Better maintenance, more features, zero downside


def calendar_aware_labels(
    data: pl.DataFrame,
    config: LabelingConfig,
    calendar: str | TradingCalendar,
    price_col: str | None = None,
    timestamp_col: str | None = None,
    group_col: str | list[str] | None = None,
    contract: DataContractConfig | None = None,
) -> pl.DataFrame:
    """Apply triple-barrier labeling with session awareness.

    Splits data by trading sessions and applies labeling within each session.
    This prevents labels from spanning session gaps (maintenance, overnight, holidays).

    Parameters
    ----------
    data : pl.DataFrame
        Input data with OHLCV and timestamp
    config : LabelingConfig
        Barrier configuration
    calendar : str or TradingCalendar
        Either:
        - Calendar name string (uses pandas_market_calendars)
        - TradingCalendar protocol implementation
        - "auto" to detect gaps automatically
    price_col : str | None, default None
        Price column name
    timestamp_col : str | None, default None
        Timestamp column name
    group_col : str | list[str] | None, default None
        Grouping column(s) for panel-aware session labeling.
    contract : DataContractConfig | None, default None
        Optional shared dataframe contract. Used after config and before defaults.

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
    - Labels that would span session breaks are truncated at the break
    - This may result in more timeout labels near session closes
    - Rows outside an explicit calendar's trading times are rejected
    - Naive datetimes passed to PandasMarketCalendar are interpreted as UTC
    - For 24/7 markets, use standard triple_barrier_labels instead
    """
    from ml4t.engineer.config import LabelingConfig as _LabelingConfig

    if not isinstance(config, _LabelingConfig):
        raise TypeError(
            "calendar_aware_labels expects LabelingConfig. "
            "Legacy BarrierConfig inputs are no longer supported; "
            "use LabelingConfig.triple_barrier(...)."
        )

    resolved_price_col, resolved_ts_col, resolved_group_cols = resolve_labeling_columns(
        data=data,
        price_col=price_col,
        timestamp_col=timestamp_col,
        group_col=group_col,
        config=config,
        contract=contract,
        require_timestamp=True,
    )
    if config.method != "triple_barrier":
        raise ValueError("calendar_aware_labels requires LabelingConfig.method='triple_barrier'.")

    # Create calendar instance if string provided
    if isinstance(calendar, str):
        if calendar == "auto":
            cal = SimpleTradingCalendar()
        else:
            # Always use pandas_market_calendars for string calendars
            cal = PandasMarketCalendar(calendar)
    else:
        cal = calendar

    session_col = _temporary_column_name(data.columns, "__ml4t_session_id")

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
        )
        data_with_session = sorted_data.with_columns(
            pl.Series(session_col, session_ids, dtype=pl.Int64)
        )

    session_group_cols = (
        [*resolved_group_cols, session_col] if resolved_group_cols else [session_col]
    )
    labeled = triple_barrier_labels(
        data=data_with_session,
        config=config,
        price_col=resolved_price_col,
        timestamp_col=resolved_ts_col,
        group_col=session_group_cols,
    )
    return labeled.drop(session_col)


__all__ = [
    "PandasMarketCalendar",
    "SimpleTradingCalendar",
    "TradingCalendar",
    "calendar_aware_labels",
]
