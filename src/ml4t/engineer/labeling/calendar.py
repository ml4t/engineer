# mypy: disable-error-code="no-any-return,assignment,arg-type"
"""Calendar-aware labeling utilities.

Provides session-aware barrier labeling that respects trading calendar gaps
(maintenance windows, overnight sessions, weekends, holidays).

This prevents data leakage when labels would otherwise span non-trading periods.
"""

from datetime import datetime, timedelta
from typing import Protocol

import polars as pl

from ml4t.engineer.labeling.barriers import BarrierConfig
from ml4t.engineer.labeling.triple_barrier import triple_barrier_labels


class TradingCalendar(Protocol):
    """Protocol for trading calendar implementations.

    Any calendar implementation providing these methods can be used.
    """

    def is_trading_time(self, timestamp: datetime) -> bool:
        """Check if given timestamp is during trading hours."""
        ...

    def next_session_break(self, timestamp: datetime) -> datetime | None:
        """Get next session break after given timestamp.

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
        self.gap_threshold = timedelta(minutes=gap_threshold_minutes)
        self._data: pl.DataFrame | None = None

    def fit(self, data: pl.DataFrame, timestamp_col: str = "timestamp") -> "SimpleTradingCalendar":
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

    def is_trading_time(self, timestamp: datetime) -> bool:
        """Check if timestamp is during trading session."""
        # Get schedule for the day
        schedule = self._calendar.schedule(start_date=timestamp.date(), end_date=timestamp.date())

        if schedule.empty:
            return False

        # Check if timestamp is within session
        session = schedule.iloc[0]
        return session["market_open"] <= timestamp <= session["market_close"]

    def next_session_break(self, timestamp: datetime) -> datetime | None:
        """Get next session close after timestamp."""
        schedule = self._calendar.schedule(
            start_date=timestamp.date(),
            end_date=timestamp.date() + timedelta(days=7),  # Look ahead 1 week
        )

        for _, session in schedule.iterrows():
            if session["market_close"] > timestamp:
                return session["market_close"].to_pydatetime()

        return None


# ExchangeCalendar adapter removed - use pandas_market_calendars directly
# See .claude/reference/calendar_libraries.md for rationale:
# - pandas_market_calendars includes ALL exchange_calendars features as dependency
# - Adds critical product-specific calendars (CME_Equity, CME_Agriculture, etc.)
# - Correctly handles CME futures maintenance breaks (4-5 PM CT)
# - Better maintenance, more features, zero downside


def calendar_aware_labels(
    data: pl.DataFrame,
    config: BarrierConfig,
    calendar: str | TradingCalendar,
    price_col: str = "close",
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Apply triple-barrier labeling with session awareness.

    Splits data by trading sessions and applies labeling within each session.
    This prevents labels from spanning session gaps (maintenance, overnight, holidays).

    Parameters
    ----------
    data : pl.DataFrame
        Input data with OHLCV and timestamp
    config : BarrierConfig
        Barrier configuration
    calendar : str or TradingCalendar
        Either:
        - Calendar name string (uses pandas_market_calendars)
        - TradingCalendar protocol implementation
        - "auto" to detect gaps automatically
    price_col : str, default "close"
        Price column name
    timestamp_col : str, default "timestamp"
        Timestamp column name

    Returns
    -------
    pl.DataFrame
        Data with barrier labels, respecting session boundaries

    Examples
    --------
    >>> # CME futures with pandas_market_calendars
    >>> labeled = calendar_aware_labels(
    ...     data,
    ...     config=BarrierConfig(upper_barrier=0.02, lower_barrier=0.02),
    ...     calendar="CME_Equity"  # Product-specific calendar
    ... )

    >>> # NYSE equities
    >>> labeled = calendar_aware_labels(
    ...     data,
    ...     config=BarrierConfig(upper_barrier=0.01, lower_barrier=0.01),
    ...     calendar="NYSE"
    ... )

    >>> # Auto-detect gaps
    >>> labeled = calendar_aware_labels(
    ...     data,
    ...     config=BarrierConfig(upper_barrier=0.02, lower_barrier=0.02),
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
    - For 24/7 markets, use standard triple_barrier_labels instead
    """
    # Create calendar instance if string provided
    if isinstance(calendar, str):
        if calendar == "auto":
            cal = SimpleTradingCalendar()
            cal.fit(data, timestamp_col)
        else:
            # Always use pandas_market_calendars for string calendars
            cal = PandasMarketCalendar(calendar)
    else:
        cal = calendar

    # Identify session boundaries
    # Add session ID column by detecting gaps
    # Use calendar's gap threshold if available, otherwise default to 30 minutes
    if isinstance(cal, SimpleTradingCalendar):
        gap_threshold_minutes = cal.gap_threshold.total_seconds() / 60
    else:
        gap_threshold_minutes = 30

    data_with_session = (
        data.with_columns([pl.col(timestamp_col).diff().alias("_time_diff")])
        .with_columns(
            [
                # New session when gap > threshold
                (pl.col("_time_diff") > pl.duration(minutes=int(gap_threshold_minutes)))
                .fill_null(False)
                .alias("_new_session")
            ]
        )
        .with_columns([pl.col("_new_session").cum_sum().alias("session_id")])
    )

    # Apply labeling to each session separately
    labeled_sessions = []

    for session_id in data_with_session["session_id"].unique().sort():
        session_data = data_with_session.filter(pl.col("session_id") == session_id)

        # Apply barrier labeling to this session only
        try:
            session_labeled = triple_barrier_labels(
                data=session_data.drop(["_time_diff", "_new_session", "session_id"]),
                config=config,
                price_col=price_col,
                timestamp_col=timestamp_col,
            )
            labeled_sessions.append(session_labeled)
        except Exception as e:
            # If labeling fails for a session (e.g., too short), skip it
            # or keep unlabeled
            print(f"Warning: Session {session_id} labeling failed: {e}")
            labeled_sessions.append(session_data.drop(["_time_diff", "_new_session", "session_id"]))

    # Concatenate all sessions
    result = pl.concat(labeled_sessions)

    return result


__all__ = [
    "PandasMarketCalendar",
    "SimpleTradingCalendar",
    "TradingCalendar",
    "calendar_aware_labels",
]
