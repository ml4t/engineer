"""Authoritative exchange schedules for equity calendar helpers."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from .base import ExchangeCalendar


class EquityCalendar(ExchangeCalendar):
    """Schedule-backed market hours with naive times in the configured exchange timezone.

    ``calendar`` is the canonical selector; ``exchange`` remains a compatible alias.
    A calendar object requires a named schedule. Calendar-free calculations belong
    to consumers accepting ``calendar=None``, not an approximate weekday schedule.
    """

    def __init__(
        self,
        exchange: str | CalendarUnset = UNSET_CALENDAR,
        timezone: str | None = None,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
    ):
        from ml4t.engineer.labeling.calendar import PandasMarketCalendar

        aliases = {"XNYS": "NYSE", "XNAS": "NASDAQ"}
        if exchange is not UNSET_CALENDAR:
            if not isinstance(exchange, str) or not exchange:
                raise ValueError("exchange must name a calendar")
            if calendar is not UNSET_CALENDAR and aliases.get(exchange, exchange) != aliases.get(
                calendar, calendar
            ):
                raise ValueError("calendar and exchange selectors conflict")
        selected = FeedSpec(time_contract_version=2).resolve_calendar(
            calendar=exchange if calendar is UNSET_CALENDAR else calendar
        )
        if selected is None:
            raise ValueError("EquityCalendar requires a named calendar, not calendar=None")
        self.exchange = exchange if isinstance(exchange, str) else selected
        self._adapter = PandasMarketCalendar(aliases.get(selected, selected))
        self._calendar = self._adapter._calendar
        self._timezone = timezone if timezone is not None else str(self._calendar.tz)
        self._tz = ZoneInfo(self._timezone)

    @property
    def name(self) -> str:
        return self.exchange

    @property
    def timezone(self) -> str:
        return self._timezone

    def _normalize(self, dt: datetime) -> datetime:
        if not isinstance(dt, datetime):
            raise TypeError("calendar timestamps must be datetime values")
        if dt.tzinfo is None or dt.utcoffset() is None:
            first = dt.replace(tzinfo=self._tz, fold=0)
            second = dt.replace(tzinfo=self._tz, fold=1)
            if first.utcoffset() != second.utcoffset():
                raise ValueError("ambiguous or nonexistent local calendar timestamp")
            dt = first
        return dt.astimezone(UTC)

    def is_session(self, dt: datetime) -> bool:
        """Check authoritative membership, excluding breaks and interruptions."""
        return self._adapter.is_trading_time(self._normalize(dt))

    def _adjacent_boundary(self, dt: datetime, *, forward: bool) -> datetime:
        instant = self._normalize(dt)
        day = instant.astimezone(self._calendar.tz).date()
        column = "market_open" if forward else "market_close"
        for days in (10, 31, 366):
            schedule = self._calendar.schedule(
                start_date=day - timedelta(days=1 if forward else days),
                end_date=day + timedelta(days=days if forward else 1),
            )
            if schedule.empty:
                continue
            boundaries = schedule[column]
            eligible = (
                boundaries[boundaries > instant] if forward else boundaries[boundaries < instant]
            )
            if not eligible.empty:
                return (eligible.iloc[0] if forward else eligible.iloc[-1]).to_pydatetime()
        raise ValueError(f"No scheduled {column} within 366 days of {instant.isoformat()}")

    def next_open(self, dt: datetime) -> datetime:
        """Return the next session open, or fail if no schedule exists within a year."""
        return self._adjacent_boundary(dt, forward=True)

    def previous_close(self, dt: datetime) -> datetime:
        """Return the previous session close, or fail if none exists within a year."""
        return self._adjacent_boundary(dt, forward=False)

    def sessions_between(self, start: datetime, end: datetime) -> list[datetime]:
        """Return authoritative session opens inside the inclusive instant interval."""
        start, end = self._normalize(start), self._normalize(end)
        if start > end:
            raise ValueError("session interval start must not follow end")
        schedule = self._calendar.schedule(
            start_date=start.astimezone(self._calendar.tz).date() - timedelta(days=1),
            end_date=end.astimezone(self._calendar.tz).date() + timedelta(days=1),
        )
        if schedule.empty:
            return []
        return [
            opened.to_pydatetime() for opened in schedule["market_open"] if start <= opened <= end
        ]

    def session_duration(self, date: datetime) -> timedelta:
        """Return tradable duration for the supplied session-label date, zero on a closure."""
        day = date.date()
        sessions = [
            (session["market_open"], session["market_close"])
            for label, session in self._adapter._schedule_around(day).iterrows()
            if label.date() == day
        ]
        return sum(
            (
                closed - opened
                for opened, closed, _, _ in self._adapter._trading_intervals_around(day)
                if any(start <= opened <= closed <= end for start, end in sessions)
            ),
            timedelta(0),
        )

    def _session_label(self, dt: datetime) -> date | None:
        instant = self._normalize(dt)
        if not self._adapter.is_trading_time(instant):
            return None
        for label, session in self._adapter._schedule_around(instant.date()).iterrows():
            if session["market_open"] <= instant <= session["market_close"]:
                return label.date()
        return None

    def add_session_info(self, df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
        """Add schedule membership, authoritative session label and exchange-local clock time."""
        timestamps = pl.col(timestamp_col)
        dtype = df.schema[timestamp_col]
        if not isinstance(dtype, pl.Datetime):
            raise TypeError("session annotations require datetime instants, not session labels")
        if dtype.time_zone is None:
            timestamps = timestamps.dt.replace_time_zone(
                self.timezone, ambiguous="raise", non_existent="raise"
            )
        return df.with_columns(
            pl.col(timestamp_col)
            .map_elements(self.is_session, return_dtype=pl.Boolean)
            .alias("is_session"),
            pl.col(timestamp_col)
            .map_elements(self._session_label, return_dtype=pl.Date)
            .alias("session_date"),
            timestamps.dt.convert_time_zone(str(self._calendar.tz)).dt.time().alias("session_time"),
        )
