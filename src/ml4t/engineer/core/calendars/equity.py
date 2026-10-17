"""Authoritative exchange schedules for equity calendar helpers."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

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
        """Return tradable duration on the exchange-local date, zero on a closure."""
        day = self._normalize(date).astimezone(self._calendar.tz).date()
        duration = timedelta(0)
        for opened, closed, _, _ in self._adapter._trading_intervals_around(day):
            # The schedule adapter includes surrounding dates for overnight lookup.
            if opened.astimezone(self._calendar.tz).date() == day:
                duration += closed - opened
        return duration
