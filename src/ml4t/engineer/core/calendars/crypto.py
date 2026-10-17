"""UTC daily crypto sessions with explicit recurring maintenance breaks."""

from datetime import UTC, date, datetime, time, timedelta

import polars as pl
from ml4t.specs import UNSET_CALENDAR, CalendarUnset, FeedSpec

from .base import ExchangeCalendar


class CryptoCalendar(ExchangeCalendar):
    """Authoritative 24/7 daily sessions, optionally shortened by UTC maintenance.

    Maintenance is a half-open daily interval, including across midnight. A
    maintenance reopen is not another exchange session. Naive instants mean UTC.
    """

    def __init__(
        self,
        maintenance_window: tuple[int, int] | None = None,
        *,
        calendar: str | None | CalendarUnset = UNSET_CALENDAR,
    ) -> None:
        from ml4t.engineer.labeling.calendar import PandasMarketCalendar

        selected = FeedSpec(time_contract_version=2, calendar="24/7").resolve_calendar(
            calendar=calendar
        )
        if selected not in {"24/7", "CRYPTO_24_7"}:
            raise ValueError("CryptoCalendar requires calendar='24/7'; None is consumer row mode")
        if maintenance_window is not None:
            if (
                not isinstance(maintenance_window, tuple)
                or len(maintenance_window) != 2
                or any(
                    isinstance(hour, bool) or not isinstance(hour, int)
                    for hour in maintenance_window
                )
            ):
                raise ValueError("maintenance_window must be a pair of integer UTC hours")
            start, end = maintenance_window
            if not 0 <= start < 24 or not 0 <= end <= 24 or start == end or (start, end) == (0, 24):
                raise ValueError("maintenance_window must leave trading time within each UTC day")
        self._adapter = PandasMarketCalendar("24/7")
        self.maintenance_window = maintenance_window

    @property
    def name(self) -> str:
        return "CRYPTO_24_7"

    @property
    def timezone(self) -> str:
        return "UTC"

    def _normalize(self, dt: datetime) -> datetime:
        return self._adapter._normalize_timestamp(dt)

    def _session_intervals(self, day: date) -> list[tuple[datetime, datetime]]:
        intervals = [
            (opened, closed)
            for opened, closed, _, _ in self._adapter._trading_intervals_around(day)
            if opened.date() == day
        ]
        if self.maintenance_window is None:
            return intervals
        start, end = self.maintenance_window
        midnight = datetime.combine(day, time(), tzinfo=UTC)
        for offset in (-1, 0):
            opened = midnight + timedelta(days=offset, hours=start)
            closed = midnight + timedelta(days=offset + (end < start), hours=end)
            remaining = []
            for first, last in intervals:
                if last <= opened or first >= closed:
                    remaining.append((first, last))
                else:
                    if first < opened:
                        remaining.append((first, opened))
                    if closed < last:
                        remaining.append((closed, last))
            intervals = remaining
        return intervals

    def is_session(self, dt: datetime) -> bool:
        """Check UTC membership with left-closed, right-open instant boundaries."""
        instant = self._normalize(dt)
        return any(
            opened <= instant < closed for opened, closed in self._session_intervals(instant.date())
        )

    def _adjacent_boundary(self, dt: datetime, *, forward: bool) -> datetime:
        instant = self._normalize(dt)
        candidates = []
        for offset in (-2, -1, 0, 1):
            intervals = self._session_intervals(instant.date() + timedelta(days=offset))
            if intervals:
                boundary = intervals[0][0] if forward else intervals[-1][1]
                eligible = boundary > instant if forward else boundary < instant
                if eligible:
                    candidates.append(boundary)
        if not candidates:
            raise ValueError(f"No scheduled crypto session boundary near {instant.isoformat()}")
        return min(candidates) if forward else max(candidates)

    def next_open(self, dt: datetime) -> datetime:
        """Return the strictly next daily session open, not a maintenance reopen."""
        return self._adjacent_boundary(dt, forward=True)

    def previous_close(self, dt: datetime) -> datetime:
        """Return the strictly previous daily session close, not an intraday break."""
        return self._adjacent_boundary(dt, forward=False)

    def sessions_between(self, start: datetime, end: datetime) -> list[datetime]:
        """Return one daily session open inside the inclusive instant interval."""
        start, end = self._normalize(start), self._normalize(end)
        if start > end:
            raise ValueError("session interval start must not follow end")
        day = start.date()
        sessions = []
        while day <= end.date():
            intervals = self._session_intervals(day)
            if intervals and start <= intervals[0][0] <= end:
                sessions.append(intervals[0][0])
            day += timedelta(days=1)
        return sessions

    def session_duration(self, date: datetime) -> timedelta:
        """Return tradable duration of the supplied UTC session-label date."""
        return sum(
            (closed - opened for opened, closed in self._session_intervals(date.date())),
            timedelta(0),
        )

    def _session_label(self, dt: datetime) -> date | None:
        instant = self._normalize(dt)
        return instant.date() if self.is_session(instant) else None

    def add_session_info(self, df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
        """Add UTC membership, session labels and clock time without changing input rows."""
        dtype = df.schema[timestamp_col]
        if not isinstance(dtype, pl.Datetime):
            raise TypeError("session annotations require datetime instants, not session labels")
        timestamps = pl.col(timestamp_col)
        if dtype.time_zone is None:
            timestamps = timestamps.dt.replace_time_zone("UTC")
        return df.with_columns(
            pl.col(timestamp_col)
            .map_elements(self.is_session, return_dtype=pl.Boolean)
            .alias("is_session"),
            pl.col(timestamp_col)
            .map_elements(self._session_label, return_dtype=pl.Date)
            .alias("session_date"),
            timestamps.dt.convert_time_zone("UTC").dt.time().alias("session_time"),
        )
