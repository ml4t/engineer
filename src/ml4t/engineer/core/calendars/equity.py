"""
Equity market calendar implementation.
"""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from .base import ExchangeCalendar


class EquityCalendar(ExchangeCalendar):
    """Traditional equity market calendar (NYSE/NASDAQ).

    Handles timezone-aware datetime operations for US equity markets.
    Defaults to America/New_York timezone for all operations.
    """

    def __init__(self, exchange: str = "NYSE", timezone: str | None = None):
        self.exchange = exchange
        self._timezone = timezone or "America/New_York"
        self._tz = ZoneInfo(self._timezone)
        self._calendar = None
        self._load_calendar()

    def _load_calendar(self) -> None:
        """Load calendar using pandas_market_calendars if available."""
        try:
            import pandas_market_calendars as mcal

            # Map exchange names to pandas_market_calendars names
            exchange_map = {
                "NYSE": "NYSE",
                "NASDAQ": "NASDAQ",
                "XNYS": "NYSE",  # Alternative name
                "XNAS": "NASDAQ",  # Alternative name
            }

            exchange_name = exchange_map.get(self.exchange, self.exchange)
            self._calendar = mcal.get_calendar(exchange_name)
        except (ImportError, ValueError):
            # Fallback to basic implementation if library not available or exchange not found
            self._calendar = None

    @property
    def name(self) -> str:
        return self.exchange

    @property
    def timezone(self) -> str:
        return self._timezone

    def is_session(self, dt: datetime) -> bool:
        """Check if datetime is during trading session."""
        if self._calendar:
            # pandas_market_calendars expects date + time check
            # First check if the date is a valid trading day
            date_only = dt.date()

            # Get trading schedule for this date
            schedule = self._calendar.schedule(start_date=date_only, end_date=date_only)

            if schedule.empty:
                return False  # Not a trading day

            # Check if time is within trading hours
            market_open = schedule.iloc[0]["market_open"]
            market_close = schedule.iloc[0]["market_close"]

            # Convert dt to timezone-aware if needed
            if dt.tzinfo is None:
                # Assume datetime is in market timezone
                dt_in_market_tz = dt.replace(tzinfo=self._tz)
            else:
                # Convert to market timezone
                dt_in_market_tz = dt.astimezone(self._tz)

            return market_open <= dt_in_market_tz <= market_close

        # Basic fallback: weekdays 9:30-16:00 ET
        return self._is_basic_session(dt)

    def _is_basic_session(self, dt: datetime) -> bool:
        """Basic session check without holidays."""
        # Convert to market timezone if needed
        dt_market = dt.replace(tzinfo=self._tz) if dt.tzinfo is None else dt.astimezone(self._tz)

        # Check weekday
        if dt_market.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check time (9:30 AM - 4:00 PM ET)
        market_open = time(9, 30)
        market_close = time(16, 0)

        current_time = dt_market.time()
        return market_open <= current_time <= market_close

    def next_open(self, dt: datetime) -> datetime:
        """Next market open after given datetime."""
        if self._calendar:
            # pandas_market_calendars has different API
            from datetime import timedelta

            # Get schedule for next few days to find next open
            end_date = dt.date() + timedelta(days=10)  # Look ahead 10 days
            schedule = self._calendar.schedule(start_date=dt.date(), end_date=end_date)

            if not schedule.empty:
                # Find first market open after current datetime
                for _, row in schedule.iterrows():
                    market_open = row["market_open"]
                    if market_open > dt:
                        return market_open.to_pydatetime()

        # Basic implementation fallback
        return self._next_basic_open(dt)

    def _next_basic_open(self, dt: datetime) -> datetime:
        """Basic next open calculation with timezone handling."""
        # Ensure we're working in market timezone
        dt = dt.replace(tzinfo=self._tz) if dt.tzinfo is None else dt.astimezone(self._tz)

        # Start from next day if after market close
        if dt.time() >= time(16, 0):
            dt = dt.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(days=1)
        else:
            dt = dt.replace(hour=9, minute=30, second=0, microsecond=0)

        # Skip weekends
        while dt.weekday() >= 5:
            dt += timedelta(days=1)

        return dt

    def previous_close(self, dt: datetime) -> datetime:
        """Previous market close before given datetime."""
        if self._calendar:
            # pandas_market_calendars approach
            from datetime import timedelta

            # Get schedule for previous few days
            start_date = dt.date() - timedelta(days=10)
            schedule = self._calendar.schedule(start_date=start_date, end_date=dt.date())

            if not schedule.empty:
                # Find last market close before current datetime
                for i in range(len(schedule) - 1, -1, -1):
                    row = schedule.iloc[i]
                    market_close = row["market_close"]
                    if market_close < dt:
                        return market_close.to_pydatetime()

        return self._previous_basic_close(dt)

    def _previous_basic_close(self, dt: datetime) -> datetime:
        """Basic previous close calculation with timezone handling."""
        # Ensure we're working in market timezone
        dt = dt.replace(tzinfo=self._tz) if dt.tzinfo is None else dt.astimezone(self._tz)

        # If before market open, go to previous day
        if dt.time() < time(9, 30):
            dt = dt.replace(hour=16, minute=0, second=0, microsecond=0) - timedelta(days=1)
        else:
            dt = dt.replace(hour=16, minute=0, second=0, microsecond=0)

        # Skip weekends backwards
        while dt.weekday() >= 5:
            dt -= timedelta(days=1)

        return dt

    def sessions_between(self, start: datetime, end: datetime) -> list[datetime]:
        """List all trading sessions between dates."""
        if self._calendar:
            # pandas_market_calendars approach
            schedule = self._calendar.schedule(start_date=start.date(), end_date=end.date())

            if not schedule.empty:
                # Return market open times as session starts
                return [
                    row["market_open"].to_pydatetime()
                    for _, row in schedule.iterrows()
                    if start <= row["market_open"].to_pydatetime() <= end
                ]

        return self._basic_sessions_between(start, end)

    def _basic_sessions_between(self, start: datetime, end: datetime) -> list[datetime]:
        """Basic session enumeration."""
        sessions = []
        current = start.replace(hour=9, minute=30, second=0, microsecond=0)

        while current <= end:
            if self._is_basic_session(current):
                sessions.append(current)
            current += timedelta(days=1)

        return sessions

    def session_duration(self, date: datetime) -> timedelta:
        """Duration of trading session on given date."""
        if self._calendar:
            # pandas_market_calendars approach
            schedule = self._calendar.schedule(start_date=date.date(), end_date=date.date())

            if not schedule.empty:
                row = schedule.iloc[0]
                market_open = row["market_open"]
                market_close = row["market_close"]
                return market_close - market_open

        # Standard 6.5 hour session (9:30 AM - 4:00 PM)
        return timedelta(hours=6, minutes=30)
