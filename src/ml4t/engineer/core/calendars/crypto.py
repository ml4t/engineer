"""
Cryptocurrency market calendar (24/7 trading).
"""

from datetime import datetime, timedelta

from .base import ExchangeCalendar


class CryptoCalendar(ExchangeCalendar):
    """24/7 crypto market calendar with optional maintenance windows."""

    def __init__(self, maintenance_window: tuple[int, int] | None = None):
        """
        Initialize crypto calendar.

        Parameters
        ----------
        maintenance_window : tuple of (start_hour, end_hour) in UTC
            Optional maintenance window when trading is paused (e.g., (4, 5) for 4-5 AM UTC)
        """
        self.maintenance_window = maintenance_window

    @property
    def name(self) -> str:
        return "CRYPTO_24_7"

    @property
    def timezone(self) -> str:
        return "UTC"

    def is_session(self, dt: datetime) -> bool:
        """Always true except during maintenance window."""
        if self.maintenance_window:
            start_hour, end_hour = self.maintenance_window
            hour = dt.hour
            return not (start_hour <= hour < end_hour)
        return True

    def next_open(self, dt: datetime) -> datetime:
        """Next market open (either immediate or after maintenance)."""
        if self.is_session(dt):
            return dt

        if self.maintenance_window:
            # Next opening is at the end of maintenance window
            start_hour, end_hour = self.maintenance_window
            next_open = dt.replace(hour=end_hour, minute=0, second=0, microsecond=0)

            # If we're already past end hour today, it's immediate
            if dt.hour >= end_hour:
                return dt

            return next_open

        return dt

    def previous_close(self, dt: datetime) -> datetime:
        """Previous market close (start of maintenance window)."""
        if self.maintenance_window:
            start_hour, end_hour = self.maintenance_window

            if dt.hour < start_hour:
                # Maintenance was yesterday
                prev_close = (dt - timedelta(days=1)).replace(
                    hour=start_hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            else:
                # Maintenance was today or we're in it
                prev_close = dt.replace(
                    hour=start_hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                )

            return prev_close

        # No maintenance, so no real "close"
        return dt - timedelta(seconds=1)

    def sessions_between(self, start: datetime, end: datetime) -> list[datetime]:
        """
        List trading sessions. For 24/7 markets, this is every hour
        outside maintenance windows.
        """
        sessions = []
        current = start.replace(minute=0, second=0, microsecond=0)

        while current <= end:
            if self.is_session(current):
                sessions.append(current)
            current += timedelta(hours=1)

        return sessions

    def session_duration(self, date: datetime) -> timedelta:  # noqa: ARG002 - interface requirement
        """Duration of trading session."""
        if self.maintenance_window:
            start_hour, end_hour = self.maintenance_window
            maintenance_duration = end_hour - start_hour
            return timedelta(hours=24 - maintenance_duration)

        return timedelta(hours=24)
