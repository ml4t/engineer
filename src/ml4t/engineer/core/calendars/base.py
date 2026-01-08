"""
Base exchange calendar interface.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import polars as pl


class ExchangeCalendar(ABC):
    """Base class for all exchange calendars."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name (e.g., 'NYSE', 'CME', 'CRYPTO')."""

    @property
    @abstractmethod
    def timezone(self) -> str:
        """Exchange timezone (e.g., 'America/New_York')."""

    @abstractmethod
    def is_session(self, dt: datetime) -> bool:
        """Check if datetime is during trading session."""

    @abstractmethod
    def next_open(self, dt: datetime) -> datetime:
        """Next market open after given datetime."""

    @abstractmethod
    def previous_close(self, dt: datetime) -> datetime:
        """Previous market close before given datetime."""

    @abstractmethod
    def sessions_between(self, start: datetime, end: datetime) -> list[datetime]:
        """List all trading sessions between dates."""

    @abstractmethod
    def session_duration(self, date: datetime) -> timedelta:
        """Duration of trading session on given date."""

    def filter_sessions(self, df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
        """Filter DataFrame to only include trading session data."""
        return df.filter(
            pl.col(timestamp_col).map_elements(
                lambda dt: self.is_session(dt),
                return_dtype=pl.Boolean,
            ),
        )

    def add_session_info(self, df: pl.DataFrame, timestamp_col: str = "timestamp") -> pl.DataFrame:
        """Add session-related columns to DataFrame."""
        return df.with_columns(
            [
                pl.col(timestamp_col)
                .map_elements(
                    lambda dt: self.is_session(dt),
                    return_dtype=pl.Boolean,
                )
                .alias("is_session"),
                pl.col(timestamp_col).dt.date().alias("session_date"),
                pl.col(timestamp_col).dt.time().alias("session_time"),
            ],
        )
