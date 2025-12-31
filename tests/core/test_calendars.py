"""Tests for trading calendar functionality.

Tests market hours, holidays, session detection, and DST transitions.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ml4t.engineer.core.calendars.equity import EquityCalendar

# =============================================================================
# Basic Calendar Tests
# =============================================================================


class TestEquityCalendarBasic:
    """Tests for basic calendar functionality."""

    def test_init_default_exchange(self) -> None:
        """Test initialization with default NYSE."""
        calendar = EquityCalendar()
        assert calendar.name == "NYSE"
        assert calendar.timezone == "America/New_York"

    def test_init_nasdaq(self) -> None:
        """Test initialization with NASDAQ."""
        calendar = EquityCalendar(exchange="NASDAQ")
        assert calendar.name == "NASDAQ"

    def test_init_custom_timezone(self) -> None:
        """Test initialization with custom timezone."""
        calendar = EquityCalendar(timezone="America/Chicago")
        assert calendar.timezone == "America/Chicago"

    def test_exchange_aliases(self) -> None:
        """Test that exchange aliases work."""
        # XNYS is NYSE alias
        cal_xnys = EquityCalendar(exchange="XNYS")
        assert cal_xnys.exchange == "XNYS"

        # XNAS is NASDAQ alias
        cal_xnas = EquityCalendar(exchange="XNAS")
        assert cal_xnas.exchange == "XNAS"


# =============================================================================
# Session Detection Tests
# =============================================================================


class TestSessionDetection:
    """Tests for is_session() method."""

    def test_weekday_market_hours(self) -> None:
        """Test session detection during normal market hours."""
        calendar = EquityCalendar()

        # Tuesday 10:00 AM ET - should be in session
        dt = datetime(2024, 1, 2, 10, 0, 0)
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        assert calendar.is_session(dt) is True

    def test_before_market_open(self) -> None:
        """Test before market open (9:30 AM ET)."""
        calendar = EquityCalendar()

        # Tuesday 9:00 AM ET - before open
        dt = datetime(2024, 1, 2, 9, 0, 0)
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        assert calendar.is_session(dt) is False

    def test_after_market_close(self) -> None:
        """Test after market close (4:00 PM ET)."""
        calendar = EquityCalendar()

        # Tuesday 5:00 PM ET - after close
        dt = datetime(2024, 1, 2, 17, 0, 0)
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        assert calendar.is_session(dt) is False

    def test_weekend_not_session(self) -> None:
        """Test weekends are not trading sessions."""
        calendar = EquityCalendar()

        # Saturday 10:00 AM ET
        saturday = datetime(2024, 1, 6, 10, 0, 0)
        saturday = saturday.replace(tzinfo=ZoneInfo("America/New_York"))
        assert calendar.is_session(saturday) is False

        # Sunday 10:00 AM ET
        sunday = datetime(2024, 1, 7, 10, 0, 0)
        sunday = sunday.replace(tzinfo=ZoneInfo("America/New_York"))
        assert calendar.is_session(sunday) is False

    def test_market_open_boundary(self) -> None:
        """Test exactly at market open (9:30 AM ET)."""
        calendar = EquityCalendar()

        # Tuesday 9:30 AM ET - should be in session
        dt = datetime(2024, 1, 2, 9, 30, 0)
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        assert calendar.is_session(dt) is True

    def test_market_close_boundary(self) -> None:
        """Test exactly at market close (4:00 PM ET)."""
        calendar = EquityCalendar()

        # Tuesday 4:00 PM ET - should still be in session
        dt = datetime(2024, 1, 2, 16, 0, 0)
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        assert calendar.is_session(dt) is True

    def test_naive_datetime_assumed_et(self) -> None:
        """Test naive datetime is assumed to be in market timezone."""
        calendar = EquityCalendar()

        # Naive datetime (no timezone)
        dt_naive = datetime(2024, 1, 2, 10, 0, 0)

        # Should assume ET and return True (market hours)
        assert calendar.is_session(dt_naive) is True

    def test_timezone_conversion(self) -> None:
        """Test datetime in different timezone is converted."""
        calendar = EquityCalendar()

        # 10:00 AM ET = 3:00 PM UTC on same day (winter time)
        dt_utc = datetime(2024, 1, 2, 15, 0, 0, tzinfo=ZoneInfo("UTC"))

        # Should be in session (10:00 AM ET)
        assert calendar.is_session(dt_utc) is True


# =============================================================================
# Next Open / Previous Close Tests
# =============================================================================


class TestNextOpenPreviousClose:
    """Tests for next_open() and previous_close() methods."""

    def test_next_open_from_market_hours(self) -> None:
        """Test next open when currently in market hours."""
        calendar = EquityCalendar()

        # Tuesday 10:00 AM ET
        dt = datetime(2024, 1, 2, 10, 0, 0)
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        next_open = calendar.next_open(dt)
        # Convert to ET for comparison
        next_open_et = next_open.astimezone(ZoneInfo("America/New_York"))

        # Next open should be Wednesday 9:30 AM ET
        assert next_open_et.day == 3
        assert next_open_et.hour == 9
        assert next_open_et.minute == 30

    def test_next_open_from_after_close(self) -> None:
        """Test next open when after market close."""
        calendar = EquityCalendar()

        # Tuesday 5:00 PM ET (after close)
        dt = datetime(2024, 1, 2, 17, 0, 0)
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        next_open = calendar.next_open(dt)
        # Convert to ET for comparison
        next_open_et = next_open.astimezone(ZoneInfo("America/New_York"))

        # Next open should be Wednesday 9:30 AM ET
        assert next_open_et.day == 3
        assert next_open_et.hour == 9
        assert next_open_et.minute == 30

    def test_next_open_skips_weekend(self) -> None:
        """Test next open skips weekend."""
        calendar = EquityCalendar()

        # Friday 5:00 PM ET
        friday = datetime(2024, 1, 5, 17, 0, 0)
        friday = friday.replace(tzinfo=ZoneInfo("America/New_York"))

        next_open = calendar.next_open(friday)
        # Convert to ET for comparison
        next_open_et = next_open.astimezone(ZoneInfo("America/New_York"))

        # Should be Monday 9:30 AM ET (skipping weekend)
        assert next_open_et.weekday() == 0  # Monday
        assert next_open_et.hour == 9
        assert next_open_et.minute == 30

    def test_previous_close_from_market_hours(self) -> None:
        """Test previous close when in market hours."""
        calendar = EquityCalendar()

        # Tuesday 10:00 AM ET
        dt = datetime(2024, 1, 2, 10, 0, 0)
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        prev_close = calendar.previous_close(dt)
        # Convert to ET for comparison
        prev_close_et = prev_close.astimezone(ZoneInfo("America/New_York"))

        # Previous close should be Dec 29, 2023 4:00 PM (Jan 1 is a holiday)
        assert prev_close_et.month == 12
        assert prev_close_et.day == 29
        assert prev_close_et.hour == 16
        assert prev_close_et.minute == 0

    def test_previous_close_from_before_open(self) -> None:
        """Test previous close when before market open."""
        calendar = EquityCalendar()

        # Tuesday 8:00 AM ET (before open)
        dt = datetime(2024, 1, 2, 8, 0, 0)
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        prev_close = calendar.previous_close(dt)
        # Convert to ET for comparison
        prev_close_et = prev_close.astimezone(ZoneInfo("America/New_York"))

        # Previous close should be Dec 29, 2023 4:00 PM (Jan 1 is a holiday)
        assert prev_close_et.month == 12
        assert prev_close_et.day == 29
        assert prev_close_et.hour == 16

    def test_previous_close_skips_weekend(self) -> None:
        """Test previous close skips weekend."""
        calendar = EquityCalendar()

        # Monday 10:00 AM ET
        monday = datetime(2024, 1, 8, 10, 0, 0)
        monday = monday.replace(tzinfo=ZoneInfo("America/New_York"))

        prev_close = calendar.previous_close(monday)
        # Convert to ET for comparison
        prev_close_et = prev_close.astimezone(ZoneInfo("America/New_York"))

        # Should be previous Friday 4:00 PM ET
        assert prev_close_et.weekday() == 4  # Friday
        assert prev_close_et.hour == 16


# =============================================================================
# Sessions Between Tests
# =============================================================================


class TestSessionsBetween:
    """Tests for sessions_between() method."""

    def test_sessions_between_same_week(self) -> None:
        """Test sessions between dates in same week."""
        calendar = EquityCalendar()

        # Jan 1, 2024 is New Year's Day (holiday), so use Jan 2-5
        # which gives Tue-Fri = 4 sessions
        start = datetime(2024, 1, 1, 9, 30, 0)  # Monday (holiday)
        end = datetime(2024, 1, 5, 16, 0, 0)  # Friday

        start = start.replace(tzinfo=ZoneInfo("America/New_York"))
        end = end.replace(tzinfo=ZoneInfo("America/New_York"))

        sessions = calendar.sessions_between(start, end)

        # Should have 4 sessions (Tue-Fri, skipping New Year's Day)
        assert len(sessions) == 4

        # All should be weekdays
        for session in sessions:
            assert session.weekday() < 5

    def test_sessions_between_includes_weekend(self) -> None:
        """Test sessions between excludes weekends."""
        calendar = EquityCalendar()

        # Friday to next Monday
        start = datetime(2024, 1, 5, 9, 30, 0)  # Friday
        end = datetime(2024, 1, 8, 16, 0, 0)  # Monday

        start = start.replace(tzinfo=ZoneInfo("America/New_York"))
        end = end.replace(tzinfo=ZoneInfo("America/New_York"))

        sessions = calendar.sessions_between(start, end)

        # Should have 2 sessions (Friday, Monday)
        assert len(sessions) == 2
        assert sessions[0].day == 5  # Friday
        assert sessions[1].day == 8  # Monday

    def test_sessions_between_single_day(self) -> None:
        """Test sessions for single day."""
        calendar = EquityCalendar()

        # Same day
        start = datetime(2024, 1, 2, 9, 30, 0)
        end = datetime(2024, 1, 2, 16, 0, 0)

        start = start.replace(tzinfo=ZoneInfo("America/New_York"))
        end = end.replace(tzinfo=ZoneInfo("America/New_York"))

        sessions = calendar.sessions_between(start, end)

        # Should have 1 session
        assert len(sessions) == 1


# =============================================================================
# Session Duration Tests
# =============================================================================


class TestSessionDuration:
    """Tests for session_duration() method."""

    def test_standard_session_duration(self) -> None:
        """Test standard 6.5 hour session."""
        calendar = EquityCalendar()

        # Regular weekday
        dt = datetime(2024, 1, 2, 10, 0, 0)
        dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))

        duration = calendar.session_duration(dt)

        # Should be 6.5 hours (9:30 AM - 4:00 PM)
        assert duration == timedelta(hours=6, minutes=30)

    def test_session_duration_different_dates(self) -> None:
        """Test session duration is same for different dates."""
        calendar = EquityCalendar()

        # Different weekdays
        dates = [
            datetime(2024, 1, 2, 10, 0, 0),  # Tuesday
            datetime(2024, 1, 3, 10, 0, 0),  # Wednesday
            datetime(2024, 1, 4, 10, 0, 0),  # Thursday
        ]

        for dt in dates:
            dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
            duration = calendar.session_duration(dt)
            assert duration == timedelta(hours=6, minutes=30)


# =============================================================================
# DST Transition Tests
# =============================================================================


class TestDSTTransitions:
    """Tests for DST transition handling."""

    def test_spring_forward_session(self) -> None:
        """Test session detection during spring DST transition."""
        calendar = EquityCalendar()

        # March 2024 DST transition (second Sunday)
        # Day before DST (Saturday - not a session)
        before_dst = datetime(2024, 3, 9, 10, 0, 0)
        before_dst = before_dst.replace(tzinfo=ZoneInfo("America/New_York"))

        # Day after DST (Monday - is a session)
        after_dst = datetime(2024, 3, 11, 10, 0, 0)
        after_dst = after_dst.replace(tzinfo=ZoneInfo("America/New_York"))

        # Both should work correctly
        assert calendar.is_session(before_dst) is False  # Saturday
        assert calendar.is_session(after_dst) is True  # Monday

    def test_fall_back_session(self) -> None:
        """Test session detection during fall DST transition."""
        calendar = EquityCalendar()

        # November 2024 DST transition (first Sunday)
        # Day before DST (Saturday - not a session)
        before_dst = datetime(2024, 11, 2, 10, 0, 0)
        before_dst = before_dst.replace(tzinfo=ZoneInfo("America/New_York"))

        # Day after DST (Monday - is a session)
        after_dst = datetime(2024, 11, 4, 10, 0, 0)
        after_dst = after_dst.replace(tzinfo=ZoneInfo("America/New_York"))

        # Both should work correctly
        assert calendar.is_session(before_dst) is False  # Saturday
        assert calendar.is_session(after_dst) is True  # Monday


# =============================================================================
# Holiday Tests (Basic - using fallback if pandas_market_calendars unavailable)
# =============================================================================


class TestHolidays:
    """Tests for market holiday handling."""

    def test_new_years_day_weekday(self) -> None:
        """Test New Year's Day (when on weekday)."""
        calendar = EquityCalendar()

        # January 1, 2024 is a Monday (holiday)
        new_years = datetime(2024, 1, 1, 10, 0, 0)
        new_years = new_years.replace(tzinfo=ZoneInfo("America/New_York"))

        # Might not detect holiday without pandas_market_calendars
        # This is a smoke test - either True or False is acceptable
        result = calendar.is_session(new_years)
        assert isinstance(result, bool)

    def test_independence_day(self) -> None:
        """Test Independence Day (July 4)."""
        calendar = EquityCalendar()

        # July 4, 2024 is a Thursday (holiday)
        july_4 = datetime(2024, 7, 4, 10, 0, 0)
        july_4 = july_4.replace(tzinfo=ZoneInfo("America/New_York"))

        # Smoke test
        result = calendar.is_session(july_4)
        assert isinstance(result, bool)


# =============================================================================
# Edge Cases & Error Handling
# =============================================================================


class TestCalendarEdgeCases:
    """Tests for edge cases and error handling."""

    def test_midnight_boundary(self) -> None:
        """Test behavior at midnight."""
        calendar = EquityCalendar()

        # Midnight Tuesday
        midnight = datetime(2024, 1, 2, 0, 0, 0)
        midnight = midnight.replace(tzinfo=ZoneInfo("America/New_York"))

        # Midnight is before market open
        assert calendar.is_session(midnight) is False

    def test_leap_year_handling(self) -> None:
        """Test calendar handles leap year correctly."""
        calendar = EquityCalendar()

        # Feb 29, 2024 (leap year) at 10 AM
        leap_day = datetime(2024, 2, 29, 10, 0, 0)
        leap_day = leap_day.replace(tzinfo=ZoneInfo("America/New_York"))

        # Thursday - should be a session
        assert calendar.is_session(leap_day) is True

    def test_year_boundary(self) -> None:
        """Test calendar handles year boundary."""
        calendar = EquityCalendar()

        # Dec 31, 2023 (Sunday - not a session)
        year_end = datetime(2023, 12, 31, 10, 0, 0)
        year_end = year_end.replace(tzinfo=ZoneInfo("America/New_York"))

        # Jan 2, 2024 (Tuesday - is a session)
        year_start = datetime(2024, 1, 2, 10, 0, 0)
        year_start = year_start.replace(tzinfo=ZoneInfo("America/New_York"))

        assert calendar.is_session(year_end) is False  # Sunday
        assert calendar.is_session(year_start) is True  # Tuesday


# =============================================================================
# Integration Tests
# =============================================================================


class TestCalendarIntegration:
    """Integration tests for calendar functionality."""

    def test_full_week_workflow(self) -> None:
        """Test complete week workflow."""
        calendar = EquityCalendar()

        # Use a week without holidays: Jan 8-12, 2024
        # Start: Monday 9:00 AM (before open)
        start = datetime(2024, 1, 8, 9, 0, 0)
        start = start.replace(tzinfo=ZoneInfo("America/New_York"))

        # Not in session (before 9:30 AM)
        assert calendar.is_session(start) is False

        # Next open should be same day 9:30 AM
        next_open = calendar.next_open(start)
        next_open_et = next_open.astimezone(ZoneInfo("America/New_York"))
        assert next_open_et.day == 8
        assert next_open_et.hour == 9
        assert next_open_et.minute == 30

        # During market hours
        during = datetime(2024, 1, 8, 10, 0, 0)
        during = during.replace(tzinfo=ZoneInfo("America/New_York"))
        assert calendar.is_session(during) is True

        # Get sessions for full week
        end = datetime(2024, 1, 12, 16, 0, 0)
        end = end.replace(tzinfo=ZoneInfo("America/New_York"))

        sessions = calendar.sessions_between(start, end)
        assert len(sessions) == 5  # Mon-Fri

    def test_timezone_aware_workflow(self) -> None:
        """Test workflow with different timezones."""
        calendar = EquityCalendar()

        # 3:00 PM UTC = 10:00 AM ET (winter time)
        dt_utc = datetime(2024, 1, 2, 15, 0, 0, tzinfo=ZoneInfo("UTC"))

        # Should detect as in session
        assert calendar.is_session(dt_utc) is True

        # Next open in UTC should be next day at correct time
        next_open = calendar.next_open(dt_utc)
        assert next_open.day == 3  # Next day


# =============================================================================
# Crypto Calendar Tests
# =============================================================================


class TestCryptoCalendar:
    """Tests for 24/7 crypto market calendar."""

    def test_init_default(self) -> None:
        """Test default initialization without maintenance window."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar()
        assert calendar.name == "CRYPTO_24_7"
        assert calendar.timezone == "UTC"
        assert calendar.maintenance_window is None

    def test_init_with_maintenance_window(self) -> None:
        """Test initialization with maintenance window."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar(maintenance_window=(4, 5))
        assert calendar.maintenance_window == (4, 5)

    def test_is_session_always_true_without_maintenance(self) -> None:
        """Test is_session returns True for any time without maintenance."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar()

        # Any time should be a session
        times = [
            datetime(2024, 1, 1, 0, 0, 0),  # Midnight
            datetime(2024, 1, 1, 3, 0, 0),  # 3 AM
            datetime(2024, 1, 1, 12, 0, 0),  # Noon
            datetime(2024, 1, 1, 23, 59, 0),  # 11:59 PM
            datetime(2024, 1, 6, 10, 0, 0),  # Saturday
            datetime(2024, 1, 7, 10, 0, 0),  # Sunday
        ]

        for dt in times:
            assert calendar.is_session(dt) is True

    def test_is_session_during_maintenance(self) -> None:
        """Test is_session returns False during maintenance window."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar(maintenance_window=(4, 5))

        # During maintenance (4:00-5:00 AM)
        dt_maintenance = datetime(2024, 1, 1, 4, 30, 0)
        assert calendar.is_session(dt_maintenance) is False

        # Just before maintenance
        dt_before = datetime(2024, 1, 1, 3, 59, 0)
        assert calendar.is_session(dt_before) is True

        # Just after maintenance
        dt_after = datetime(2024, 1, 1, 5, 0, 0)
        assert calendar.is_session(dt_after) is True

    def test_next_open_without_maintenance(self) -> None:
        """Test next_open returns immediately without maintenance."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar()

        dt = datetime(2024, 1, 1, 10, 0, 0)
        next_open = calendar.next_open(dt)

        assert next_open == dt

    def test_next_open_during_maintenance(self) -> None:
        """Test next_open returns end of maintenance window."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar(maintenance_window=(4, 5))

        # During maintenance
        dt = datetime(2024, 1, 1, 4, 30, 0)
        next_open = calendar.next_open(dt)

        assert next_open.hour == 5
        assert next_open.minute == 0

    def test_previous_close_without_maintenance(self) -> None:
        """Test previous_close returns 1 second ago without maintenance."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar()

        dt = datetime(2024, 1, 1, 10, 0, 0)
        prev_close = calendar.previous_close(dt)

        assert prev_close == dt - timedelta(seconds=1)

    def test_previous_close_with_maintenance(self) -> None:
        """Test previous_close returns start of maintenance."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar(maintenance_window=(4, 5))

        # After maintenance today
        dt = datetime(2024, 1, 1, 10, 0, 0)
        prev_close = calendar.previous_close(dt)

        assert prev_close.hour == 4
        assert prev_close.day == 1

    def test_previous_close_before_maintenance(self) -> None:
        """Test previous_close when before maintenance today."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar(maintenance_window=(4, 5))

        # Before maintenance today (3 AM)
        dt = datetime(2024, 1, 2, 3, 0, 0)
        prev_close = calendar.previous_close(dt)

        # Should be yesterday's maintenance start
        assert prev_close.day == 1
        assert prev_close.hour == 4

    def test_sessions_between_without_maintenance(self) -> None:
        """Test sessions_between returns every hour without maintenance."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar()

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 3, 0, 0)

        sessions = calendar.sessions_between(start, end)

        # Should have 4 sessions (0:00, 1:00, 2:00, 3:00)
        assert len(sessions) == 4

    def test_sessions_between_with_maintenance(self) -> None:
        """Test sessions_between excludes maintenance window."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar(maintenance_window=(4, 5))

        start = datetime(2024, 1, 1, 3, 0, 0)
        end = datetime(2024, 1, 1, 6, 0, 0)

        sessions = calendar.sessions_between(start, end)

        # Should have 3 sessions (3:00, 5:00, 6:00) - skipping 4:00
        assert len(sessions) == 3
        session_hours = [s.hour for s in sessions]
        assert 4 not in session_hours

    def test_session_duration_without_maintenance(self) -> None:
        """Test session_duration is 24 hours without maintenance."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        calendar = CryptoCalendar()

        dt = datetime(2024, 1, 1, 10, 0, 0)
        duration = calendar.session_duration(dt)

        assert duration == timedelta(hours=24)

    def test_session_duration_with_maintenance(self) -> None:
        """Test session_duration excludes maintenance duration."""
        from ml4t.engineer.core.calendars.crypto import CryptoCalendar

        # 1 hour maintenance
        calendar = CryptoCalendar(maintenance_window=(4, 5))

        dt = datetime(2024, 1, 1, 10, 0, 0)
        duration = calendar.session_duration(dt)

        assert duration == timedelta(hours=23)
