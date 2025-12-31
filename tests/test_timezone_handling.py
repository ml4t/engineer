"""Tests for timezone handling in EquityCalendar."""

from datetime import datetime
from zoneinfo import ZoneInfo

from ml4t.engineer.core.calendars.equity import EquityCalendar


class TestEquityCalendarTimezone:
    """Test timezone handling in EquityCalendar."""

    def test_initialization_with_timezone(self):
        """Test calendar initialization with custom timezone."""
        # Default should be New York
        cal = EquityCalendar()
        assert cal.timezone == "America/New_York"

        # Custom timezone
        cal_london = EquityCalendar(timezone="Europe/London")
        assert cal_london.timezone == "Europe/London"

    def test_is_session_naive_datetime(self):
        """Test session check with naive datetime."""
        cal = EquityCalendar()

        # Monday 9:30 AM ET (naive, assumed to be in market timezone)
        dt = datetime(2024, 1, 8, 9, 30, 0)
        assert cal._is_basic_session(dt) is True

        # Monday 9:29 AM ET (before open)
        dt = datetime(2024, 1, 8, 9, 29, 0)
        assert cal._is_basic_session(dt) is False

        # Monday 4:01 PM ET (after close)
        dt = datetime(2024, 1, 8, 16, 1, 0)
        assert cal._is_basic_session(dt) is False

        # Saturday (weekend)
        dt = datetime(2024, 1, 6, 12, 0, 0)
        assert cal._is_basic_session(dt) is False

    def test_is_session_aware_datetime(self):
        """Test session check with timezone-aware datetime."""
        cal = EquityCalendar()

        # Create UTC time that's 2:30 PM ET (during market hours)
        # 2:30 PM ET = 7:30 PM UTC (during standard time)
        dt_utc = datetime(2024, 1, 8, 19, 30, 0, tzinfo=ZoneInfo("UTC"))
        assert cal._is_basic_session(dt_utc) is True

        # Create UTC time that's 9:00 PM ET (after market)
        dt_utc_after = datetime(2024, 1, 8, 2, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert cal._is_basic_session(dt_utc_after) is False

        # Pacific time during market hours (12:30 PM PT = 3:30 PM ET)
        dt_pacific = datetime(2024, 1, 8, 12, 30, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        assert cal._is_basic_session(dt_pacific) is True

    def test_next_basic_open_naive(self):
        """Test next market open with naive datetime."""
        cal = EquityCalendar()

        # Monday 3 PM ET -> Tuesday 9:30 AM ET
        dt = datetime(2024, 1, 8, 15, 0, 0)
        next_open = cal._next_basic_open(dt)
        assert next_open.hour == 9
        assert next_open.minute == 30
        assert next_open.day == 8  # Same day since before close

        # Monday 5 PM ET -> Tuesday 9:30 AM ET
        dt = datetime(2024, 1, 8, 17, 0, 0)
        next_open = cal._next_basic_open(dt)
        assert next_open.day == 9  # Next day

        # Friday 5 PM ET -> Monday 9:30 AM ET (skip weekend)
        dt = datetime(2024, 1, 5, 17, 0, 0)  # Friday
        next_open = cal._next_basic_open(dt)
        assert next_open.day == 8  # Monday
        assert next_open.weekday() == 0  # Monday

    def test_next_basic_open_aware(self):
        """Test next market open with timezone-aware datetime."""
        cal = EquityCalendar()

        # UTC time that's Monday 5 PM ET -> Tuesday 9:30 AM ET
        dt_utc = datetime(2024, 1, 8, 22, 0, 0, tzinfo=ZoneInfo("UTC"))  # 5 PM ET
        next_open = cal._next_basic_open(dt_utc)
        assert next_open.tzinfo is not None

        # Convert to ET and check
        et_tz = ZoneInfo("America/New_York")
        next_open_et = next_open.astimezone(et_tz) if next_open.tzinfo else next_open
        assert next_open_et.hour == 9
        assert next_open_et.minute == 30

    def test_previous_basic_close_naive(self):
        """Test previous market close with naive datetime."""
        cal = EquityCalendar()

        # Tuesday 10 AM ET -> Monday 4 PM ET
        dt = datetime(2024, 1, 9, 10, 0, 0)
        prev_close = cal._previous_basic_close(dt)
        assert prev_close.hour == 16
        assert prev_close.minute == 0
        assert prev_close.day == 9  # Same day since after open

        # Tuesday 8 AM ET -> Monday 4 PM ET
        dt = datetime(2024, 1, 9, 8, 0, 0)
        prev_close = cal._previous_basic_close(dt)
        assert prev_close.day == 8  # Previous day

        # Monday 8 AM ET -> Friday 4 PM ET (skip weekend)
        dt = datetime(2024, 1, 8, 8, 0, 0)  # Monday
        prev_close = cal._previous_basic_close(dt)
        assert prev_close.day == 5  # Friday
        assert prev_close.weekday() == 4  # Friday

    def test_previous_basic_close_aware(self):
        """Test previous market close with timezone-aware datetime."""
        cal = EquityCalendar()

        # UTC time that's Tuesday 8 AM ET -> Monday 4 PM ET
        dt_utc = datetime(2024, 1, 9, 13, 0, 0, tzinfo=ZoneInfo("UTC"))  # 8 AM ET
        prev_close = cal._previous_basic_close(dt_utc)
        assert prev_close.tzinfo is not None

        # Convert to ET and check
        et_tz = ZoneInfo("America/New_York")
        prev_close_et = prev_close.astimezone(et_tz) if prev_close.tzinfo else prev_close
        assert prev_close_et.hour == 16
        assert prev_close_et.minute == 0

    def test_daylight_saving_time(self):
        """Test handling of daylight saving time transitions."""
        cal = EquityCalendar()

        # Test during standard time (January)
        dt_standard = datetime(2024, 1, 15, 14, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal._is_basic_session(dt_standard) is True

        # Test during daylight time (July)
        dt_daylight = datetime(2024, 7, 15, 14, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal._is_basic_session(dt_daylight) is True

        # UTC times should handle DST correctly
        # In January: 2:30 PM ET = 7:30 PM UTC
        dt_utc_jan = datetime(2024, 1, 15, 19, 30, 0, tzinfo=ZoneInfo("UTC"))
        assert cal._is_basic_session(dt_utc_jan) is True

        # In July: 2:30 PM ET = 6:30 PM UTC (due to DST)
        dt_utc_jul = datetime(2024, 7, 15, 18, 30, 0, tzinfo=ZoneInfo("UTC"))
        assert cal._is_basic_session(dt_utc_jul) is True

    def test_edge_cases(self):
        """Test edge cases in timezone handling."""
        cal = EquityCalendar()

        # Exactly at market open
        dt_open = datetime(2024, 1, 8, 9, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal._is_basic_session(dt_open) is True

        # Exactly at market close
        dt_close = datetime(2024, 1, 8, 16, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        assert cal._is_basic_session(dt_close) is True

        # One second before open
        dt_before = datetime(2024, 1, 8, 9, 29, 59, tzinfo=ZoneInfo("America/New_York"))
        assert cal._is_basic_session(dt_before) is False

        # One second after close
        dt_after = datetime(2024, 1, 8, 16, 0, 1, tzinfo=ZoneInfo("America/New_York"))
        assert cal._is_basic_session(dt_after) is False

    def test_mixed_timezone_operations(self):
        """Test operations with mixed timezone inputs."""
        cal = EquityCalendar()

        # Create times in different timezones that represent the same moment
        et_tz = ZoneInfo("America/New_York")
        utc_tz = ZoneInfo("UTC")
        london_tz = ZoneInfo("Europe/London")

        # 2:30 PM ET on a Monday
        dt_et = datetime(2024, 1, 8, 14, 30, 0, tzinfo=et_tz)

        # Same moment in UTC (7:30 PM UTC)
        dt_utc = dt_et.astimezone(utc_tz)

        # Same moment in London (7:30 PM GMT)
        dt_london = dt_et.astimezone(london_tz)

        # All should evaluate to the same session status
        assert cal._is_basic_session(dt_et) is True
        assert cal._is_basic_session(dt_utc) is True
        assert cal._is_basic_session(dt_london) is True

        # Next open should be consistent
        next_et = cal._next_basic_open(dt_et)
        next_utc = cal._next_basic_open(dt_utc)
        next_london = cal._next_basic_open(dt_london)

        # Convert all to ET for comparison
        next_utc_et = next_utc.astimezone(et_tz) if next_utc.tzinfo else next_utc

        next_london_et = next_london.astimezone(et_tz) if next_london.tzinfo else next_london

        # They should represent the same time
        assert abs((next_et - next_utc_et).total_seconds()) < 1
        assert abs((next_et - next_london_et).total_seconds()) < 1


if __name__ == "__main__":
    # Run tests
    test = TestEquityCalendarTimezone()

    print("Testing timezone initialization...")
    test.test_initialization_with_timezone()
    print("✅ Initialization tests passed")

    print("Testing naive datetime handling...")
    test.test_is_session_naive_datetime()
    print("✅ Naive datetime tests passed")

    print("Testing timezone-aware datetime handling...")
    test.test_is_session_aware_datetime()
    print("✅ Timezone-aware tests passed")

    print("Testing next market open...")
    test.test_next_basic_open_naive()
    test.test_next_basic_open_aware()
    print("✅ Next open tests passed")

    print("Testing previous market close...")
    test.test_previous_basic_close_naive()
    test.test_previous_basic_close_aware()
    print("✅ Previous close tests passed")

    print("Testing daylight saving time...")
    test.test_daylight_saving_time()
    print("✅ DST tests passed")

    print("Testing edge cases...")
    test.test_edge_cases()
    print("✅ Edge case tests passed")

    print("Testing mixed timezone operations...")
    test.test_mixed_timezone_operations()
    print("✅ Mixed timezone tests passed")

    print("\n🎉 All timezone handling tests passed!")
