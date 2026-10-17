from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from ml4t.engineer.core import EquityCalendar


def test_holiday_has_no_fabricated_session_open_or_duration():
    calendar = EquityCalendar("NYSE")
    start = datetime(2025, 7, 4, tzinfo=ZoneInfo("America/New_York"))
    end = start + timedelta(days=1)
    assert calendar.sessions_between(start, end) == []
    assert calendar.session_duration(start) == timedelta(0)


@pytest.mark.parametrize("naive", [True, False])
def test_holiday_open_close_lookup_uses_same_timezone_and_real_early_close(naive):
    calendar = EquityCalendar("NYSE")
    instant = datetime(2025, 7, 4, 12)
    if not naive:
        instant = instant.replace(tzinfo=ZoneInfo("America/New_York"))
    assert calendar.next_open(instant) == datetime(2025, 7, 7, 13, 30, tzinfo=UTC)
    assert calendar.previous_close(instant) == datetime(2025, 7, 3, 17, tzinfo=UTC)


def test_naive_non_us_session_filter_uses_selected_exchange_timezone():
    calendar = EquityCalendar("LSE")
    data = pl.DataFrame(
        {"timestamp": [datetime(2025, 7, 7, 8, 30), datetime(2025, 7, 7, 17)], "row": [0, 1]}
    )
    assert calendar.timezone == "Europe/London"
    assert calendar.filter_sessions(data)["row"].to_list() == [0]


def test_non_us_lunch_break_is_not_a_tradable_session():
    calendar = EquityCalendar("HKEX")
    data = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 7, 7, hour, 30, tzinfo=UTC) for hour in (3, 4, 5)],
            "row": ["before", "lunch", "after"],
        }
    )
    assert calendar.filter_sessions(data)["row"].to_list() == ["before", "after"]
    assert calendar.session_duration(datetime(2025, 7, 7)) == timedelta(hours=5, minutes=30)


def test_naive_session_enumeration_matches_aware_equivalents():
    calendar = EquityCalendar("NYSE")
    start, end = datetime(2025, 7, 3), datetime(2025, 7, 8)
    expected = [datetime(2025, 7, 3, 13, 30, tzinfo=UTC), datetime(2025, 7, 7, 13, 30, tzinfo=UTC)]
    assert calendar.sessions_between(start, end) == expected
    assert (
        calendar.sessions_between(
            start.replace(tzinfo=ZoneInfo("America/New_York")),
            end.replace(tzinfo=ZoneInfo("America/New_York")),
        )
        == expected
    )


def test_missing_schedule_provider_rejects_instead_of_trading_on_holiday(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    with pytest.raises(ImportError, match="pandas.market.calendars"):
        EquityCalendar("NYSE")


def test_canonical_calendar_selector_preserves_aliases_and_rejects_conflicts():
    assert EquityCalendar(calendar="LSE").timezone == "Europe/London"
    assert EquityCalendar(exchange="XNYS", calendar="NYSE").name == "XNYS"
    with pytest.raises(ValueError, match="conflict"):
        EquityCalendar(exchange="NYSE", calendar="LSE")
    with pytest.raises(ValueError, match="named calendar"):
        EquityCalendar(calendar=None)


def test_regular_early_close_is_not_rounded_to_standard_session_duration():
    calendar = EquityCalendar("NYSE")
    assert calendar.session_duration(datetime(2025, 7, 3)) == timedelta(hours=3, minutes=30)
    assert not calendar.is_session(datetime(2025, 7, 4, 15, tzinfo=UTC))


def test_schedule_lookup_failure_never_uses_weekday_fallback(monkeypatch):
    calendar = EquityCalendar("NYSE")
    import pandas as pd

    monkeypatch.setattr(calendar._calendar, "schedule", lambda **_kwargs: pd.DataFrame())
    with pytest.raises(ValueError, match="scheduled"):
        calendar.next_open(datetime(2025, 7, 7, tzinfo=UTC))
    with pytest.raises(ValueError, match="scheduled"):
        calendar.previous_close(datetime(2025, 7, 7, tzinfo=UTC))


@pytest.mark.parametrize("timestamp", [datetime(2025, 3, 9, 2, 30), datetime(2025, 11, 2, 1, 30)])
def test_invalid_naive_dst_timestamp_does_not_get_silently_assigned_an_offset(timestamp):
    calendar = EquityCalendar("NYSE")
    with pytest.raises(ValueError, match="ambiguous or nonexistent"):
        calendar.is_session(timestamp)


@pytest.mark.parametrize("zone", ["UTC", "Asia/Tokyo"])
def test_session_annotations_use_authoritative_session_label_not_input_civil_date(zone):
    calendar = EquityCalendar("CME_Equity")
    data = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 7, 6, 23, 30, tzinfo=UTC),
                datetime(2025, 7, 4, 23, 30, tzinfo=UTC),
            ],
            "close": [100.0, 101.0],
        }
    ).with_columns(pl.col("timestamp").dt.convert_time_zone(zone))
    result = calendar.add_session_info(data)
    assert result["is_session"].to_list() == [True, False]
    assert result["session_date"].to_list() == [date(2025, 7, 7), None]
    assert result.select(data.columns).equals(data)


def test_session_duration_uses_session_label_not_overnight_open_date():
    calendar = EquityCalendar("CME_Equity")
    assert calendar.session_duration(datetime(2025, 7, 7)) == timedelta(hours=23)
    assert calendar.session_duration(datetime(2025, 11, 28)) == timedelta(hours=19)
