from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from ml4t.engineer.core.calendars import CryptoCalendar


@pytest.mark.parametrize("maintenance", [None, (4, 5), (23, 1)])
def test_crypto_sessions_are_daily_opens_not_hourly_observations(maintenance):
    calendar = CryptoCalendar(maintenance_window=maintenance)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 3, 12, tzinfo=UTC)
    hour = 1 if maintenance == (23, 1) else 0
    assert calendar.sessions_between(start, end) == [
        datetime(2025, 1, day, hour, tzinfo=UTC) for day in (1, 2, 3)
    ]
    assert calendar.sessions_between(start + timedelta(hours=12), start + timedelta(hours=18)) == []


@pytest.mark.parametrize("zone", ["UTC", "Asia/Tokyo", "America/New_York"])
def test_maintenance_membership_is_invariant_to_timestamp_representation(zone):
    calendar = CryptoCalendar(maintenance_window=(4, 5))
    closed = datetime(2025, 1, 1, 4, 30, tzinfo=UTC)
    assert not calendar.is_session(closed.astimezone(ZoneInfo(zone)))
    assert not calendar.is_session(closed.replace(tzinfo=None))
    assert calendar.is_session((closed + timedelta(hours=1)).astimezone(ZoneInfo(zone)))


def test_overnight_maintenance_excludes_both_dates_and_does_not_inflate_duration():
    calendar = CryptoCalendar(maintenance_window=(23, 1))
    for stamp in [datetime(2025, 1, 1, 23, 30), datetime(2025, 1, 2, 0, 30)]:
        assert not calendar.is_session(stamp.replace(tzinfo=UTC))
    assert calendar.is_session(datetime(2025, 1, 2, 1, tzinfo=UTC))
    assert not calendar.is_session(datetime(2025, 1, 2, 23, tzinfo=UTC))
    assert calendar.session_duration(datetime(2025, 1, 2)) == timedelta(hours=22)


@pytest.mark.parametrize("maintenance", [(0, 1), (23, 24), (23, 0)])
def test_midnight_maintenance_endpoints_keep_positive_daily_duration(maintenance):
    calendar = CryptoCalendar(maintenance_window=maintenance)
    day = datetime(2025, 1, 2, tzinfo=UTC)
    assert calendar.session_duration(day) == timedelta(hours=23)
    assert calendar.sessions_between(day, day + timedelta(hours=23, minutes=30)) == [
        day + timedelta(hours=1 if maintenance == (0, 1) else 0)
    ]
    assert calendar.next_open(day + timedelta(minutes=30)) == (
        day + timedelta(hours=1) if maintenance == (0, 1) else day + timedelta(days=1)
    )


@pytest.mark.parametrize("maintenance", [None, (4, 5), (23, 1)])
def test_crypto_boundary_helpers_name_sessions_not_immediate_or_maintenance_reopens(maintenance):
    calendar = CryptoCalendar(maintenance_window=maintenance)
    stamp = datetime(2025, 1, 2, 4, 30, tzinfo=UTC)
    opened = datetime(2025, 1, 3, 1 if maintenance == (23, 1) else 0, tzinfo=UTC)
    closed = (
        datetime(2025, 1, 1, 23, tzinfo=UTC)
        if maintenance == (23, 1)
        else stamp.replace(hour=0, minute=0)
    )
    assert calendar.next_open(stamp) == opened
    assert calendar.previous_close(stamp) == closed
    assert calendar.next_open(opened) == opened + timedelta(days=1)
    assert calendar.previous_close(closed) == closed - timedelta(days=1)


@pytest.mark.parametrize("zone", [None, "UTC", "Asia/Tokyo"])
def test_crypto_annotations_use_utc_session_labels_without_changing_input(zone):
    stamps = [datetime(2025, 1, 1, 23, 30, tzinfo=UTC), datetime(2025, 1, 2, 4, 30, tzinfo=UTC)]
    frame = pl.DataFrame({"when": stamps, "value": [1, 2]})
    if zone is None:
        frame = frame.with_columns(pl.col("when").dt.replace_time_zone(None))
    else:
        frame = frame.with_columns(pl.col("when").dt.convert_time_zone(zone))
    original = frame.clone()
    calendar = CryptoCalendar(maintenance_window=(4, 5))
    result = calendar.add_session_info(frame, timestamp_col="when")
    assert result["session_date"].to_list() == [date(2025, 1, 1), None]
    assert result["session_time"].to_list() == [time(23, 30), time(4, 30)]
    assert result["is_session"].to_list() == [True, False]
    assert calendar.filter_sessions(frame, timestamp_col="when")["value"].to_list() == [1]
    assert result.select(frame.columns).equals(original)
    assert frame.equals(original)


@pytest.mark.parametrize(
    "maintenance", [(-1, 5), (4, 25), (24, 1), (4, 4), (0, 24), (True, 5), (4.5, 5), (4,), "4-5"]
)
def test_invalid_maintenance_does_not_construct_a_misleading_schedule(maintenance):
    with pytest.raises(ValueError, match="maintenance_window"):
        CryptoCalendar(maintenance_window=maintenance)


@pytest.mark.parametrize("name", ["24/7", "CRYPTO_24_7"])
def test_canonical_calendar_keyword_selects_the_continuous_schedule(name):
    calendar = CryptoCalendar(calendar=name)
    assert calendar.name == "CRYPTO_24_7"
    assert calendar.session_duration(datetime(2025, 1, 1)) == timedelta(days=1)


@pytest.mark.parametrize("name", [None, "NYSE", "", False])
def test_crypto_calendar_object_never_treats_disable_or_an_exchange_as_continuous(name):
    with pytest.raises(ValueError, match="calendar"):
        CryptoCalendar(calendar=name)


def test_crypto_session_interval_rejects_reversed_bounds():
    calendar = CryptoCalendar()
    with pytest.raises(ValueError, match="start.*end"):
        calendar.sessions_between(datetime(2025, 1, 2), datetime(2025, 1, 1))


def test_crypto_annotations_do_not_interpret_session_label_dates_as_instants():
    frame = pl.DataFrame({"timestamp": [date(2025, 1, 1)]})
    with pytest.raises(TypeError, match="datetime instants"):
        CryptoCalendar().add_session_info(frame)


def test_crypto_calendar_does_not_substitute_a_schedule_when_provider_is_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    with pytest.raises(ImportError, match="pandas.market.calendars"):
        CryptoCalendar()
