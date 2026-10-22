from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from math import log
from statistics import stdev
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from ml4t.specs import UNSET_CALENDAR, FeedSpec, read_spec_payload, write_spec_payload

from ml4t.engineer import compute_features
from ml4t.engineer.core.exceptions import DataSchemaError

REQUEST = [
    {"name": "sma", "params": {"period": 2}},
    {"name": "realized_volatility", "params": {"period": 2, "annualize": False}},
]


def prices(days, symbol="TEST"):
    close = [100.0 + 3 * index for index in range(len(days))]
    return pl.DataFrame(
        {
            "timestamp": days,
            "symbol": symbol,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


def contract(calendar=UNSET_CALENDAR):
    return FeedSpec(
        time_contract_version=2,
        calendar=calendar,
        entity_col="symbol",
        timestamp_semantics="session_label",
        data_frequency="1d",
        bar_type="time",
    )


def materialize(data):
    return data.collect() if isinstance(data, pl.LazyFrame) else data


def derive_returns(data):
    return data.with_columns(pl.col("close").log().diff().over("symbol").alias("returns"))


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("representation", ["date", "naive", "UTC", "Asia/Tokyo"])
def test_daily_price_returns_features_preserve_session_identifiers_across_closures(
    lazy, representation
):
    days = [date(2025, 7, day) for day in (3, 7, 8, 9)]
    if representation != "date":
        zone = None if representation == "naive" else ZoneInfo(representation)
        days = [datetime.combine(day, time(0, 15), tzinfo=zone) for day in days]
    frame = prices(days)
    original = frame.clone()
    data = frame.lazy() if lazy else frame
    checked = compute_features(data.reverse(), [], feed_spec=contract())
    assert isinstance(checked, type(data))
    assert materialize(checked).equals(original)
    result = materialize(compute_features(derive_returns(checked), REQUEST, feed_spec=contract()))
    assert result.select(frame.columns).equals(frame)
    assert result["sma"].to_list() == [None, 101.5, 104.5, 107.5]
    assert result["realized_volatility"][2] == pytest.approx(
        stdev([log(103 / 100), log(106 / 103)])
    )
    assert frame.equals(original)


@pytest.mark.parametrize("lazy", [False, True])
def test_removed_daily_price_cannot_be_interpreted_as_a_one_bar_return(lazy):
    complete = prices([date(2025, 7, day) for day in range(14, 19)])
    incomplete = complete.filter(pl.col("timestamp") != date(2025, 7, 16))
    original = incomplete.clone()
    for frame, requests in [(incomplete, []), (derive_returns(incomplete), REQUEST)]:
        data = frame.lazy() if lazy else frame
        with pytest.raises(ValueError, match="missing scheduled session.*2025-07-16"):
            compute_features(data, requests, feed_spec=contract("NYSE"))
    assert incomplete.equals(original)


def test_complete_asset_does_not_hide_another_assets_missing_daily_price():
    complete = prices([date(2025, 7, day) for day in range(14, 19)], "COMPLETE")
    missing = complete.with_columns(pl.lit("MISSING").alias("symbol")).filter(
        pl.col("timestamp") != date(2025, 7, 16)
    )
    with pytest.raises(ValueError, match="missing scheduled session.*MISSING.*2025-07-16"):
        compute_features(
            derive_returns(pl.concat([complete, missing])), REQUEST, feed_spec=contract()
        )


@pytest.mark.parametrize("frequency", ["1d", "1D", "24h", timedelta(days=1)])
def test_complete_continuous_daily_grid_includes_weekends_without_equity_annualization(frequency):
    frame = derive_returns(prices([date(2025, 7, day) for day in range(10, 15)]))
    result = compute_features(
        frame, REQUEST, feed_spec=replace(contract("24/7"), data_frequency=frequency)
    )
    assert result.height == 5
    assert result["realized_volatility"][2] == pytest.approx(
        stdev([log(103 / 100), log(106 / 103)])
    )
    with pytest.raises(ValueError, match="off-schedule"):
        compute_features(frame, [], feed_spec=contract("NYSE"))


def test_reindexing_derived_daily_returns_does_not_hide_a_cross_session_return():
    complete = prices([date(2025, 7, day) for day in range(14, 19)])
    missing = complete.filter(pl.col("timestamp") != date(2025, 7, 16))
    unsafe = (
        complete.select("timestamp")
        .join(derive_returns(missing), on="timestamp", how="left")
        .with_columns(pl.col("symbol").fill_null("TEST"))
    )
    original = unsafe.clone()
    with pytest.raises(ValueError, match="Return adjacent to a missing price"):
        compute_features(unsafe, REQUEST, feed_spec=contract())
    assert unsafe.equals(original)
    safe_prices = (
        complete.select("timestamp")
        .join(missing, on="timestamp", how="left")
        .with_columns(pl.col("symbol").fill_null("TEST"))
    )
    safe = compute_features(derive_returns(safe_prices), REQUEST, feed_spec=contract())
    assert safe.height == 5
    assert safe["returns"][3] is None
    assert safe["realized_volatility"][3] is None


@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("lazy", [False, True])
def test_empty_and_single_daily_grids_preserve_declared_schema_and_shape(count, lazy):
    frame = prices([date(2025, 7, 3)]).head(count)
    data = frame.lazy() if lazy else frame
    assert materialize(compute_features(data, [], feed_spec=contract())).equals(frame)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_serialized_daily_calendar_selection_does_not_turn_continuous_weekends_into_closures(
    tmp_path, suffix
):
    frame = prices([date(2025, 7, 11), date(2025, 7, 14)])
    path = tmp_path / f"feed{suffix}"
    write_spec_payload(contract("24/7").to_dict(), path)
    metadata = read_spec_payload(path)
    with pytest.raises(ValueError, match="missing scheduled session.*2025-07-12"):
        compute_features(frame, [], feed_spec=metadata)
    assert compute_features(frame, [], feed_spec=metadata, calendar="NYSE").equals(frame)
    assert compute_features(frame, [], feed_spec=metadata, calendar=None).equals(frame)
    write_spec_payload(contract().to_dict(), path)
    payload = read_spec_payload(path)
    assert "calendar" not in payload
    assert compute_features(frame, [], feed_spec=payload).equals(frame)


@pytest.mark.parametrize("kind", ["duplicate", "closed", "null"])
def test_invalid_daily_session_labels_fail_without_changing_prices(kind):
    days = [date(2025, 7, day) for day in (3, 7, 8)]
    if kind == "null":
        days[1] = None
    else:
        days.insert(1, days[0] if kind == "duplicate" else date(2025, 7, 4))
    frame = prices(days)
    original = frame.clone()
    with pytest.raises((ValueError, DataSchemaError), match="duplicate|off-schedule|null"):
        compute_features(frame, [], feed_spec=contract())
    assert frame.equals(original)


def test_distinct_instants_with_the_same_session_identifier_are_duplicate_daily_bars():
    frame = prices([datetime(2025, 7, 3, hour, tzinfo=UTC) for hour in (10, 11)])
    with pytest.raises(ValueError, match="duplicate session label"):
        compute_features(frame, [], feed_spec=contract())


def test_overnight_exchange_uses_session_labels_not_the_open_instant_date():
    frame = prices([date(2025, 7, day) for day in (7, 8)])
    assert compute_features(frame, [], feed_spec=contract("CME_Equity")).equals(frame)
    sunday = prices([date(2025, 7, 6)])
    with pytest.raises(ValueError, match="off-schedule session label"):
        compute_features(sunday, [], feed_spec=contract("CME_Equity"))


@pytest.mark.parametrize("frequency", [None, "1h", "2d", timedelta(0)])
def test_daily_labels_require_a_declared_single_session_frequency(frequency):
    frame = prices([date(2025, 7, 3)])
    with pytest.raises(ValueError, match="frequency|declared bar grid"):
        compute_features(frame, [], feed_spec=replace(contract(), data_frequency=frequency))


def test_daily_frequency_does_not_reclassify_event_instants_as_session_labels():
    frame = prices([datetime(2025, 7, 3, tzinfo=UTC)])
    instant = replace(
        contract(),
        timestamp_semantics="event_time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
        timezone="UTC",
    )
    with pytest.raises(ValueError, match="frequency.*below one day"):
        compute_features(frame, [], feed_spec=instant)


def test_disabled_calendar_keeps_daily_observation_order_without_loading_a_schedule(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    frame = derive_returns(prices([date(2025, 7, day) for day in (8, 4, 20, 7)]))
    result = compute_features(frame, REQUEST, feed_spec=contract(None))
    assert result.select(frame.columns).equals(frame)
    assert result["sma"].to_list() == [None, 101.5, 104.5, 107.5]
