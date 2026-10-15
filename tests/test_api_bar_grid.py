from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from ml4t.specs import FeedSpec, TimestampSemantics

from ml4t.engineer import compute_features


def prices(times):
    close = [100.0 + i * 3 for i in range(len(times))]
    return pl.DataFrame(
        {
            "timestamp": times,
            "symbol": "TEST",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


def returns(frame):
    return frame.with_columns(pl.col("close").log().diff().over("symbol").alias("returns"))


REQUEST = [{"name": "realized_volatility", "params": {"period": 3, "annualize": False}}]


def feed(calendar):
    return FeedSpec(
        calendar=calendar,
        entity_col="symbol",
        data_frequency="1h",
        timezone="UTC",
        bar_type="time",
        timestamp_semantics=TimestampSemantics.BAR_CLOSE,
    )


def run(frame, lazy, **kwargs):
    result = compute_features(frame.lazy() if lazy else frame, REQUEST, **kwargs)
    return result.collect() if isinstance(result, pl.LazyFrame) else result


@pytest.mark.parametrize("lazy", [False, True])
def test_undeclared_grid_does_not_interpret_outage_return_as_one_bar(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    frame = returns(prices([start + timedelta(hours=h) for h in [0, 1, 2, 50, 51, 52]]))
    original = frame.clone()
    with pytest.raises(ValueError, match="declared.*grid"):
        run(frame, lazy)
    assert frame.equals(original)


@pytest.mark.parametrize("lazy", [False, True])
def test_continuous_hourly_outage_is_rejected_without_mutating_prices_or_returns(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    frame = returns(prices([start + timedelta(hours=h) for h in [0, 1, 2, 50, 51, 52]]))
    original = frame.clone()
    with pytest.raises(ValueError, match="missing.*2025-07-11.*03:00"):
        run(frame, lazy, calendar="24/7", feed_spec=feed("24/7"))
    assert frame.equals(original)


@pytest.mark.parametrize("lazy", [False, True])
def test_complete_continuous_hourly_prices_keep_row_count_and_expected_volatility(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    frame = returns(prices([start + timedelta(hours=h) for h in range(8)]))
    result = run(frame, lazy, calendar="24/7", feed_spec=feed("24/7"))
    expected = frame.select(pl.col("returns").rolling_std(3)).to_series()
    assert result.height == frame.height
    assert result["realized_volatility"].equals(expected.rename("realized_volatility"))


def nyse_times():
    zone = ZoneInfo("America/New_York")
    july_3 = [datetime(2025, 7, 3, h, 30, tzinfo=zone) for h in [10, 11, 12]]
    july_3.append(datetime(2025, 7, 3, 13, tzinfo=zone))
    july_7 = [datetime(2025, 7, 7, h, 30, tzinfo=zone) for h in range(10, 16)]
    july_7.append(datetime(2025, 7, 7, 16, tzinfo=zone))
    return july_3 + july_7


@pytest.mark.parametrize("lazy", [False, True])
def test_nyse_holiday_weekend_and_early_close_are_not_missing_hourly_bars(lazy):
    frame = returns(prices(nyse_times()))
    result = run(frame, lazy, calendar="NYSE", feed_spec=feed("NYSE"))
    assert result.height == frame.height
    assert result["timestamp"].equals(frame["timestamp"])


@pytest.mark.parametrize("lazy", [False, True])
def test_nyse_actual_missing_scheduled_bar_is_not_reclassified_as_a_closure(lazy):
    times = nyse_times()
    del times[5]  # Delete the price observation before deriving the returns.
    frame = returns(prices(times))
    original = frame.clone()
    with pytest.raises(ValueError, match="missing.*2025-07-07.*15:30"):
        run(frame, lazy, calendar="NYSE", feed_spec=feed("NYSE"))
    assert frame.equals(original)


@pytest.mark.parametrize("lazy", [False, True])
def test_explicit_none_is_row_invariant_on_regular_and_irregular_timestamps(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    results = []
    for hours in [range(8), [0, 1, 3, 73, 74, 200, 201, 300]]:
        frame = returns(prices([start + timedelta(hours=h) for h in hours]))
        result = run(frame, lazy, calendar=None, feed_spec=feed("NYSE"))
        assert result.height == frame.height
        results.append(result["realized_volatility"])
    assert results[0].equals(results[1])
