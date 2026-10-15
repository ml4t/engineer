from datetime import UTC, datetime, timedelta
from math import log
from statistics import stdev
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from ml4t.engineer import compute_features
from ml4t.engineer.core.bar_grid import validate_bar_grid


def frame(times, symbol="TEST"):
    close = [100.0 + i for i in range(len(times))]
    return pl.DataFrame(
        {
            "timestamp": times,
            "symbol": symbol,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


def validate(data, calendar="24/7", **overrides):
    options = {
        "calendar": calendar,
        "frequency": timedelta(hours=1),
        "timezone": "UTC",
        "timestamp_col": "timestamp",
        "group_cols": ["symbol"],
        "anchor": "session_open",
        "label_side": "right",
        "include_partial": True,
    }
    options.update(overrides)
    validate_bar_grid(data, **options)


@pytest.mark.parametrize("lazy", [False, True])
def test_validated_hourly_prices_reach_public_returns_and_feature_pipeline_unchanged(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    prices = frame([start + timedelta(hours=h) for h in range(8)])
    data = prices.lazy() if lazy else prices
    validate(data)
    returns = data.with_columns(pl.col("close").log().diff().over("symbol").alias("returns"))
    result = compute_features(
        returns,
        [{"name": "realized_volatility", "params": {"period": 3, "annualize": False}}],
        calendar=None,
    )
    result = result.collect() if isinstance(result, pl.LazyFrame) else result
    assert result.height == prices.height
    assert result.select(prices.columns).equals(prices)
    expected = stdev([log(101 / 100), log(102 / 101), log(103 / 102)])
    assert result["realized_volatility"][3] == pytest.approx(expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_hourly_price_outage_fails_before_return_derivation_and_preserves_input(lazy):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    prices = frame([start + timedelta(hours=h) for h in [0, 1, 2, 50, 51, 52]])
    original = prices.clone()
    with pytest.raises(ValueError, match="missing.*2025-07-11T03:00"):
        validate(prices.lazy() if lazy else prices)
        pytest.fail("Incomplete prices must not reach return derivation")
    assert prices.equals(original)


def nyse_times():
    zone = ZoneInfo("America/New_York")
    return (
        [datetime(2025, 7, 3, h, 30, tzinfo=zone) for h in [10, 11, 12]]
        + [datetime(2025, 7, 3, 13, tzinfo=zone)]
        + [datetime(2025, 7, 7, h, 30, tzinfo=zone) for h in range(10, 16)]
        + [datetime(2025, 7, 7, 16, tzinfo=zone)]
    )


def test_real_nyse_holiday_weekend_and_early_close_preserve_all_prices():
    prices = frame(nyse_times())
    original = prices.clone()
    validate(prices, "NYSE")
    assert prices.equals(original)


def test_real_nyse_missing_bar_is_not_inferred_to_be_a_holiday():
    times = nyse_times()
    del times[5]
    with pytest.raises(ValueError, match="missing.*2025-07-07T15:30"):
        validate(frame(times), "NYSE")


def test_one_missing_asset_is_not_hidden_by_another_assets_complete_grid():
    start = datetime(2025, 7, 11, tzinfo=UTC)
    times = [start + timedelta(hours=h) for h in range(8)]
    complete = frame(times, "COMPLETE")
    del times[3]
    prices = pl.concat([complete, frame(times, "MISSING")])
    with pytest.raises(ValueError, match="missing.*MISSING.*03:00"):
        validate(prices)


@pytest.mark.parametrize("kind", ["duplicate", "off_grid", "null"])
def test_invalid_bar_timestamps_fail_without_changing_rows(kind):
    start = datetime(2025, 7, 11, tzinfo=UTC)
    times = [start + timedelta(hours=h) for h in range(8)]
    if kind == "duplicate":
        times[3] = times[2]
    elif kind == "off_grid":
        times = [value + timedelta(minutes=1) for value in times]
    else:
        times[3] = None
    prices = frame(times)
    original = prices.clone()
    with pytest.raises(
        ValueError,
        match={"duplicate": "duplicate", "off_grid": "missing|off-grid", "null": "null"}[kind],
    ):
        validate(prices)
    assert prices.equals(original)


def test_explicit_left_labels_and_no_partial_bar_use_session_open_not_first_observation():
    zone = ZoneInfo("America/New_York")
    prices = frame([datetime(2025, 7, 3, h, 30, tzinfo=zone) for h in [9, 10, 11]])
    validate(prices, "NYSE", label_side="left", include_partial=False)
    with pytest.raises(ValueError, match="off-grid"):
        validate(prices, "NYSE", label_side="right", include_partial=False)


@pytest.mark.parametrize("frequency", [None, timedelta(0), timedelta(days=1)])
def test_undeclared_or_unsupported_duration_does_not_trigger_frequency_inference(frequency):
    prices = frame([datetime(2025, 7, 11, tzinfo=UTC)])
    with pytest.raises(ValueError, match="declared grid frequency"):
        validate(prices, frequency=frequency)


def test_same_instants_in_other_timezone_use_identical_scheduled_bars():
    zone = ZoneInfo("America/New_York")
    times = [datetime(2025, 7, 7, h, 30, tzinfo=zone) for h in range(10, 14)]
    validate(frame(times), "NYSE")
    validate(frame([value.astimezone(UTC) for value in times]), "NYSE")


def test_missing_whole_nyse_session_is_not_hidden_by_observed_session_dates():
    zone = ZoneInfo("America/New_York")
    times = []
    for day in [14, 16]:
        times.extend(datetime(2025, 7, day, h, 30, tzinfo=zone) for h in range(10, 16))
        times.append(datetime(2025, 7, day, 16, tzinfo=zone))
    with pytest.raises(ValueError, match="missing.*2025-07-15"):
        validate(frame(times), "NYSE")


def test_hong_kong_lunch_break_does_not_look_like_a_missing_hourly_bar():
    zone = ZoneInfo("Asia/Hong_Kong")
    prices = frame(
        [datetime(2025, 7, 14, h, 30, tzinfo=zone) for h in [10, 11]]
        + [datetime(2025, 7, 14, 12, tzinfo=zone)]
        + [datetime(2025, 7, 14, h, tzinfo=zone) for h in [14, 15, 16]]
    )
    validate(prices, "HKEX")


def test_real_nyse_dst_weekend_changes_utc_hours_not_local_bar_anchoring():
    zone = ZoneInfo("America/New_York")
    times = []
    for day in [7, 10]:
        times.extend(datetime(2025, 3, day, h, 30, tzinfo=zone) for h in range(10, 16))
        times.append(datetime(2025, 3, day, 16, tzinfo=zone))
    validate(frame(times), "NYSE")


def test_unknown_calendar_cannot_fall_back_to_us_weekdays():
    with pytest.raises(RuntimeError):
        validate(frame([datetime(2025, 7, 11, tzinfo=UTC)]), "NOT_A_REAL_EXCHANGE_20261015")


def test_missing_provider_cannot_fall_back_to_us_weekdays(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def no_provider(name, *args, **kwargs):
        if name == "pandas_market_calendars":
            raise ImportError("Absent optional schedule provider")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_provider)
    with pytest.raises(ImportError, match="pandas_market_calendars required"):
        validate(frame([datetime(2025, 7, 11, tzinfo=UTC)]))
