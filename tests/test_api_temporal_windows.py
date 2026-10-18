from datetime import UTC, date, datetime, timedelta
from statistics import mean, pstdev, stdev

import polars as pl
import pytest
import yaml
from ml4t.specs import FeedSpec

from ml4t.engineer import compute_features


def _prices():
    from ml4t.engineer.core.bar_grid import _expected_bar_labels
    from ml4t.engineer.labeling.calendar import PandasMarketCalendar

    timestamps = list(
        _expected_bar_labels(
            PandasMarketCalendar("NYSE"),
            datetime(2025, 7, 2, tzinfo=UTC),
            datetime(2025, 7, 9, tzinfo=UTC),
            timedelta(hours=1),
            "right",
            True,
        )
    )
    values = [100.0 + index for index in range(len(timestamps))]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "A",
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": 1.0,
        }
    )


def _feed():
    return FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timezone="UTC",
        entity_col="symbol",
        timestamp_semantics="bar_close",
        data_frequency="1h",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="right",
        partial_bar_policy="include",
    )


def _run(prices, period, unit, lazy=False):
    checked = compute_features(prices, [], feed_spec=_feed())
    returns = checked.with_columns(pl.col("close").pct_change().over("symbol").alias("returns"))
    features = [
        {"name": "sma", "params": {"period": period}},
        {"name": "stddev", "params": {"period": period}},
        {"name": "realized_volatility", "params": {"period": period, "annualize": False}},
    ]
    data = returns.lazy() if lazy else returns
    result = compute_features(data, features, feed_spec=_feed(), window_unit=unit)
    assert isinstance(result, type(data))
    return result.collect() if lazy else result


@pytest.mark.parametrize("lazy", [False, True])
def test_session_feature_windows_use_original_causal_bars_and_actual_shortened_sessions(lazy):
    prices = _prices()
    result = _run(prices.reverse(), 2, "session", lazy)
    assert result.select(prices.columns).equals(prices)
    assert result["sma"][:7].to_list() == [None] * 7
    # First July3 bar includes seven July2 bars and itself, not two rows or two closes.
    assert result["sma"][7] == pytest.approx(mean(range(100, 108)))
    assert result["stddev"][7] == pytest.approx(pstdev(range(100, 108)))
    # July3 has only four bars; July4 is closed. First July7 bar sees July3 + itself.
    assert result["sma"][11] == pytest.approx(mean(range(107, 112)))
    assert result["stddev"][11] == pytest.approx(pstdev(range(107, 112)))
    assert result["realized_volatility"][11] == pytest.approx(
        stdev([i / (i - 1) - 1 for i in range(107, 112)])
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_elapsed_feature_windows_use_open_left_utc_duration_not_rows_or_sessions(lazy):
    prices = _prices()
    result = _run(prices, "2h", "elapsed", lazy)
    assert result["sma"][:2].to_list() == [None, None]
    assert result["sma"][2] == pytest.approx(101.5)
    # First July3 bar has no bar from July2 inside the elapsed duration.
    assert result["sma"][7] == pytest.approx(107.0)
    assert result["realized_volatility"][7] is None
    # The final half-hour July3 bar puts three observations inside two physical hours.
    assert result["sma"][10] == pytest.approx(mean([108, 109, 110]))
    assert result["stddev"][10] == pytest.approx(pstdev([108, 109, 110]))
    assert result["realized_volatility"][10] == pytest.approx(
        stdev([108 / 107 - 1, 109 / 108 - 1, 110 / 109 - 1])
    )


@pytest.mark.parametrize("unit,period", [("session", 2), ("elapsed", "2h")])
def test_temporal_feature_panels_are_isolated_and_future_prices_do_not_enter_earlier_windows(
    unit, period
):
    first = _prices()
    second = first.with_columns(
        *[(pl.col(name) * 2).alias(name) for name in ("open", "high", "low", "close")],
        pl.lit("B").alias("symbol"),
    )
    result = _run(pl.concat([second, first]).reverse(), period, unit, True)
    for frame in (first, second):
        symbol = frame["symbol"][0]
        assert result.filter(pl.col("symbol") == symbol).equals(_run(frame, period, unit))
    changed = first.with_columns(
        *[
            pl.when(pl.col("timestamp").dt.date() >= date(2025, 7, 7))
            .then(pl.col(name) * 4)
            .otherwise(pl.col(name))
            .alias(name)
            for name in ("open", "high", "low", "close")
        ]
    )
    before = _run(first, period, unit).head(11)
    after = _run(changed, period, unit).head(11)
    assert before.equals(after)


@pytest.mark.parametrize("unit,period", [("session", 2), ("elapsed", "2h")])
def test_lazy_filter_projection_and_slice_do_not_remove_temporal_warmup_history(unit, period):
    prices = _prices()
    features = [{"name": "sma", "params": {"period": period}}]
    full = compute_features(prices, features, feed_spec=_feed(), window_unit=unit)
    predicate = pl.col("timestamp").dt.date() == date(2025, 7, 7)
    lazy = compute_features(prices.lazy(), features, feed_spec=_feed(), window_unit=unit)
    assert (
        lazy.filter(predicate).select("sma").collect().equals(full.filter(predicate).select("sma"))
    )
    assert lazy.slice(7, 4).collect().equals(full.slice(7, 4))


@pytest.mark.parametrize("unit,period", [("session", 2), ("elapsed", "2h")])
def test_serialized_existing_feature_requests_keep_explicit_temporal_units_and_parameters(
    tmp_path, unit, period
):
    features = [{"name": "sma", "params": {"period": period}, "output": "average"}]
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump({"features": features}))
    expected = compute_features(_prices(), features, feed_spec=_feed(), window_unit=unit)
    actual = compute_features(_prices(), path, feed_spec=_feed().to_dict(), window_unit=unit)
    assert actual.equals(expected)


@pytest.mark.parametrize("unit,period", [("session", 2), ("elapsed", "2h")])
def test_missing_complete_session_is_rejected_before_temporal_feature_returns_are_interpreted(
    unit, period
):
    prices = _prices().filter(pl.col("timestamp").dt.date() != date(2025, 7, 3))
    before = prices.clone()
    with pytest.raises(ValueError, match="missing scheduled bar.*2025-07-03"):
        compute_features(
            prices,
            [{"name": "sma", "params": {"period": period}}],
            feed_spec=_feed(),
            window_unit=unit,
        )
    assert prices.equals(before)


@pytest.mark.parametrize("unit,period", [("session", 2), ("elapsed", "2h")])
@pytest.mark.parametrize("feature", ["ema", "rsi"])
def test_temporal_units_do_not_silently_redefine_recursive_indicators(feature, unit, period):
    with pytest.raises(ValueError, match=f"{feature}.*temporal"):
        compute_features(
            _prices(),
            [{"name": feature, "params": {"period": period}}],
            feed_spec=_feed(),
            window_unit=unit,
        )


@pytest.mark.parametrize(
    "unit,period",
    [("session", 0), ("session", 1.5), ("session", True), ("elapsed", 2), ("elapsed", "0h")],
)
def test_temporal_feature_periods_reject_unit_mismatches_instead_of_converting_rows(unit, period):
    with pytest.raises(ValueError, match="period"):
        compute_features(
            _prices(),
            [{"name": "sma", "params": {"period": period}}],
            feed_spec=_feed(),
            window_unit=unit,
        )


@pytest.mark.parametrize("unit,period", [("session", 2), ("elapsed", "2h")])
def test_calendar_none_remains_observation_mode_even_when_temporal_windows_are_requested(
    unit, period
):
    with pytest.raises(ValueError, match="observation"):
        compute_features(
            _prices(),
            [{"name": "sma", "params": {"period": period}}],
            calendar=None,
            window_unit=unit,
        )


@pytest.mark.parametrize("naive", [False, True])
def test_elapsed_window_counts_physical_hours_through_spring_dst_in_declared_timezone(naive):
    timestamps = pl.Series(
        "timestamp", [datetime(2025, 3, 9, hour, tzinfo=UTC) for hour in range(5, 11)]
    ).dt.convert_time_zone("America/New_York")
    if naive:
        timestamps = timestamps.dt.replace_time_zone(None)
    values = [100.0 + i for i in range(6)]
    prices = pl.DataFrame(
        {
            "timestamp": timestamps,
            "close": values,
            "open": values,
            "high": values,
            "low": values,
            "volume": 1.0,
        }
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="24/7",
        timezone="America/New_York",
        timestamp_semantics="event_time",
        data_frequency="1h",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label="left",
        partial_bar_policy="include",
    )
    result = compute_features(
        prices, [{"name": "sma", "params": {"period": "2h"}}], feed_spec=feed, window_unit="elapsed"
    )
    assert result["sma"].to_list() == [None, None, 101.5, 102.5, 103.5, 104.5]


@pytest.mark.parametrize("lazy", [False, True])
def test_daily_feature_session_windows_count_sessions_without_inventing_intraday_instants(lazy):
    values = [100.0, 101.0, 104.0, 105.0]
    prices = pl.DataFrame(
        {
            "timestamp": [date(2025, 7, day) for day in (2, 3, 7, 8)],
            "close": values,
            "open": values,
            "high": values,
            "low": values,
            "volume": 1.0,
        }
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timestamp_semantics="session_label",
        data_frequency="1d",
        bar_type="time",
    )
    result = compute_features(
        prices.lazy() if lazy else prices,
        [{"name": "sma", "params": {"period": 2}}],
        feed_spec=feed,
        window_unit="session",
    )
    result = result.collect() if lazy else result
    assert result["sma"].to_list() == [None, 100.5, 102.5, 104.5]
    with pytest.raises(ValueError, match="session_label.*elapsed"):
        compute_features(
            prices,
            [{"name": "sma", "params": {"period": "2h"}}],
            feed_spec=feed,
            window_unit="elapsed",
        )


@pytest.mark.parametrize("lazy", [False, True])
def test_temporal_windows_preserve_ddof_scale_annualization_and_input_parameter_values(lazy):
    from copy import deepcopy
    from math import sqrt

    prices = _prices().with_columns(pl.col("close").pct_change().over("symbol").alias("returns"))
    features = [
        {"name": "stddev", "params": {"period": 2, "nbdev": 2.0, "ddof": 1}},
        {"name": "realized_volatility", "params": {"period": 2, "trading_periods": 365}},
        {"name": "sma", "params": {"period": 1}, "output": "current_session"},
        {"name": "sma", "params": {"period": 2}, "output": "two_sessions"},
    ]
    original = deepcopy(features)
    result = compute_features(
        prices.lazy() if lazy else prices, features, feed_spec=_feed(), window_unit="session"
    )
    result = result.collect() if lazy else result
    assert result["stddev"][11] == pytest.approx(2 * stdev(range(107, 112)))
    assert result["realized_volatility"][11] == pytest.approx(
        stdev([i / (i - 1) - 1 for i in range(107, 112)]) * sqrt(365)
    )
    assert result["current_session"][11] == 111.0
    assert result["two_sessions"][11] == 109.0
    assert features == original


@pytest.mark.parametrize("unit,period", [("session", 10**100), ("elapsed", "365d")])
def test_temporal_window_exceeding_available_coverage_stays_null_without_integer_overflow(
    unit, period
):
    result = compute_features(
        _prices(),
        [{"name": "sma", "params": {"period": period}}],
        feed_spec=_feed(),
        window_unit=unit,
    )
    assert result["sma"].is_null().all()


@pytest.mark.parametrize("lazy", [False, True])
def test_empty_temporal_feature_input_preserves_typed_columns_and_no_observations(lazy):
    prices = _prices().head(0)
    result = compute_features(
        prices.lazy() if lazy else prices,
        [{"name": "sma", "params": {"period": 1}}],
        feed_spec=_feed(),
        window_unit="session",
    )
    result = result.collect() if lazy else result
    assert result.height == 0
    assert result.schema["sma"] == pl.Float64
    assert result.select(prices.columns).equals(prices)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("lazy", [False, True])
def test_feature_session_windows_count_short_sessions_even_when_partial_policy_excludes_all_bars(
    side, lazy
):
    from ml4t.engineer.core.bar_grid import _expected_bar_labels
    from ml4t.engineer.labeling.calendar import PandasMarketCalendar

    timestamps = list(
        _expected_bar_labels(
            PandasMarketCalendar("NYSE"),
            datetime(2025, 7, 2, tzinfo=UTC),
            datetime(2025, 7, 9, tzinfo=UTC),
            timedelta(hours=5),
            side,
            False,
        )
    )
    values = [100.0, 101.0, 102.0]
    prices = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": 1.0,
        }
    )
    feed = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timezone="UTC",
        data_frequency="5h",
        timestamp_semantics="bar_close" if side == "right" else "event_time",
        bar_type="time",
        bar_grid_anchor="session_open",
        bar_label=side,
        partial_bar_policy="exclude",
    )
    result = compute_features(
        prices.lazy() if lazy else prices,
        [{"name": "sma", "params": {"period": 2}}],
        feed_spec=feed,
        window_unit="session",
    )
    result = result.collect() if lazy else result
    # July3 counts, despite having no5h bar. July7 must not include the July2 price.
    assert result["sma"].to_list() == [None, 101.0, 101.5]
