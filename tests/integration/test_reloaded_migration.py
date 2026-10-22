"""Actual native recipe workflows retain independent numerical and boundary expectations."""

import importlib.util
import math
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal


def recipes():
    path = Path(__file__).resolve().parents[2] / "examples/reloaded_workflows.py"
    spec = importlib.util.spec_from_file_location("native_migration_recipes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wide_integer_price_recipe_keeps_ticks_assets_and_leading_rows():
    import pandas as pd

    stamps = pd.date_range("2025-07-07", periods=3, tz="UTC")
    wide = pd.DataFrame(
        {
            17: pd.array([2**53 + 2, 2**53 + 1, 2**53], dtype="UInt64"),
            23: pd.array([2**63 + 2, 2**63 + 1, 2**63], dtype="UInt64"),
        },
        index=stamps,
    )
    result = recipes().precise_wide_returns(wide)
    assert result.height == 6
    assert result["close"].dtype == pl.UInt64
    for asset, base in ((17, 2**53), (23, 2**63)):
        selected = result.filter(pl.col("asset") == asset)
        assert selected["close"].to_list() == [base + 2, base + 1, base]
        actual = selected["return"].to_list()
        assert actual[0] is None
        for value, expected in zip(actual[1:], (-1 / (base + 2), -1 / (base + 1)), strict=True):
            # Division produces Float64; retain ticks and allow only its final rounding unit.
            assert abs(value - expected) <= math.ulp(expected)


def test_wide_price_missing_cells_never_compress_into_a_false_one_bar_return():
    import pandas as pd

    wide = pd.DataFrame(
        {17: pd.array([100.0, None, 120.0, 150.0], dtype="Float64")},
        index=pd.date_range("2025-07-07", periods=4, tz="UTC"),
    )
    result = recipes().precise_wide_returns(wide)
    assert result["return"].to_list() == [None, None, None, 0.25]
    assert result["close"].to_list() == [100.0, None, 120.0, 150.0]


@pytest.mark.parametrize("calendar", [None, "NYSE"])
def test_split_rename_momentum_volatility_liquidity_sector_screen_preserves_financial_units(
    calendar,
):
    if calendar:
        pytest.importorskip("pandas_market_calendars")
    recipe = recipes()
    prices, decisions, universe, feed = recipe.captured_snapshot()
    result = recipe.snapshot_screen(prices, decisions, universe, feed_spec=feed, calendar=calendar)
    assert result["asset"].to_list() == [17]
    assert result["momentum"][0] == pytest.approx(0.01, rel=1e-14, abs=0)
    assert result["volatility"][0] == pytest.approx((3 / 98 + 0.02) / 2 * math.sqrt(252))
    assert result["lagged_mean_notional_volume"].to_list() == [100343.0]
    assert result["sector"].to_list() == ["bank"]
    assert result["screen"].to_list() == [True]
    assert result["effective_session"].to_list() == [date(2025, 7, 8)]
    assert result["liquidity_available_at"][0] == datetime(2025, 7, 3, 17, tzinfo=UTC)


def test_optional_missing_volume_does_not_remove_independent_price_features():
    recipe = recipes()
    prices, decisions, universe, feed = recipe.captured_snapshot()
    original = recipe.snapshot_screen(prices, decisions, universe, feed_spec=feed, calendar=None)
    missing = recipe.snapshot_screen(
        prices.drop("adj_volume"), decisions, universe, feed_spec=feed, calendar=None
    )
    assert_frame_equal(
        missing.select("momentum", "volatility", "eligible"),
        original.select("momentum", "volatility", "eligible"),
    )
    assert missing["lagged_mean_notional_volume"].to_list() == [None]
    assert missing["screen"].to_list() == [False]


def test_custom_multioutput_missing_input_and_future_universe_facts_never_qualify():
    recipe = recipes()
    prices, decisions, universe, feed = recipe.captured_snapshot()
    result = recipe.custom_bands(prices, decisions, universe, feed_spec=feed, calendar=None)
    assert result["bands"].struct.field("mid")[0] == pytest.approx(299 / 3)
    assert result["bands"].struct.field("width")[0] == pytest.approx(math.sqrt(14) / 3)
    delayed = prices.with_columns(
        pl.when(pl.col("timestamp") == date(2025, 7, 3))
        .then(pl.lit(datetime(2025, 7, 20, tzinfo=UTC)))
        .otherwise("available_at")
        .alias("available_at")
    )
    unavailable = recipe.custom_bands(delayed, decisions, universe, feed_spec=feed, calendar=None)
    assert unavailable["eligible"].to_list() == [False]
    assert unavailable["bands"].struct.field("mid").to_list() == [None]
    future = universe.with_columns(pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("available_at"))
    unknown_sector = recipe.snapshot_screen(
        prices, decisions, future, feed_spec=feed, calendar=None
    )
    assert unknown_sector["screen"].to_list() == [False]
    assert unknown_sector["sector"].to_list() == [None]


def test_cadence_carries_sampled_rows_including_unknown_samples_without_future_values():
    days = [date(2025, 7, d) for d in (7, 8, 9, 14)]
    frame = pl.DataFrame(
        {
            "effective_session": days,
            "asset": [17] * 4,
            "rsi": [10.0, 20.0, 30.0, None],
            "eligible": [True, True, True, False],
            "available_at": [datetime(2025, 7, d.day, 12, tzinfo=UTC) for d in days],
            "decision_time": [datetime(2025, 7, d.day, 13, tzinfo=UTC) for d in days],
        }
    )
    result = recipes().sampled_values(frame.reverse(), [days[0], days[-1]])
    assert_frame_equal(result.select(frame.columns), frame.reverse())
    assert result.sort("effective_session")["sampled_rsi"].to_list() == [10.0, 10.0, 10.0, None]
    with pytest.raises(ValueError, match="unique"):
        recipes().sampled_values(pl.concat([frame, frame.head(1)]), [days[0]])
    future = frame.with_columns(pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("available_at"))
    with pytest.raises(ValueError, match="not known"):
        recipes().sampled_values(future, [days[0]])


def test_actual_price_preparation_keeps_unavailable_sample_and_holds_only_known_features():
    result = recipes().cadence_example()
    assert result["effective_session"].to_list() == [date(2025, 7, day) for day in (7, 8, 9, 14)]
    assert result["sampled_rsi"].to_list()[:3] == pytest.approx(
        [10.0, 10.0, 10.0], rel=1e-12, abs=0
    )
    assert result["sampled_rsi"][-1] is None
    assert result["eligible"].to_list()[-1] is False
    assert result["sample_session"].to_list() == [date(2025, 7, 7)] * 3 + [date(2025, 7, 14)]
