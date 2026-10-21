"""Actual optional inputs preserve their units, coverage and decision information set."""

import importlib
from datetime import UTC, date, datetime

import polars as pl
import pytest
from ml4t.specs import FeedSpec
from polars.testing import assert_frame_equal

DAYS = [date(2025, 7, d) for d in (7, 8, 9, 10, 11)]


def inputs():
    return importlib.import_module("ml4t.engineer.features.inputs")


def feed(**options):
    return FeedSpec(time_contract_version=2, calendar=None, entity_col="asset", **options)


def market():
    return pl.DataFrame(
        {
            "timestamp": DAYS * 2,
            "asset": [17] * 5 + [23] * 5,
            "close": [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 50.0, 25.0, 12.5, 6.25],
            "volume": [100.0, 0.0, None, 20.0, 10.0, 10.0, 20.0, 40.0, 80.0, 160.0],
            "available_at": [datetime(2025, 7, d.day, 20, tzinfo=UTC) for d in DAYS] * 2,
            "decision_time": [datetime(2025, 7, d.day, 13, tzinfo=UTC) for d in DAYS] * 2,
            "currency": ["USD"] * 10,
            "multiplier": [1.0] * 10,
            "volume_unit": ["shares"] * 10,
        }
    )


def prepare_market(frame, **options):
    return inputs().prepare_market_inputs(
        frame,
        **{
            "feed_spec": feed(),
            "price_basis": "raw",
            "volume_basis": "raw",
            "purpose": "causal",
            "liquidity_window": 2,
            "completed_at_col": "available_at",
            **options,
        },
    )


@pytest.mark.parametrize("lazy", [False, True])
def test_lagged_liquidity_uses_completed_price_volume_products_once_not_current_price(lazy):
    original = market().sort("timestamp", "asset", descending=[False, True])
    result = prepare_market(original.lazy() if lazy else original)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    result = result.collect() if lazy else result
    assert_frame_equal(result.select(original.columns), original)
    a, b = (result.filter(pl.col("asset") == asset) for asset in (17, 23))
    assert a["notional_volume"].to_list() == [None] * 5
    assert a["lagged_mean_notional_volume"].to_list() == [None, None, 500.0, None, None]
    assert b["lagged_mean_notional_volume"].to_list() == [None, None, 1000.0, 1000.0, 1000.0]
    assert a["liquidity_covered"].to_list() == [False, False, True, False, False]
    assert a["liquidity_available_at"][2] == datetime(2025, 7, 8, 20, tzinfo=UTC)
    assert a["liquidity_lag"].to_list() == [1] * 5
    assert_frame_equal(original, market().sort("timestamp", "asset", descending=[False, True]))


def test_retrospective_raw_products_distinguish_real_zero_from_uncovered_volume():
    result = prepare_market(market(), purpose="retrospective")
    assert result.filter(pl.col("asset") == 17)["notional_volume"].to_list() == [
        1000.0,
        0.0,
        None,
        800.0,
        500.0,
    ]
    assert result["preparation_purpose"].unique().to_list() == ["retrospective"]
    no_volume = prepare_market(market().drop("volume"))
    assert no_volume["lagged_mean_notional_volume"].null_count() == no_volume.height
    assert_frame_equal(
        no_volume.select("timestamp", "asset", "close"),
        market().select("timestamp", "asset", "close"),
    )


def test_explicit_contract_multiplier_and_split_compatible_basis_preserve_notional_units():
    contracts = market().with_columns(
        pl.lit(50.0).alias("multiplier"), pl.lit("contracts").alias("volume_unit")
    )
    result = prepare_market(contracts)
    assert result.filter(pl.col("asset") == 23)["lagged_mean_notional_volume"][2] == 50000.0
    split = market().with_columns(
        (pl.col("close") / 2).alias("close"), (pl.col("volume") * 2).alias("volume")
    )
    result = prepare_market(split, price_basis="split_adjusted", volume_basis="split_adjusted")
    assert result.filter(pl.col("asset") == 17)["lagged_mean_notional_volume"][2] == 500.0
    with pytest.raises(ValueError, match="basis"):
        prepare_market(split, price_basis="split_adjusted", volume_basis="raw")


def test_delayed_publication_or_uncompleted_bar_cannot_supply_prior_causal_liquidity():
    delayed = market().with_columns(
        pl.when((pl.col("asset") == 17) & (pl.col("timestamp") == DAYS[1]))
        .then(pl.lit(datetime(2025, 7, 10, tzinfo=UTC)))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    result = prepare_market(delayed)
    assert result.filter(pl.col("asset") == 17)["lagged_mean_notional_volume"][2] is None
    provisional = market().with_columns(pl.lit(datetime(2025, 7, 20, tzinfo=UTC)).alias("finished"))
    assert not prepare_market(provisional, completed_at_col="finished")["liquidity_covered"].any()


def test_native_market_validation_without_derived_features_preserves_the_input_exactly():
    result = prepare_market(market(), liquidity_window=None)
    assert_frame_equal(result, market())
