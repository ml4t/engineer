"""
Tests for microstructure features - Decorator-based system.

Tests cover:
- Basic functionality for all 6 features
- Edge cases (insufficient data, NaN handling)
- Parameter validation
- Output structure
- Registry verification
- Academic formula correctness
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.microstructure import (
    amihud_illiquidity,
    effective_tick_rule,
    kyle_lambda,
    order_flow_imbalance,
    realized_spread,
    roll_spread_estimator,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_microstructure_data():
    """Sample OHLCV + returns data for microstructure testing."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 30),
                interval="1d",
                eager=True,
            ),
            "open": [
                100.0,
                101.5,
                102.2,
                101.8,
                103.0,
                102.5,
                104.0,
                103.3,
                105.1,
                104.5,
                106.2,
                105.8,
                107.0,
                106.5,
                108.2,
                107.5,
                109.0,
                108.7,
                110.1,
                109.5,
                111.0,
                110.5,
                112.2,
                111.8,
                113.0,
                112.5,
                114.1,
                113.5,
                115.0,
                114.5,
            ],
            "high": [
                102.0,
                103.0,
                104.0,
                103.5,
                105.0,
                104.0,
                106.0,
                105.0,
                107.0,
                106.0,
                108.0,
                107.0,
                109.0,
                108.0,
                110.0,
                109.0,
                111.0,
                110.0,
                112.0,
                111.0,
                113.0,
                112.0,
                114.0,
                113.0,
                115.0,
                114.0,
                116.0,
                115.0,
                117.0,
                116.0,
            ],
            "low": [
                98.0,
                99.5,
                100.0,
                99.5,
                101.0,
                100.5,
                102.0,
                101.5,
                103.0,
                102.5,
                104.0,
                103.5,
                105.0,
                104.5,
                106.0,
                105.5,
                107.0,
                106.5,
                108.0,
                107.5,
                109.0,
                108.5,
                110.0,
                109.5,
                111.0,
                110.5,
                112.0,
                111.5,
                113.0,
                112.5,
            ],
            "close": [
                101.0,
                102.0,
                101.5,
                103.0,
                102.5,
                104.0,
                103.5,
                105.0,
                104.5,
                106.0,
                105.5,
                107.0,
                106.5,
                108.0,
                107.5,
                109.0,
                108.5,
                110.0,
                109.5,
                111.0,
                110.5,
                112.0,
                111.5,
                113.0,
                112.5,
                114.0,
                113.5,
                115.0,
                114.5,
                116.0,
            ],
            "volume": [
                1000.0,
                1100.0,
                900.0,
                1200.0,
                800.0,
                1300.0,
                700.0,
                1400.0,
                600.0,
                1500.0,
                500.0,
                1600.0,
                400.0,
                1700.0,
                300.0,
                1800.0,
                200.0,
                1900.0,
                100.0,
                2000.0,
                2100.0,
                2200.0,
                2300.0,
                2400.0,
                2500.0,
                2600.0,
                2700.0,
                2800.0,
                2900.0,
                3000.0,
            ],
        },
    ).with_columns(
        # Add returns column for features that need it
        (pl.col("close").pct_change()).alias("returns")
    )


@pytest.fixture
def small_microstructure_data():
    """Small dataset for edge case testing."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 5),
                interval="1d",
                eager=True,
            ),
            "open": [100.0, 101.0, 102.0, 101.5, 103.0],
            "high": [102.0, 103.0, 104.0, 103.0, 105.0],
            "low": [98.0, 99.0, 100.0, 99.5, 101.0],
            "close": [101.0, 102.0, 101.5, 103.0, 102.5],
            "volume": [1000.0, 1100.0, 900.0, 1200.0, 800.0],
        },
    ).with_columns((pl.col("close").pct_change()).alias("returns"))


# =============================================================================
# Kyle Lambda Tests
# =============================================================================


def test_kyle_lambda_basic_functionality(sample_microstructure_data):
    """Test Kyle's Lambda basic calculation."""
    result = sample_microstructure_data.with_columns(
        kyle_lambda("returns", "volume", period=20).alias("kyle_lambda_20")
    )

    assert "kyle_lambda_20" in result.columns
    assert len(result) == len(sample_microstructure_data)
    assert result["kyle_lambda_20"].dtype == pl.Float64

    # Should have NaNs in warmup period
    lambda_values = result["kyle_lambda_20"].to_numpy()
    assert np.isnan(lambda_values[0])  # First value should be NaN


def test_kyle_lambda_with_custom_period(sample_microstructure_data):
    """Test Kyle's Lambda with custom period."""
    result = sample_microstructure_data.with_columns(
        kyle_lambda("returns", "volume", period=10).alias("kyle_lambda_10")
    )

    assert "kyle_lambda_10" in result.columns
    lambda_values = result["kyle_lambda_10"].drop_nulls().drop_nans()
    assert len(lambda_values) > 0  # Should have some valid values


def test_kyle_lambda_positive_values(sample_microstructure_data):
    """Test that Kyle's Lambda produces non-negative values."""
    result = sample_microstructure_data.with_columns(
        kyle_lambda("returns", "volume", period=5).alias("kyle_lambda_5")
    )
    lambda_values = result["kyle_lambda_5"].drop_nulls().drop_nans().to_numpy()

    # Kyle lambda should be non-negative (price impact measure)
    assert np.all(lambda_values >= 0)


# =============================================================================
# Amihud Illiquidity Tests
# =============================================================================


def test_amihud_basic_functionality(sample_microstructure_data):
    """Test Amihud illiquidity basic calculation."""
    result = sample_microstructure_data.with_columns(
        amihud_illiquidity("returns", "volume", "close", period=20).alias("amihud_20")
    )

    assert "amihud_20" in result.columns
    assert len(result) == len(sample_microstructure_data)
    assert result["amihud_20"].dtype == pl.Float64


def test_amihud_with_custom_period(sample_microstructure_data):
    """Test Amihud illiquidity with custom period."""
    result = sample_microstructure_data.with_columns(
        amihud_illiquidity("returns", "volume", "close", period=10).alias("amihud_10")
    )

    assert "amihud_10" in result.columns
    amihud_values = result["amihud_10"].drop_nulls().drop_nans()
    assert len(amihud_values) > 0


def test_amihud_positive_values(sample_microstructure_data):
    """Test that Amihud illiquidity produces non-negative values."""
    result = sample_microstructure_data.with_columns(
        amihud_illiquidity("returns", "volume", "close", period=5).alias("amihud_5")
    )
    amihud_values = result["amihud_5"].drop_nulls().drop_nans().to_numpy()

    # Amihud should be non-negative (illiquidity measure)
    assert np.all(amihud_values >= 0)


# =============================================================================
# Roll Spread Tests
# =============================================================================


def test_roll_spread_basic_functionality(sample_microstructure_data):
    """Test Roll spread estimator basic calculation."""
    result = sample_microstructure_data.with_columns(
        roll_spread_estimator("close", period=20).alias("roll_spread_20")
    )

    assert "roll_spread_20" in result.columns
    assert len(result) == len(sample_microstructure_data)
    assert result["roll_spread_20"].dtype == pl.Float64


def test_roll_spread_with_custom_period(sample_microstructure_data):
    """Test Roll spread with custom period."""
    result = sample_microstructure_data.with_columns(
        roll_spread_estimator("close", period=10).alias("roll_spread_10")
    )

    assert "roll_spread_10" in result.columns
    spread_values = result["roll_spread_10"].drop_nulls().drop_nans()
    assert len(spread_values) > 0


def test_roll_spread_non_negative(sample_microstructure_data):
    """Test that Roll spread produces non-negative values."""
    result = sample_microstructure_data.with_columns(
        roll_spread_estimator("close", period=5).alias("roll_spread_5")
    )
    spread_values = result["roll_spread_5"].drop_nulls().drop_nans().to_numpy()

    # Spread should be non-negative
    assert np.all(spread_values >= 0)


# =============================================================================
# Tick Rule Tests
# =============================================================================


def test_tick_rule_basic_functionality(sample_microstructure_data):
    """Test effective tick rule basic calculation."""
    result = sample_microstructure_data.with_columns(
        effective_tick_rule("close").alias("tick_rule")
    )

    assert "tick_rule" in result.columns
    assert len(result) == len(sample_microstructure_data)
    # Tick rule returns Int32 (1, -1, 0)
    assert result["tick_rule"].dtype in [pl.Int32, pl.Int64, pl.Float64]


def test_tick_rule_values(sample_microstructure_data):
    """Test that tick rule produces valid classification values."""
    result = sample_microstructure_data.with_columns(
        effective_tick_rule("close").alias("tick_rule")
    )
    tick_values = result["tick_rule"].to_numpy()

    # Should only contain -1, 0, or 1 (after dropping NaNs)
    valid_values = tick_values[~np.isnan(tick_values)]
    assert np.all(np.isin(valid_values, [-1, 0, 1]))


def test_tick_rule_logic(small_microstructure_data):
    """Test tick rule classification logic."""
    result = small_microstructure_data.with_columns(effective_tick_rule("close").alias("tick_rule"))
    tick_values = result["tick_rule"].to_numpy()

    # [0]: NaN (no previous price)
    # [1]: close=102 > 101 -> BUY (1)
    # [2]: close=101.5 < 102 -> SELL (-1)
    # [3]: close=103 > 101.5 -> BUY (1)
    # [4]: close=102.5 < 103 -> SELL (-1)

    assert np.isnan(tick_values[0]) or tick_values[0] == 0
    assert tick_values[1] == 1  # Price up
    assert tick_values[2] == -1  # Price down
    assert tick_values[3] == 1  # Price up
    assert tick_values[4] == -1  # Price down


# =============================================================================
# Realized Spread Tests
# =============================================================================


def test_realized_spread_basic_functionality(sample_microstructure_data):
    """Test realized spread basic calculation."""
    result = sample_microstructure_data.with_columns(
        realized_spread("high", "low", "close", period=20).alias("realized_spread_20")
    )

    assert "realized_spread_20" in result.columns
    assert len(result) == len(sample_microstructure_data)
    assert result["realized_spread_20"].dtype == pl.Float64


def test_realized_spread_with_custom_period(sample_microstructure_data):
    """Test realized spread with custom period."""
    result = sample_microstructure_data.with_columns(
        realized_spread("high", "low", "close", period=10).alias("realized_spread_10")
    )

    assert "realized_spread_10" in result.columns
    spread_values = result["realized_spread_10"].drop_nulls().drop_nans()
    assert len(spread_values) > 0


def test_realized_spread_bounded(sample_microstructure_data):
    """Test that realized spread is bounded (0 to 1 roughly)."""
    result = sample_microstructure_data.with_columns(
        realized_spread("high", "low", "close", period=5).alias("realized_spread_5")
    )
    spread_values = result["realized_spread_5"].drop_nulls().drop_nans().to_numpy()

    # Spread should be non-negative
    assert np.all(spread_values >= 0)
    # Should generally be less than 1 (100% of midpoint)
    assert np.percentile(spread_values, 90) < 1.0


# =============================================================================
# Order Flow Imbalance Tests
# =============================================================================


def test_order_flow_imbalance_basic_functionality(sample_microstructure_data):
    """Test order flow imbalance basic calculation."""
    result = sample_microstructure_data.with_columns(
        order_flow_imbalance("volume", "close").alias("order_flow_imbalance")
    )

    assert "order_flow_imbalance" in result.columns
    assert len(result) == len(sample_microstructure_data)
    assert result["order_flow_imbalance"].dtype == pl.Float64


def test_order_flow_imbalance_bounded(sample_microstructure_data):
    """Test that order flow imbalance is bounded between -1 and 1."""
    result = sample_microstructure_data.with_columns(
        order_flow_imbalance("volume", "close").alias("order_flow_imbalance")
    )
    imbalance_values = result["order_flow_imbalance"].drop_nulls().drop_nans().to_numpy()

    # Should be bounded between -1 (all selling) and 1 (all buying)
    assert np.all(imbalance_values >= -1)
    assert np.all(imbalance_values <= 1)


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_insufficient_data():
    """Test all features handle insufficient data gracefully."""
    tiny_df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 2),
                interval="1d",
                eager=True,
            ),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [98.0, 99.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        },
    ).with_columns((pl.col("close").pct_change()).alias("returns"))

    # All features should handle insufficient data gracefully
    result_kyle = tiny_df.with_columns(
        kyle_lambda("returns", "volume", period=20).alias("kyle_lambda_20")
    )
    result_amihud = tiny_df.with_columns(
        amihud_illiquidity("returns", "volume", "close", period=20).alias("amihud_20")
    )
    result_roll = tiny_df.with_columns(
        roll_spread_estimator("close", period=20).alias("roll_spread_20")
    )
    result_tick = tiny_df.with_columns(effective_tick_rule("close").alias("tick_rule"))
    result_realized = tiny_df.with_columns(
        realized_spread("high", "low", "close", period=20).alias("realized_spread_20")
    )
    result_ofi = tiny_df.with_columns(
        order_flow_imbalance("volume", "close").alias("order_flow_imbalance")
    )

    # Should all complete without error
    assert len(result_kyle) == 2
    assert len(result_amihud) == 2
    assert len(result_roll) == 2
    assert len(result_tick) == 2
    assert len(result_realized) == 2
    assert len(result_ofi) == 2


def test_nan_data():
    """Test features handle NaN values in data."""
    data_with_nan = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 10),
                interval="1d",
                eager=True,
            ),
            "open": [100.0, np.nan, 102.0, 101.0, 103.0, np.nan, 105.0, 104.0, 106.0, 105.0],
            "high": [102.0, 103.0, 104.0, np.nan, 105.0, 106.0, 107.0, 108.0, np.nan, 110.0],
            "low": [98.0, 99.0, 100.0, 99.0, np.nan, 102.0, 103.0, 104.0, 105.0, 106.0],
            "close": [101.0, 102.0, np.nan, 103.0, 104.0, 105.0, np.nan, 107.0, 108.0, 109.0],
            "volume": [1000.0, 1100.0, 900.0, np.nan, 800.0, 1300.0, 700.0, np.nan, 600.0, 1500.0],
        },
    ).with_columns((pl.col("close").pct_change()).alias("returns"))

    # All features should handle NaN gracefully
    result_kyle = data_with_nan.with_columns(
        kyle_lambda("returns", "volume", period=5).alias("kyle_lambda_5")
    )
    result_amihud = data_with_nan.with_columns(
        amihud_illiquidity("returns", "volume", "close", period=5).alias("amihud_5")
    )
    result_roll = data_with_nan.with_columns(
        roll_spread_estimator("close", period=5).alias("roll_spread_5")
    )

    # Should complete without error
    assert len(result_kyle) == 10
    assert len(result_amihud) == 10
    assert len(result_roll) == 10


# =============================================================================
# Registry Tests
# =============================================================================


def test_registry_has_microstructure_features():
    """Test that microstructure features are registered in the global registry."""
    from ml4t.engineer.core.registry import get_registry

    registry = get_registry()

    # Check all microstructure features are registered
    assert registry.get("kyle_lambda") is not None
    assert registry.get("amihud_illiquidity") is not None
    assert registry.get("roll_spread_estimator") is not None
    assert registry.get("effective_tick_rule") is not None
    assert registry.get("realized_spread") is not None
    assert registry.get("order_flow_imbalance") is not None

    # Verify category
    meta = registry.get("kyle_lambda")
    assert meta.category == "microstructure"
