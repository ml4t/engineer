"""Tests for labeling features."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.labeling import (
    BarrierConfig,
    triple_barrier_labels,
)


@pytest.fixture
def sample_price_data():
    """Sample price data for labeling tests."""
    np.random.seed(42)
    n = 100

    # Generate realistic price movement
    returns = np.random.randn(n) * 0.01
    returns[0] = 0  # First return is zero
    prices = 100 * np.exp(np.cumsum(returns))

    # Compute simple returns for fixed horizon
    simple_returns = np.concatenate([[np.nan], np.diff(prices) / prices[:-1]])

    return pl.DataFrame(
        {
            "timestamp": pl.int_range(0, n, eager=True),
            "close": prices,
            "returns": simple_returns,
        }
    )


@pytest.fixture
def trending_price_data():
    """Price data with clear trend for trend scanning tests."""

    # Create uptrend then downtrend
    uptrend = np.linspace(100, 110, 25)
    downtrend = np.linspace(110, 95, 25)
    prices = np.concatenate([uptrend, downtrend])

    return pl.DataFrame(
        {
            "close": prices,
        }
    )


# ============================================================================
# Triple Barrier Tests
# ============================================================================


def test_triple_barrier_basic_functionality(sample_price_data):
    """Test triple barrier basic calculation."""
    config = BarrierConfig(
        upper_barrier=0.02,
        lower_barrier=0.01,
        max_holding_period=10,
    )
    result = triple_barrier_labels(
        sample_price_data,
        config,
        price_col="close",
    )

    assert "label" in result.columns
    assert "label_price" in result.columns
    assert "label_return" in result.columns
    assert "label_bars" in result.columns
    assert "barrier_hit" in result.columns
    assert len(result) == len(sample_price_data)


def test_triple_barrier_label_values(sample_price_data):
    """Test that labels are in expected range."""
    config = BarrierConfig(
        upper_barrier=0.02,
        lower_barrier=0.01,
        max_holding_period=10,
    )
    result = triple_barrier_labels(
        sample_price_data,
        config,
    )

    labels = result["label"].drop_nulls()
    assert labels.is_in([-1, 0, 1]).all()


def test_triple_barrier_with_timestamp(sample_price_data):
    """Test triple barrier with timestamp column."""
    config = BarrierConfig(
        upper_barrier=0.03,
        lower_barrier=0.015,
        max_holding_period=15,
    )
    result = triple_barrier_labels(
        sample_price_data,
        config,
        price_col="close",
        timestamp_col="timestamp",
    )

    assert "label_duration" in result.columns
    # Duration should be present for some samples
    durations = result["label_duration"].drop_nulls()
    assert len(durations) > 0


def test_triple_barrier_long_position(sample_price_data):
    """Test triple barrier with long position side."""
    config = BarrierConfig(
        upper_barrier=0.02,
        lower_barrier=0.01,
        max_holding_period=10,
        side=1,  # Long
    )
    result = triple_barrier_labels(
        sample_price_data,
        config,
    )

    assert "label" in result.columns
    # Labels should be valid
    labels = result["label"].drop_nulls()
    assert len(labels) > 0


def test_triple_barrier_short_position(sample_price_data):
    """Test triple barrier with short position side."""
    config = BarrierConfig(
        upper_barrier=0.02,
        lower_barrier=0.01,
        max_holding_period=10,
        side=-1,  # Short
    )
    result = triple_barrier_labels(
        sample_price_data,
        config,
    )

    assert "label" in result.columns
    labels = result["label"].drop_nulls()
    assert len(labels) > 0


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_insufficient_data_triple_barrier():
    """Test triple barrier with insufficient data."""
    tiny_df = pl.DataFrame(
        {
            "close": [100.0, 101.0, 100.5],
        }
    )

    # Should not raise, but results may be limited
    config = BarrierConfig(
        upper_barrier=0.02,
        lower_barrier=0.01,
        max_holding_period=10,
    )
    result = triple_barrier_labels(
        tiny_df,
        config,
    )

    assert "label" in result.columns
    assert len(result) == len(tiny_df)


def test_constant_prices_triple_barrier():
    """Test triple barrier with constant prices (no movement)."""
    const_df = pl.DataFrame(
        {
            "close": [100.0] * 50,
        }
    )

    config = BarrierConfig(
        upper_barrier=0.02,
        lower_barrier=0.01,
        max_holding_period=10,
    )
    result = triple_barrier_labels(
        const_df,
        config,
    )

    # All should timeout (label=0) since no price movement
    labels = result["label"].drop_nulls()
    if len(labels) > 0:
        # Most should be timeout labels
        n_timeout = (labels == 0).sum()
        assert n_timeout > len(labels) * 0.8  # At least 80% timeout


def test_nan_handling_triple_barrier(sample_price_data):
    """Test triple barrier with NaN in prices."""
    # Add some NaN values
    data_with_nan = sample_price_data.with_columns(
        pl.when(pl.col("close").is_between(105, 107))
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    # Should not crash
    config = BarrierConfig(
        upper_barrier=0.02,
        lower_barrier=0.01,
        max_holding_period=10,
    )
    result = triple_barrier_labels(
        data_with_nan,
        config,
    )

    assert "label" in result.columns


# ============================================================================
# Registry Tests
# ============================================================================


def test_registry_has_labeling_features():
    """Test that labeling features are registered in the global registry."""
    from ml4t.engineer.core.registry import get_registry
    from ml4t.engineer.labeling import register_labeling_features

    registry = get_registry()

    # Ensure features are registered
    register_labeling_features(registry)

    # Check triple_barrier is registered
    triple_barrier = registry.get("triple_barrier")
    assert triple_barrier is not None
    assert triple_barrier.category == "labeling"
    assert triple_barrier.normalized is False  # Forward-looking
    # lookback is a callable that returns 0 (forward-looking)
    assert callable(triple_barrier.lookback)
    assert triple_barrier.lookback() == 0


def test_all_features_execute(sample_price_data):
    """Test that triple_barrier can execute without errors."""
    config = BarrierConfig(
        upper_barrier=0.02,
        lower_barrier=0.01,
    )
    result = triple_barrier_labels(
        sample_price_data,
        config,
        price_col="close",
    )
    assert isinstance(result, pl.DataFrame)


# ============================================================================
# Note: Comprehensive tests for fixed_time_horizon_labels and
# trend_scanning_labels are in test_new_labeling_methods.py
# ============================================================================
