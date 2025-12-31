"""
Tests for advanced volatility features.

Tests cover:
- Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang volatility
- Realized volatility and GARCH forecast
- Basic functionality, edge cases, parameter validation
- Output structure and positive values validation
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.volatility import (
    garch_forecast,
    garman_klass_volatility,
    parkinson_volatility,
    realized_volatility,
    rogers_satchell_volatility,
    yang_zhang_volatility,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlc_data():
    """Sample OHLC data for volatility testing."""
    np.random.seed(42)
    n = 100

    # Generate realistic price movement
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open = close + np.random.randn(n) * 0.2

    returns = np.concatenate([[np.nan], np.diff(np.log(close))])

    return pl.DataFrame(
        {
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "returns": returns,
        }
    )


# =============================================================================
# Parkinson Volatility Tests
# =============================================================================


def test_parkinson_basic_functionality(sample_ohlc_data):
    """Test Parkinson volatility basic calculation."""
    result = sample_ohlc_data.with_columns(
        parkinson_volatility("high", "low", period=20).alias("parkinson_20_ann252")
    )

    assert "parkinson_20_ann252" in result.columns
    assert len(result) == len(sample_ohlc_data)
    assert result["parkinson_20_ann252"].dtype == pl.Float64


def test_parkinson_annualized_vs_non_annualized(sample_ohlc_data):
    """Test Parkinson with and without annualization."""
    annualized = sample_ohlc_data.with_columns(
        parkinson_volatility("high", "low", period=20, annualize=True).alias("parkinson_20_ann252")
    )
    non_annualized = sample_ohlc_data.with_columns(
        parkinson_volatility("high", "low", period=20, annualize=False).alias("parkinson_20")
    )

    # Annualized should be higher by sqrt(trading_periods)
    ann_values = annualized["parkinson_20_ann252"].drop_nulls().drop_nans()
    non_ann_values = non_annualized["parkinson_20"].drop_nulls().drop_nans()

    assert len(ann_values) > 0
    assert len(non_ann_values) > 0
    # First valid values should differ by ~sqrt(252)
    assert ann_values[0] > non_ann_values[0]


def test_parkinson_positive_values(sample_ohlc_data):
    """Test that Parkinson volatility is always positive."""
    result = sample_ohlc_data.with_columns(
        parkinson_volatility("high", "low", period=10).alias("parkinson_10_ann252")
    )
    valid_values = result["parkinson_10_ann252"].drop_nulls().drop_nans()

    assert len(valid_values) > 0
    assert (valid_values > 0).all()


# =============================================================================
# Garman-Klass Volatility Tests
# =============================================================================


def test_garman_klass_basic_functionality(sample_ohlc_data):
    """Test Garman-Klass volatility basic calculation."""
    result = sample_ohlc_data.with_columns(
        garman_klass_volatility("high", "low", "open", "close", period=20).alias(
            "garman_klass_20_ann252"
        )
    )

    assert "garman_klass_20_ann252" in result.columns
    assert len(result) == len(sample_ohlc_data)
    assert result["garman_klass_20_ann252"].dtype == pl.Float64


def test_garman_klass_positive_values(sample_ohlc_data):
    """Test that Garman-Klass volatility is always positive."""
    result = sample_ohlc_data.with_columns(
        garman_klass_volatility("high", "low", "open", "close", period=15).alias(
            "garman_klass_15_ann252"
        )
    )
    valid_values = result["garman_klass_15_ann252"].drop_nulls().drop_nans()

    assert len(valid_values) > 0
    assert (valid_values > 0).all()


# =============================================================================
# Rogers-Satchell Volatility Tests
# =============================================================================


def test_rogers_satchell_basic_functionality(sample_ohlc_data):
    """Test Rogers-Satchell volatility basic calculation."""
    result = sample_ohlc_data.with_columns(
        rogers_satchell_volatility("high", "low", "open", "close", period=20).alias(
            "rogers_satchell_20_ann252"
        )
    )

    assert "rogers_satchell_20_ann252" in result.columns
    assert len(result) == len(sample_ohlc_data)


def test_rogers_satchell_positive_values(sample_ohlc_data):
    """Test that Rogers-Satchell volatility is always positive."""
    result = sample_ohlc_data.with_columns(
        rogers_satchell_volatility("high", "low", "open", "close", period=15).alias(
            "rogers_satchell_15_ann252"
        )
    )
    valid_values = result["rogers_satchell_15_ann252"].drop_nulls().drop_nans()

    assert len(valid_values) > 0
    assert (valid_values > 0).all()


# =============================================================================
# Yang-Zhang Volatility Tests
# =============================================================================


def test_yang_zhang_basic_functionality(sample_ohlc_data):
    """Test Yang-Zhang volatility basic calculation."""
    result = sample_ohlc_data.with_columns(
        yang_zhang_volatility("high", "low", "open", "close", period=20).alias(
            "yang_zhang_20_ann252"
        )
    )

    assert "yang_zhang_20_ann252" in result.columns
    assert len(result) == len(sample_ohlc_data)


def test_yang_zhang_positive_values(sample_ohlc_data):
    """Test that Yang-Zhang volatility is always positive."""
    result = sample_ohlc_data.with_columns(
        yang_zhang_volatility("high", "low", "open", "close", period=20).alias(
            "yang_zhang_20_ann252"
        )
    )
    valid_values = result["yang_zhang_20_ann252"].drop_nulls().drop_nans()

    assert len(valid_values) > 0
    assert (valid_values > 0).all()


# =============================================================================
# Realized Volatility Tests
# =============================================================================


def test_realized_vol_basic_functionality(sample_ohlc_data):
    """Test realized volatility basic calculation."""
    result = sample_ohlc_data.with_columns(
        realized_volatility("returns", period=20).alias("realized_vol_20_ann252")
    )

    assert "realized_vol_20_ann252" in result.columns
    assert len(result) == len(sample_ohlc_data)


def test_realized_vol_positive_values(sample_ohlc_data):
    """Test that realized volatility is always positive."""
    result = sample_ohlc_data.with_columns(
        realized_volatility("returns", period=15).alias("realized_vol_15_ann252")
    )
    valid_values = result["realized_vol_15_ann252"].drop_nulls().drop_nans()

    assert len(valid_values) > 0
    assert (valid_values > 0).all()


# =============================================================================
# GARCH Volatility Tests
# =============================================================================


def test_garch_basic_functionality(sample_ohlc_data):
    """Test GARCH volatility forecast basic calculation."""
    result = sample_ohlc_data.with_columns(
        garch_forecast("returns", horizon=1).alias("garch_forecast_1")
    )

    assert "garch_forecast_1" in result.columns
    assert len(result) == len(sample_ohlc_data)


def test_garch_positive_values(sample_ohlc_data):
    """Test that GARCH forecast is always positive."""
    result = sample_ohlc_data.with_columns(
        garch_forecast("returns", horizon=5).alias("garch_forecast_5")
    )
    valid_values = result["garch_forecast_5"].drop_nulls().drop_nans()

    assert len(valid_values) > 0
    assert (valid_values > 0).all()


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_insufficient_data():
    """Test advanced volatility features with insufficient data."""
    tiny_df = pl.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "returns": [np.nan, 0.01],
        }
    )

    # Should not raise, but results may be all null/nan
    result = tiny_df.with_columns(
        parkinson_volatility("high", "low", period=20).alias("parkinson_20_ann252")
    )
    assert "parkinson_20_ann252" in result.columns


def test_constant_prices():
    """Test with constant prices (no volatility)."""
    const_df = pl.DataFrame(
        {
            "open": [100.0] * 50,
            "high": [100.0] * 50,
            "low": [100.0] * 50,
            "close": [100.0] * 50,
            "returns": [0.0] * 50,
        }
    )

    result = const_df.with_columns(
        parkinson_volatility("high", "low", period=20).alias("parkinson_20_ann252")
    )
    # Should have very low/zero volatility
    valid_values = result["parkinson_20_ann252"].drop_nulls().drop_nans()
    if len(valid_values) > 0:
        # All values should be zero or very close to zero
        assert (valid_values < 1e-10).all()


# =============================================================================
# Registry Tests
# =============================================================================


def test_registry_has_advanced_volatility_features():
    """Test that advanced volatility features are registered in the global registry."""
    from ml4t.engineer.core.registry import get_registry

    registry = get_registry()

    # Check all advanced volatility features are registered
    assert registry.get("parkinson_volatility") is not None
    assert registry.get("garman_klass_volatility") is not None
    assert registry.get("rogers_satchell_volatility") is not None
    assert registry.get("yang_zhang_volatility") is not None
    assert registry.get("realized_volatility") is not None
    assert registry.get("garch_forecast") is not None

    # Verify they're in volatility category
    assert registry.get("parkinson_volatility").category == "volatility"
    assert registry.get("garman_klass_volatility").category == "volatility"
    assert registry.get("rogers_satchell_volatility").category == "volatility"
    assert registry.get("yang_zhang_volatility").category == "volatility"
    assert registry.get("realized_volatility").category == "volatility"
    assert registry.get("garch_forecast").category == "volatility"


def test_all_features_execute(sample_ohlc_data):
    """Test that all features can execute without errors."""
    # Test OHLC-based features
    result = sample_ohlc_data.with_columns(
        [
            parkinson_volatility("high", "low").alias("parkinson"),
            garman_klass_volatility("high", "low", "open", "close").alias("garman_klass"),
            rogers_satchell_volatility("high", "low", "open", "close").alias("rogers_satchell"),
            yang_zhang_volatility("high", "low", "open", "close").alias("yang_zhang"),
        ]
    )
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(sample_ohlc_data)

    # Test returns-based features separately
    result = sample_ohlc_data.with_columns(
        [
            realized_volatility("returns").alias("realized_vol"),
            garch_forecast("returns").alias("garch"),
        ]
    )
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(sample_ohlc_data)
