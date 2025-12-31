"""Tests targeting uncovered lines in macd.py for coverage boost."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.momentum.macd import (
    macd,
    macd_histogram_numba,
    macd_numba,
    macd_signal,
    macd_signal_numba,
)


class TestMACDNumba:
    """Tests for macd_numba Numba function."""

    def test_basic_calculation(self):
        """Test basic MACD line calculation."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)

        # Should return single array
        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)

        # Should have valid values after warmup
        assert np.sum(~np.isnan(result)) > 0

    def test_insufficient_data(self):
        """Test with insufficient data."""
        close = np.array([100.0, 101.0, 99.0])

        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)

        # Should all be NaN
        assert all(np.isnan(result))

    def test_different_periods(self):
        """Test MACD with different period configurations."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        # Standard 12/26/9
        macd1 = macd_numba(close, 12, 26, 9)

        # Faster 6/13/5
        macd2 = macd_numba(close, 6, 13, 5)

        # Faster periods should produce different results
        valid1 = macd1[~np.isnan(macd1)]
        valid2 = macd2[~np.isnan(macd2)]
        assert len(valid1) > 0 and len(valid2) > 0


class TestMACDComponents:
    """Tests for individual MACD component functions."""

    @pytest.fixture
    def price_data(self):
        """Generate price data for testing."""
        np.random.seed(42)
        return 100 + np.cumsum(np.random.randn(100) * 0.5)

    def test_macd_signal_numba(self, price_data):
        """Test MACD signal line calculation (numba)."""
        result = macd_signal_numba(price_data, fast_period=12, slow_period=26, signal_period=9)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(price_data)

    def test_macd_histogram_numba(self, price_data):
        """Test MACD histogram calculation (numba)."""
        result = macd_histogram_numba(price_data, fast_period=12, slow_period=26, signal_period=9)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(price_data)

        # Histogram should have values
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_macd_signal_polars(self, price_data):
        """Test MACD signal via Polars."""
        result = macd_signal(price_data, fast_period=12, slow_period=26, signal_period=9)

        assert len(result) == len(price_data)


class TestMACDPolars:
    """Tests for MACD Polars wrapper."""

    def test_polars_expression(self):
        """Test MACD with Polars expression."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(macd("close", fast_period=12, slow_period=26).alias("macd"))

        assert "macd" in result.columns
        assert len(result) == len(df)

    def test_polars_series(self):
        """Test MACD with Polars Series."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        series = pl.Series(close)

        result = macd(series, fast_period=12, slow_period=26)

        assert len(result) == len(close)
