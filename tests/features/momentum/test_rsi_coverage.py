"""Tests targeting uncovered lines in rsi.py for coverage boost."""

import numpy as np
import polars as pl

from ml4t.engineer.features.momentum.rsi import rsi, rsi_numba


class TestRSINumba:
    """Tests for rsi_numba Numba function."""

    def test_basic_calculation(self):
        """Test basic RSI calculation."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)

        result = rsi_numba(close, period=14)

        # RSI should be in 0-100 range
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(0 <= v <= 100 for v in valid)

    def test_insufficient_data(self):
        """Test with insufficient data for period."""
        close = np.array([100.0, 101.0, 99.0])
        result = rsi_numba(close, period=14)

        # All should be NaN
        assert all(np.isnan(result))

    def test_trending_market(self):
        """Test RSI in trending market."""
        # Strong uptrend - RSI should be high
        close = np.linspace(100, 150, 50)
        result = rsi_numba(close, period=14)

        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.mean(valid) > 50  # Should be above 50 in uptrend

    def test_downtrending_market(self):
        """Test RSI in downtrending market."""
        close = np.linspace(150, 100, 50)
        result = rsi_numba(close, period=14)

        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.mean(valid) < 50  # Should be below 50 in downtrend

    def test_different_periods(self):
        """Test RSI with different time periods."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        rsi_7 = rsi_numba(close, period=7)
        rsi_21 = rsi_numba(close, period=21)

        # Shorter period should have values earlier
        valid_7 = np.sum(~np.isnan(rsi_7))
        valid_21 = np.sum(~np.isnan(rsi_21))
        assert valid_7 > valid_21


class TestRSIPolars:
    """Tests for RSI Polars wrapper."""

    def test_polars_expression(self):
        """Test RSI with Polars expression."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(rsi("close", period=14).alias("rsi"))

        assert "rsi" in result.columns
        assert len(result) == len(df)

    def test_polars_series(self):
        """Test RSI with Polars Series input."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        series = pl.Series(close)

        result = rsi(series, period=14)

        assert len(result) == len(close)

    def test_numpy_array(self):
        """Test RSI with NumPy array input."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)

        result = rsi(close, period=14)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)
