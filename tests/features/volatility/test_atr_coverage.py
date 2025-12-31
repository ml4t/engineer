"""Tests targeting uncovered lines in atr.py for coverage boost."""

import numpy as np
import polars as pl

from ml4t.engineer.features.volatility.atr import atr, atr_numba


class TestATRNumba:
    """Tests for atr_numba Numba function."""

    def test_basic_calculation(self):
        """Test basic ATR calculation."""
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = atr_numba(high, low, close, period=14)

        # ATR should be positive
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(v >= 0 for v in valid)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([100.0, 101.0, 102.0])

        result = atr_numba(high, low, close, period=14)
        assert all(np.isnan(result))

    def test_volatile_market(self):
        """Test ATR in volatile market."""
        np.random.seed(42)
        n = 50
        # More volatile market
        close = 100 + np.cumsum(np.random.randn(n) * 2.0)
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)

        result = atr_numba(high, low, close, period=14)
        valid = result[~np.isnan(result)]

        # ATR should be relatively high in volatile market
        if len(valid) > 0:
            assert np.mean(valid) > 0.5

    def test_constant_price(self):
        """Test ATR with constant prices."""
        high = np.ones(30) * 100
        low = np.ones(30) * 100
        close = np.ones(30) * 100

        result = atr_numba(high, low, close, period=14)

        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            # ATR should be very low/zero
            assert all(v < 0.01 for v in valid)


class TestATRPolars:
    """Tests for ATR Polars wrapper."""

    def test_polars_expression(self):
        """Test ATR with Polars expression."""
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        df = pl.DataFrame({"high": high, "low": low, "close": close})

        result = df.select(atr("high", "low", "close", period=14).alias("atr"))

        assert "atr" in result.columns
        assert len(result) == len(df)

    def test_different_periods(self):
        """Test ATR with different time periods."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        atr_7 = atr_numba(high, low, close, period=7)
        atr_21 = atr_numba(high, low, close, period=21)

        # Shorter period should have more valid values
        valid_7 = np.sum(~np.isnan(atr_7))
        valid_21 = np.sum(~np.isnan(atr_21))
        assert valid_7 > valid_21
