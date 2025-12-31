"""Tests targeting uncovered lines in ema.py for coverage boost."""

import numpy as np
import polars as pl

from ml4t.engineer.features.trend.ema import ema, ema_numba


class TestEMANumba:
    """Tests for ema_numba Numba function."""

    def test_basic_calculation(self):
        """Test basic EMA calculation."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)

        result = ema_numba(close, period=10)

        # EMA should be smooth
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_insufficient_data(self):
        """Test with insufficient data for period."""
        close = np.array([100.0, 101.0, 99.0])
        result = ema_numba(close, period=10)

        # All should be NaN
        assert all(np.isnan(result))

    def test_trending_market(self):
        """Test EMA follows trend."""
        close = np.linspace(100, 150, 50)
        result = ema_numba(close, period=10)

        valid = result[~np.isnan(result)]
        if len(valid) > 5:
            # EMA should be increasing
            diffs = np.diff(valid)
            assert np.mean(diffs > 0) > 0.8

    def test_constant_price(self):
        """Test EMA with constant prices."""
        close = np.ones(50) * 100

        result = ema_numba(close, period=10)

        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            # EMA of constant should be constant
            assert np.allclose(valid, 100.0, rtol=1e-6)

    def test_different_periods(self):
        """Test EMA with different time periods."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        ema_5 = ema_numba(close, period=5)
        ema_20 = ema_numba(close, period=20)

        valid_5 = np.sum(~np.isnan(ema_5))
        valid_20 = np.sum(~np.isnan(ema_20))
        assert valid_5 > valid_20


class TestEMAPolars:
    """Tests for EMA Polars wrapper."""

    def test_polars_expression(self):
        """Test EMA with Polars expression."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(ema("close", period=10).alias("ema"))

        assert "ema" in result.columns
        assert len(result) == len(df)

    def test_polars_series(self):
        """Test EMA with Polars Series."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        series = pl.Series(close)

        result = ema(series, period=10)

        assert len(result) == len(close)

    def test_numpy_array(self):
        """Test EMA with NumPy array."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)

        result = ema(close, period=10)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(close)
