"""
Comprehensive coverage tests for volatility indicators.

Tests edge cases and various code paths to improve coverage.
"""

import numpy as np
import polars as pl
import pytest


class TestATRCoverage:
    """Coverage tests for Average True Range."""

    def test_atr_basic(self):
        """Test basic ATR calculation."""
        from ml4t.engineer.features.volatility.atr import atr

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = atr(high, low, close, period=14)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # ATR should be non-negative
        assert np.all(valid_values >= 0)

    def test_atr_short_period(self):
        """Test ATR with short period."""
        from ml4t.engineer.features.volatility.atr import atr

        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = atr(high, low, close, period=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0


class TestNATRCoverage:
    """Coverage tests for Normalized Average True Range."""

    def test_natr_basic(self):
        """Test basic NATR calculation."""
        from ml4t.engineer.features.volatility.natr import natr

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = natr(high, low, close, period=14)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0


class TestTRANGECoverage:
    """Coverage tests for True Range."""

    def test_trange_basic(self):
        """Test basic TRANGE calculation."""
        from ml4t.engineer.features.volatility.trange import trange

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = trange(high, low, close)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # True range should be non-negative
        assert np.all(valid_values >= 0)


class TestAdvancedVolatilityPolars:
    """Coverage tests for advanced volatility estimators using Polars."""

    @pytest.fixture
    def ohlc_df(self):
        """Create OHLC DataFrame for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_prices = close + np.random.randn(n) * 0.1
        return pl.DataFrame(
            {
                "open": open_prices,
                "high": high,
                "low": low,
                "close": close,
            }
        )

    def test_parkinson_polars(self, ohlc_df):
        """Test Parkinson volatility with Polars."""
        from ml4t.engineer.features.volatility.parkinson_volatility import parkinson_volatility

        result = ohlc_df.select(parkinson_volatility("high", "low", period=20))
        assert result is not None
        assert len(result) == len(ohlc_df)

    def test_garman_klass_polars(self, ohlc_df):
        """Test Garman-Klass volatility with Polars."""
        from ml4t.engineer.features.volatility.garman_klass_volatility import (
            garman_klass_volatility,
        )

        result = ohlc_df.select(garman_klass_volatility("open", "high", "low", "close", period=20))
        assert result is not None
        assert len(result) == len(ohlc_df)

    def test_rogers_satchell_polars(self, ohlc_df):
        """Test Rogers-Satchell volatility with Polars."""
        from ml4t.engineer.features.volatility.rogers_satchell_volatility import (
            rogers_satchell_volatility,
        )

        result = ohlc_df.select(
            rogers_satchell_volatility("open", "high", "low", "close", period=20)
        )
        assert result is not None
        assert len(result) == len(ohlc_df)

    def test_yang_zhang_polars(self, ohlc_df):
        """Test Yang-Zhang volatility with Polars."""
        from ml4t.engineer.features.volatility.yang_zhang_volatility import yang_zhang_volatility

        result = ohlc_df.select(yang_zhang_volatility("open", "high", "low", "close", period=20))
        assert result is not None
        assert len(result) == len(ohlc_df)


class TestMLVolatilityPolars:
    """Coverage tests for ML volatility features using Polars."""

    @pytest.fixture
    def price_df(self):
        """Create price DataFrame for testing."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        return pl.DataFrame({"close": close})

    def test_ewma_volatility_polars(self, price_df):
        """Test EWMA volatility with Polars."""
        from ml4t.engineer.features.volatility.ewma_volatility import ewma_volatility

        result = price_df.select(ewma_volatility("close", span=20))
        assert result is not None
        assert len(result) == len(price_df)

    def test_realized_volatility_polars(self, price_df):
        """Test realized volatility with Polars."""
        from ml4t.engineer.features.volatility.realized_volatility import realized_volatility

        result = price_df.select(realized_volatility("close", period=20))
        assert result is not None
        assert len(result) == len(price_df)

    def test_volatility_percentile_rank_polars(self, price_df):
        """Test volatility percentile rank with Polars."""
        from ml4t.engineer.features.volatility.volatility_percentile_rank import (
            volatility_percentile_rank,
        )

        result = price_df.select(volatility_percentile_rank("close", period=20, lookback=100))
        assert result is not None
        assert len(result) == len(price_df)


class TestEdgeCases:
    """Test edge cases for volatility indicators."""

    def test_constant_prices(self):
        """Test ATR with constant prices."""
        from ml4t.engineer.features.volatility.atr import atr

        n = 50
        close = np.array([100.0] * n)
        high = np.array([100.0] * n)
        low = np.array([100.0] * n)

        result = atr(high, low, close, period=14)

        # Should handle gracefully
        assert result is not None
        assert len(result) == n

    def test_short_data(self):
        """Test with data shorter than period."""
        from ml4t.engineer.features.volatility.atr import atr

        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([100.0, 101.0, 102.0])

        result = atr(high, low, close, period=14)

        # Should handle gracefully
        assert result is not None
        assert len(result) == 3

    def test_high_volatility(self):
        """Test with high volatility data."""
        from ml4t.engineer.features.volatility.atr import atr

        np.random.seed(42)
        n = 100
        # Large random jumps
        close = 100 + np.cumsum(np.random.randn(n) * 5)
        high = close + np.abs(np.random.randn(n) * 3)
        low = close - np.abs(np.random.randn(n) * 3)

        result = atr(high, low, close, period=14)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # High volatility should give higher ATR
        assert valid_values.mean() > 1.0
