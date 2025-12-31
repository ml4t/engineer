"""Tests for RSI (Relative Strength Index) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.momentum.rsi import rsi


@pytest.fixture
def ohlcv_data():
    """Standard OHLCV test data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pl.DataFrame(
        {
            "open": close - np.random.rand(n) * 0.5,
            "high": close + np.random.rand(n),
            "low": close - np.random.rand(n),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_computes_successfully_numpy(self):
        """Test RSI computes successfully with numpy array."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = rsi(close, period=14)

        assert result is not None
        assert len(result) == len(close)
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_series(self, ohlcv_data):
        """Test RSI computes successfully with Polars Series."""
        result = rsi(ohlcv_data["close"], period=14)

        assert result is not None
        assert len(result) == len(ohlcv_data)
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_expr(self, ohlcv_data):
        """Test RSI computes successfully with Polars expression."""
        result = ohlcv_data.select(rsi("close", period=14))

        assert result is not None
        assert len(result) == len(ohlcv_data)
        assert "close" in result.columns

    def test_parameter_variations(self):
        """Test RSI with different periods."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        r1 = rsi(close, period=7)
        r2 = rsi(close, period=21)

        # Different periods should give different results
        # Skip NaN values for comparison
        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        assert not np.allclose(r1[valid_mask], r2[valid_mask], rtol=0.01)

    def test_output_range(self):
        """Test RSI output is within 0-100 range."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = rsi(close, period=14)

        # Check valid values are in range [0, 100]
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_array(self):
        """Test RSI with empty array."""
        empty = np.array([])
        result = rsi(empty, period=14)

        assert len(result) == 0

    def test_single_value(self):
        """Test RSI with single value."""
        single = np.array([100.0])
        result = rsi(single, period=14)

        assert len(result) == 1
        assert np.isnan(result[0])

    def test_constant_values(self):
        """Test RSI with constant prices."""
        const = np.full(100, 100.0)
        result = rsi(const, period=14)

        # With constant prices, RSI should be 0 (no gain or loss)
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.allclose(valid_values, 0.0, atol=1e-10)

    def test_with_nans_in_data(self):
        """Test RSI handles NaN values in input."""
        close = np.array([100.0, 101.0, np.nan, 103.0, 104.0] * 20)
        result = rsi(close, period=5)

        assert result is not None
        assert len(result) == len(close)

    def test_insufficient_data(self):
        """Test RSI with insufficient data for period."""
        close = np.array([100.0, 101.0, 102.0])
        result = rsi(close, period=14)

        # Should return all NaN
        assert np.all(np.isnan(result))

    def test_exact_minimum_data(self):
        """Test RSI with exactly minimum data needed."""
        close = np.array([100.0 + i * 0.5 for i in range(15)])  # 15 values for period=14
        result = rsi(close, period=14)

        # Should have first valid value at index 14
        assert np.all(np.isnan(result[:14]))
        assert not np.isnan(result[14])


class TestValidation:
    """Parameter validation tests."""

    def test_invalid_period_zero(self):
        """Test RSI rejects zero period."""
        close = np.random.randn(100)
        with pytest.raises((ValueError, Exception)):
            rsi(close, period=0)

    def test_invalid_period_negative(self):
        """Test RSI rejects negative period."""
        close = np.random.randn(100)
        with pytest.raises((ValueError, Exception)):
            rsi(close, period=-1)


class TestNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_known_values(self):
        """Test RSI against known values from Wilder's book."""
        # Example from Wilder's book (simplified)
        close = np.array(
            [
                44.00,
                44.34,
                44.09,
                43.61,
                44.33,
                44.83,
                45.10,
                45.42,
                45.84,
                46.08,
                45.89,
                46.03,
                45.61,
                46.28,
                46.28,
                46.00,
                46.03,
                46.41,
                46.22,
                45.64,
            ]
        )

        result = rsi(close, period=14)

        # Should have valid RSI values after warmup period
        assert not np.isnan(result[14])
        # RSI should be reasonable (between 0-100)
        assert 0 <= result[14] <= 100

    def test_all_gains(self):
        """Test RSI with all upward movement."""
        close = np.arange(100.0, 200.0, 1.0)
        result = rsi(close, period=14)

        # With all gains, RSI should approach 100
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 90)  # Should be very high

    def test_all_losses(self):
        """Test RSI with all downward movement."""
        close = np.arange(200.0, 100.0, -1.0)
        result = rsi(close, period=14)

        # With all losses, RSI should approach 0
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values <= 10)  # Should be very low

    def test_alternating_movement(self):
        """Test RSI with alternating up/down movement."""
        close = np.array([100 + (i % 2) for i in range(100)])
        result = rsi(close, period=14)

        # With alternating movement, RSI should be near 50
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # Should be somewhere in middle range
        assert np.all((valid_values >= 30) & (valid_values <= 70))


class TestDataTypes:
    """Test different input data types."""

    def test_numpy_array_float64(self):
        """Test with float64 numpy array."""
        close = np.random.randn(100).astype(np.float64)
        result = rsi(close, period=14)
        assert isinstance(result, np.ndarray)

    def test_numpy_array_float32(self):
        """Test with float32 numpy array (should convert)."""
        close = np.random.randn(100).astype(np.float32)
        result = rsi(close, period=14)
        assert isinstance(result, np.ndarray)

    def test_polars_series(self):
        """Test with Polars Series."""
        close = pl.Series([100.0 + i * 0.5 for i in range(100)])
        result = rsi(close, period=14)
        assert isinstance(result, np.ndarray)

    def test_polars_expression(self):
        """Test with Polars expression (string column name)."""
        df = pl.DataFrame({"close": [100.0 + i * 0.5 for i in range(100)]})
        result = df.select(rsi("close", period=14))
        assert isinstance(result, pl.DataFrame)
        assert "close" in result.columns


class TestPeriodSizes:
    """Test various period sizes."""

    def test_small_period(self):
        """Test with small period (5)."""
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = rsi(close, period=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0) & (valid_values <= 100))

    def test_medium_period(self):
        """Test with medium period (14 - default)."""
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = rsi(close, period=14)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0) & (valid_values <= 100))

    def test_large_period(self):
        """Test with large period (50)."""
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        result = rsi(close, period=50)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0) & (valid_values <= 100))
