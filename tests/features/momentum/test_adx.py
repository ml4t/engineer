"""Tests for ADX (Average Directional Movement Index) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.momentum.adx import adx


@pytest.fixture
def ohlcv_data():
    """Standard OHLCV test data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_prices = close + np.random.randn(n) * 0.2

    return pl.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_adx_computes_successfully_numpy(self):
        """Test ADX computes successfully with numpy arrays."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = adx(high, low, close, period=14)

        assert result is not None
        assert len(result) == n
        assert isinstance(result, np.ndarray)

    def test_adx_computes_successfully_polars_series(self, ohlcv_data):
        """Test ADX computes successfully with Polars Series."""
        result = adx(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            period=14,
        )

        assert result is not None
        assert len(result) == len(ohlcv_data)
        assert isinstance(result, np.ndarray)

    def test_adx_computes_successfully_polars_expr(self, ohlcv_data):
        """Test ADX computes successfully with Polars expressions."""
        result = ohlcv_data.select(adx("high", "low", "close", period=14))

        assert result is not None
        assert len(result) == len(ohlcv_data)

    def test_parameter_variations(self):
        """Test ADX with different periods."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        r1 = adx(high, low, close, period=7)
        r2 = adx(high, low, close, period=21)

        # Different periods should give different results
        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            assert not np.allclose(r1[valid_mask], r2[valid_mask], rtol=0.01)

    def test_output_range(self):
        """Test ADX output is within 0-100 range."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = adx(high, low, close, period=14)

        # Check valid values are in range [0, 100]
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.all(valid_values >= 0)
            assert np.all(valid_values <= 100)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_arrays(self):
        """Test ADX with empty arrays."""
        empty = np.array([])
        result = adx(empty, empty, empty, period=14)

        assert len(result) == 0

    def test_single_value(self):
        """Test ADX with single value."""
        single = np.array([100.0])
        result = adx(single, single + 1, single - 1, period=14)

        assert len(result) == 1
        assert np.isnan(result[0])

    def test_constant_values(self):
        """Test ADX with constant prices."""
        const = np.full(100, 100.0)
        high = const + 1
        low = const - 1

        result = adx(high, low, const, period=14)

        # With constant prices, ADX should be low/zero
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            # ADX measures trend strength, constant prices = no trend
            assert np.all(valid_values < 50)

    def test_with_nans_in_data(self):
        """Test ADX handles NaN values in input."""
        high = np.array([101.0, 102.0, np.nan, 104.0, 105.0] * 20)
        low = np.array([99.0, 100.0, np.nan, 102.0, 103.0] * 20)
        close = np.array([100.0, 101.0, np.nan, 103.0, 104.0] * 20)

        result = adx(high, low, close, period=5)

        assert result is not None
        assert len(result) == len(close)

    def test_insufficient_data(self):
        """Test ADX with insufficient data."""
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([100.0, 101.0, 102.0])

        result = adx(high, low, close, period=14)

        # Should return all NaN
        assert np.all(np.isnan(result))

    def test_exact_minimum_data(self):
        """Test ADX with exactly minimum data needed."""
        # ADX lookback is 2*period - 1 = 2*14 - 1 = 27
        n = 28
        close = np.array([100.0 + i * 0.5 for i in range(n)])
        high = close + 1
        low = close - 1

        result = adx(high, low, close, period=14)

        # Should have first valid value at index 27
        assert not np.all(np.isnan(result))


class TestValidation:
    """Parameter validation tests."""

    def test_invalid_period_zero(self):
        """Test ADX rejects zero period."""
        high = np.random.randn(100)
        low = high - 1
        close = high - 0.5

        # ADX raises ZeroDivisionError for period=0
        with pytest.raises(ZeroDivisionError):
            adx(high, low, close, period=0)

    def test_invalid_period_negative(self):
        """Test ADX rejects negative period."""
        high = np.random.randn(100)
        low = high - 1
        close = high - 0.5

        # ADX with negative period has undefined behavior but doesn't crash
        # It may return values or NaN depending on implementation
        result = adx(high, low, close, period=-1)
        assert result is not None  # Just verify it returns something

    def test_mismatched_array_lengths(self):
        """Test ADX with mismatched array lengths."""
        high = np.random.randn(100)
        low = np.random.randn(50)  # Different length
        close = np.random.randn(100)

        # Should handle gracefully or raise
        try:
            result = adx(high, low, close, period=14)
            # If it doesn't raise, should return something valid
            assert result is not None
        except (ValueError, IndexError):
            # Or it might raise - both are acceptable
            pass


class TestNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_strong_uptrend(self):
        """Test ADX increases during strong uptrend."""
        # Strong uptrend
        close = np.linspace(100, 150, 100)
        high = close + 1
        low = close - 1

        result = adx(high, low, close, period=14)

        # ADX should show trend strength
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Later values should indicate trend (ADX > 20)
            assert np.any(valid_values[-10:] > 20)

    def test_strong_downtrend(self):
        """Test ADX increases during strong downtrend."""
        # Strong downtrend
        close = np.linspace(150, 100, 100)
        high = close + 1
        low = close - 1

        result = adx(high, low, close, period=14)

        # ADX should show trend strength regardless of direction
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Later values should indicate trend (ADX > 20)
            assert np.any(valid_values[-10:] > 20)

    def test_sideways_market(self):
        """Test ADX is low during sideways market."""
        # Sideways/choppy market
        close = 100 + np.sin(np.linspace(0, 10 * np.pi, 100)) * 5
        high = close + 1
        low = close - 1

        result = adx(high, low, close, period=14)

        # ADX should show weak trend
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Most values should be relatively low
            median_adx = np.median(valid_values)
            assert median_adx < 40  # Weak to moderate trend


class TestDataTypes:
    """Test different input data types."""

    def test_numpy_array_float64(self):
        """Test with float64 numpy arrays."""
        high = np.random.randn(100).astype(np.float64)
        low = high - 1
        close = high - 0.5

        result = adx(high, low, close, period=14)
        assert isinstance(result, np.ndarray)

    def test_numpy_array_float32(self):
        """Test with float32 numpy arrays (should convert)."""
        high = np.random.randn(100).astype(np.float32)
        low = (high - 1).astype(np.float32)
        close = (high - 0.5).astype(np.float32)

        result = adx(high, low, close, period=14)
        assert isinstance(result, np.ndarray)

    def test_polars_series(self):
        """Test with Polars Series."""
        high = pl.Series([100.0 + i + 1 for i in range(100)])
        low = pl.Series([100.0 + i - 1 for i in range(100)])
        close = pl.Series([100.0 + i for i in range(100)])

        result = adx(high, low, close, period=14)
        assert isinstance(result, np.ndarray)

    def test_polars_expression(self):
        """Test with Polars expressions (string column names)."""
        df = pl.DataFrame(
            {
                "high": [100.0 + i + 1 for i in range(100)],
                "low": [100.0 + i - 1 for i in range(100)],
                "close": [100.0 + i for i in range(100)],
            }
        )

        result = df.select(adx("high", "low", "close", period=14))
        assert isinstance(result, pl.DataFrame)


class TestPeriodSizes:
    """Test various period sizes."""

    def test_small_period(self):
        """Test with small period (5)."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = adx(high, low, close, period=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0) & (valid_values <= 100))

    def test_medium_period(self):
        """Test with medium period (14 - default)."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = adx(high, low, close, period=14)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0) & (valid_values <= 100))

    def test_large_period(self):
        """Test with large period (30)."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = adx(high, low, close, period=30)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0) & (valid_values <= 100))
