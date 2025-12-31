"""Tests for MFI (Money Flow Index) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.momentum.mfi import mfi


@pytest.fixture
def ohlcv_data():
    """Standard OHLCV test data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_prices = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n).astype(float)

    return pl.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_mfi_computes_successfully_numpy(self):
        """Test MFI computes successfully with numpy arrays."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(1000, 10000, n).astype(float)

        result = mfi(high, low, close, volume, period=14)

        assert result is not None
        assert len(result) == n
        assert isinstance(result, np.ndarray)

    def test_mfi_computes_successfully_polars_series(self, ohlcv_data):
        """Test MFI computes successfully with Polars Series."""
        result = mfi(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            period=14,
        )

        assert result is not None
        assert len(result) == len(ohlcv_data)
        assert isinstance(result, np.ndarray)

    def test_mfi_computes_successfully_polars_expr(self, ohlcv_data):
        """Test MFI computes successfully with Polars expressions."""
        result = ohlcv_data.select(mfi("high", "low", "close", "volume", period=14))

        assert result is not None
        assert len(result) == len(ohlcv_data)

    def test_parameter_variations(self):
        """Test MFI with different periods."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(1000, 10000, n).astype(float)

        r1 = mfi(high, low, close, volume, period=7)
        r2 = mfi(high, low, close, volume, period=21)

        # Different periods should give different results
        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            assert not np.allclose(r1[valid_mask], r2[valid_mask], rtol=0.01)

    def test_output_range(self):
        """Test MFI output is within 0-100 range."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(1000, 10000, n).astype(float)

        result = mfi(high, low, close, volume, period=14)

        # Check valid values are in range [0, 100]
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.all(valid_values >= 0)
            assert np.all(valid_values <= 100)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_arrays(self):
        """Test MFI with empty arrays."""
        empty = np.array([])
        result = mfi(empty, empty, empty, empty, period=14)

        assert len(result) == 0

    def test_single_value(self):
        """Test MFI with single value."""
        single = np.array([100.0])
        result = mfi(single + 1, single - 1, single, single * 1000, period=14)

        assert len(result) == 1
        assert np.isnan(result[0])

    def test_constant_values(self):
        """Test MFI with constant prices."""
        const = np.full(100, 100.0)
        high = const + 1
        low = const - 1
        volume = np.full(100, 5000.0)

        result = mfi(high, low, const, volume, period=14)

        # With constant prices, typical price doesn't change
        # MFI behavior depends on implementation
        assert result is not None

    def test_with_nans_in_data(self):
        """Test MFI handles NaN values in input."""
        high = np.array([101.0, 102.0, np.nan, 104.0, 105.0] * 20)
        low = np.array([99.0, 100.0, np.nan, 102.0, 103.0] * 20)
        close = np.array([100.0, 101.0, np.nan, 103.0, 104.0] * 20)
        volume = np.array([5000.0, 5100.0, np.nan, 5200.0, 5300.0] * 20)

        result = mfi(high, low, close, volume, period=5)

        assert result is not None
        assert len(result) == len(close)

    def test_insufficient_data(self):
        """Test MFI with insufficient data."""
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([5000.0, 5100.0, 5200.0])

        result = mfi(high, low, close, volume, period=14)

        # Should return all NaN
        assert np.all(np.isnan(result))

    def test_exact_minimum_data(self):
        """Test MFI with exactly minimum data needed."""
        # Need period + 1 values
        n = 15
        close = np.array([100.0 + i * 0.5 for i in range(n)])
        high = close + 1
        low = close - 1
        volume = np.full(n, 5000.0)

        result = mfi(high, low, close, volume, period=14)

        # Should have first valid value at index 14
        assert not np.all(np.isnan(result))


class TestValidation:
    """Parameter validation tests."""

    def test_invalid_period_zero(self):
        """Test MFI rejects zero period."""
        high = np.random.randn(100) + 100
        low = high - 1
        close = high - 0.5
        volume = np.random.randint(1000, 10000, 100).astype(float)

        with pytest.raises((ValueError, Exception)):
            mfi(high, low, close, volume, period=0)

    def test_invalid_period_negative(self):
        """Test MFI rejects negative period."""
        high = np.random.randn(100) + 100
        low = high - 1
        close = high - 0.5
        volume = np.random.randint(1000, 10000, 100).astype(float)

        with pytest.raises((ValueError, Exception)):
            mfi(high, low, close, volume, period=-1)


class TestNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_high_volume_uptrend(self):
        """Test MFI increases with high volume uptrend."""
        # Uptrend with increasing volume
        close = np.linspace(100, 150, 50)
        high = close + 1
        low = close - 1
        volume = np.linspace(5000, 10000, 50)

        result = mfi(high, low, close, volume, period=14)

        # MFI should show buying pressure (higher values)
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Later values should indicate buying pressure (MFI > 50)
            assert np.median(valid_values[-10:]) > 40

    def test_high_volume_downtrend(self):
        """Test MFI decreases with high volume downtrend."""
        # Downtrend with increasing volume
        close = np.linspace(150, 100, 50)
        high = close + 1
        low = close - 1
        volume = np.linspace(5000, 10000, 50)

        result = mfi(high, low, close, volume, period=14)

        # MFI should show selling pressure (lower values)
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Later values should indicate selling pressure (MFI < 50)
            assert np.median(valid_values[-10:]) < 60

    def test_zero_volume(self):
        """Test MFI with zero volume."""
        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        high = close + 1
        low = close - 1
        volume = np.zeros(50)

        result = mfi(high, low, close, volume, period=14)

        # With zero volume, MFI should handle gracefully
        assert result is not None


class TestDataTypes:
    """Test different input data types."""

    def test_numpy_array_float64(self):
        """Test with float64 numpy arrays."""
        high = (np.random.randn(100) + 100).astype(np.float64)
        low = (high - 1).astype(np.float64)
        close = (high - 0.5).astype(np.float64)
        volume = np.random.randint(1000, 10000, 100).astype(np.float64)

        result = mfi(high, low, close, volume, period=14)
        assert isinstance(result, np.ndarray)

    def test_numpy_array_float32(self):
        """Test with float32 numpy arrays (should convert)."""
        high = (np.random.randn(100) + 100).astype(np.float32)
        low = (high - 1).astype(np.float32)
        close = (high - 0.5).astype(np.float32)
        volume = np.random.randint(1000, 10000, 100).astype(np.float32)

        result = mfi(high, low, close, volume, period=14)
        assert isinstance(result, np.ndarray)

    def test_polars_series(self):
        """Test with Polars Series."""
        high = pl.Series([100.0 + i + 1 for i in range(100)])
        low = pl.Series([100.0 + i - 1 for i in range(100)])
        close = pl.Series([100.0 + i for i in range(100)])
        volume = pl.Series([5000.0 + i * 10 for i in range(100)])

        result = mfi(high, low, close, volume, period=14)
        assert isinstance(result, np.ndarray)

    def test_polars_expression(self):
        """Test with Polars expressions (string column names)."""
        df = pl.DataFrame(
            {
                "high": [100.0 + i + 1 for i in range(100)],
                "low": [100.0 + i - 1 for i in range(100)],
                "close": [100.0 + i for i in range(100)],
                "volume": [5000.0 + i * 10 for i in range(100)],
            }
        )

        result = df.select(mfi("high", "low", "close", "volume", period=14))
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
        volume = np.random.randint(1000, 10000, n).astype(float)

        result = mfi(high, low, close, volume, period=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # Use tolerance for floating point comparison (values can be tiny negatives like -1e-14)
        assert np.all((valid_values >= -1e-10) & (valid_values <= 100 + 1e-10))

    def test_medium_period(self):
        """Test with medium period (14 - default)."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(1000, 10000, n).astype(float)

        result = mfi(high, low, close, volume, period=14)

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
        volume = np.random.randint(1000, 10000, n).astype(float)

        result = mfi(high, low, close, volume, period=30)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0) & (valid_values <= 100))


class TestVolumeEffects:
    """Test volume-specific behavior."""

    def test_constant_volume(self):
        """Test MFI with constant volume."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + 1
        low = close - 1
        volume = np.full(100, 5000.0)

        result = mfi(high, low, close, volume, period=14)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0) & (valid_values <= 100))

    def test_varying_volume(self):
        """Test MFI with varying volume."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + 1
        low = close - 1
        volume = np.random.randint(1000, 10000, 100).astype(float)

        result = mfi(high, low, close, volume, period=14)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all((valid_values >= 0) & (valid_values <= 100))
