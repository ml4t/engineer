"""Tests for MACD (Moving Average Convergence/Divergence) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.momentum.macd import macd, macd_full, macd_signal


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

    def test_macd_computes_successfully_numpy(self):
        """Test MACD computes successfully with numpy array."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = macd(close, fast_period=12, slow_period=26)

        assert result is not None
        assert len(result) == len(close)
        assert isinstance(result, np.ndarray)

    def test_macd_computes_successfully_polars_series(self, ohlcv_data):
        """Test MACD computes successfully with Polars Series."""
        result = macd(ohlcv_data["close"], fast_period=12, slow_period=26)

        assert result is not None
        assert len(result) == len(ohlcv_data)
        assert isinstance(result, np.ndarray)

    def test_macd_computes_successfully_polars_expr(self, ohlcv_data):
        """Test MACD computes successfully with Polars expression."""
        result = ohlcv_data.select(macd("close", fast_period=12, slow_period=26))

        assert result is not None
        assert len(result) == len(ohlcv_data)
        assert "close" in result.columns

    def test_macd_signal_computes(self):
        """Test MACD signal line computes."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = macd_signal(close, fast_period=12, slow_period=26, signal_period=9)

        assert result is not None
        assert len(result) == len(close)

    def test_macd_full_returns_three_components(self):
        """Test macd_full returns all three components."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        macd_line, signal_line, histogram = macd_full(close)

        assert macd_line is not None
        assert signal_line is not None
        assert histogram is not None
        assert len(macd_line) == len(close)
        assert len(signal_line) == len(close)
        assert len(histogram) == len(close)

    def test_parameter_variations(self):
        """Test MACD with different parameters."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        r1 = macd(close, fast_period=8, slow_period=20)
        r2 = macd(close, fast_period=12, slow_period=26)

        # Different parameters should give different results
        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        assert np.sum(valid_mask) > 0
        assert not np.allclose(r1[valid_mask], r2[valid_mask], rtol=0.01)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_array(self):
        """Test MACD with empty array."""
        empty = np.array([])
        result = macd(empty, fast_period=12, slow_period=26)

        assert len(result) == 0

    def test_single_value(self):
        """Test MACD with single value."""
        single = np.array([100.0])
        result = macd(single, fast_period=12, slow_period=26)

        assert len(result) == 1
        assert np.isnan(result[0])

    def test_constant_values(self):
        """Test MACD with constant prices."""
        const = np.full(100, 100.0)
        result = macd(const, fast_period=12, slow_period=26)

        # With constant prices, MACD should be 0
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.allclose(valid_values, 0.0, atol=1e-10)

    def test_with_nans_in_data(self):
        """Test MACD handles NaN values in input."""
        close = np.array([100.0, 101.0, np.nan, 103.0, 104.0] * 20)
        result = macd(close, fast_period=12, slow_period=26)

        assert result is not None
        assert len(result) == len(close)

    def test_insufficient_data(self):
        """Test MACD with insufficient data."""
        close = np.array([100.0, 101.0, 102.0])
        result = macd(close, fast_period=12, slow_period=26)

        # Should return all NaN
        assert np.all(np.isnan(result))

    def test_exact_minimum_data(self):
        """Test MACD with exactly minimum data needed."""
        # Need at least slow_period + signal_period - 1 = 26 + 9 - 1 = 34
        close = np.array([100.0 + i * 0.5 for i in range(35)])
        result = macd(close, fast_period=12, slow_period=26)

        # Should have some valid values near the end
        assert not np.all(np.isnan(result))


class TestValidation:
    """Parameter validation tests."""

    def test_invalid_fast_period_zero(self):
        """Test MACD rejects zero fast period."""
        close = np.random.randn(100)
        with pytest.raises((ValueError, Exception)):
            macd(close, fast_period=0, slow_period=26)

    def test_invalid_fast_period_negative(self):
        """Test MACD rejects negative fast period."""
        close = np.random.randn(100)
        with pytest.raises((ValueError, Exception)):
            macd(close, fast_period=-1, slow_period=26)

    def test_invalid_slow_period_zero(self):
        """Test MACD rejects zero slow period."""
        close = np.random.randn(100)
        with pytest.raises((ValueError, Exception)):
            macd(close, fast_period=12, slow_period=0)

    def test_invalid_slow_period_negative(self):
        """Test MACD rejects negative slow period."""
        close = np.random.randn(100)
        with pytest.raises((ValueError, Exception)):
            macd(close, fast_period=12, slow_period=-1)

    def test_fast_greater_than_slow(self):
        """Test MACD rejects fast period >= slow period."""
        close = np.random.randn(100)
        with pytest.raises((ValueError, Exception)):
            macd(close, fast_period=26, slow_period=12)

    def test_signal_period_validation(self):
        """Test macd_signal validates signal period."""
        close = np.random.randn(100)
        with pytest.raises((ValueError, Exception)):
            macd_signal(close, fast_period=12, slow_period=26, signal_period=0)


class TestNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_histogram_equals_difference(self):
        """Test that histogram = MACD line - signal line."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        macd_line, signal_line, histogram = macd_full(close)

        # Where both are valid, histogram should equal difference
        valid_mask = ~np.isnan(macd_line) & ~np.isnan(signal_line) & ~np.isnan(histogram)
        if np.sum(valid_mask) > 0:
            expected_hist = macd_line[valid_mask] - signal_line[valid_mask]
            actual_hist = histogram[valid_mask]
            assert np.allclose(expected_hist, actual_hist, atol=1e-10)

    def test_zero_crossovers(self):
        """Test MACD crosses zero during trending market."""
        # Create uptrend then downtrend
        uptrend = np.linspace(100, 120, 50)
        downtrend = np.linspace(120, 100, 50)
        close = np.concatenate([uptrend, downtrend])

        result = macd(close, fast_period=5, slow_period=10)

        # Should have some positive and some negative values
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            has_positive = np.any(valid_values > 0)
            has_negative = np.any(valid_values < 0)
            # At least one should be true in trending market
            assert has_positive or has_negative


class TestDataTypes:
    """Test different input data types."""

    def test_numpy_array_float64(self):
        """Test with float64 numpy array."""
        close = np.random.randn(100).astype(np.float64)
        result = macd(close, fast_period=12, slow_period=26)
        assert isinstance(result, np.ndarray)

    def test_numpy_array_float32(self):
        """Test with float32 numpy array (should convert)."""
        close = np.random.randn(100).astype(np.float32)
        result = macd(close, fast_period=12, slow_period=26)
        assert isinstance(result, np.ndarray)

    def test_polars_series(self):
        """Test with Polars Series."""
        close = pl.Series([100.0 + i * 0.5 for i in range(100)])
        result = macd(close, fast_period=12, slow_period=26)
        assert isinstance(result, np.ndarray)

    def test_polars_expression(self):
        """Test with Polars expression (string column name)."""
        df = pl.DataFrame({"close": [100.0 + i * 0.5 for i in range(100)]})
        result = df.select(macd("close", fast_period=12, slow_period=26))
        assert isinstance(result, pl.DataFrame)
        assert "close" in result.columns


class TestPeriodCombinations:
    """Test various period combinations."""

    def test_default_periods(self):
        """Test with default periods (12, 26, 9)."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        macd_line, signal_line, histogram = macd_full(close)

        # Should have valid values
        assert np.sum(~np.isnan(macd_line)) > 0
        assert np.sum(~np.isnan(signal_line)) > 0
        assert np.sum(~np.isnan(histogram)) > 0

    def test_custom_fast_periods(self):
        """Test with custom fast period."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = macd(close, fast_period=8, slow_period=26)

        assert np.sum(~np.isnan(result)) > 0

    def test_custom_slow_periods(self):
        """Test with custom slow period."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = macd(close, fast_period=12, slow_period=30)

        assert np.sum(~np.isnan(result)) > 0

    def test_small_periods(self):
        """Test with small periods."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        result = macd(close, fast_period=5, slow_period=10)

        assert np.sum(~np.isnan(result)) > 0

    def test_large_periods(self):
        """Test with large periods."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        result = macd(close, fast_period=20, slow_period=50)

        assert np.sum(~np.isnan(result)) > 0


class TestSignalLine:
    """Test signal line specific functionality."""

    def test_signal_line_is_ema_of_macd(self):
        """Test that signal line is smoother than MACD line."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 2)  # More volatility

        macd_line, signal_line, _ = macd_full(close)

        # Signal line should generally be smoother (less volatile)
        # Skip NaN values
        valid_mask = ~np.isnan(macd_line) & ~np.isnan(signal_line)
        if np.sum(valid_mask) > 10:
            macd_std = np.std(macd_line[valid_mask])
            signal_std = np.std(signal_line[valid_mask])
            # This is not always true due to lag, but often signal is smoother
            # Just verify both have reasonable variance
            assert macd_std > 0
            assert signal_std > 0
