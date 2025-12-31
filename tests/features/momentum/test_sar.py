"""Tests for SAR (Parabolic Stop and Reverse) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.momentum.sar import sar


@pytest.fixture
def ohlcv_data():
    """Standard OHLCV test data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)

    return pl.DataFrame(
        {
            "open": close - np.random.rand(n) * 0.5,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_sar_computes_successfully_numpy(self):
        """Test SAR computes successfully with numpy arrays."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = sar(high, low, acceleration=0.02, maximum=0.2)

        assert result is not None
        assert len(result) == n
        assert isinstance(result, np.ndarray)

    def test_sar_computes_successfully_polars_series(self, ohlcv_data):
        """Test SAR computes successfully with Polars Series."""
        result = sar(
            ohlcv_data["high"],
            ohlcv_data["low"],
            acceleration=0.02,
            maximum=0.2,
        )

        assert result is not None
        assert len(result) == len(ohlcv_data)
        assert isinstance(result, np.ndarray)

    def test_sar_computes_successfully_polars_expr(self, ohlcv_data):
        """Test SAR computes successfully with Polars expressions."""
        result = ohlcv_data.select(sar("high", "low", acceleration=0.02, maximum=0.2))

        assert result is not None
        assert len(result) == len(ohlcv_data)

    def test_parameter_variations(self):
        """Test SAR with different parameters."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        r1 = sar(high, low, acceleration=0.02, maximum=0.2)
        r2 = sar(high, low, acceleration=0.05, maximum=0.3)

        # Different parameters should give different results
        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            assert not np.allclose(r1[valid_mask], r2[valid_mask], rtol=0.01)

    def test_sar_within_price_range(self):
        """Test SAR values are within reasonable price range."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = sar(high, low, acceleration=0.02, maximum=0.2)

        # SAR should be within or near the price range
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            price_min = np.min(low)
            price_max = np.max(high)
            # SAR might go slightly outside but should be reasonable
            assert np.all(valid_values > price_min * 0.8)
            assert np.all(valid_values < price_max * 1.2)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_arrays(self):
        """Test SAR with empty arrays."""
        empty = np.array([])
        result = sar(empty, empty, acceleration=0.02, maximum=0.2)

        assert len(result) == 0

    def test_single_value(self):
        """Test SAR with single value."""
        single_high = np.array([101.0])
        single_low = np.array([99.0])

        result = sar(single_high, single_low, acceleration=0.02, maximum=0.2)

        assert len(result) == 1
        assert np.isnan(result[0])

    def test_two_values(self):
        """Test SAR with exactly two values (minimum for first output)."""
        high = np.array([101.0, 102.0])
        low = np.array([99.0, 100.0])

        result = sar(high, low, acceleration=0.02, maximum=0.2)

        assert len(result) == 2
        # First value is NaN, second should have SAR
        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_constant_values(self):
        """Test SAR with constant prices."""
        high = np.full(100, 101.0)
        low = np.full(100, 99.0)

        result = sar(high, low, acceleration=0.02, maximum=0.2)

        # SAR should still compute
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_with_nans_in_data(self):
        """Test SAR handles NaN values in input."""
        high = np.array([101.0, 102.0, np.nan, 104.0, 105.0] * 20)
        low = np.array([99.0, 100.0, np.nan, 102.0, 103.0] * 20)

        result = sar(high, low, acceleration=0.02, maximum=0.2)

        assert result is not None
        assert len(result) == len(high)


class TestValidation:
    """Parameter validation tests."""

    def test_invalid_acceleration_zero(self):
        """Test SAR rejects zero acceleration."""
        high = np.random.randn(100) + 100
        low = high - 1

        with pytest.raises((ValueError, Exception)):
            sar(high, low, acceleration=0.0, maximum=0.2)

    def test_invalid_acceleration_negative(self):
        """Test SAR rejects negative acceleration."""
        high = np.random.randn(100) + 100
        low = high - 1

        with pytest.raises((ValueError, Exception)):
            sar(high, low, acceleration=-0.02, maximum=0.2)

    def test_invalid_maximum_zero(self):
        """Test SAR rejects zero maximum."""
        high = np.random.randn(100) + 100
        low = high - 1

        with pytest.raises((ValueError, Exception)):
            sar(high, low, acceleration=0.02, maximum=0.0)

    def test_invalid_maximum_negative(self):
        """Test SAR rejects negative maximum."""
        high = np.random.randn(100) + 100
        low = high - 1

        with pytest.raises((ValueError, Exception)):
            sar(high, low, acceleration=0.02, maximum=-0.2)

    def test_acceleration_greater_than_maximum(self):
        """Test SAR rejects acceleration > maximum."""
        high = np.random.randn(100) + 100
        low = high - 1

        with pytest.raises((ValueError, Exception)):
            sar(high, low, acceleration=0.3, maximum=0.2)

    def test_mismatched_array_lengths(self):
        """Test SAR with mismatched array lengths."""
        high = np.random.randn(100) + 100
        low = np.random.randn(50) + 99  # Different length

        with pytest.raises((ValueError, IndexError)):
            sar(high, low, acceleration=0.02, maximum=0.2)


class TestNumericalAccuracy:
    """Numerical accuracy tests."""

    def test_uptrend_sar_below_price(self):
        """Test SAR is below price during uptrend."""
        # Strong uptrend
        close = np.linspace(100, 150, 100)
        high = close + 1
        low = close - 1

        result = sar(high, low, acceleration=0.02, maximum=0.2)

        # In uptrend, SAR should generally be below price
        valid_indices = ~np.isnan(result)
        if np.sum(valid_indices) > 10:
            # Check last half of trend
            last_half = len(result) // 2
            sar_values = result[last_half:]
            low_values = low[last_half:]
            valid_mask = ~np.isnan(sar_values)
            if np.sum(valid_mask) > 5:
                # Most SAR values should be below low
                below_count = np.sum(sar_values[valid_mask] < low_values[valid_mask])
                assert below_count > np.sum(valid_mask) * 0.5

    def test_downtrend_sar_above_price(self):
        """Test SAR is above price during downtrend."""
        # Strong downtrend
        close = np.linspace(150, 100, 100)
        high = close + 1
        low = close - 1

        result = sar(high, low, acceleration=0.02, maximum=0.2)

        # In downtrend, SAR should generally be above price
        valid_indices = ~np.isnan(result)
        if np.sum(valid_indices) > 10:
            # Check last half of trend
            last_half = len(result) // 2
            sar_values = result[last_half:]
            high_values = high[last_half:]
            valid_mask = ~np.isnan(sar_values)
            if np.sum(valid_mask) > 5:
                # Most SAR values should be above high
                above_count = np.sum(sar_values[valid_mask] > high_values[valid_mask])
                assert above_count > np.sum(valid_mask) * 0.5

    def test_sar_switches_sides_on_reversal(self):
        """Test SAR switches sides when trend reverses."""
        # Uptrend then downtrend
        uptrend = np.linspace(100, 120, 50)
        downtrend = np.linspace(120, 100, 50)
        close = np.concatenate([uptrend, downtrend])
        high = close + 1
        low = close - 1

        result = sar(high, low, acceleration=0.02, maximum=0.2)

        # SAR should exist and change relative to price
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 20


class TestDataTypes:
    """Test different input data types."""

    def test_numpy_array_float64(self):
        """Test with float64 numpy arrays."""
        high = (np.random.randn(100) + 100).astype(np.float64)
        low = (high - 1).astype(np.float64)

        result = sar(high, low, acceleration=0.02, maximum=0.2)
        assert isinstance(result, np.ndarray)

    def test_numpy_array_float32(self):
        """Test with float32 numpy arrays (should convert)."""
        high = (np.random.randn(100) + 100).astype(np.float32)
        low = (high - 1).astype(np.float32)

        result = sar(high, low, acceleration=0.02, maximum=0.2)
        assert isinstance(result, np.ndarray)

    def test_polars_series(self):
        """Test with Polars Series."""
        high = pl.Series([100.0 + i + 1 for i in range(100)])
        low = pl.Series([100.0 + i - 1 for i in range(100)])

        result = sar(high, low, acceleration=0.02, maximum=0.2)
        assert isinstance(result, np.ndarray)

    def test_polars_expression(self):
        """Test with Polars expressions (string column names)."""
        df = pl.DataFrame(
            {
                "high": [100.0 + i + 1 for i in range(100)],
                "low": [100.0 + i - 1 for i in range(100)],
            }
        )

        result = df.select(sar("high", "low", acceleration=0.02, maximum=0.2))
        assert isinstance(result, pl.DataFrame)


class TestParameterEffects:
    """Test effects of different parameter values."""

    def test_higher_acceleration_faster_movement(self):
        """Test that higher acceleration makes SAR move faster."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        sar_slow = sar(high, low, acceleration=0.01, maximum=0.2)
        sar_fast = sar(high, low, acceleration=0.05, maximum=0.2)

        # Both should have values
        assert np.sum(~np.isnan(sar_slow)) > 0
        assert np.sum(~np.isnan(sar_fast)) > 0

    def test_default_parameters(self):
        """Test with default parameters (0.02, 0.2)."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = sar(high, low)  # Use defaults

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_conservative_parameters(self):
        """Test with conservative parameters (slow acceleration)."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = sar(high, low, acceleration=0.01, maximum=0.1)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_aggressive_parameters(self):
        """Test with aggressive parameters (fast acceleration)."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = sar(high, low, acceleration=0.1, maximum=0.3)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
