"""Tests targeting uncovered lines in directional.py for coverage boost."""

import numpy as np

from ml4t.engineer.features.momentum.directional import (
    calculate_directional_movement_nb,
    dx_numba,
    minus_di_numba,
    plus_di_numba,
    wilders_smoothing_nb,
)


class TestDirectionalMovementNumba:
    """Tests for calculate_directional_movement_nb."""

    def test_basic_calculation(self):
        """Test basic directional movement calculation."""
        high = np.array([10.0, 11.0, 12.0, 11.5, 13.0])
        low = np.array([9.0, 10.0, 11.0, 10.5, 12.0])

        plus_dm, minus_dm = calculate_directional_movement_nb(high, low)

        # First value should be 0 (no previous)
        assert plus_dm[0] == 0.0
        assert minus_dm[0] == 0.0

        # Check some movement detected
        assert np.sum(plus_dm) > 0 or np.sum(minus_dm) > 0

    def test_both_branches(self):
        """Test both plus and minus DM branches."""
        # Create data with clear upward movement
        high = np.array([10.0, 12.0, 14.0])
        low = np.array([9.0, 11.0, 13.0])

        plus_dm, minus_dm = calculate_directional_movement_nb(high, low)
        assert np.any(plus_dm > 0)

        # Create data with clear downward movement
        high = np.array([14.0, 13.0, 12.0])
        low = np.array([13.0, 11.0, 10.0])

        plus_dm, minus_dm = calculate_directional_movement_nb(high, low)
        assert np.any(minus_dm > 0)


class TestWildersSmoothing:
    """Tests for wilders_smoothing_nb."""

    def test_basic_smoothing(self):
        """Test basic Wilder's smoothing."""
        values = np.arange(30, dtype=np.float64)
        result = wilders_smoothing_nb(values, period=14)

        # First (period-1) values should be NaN
        assert all(np.isnan(result[:13]))

        # First output at index period-1
        assert not np.isnan(result[13])

    def test_insufficient_data(self):
        """Test with insufficient data."""
        values = np.array([1.0, 2.0, 3.0])
        result = wilders_smoothing_nb(values, period=14)

        assert all(np.isnan(result))

    def test_with_nan_values(self):
        """Test handling of NaN values in input."""
        values = np.array([1.0, np.nan, 3.0, 4.0, 5.0] * 5, dtype=np.float64)
        result = wilders_smoothing_nb(values, period=5)

        # Should still produce output
        assert len(result) == len(values)


class TestPlusDINumba:
    """Tests for plus_di_numba."""

    def test_basic_calculation(self):
        """Test basic Plus DI calculation."""
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = plus_di_numba(high, low, close, timeperiod=14)

        # First 14 values should be NaN
        assert all(np.isnan(result[:14]))

        # Values after period should be valid (0-100 range)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert all(v >= 0 for v in valid)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])

        result = plus_di_numba(high, low, close, timeperiod=14)
        assert all(np.isnan(result))

    def test_zero_tr_handling(self):
        """Test handling when TR is zero."""
        # Constant price scenario
        high = np.ones(30) * 100
        low = np.ones(30) * 100
        close = np.ones(30) * 100

        result = plus_di_numba(high, low, close, timeperiod=14)

        # Should return 0 when TR is 0
        valid = result[~np.isnan(result)]
        assert all(v == 0 for v in valid)


class TestMinusDINumba:
    """Tests for minus_di_numba."""

    def test_basic_calculation(self):
        """Test basic Minus DI calculation."""
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = minus_di_numba(high, low, close, timeperiod=14)

        # First 14 values should be NaN
        assert all(np.isnan(result[:14]))

        # Values after period should be valid
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestDXNumba:
    """Tests for dx_numba."""

    def test_basic_calculation(self):
        """Test basic DX calculation."""
        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = dx_numba(high, low, close, timeperiod=14)

        # Should have valid values after warmup
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

        # DX should be in 0-100 range
        assert all(0 <= v <= 100 for v in valid)

    def test_insufficient_data(self):
        """Test with insufficient data."""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])

        result = dx_numba(high, low, close, timeperiod=14)
        assert all(np.isnan(result))
