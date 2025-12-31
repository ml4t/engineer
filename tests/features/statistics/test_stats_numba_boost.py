"""
Comprehensive tests for Numba implementations in statistics and price_transform modules.

This test suite directly tests the Numba functions to ensure maximum code coverage.
It covers edge cases, insufficient data handling, NaN propagation, and validates
the first valid output index.

Target coverage improvements:
- statistics/stddev.py: 25 missing lines (42% coverage)
- statistics/var.py: 24 missing lines (45% coverage)
- statistics/avgdev.py: 14 missing lines (59% coverage)
- statistics/linearreg.py: 19 missing lines (~50% coverage)
- statistics/linearreg_slope.py: 20 missing lines (~50% coverage)
- statistics/linearreg_intercept.py: 19 missing lines (~50% coverage)
- statistics/linearreg_angle.py: 20 missing lines (~50% coverage)
- statistics/tsf.py: 30 missing lines (41% coverage)
- price_transform/midprice.py: 38 missing lines (48.6% coverage)
"""

import numpy as np

from ml4t.engineer.features.price_transform.midprice import midprice_numba, midprice_numpy
from ml4t.engineer.features.statistics.avgdev import avgdev_numba
from ml4t.engineer.features.statistics.linearreg import linearreg_numba
from ml4t.engineer.features.statistics.linearreg_angle import linearreg_angle_numba
from ml4t.engineer.features.statistics.linearreg_intercept import linearreg_intercept_numba
from ml4t.engineer.features.statistics.linearreg_slope import linearreg_slope_numba
from ml4t.engineer.features.statistics.stddev import stddev_numba
from ml4t.engineer.features.statistics.tsf import tsf_numba
from ml4t.engineer.features.statistics.var import var_numba


class TestStddevNumba:
    """Test stddev_numba() Numba implementation directly."""

    def test_basic_computation(self):
        """Basic test with valid data - verify output shape and NaN positions."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = stddev_numba(close, period=5)

        assert len(result) == len(close)
        # First 4 values should be NaN (period-1)
        assert np.all(np.isnan(result[:4]))
        # Fifth value onwards should have valid data
        assert not np.isnan(result[4])
        assert result[4] > 0  # Standard deviation should be positive

    def test_insufficient_data(self):
        """Test when data length is less than period."""
        close = np.array([1.0, 2.0])
        result = stddev_numba(close, period=5)

        # All values should be NaN
        assert np.all(np.isnan(result))

    def test_period_equals_length(self):
        """Test when period exactly equals data length."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = stddev_numba(close, period=5)

        # First 4 should be NaN, last should be valid
        assert np.all(np.isnan(result[:4]))
        assert not np.isnan(result[4])

    def test_period_one(self):
        """Test edge case with period=1."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = stddev_numba(close, period=1)

        # Period < 2 should return all NaN (std dev needs at least 2 points)
        assert np.all(np.isnan(result))

    def test_period_two(self):
        """Test edge case with period=2."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = stddev_numba(close, period=2)

        # First value should be NaN (period-1)
        assert np.isnan(result[0])
        # Remaining should be valid
        assert not np.isnan(result[1])

    def test_nan_in_input(self):
        """Test NaN handling in input data."""
        close = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = stddev_numba(close, period=5)

        # Windows containing NaN should produce NaN
        # Window indices 0-4 contains NaN at index 2
        assert np.isnan(result[4])
        # Window indices 5-9 should not contain NaN
        assert not np.isnan(result[9])

    def test_nbdev_scaling(self):
        """Test nbdev scaling parameter."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result_1 = stddev_numba(close, period=5, nbdev=1.0)
        result_2 = stddev_numba(close, period=5, nbdev=2.0)

        # With nbdev=2, results should be exactly 2x
        assert np.allclose(result_2[4:], result_1[4:] * 2.0, rtol=1e-9)

    def test_ddof_parameter(self):
        """Test degrees of freedom parameter."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result_0 = stddev_numba(close, period=5, ddof=0)  # Population std
        result_1 = stddev_numba(close, period=5, ddof=1)  # Sample std

        # Sample std should be larger than population std
        assert result_1[4] > result_0[4]


class TestVarNumba:
    """Test var_numba() Numba implementation directly."""

    def test_basic_computation(self):
        """Basic test with valid data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = var_numba(close, timeperiod=5)

        assert len(result) == len(close)
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        # Fifth value onwards should be valid
        assert not np.isnan(result[4])
        assert result[4] >= 0  # Variance is always non-negative

    def test_insufficient_data(self):
        """Test when data length is less than period."""
        close = np.array([1.0, 2.0])
        result = var_numba(close, timeperiod=5)

        assert np.all(np.isnan(result))

    def test_period_one(self):
        """Test with period=1."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = var_numba(close, timeperiod=1)

        # With period=1, variance should be 0 (single point has no variance)
        assert np.all(result[0:] == 0.0)

    def test_period_two(self):
        """Test with period=2."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = var_numba(close, timeperiod=2)

        # First value should be NaN, rest should be valid
        assert np.isnan(result[0])
        assert not np.isnan(result[1])
        assert result[1] >= 0

    def test_constant_input(self):
        """Test with constant input - variance should be zero."""
        close = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = var_numba(close, timeperiod=3)

        # All valid windows should have zero variance (within floating-point tolerance)
        assert np.isclose(result[2], 0.0, atol=1e-10)
        assert np.isclose(result[3], 0.0, atol=1e-10)
        assert np.isclose(result[4], 0.0, atol=1e-10)

    def test_variance_vs_stddev(self):
        """Verify variance is stddev squared."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        var_result = var_numba(close, timeperiod=5)
        stddev_result = stddev_numba(close, period=5, ddof=0)

        # Variance should equal stddev squared
        assert np.allclose(var_result[4:], stddev_result[4:] ** 2, rtol=1e-9)


class TestAvgdevNumba:
    """Test avgdev_numba() Numba implementation directly."""

    def test_basic_computation(self):
        """Basic test with valid data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = avgdev_numba(close, timeperiod=5)

        assert len(result) == len(close)
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        # Fifth value onwards should be valid
        assert not np.isnan(result[4])
        assert result[4] >= 0  # Average deviation is always non-negative

    def test_insufficient_data(self):
        """Test when data length is less than period."""
        close = np.array([1.0, 2.0])
        result = avgdev_numba(close, timeperiod=5)

        assert np.all(np.isnan(result))

    def test_period_one(self):
        """Test with period=1."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = avgdev_numba(close, timeperiod=1)

        # With period=1, average deviation should be 0
        assert np.all(result[0:] == 0.0)

    def test_period_two(self):
        """Test with period=2."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = avgdev_numba(close, timeperiod=2)

        # First value should be NaN, rest should be valid
        assert np.isnan(result[0])
        assert not np.isnan(result[1])
        assert result[1] >= 0

    def test_constant_input(self):
        """Test with constant input - average deviation should be zero."""
        close = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = avgdev_numba(close, timeperiod=3)

        # All valid windows should have zero average deviation
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert result[4] == 0.0

    def test_symmetric_data(self):
        """Test with symmetric data around mean."""
        close = np.array([1.0, 3.0, 5.0, 3.0, 1.0])
        result = avgdev_numba(close, timeperiod=5)

        # For symmetric data, average deviation should be computable
        assert not np.isnan(result[4])
        assert result[4] > 0


class TestLinearregNumba:
    """Test linearreg_numba() Numba implementation directly."""

    def test_basic_computation(self):
        """Basic test with valid data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = linearreg_numba(close, timeperiod=5)

        assert len(result) == len(close)
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        # Fifth value onwards should be valid
        assert not np.isnan(result[4])

    def test_insufficient_data(self):
        """Test when data length is less than period."""
        close = np.array([1.0, 2.0])
        result = linearreg_numba(close, timeperiod=5)

        assert np.all(np.isnan(result))

    def test_period_two(self):
        """Test with period=2 (minimum for linear regression)."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = linearreg_numba(close, timeperiod=2)

        # First value should be NaN, rest should be valid
        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_linear_trend(self):
        """Test with perfectly linear data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = linearreg_numba(close, timeperiod=5)

        # For perfect linear trend, regression should match last point
        assert np.isclose(result[4], 5.0, rtol=1e-9)

    def test_constant_input(self):
        """Test with constant input."""
        close = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = linearreg_numba(close, timeperiod=3)

        # For constant input, regression should equal the constant
        assert np.allclose(result[2:], 5.0, rtol=1e-9)

    def test_first_valid_index(self):
        """Verify first valid output is at index period-1."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        period = 3
        result = linearreg_numba(close, timeperiod=period)

        # First period-1 values should be NaN
        assert np.all(np.isnan(result[: period - 1]))
        # Value at index period-1 should be valid
        assert not np.isnan(result[period - 1])


class TestLinearregSlopeNumba:
    """Test linearreg_slope_numba() Numba implementation directly."""

    def test_basic_computation(self):
        """Basic test with valid data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = linearreg_slope_numba(close, timeperiod=5)

        assert len(result) == len(close)
        assert np.all(np.isnan(result[:4]))
        assert not np.isnan(result[4])

    def test_insufficient_data(self):
        """Test when data length is less than period."""
        close = np.array([1.0, 2.0])
        result = linearreg_slope_numba(close, timeperiod=5)

        assert np.all(np.isnan(result))

    def test_period_two(self):
        """Test with period=2."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = linearreg_slope_numba(close, timeperiod=2)

        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_linear_trend(self):
        """Test with perfectly linear data - slope should be constant."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = linearreg_slope_numba(close, timeperiod=5)

        # For perfect linear trend with slope 1, result should be 1
        assert np.isclose(result[4], 1.0, rtol=1e-9)

    def test_constant_input(self):
        """Test with constant input - slope should be zero."""
        close = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = linearreg_slope_numba(close, timeperiod=3)

        # For constant input, slope should be 0
        assert np.allclose(result[2:], 0.0, atol=1e-9)

    def test_negative_slope(self):
        """Test with decreasing data - slope should be negative."""
        close = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
        result = linearreg_slope_numba(close, timeperiod=5)

        # Slope should be negative for decreasing trend
        assert result[4] < 0


class TestLinearregInterceptNumba:
    """Test linearreg_intercept_numba() Numba implementation directly."""

    def test_basic_computation(self):
        """Basic test with valid data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = linearreg_intercept_numba(close, timeperiod=5)

        assert len(result) == len(close)
        assert np.all(np.isnan(result[:4]))
        assert not np.isnan(result[4])

    def test_insufficient_data(self):
        """Test when data length is less than period."""
        close = np.array([1.0, 2.0])
        result = linearreg_intercept_numba(close, timeperiod=5)

        assert np.all(np.isnan(result))

    def test_period_two(self):
        """Test with period=2."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = linearreg_intercept_numba(close, timeperiod=2)

        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_constant_input(self):
        """Test with constant input - intercept should equal constant."""
        close = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = linearreg_intercept_numba(close, timeperiod=3)

        # For constant input, intercept should be the constant
        assert np.allclose(result[2:], 5.0, rtol=1e-9)

    def test_relationship_with_slope(self):
        """Verify intercept and slope form the regression line."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        intercept = linearreg_intercept_numba(close, timeperiod=5)
        slope = linearreg_slope_numba(close, timeperiod=5)
        linearreg = linearreg_numba(close, timeperiod=5)

        # linearreg = intercept + slope * (timeperiod-1)
        expected = intercept[4] + slope[4] * 4
        assert np.isclose(expected, linearreg[4], rtol=1e-9)


class TestLinearregAngleNumba:
    """Test linearreg_angle_numba() Numba implementation directly."""

    def test_basic_computation(self):
        """Basic test with valid data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = linearreg_angle_numba(close, timeperiod=5)

        assert len(result) == len(close)
        assert np.all(np.isnan(result[:4]))
        assert not np.isnan(result[4])

    def test_insufficient_data(self):
        """Test when data length is less than period."""
        close = np.array([1.0, 2.0])
        result = linearreg_angle_numba(close, timeperiod=5)

        assert np.all(np.isnan(result))

    def test_period_two(self):
        """Test with period=2."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = linearreg_angle_numba(close, timeperiod=2)

        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_constant_input(self):
        """Test with constant input - angle should be zero."""
        close = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = linearreg_angle_numba(close, timeperiod=3)

        # For constant input, angle should be 0 degrees
        assert np.allclose(result[2:], 0.0, atol=1e-9)

    def test_positive_angle(self):
        """Test with increasing data - angle should be positive."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = linearreg_angle_numba(close, timeperiod=5)

        # Angle should be positive for increasing trend
        assert result[4] > 0

    def test_negative_angle(self):
        """Test with decreasing data - angle should be negative."""
        close = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
        result = linearreg_angle_numba(close, timeperiod=5)

        # Angle should be negative for decreasing trend
        assert result[4] < 0

    def test_angle_from_slope(self):
        """Verify angle is arctan(slope) in degrees."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        angle = linearreg_angle_numba(close, timeperiod=5)
        slope = linearreg_slope_numba(close, timeperiod=5)

        # angle = arctan(slope) * 180/π
        expected_angle = np.arctan(slope[4]) * (180.0 / np.pi)
        assert np.isclose(angle[4], expected_angle, rtol=1e-9)


class TestTsfNumba:
    """Test tsf_numba() Numba implementation directly."""

    def test_basic_computation(self):
        """Basic test with valid data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = tsf_numba(close, timeperiod=5)

        assert len(result) == len(close)
        assert np.all(np.isnan(result[:4]))
        assert not np.isnan(result[4])

    def test_insufficient_data(self):
        """Test when data length is less than period."""
        close = np.array([1.0, 2.0])
        result = tsf_numba(close, timeperiod=5)

        assert np.all(np.isnan(result))

    def test_period_two(self):
        """Test with period=2."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tsf_numba(close, timeperiod=2)

        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_linear_trend(self):
        """Test TSF with perfectly linear data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tsf_numba(close, timeperiod=5)

        # For perfect linear trend, TSF should project one period ahead
        # Expected: y = x + 1, at x=5 (next period), y=6
        assert np.isclose(result[4], 6.0, rtol=1e-9)

    def test_constant_input(self):
        """Test with constant input - TSF should equal constant."""
        close = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = tsf_numba(close, timeperiod=3)

        # For constant input, TSF should be the constant
        assert np.allclose(result[2:], 5.0, rtol=1e-9)

    def test_tsf_vs_linearreg(self):
        """Verify TSF = LINEARREG + SLOPE."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        tsf_result = tsf_numba(close, timeperiod=5)
        linearreg = linearreg_numba(close, timeperiod=5)
        slope = linearreg_slope_numba(close, timeperiod=5)

        # TSF should be linearreg + slope (projecting one period ahead)
        expected = linearreg[4:] + slope[4:]
        assert np.allclose(tsf_result[4:], expected, rtol=1e-9)

    def test_edge_case_period_one(self):
        """Test edge case with period=1."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tsf_numba(close, timeperiod=1)

        # Period=1: denominator becomes 0, should return all NaN
        assert np.all(np.isnan(result))


class TestMidpriceNumba:
    """Test midprice_numba() Numba implementation directly."""

    def test_basic_computation(self):
        """Basic test with valid data."""
        high = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 11.0, 15.0, 13.0, 12.0])
        low = np.array([9.0, 10.0, 10.0, 11.0, 12.0, 10.0, 9.0, 13.0, 11.0, 10.0])
        result = midprice_numba(high, low, timeperiod=5)

        assert len(result) == len(high)
        # First 4 values should be NaN
        assert np.all(np.isnan(result[:4]))
        # Fifth value onwards should be valid
        assert not np.isnan(result[4])

    def test_insufficient_data(self):
        """Test when data length is less than period."""
        high = np.array([10.0, 12.0])
        low = np.array([9.0, 10.0])
        result = midprice_numba(high, low, timeperiod=5)

        assert np.all(np.isnan(result))

    def test_period_two(self):
        """Test with period=2."""
        high = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 10.0, 11.0, 12.0])
        result = midprice_numba(high, low, timeperiod=2)

        # First value should be NaN
        assert np.isnan(result[0])
        # Second value onwards should be valid
        assert not np.isnan(result[1])

    def test_constant_range(self):
        """Test with constant high and low."""
        high = np.array([15.0, 15.0, 15.0, 15.0, 15.0])
        low = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        result = midprice_numba(high, low, timeperiod=3)

        # Midprice should be constant (15 + 10) / 2 = 12.5
        assert np.allclose(result[2:], 12.5, rtol=1e-9)

    def test_identical_high_low(self):
        """Test when high equals low."""
        high = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        low = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        result = midprice_numba(high, low, timeperiod=3)

        # When high = low, midprice should equal that value
        assert np.allclose(result[2:], 10.0, rtol=1e-9)

    def test_first_valid_index(self):
        """Verify first valid output is at index period-1."""
        high = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 11.0])
        low = np.array([9.0, 10.0, 10.0, 11.0, 12.0, 10.0, 9.0])
        period = 3
        result = midprice_numba(high, low, timeperiod=period)

        # First period-1 values should be NaN
        assert np.all(np.isnan(result[: period - 1]))
        # Value at index period-1 should be valid
        assert not np.isnan(result[period - 1])

    def test_monotonic_deque_correctness(self):
        """Test that monotonic deque algorithm produces correct results."""
        high = np.array([5.0, 8.0, 3.0, 9.0, 2.0, 7.0, 6.0])
        low = np.array([1.0, 2.0, 0.5, 3.0, 0.2, 1.5, 1.0])
        period = 3
        result = midprice_numba(high, low, timeperiod=period)

        # Manually verify window [0:3]: high_max=8, low_min=0.5 -> (8+0.5)/2 = 4.25
        assert np.isclose(result[2], 4.25, rtol=1e-9)

        # Manually verify window [1:4]: high_max=9, low_min=0.5 -> (9+0.5)/2 = 4.75
        assert np.isclose(result[3], 4.75, rtol=1e-9)


class TestMidpriceNumpy:
    """Test midprice_numpy() NumPy implementation directly."""

    def test_basic_computation(self):
        """Basic test with valid data."""
        high = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 11.0, 15.0, 13.0, 12.0])
        low = np.array([9.0, 10.0, 10.0, 11.0, 12.0, 10.0, 9.0, 13.0, 11.0, 10.0])
        result = midprice_numpy(high, low, timeperiod=5)

        assert len(result) == len(high)
        assert np.all(np.isnan(result[:4]))
        assert not np.isnan(result[4])

    def test_insufficient_data(self):
        """Test when data length is less than period."""
        high = np.array([10.0, 12.0])
        low = np.array([9.0, 10.0])
        result = midprice_numpy(high, low, timeperiod=5)

        assert np.all(np.isnan(result))

    def test_period_two(self):
        """Test with period=2."""
        high = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 10.0, 11.0, 12.0])
        result = midprice_numpy(high, low, timeperiod=2)

        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_numba_numpy_equivalence(self):
        """Verify Numba and NumPy implementations produce identical results."""
        high = np.array([10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 11.0, 15.0, 13.0, 12.0])
        low = np.array([9.0, 10.0, 10.0, 11.0, 12.0, 10.0, 9.0, 13.0, 11.0, 10.0])

        numba_result = midprice_numba(high, low, timeperiod=5)
        numpy_result = midprice_numpy(high, low, timeperiod=5)

        # Both implementations should produce identical results
        assert np.allclose(numba_result, numpy_result, rtol=1e-9, equal_nan=True)

    def test_mismatched_lengths(self):
        """Test error handling for mismatched array lengths."""
        high = np.array([10.0, 12.0, 11.0])
        low = np.array([9.0, 10.0])

        # Should return all NaN
        result = midprice_numpy(high, low, timeperiod=2)
        assert np.all(np.isnan(result))


class TestCrossValidationScenarios:
    """Cross-validation tests across multiple Numba functions."""

    def test_all_functions_handle_empty_input(self):
        """Verify all functions handle empty arrays gracefully."""
        empty = np.array([])

        # All should return empty array
        assert len(stddev_numba(empty, period=5)) == 0
        assert len(var_numba(empty, timeperiod=5)) == 0
        assert len(avgdev_numba(empty, timeperiod=5)) == 0
        assert len(linearreg_numba(empty, timeperiod=5)) == 0
        assert len(linearreg_slope_numba(empty, timeperiod=5)) == 0
        assert len(linearreg_intercept_numba(empty, timeperiod=5)) == 0
        assert len(linearreg_angle_numba(empty, timeperiod=5)) == 0
        assert len(tsf_numba(empty, timeperiod=5)) == 0
        assert len(midprice_numba(empty, empty, timeperiod=5)) == 0

    def test_all_functions_with_single_element(self):
        """Test all functions with single-element array."""
        single = np.array([5.0])

        # All should return array with NaN
        assert np.isnan(stddev_numba(single, period=2)[0])
        assert np.isnan(var_numba(single, timeperiod=2)[0])
        assert np.isnan(avgdev_numba(single, timeperiod=2)[0])
        assert np.isnan(linearreg_numba(single, timeperiod=2)[0])
        assert np.isnan(linearreg_slope_numba(single, timeperiod=2)[0])
        assert np.isnan(linearreg_intercept_numba(single, timeperiod=2)[0])
        assert np.isnan(linearreg_angle_numba(single, timeperiod=2)[0])
        assert np.isnan(tsf_numba(single, timeperiod=2)[0])
        assert np.isnan(midprice_numba(single, single, timeperiod=2)[0])

    def test_consistency_across_functions(self):
        """Verify mathematical relationships across functions."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Variance = StdDev^2
        var_result = var_numba(close, timeperiod=5)
        stddev_result = stddev_numba(close, period=5, ddof=0)
        assert np.allclose(var_result[4:], stddev_result[4:] ** 2, rtol=1e-9)

        # TSF = LINEARREG + SLOPE
        tsf_result = tsf_numba(close, timeperiod=5)
        linearreg = linearreg_numba(close, timeperiod=5)
        slope = linearreg_slope_numba(close, timeperiod=5)
        assert np.allclose(tsf_result[4:], linearreg[4:] + slope[4:], rtol=1e-9)

        # LINEARREG = INTERCEPT + SLOPE * (period-1)
        intercept = linearreg_intercept_numba(close, timeperiod=5)
        expected_linearreg = intercept[4:] + slope[4:] * 4
        assert np.allclose(linearreg[4:], expected_linearreg, rtol=1e-9)

        # ANGLE = arctan(SLOPE) * 180/π
        angle = linearreg_angle_numba(close, timeperiod=5)
        expected_angle = np.arctan(slope[4:]) * (180.0 / np.pi)
        assert np.allclose(angle[4:], expected_angle, rtol=1e-9)
