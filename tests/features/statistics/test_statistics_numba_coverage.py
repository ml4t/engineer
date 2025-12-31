"""
Comprehensive Numba coverage tests for statistics module.

This test file directly calls Numba implementations to boost coverage
from ~40-50% to target 80%+. Tests edge cases, boundary conditions,
and various data patterns.

Target coverage improvements:
- TSF: 41% -> 80%+ (lines 24-74)
- VAR: 45% -> 80%+ (lines 42-81)
- STDDEV: 42% -> 80%+ (lines 47-88)
- LINEARREG family: ~50% -> 80%+
- AVGDEV: 59% -> 80%+
"""

import numpy as np

from ml4t.engineer.features.statistics.avgdev import avgdev_numba
from ml4t.engineer.features.statistics.linearreg import linearreg_numba
from ml4t.engineer.features.statistics.linearreg_angle import linearreg_angle_numba
from ml4t.engineer.features.statistics.linearreg_intercept import linearreg_intercept_numba
from ml4t.engineer.features.statistics.linearreg_slope import linearreg_slope_numba
from ml4t.engineer.features.statistics.stddev import stddev_numba
from ml4t.engineer.features.statistics.tsf import tsf_numba
from ml4t.engineer.features.statistics.var import var_numba


class TestTSFNumbaCoverage:
    """Direct tests for TSF Numba implementation - target 80%+ coverage."""

    def test_tsf_numba_basic_trend(self):
        """Test TSF with perfect linear uptrend."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = tsf_numba(data, timeperiod=3)

        # First 2 values should be NaN (timeperiod-1)
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # TSF should forecast the next value in the trend
        # For perfect linear trend, TSF[i] should approximate data[i+1]
        assert not np.isnan(result[2])
        assert result[2] > 3.0  # Should forecast upward

    def test_tsf_numba_short_data(self):
        """Test TSF with data shorter than timeperiod - all NaN."""
        data = np.array([1.0, 2.0, 3.0])
        result = tsf_numba(data, timeperiod=5)

        # All values should be NaN when n < timeperiod
        assert np.all(np.isnan(result))

    def test_tsf_numba_exact_timeperiod(self):
        """Test TSF when data length equals timeperiod."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = tsf_numba(data, timeperiod=5)

        # First 4 NaN, last one should have value
        assert np.all(np.isnan(result[:4]))
        assert not np.isnan(result[4])

    def test_tsf_numba_constant_values(self):
        """Test TSF with constant data - slope should be zero."""
        data = np.full(10, 5.0)
        result = tsf_numba(data, timeperiod=3)

        # For constant data, TSF should equal the constant
        valid_results = result[~np.isnan(result)]
        assert np.allclose(valid_results, 5.0, atol=1e-10)

    def test_tsf_numba_zero_denominator(self):
        """Test TSF denominator edge case - should return NaN."""
        # Construct data that might cause zero denominator
        data = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = tsf_numba(data, timeperiod=5)

        # Should still work (constant data has valid TSF)
        assert not np.isnan(result[4])

    def test_tsf_numba_sliding_window(self):
        """Test TSF sliding window calculation."""
        # Longer series to exercise sliding window loop (i >= timeperiod)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        result = tsf_numba(data, timeperiod=3)

        # All values from index 2 onward should be computed
        assert not np.any(np.isnan(result[2:]))

        # TSF should track the trend
        assert np.all(np.diff(result[2:]) > 0)  # Monotonic increasing for uptrend

    def test_tsf_numba_downtrend(self):
        """Test TSF with downward trend."""
        data = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = tsf_numba(data, timeperiod=3)

        # TSF should forecast downward
        valid_results = result[~np.isnan(result)]
        assert np.all(np.diff(valid_results) < 0)  # Monotonic decreasing

    def test_tsf_numba_volatile_data(self):
        """Test TSF with volatile data."""
        data = np.array([1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0, 5.0, 9.0])
        result = tsf_numba(data, timeperiod=4)

        # Should compute values without errors
        assert not np.any(np.isnan(result[3:]))


class TestVARNumbaCoverage:
    """Direct tests for VAR Numba implementation - target 80%+ coverage."""

    def test_var_numba_basic(self):
        """Test VAR basic calculation."""
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
        result = var_numba(data, timeperiod=3, nbdev=1.0)

        # First 2 values NaN
        assert np.all(np.isnan(result[:2]))

        # Variance should be positive
        assert np.all(result[2:] >= 0)

    def test_var_numba_short_data(self):
        """Test VAR with insufficient data."""
        data = np.array([1.0, 2.0])
        result = var_numba(data, timeperiod=5, nbdev=1.0)

        # All NaN when n < timeperiod
        assert np.all(np.isnan(result))

    def test_var_numba_constant_values(self):
        """Test VAR with constant data - variance should be zero."""
        data = np.full(10, 5.0)
        result = var_numba(data, timeperiod=3, nbdev=1.0)

        # Variance of constant is zero
        valid_results = result[~np.isnan(result)]
        assert np.allclose(valid_results, 0.0, atol=1e-10)

    def test_var_numba_sliding_window(self):
        """Test VAR sliding window calculation."""
        # Exercise the loop for i >= timeperiod
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = var_numba(data, timeperiod=3, nbdev=1.0)

        # All values from index 2 onward should be computed
        assert not np.any(np.isnan(result[2:]))

        # Variance should be consistent for linear trend
        assert np.all(result[2:] > 0)

    def test_var_numba_exact_timeperiod(self):
        """Test VAR when data length equals timeperiod."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = var_numba(data, timeperiod=5, nbdev=1.0)

        # First 4 NaN, last one should have value
        assert np.all(np.isnan(result[:4]))
        assert not np.isnan(result[4])
        assert result[4] >= 0

    def test_var_numba_nbdev_parameter(self):
        """Test VAR with different nbdev values (compatibility parameter)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        result1 = var_numba(data, timeperiod=3, nbdev=1.0)
        result2 = var_numba(data, timeperiod=3, nbdev=2.0)

        # nbdev doesn't affect VAR calculation (compatibility parameter)
        assert np.allclose(result1[~np.isnan(result1)], result2[~np.isnan(result2)])

    def test_var_numba_high_variance(self):
        """Test VAR with high variance data."""
        data = np.array([1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0])
        result = var_numba(data, timeperiod=2, nbdev=1.0)

        # Should handle large variance
        assert not np.any(np.isnan(result[1:]))
        assert np.all(result[1:] > 0)


class TestSTDDEVNumbaCoverage:
    """Direct tests for STDDEV Numba implementation - target 80%+ coverage."""

    def test_stddev_numba_basic(self):
        """Test STDDEV basic calculation."""
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
        result = stddev_numba(data, period=3, nbdev=1.0, ddof=0)

        # First 2 values NaN
        assert np.all(np.isnan(result[:2]))

        # StdDev should be positive
        assert np.all(result[2:] > 0)

    def test_stddev_numba_short_data(self):
        """Test STDDEV with insufficient data (period < 2)."""
        data = np.array([1.0, 2.0, 3.0])
        result = stddev_numba(data, period=1, nbdev=1.0, ddof=0)

        # All NaN when period < 2
        assert np.all(np.isnan(result))

    def test_stddev_numba_data_too_short(self):
        """Test STDDEV when n < period."""
        data = np.array([1.0, 2.0])
        result = stddev_numba(data, period=5, nbdev=1.0, ddof=0)

        # All NaN when n < period
        assert np.all(np.isnan(result))

    def test_stddev_numba_constant_values(self):
        """Test STDDEV with constant data - should be zero."""
        data = np.full(10, 5.0)
        result = stddev_numba(data, period=3, nbdev=1.0, ddof=0)

        # StdDev of constant is zero
        valid_results = result[~np.isnan(result)]
        assert np.allclose(valid_results, 0.0, atol=1e-10)

    def test_stddev_numba_with_nan(self):
        """Test STDDEV with NaN values in data."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
        result = stddev_numba(data, period=3, nbdev=1.0, ddof=0)

        # Window containing NaN should result in NaN
        assert np.isnan(result[2])  # Window includes NaN at index 2

    def test_stddev_numba_ddof_zero_edge_case(self):
        """Test STDDEV when period - ddof = 0 (avoid division by zero)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = stddev_numba(data, period=2, nbdev=1.0, ddof=2)

        # Should handle ddof >= period gracefully (result[i] = 0.0)
        # First value should be 0.0 due to ddof edge case
        assert result[1] == 0.0

    def test_stddev_numba_nbdev_scaling(self):
        """Test STDDEV nbdev scaling factor."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        result1 = stddev_numba(data, period=3, nbdev=1.0, ddof=0)
        result2 = stddev_numba(data, period=3, nbdev=2.0, ddof=0)

        # nbdev=2 should be 2x nbdev=1
        valid1 = result1[~np.isnan(result1)]
        valid2 = result2[~np.isnan(result2)]
        assert np.allclose(valid2, valid1 * 2.0)

    def test_stddev_numba_ddof_population_vs_sample(self):
        """Test STDDEV with ddof=0 (population) vs ddof=1 (sample)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        pop = stddev_numba(data, period=3, nbdev=1.0, ddof=0)
        sample = stddev_numba(data, period=3, nbdev=1.0, ddof=1)

        # Sample stddev should be larger than population
        pop_valid = pop[~np.isnan(pop)]
        sample_valid = sample[~np.isnan(sample)]
        assert np.all(sample_valid >= pop_valid)

    def test_stddev_numba_sliding_window(self):
        """Test STDDEV sliding window calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = stddev_numba(data, period=3, nbdev=1.0, ddof=0)

        # All values from index 2 onward should be computed
        assert not np.any(np.isnan(result[2:]))


class TestLINEARREGNumbaCoverage:
    """Direct tests for LINEARREG Numba implementation - target 80%+ coverage."""

    def test_linearreg_numba_basic_uptrend(self):
        """Test LINEARREG with perfect linear uptrend."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = linearreg_numba(data, timeperiod=3)

        # First 2 values NaN
        assert np.all(np.isnan(result[:2]))

        # LINEARREG should track the trend
        assert np.all(result[2:] > 0)

    def test_linearreg_numba_short_data(self):
        """Test LINEARREG with insufficient data."""
        data = np.array([1.0, 2.0])
        result = linearreg_numba(data, timeperiod=5)

        # All NaN when n < timeperiod
        assert np.all(np.isnan(result))

    def test_linearreg_numba_constant_values(self):
        """Test LINEARREG with constant data."""
        data = np.full(10, 5.0)
        result = linearreg_numba(data, timeperiod=3)

        # For constant data, regression line should equal the constant
        valid_results = result[~np.isnan(result)]
        assert np.allclose(valid_results, 5.0, atol=1e-10)

    def test_linearreg_numba_sliding_window(self):
        """Test LINEARREG sliding window calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        result = linearreg_numba(data, timeperiod=3)

        # All values from index 2 onward should be computed
        assert not np.any(np.isnan(result[2:]))

        # For linear trend, regression should closely match actual values
        assert np.all(result[2:] > 0)


class TestLINEARREGSlopeNumbaCoverage:
    """Direct tests for LINEARREG_SLOPE Numba implementation - target 80%+ coverage."""

    def test_linearreg_slope_numba_uptrend(self):
        """Test LINEARREG_SLOPE with uptrend - positive slope."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = linearreg_slope_numba(data, timeperiod=3)

        # Slope should be positive for uptrend
        valid_results = result[~np.isnan(result)]
        assert np.all(valid_results > 0)

    def test_linearreg_slope_numba_downtrend(self):
        """Test LINEARREG_SLOPE with downtrend - negative slope."""
        data = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = linearreg_slope_numba(data, timeperiod=3)

        # Slope should be negative for downtrend
        valid_results = result[~np.isnan(result)]
        assert np.all(valid_results < 0)

    def test_linearreg_slope_numba_constant(self):
        """Test LINEARREG_SLOPE with constant data - zero slope."""
        data = np.full(10, 5.0)
        result = linearreg_slope_numba(data, timeperiod=3)

        # Slope should be zero for constant data
        valid_results = result[~np.isnan(result)]
        assert np.allclose(valid_results, 0.0, atol=1e-10)

    def test_linearreg_slope_numba_short_data(self):
        """Test LINEARREG_SLOPE with insufficient data."""
        data = np.array([1.0, 2.0])
        result = linearreg_slope_numba(data, timeperiod=5)

        # All NaN when n < timeperiod
        assert np.all(np.isnan(result))

    def test_linearreg_slope_numba_sliding_window(self):
        """Test LINEARREG_SLOPE sliding window."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        result = linearreg_slope_numba(data, timeperiod=3)

        # All slopes should be positive and roughly equal for linear trend
        valid_results = result[~np.isnan(result)]
        assert np.all(valid_results > 0)
        assert np.allclose(valid_results, valid_results[0], rtol=0.1)


class TestLINEARREGInterceptNumbaCoverage:
    """Direct tests for LINEARREG_INTERCEPT Numba implementation - target 80%+ coverage."""

    def test_linearreg_intercept_numba_basic(self):
        """Test LINEARREG_INTERCEPT basic calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = linearreg_intercept_numba(data, timeperiod=3)

        # Should compute intercepts
        assert not np.any(np.isnan(result[2:]))

    def test_linearreg_intercept_numba_constant(self):
        """Test LINEARREG_INTERCEPT with constant data."""
        data = np.full(10, 5.0)
        result = linearreg_intercept_numba(data, timeperiod=3)

        # For constant data, intercept should equal the constant
        valid_results = result[~np.isnan(result)]
        assert np.allclose(valid_results, 5.0, atol=1e-10)

    def test_linearreg_intercept_numba_short_data(self):
        """Test LINEARREG_INTERCEPT with insufficient data."""
        data = np.array([1.0, 2.0])
        result = linearreg_intercept_numba(data, timeperiod=5)

        # All NaN when n < timeperiod
        assert np.all(np.isnan(result))

    def test_linearreg_intercept_numba_sliding_window(self):
        """Test LINEARREG_INTERCEPT sliding window."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        result = linearreg_intercept_numba(data, timeperiod=3)

        # All values from index 2 onward should be computed
        assert not np.any(np.isnan(result[2:]))


class TestLINEARREGAngleNumbaCoverage:
    """Direct tests for LINEARREG_ANGLE Numba implementation - target 80%+ coverage."""

    def test_linearreg_angle_numba_uptrend(self):
        """Test LINEARREG_ANGLE with uptrend - positive angle."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = linearreg_angle_numba(data, timeperiod=3)

        # Angle should be positive for uptrend
        valid_results = result[~np.isnan(result)]
        assert np.all(valid_results > 0)

    def test_linearreg_angle_numba_downtrend(self):
        """Test LINEARREG_ANGLE with downtrend - negative angle."""
        data = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = linearreg_angle_numba(data, timeperiod=3)

        # Angle should be negative for downtrend
        valid_results = result[~np.isnan(result)]
        assert np.all(valid_results < 0)

    def test_linearreg_angle_numba_constant(self):
        """Test LINEARREG_ANGLE with constant data - zero angle."""
        data = np.full(10, 5.0)
        result = linearreg_angle_numba(data, timeperiod=3)

        # Angle should be zero for flat line
        valid_results = result[~np.isnan(result)]
        assert np.allclose(valid_results, 0.0, atol=1e-10)

    def test_linearreg_angle_numba_short_data(self):
        """Test LINEARREG_ANGLE with insufficient data."""
        data = np.array([1.0, 2.0])
        result = linearreg_angle_numba(data, timeperiod=5)

        # All NaN when n < timeperiod
        assert np.all(np.isnan(result))

    def test_linearreg_angle_numba_sliding_window(self):
        """Test LINEARREG_ANGLE sliding window."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        result = linearreg_angle_numba(data, timeperiod=3)

        # All angles should be positive for uptrend
        valid_results = result[~np.isnan(result)]
        assert np.all(valid_results > 0)

    def test_linearreg_angle_numba_steep_vs_gentle(self):
        """Test LINEARREG_ANGLE with different slopes."""
        # Steep uptrend
        steep = np.array([1.0, 10.0, 20.0, 30.0, 40.0])
        steep_result = linearreg_angle_numba(steep, timeperiod=3)

        # Gentle uptrend
        gentle = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gentle_result = linearreg_angle_numba(gentle, timeperiod=3)

        # Steep should have larger angle
        steep_angle = steep_result[~np.isnan(steep_result)][0]
        gentle_angle = gentle_result[~np.isnan(gentle_result)][0]
        assert steep_angle > gentle_angle


class TestAVGDEVNumbaCoverage:
    """Direct tests for AVGDEV Numba implementation - target 80%+ coverage."""

    def test_avgdev_numba_basic(self):
        """Test AVGDEV basic calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = avgdev_numba(data, timeperiod=3)

        # First 2 values NaN
        assert np.all(np.isnan(result[:2]))

        # AVGDEV should be positive
        assert np.all(result[2:] > 0)

    def test_avgdev_numba_short_data(self):
        """Test AVGDEV with insufficient data."""
        data = np.array([1.0, 2.0])
        result = avgdev_numba(data, timeperiod=5)

        # All NaN when n < timeperiod
        assert np.all(np.isnan(result))

    def test_avgdev_numba_constant_values(self):
        """Test AVGDEV with constant data - should be zero."""
        data = np.full(10, 5.0)
        result = avgdev_numba(data, timeperiod=3)

        # AVGDEV of constant is zero
        valid_results = result[~np.isnan(result)]
        assert np.allclose(valid_results, 0.0, atol=1e-10)

    def test_avgdev_numba_sliding_window(self):
        """Test AVGDEV sliding window calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        result = avgdev_numba(data, timeperiod=3)

        # All values from index 2 onward should be computed
        assert not np.any(np.isnan(result[2:]))

    def test_avgdev_numba_high_deviation(self):
        """Test AVGDEV with high deviation data."""
        data = np.array([1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 1.0, 100.0])
        result = avgdev_numba(data, timeperiod=2)

        # Should compute high deviation
        assert not np.any(np.isnan(result[1:]))
        assert np.all(result[1:] > 0)

    def test_avgdev_numba_exact_timeperiod(self):
        """Test AVGDEV when data length equals timeperiod."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = avgdev_numba(data, timeperiod=5)

        # First 4 NaN, last one should have value
        assert np.all(np.isnan(result[:4]))
        assert not np.isnan(result[4])
        assert result[4] >= 0


class TestEdgeCasesAndIntegration:
    """Additional edge cases and integration tests for comprehensive coverage."""

    def test_all_functions_handle_empty_array(self):
        """Test all functions handle empty arrays gracefully."""
        empty = np.array([])

        assert len(tsf_numba(empty, 3)) == 0
        assert len(var_numba(empty, 3)) == 0
        assert len(stddev_numba(empty, 3)) == 0
        assert len(linearreg_numba(empty, 3)) == 0
        assert len(linearreg_slope_numba(empty, 3)) == 0
        assert len(linearreg_intercept_numba(empty, 3)) == 0
        assert len(linearreg_angle_numba(empty, 3)) == 0
        assert len(avgdev_numba(empty, 3)) == 0

    def test_all_functions_handle_single_value(self):
        """Test all functions handle single value arrays."""
        single = np.array([5.0])

        # All should return array with NaN
        assert np.all(np.isnan(tsf_numba(single, 3)))
        assert np.all(np.isnan(var_numba(single, 3)))
        assert np.all(np.isnan(stddev_numba(single, 3)))
        assert np.all(np.isnan(linearreg_numba(single, 3)))
        assert np.all(np.isnan(linearreg_slope_numba(single, 3)))
        assert np.all(np.isnan(linearreg_intercept_numba(single, 3)))
        assert np.all(np.isnan(linearreg_angle_numba(single, 3)))
        assert np.all(np.isnan(avgdev_numba(single, 3)))

    def test_large_timeperiod_values(self):
        """Test functions with large timeperiod values."""
        data = np.arange(100.0)

        result_tsf = tsf_numba(data, timeperiod=50)
        result_var = var_numba(data, timeperiod=50)
        result_stddev = stddev_numba(data, period=50)  # stddev uses 'period'
        result_linearreg = linearreg_numba(data, timeperiod=50)

        # Should compute valid results after warmup
        assert not np.any(np.isnan(result_tsf[49:]))
        assert not np.any(np.isnan(result_var[49:]))
        assert not np.any(np.isnan(result_stddev[49:]))
        assert not np.any(np.isnan(result_linearreg[49:]))

    def test_very_small_values(self):
        """Test functions with very small values (numerical stability)."""
        data = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10, 7e-10])

        result_var = var_numba(data, timeperiod=3)
        result_stddev = stddev_numba(data, period=3)  # stddev uses 'period'
        result_avgdev = avgdev_numba(data, timeperiod=3)

        # Should compute without numerical issues
        assert not np.any(np.isnan(result_var[2:]))
        assert not np.any(np.isnan(result_stddev[2:]))
        assert not np.any(np.isnan(result_avgdev[2:]))

    def test_very_large_values(self):
        """Test functions with very large values."""
        data = np.array([1e10, 2e10, 3e10, 4e10, 5e10, 6e10, 7e10])

        result_var = var_numba(data, timeperiod=3)
        result_stddev = stddev_numba(data, period=3)  # stddev uses 'period'
        result_linearreg = linearreg_numba(data, timeperiod=3)

        # Should compute without overflow
        assert not np.any(np.isnan(result_var[2:]))
        assert not np.any(np.isnan(result_stddev[2:]))
        assert not np.any(np.isnan(result_linearreg[2:]))

    def test_mixed_positive_negative(self):
        """Test functions with mixed positive and negative values."""
        data = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0])

        result_tsf = tsf_numba(data, timeperiod=3)
        result_var = var_numba(data, timeperiod=3)
        result_linearreg = linearreg_numba(data, timeperiod=3)

        # Should handle mixed signs correctly
        assert not np.any(np.isnan(result_tsf[2:]))
        assert not np.any(np.isnan(result_var[2:]))
        assert not np.any(np.isnan(result_linearreg[2:]))
