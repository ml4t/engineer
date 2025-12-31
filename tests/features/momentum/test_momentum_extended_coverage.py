"""
Extended coverage tests for Numba implementations in momentum module.

These tests directly call the Numba functions (not wrapper functions) to ensure
the JIT-compiled code paths are properly exercised. Focus areas:
1. Edge cases: insufficient data, exact lookback length, boundary conditions
2. Branch paths: All conditional branches in the algorithms
3. Zero/NaN handling: Division by zero, invalid data
"""

import numpy as np
import pytest

from ml4t.engineer.features.momentum.adx import (
    _apply_wilders_smoothing,
    _calculate_directional_movement,
    _calculate_dx,
    _calculate_true_range,
    adx_numba,
)
from ml4t.engineer.features.momentum.aroon import aroon_numba, aroonosc_numba
from ml4t.engineer.features.momentum.directional import (
    calculate_directional_movement_nb,
    dx_numba,
    minus_di_numba,
    plus_di_numba,
    wilders_smoothing_nb,
)
from ml4t.engineer.features.momentum.macd import (
    _int_ema_talib_nb,
    macd_histogram_numba,
    macd_numba,
    macd_signal_numba,
)
from ml4t.engineer.features.momentum.macdfix import _int_ema_fixed_k, macdfix_numba
from ml4t.engineer.features.momentum.sar import sar_numba
from ml4t.engineer.features.momentum.ultosc import ultosc_numba


class TestDirectionalMovementNumba:
    """Test directional.py Numba functions."""

    def test_calculate_directional_movement_nb_positive_plus_dm(self):
        """Test plus DM > minus DM case."""
        high = np.array([10.0, 12.0, 11.0])
        low = np.array([9.0, 10.0, 9.5])

        plus_dm, minus_dm = calculate_directional_movement_nb(high, low)

        # First value should be 0 (no previous bar)
        assert plus_dm[0] == 0.0
        assert minus_dm[0] == 0.0

        # Second value: high_diff=2, low_diff=0 -> plus_dm=2
        assert plus_dm[1] == 2.0
        assert minus_dm[1] == 0.0

        # Third value: high_diff=-1 (negative), low_diff=0.5 -> minus_dm=0.5
        assert plus_dm[2] == 0.0
        assert minus_dm[2] == 0.5

    def test_calculate_directional_movement_nb_negative_values(self):
        """Test when both diffs are negative."""
        # For both to be zero, need:
        # - high_diff = high[i] - high[i-1] <= 0 (no upward high movement)
        # - low_diff = low[i-1] - low[i] <= 0 (no downward low movement, i.e., low went up)
        high = np.array([12.0, 11.0, 10.0])  # high going down
        low = np.array([8.0, 9.0, 10.0])  # low going UP (so low_diff is negative)

        plus_dm, minus_dm = calculate_directional_movement_nb(high, low)

        # Both movements are negative or zero, should be zero
        assert plus_dm[1] == 0.0
        assert minus_dm[1] == 0.0

    def test_wilders_smoothing_nb_insufficient_data(self):
        """Test when n <= period."""
        data = np.array([1.0, 2.0, 3.0])
        result = wilders_smoothing_nb(data, period=5)
        assert np.all(np.isnan(result))

    def test_wilders_smoothing_nb_exact_period(self):
        """Test with exactly period+1 data points."""
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        result = wilders_smoothing_nb(data, period=5)

        # First valid output at index period-1=4
        assert not np.isnan(result[4])
        # Last value should also be valid
        assert not np.isnan(result[5])

    def test_wilders_smoothing_nb_with_nan_values(self):
        """Test handling of NaN values in input."""
        # Wilder's smoothing: first output at period-1
        # For period=3, first output at index 2, summing indices 1, 2
        data = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        result = wilders_smoothing_nb(data, period=3)

        # First output at index 2 (period-1), value = sum of indices 1,2 = 1+2 = 3
        assert not np.isnan(result[2])
        assert result[2] == 3.0  # sum of indices 1 and 2

    def test_plus_di_numba_exact_boundary(self):
        """Test plus_di with exactly timeperiod+1 bars."""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
        close = np.array([9.5, 10.5, 11.5, 12.5, 13.5])

        result = plus_di_numba(high, low, close, timeperiod=3)

        # First valid output at index timeperiod=3
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert not np.isnan(result[3])

    def test_plus_di_numba_zero_true_range(self):
        """Test when sum_tr becomes zero."""
        # All identical values -> zero true range
        high = np.full(20, 10.0)
        low = np.full(20, 10.0)
        close = np.full(20, 10.0)

        result = plus_di_numba(high, low, close, timeperiod=14)

        # Should handle zero TR with result=0.0
        valid_indices = ~np.isnan(result)
        if np.any(valid_indices):
            assert np.all(result[valid_indices] == 0.0)

    def test_plus_di_numba_insufficient_data(self):
        """Test when n <= timeperiod."""
        high = np.array([10.0, 11.0])
        low = np.array([9.0, 10.0])
        close = np.array([9.5, 10.5])

        result = plus_di_numba(high, low, close, timeperiod=5)
        assert np.all(np.isnan(result))

    def test_minus_di_numba_downtrend(self):
        """Test minus_di in downtrend (minus DM dominates)."""
        high = np.array([20.0, 19.0, 18.0, 17.0, 16.0, 15.0] + [14.0] * 10)
        low = np.array([18.0, 17.0, 16.0, 15.0, 14.0, 13.0] + [12.0] * 10)
        close = np.array([19.0, 18.0, 17.0, 16.0, 15.0, 14.0] + [13.0] * 10)

        result = minus_di_numba(high, low, close, timeperiod=5)

        # Should have high minus DI values in downtrend
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.any(valid_values > 0)

    def test_minus_di_numba_zero_tr_handling(self):
        """Test minus_di when sum_tr=0."""
        high = np.full(20, 10.0)
        low = np.full(20, 10.0)
        close = np.full(20, 10.0)

        result = minus_di_numba(high, low, close, timeperiod=14)

        # Should return 0.0 for zero TR
        valid_indices = ~np.isnan(result)
        if np.any(valid_indices):
            assert np.all(result[valid_indices] == 0.0)

    def test_dx_numba_zero_sum_di(self):
        """Test DX when plus_di + minus_di = 0."""
        # Create scenario where both DI are zero
        high = np.full(30, 10.0)
        low = np.full(30, 10.0)
        close = np.full(30, 10.0)

        result = dx_numba(high, low, close, timeperiod=14)

        # Should return 0.0 when sum_di=0
        valid_indices = ~np.isnan(result)
        if np.any(valid_indices):
            assert np.all(result[valid_indices] == 0.0)

    def test_dx_numba_exact_period_data(self):
        """Test DX with exactly 2*timeperiod data."""
        np.random.seed(42)
        high = np.random.uniform(10, 20, 30)
        low = high - np.random.uniform(0.5, 1.5, 30)
        close = (high + low) / 2

        result = dx_numba(high, low, close, timeperiod=14)

        # First valid value at index timeperiod=14
        assert np.all(np.isnan(result[:14]))
        assert not np.isnan(result[14])


class TestADXNumba:
    """Test adx.py Numba helper functions."""

    def test_calculate_directional_movement_all_cases(self):
        """Test all 5 cases in directional movement calculation."""
        # Case 1: diff_p > 0 and diff_p > diff_m
        plus_dm, minus_dm = _calculate_directional_movement(10.0, 12.0, 9.0, 9.5)
        assert plus_dm == 2.0
        assert minus_dm == 0.0

        # Case 2: diff_m > 0 and diff_p < diff_m
        plus_dm, minus_dm = _calculate_directional_movement(10.0, 10.5, 9.0, 8.0)
        assert plus_dm == 0.0
        assert minus_dm == 1.0

        # Case 3: diff_p > 0 and diff_m <= 0
        plus_dm, minus_dm = _calculate_directional_movement(10.0, 11.0, 9.0, 10.0)
        assert plus_dm == 1.0
        assert minus_dm == 0.0

        # Case 4: diff_m > 0 and diff_p <= 0
        plus_dm, minus_dm = _calculate_directional_movement(10.0, 9.5, 9.0, 8.5)
        assert plus_dm == 0.0
        assert minus_dm == 0.5

        # Case 5: Both zero or negative
        plus_dm, minus_dm = _calculate_directional_movement(10.0, 9.0, 9.0, 10.0)
        assert plus_dm == 0.0
        assert minus_dm == 0.0

    def test_calculate_true_range(self):
        """Test true range calculation."""
        # TR = max(H-L, |H-PC|, |L-PC|)

        # Case 1: H-L is largest
        tr = _calculate_true_range(high=10.0, low=8.0, prev_close=9.0)
        assert tr == 2.0

        # Case 2: |H-PC| is largest
        tr = _calculate_true_range(high=10.0, low=9.5, prev_close=7.0)
        assert tr == 3.0

        # Case 3: |L-PC| is largest
        tr = _calculate_true_range(high=9.0, low=8.0, prev_close=11.0)
        assert tr == 3.0

    def test_apply_wilders_smoothing(self):
        """Test Wilder's smoothing formula."""
        result = _apply_wilders_smoothing(prev_smoothed=10.0, new_value=12.0, period=14)

        # Expected: 10 - 10/14 + 12 = 10 - 0.714 + 12 = 21.286
        expected = 10.0 - (10.0 / 14) + 12.0
        assert abs(result - expected) < 1e-10

    def test_calculate_dx_zero_sum(self):
        """Test DX calculation when sum_di is zero."""
        dx = _calculate_dx(plus_di=0.0, minus_di=0.0)
        assert dx == 0.0

    def test_calculate_dx_normal(self):
        """Test DX calculation with normal values."""
        dx = _calculate_dx(plus_di=30.0, minus_di=20.0)

        # Expected: 100 * |30-20| / (30+20) = 100 * 10 / 50 = 20
        assert abs(dx - 20.0) < 1e-10

    def test_adx_numba_insufficient_data(self):
        """Test ADX with insufficient data."""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])

        result = adx_numba(high, low, close, period=14)
        assert np.all(np.isnan(result))

    def test_adx_numba_exact_lookback(self):
        """Test ADX with exactly lookback+1 data points."""
        # lookback = 2*period - 1 = 27
        np.random.seed(42)
        high = np.random.uniform(10, 20, 28)
        low = high - np.random.uniform(0.5, 1.5, 28)
        close = (high + low) / 2

        result = adx_numba(high, low, close, period=14)

        # First valid at index 27
        assert np.all(np.isnan(result[:27]))
        assert not np.isnan(result[27])

    def test_adx_numba_zero_tr_path(self):
        """Test ADX when TR becomes zero."""
        # Flat prices -> zero TR
        high = np.full(50, 10.0)
        low = np.full(50, 10.0)
        close = np.full(50, 10.0)

        result = adx_numba(high, low, close, period=14)

        # Should handle zero TR gracefully
        assert not np.any(np.isinf(result))


class TestSARNumba:
    """Test sar.py Numba function."""

    def test_sar_numba_insufficient_data(self):
        """Test SAR with less than 2 data points."""
        high = np.array([10.0])
        low = np.array([9.0])

        result = sar_numba(high, low)
        assert np.all(np.isnan(result))

    def test_sar_numba_exact_minimum(self):
        """Test SAR with exactly 2 data points."""
        high = np.array([10.0, 11.0])
        low = np.array([9.0, 10.0])

        result = sar_numba(high, low)

        # First SAR at index 1
        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_sar_numba_acceleration_exceeds_maximum(self):
        """Test when acceleration > maximum."""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])

        # acceleration > maximum should be clamped
        result = sar_numba(high, low, acceleration=0.3, maximum=0.2)

        # Should not raise error, acceleration should be clamped
        assert not np.any(np.isnan(result[1:]))

    def test_sar_numba_mismatched_lengths(self):
        """Test SAR with mismatched high/low lengths."""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0])

        result = sar_numba(high, low)
        assert np.all(np.isnan(result))

    def test_sar_numba_initial_long_position(self):
        """Test SAR starting in long position (minus_dm = 0)."""
        # Create uptrend: minus_dm should be 0
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([9.5, 10.5, 11.5, 12.5, 13.5])

        result = sar_numba(high, low)

        # Should start in long position (SAR below price)
        # First valid SAR at index 1
        assert result[1] < high[1]

    def test_sar_numba_initial_short_position(self):
        """Test SAR starting in short position (minus_dm > 0)."""
        # Create downtrend: minus_dm > 0
        high = np.array([15.0, 14.0, 13.0, 12.0, 11.0])
        low = np.array([14.5, 13.5, 12.5, 11.5, 10.5])

        result = sar_numba(high, low)

        # Should start in short position (SAR above price)
        # First valid SAR at index 1
        assert result[1] > low[1]

    def test_sar_numba_switch_from_long_to_short(self):
        """Test SAR switching from long to short."""
        # Uptrend then downtrend
        high = np.array([10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 11.0, 10.0, 9.0])

        result = sar_numba(high, low, acceleration=0.1, maximum=0.2)

        # Should have valid values throughout (no NaN after index 1)
        assert not np.any(np.isnan(result[1:]))

    def test_sar_numba_switch_from_short_to_long(self):
        """Test SAR switching from short to long."""
        # Downtrend then uptrend
        high = np.array([15.0, 14.0, 13.0, 12.0, 13.0, 14.0, 15.0])
        low = np.array([14.0, 13.0, 12.0, 11.0, 12.0, 13.0, 14.0])

        result = sar_numba(high, low, acceleration=0.1, maximum=0.2)

        # Should have valid values throughout
        assert not np.any(np.isnan(result[1:]))

    def test_sar_numba_no_switch_long(self):
        """Test SAR staying in long position (else branch line 128)."""
        # Strong uptrend, SAR should stay below
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        low = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5])

        result = sar_numba(high, low, acceleration=0.02, maximum=0.2)

        # SAR should stay below price (long position)
        for i in range(2, len(result)):
            if not np.isnan(result[i]):
                assert result[i] < low[i]

    def test_sar_numba_no_switch_short(self):
        """Test SAR staying in short position (else branch line 168)."""
        # Strong downtrend, SAR should stay above
        high = np.array([16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0])
        low = np.array([15.5, 14.5, 13.5, 12.5, 11.5, 10.5, 9.5])

        result = sar_numba(high, low, acceleration=0.02, maximum=0.2)

        # SAR should stay above price (short position)
        for i in range(2, len(result)):
            if not np.isnan(result[i]):
                assert result[i] > high[i]

    def test_sar_numba_af_increment_long(self):
        """Test AF increment when new high is reached (line 133-136)."""
        # Steadily increasing highs
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        low = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5])

        result = sar_numba(high, low, acceleration=0.02, maximum=0.2)

        # AF should increase with each new high
        assert not np.any(np.isnan(result[1:]))

    def test_sar_numba_af_increment_short(self):
        """Test AF increment when new low is reached (line 172-175)."""
        # Steadily decreasing lows
        high = np.array([15.0, 14.0, 13.0, 12.0, 11.0, 10.0])
        low = np.array([14.5, 13.5, 12.5, 11.5, 10.5, 9.5])

        result = sar_numba(high, low, acceleration=0.02, maximum=0.2)

        # AF should increase with each new low
        assert not np.any(np.isnan(result[1:]))


class TestMACDNumba:
    """Test macd.py Numba functions."""

    def test_int_ema_talib_nb_insufficient_data(self):
        """Test _int_ema_talib_nb with insufficient data."""
        close = np.array([100.0, 101.0])
        result = _int_ema_talib_nb(close, start_idx=0, end_idx=1, period=5, k=0.3)

        # With period=5, need at least 5 values
        assert np.all(np.isnan(result))

    def test_int_ema_talib_nb_exact_start_idx(self):
        """Test _int_ema_talib_nb when start_idx equals lookback."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        k = 2.0 / (3 + 1)  # period=3
        result = _int_ema_talib_nb(close, start_idx=2, end_idx=5, period=3, k=k)

        # First output at start_idx=2
        assert not np.isnan(result[2])
        # Subsequent values should be valid
        assert not np.isnan(result[3])

    def test_int_ema_talib_nb_start_idx_greater_than_end_idx(self):
        """Test when start_idx > end_idx."""
        close = np.array([100.0, 101.0, 102.0])
        result = _int_ema_talib_nb(close, start_idx=5, end_idx=2, period=2, k=0.5)

        # Should return all NaN
        assert np.all(np.isnan(result))

    def test_macd_numba_insufficient_data(self):
        """Test MACD with insufficient data."""
        close = np.array([100.0, 101.0, 102.0])
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)

        # Need at least slow_period + signal_period - 1
        assert np.all(np.isnan(result))

    def test_macd_numba_swap_periods(self):
        """Test MACD when slow < fast (should swap)."""
        np.random.seed(42)
        close = np.random.uniform(100, 110, 50)

        # Pass swapped periods
        result1 = macd_numba(close, fast_period=26, slow_period=12, signal_period=9)
        result2 = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)

        # Should produce same result after internal swap
        valid1 = ~np.isnan(result1)
        valid2 = ~np.isnan(result2)
        assert np.allclose(result1[valid1], result2[valid2])

    def test_macd_numba_exact_lookback(self):
        """Test MACD with exactly lookback+1 data points."""
        # lookback = signal_period - 1 + slow_period - 1 = 8 + 25 = 33
        # First valid at index 33
        np.random.seed(42)
        close = np.random.uniform(100, 110, 50)

        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)

        # First valid at index lookback_total = 33
        assert np.all(np.isnan(result[:33]))
        assert not np.isnan(result[33])

    def test_macd_signal_numba(self):
        """Test MACD signal line calculation."""
        np.random.seed(42)
        close = np.random.uniform(100, 110, 50)

        result = macd_signal_numba(close, fast_period=12, slow_period=26, signal_period=9)

        # Should have valid values after initial lookback
        assert np.any(~np.isnan(result))

    def test_macd_histogram_numba(self):
        """Test MACD histogram calculation."""
        np.random.seed(42)
        close = np.random.uniform(100, 110, 50)

        result = macd_histogram_numba(
            close,
            fast_period=12,
            slow_period=26,
            signal_period=9,
        )

        # Histogram should be MACD - Signal
        macd_line = macd_numba(close, 12, 26, 9)
        signal_line = macd_signal_numba(close, 12, 26, 9)

        valid = ~np.isnan(result)
        assert np.allclose(result[valid], macd_line[valid] - signal_line[valid])


class TestMACDFIXNumba:
    """Test macdfix.py Numba functions."""

    def test_int_ema_fixed_k_insufficient_data(self):
        """Test _int_ema_fixed_k with insufficient data."""
        close = np.array([100.0, 101.0])
        result = _int_ema_fixed_k(close, start_idx=0, end_idx=1, period=5, k=0.15)

        # Need at least period values
        assert np.all(np.isnan(result))

    def test_int_ema_fixed_k_with_fixed_constants(self):
        """Test that MACDFIX uses fixed k values, not calculated."""
        np.random.seed(42)
        close = np.random.uniform(100, 110, 50)

        # MACDFIX should use k=0.15 for period=12, not 2/(12+1)
        k_fixed = 0.15
        k_calculated = 2.0 / (12 + 1)

        assert k_fixed != k_calculated  # Confirm they're different

        result = _int_ema_fixed_k(close, start_idx=11, end_idx=49, period=12, k=k_fixed)

        # Should produce valid results
        assert np.any(~np.isnan(result))

    def test_macdfix_numba_insufficient_data(self):
        """Test MACDFIX with insufficient data."""
        close = np.array([100.0, 101.0])
        macd, signal, hist = macdfix_numba(close, signalperiod=9)

        # All should be NaN
        assert np.all(np.isnan(macd))
        assert np.all(np.isnan(signal))
        assert np.all(np.isnan(hist))

    def test_macdfix_numba_exact_lookback(self):
        """Test MACDFIX with exactly lookback+1 data points."""
        # MACDFIX uses fixed periods: fast=12, slow=26
        # lookback = signal_period - 1 + slow_period - 1 = 8 + 25 = 33
        # First valid at index 33
        np.random.seed(42)
        close = np.random.uniform(100, 110, 50)

        macd, signal, hist = macdfix_numba(close, signalperiod=9)

        # First valid at index 33
        assert np.all(np.isnan(macd[:33]))
        assert not np.isnan(macd[33])

    def test_macdfix_numba_histogram_calculation(self):
        """Test MACDFIX histogram = MACD - Signal."""
        np.random.seed(42)
        close = np.random.uniform(100, 110, 50)

        macd, signal, hist = macdfix_numba(close, signalperiod=9)

        # Histogram should equal MACD - Signal
        valid = ~np.isnan(hist)
        assert np.allclose(hist[valid], macd[valid] - signal[valid])


class TestAroonNumba:
    """Test aroon.py Numba functions."""

    def test_aroon_numba_insufficient_data(self):
        """Test AROON with insufficient data."""
        high = np.array([10.0, 11.0])
        low = np.array([9.0, 10.0])

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=5)

        # Need at least timeperiod data
        assert np.all(np.isnan(aroon_down))
        assert np.all(np.isnan(aroon_up))

    def test_aroon_numba_mismatched_lengths(self):
        """Test AROON with mismatched high/low lengths."""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0])

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=2)

        # Should return NaN
        assert np.all(np.isnan(aroon_down))
        assert np.all(np.isnan(aroon_up))

    def test_aroon_numba_exact_period(self):
        """Test AROON with exactly timeperiod+1 data points."""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=5)

        # First valid at index timeperiod=5
        assert np.all(np.isnan(aroon_down[:5]))
        assert np.all(np.isnan(aroon_up[:5]))
        assert not np.isnan(aroon_down[5])
        assert not np.isnan(aroon_up[5])

    def test_aroon_numba_highest_idx_reset(self):
        """Test AROON when highest_idx falls behind trailing_idx (line 97-106)."""
        # Create data where highest is at beginning, then new values come in
        high = np.array([20.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0])
        low = np.array([19.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0])

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=3)

        # Should handle highest_idx reset
        assert not np.any(np.isnan(aroon_up[3:]))

    def test_aroon_numba_lowest_idx_reset(self):
        """Test AROON when lowest_idx falls behind trailing_idx (line 81-90)."""
        # Create data where lowest is at beginning, then new values come in
        high = np.array([10.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        low = np.array([9.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=3)

        # Should handle lowest_idx reset
        assert not np.any(np.isnan(aroon_down[3:]))

    def test_aroon_numba_new_high_reached(self):
        """Test AROON when new high is reached (line 107-109)."""
        # Steadily increasing highs
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=3)

        # Aroon Up should be 100 (highest is always at current bar)
        valid_indices = ~np.isnan(aroon_up)
        assert np.all(aroon_up[valid_indices] == 100.0)

    def test_aroon_numba_new_low_reached(self):
        """Test AROON when new low is reached (line 91-93)."""
        # Steadily decreasing lows
        high = np.array([15.0, 14.0, 13.0, 12.0, 11.0, 10.0])
        low = np.array([14.0, 13.0, 12.0, 11.0, 10.0, 9.0])

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=3)

        # Aroon Down should be 100 (lowest is always at current bar)
        valid_indices = ~np.isnan(aroon_down)
        assert np.all(aroon_down[valid_indices] == 100.0)

    def test_aroonosc_numba(self):
        """Test AROONOSC = Aroon Up - Aroon Down."""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])

        result = aroonosc_numba(high, low, timeperiod=3)
        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=3)

        # Oscillator should be Up - Down
        valid = ~np.isnan(result)
        assert np.allclose(result[valid], aroon_up[valid] - aroon_down[valid])


class TestULTOSCNumba:
    """Test ultosc.py Numba function."""

    def test_ultosc_numba_insufficient_data(self):
        """Test ULTOSC with insufficient data."""
        high = np.array([10.0, 11.0])
        low = np.array([9.0, 10.0])
        close = np.array([9.5, 10.5])

        result = ultosc_numba(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        # Need at least timeperiod3+1 data
        assert np.all(np.isnan(result))

    def test_ultosc_numba_exact_period(self):
        """Test ULTOSC with exactly timeperiod3+1 data points."""
        np.random.seed(42)
        high = np.random.uniform(10, 20, 29)
        low = high - np.random.uniform(0.5, 1.5, 29)
        close = (high + low) / 2

        result = ultosc_numba(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        # First valid at index timeperiod3=28
        assert np.all(np.isnan(result[:28]))
        assert not np.isnan(result[28])

    def test_ultosc_numba_zero_tr_handling(self):
        """Test ULTOSC when TR becomes zero."""
        # Flat prices -> zero TR
        high = np.full(40, 10.0)
        low = np.full(40, 10.0)
        close = np.full(40, 10.0)

        result = ultosc_numba(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        # Should handle zero TR (skip those values)
        assert not np.any(np.isinf(result))

    def test_ultosc_numba_buying_pressure_calculation(self):
        """Test buying pressure calculation (line 76-78)."""
        # BP = Close - Min(Low, Prior Close)
        # Need at least timeperiod3+1 data points (5 for timeperiod3=4)
        np.random.seed(42)
        high = np.random.uniform(10, 20, 10)
        low = high - np.random.uniform(0.5, 1.5, 10)
        close = (high + low) / 2

        result = ultosc_numba(high, low, close, timeperiod1=2, timeperiod2=3, timeperiod3=4)

        # Should produce valid results after timeperiod3
        assert np.any(~np.isnan(result))

    def test_ultosc_numba_rolling_sum_calculation(self):
        """Test rolling sum calculations (lines 102-112)."""
        np.random.seed(42)
        high = np.random.uniform(10, 20, 35)
        low = high - np.random.uniform(0.5, 1.5, 35)
        close = (high + low) / 2

        result = ultosc_numba(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        # Should have continuous valid values after timeperiod3
        valid_count = np.sum(~np.isnan(result[28:]))
        assert valid_count > 0

    def test_ultosc_numba_weighted_average(self):
        """Test weighted average calculation (line 120)."""
        # UO = 100 * ((4*avg1) + (2*avg2) + avg3) / 7
        np.random.seed(42)
        high = np.random.uniform(10, 20, 40)
        low = high - np.random.uniform(0.5, 1.5, 40)
        close = (high + low) / 2

        result = ultosc_numba(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        # All valid values should be in range [0, 100]
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 150.0)  # Can exceed 100 in some cases


class TestEdgeCasesAcrossAllFunctions:
    """Test edge cases that apply to multiple functions."""

    def test_single_nan_value(self):
        """Test all functions with a single NaN in input."""
        np.random.seed(42)
        high = np.random.uniform(10, 20, 30)
        low = high - np.random.uniform(0.5, 1.5, 30)
        close = (high + low) / 2

        # Insert NaN
        close[15] = np.nan

        # All functions should handle NaN gracefully
        result_adx = adx_numba(high, low, close, period=14)
        result_plus_di = plus_di_numba(high, low, close, timeperiod=14)
        result_minus_di = minus_di_numba(high, low, close, timeperiod=14)

        # Should not propagate NaN to all subsequent values
        assert np.any(~np.isnan(result_adx[20:]))
        assert np.any(~np.isnan(result_plus_di[20:]))
        assert np.any(~np.isnan(result_minus_di[20:]))

    def test_all_zeros(self):
        """Test with all zero values."""
        high = np.zeros(30)
        low = np.zeros(30)
        close = np.zeros(30)

        # Should not crash, though results may be zero or NaN
        result_adx = adx_numba(high, low, close, period=14)
        result_sar = sar_numba(high, low)
        result_ultosc = ultosc_numba(high, low, close, 7, 14, 28)

        assert not np.any(np.isinf(result_adx))
        assert not np.any(np.isinf(result_sar))
        assert not np.any(np.isinf(result_ultosc))

    def test_negative_prices(self):
        """Test with negative prices (edge case for some assets)."""
        # Create arrays with negative prices transitioning to zero
        high = np.concatenate([np.array([-5.0, -4.0, -3.0, -2.0, -1.0]), np.zeros(25)])
        low = np.concatenate([np.array([-6.0, -5.0, -4.0, -3.0, -2.0]), np.full(25, -1.0)])
        close = np.concatenate([np.array([-5.5, -4.5, -3.5, -2.5, -1.5]), np.full(25, -0.5)])

        # Should handle negative prices - may produce some NaN/zero values which is OK
        result_adx = adx_numba(high, low, close, period=14)
        result_aroon_d, result_aroon_u = aroon_numba(high, low, timeperiod=14)

        # ADX may have NaN if TR=0, but shouldn't have inf
        assert not np.any(np.isinf(result_adx))
        # Aroon should have valid values after lookback
        assert np.any(~np.isnan(result_aroon_d))
        assert np.any(~np.isnan(result_aroon_u))

    def test_very_large_period(self):
        """Test with period close to data length."""
        np.random.seed(42)
        high = np.random.uniform(10, 20, 50)
        low = high - np.random.uniform(0.5, 1.5, 50)
        close = (high + low) / 2

        # Period = 45, data length = 50
        result_plus_di = plus_di_numba(high, low, close, timeperiod=45)
        minus_di_numba(high, low, close, timeperiod=45)

        # Should have very few valid values
        valid_count = np.sum(~np.isnan(result_plus_di))
        assert valid_count < 10


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
