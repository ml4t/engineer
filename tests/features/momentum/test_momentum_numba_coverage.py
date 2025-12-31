"""
Comprehensive Numba implementation tests for momentum module.

This test suite directly exercises Numba code paths to maximize coverage
of lines 40-94 (RSI), 51-144 (MFI), 49-186 (SAR), 48-287 (ADX), and 34-191 (MACD).
"""

import numpy as np

from ml4t.engineer.features.momentum.adx import (
    _apply_wilders_smoothing,
    _calculate_directional_movement,
    _calculate_dx,
    _calculate_true_range,
    adx_numba,
)
from ml4t.engineer.features.momentum.macd import (
    _int_ema_talib_nb,
    macd_histogram_numba,
    macd_numba,
    macd_signal_numba,
)
from ml4t.engineer.features.momentum.mfi import mfi_numba
from ml4t.engineer.features.momentum.rsi import rsi_numba
from ml4t.engineer.features.momentum.sar import sar_numba


class TestRSINumbaDirectCoverage:
    """Test RSI Numba implementation directly to cover lines 40-94."""

    def test_rsi_numba_insufficient_data(self):
        """Test with data shorter than period (line 43-44)."""
        close = np.array([44.0, 44.34, 44.09])
        result = rsi_numba(close, period=5)
        assert np.all(np.isnan(result))

    def test_rsi_numba_exact_period_length(self):
        """Test with data exactly equal to period (line 43-44)."""
        close = np.array([44.0, 44.34, 44.09, 44.15, 43.61])
        result = rsi_numba(close, period=5)
        assert np.all(np.isnan(result))

    def test_rsi_numba_initial_sums(self):
        """Test initial gain/loss calculation (lines 46-56)."""
        # Construct data with known gains and losses
        close = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0])
        result = rsi_numba(close, period=3)
        # Should calculate initial sums: gain=4, loss=2
        assert not np.isnan(result[3])

    def test_rsi_numba_first_rsi_with_zero_loss(self):
        """Test first RSI when avg_loss is zero but avg_gain is also zero (lines 64-66)."""
        # Flat prices -> no gains, no losses
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        result = rsi_numba(close, period=3)
        assert result[3] == 0.0

    def test_rsi_numba_first_rsi_with_zero_loss_nonzero_gain(self):
        """Test first RSI when avg_loss is zero but avg_gain is not (lines 67-68)."""
        # Only gains, no losses
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = rsi_numba(close, period=3)
        assert result[3] == 100.0

    def test_rsi_numba_subsequent_positive_change(self):
        """Test Wilder's smoothing with positive changes (lines 78-80)."""
        close = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 107.0, 109.0])
        result = rsi_numba(close, period=3)
        # Should trigger lines 78-80 for positive changes
        assert not np.isnan(result[5])

    def test_rsi_numba_subsequent_negative_change(self):
        """Test Wilder's smoothing with negative changes (lines 81-83)."""
        close = np.array([100.0, 102.0, 104.0, 106.0, 104.0, 102.0, 100.0])
        result = rsi_numba(close, period=3)
        # Should trigger lines 81-83 for negative changes
        assert not np.isnan(result[5])

    def test_rsi_numba_subsequent_zero_loss(self):
        """Test subsequent RSI when avg_loss becomes zero (lines 85-92)."""
        # Strong uptrend -> avg_loss approaches zero
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        result = rsi_numba(close, period=3)
        # Should handle zero loss case (lines 89-92)
        assert np.all(result[3:] <= 100.0)

    def test_rsi_numba_alternating_values(self):
        """Test with alternating gains/losses."""
        close = np.array([100.0, 102.0, 100.0, 102.0, 100.0, 102.0, 100.0])
        result = rsi_numba(close, period=3)
        # Should exercise both gain and loss paths
        assert not np.all(np.isnan(result[3:]))
        assert np.all((result[3:] >= 0.0) & (result[3:] <= 100.0) | np.isnan(result[3:]))

    def test_rsi_numba_extreme_volatility(self):
        """Test with extreme price swings."""
        close = np.array([100.0, 150.0, 50.0, 200.0, 25.0, 175.0])
        result = rsi_numba(close, period=3)
        assert not np.isnan(result[3])

    def test_rsi_numba_small_changes(self):
        """Test with very small price changes."""
        close = np.array([100.0, 100.001, 100.002, 99.999, 100.001, 100.0])
        result = rsi_numba(close, period=3)
        assert not np.isnan(result[3])

    def test_rsi_numba_period_one(self):
        """Test with period=1 (edge case, line 43)."""
        close = np.array([100.0, 101.0, 102.0])
        result = rsi_numba(close, period=1)
        # Period 1 should calculate from index 1
        assert not np.isnan(result[1])


class TestMFINumbaDirectCoverage:
    """Test MFI Numba implementation directly to cover lines 51-144."""

    def test_mfi_numba_insufficient_data(self):
        """Test with data shorter than period (lines 54-55)."""
        high = np.array([127.0, 128.0])
        low = np.array([125.0, 126.0])
        close = np.array([126.0, 127.0])
        volume = np.array([5000.0, 5500.0])
        result = mfi_numba(high, low, close, volume, period=5)
        assert np.all(np.isnan(result))

    def test_mfi_numba_not_enough_data(self):
        """Test when n < period + 1 (lines 58-59)."""
        high = np.array([127.0, 128.0, 129.0])
        low = np.array([125.0, 126.0, 127.0])
        close = np.array([126.0, 127.0, 128.0])
        volume = np.array([5000.0, 5500.0, 6000.0])
        result = mfi_numba(high, low, close, volume, period=3)
        assert np.all(np.isnan(result))

    def test_mfi_numba_positive_flow_only(self):
        """Test with only positive money flow (lines 85-88)."""
        # Increasing typical prices
        high = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
        low = np.array([98.0, 100.0, 102.0, 104.0, 106.0])
        close = np.array([99.0, 101.0, 103.0, 105.0, 107.0])
        volume = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        result = mfi_numba(high, low, close, volume, period=3)
        # Should hit positive flow path
        assert result[3] == 100.0  # All positive flow -> MFI = 100

    def test_mfi_numba_negative_flow_only(self):
        """Test with only negative money flow (lines 89-92)."""
        # Decreasing typical prices
        high = np.array([108.0, 106.0, 104.0, 102.0, 100.0])
        low = np.array([106.0, 104.0, 102.0, 100.0, 98.0])
        close = np.array([107.0, 105.0, 103.0, 101.0, 99.0])
        volume = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        result = mfi_numba(high, low, close, volume, period=3)
        # Should hit negative flow path
        assert result[3] == 0.0  # All negative flow -> MFI = 0

    def test_mfi_numba_no_change_flow(self):
        """Test with no change in typical price (lines 93-95)."""
        # Constant typical prices
        high = np.array([102.0, 102.0, 102.0, 102.0, 102.0])
        low = np.array([98.0, 98.0, 98.0, 98.0, 98.0])
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        volume = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        result = mfi_numba(high, low, close, volume, period=3)
        # Should hit no-change path (lines 93-95)
        assert result[3] == 0.0

    def test_mfi_numba_small_total_flow(self):
        """Test with total money flow < 1.0 (lines 103-104)."""
        high = np.array([1.0, 1.01, 1.02, 1.03, 1.04])
        low = np.array([0.99, 1.0, 1.01, 1.02, 1.03])
        close = np.array([0.995, 1.005, 1.015, 1.025, 1.035])
        volume = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001])  # Very small volume
        result = mfi_numba(high, low, close, volume, period=3)
        # Should hit total_mf < 1.0 path (line 103)
        assert result[3] == 0.0

    def test_mfi_numba_circular_buffer_wraparound(self):
        """Test circular buffer wraparound (line 98, 141)."""
        high = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0])
        low = np.array([98.0, 100.0, 99.0, 101.0, 100.0, 102.0, 101.0, 103.0])
        close = np.array([99.0, 101.0, 100.0, 102.0, 101.0, 103.0, 102.0, 104.0])
        volume = np.array([1000.0, 1100.0, 1000.0, 1200.0, 1000.0, 1300.0, 1000.0, 1400.0])
        result = mfi_numba(high, low, close, volume, period=3)
        # Should exercise circular buffer logic
        assert not np.all(np.isnan(result[3:]))

    def test_mfi_numba_subsequent_positive_flow(self):
        """Test subsequent calculations with positive flow (lines 121-124)."""
        # Mix of positive and negative flows
        high = np.array([100.0, 102.0, 104.0, 103.0, 105.0, 107.0])
        low = np.array([98.0, 100.0, 102.0, 101.0, 103.0, 105.0])
        close = np.array([99.0, 101.0, 103.0, 102.0, 104.0, 106.0])
        volume = np.array([1000.0, 1100.0, 1200.0, 1100.0, 1300.0, 1400.0])
        result = mfi_numba(high, low, close, volume, period=3)
        assert not np.isnan(result[4])

    def test_mfi_numba_subsequent_negative_flow(self):
        """Test subsequent calculations with negative flow (lines 125-128)."""
        high = np.array([107.0, 105.0, 103.0, 104.0, 102.0, 100.0])
        low = np.array([105.0, 103.0, 101.0, 102.0, 100.0, 98.0])
        close = np.array([106.0, 104.0, 102.0, 103.0, 101.0, 99.0])
        volume = np.array([1400.0, 1300.0, 1100.0, 1200.0, 1000.0, 900.0])
        result = mfi_numba(high, low, close, volume, period=3)
        assert not np.isnan(result[4])

    def test_mfi_numba_subsequent_no_change(self):
        """Test subsequent calculations with no price change (lines 129-131)."""
        high = np.array([102.0, 102.0, 102.0, 102.0, 102.0, 102.0])
        low = np.array([98.0, 98.0, 98.0, 98.0, 98.0, 98.0])
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        volume = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        result = mfi_numba(high, low, close, volume, period=3)
        assert result[3] == 0.0

    def test_mfi_numba_subsequent_small_total(self):
        """Test subsequent calculations with total_mf < 1.0 (lines 135-136)."""
        high = np.array([1.0, 1.01, 1.0, 1.01, 1.0, 1.01])
        low = np.array([0.99, 1.0, 0.99, 1.0, 0.99, 1.0])
        close = np.array([0.995, 1.005, 0.995, 1.005, 0.995, 1.005])
        volume = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
        result = mfi_numba(high, low, close, volume, period=3)
        # Should hit small total flow path repeatedly
        assert not np.all(np.isnan(result[3:]))


class TestSARNumbaDirectCoverage:
    """Test SAR Numba implementation directly to cover lines 49-186."""

    def test_sar_numba_insufficient_data(self):
        """Test with less than 2 data points (lines 56-57)."""
        high = np.array([10.0])
        low = np.array([9.0])
        result = sar_numba(high, low)
        assert np.all(np.isnan(result))

    def test_sar_numba_mismatched_lengths(self):
        """Test with mismatched high/low lengths (lines 50-51)."""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0])
        result = sar_numba(high, low)
        assert np.all(np.isnan(result))

    def test_sar_numba_acceleration_exceeds_maximum(self):
        """Test when acceleration > maximum (lines 68-70)."""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
        result = sar_numba(high, low, acceleration=0.3, maximum=0.2)
        # Should adjust acceleration to maximum
        assert not np.isnan(result[1])

    def test_sar_numba_initial_short_position(self):
        """Test initial direction as short (lines 73-77, 89-91)."""
        # Make dm_minus > 0 to trigger short
        high = np.array([15.0, 14.0, 13.0, 12.0, 11.0])
        low = np.array([14.0, 13.0, 12.0, 11.0, 10.0])
        result = sar_numba(high, low)
        # Should start in short position (is_long=0)
        assert not np.isnan(result[1])

    def test_sar_numba_initial_long_position(self):
        """Test initial direction as long (lines 86-88)."""
        # Make dm_minus == 0 to trigger long
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
        result = sar_numba(high, low)
        # Should start in long position (is_long=1)
        assert not np.isnan(result[1])

    def test_sar_numba_long_switch_to_short(self):
        """Test switching from long to short (lines 106-123)."""
        # Start long, then price drops through SAR
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0])
        result = sar_numba(high, low)
        # Should trigger switch from long to short
        assert not np.all(np.isnan(result[1:]))

    def test_sar_numba_long_no_switch(self):
        """Test long position without switch (lines 128-143)."""
        # Sustained uptrend
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        result = sar_numba(high, low)
        # Should stay long, adjusting SAR
        assert not np.all(np.isnan(result[1:]))

    def test_sar_numba_long_new_extreme(self):
        """Test long position with new extreme point (lines 133-136)."""
        # New highs trigger AF increase
        high = np.array([10.0, 11.0, 12.0, 13.0, 15.0, 18.0, 22.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 14.0, 17.0, 21.0])
        result = sar_numba(high, low)
        # Should hit new extreme path (line 133)
        assert not np.isnan(result[2])

    def test_sar_numba_short_switch_to_long(self):
        """Test switching from short to long (lines 145-166)."""
        # Start short, then price rises through SAR
        high = np.array([15.0, 14.0, 13.0, 12.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([14.0, 13.0, 12.0, 11.0, 10.0, 11.0, 12.0, 13.0])
        result = sar_numba(high, low)
        # Should trigger switch from short to long
        assert not np.all(np.isnan(result[1:]))

    def test_sar_numba_short_no_switch(self):
        """Test short position without switch (lines 167-182)."""
        # Sustained downtrend
        high = np.array([17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0])
        low = np.array([16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0])
        result = sar_numba(high, low)
        # Should stay short, adjusting SAR
        assert not np.all(np.isnan(result[1:]))

    def test_sar_numba_short_new_extreme(self):
        """Test short position with new extreme point (lines 172-175)."""
        # New lows trigger AF increase
        high = np.array([22.0, 18.0, 15.0, 13.0, 10.0, 7.0, 5.0])
        low = np.array([21.0, 17.0, 14.0, 12.0, 9.0, 6.0, 4.0])
        result = sar_numba(high, low)
        # Should hit new extreme path (line 172)
        assert not np.isnan(result[2])

    def test_sar_numba_multiple_switches(self):
        """Test multiple switches between long and short."""
        # Oscillating prices
        high = np.array([10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 12.0, 11.0])
        low = np.array([9.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 11.0, 10.0])
        result = sar_numba(high, low)
        assert not np.all(np.isnan(result[1:]))

    def test_sar_numba_extreme_acceleration(self):
        """Test with maximum acceleration factor."""
        high = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
        low = np.array([9.0, 14.0, 19.0, 24.0, 29.0, 34.0, 39.0])
        result = sar_numba(high, low, acceleration=0.2, maximum=0.2)
        assert not np.all(np.isnan(result[1:]))


class TestADXNumbaDirectCoverage:
    """Test ADX Numba implementation and helper functions to cover lines 48-287."""

    def test_calculate_directional_movement_case1(self):
        """Test +DM dominant case (lines 55-57)."""
        plus_dm, minus_dm = _calculate_directional_movement(10.0, 12.0, 8.0, 8.5)
        # diff_p=2.0, diff_m=0 -> Case 1: +DM=diffP, -DM=0
        assert plus_dm == 2.0
        assert minus_dm == 0.0

    def test_calculate_directional_movement_case2(self):
        """Test case with no directional movement (both diffs negative)."""
        plus_dm, minus_dm = _calculate_directional_movement(12.0, 11.0, 8.0, 9.0)
        # diff_p = 11 - 12 = -1 (negative, no upward movement)
        # diff_m = 8 - 9 = -1 (negative, no downward movement)
        # Both negative -> both DM = 0
        assert plus_dm == 0.0
        assert minus_dm == 0.0

    def test_calculate_directional_movement_case3(self):
        """Test +DM > -DM case (lines 55-57)."""
        plus_dm, minus_dm = _calculate_directional_movement(10.0, 13.0, 8.0, 8.5)
        # diff_p=3.0, diff_m=0.5 -> +DM dominant
        assert plus_dm == 3.0
        assert minus_dm == 0.0

    def test_calculate_directional_movement_case4(self):
        """Test +DM positive but -DM negative (only +DM counts)."""
        plus_dm, minus_dm = _calculate_directional_movement(12.0, 12.5, 8.0, 9.0)
        # diff_p = 12.5 - 12.0 = 0.5 (positive, upward movement)
        # diff_m = 8.0 - 9.0 = -1.0 (negative, no downward movement)
        # +DM wins since it's positive and -DM is negative
        assert plus_dm == 0.5
        assert minus_dm == 0.0

    def test_calculate_directional_movement_case5(self):
        """Test no movement case (lines 59)."""
        plus_dm, minus_dm = _calculate_directional_movement(12.0, 12.0, 8.0, 8.0)
        # No change -> both zero
        assert plus_dm == 0.0
        assert minus_dm == 0.0

    def test_calculate_true_range(self):
        """Test True Range calculation (lines 80-83)."""
        tr = _calculate_true_range(12.0, 10.0, 11.0)
        # max(12-10, |12-11|, |10-11|) = max(2, 1, 1) = 2
        assert tr == 2.0

    def test_calculate_true_range_gap_up(self):
        """Test True Range with gap up."""
        tr = _calculate_true_range(15.0, 13.0, 11.0)
        # max(15-13, |15-11|, |13-11|) = max(2, 4, 2) = 4
        assert tr == 4.0

    def test_calculate_true_range_gap_down(self):
        """Test True Range with gap down."""
        tr = _calculate_true_range(10.0, 8.0, 13.0)
        # max(10-8, |10-13|, |8-13|) = max(2, 3, 5) = 5
        assert tr == 5.0

    def test_apply_wilders_smoothing(self):
        """Test Wilder's smoothing formula (lines 110)."""
        result = _apply_wilders_smoothing(100.0, 10.0, 14)
        # 100 - (100/14) + 10 = 100 - 7.142857 + 10 = 102.857143
        expected = 100.0 - (100.0 / 14) + 10.0
        assert abs(result - expected) < 1e-6

    def test_calculate_dx_normal(self):
        """Test DX calculation (lines 130-132)."""
        dx = _calculate_dx(25.0, 15.0)
        # |25-15| / (25+15) * 100 = 10/40 * 100 = 25
        assert abs(dx - 25.0) < 1e-6

    def test_calculate_dx_zero_sum(self):
        """Test DX when sum is zero (lines 130-131)."""
        dx = _calculate_dx(0.0, 0.0)
        assert dx == 0.0

    def test_adx_numba_insufficient_data(self):
        """Test ADX with insufficient data (lines 169-170)."""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])
        result = adx_numba(high, low, close, period=14)
        # Need at least 2*period-1 = 27 bars
        assert np.all(np.isnan(result))

    def test_adx_numba_exact_lookback(self):
        """Test ADX with exactly lookback bars."""
        # For period=3, lookback = 2*3-1 = 5
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
        close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5])
        result = adx_numba(high, low, close, period=3)
        # Should have first value at index 5
        assert not np.isnan(result[5])
        assert np.all(np.isnan(result[:5]))

    def test_adx_numba_initial_accumulation(self):
        """Test initial DM/TR accumulation (lines 186-207)."""
        # Small period to test initial loop
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5])
        result = adx_numba(high, low, close, period=3)
        assert not np.isnan(result[5])

    def test_adx_numba_dx_accumulation(self):
        """Test DX accumulation loop (lines 210-242)."""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5])
        result = adx_numba(high, low, close, period=3)
        # Should calculate sum_dx over period
        assert not np.isnan(result[5])

    def test_adx_numba_zero_tr(self):
        """Test when prev_tr is zero (lines 238)."""
        # Flat prices -> zero TR
        high = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        low = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        close = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        result = adx_numba(high, low, close, period=3)
        # With zero TR, DX calculation is skipped
        assert result[5] == 0.0

    def test_adx_numba_first_adx_output(self):
        """Test first ADX output (lines 245-248)."""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5])
        result = adx_numba(high, low, close, period=3)
        # First ADX = sum_dx / period
        assert not np.isnan(result[5])

    def test_adx_numba_subsequent_calculations(self):
        """Test subsequent ADX calculations (lines 251-285)."""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
        close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5])
        result = adx_numba(high, low, close, period=3)
        # Should calculate multiple ADX values
        assert not np.isnan(result[6])
        assert not np.isnan(result[7])

    def test_adx_numba_subsequent_zero_tr(self):
        """Test subsequent calculation with zero TR (lines 277)."""
        # Start with movement, then flat
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 14.0, 14.0, 14.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 14.0, 14.0])
        close = np.array([9.5, 10.5, 11.5, 12.5, 13.5, 14.0, 14.0, 14.0])
        result = adx_numba(high, low, close, period=3)
        # Should handle zero TR in subsequent calculations
        assert not np.all(np.isnan(result[5:]))


class TestMACDNumbaDirectCoverage:
    """Test MACD Numba implementation and helper functions to cover lines 34-191."""

    def test_int_ema_talib_insufficient_data(self):
        """Test INT_EMA with insufficient data (lines 43-45)."""
        close = np.array([100.0, 101.0])
        result = _int_ema_talib_nb(close, start_idx=5, end_idx=10, period=5, k=0.333)
        # start_idx > len(close) -> all NaN
        assert np.all(np.isnan(result))

    def test_int_ema_talib_start_after_end(self):
        """Test INT_EMA when start > end (lines 43-45)."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = _int_ema_talib_nb(close, start_idx=10, end_idx=5, period=3, k=0.5)
        assert np.all(np.isnan(result))

    def test_int_ema_talib_seed_calculation(self):
        """Test seed calculation (lines 48-56)."""
        close = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])
        result = _int_ema_talib_nb(close, start_idx=2, end_idx=5, period=3, k=0.5)
        # Seed at index 2 = mean(100, 102, 104) = 102
        assert abs(result[2] - 102.0) < 1e-6

    def test_int_ema_talib_seed_output(self):
        """Test that seed is output at start_idx (line 59)."""
        close = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
        result = _int_ema_talib_nb(close, start_idx=2, end_idx=4, period=3, k=0.5)
        assert not np.isnan(result[2])

    def test_int_ema_talib_unstable_period_skip(self):
        """Test unstable period skip (lines 62-65)."""
        close = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0])
        result = _int_ema_talib_nb(close, start_idx=3, end_idx=6, period=3, k=0.5)
        # Should skip calculations until today > start_idx
        assert not np.isnan(result[3])

    def test_int_ema_talib_remaining_range(self):
        """Test remaining range calculation (lines 68-75)."""
        close = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0])
        result = _int_ema_talib_nb(close, start_idx=3, end_idx=7, period=3, k=0.5)
        # Should output from start_idx+1 to end_idx
        assert not np.isnan(result[4])
        assert not np.isnan(result[5])

    def test_macd_numba_swap_periods(self):
        """Test period swapping when slow < fast (lines 111-114)."""
        close = np.array(np.random.randn(100).cumsum() + 100)
        # Intentionally reversed periods
        result = macd_numba(close, fast_period=26, slow_period=12, signal_period=9)
        # Should swap them internally
        assert not np.all(np.isnan(result))

    def test_macd_numba_k_calculation(self):
        """Test k calculation (lines 117-118)."""
        close = np.array(np.random.randn(50).cumsum() + 100)
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # k values should be calculated as 2/(period+1)
        assert not np.all(np.isnan(result[34:]))  # First valid at 26+9-1=34

    def test_macd_numba_lookback_calculation(self):
        """Test lookback calculation (lines 121-122)."""
        close = np.array(np.random.randn(50).cumsum() + 100)
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # lookback_total = signal_period-1 + slow_period-1 = 9-1 + 26-1 = 33
        assert np.all(np.isnan(result[:33]))

    def test_macd_numba_insufficient_data(self):
        """Test with insufficient data (lines 128-129)."""
        close = np.array([100.0, 101.0, 102.0])
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)
        assert np.all(np.isnan(result))

    def test_macd_numba_temp_integer(self):
        """Test tempInteger calculation (line 132)."""
        close = np.array(np.random.randn(50).cumsum() + 100)
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # temp_integer = start_idx - lookback_signal
        assert not np.all(np.isnan(result))

    def test_macd_numba_slow_ema(self):
        """Test slow EMA calculation (line 135)."""
        close = np.array(np.random.randn(50).cumsum() + 100)
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # Should calculate slow EMA
        assert not np.all(np.isnan(result))

    def test_macd_numba_fast_ema(self):
        """Test fast EMA calculation (line 138)."""
        close = np.array(np.random.randn(50).cumsum() + 100)
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # Should calculate fast EMA
        assert not np.all(np.isnan(result))

    def test_macd_numba_difference_calculation(self):
        """Test MACD line calculation (lines 141-143)."""
        close = np.array(np.random.randn(50).cumsum() + 100)
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # MACD = fast - slow
        assert not np.all(np.isnan(result[34:]))

    def test_macd_numba_copy_result(self):
        """Test result copying (lines 148-153)."""
        close = np.array(np.random.randn(50).cumsum() + 100)
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # Should copy from buffer to result
        assert not np.all(np.isnan(result[34:]))

    def test_macd_signal_numba_delegation(self):
        """Test signal line calculation (lines 188-191)."""
        close = np.array(np.random.randn(50).cumsum() + 100)
        result = macd_signal_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # Should apply EMA to MACD line
        assert not np.all(np.isnan(result[34:]))

    def test_macd_histogram_numba_calculation(self):
        """Test histogram calculation (lines 220-228)."""
        close = np.array(np.random.randn(50).cumsum() + 100)
        result = macd_histogram_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # Histogram = MACD - Signal
        assert not np.all(np.isnan(result[34:]))

    def test_macd_histogram_nan_handling(self):
        """Test histogram NaN handling (line 227)."""
        close = np.array([100.0] * 10)
        result = macd_histogram_numba(close, fast_period=3, slow_period=6, signal_period=3)
        # Should handle NaN values correctly
        assert not np.all(np.isnan(result))

    def test_macd_numba_constant_prices(self):
        """Test MACD with constant prices."""
        close = np.array([100.0] * 50)
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # Constant prices -> MACD should be ~0
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.allclose(valid, 0.0, atol=1e-10)

    def test_macd_numba_trending_prices(self):
        """Test MACD with trending prices."""
        close = np.linspace(100, 150, 50)
        result = macd_numba(close, fast_period=12, slow_period=26, signal_period=9)
        # Uptrend -> MACD should be positive
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert valid[-1] > 0

    def test_macd_numba_small_periods(self):
        """Test MACD with small periods."""
        close = np.array(np.random.randn(20).cumsum() + 100)
        result = macd_numba(close, fast_period=3, slow_period=6, signal_period=3)
        # Should calculate with small periods
        assert not np.all(np.isnan(result))

    def test_macd_signal_constant_prices(self):
        """Test signal line with constant prices."""
        close = np.array([100.0] * 50)
        result = macd_signal_numba(close, fast_period=12, slow_period=26, signal_period=9)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.allclose(valid, 0.0, atol=1e-10)


class TestIntegrationEdgeCases:
    """Integration tests for complex edge cases across all functions."""

    def test_rsi_extreme_values(self):
        """Test RSI with extreme price values."""
        close = np.array([1e-10, 1e10, 1e-5, 1e8, 1e-3])
        result = rsi_numba(close, period=3)
        assert not np.all(np.isnan(result[3:]))

    def test_mfi_zero_volume(self):
        """Test MFI with zero volume."""
        high = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
        low = np.array([98.0, 100.0, 102.0, 104.0, 106.0])
        close = np.array([99.0, 101.0, 103.0, 105.0, 107.0])
        volume = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = mfi_numba(high, low, close, volume, period=3)
        assert result[3] == 0.0  # Zero volume -> zero MFI

    def test_sar_extreme_parameters(self):
        """Test SAR with extreme but valid parameters."""
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0])
        result = sar_numba(high, low, acceleration=0.001, maximum=0.001)
        assert not np.all(np.isnan(result[1:]))

    def test_adx_minimal_period(self):
        """Test ADX with minimal valid period."""
        # period=1 -> lookback=1
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])
        result = adx_numba(high, low, close, period=1)
        assert not np.isnan(result[1])

    def test_macd_equal_fast_slow(self):
        """Test MACD behavior when fast approaches slow (after swap)."""
        close = np.array(np.random.randn(50).cumsum() + 100)
        # After swap: fast=25, slow=26
        result = macd_numba(close, fast_period=25, slow_period=26, signal_period=9)
        # Very small difference expected
        assert not np.all(np.isnan(result))
