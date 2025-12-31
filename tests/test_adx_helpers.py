"""
Unit tests for ADX helper functions.

Tests the individual modular components created during the ADX refactoring
to ensure they maintain mathematical correctness and handle edge cases properly.
"""

import os

# Import the helper functions from the ADX module
# These functions are internal but we need to test them directly
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ml4t.engineer.features.momentum.adx import (
    _apply_wilders_smoothing,
    _calculate_directional_movement,
    _calculate_dx,
    _calculate_true_range,
)


class TestDirectionalMovement:
    """Test _calculate_directional_movement helper function."""

    def test_plus_dm_only(self):
        """Test case where only plus DM should be non-zero."""
        # Price moves up more significantly than down
        prev_high, current_high = 100.0, 102.0  # +2.0
        prev_low, current_low = 99.0, 100.0  # +1.0 (less movement)

        plus_dm, minus_dm = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )

        assert plus_dm == 2.0
        assert minus_dm == 0.0

    def test_minus_dm_only(self):
        """Test case where only minus DM should be non-zero."""
        # Price moves down more significantly than up
        prev_high, current_high = 100.0, 100.5  # +0.5
        prev_low, current_low = 99.0, 97.0  # -2.0 (more movement)

        plus_dm, minus_dm = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )

        assert plus_dm == 0.0
        assert minus_dm == 2.0

    def test_both_zero_equal_movement(self):
        """Test case where both movements are equal (both should be zero)."""
        # Equal positive and negative movement
        prev_high, current_high = 100.0, 101.5  # +1.5
        prev_low, current_low = 99.0, 97.5  # -1.5 (equal movement)

        plus_dm, minus_dm = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )

        assert plus_dm == 0.0
        assert minus_dm == 0.0

    def test_both_zero_no_movement(self):
        """Test case where there's no significant movement."""
        # No real directional movement
        prev_high, current_high = 100.0, 100.0  # 0.0
        prev_low, current_low = 99.0, 99.0  # 0.0

        plus_dm, minus_dm = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )

        assert plus_dm == 0.0
        assert minus_dm == 0.0

    def test_negative_movements(self):
        """Test case with negative price movements."""
        # Both prices move down, but high moves down more
        prev_high, current_high = 100.0, 98.0  # -2.0
        prev_low, current_low = 99.0, 98.5  # +0.5 (low goes up relative)

        plus_dm, minus_dm = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )

        # Since diff_p = -2.0 (negative) and diff_m = 0.5 (positive)
        # and diff_m > 0 and diff_p < diff_m, we get minus_dm = 0.5
        assert plus_dm == 0.0
        assert minus_dm == 0.5

    def test_edge_case_very_small_differences(self):
        """Test with very small price differences."""
        prev_high, current_high = 100.0, 100.0001
        prev_low, current_low = 99.0, 99.0002

        plus_dm, minus_dm = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )

        # diff_p = 0.0001, diff_m = -0.0002 (negative, so no movement)
        # Since diff_p > 0 and diff_p > diff_m, plus_dm = 0.0001
        assert abs(plus_dm - 0.0001) < 1e-10
        assert minus_dm == 0.0


class TestTrueRange:
    """Test _calculate_true_range helper function."""

    def test_high_low_range_largest(self):
        """Test case where high-low range is the largest."""
        high, low, prev_close = 102.0, 98.0, 99.5

        tr = _calculate_true_range(high, low, prev_close)

        # high-low = 4.0, high-prev_close = 2.5, low-prev_close = 1.5
        assert tr == 4.0

    def test_high_prev_close_largest(self):
        """Test case where high-prev_close gap is largest."""
        high, low, prev_close = 101.0, 100.0, 95.0

        tr = _calculate_true_range(high, low, prev_close)

        # high-low = 1.0, high-prev_close = 6.0, low-prev_close = 5.0
        assert tr == 6.0

    def test_low_prev_close_largest(self):
        """Test case where low-prev_close gap is largest."""
        high, low, prev_close = 101.0, 95.0, 100.0

        tr = _calculate_true_range(high, low, prev_close)

        # high-low = 6.0, high-prev_close = 1.0, low-prev_close = 5.0
        assert tr == 6.0

    def test_gap_down_scenario(self):
        """Test gap down scenario."""
        high, low, prev_close = 96.0, 94.0, 100.0

        tr = _calculate_true_range(high, low, prev_close)

        # high-low = 2.0, high-prev_close = 4.0, low-prev_close = 6.0
        assert tr == 6.0

    def test_gap_up_scenario(self):
        """Test gap up scenario."""
        high, low, prev_close = 105.0, 103.0, 100.0

        tr = _calculate_true_range(high, low, prev_close)

        # high-low = 2.0, high-prev_close = 5.0, low-prev_close = 3.0
        assert tr == 5.0

    def test_zero_range(self):
        """Test case with zero range (flat prices)."""
        high, low, prev_close = 100.0, 100.0, 100.0

        tr = _calculate_true_range(high, low, prev_close)

        assert tr == 0.0

    def test_negative_range_handling(self):
        """Test that absolute values are used correctly."""
        high, low, prev_close = 98.0, 97.0, 100.0

        tr = _calculate_true_range(high, low, prev_close)

        # high-low = 1.0, abs(high-prev_close) = 2.0, abs(low-prev_close) = 3.0
        assert tr == 3.0


class TestWildersSmoothing:
    """Test _apply_wilders_smoothing helper function."""

    def test_basic_smoothing(self):
        """Test basic Wilder's smoothing calculation."""
        prev_smoothed = 10.0
        new_value = 12.0
        period = 14

        result = _apply_wilders_smoothing(prev_smoothed, new_value, period)

        # Expected: 10.0 - (10.0 / 14) + 12.0 = 10.0 - 0.714... + 12.0 = 21.285...
        expected = prev_smoothed - (prev_smoothed / period) + new_value
        assert abs(result - expected) < 1e-10
        assert abs(result - 21.285714285714285) < 1e-10

    def test_zero_previous_value(self):
        """Test smoothing when previous value is zero."""
        prev_smoothed = 0.0
        new_value = 5.0
        period = 10

        result = _apply_wilders_smoothing(prev_smoothed, new_value, period)

        # Expected: 0.0 - (0.0 / 10) + 5.0 = 5.0
        assert result == 5.0

    def test_zero_new_value(self):
        """Test smoothing when new value is zero."""
        prev_smoothed = 8.0
        new_value = 0.0
        period = 5

        result = _apply_wilders_smoothing(prev_smoothed, new_value, period)

        # Expected: 8.0 - (8.0 / 5) + 0.0 = 8.0 - 1.6 = 6.4
        assert abs(result - 6.4) < 1e-10

    def test_different_periods(self):
        """Test smoothing with different periods."""
        prev_smoothed = 100.0
        new_value = 110.0

        for period in [2, 5, 10, 14, 20]:
            result = _apply_wilders_smoothing(prev_smoothed, new_value, period)
            expected = prev_smoothed - (prev_smoothed / period) + new_value
            assert abs(result - expected) < 1e-10

            # For period=2: 100 - 50 + 110 = 160
            # For period=20: 100 - 5 + 110 = 205
            if period == 2:
                assert abs(result - 160.0) < 1e-10
            elif period == 20:
                assert abs(result - 205.0) < 1e-10

    def test_smoothing_mathematical_relationship(self):
        """Test that smoothing follows the exact mathematical formula."""
        # Test the exact mathematical relationship
        prev_smoothed = 100.0
        new_value = 110.0
        period = 14

        result = _apply_wilders_smoothing(prev_smoothed, new_value, period)
        expected = prev_smoothed - (prev_smoothed / period) + new_value

        # Should match exact formula
        assert abs(result - expected) < 1e-15

        # Test with zero values
        result_zero = _apply_wilders_smoothing(0.0, 50.0, 10)
        expected_zero = 0.0 - (0.0 / 10) + 50.0  # = 50.0
        assert abs(result_zero - expected_zero) < 1e-15

    def test_large_numbers(self):
        """Test smoothing with large numbers."""
        prev_smoothed = 1e6
        new_value = 1.1e6
        period = 14

        result = _apply_wilders_smoothing(prev_smoothed, new_value, period)
        expected = prev_smoothed - (prev_smoothed / period) + new_value

        # Use relative tolerance for large numbers
        assert abs((result - expected) / expected) < 1e-10


class TestCalculateDX:
    """Test _calculate_dx helper function."""

    def test_equal_di_values(self):
        """Test DX calculation when +DI equals -DI."""
        plus_di = 25.0
        minus_di = 25.0

        dx = _calculate_dx(plus_di, minus_di)

        # When +DI = -DI, DX = 100 * (0 / 50) = 0
        assert dx == 0.0

    def test_plus_di_dominant(self):
        """Test DX when +DI is dominant."""
        plus_di = 40.0
        minus_di = 10.0

        dx = _calculate_dx(plus_di, minus_di)

        # DX = 100 * (|40 - 10| / (40 + 10)) = 100 * (30 / 50) = 60.0
        assert abs(dx - 60.0) < 1e-10

    def test_minus_di_dominant(self):
        """Test DX when -DI is dominant."""
        plus_di = 15.0
        minus_di = 35.0

        dx = _calculate_dx(plus_di, minus_di)

        # DX = 100 * (|15 - 35| / (15 + 35)) = 100 * (20 / 50) = 40.0
        assert abs(dx - 40.0) < 1e-10

    def test_zero_sum_di(self):
        """Test DX when both DI values are zero."""
        plus_di = 0.0
        minus_di = 0.0

        dx = _calculate_dx(plus_di, minus_di)

        # When sum is zero, should return 0 to avoid division by zero
        assert dx == 0.0

    def test_one_zero_di(self):
        """Test DX when one DI is zero."""
        plus_di = 30.0
        minus_di = 0.0

        dx = _calculate_dx(plus_di, minus_di)

        # DX = 100 * (|30 - 0| / (30 + 0)) = 100 * (30 / 30) = 100.0
        assert abs(dx - 100.0) < 1e-10

    def test_maximum_dx_value(self):
        """Test that DX can reach maximum value of 100."""
        plus_di = 50.0
        minus_di = 0.0

        dx = _calculate_dx(plus_di, minus_di)
        assert abs(dx - 100.0) < 1e-10

        # Test reverse case
        plus_di = 0.0
        minus_di = 25.0

        dx = _calculate_dx(plus_di, minus_di)
        assert abs(dx - 100.0) < 1e-10

    def test_dx_range_properties(self):
        """Test that DX is always in range [0, 100]."""
        test_cases = [
            (10.0, 20.0),
            (25.0, 25.0),
            (50.0, 10.0),
            (5.0, 45.0),
            (100.0, 0.0),
            (0.0, 100.0),
            (1.0, 1.0),
        ]

        for plus_di, minus_di in test_cases:
            dx = _calculate_dx(plus_di, minus_di)
            assert 0.0 <= dx <= 100.0, f"DX out of range for +DI={plus_di}, -DI={minus_di}"

    def test_dx_symmetry(self):
        """Test that DX is symmetric with respect to +DI and -DI."""
        plus_di = 30.0
        minus_di = 20.0

        dx1 = _calculate_dx(plus_di, minus_di)
        dx2 = _calculate_dx(minus_di, plus_di)

        # DX should be the same regardless of which DI is larger
        assert abs(dx1 - dx2) < 1e-10

    def test_very_small_di_values(self):
        """Test DX with very small DI values."""
        plus_di = 1e-6
        minus_di = 2e-6

        dx = _calculate_dx(plus_di, minus_di)

        # Should still calculate correctly
        expected = 100.0 * (1e-6 / 3e-6)  # 100 * (1/3) = 33.333...
        assert abs(dx - expected) < 1e-8

    def test_large_di_values(self):
        """Test DX with large DI values."""
        plus_di = 1e6
        minus_di = 2e6

        dx = _calculate_dx(plus_di, minus_di)

        expected = 100.0 * (1e6 / 3e6)  # 100 * (1/3) = 33.333...
        assert abs(dx - expected) < 1e-8


class TestHelperFunctionsIntegration:
    """Integration tests for helper functions working together."""

    def test_realistic_price_sequence(self):
        """Test helper functions with a realistic price sequence."""
        # Simulate a few periods of price data
        highs = [100.0, 102.0, 101.5, 103.0, 102.5]
        lows = [99.0, 100.5, 100.0, 101.0, 101.5]

        # Test directional movements
        for i in range(1, len(highs)):
            plus_dm, minus_dm = _calculate_directional_movement(
                highs[i - 1],
                highs[i],
                lows[i - 1],
                lows[i],
            )

            # Directional movements should be non-negative
            assert plus_dm >= 0.0
            assert minus_dm >= 0.0

            # At most one should be non-zero (TA-Lib logic)
            assert plus_dm == 0.0 or minus_dm == 0.0

    def test_true_range_sequence(self):
        """Test True Range calculations across a price sequence."""
        highs = [100.0, 102.0, 101.5, 103.0, 102.5]
        lows = [99.0, 100.5, 100.0, 101.0, 101.5]
        closes = [99.5, 101.0, 100.5, 102.0, 102.0]

        for i in range(1, len(highs)):
            tr = _calculate_true_range(highs[i], lows[i], closes[i - 1])

            # True Range should always be positive or zero
            assert tr >= 0.0

            # True Range should be at least as large as high-low range
            hl_range = highs[i] - lows[i]
            assert tr >= hl_range

    def test_smoothing_accumulation(self):
        """Test accumulating values with Wilder's smoothing."""
        values = [5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0]
        period = 3

        # Start with first value
        smoothed = values[0]

        # Apply smoothing for subsequent values
        for i in range(1, len(values)):
            prev_smoothed = smoothed
            smoothed = _apply_wilders_smoothing(smoothed, values[i], period)

            # Smoothed value should be positive
            assert smoothed >= 0.0

            # Check the mathematical relationship
            expected = prev_smoothed - (prev_smoothed / period) + values[i]
            assert abs(smoothed - expected) < 1e-10

    def test_dx_calculation_sequence(self):
        """Test DX calculations with varying DI values."""
        plus_di_values = [20.0, 25.0, 30.0, 15.0, 35.0]
        minus_di_values = [15.0, 20.0, 10.0, 25.0, 5.0]

        dx_values = []
        for plus_di, minus_di in zip(plus_di_values, minus_di_values, strict=False):
            dx = _calculate_dx(plus_di, minus_di)
            dx_values.append(dx)

            # Each DX should be in valid range
            assert 0.0 <= dx <= 100.0

        # Check that we get reasonable variation
        assert max(dx_values) > min(dx_values)  # Should have some variation

    def test_helper_functions_consistency(self):
        """Test that helper functions produce consistent results."""
        # Test multiple calls with same inputs
        prev_high, current_high = 100.0, 102.0
        prev_low, current_low = 99.0, 100.5

        # Multiple calls should return identical results
        result1 = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )
        result2 = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )

        assert result1 == result2

        # Same for other functions
        high, low, prev_close = 102.0, 100.5, 99.5
        tr1 = _calculate_true_range(high, low, prev_close)
        tr2 = _calculate_true_range(high, low, prev_close)
        assert tr1 == tr2

        # Wilder's smoothing
        smoothed1 = _apply_wilders_smoothing(10.0, 12.0, 14)
        smoothed2 = _apply_wilders_smoothing(10.0, 12.0, 14)
        assert smoothed1 == smoothed2

        # DX calculation
        dx1 = _calculate_dx(25.0, 15.0)
        dx2 = _calculate_dx(25.0, 15.0)
        assert dx1 == dx2


class TestHelperFunctionsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_values(self):
        """Test helper functions with extreme values."""
        # Very large numbers
        large_val = 1e10
        plus_dm, minus_dm = _calculate_directional_movement(
            large_val,
            large_val * 1.1,
            large_val * 0.9,
            large_val,
        )
        assert plus_dm >= 0.0 and minus_dm >= 0.0

        # Very small numbers
        small_val = 1e-10
        tr = _calculate_true_range(small_val * 2, small_val, small_val * 1.5)
        assert tr >= 0.0

        # Large period for smoothing - result should be close to original + (new - original/period)
        result = _apply_wilders_smoothing(100.0, 110.0, 1000)
        expected = 100.0 - (100.0 / 1000) + 110.0  # ≈ 209.9
        assert abs(result - expected) < 1e-10

        # Extreme DI values
        dx = _calculate_dx(1e8, 1e-8)
        assert 0.0 <= dx <= 100.0

    def test_precision_handling(self):
        """Test precision with floating point arithmetic."""
        # Test with values that might cause precision issues
        prev_high = 100.00000001
        current_high = 100.00000002
        prev_low = 99.99999999
        current_low = 100.00000001

        plus_dm, minus_dm = _calculate_directional_movement(
            prev_high,
            current_high,
            prev_low,
            current_low,
        )

        # Should handle tiny differences correctly
        assert plus_dm >= 0.0 and minus_dm >= 0.0

        # Test True Range precision
        tr = _calculate_true_range(100.00000001, 99.99999999, 100.0)
        assert tr >= 0.0

        # Test DX with very close values
        dx = _calculate_dx(25.0000001, 25.0000002)
        assert 0.0 <= dx <= 100.0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
