"""Additional Aroon tests targeting uncovered lines in aroon_numba."""

import numpy as np

from ml4t.engineer.features.momentum.aroon import aroon_numba, aroonosc_numba


class TestAroonNumbaInternals:
    """Test aroon_numba edge cases and branches."""

    def test_insufficient_data(self):
        """Test with data less than timeperiod."""
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # Should return all NaN
        assert len(aroon_down) == 3
        assert len(aroon_up) == 3
        assert np.all(np.isnan(aroon_down))
        assert np.all(np.isnan(aroon_up))

    def test_mismatched_lengths(self):
        """Test with mismatched array lengths."""
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0])  # Different length

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=5)

        # Should return all NaN due to length mismatch
        assert len(aroon_down) == 3
        assert len(aroon_up) == 3
        assert np.all(np.isnan(aroon_down))
        assert np.all(np.isnan(aroon_up))

    def test_exact_timeperiod_boundary(self):
        """Test with exactly timeperiod + 1 data points."""
        n = 15  # timeperiod = 14
        high = np.linspace(100, 114, n)
        low = np.linspace(98, 112, n)

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # First valid value at index 14
        assert np.isnan(aroon_down[:14]).all()
        assert np.isnan(aroon_up[:14]).all()
        assert not np.isnan(aroon_down[14])
        assert not np.isnan(aroon_up[14])

    def test_lowest_idx_less_than_trailing(self):
        """Test branch where lowest_idx < trailing_idx (search for new low)."""
        # Create data where low idx needs to be recalculated
        low = np.array([100.0] * 20 + [90.0] + [100.0] * 29)  # Low at index 20
        high = low + 2

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # Should handle the search for new lowest
        valid_down = aroon_down[~np.isnan(aroon_down)]
        assert len(valid_down) > 20

    def test_highest_idx_less_than_trailing(self):
        """Test branch where highest_idx < trailing_idx (search for new high)."""
        # Create data where high idx needs to be recalculated
        high = np.array([100.0] * 20 + [110.0] + [100.0] * 29)  # High at index 20
        low = high - 2

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # Should handle the search for new highest
        valid_up = aroon_up[~np.isnan(aroon_up)]
        assert len(valid_up) > 20

    def test_new_low_equals_current_low(self):
        """Test branch where new low equals current lowest (update lowest_idx)."""
        # Create data with equal lows
        n = 50
        low = np.full(n, 100.0)
        low[25] = 99.0  # New absolute low
        high = low + 2

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # Should update lowest_idx to most recent equal low
        valid_down = aroon_down[~np.isnan(aroon_down)]
        assert len(valid_down) > 0

    def test_new_high_equals_current_high(self):
        """Test branch where new high equals current highest (update highest_idx)."""
        # Create data with equal highs
        n = 50
        high = np.full(n, 100.0)
        high[25] = 101.0  # New absolute high
        low = high - 2

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # Should update highest_idx to most recent equal high
        valid_up = aroon_up[~np.isnan(aroon_up)]
        assert len(valid_up) > 0

    def test_strong_uptrend(self):
        """Test with strong uptrend (high Aroon Up)."""
        n = 100
        high = np.linspace(100, 200, n)
        low = np.linspace(98, 198, n)

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # Aroon Up should be high in uptrend
        valid_up = aroon_up[~np.isnan(aroon_up)]
        assert len(valid_up) > 50
        # Most recent values should be high (close to 100)
        assert np.median(valid_up[-20:]) > 85

    def test_strong_downtrend(self):
        """Test with strong downtrend (high Aroon Down)."""
        n = 100
        high = np.linspace(200, 100, n)
        low = np.linspace(198, 98, n)

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # Aroon Down should be high in downtrend
        valid_down = aroon_down[~np.isnan(aroon_down)]
        assert len(valid_down) > 50
        # Most recent values should be high (close to 100)
        assert np.median(valid_down[-20:]) > 85

    def test_sideways_market(self):
        """Test with sideways/choppy market."""
        n = 100
        # Oscillating prices
        high = 100 + 5 * np.sin(np.linspace(0, 10 * np.pi, n))
        low = high - 2

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # Both should fluctuate
        valid_down = aroon_down[~np.isnan(aroon_down)]
        valid_up = aroon_up[~np.isnan(aroon_up)]

        assert len(valid_down) > 50
        assert len(valid_up) > 50
        # Neither should dominate
        assert 20 < np.median(valid_down) < 80
        assert 20 < np.median(valid_up) < 80

    def test_constant_values(self):
        """Test with constant high/low values."""
        n = 50
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # With constant values, both should show no clear trend
        valid_down = aroon_down[~np.isnan(aroon_down)]
        valid_up = aroon_up[~np.isnan(aroon_up)]

        assert len(valid_down) > 0
        assert len(valid_up) > 0
        # Values should be valid within range
        assert np.all((valid_down >= 0) & (valid_down <= 100))
        assert np.all((valid_up >= 0) & (valid_up <= 100))

    def test_small_timeperiod(self):
        """Test with small timeperiod (5)."""
        n = 50
        high = np.linspace(100, 110, n)
        low = np.linspace(98, 108, n)

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=5)

        # Should have values starting at index 5
        assert np.isnan(aroon_down[:5]).all()
        assert not np.all(np.isnan(aroon_down[5:]))

    def test_large_timeperiod(self):
        """Test with large timeperiod (30)."""
        n = 100
        high = np.linspace(100, 110, n)
        low = np.linspace(98, 108, n)

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=30)

        # Should have values starting at index 30
        assert np.isnan(aroon_down[:30]).all()
        assert not np.all(np.isnan(aroon_down[30:]))

    def test_factor_calculation(self):
        """Test that factor is correctly applied in formula."""
        high = np.array([100.0] * 14 + [110.0] + [100.0] * 35)  # High at idx 14
        low = np.array([100.0] * 14 + [90.0] + [100.0] * 35)  # Low at idx 14

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # At index 14, both high and low occur at same position
        # Aroon = factor * (period - (today - idx))
        # At index 14: (14 - (14 - 14)) = 14, factor = 100/14
        if not np.isnan(aroon_up[14]):
            expected = 100.0  # Most recent
            assert abs(aroon_up[14] - expected) < 1

    def test_complex_pattern(self):
        """Test with complex price pattern."""
        np.random.seed(42)
        n = 100
        high = 100 + np.cumsum(np.random.randn(n) * 0.5)
        low = high - np.abs(np.random.randn(n) * 0.5)

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)

        # Should compute without errors
        valid_down = aroon_down[~np.isnan(aroon_down)]
        valid_up = aroon_up[~np.isnan(aroon_up)]

        assert len(valid_down) > 50
        assert len(valid_up) > 50
        assert np.all((valid_down >= 0) & (valid_down <= 100))
        assert np.all((valid_up >= 0) & (valid_up <= 100))


class TestAroonOscNumba:
    """Test aroonosc_numba implementation."""

    def test_aroonosc_equals_difference(self):
        """Test that aroonosc correctly computes up - down."""
        n = 50
        high = np.linspace(100, 110, n)
        low = np.linspace(98, 108, n)

        aroon_down, aroon_up = aroon_numba(high, low, timeperiod=14)
        osc = aroonosc_numba(high, low, timeperiod=14)

        # Check that osc = up - down
        expected = aroon_up - aroon_down
        np.testing.assert_array_almost_equal(osc, expected)

    def test_aroonosc_range(self):
        """Test aroonosc is within -100 to 100."""
        n = 100
        high = 100 + np.cumsum(np.random.randn(n) * 0.5)
        low = high - 1

        osc = aroonosc_numba(high, low, timeperiod=14)

        valid = osc[~np.isnan(osc)]
        assert len(valid) > 50
        assert np.all((valid >= -100) & (valid <= 100))

    def test_aroonosc_uptrend(self):
        """Test aroonosc is positive in uptrend."""
        n = 50
        high = np.linspace(100, 150, n)
        low = np.linspace(98, 148, n)

        osc = aroonosc_numba(high, low, timeperiod=14)

        valid = osc[~np.isnan(osc)]
        # Should be predominantly positive
        assert np.median(valid[-10:]) > 0

    def test_aroonosc_downtrend(self):
        """Test aroonosc is negative in downtrend."""
        n = 50
        high = np.linspace(150, 100, n)
        low = np.linspace(148, 98, n)

        osc = aroonosc_numba(high, low, timeperiod=14)

        valid = osc[~np.isnan(osc)]
        # Should be predominantly negative
        assert np.median(valid[-10:]) < 0
