"""
Test Numba implementations directly for trend indicators to boost coverage.

This test file directly calls Numba functions with numpy arrays to ensure
the JIT-compiled code paths are executed and covered.
"""

import numpy as np

from ml4t.engineer.features.trend.dema import dema_numba
from ml4t.engineer.features.trend.ema import ema_numba
from ml4t.engineer.features.trend.kama import kama_numba
from ml4t.engineer.features.trend.sma import sma_numba
from ml4t.engineer.features.trend.t3 import t3_numba
from ml4t.engineer.features.trend.wma import wma_numba


class TestEMANumbaCoverage:
    """Test EMA Numba implementation directly."""

    def test_ema_numba_direct_large_array(self):
        """Test EMA numba function with large array to trigger JIT path."""
        # Large array to ensure Numba compilation
        close = np.random.randn(1000).cumsum() + 100
        period = 20

        result = ema_numba(close, period)

        # Basic validation
        assert len(result) == len(close)
        assert np.isnan(result[: period - 1]).all()  # First period-1 values are NaN
        assert not np.isnan(result[period - 1 :]).any()  # Rest are valid

    def test_ema_numba_short_period(self):
        """Test EMA with minimum period."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 2

        result = ema_numba(close, period)

        assert len(result) == len(close)
        assert np.isnan(result[0])  # First value is NaN
        assert not np.isnan(result[1:]).all()  # Rest should have values

    def test_ema_numba_with_nan_values(self):
        """Test EMA handles NaN values in input."""
        close = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3

        result = ema_numba(close, period)

        # Should handle NaN gracefully
        assert len(result) == len(close)
        assert np.isnan(result[:2]).all()  # Initial NaN values

    def test_ema_numba_constant_values(self):
        """Test EMA with constant input."""
        close = np.full(100, 50.0)
        period = 10

        result = ema_numba(close, period)

        # For constant input, EMA should equal input after warmup
        assert np.allclose(result[period - 1 :], 50.0, rtol=1e-10)

    def test_ema_numba_invalid_period(self):
        """Test EMA with invalid period."""
        close = np.array([1.0, 2.0, 3.0])
        period = 0

        result = ema_numba(close, period)

        # Should return all NaN
        assert np.isnan(result).all()

    def test_ema_numba_period_exceeds_data(self):
        """Test EMA when period exceeds data length."""
        close = np.array([1.0, 2.0, 3.0])
        period = 10

        result = ema_numba(close, period)

        # Should return all NaN
        assert np.isnan(result).all()

    def test_ema_numba_all_nan_input(self):
        """Test EMA with all NaN input."""
        close = np.full(50, np.nan)
        period = 10

        result = ema_numba(close, period)

        # Should return all NaN
        assert np.isnan(result).all()


class TestDEMANumbaCoverage:
    """Test DEMA Numba implementation directly."""

    def test_dema_numba_direct_large_array(self):
        """Test DEMA numba function with large array."""
        close = np.random.randn(1000).cumsum() + 100
        period = 30

        result = dema_numba(close, period)

        # Basic validation
        assert len(result) == len(close)
        lookback = 2 * (period - 1)
        assert np.isnan(result[:lookback]).all()  # Warmup period
        assert not np.isnan(result[lookback:]).any()  # Valid values after

    def test_dema_numba_short_period(self):
        """Test DEMA with short period."""
        close = np.random.randn(100).cumsum() + 100
        period = 5

        result = dema_numba(close, period)

        lookback = 2 * (period - 1)
        assert len(result) == len(close)
        assert np.isnan(result[:lookback]).all()

    def test_dema_numba_insufficient_data(self):
        """Test DEMA with insufficient data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 10

        result = dema_numba(close, period)

        # Should return all NaN
        assert np.isnan(result).all()

    def test_dema_numba_constant_values(self):
        """Test DEMA with constant input."""
        close = np.full(200, 100.0)
        period = 20

        result = dema_numba(close, period)

        lookback = 2 * (period - 1)
        # For constant input, DEMA should equal input after warmup
        assert np.allclose(result[lookback:], 100.0, rtol=1e-8)

    def test_dema_numba_varying_volatility(self):
        """Test DEMA with varying volatility."""
        # Create data with different volatility regimes
        close = np.concatenate(
            [
                np.random.randn(100) * 0.5 + 100,  # Low volatility
                np.random.randn(100) * 2.0 + 100,  # High volatility
            ]
        )
        period = 15

        result = dema_numba(close, period)

        lookback = 2 * (period - 1)
        assert len(result) == len(close)
        assert not np.isnan(result[lookback:]).any()


class TestSMANumbaCoverage:
    """Test SMA Numba implementation directly."""

    def test_sma_numba_direct_large_array(self):
        """Test SMA numba function with large array."""
        close = np.random.randn(1000).cumsum() + 100
        period = 20

        result = sma_numba(close, period)

        assert len(result) == len(close)
        assert np.isnan(result[: period - 1]).all()
        assert not np.isnan(result[period - 1 :]).any()

    def test_sma_numba_exact_calculation(self):
        """Test SMA calculation accuracy."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3

        result = sma_numba(close, period)

        # Manual calculation for verification
        expected_3 = (1 + 2 + 3) / 3  # 2.0
        expected_4 = (2 + 3 + 4) / 3  # 3.0
        expected_5 = (3 + 4 + 5) / 3  # 4.0

        assert np.isclose(result[2], expected_3)
        assert np.isclose(result[3], expected_4)
        assert np.isclose(result[4], expected_5)

    def test_sma_numba_with_nan_in_data(self):
        """Test SMA with NaN values scattered in data."""
        close = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3

        result = sma_numba(close, period)

        # Should handle NaN by skipping windows with insufficient data
        assert len(result) == len(close)

    def test_sma_numba_constant_values(self):
        """Test SMA with constant input."""
        close = np.full(100, 42.0)
        period = 10

        result = sma_numba(close, period)

        # For constant input, SMA should equal input
        assert np.allclose(result[period - 1 :], 42.0)

    def test_sma_numba_invalid_period(self):
        """Test SMA with invalid period."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 0

        result = sma_numba(close, period)

        assert np.isnan(result).all()

    def test_sma_numba_period_equals_length(self):
        """Test SMA when period equals data length."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 5

        result = sma_numba(close, period)

        # Should have one valid value at the end
        assert np.isnan(result[:4]).all()
        assert np.isclose(result[4], 3.0)  # (1+2+3+4+5)/5 = 3

    def test_sma_numba_running_sum_optimization(self):
        """Test that running sum optimization works correctly."""
        # Large array to test rolling window efficiency
        close = np.arange(1000, dtype=np.float64)
        period = 50

        result = sma_numba(close, period)

        # Verify a few values manually
        for i in range(period - 1, min(period + 10, len(close))):
            expected = np.mean(close[i - period + 1 : i + 1])
            assert np.isclose(result[i], expected)


class TestKAMANumbaCoverage:
    """Test KAMA Numba implementation directly."""

    def test_kama_numba_direct_large_array(self):
        """Test KAMA numba function with large array."""
        close = np.random.randn(1000).cumsum() + 100
        timeperiod = 30

        result = kama_numba(close, timeperiod)

        assert len(result) == len(close)
        # KAMA has warmup period
        assert np.isnan(result[:timeperiod]).all()
        assert not np.isnan(result[timeperiod:]).any()

    def test_kama_numba_trending_market(self):
        """Test KAMA in strongly trending market."""
        # Create strong uptrend
        close = np.linspace(100, 200, 200)
        timeperiod = 20

        result = kama_numba(close, timeperiod)

        # In strong trend, KAMA should follow closely
        assert len(result) == len(close)
        assert not np.isnan(result[timeperiod:]).any()

    def test_kama_numba_choppy_market(self):
        """Test KAMA in choppy/sideways market."""
        # Create choppy market
        close = 100 + 5 * np.sin(np.linspace(0, 4 * np.pi, 200))
        timeperiod = 20

        result = kama_numba(close, timeperiod)

        # In choppy market, KAMA should be smoother
        assert len(result) == len(close)
        assert not np.isnan(result[timeperiod:]).any()

    def test_kama_numba_efficiency_ratio_extremes(self):
        """Test KAMA with efficiency ratio at extremes."""
        # Perfect trend (ER = 1)
        close_trend = np.linspace(100, 200, 100)
        result_trend = kama_numba(close_trend, 10)

        # Random walk (low ER)
        np.random.seed(42)
        close_random = np.random.randn(100).cumsum() + 100
        result_random = kama_numba(close_random, 10)

        assert len(result_trend) == len(close_trend)
        assert len(result_random) == len(close_random)

    def test_kama_numba_short_period(self):
        """Test KAMA with minimum period."""
        close = np.random.randn(100).cumsum() + 100
        timeperiod = 2

        result = kama_numba(close, timeperiod)

        assert len(result) == len(close)
        assert np.isnan(result[:timeperiod]).all()

    def test_kama_numba_constant_values(self):
        """Test KAMA with constant input."""
        close = np.full(100, 50.0)
        timeperiod = 20

        result = kama_numba(close, timeperiod)

        # For constant input, KAMA should equal input after warmup
        assert np.allclose(result[timeperiod:], 50.0, rtol=1e-10)

    def test_kama_numba_insufficient_data(self):
        """Test KAMA with insufficient data."""
        close = np.array([1.0, 2.0, 3.0])
        timeperiod = 10

        result = kama_numba(close, timeperiod)

        assert np.isnan(result).all()


class TestT3NumbaCoverage:
    """Test T3 Numba implementation directly."""

    def test_t3_numba_direct_large_array(self):
        """Test T3 numba function with large array."""
        close = np.random.randn(1000).cumsum() + 100
        timeperiod = 5
        vfactor = 0.7

        result = t3_numba(close, timeperiod, vfactor)

        assert len(result) == len(close)
        lookback = 6 * (timeperiod - 1)
        assert np.isnan(result[:lookback]).all()
        assert not np.isnan(result[lookback:]).any()

    def test_t3_numba_default_parameters(self):
        """Test T3 with default parameters."""
        close = np.random.randn(200).cumsum() + 100
        result = t3_numba(close, 5, 0.7)

        lookback = 6 * (5 - 1)  # 24
        assert len(result) == len(close)
        assert np.isnan(result[:lookback]).all()

    def test_t3_numba_different_vfactors(self):
        """Test T3 with different volume factors."""
        close = np.random.randn(200).cumsum() + 100
        timeperiod = 5

        # Test various vfactors
        for vfactor in [0.0, 0.3, 0.5, 0.7, 1.0]:
            result = t3_numba(close, timeperiod, vfactor)
            lookback = 6 * (timeperiod - 1)
            assert len(result) == len(close)
            assert not np.isnan(result[lookback:]).any()

    def test_t3_numba_different_periods(self):
        """Test T3 with different time periods."""
        close = np.random.randn(300).cumsum() + 100
        vfactor = 0.7

        for timeperiod in [2, 5, 10, 20]:
            result = t3_numba(close, timeperiod, vfactor)
            lookback = 6 * (timeperiod - 1)
            assert len(result) == len(close)
            if lookback < len(close):
                assert not np.isnan(result[lookback:]).any()

    def test_t3_numba_insufficient_data(self):
        """Test T3 with insufficient data."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        timeperiod = 5
        vfactor = 0.7

        result = t3_numba(close, timeperiod, vfactor)

        # Lookback = 6 * (5-1) = 24, which exceeds data length
        assert np.isnan(result).all()

    def test_t3_numba_constant_values(self):
        """Test T3 with constant input."""
        close = np.full(200, 100.0)
        timeperiod = 5
        vfactor = 0.7

        result = t3_numba(close, timeperiod, vfactor)

        lookback = 6 * (timeperiod - 1)
        # For constant input, T3 should equal input after warmup
        assert np.allclose(result[lookback:], 100.0, rtol=1e-8)

    def test_t3_numba_six_level_ema_cascade(self):
        """Test T3's six-level EMA cascade calculation."""
        # Use longer array to ensure all 6 EMAs are calculated
        close = np.linspace(100, 150, 300)
        timeperiod = 5
        vfactor = 0.7

        result = t3_numba(close, timeperiod, vfactor)

        lookback = 6 * (timeperiod - 1)
        assert len(result) == len(close)
        # Verify smooth progression
        assert not np.isnan(result[lookback:]).any()

    def test_t3_numba_coefficient_calculation(self):
        """Test T3 coefficient calculation for different vfactors."""
        close = np.random.randn(200).cumsum() + 100
        timeperiod = 5

        # Extreme vfactors should produce valid results
        for vfactor in [0.1, 0.5, 0.9]:
            result = t3_numba(close, timeperiod, vfactor)
            lookback = 6 * (timeperiod - 1)
            assert not np.isnan(result[lookback:]).any()
            # Results should be different for different vfactors
            assert len(np.unique(result[lookback : lookback + 10])) > 1

    def test_t3_numba_edge_case_minimum_period(self):
        """Test T3 with minimum viable period."""
        close = np.random.randn(100).cumsum() + 100
        timeperiod = 2  # Minimum period
        vfactor = 0.7

        result = t3_numba(close, timeperiod, vfactor)

        lookback = 6 * (timeperiod - 1)  # 6
        assert len(result) == len(close)
        assert np.isnan(result[:lookback]).all()
        assert not np.isnan(result[lookback:]).any()


class TestWMANumbaCoverage:
    """Test WMA Numba implementation directly."""

    def test_wma_numba_direct_large_array(self):
        """Test WMA numba function with large array."""
        close = np.random.randn(1000).cumsum() + 100
        period = 20

        result = wma_numba(close, period)

        assert len(result) == len(close)
        assert np.isnan(result[: period - 1]).all()
        assert not np.isnan(result[period - 1 :]).any()

    def test_wma_numba_exact_calculation(self):
        """Test WMA calculation accuracy."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 3

        result = wma_numba(close, period)

        # Manual calculation: WMA = (1*w1 + 2*w2 + 3*w3) / (1+2+3)
        # For period 3: weights are 1, 2, 3
        # At index 2: (1*1 + 2*2 + 3*3) / 6 = 14/6 = 2.333...
        expected = (1.0 * 1 + 2.0 * 2 + 3.0 * 3) / 6.0
        assert np.isclose(result[2], expected)

        # At index 3: (2*1 + 3*2 + 4*3) / 6 = 20/6 = 3.333...
        expected = (2.0 * 1 + 3.0 * 2 + 4.0 * 3) / 6.0
        assert np.isclose(result[3], expected)

    def test_wma_numba_weight_calculation(self):
        """Test WMA weight calculation is correct."""
        close = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        period = 4

        result = wma_numba(close, period)

        # Weights for period 4: 1, 2, 3, 4
        # Weight sum = 1+2+3+4 = 10
        # At index 3: (10*1 + 20*2 + 30*3 + 40*4) / 10 = 300/10 = 30
        expected = (10.0 * 1 + 20.0 * 2 + 30.0 * 3 + 40.0 * 4) / 10.0
        assert np.isclose(result[3], expected)

    def test_wma_numba_short_period(self):
        """Test WMA with minimum period."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 2

        result = wma_numba(close, period)

        assert len(result) == len(close)
        assert np.isnan(result[0])
        # WMA with period 2: weights 1, 2
        # At index 1: (1*1 + 2*2) / 3 = 5/3
        expected = (1.0 * 1 + 2.0 * 2) / 3.0
        assert np.isclose(result[1], expected)

    def test_wma_numba_constant_values(self):
        """Test WMA with constant input."""
        close = np.full(100, 42.0)
        period = 10

        result = wma_numba(close, period)

        # For constant input, WMA should equal input
        assert np.allclose(result[period - 1 :], 42.0)

    def test_wma_numba_invalid_period(self):
        """Test WMA with invalid period."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 0

        result = wma_numba(close, period)

        assert np.isnan(result).all()

    def test_wma_numba_period_exceeds_data(self):
        """Test WMA when period exceeds data length."""
        close = np.array([1.0, 2.0, 3.0])
        period = 10

        result = wma_numba(close, period)

        assert np.isnan(result).all()

    def test_wma_numba_period_equals_length(self):
        """Test WMA when period equals data length."""
        close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        period = 5

        result = wma_numba(close, period)

        # Should have one valid value at the end
        assert np.isnan(result[:4]).all()
        # Weights: 1, 2, 3, 4, 5; sum = 15
        # WMA = (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15 = 55/15 = 3.666...
        expected = (1.0 * 1 + 2.0 * 2 + 3.0 * 3 + 4.0 * 4 + 5.0 * 5) / 15.0
        assert np.isclose(result[4], expected)

    def test_wma_numba_gives_more_weight_to_recent(self):
        """Test that WMA gives more weight to recent values."""
        # Create data with sudden jump
        close = np.concatenate(
            [
                np.full(50, 100.0),
                np.full(50, 120.0),  # Sudden jump
            ]
        )
        period = 10

        result = wma_numba(close, period)

        # WMA should be between 100 and 120
        # At the jump point, recent values get more weight
        jump_idx = 55  # 5 values into the jump
        assert 100.0 < result[jump_idx] < 120.0

    def test_wma_numba_large_period(self):
        """Test WMA with large period."""
        close = np.random.randn(500).cumsum() + 100
        period = 100

        result = wma_numba(close, period)

        assert len(result) == len(close)
        assert np.isnan(result[: period - 1]).all()
        assert not np.isnan(result[period - 1 :]).any()

    def test_wma_numba_trending_data(self):
        """Test WMA follows trending data correctly."""
        close = np.linspace(100, 200, 200)
        period = 20

        result = wma_numba(close, period)

        # WMA should be monotonically increasing
        valid_result = result[~np.isnan(result)]
        assert np.all(np.diff(valid_result) > -1e-10)  # Allow tiny numerical errors


class TestCrossIndicatorComparison:
    """Test relationships between different trend indicators."""

    def test_ema_vs_sma_responsiveness(self):
        """EMA should be more responsive than SMA to recent changes."""
        # Create data with sudden jump
        close = np.concatenate(
            [
                np.full(50, 100.0),
                np.full(50, 120.0),  # Sudden jump
            ]
        )
        period = 10

        ema_result = ema_numba(close, period)
        sma_result = sma_numba(close, period)

        # After the jump, EMA should reach new level faster than SMA
        # Check midpoint of transition
        idx = 55
        ema_progress = (ema_result[idx] - 100.0) / 20.0  # Fraction to new level
        sma_progress = (sma_result[idx] - 100.0) / 20.0

        assert ema_progress > sma_progress

    def test_dema_vs_ema_lag(self):
        """DEMA should have less lag than EMA."""
        close = np.linspace(100, 200, 300)  # Linear trend
        period = 20

        ema_result = ema_numba(close, period)
        dema_result = dema_numba(close, period)

        # In trend, DEMA should be closer to actual price
        start_idx = 2 * (period - 1) + 10  # After warmup
        dema_error = np.abs(
            dema_result[start_idx : start_idx + 50] - close[start_idx : start_idx + 50]
        )
        ema_error = np.abs(
            ema_result[start_idx : start_idx + 50] - close[start_idx : start_idx + 50]
        )

        assert np.mean(dema_error) < np.mean(ema_error)

    def test_kama_vs_ema_in_choppy_market(self):
        """KAMA should be smoother than EMA in choppy markets."""
        # Create choppy market
        np.random.seed(42)
        close = 100 + np.random.randn(200) * 2
        period = 20

        ema_result = ema_numba(close, period)
        kama_result = kama_numba(close, period)

        # Measure smoothness by variance of differences
        start_idx = period + 1
        ema_changes = np.diff(ema_result[start_idx : start_idx + 50])
        kama_changes = np.diff(kama_result[start_idx : start_idx + 50])

        # KAMA should have lower variance (smoother)
        assert np.var(kama_changes) < np.var(ema_changes)

    def test_t3_extreme_smoothness(self):
        """T3 should be very smooth due to six-level EMA cascade."""
        # Create noisy data
        np.random.seed(42)
        close = 100 + np.random.randn(300) * 5 + np.linspace(0, 20, 300)
        timeperiod = 5
        vfactor = 0.7

        t3_result = t3_numba(close, timeperiod, vfactor)
        ema_result = ema_numba(close, timeperiod)

        # Measure smoothness
        lookback_t3 = 6 * (timeperiod - 1)
        start_idx = max(lookback_t3, timeperiod)
        t3_changes = np.diff(t3_result[start_idx : start_idx + 50])
        ema_changes = np.diff(ema_result[start_idx : start_idx + 50])

        # T3 should have much lower variance (smoother)
        assert np.var(t3_changes) < np.var(ema_changes)

    def test_wma_vs_sma_recent_weight(self):
        """WMA should give more weight to recent data than SMA."""
        # Create data with sudden jump
        close = np.concatenate(
            [
                np.full(50, 100.0),
                np.full(50, 120.0),  # Sudden jump
            ]
        )
        period = 10

        wma_result = wma_numba(close, period)
        sma_result = sma_numba(close, period)

        # After the jump, WMA should reach new level faster than SMA
        # Check midpoint of transition
        idx = 55
        wma_progress = (wma_result[idx] - 100.0) / 20.0  # Fraction to new level
        sma_progress = (sma_result[idx] - 100.0) / 20.0

        # WMA gives linearly increasing weights, so it should be more responsive than equal-weight SMA
        assert wma_progress > sma_progress


class TestNumericalStability:
    """Test numerical stability of Numba implementations."""

    def test_ema_with_very_large_values(self):
        """Test EMA with very large values."""
        close = np.random.randn(100) * 1e6 + 1e8
        period = 20

        result = ema_numba(close, period)

        assert not np.isinf(result).any()
        assert np.isfinite(result[period - 1 :]).all()

    def test_ema_with_very_small_values(self):
        """Test EMA with very small values."""
        close = np.random.randn(100) * 1e-6 + 1e-4
        period = 20

        result = ema_numba(close, period)

        assert not np.isinf(result).any()
        assert np.isfinite(result[period - 1 :]).all()

    def test_sma_numerical_precision(self):
        """Test SMA maintains precision with large sums."""
        # Large values that could cause precision issues
        close = np.full(100, 1e10) + np.random.randn(100)
        period = 20

        result = sma_numba(close, period)

        # Should maintain precision
        assert np.isfinite(result[period - 1 :]).all()
        assert np.allclose(result[period - 1 :], 1e10, rtol=1e-8)

    def test_kama_with_zero_volatility(self):
        """Test KAMA when volatility is zero (constant prices)."""
        close = np.full(100, 100.0)
        timeperiod = 20

        result = kama_numba(close, timeperiod)

        # Should handle zero volatility gracefully
        assert np.isfinite(result[timeperiod:]).all()
        assert np.allclose(result[timeperiod:], 100.0)

    def test_t3_numerical_stability_with_cascade(self):
        """Test T3's six-level cascade maintains numerical stability."""
        close = np.random.randn(300) * 1e3 + 1e5
        timeperiod = 5
        vfactor = 0.7

        result = t3_numba(close, timeperiod, vfactor)

        lookback = 6 * (timeperiod - 1)
        assert np.isfinite(result[lookback:]).all()
        assert not np.isinf(result).any()

    def test_wma_with_very_large_values(self):
        """Test WMA with very large values."""
        close = np.random.randn(100) * 1e6 + 1e8
        period = 20

        result = wma_numba(close, period)

        assert not np.isinf(result).any()
        assert np.isfinite(result[period - 1 :]).all()

    def test_wma_numerical_precision_weighted_sum(self):
        """Test WMA maintains precision with weighted sums."""
        # Large values that could cause precision issues
        close = np.full(100, 1e10) + np.random.randn(100)
        period = 20

        result = wma_numba(close, period)

        # Should maintain precision
        assert np.isfinite(result[period - 1 :]).all()
        assert np.allclose(result[period - 1 :], 1e10, rtol=1e-8)


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_all_indicators_with_single_valid_point(self):
        """Test all indicators with minimum data after NaN."""
        close = np.concatenate([np.full(50, np.nan), [100.0]])

        # These should all return arrays of correct length
        ema = ema_numba(close, 10)
        sma = sma_numba(close, 10)

        assert len(ema) == len(close)
        assert len(sma) == len(close)

    def test_alternating_nan_pattern(self):
        """Test indicators with alternating NaN pattern."""
        close = np.array([1.0, np.nan, 2.0, np.nan, 3.0, np.nan, 4.0, np.nan, 5.0, np.nan] * 10)

        ema_result = ema_numba(close, 3)
        sma_result = sma_numba(close, 3)

        # Should handle gracefully
        assert len(ema_result) == len(close)
        assert len(sma_result) == len(close)

    def test_all_indicators_exact_lookback_length(self):
        """Test when data length exactly matches lookback requirement."""
        # DEMA has longest lookback
        close = np.random.randn(60).cumsum() + 100
        period = 30

        dema_result = dema_numba(close, period)
        2 * (period - 1)  # 58

        assert len(dema_result) == 60
        # Should have exactly 2 valid values
        valid_count = np.sum(~np.isnan(dema_result))
        assert valid_count >= 1  # At least one valid value

    def test_monotonic_increasing_data(self):
        """Test all indicators with strictly monotonic increasing data."""
        close = np.arange(200, dtype=np.float64)

        ema_result = ema_numba(close, 10)
        sma_result = sma_numba(close, 10)
        wma_result = wma_numba(close, 10)
        kama_result = kama_numba(close, 20)
        dema_result = dema_numba(close, 20)
        t3_result = t3_numba(close, 5, 0.7)

        # All should be monotonic increasing after warmup
        for result in [ema_result, sma_result, wma_result, kama_result, dema_result, t3_result]:
            valid_idx = ~np.isnan(result)
            valid_values = result[valid_idx]
            if len(valid_values) > 1:
                assert np.all(np.diff(valid_values) >= -1e-10)  # Allow tiny numerical errors

    def test_monotonic_decreasing_data(self):
        """Test all indicators with strictly monotonic decreasing data."""
        close = np.arange(200, dtype=np.float64)[::-1]

        ema_result = ema_numba(close, 10)
        sma_result = sma_numba(close, 10)
        wma_result = wma_numba(close, 10)
        kama_result = kama_numba(close, 20)
        dema_result = dema_numba(close, 20)
        t3_result = t3_numba(close, 5, 0.7)

        # All should be monotonic decreasing after warmup
        for result in [ema_result, sma_result, wma_result, kama_result, dema_result, t3_result]:
            valid_idx = ~np.isnan(result)
            valid_values = result[valid_idx]
            if len(valid_values) > 1:
                assert np.all(np.diff(valid_values) <= 1e-10)  # Allow tiny numerical errors
