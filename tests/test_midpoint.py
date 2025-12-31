"""
Test MIDPOINT (Midpoint over period) indicator.
"""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    pytest.skip("TA-Lib not available", allow_module_level=True)

from ml4t.engineer.features.trend import midpoint, sma


class TestMIDPOINT:
    """Test MIDPOINT indicator."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    @pytest.fixture
    def oscillating_data(self):
        """Generate oscillating data."""
        # Data that swings between highs and lows
        return np.array(
            [10, 15, 12, 18, 14, 20, 16, 22, 18, 24, 20, 26, 22, 28, 24],
            dtype=float,
        )

    @pytest.fixture
    def random_data(self):
        """Generate random test data."""
        np.random.seed(42)
        return np.random.randn(1000) * 10 + 100

    def test_midpoint_accuracy(self, simple_data, oscillating_data, random_data):
        """Test MIDPOINT matches TA-Lib exactly."""
        for data in [simple_data, oscillating_data, random_data]:
            for period in [2, 5, 10, 14, 20]:
                if len(data) < period:
                    continue

                expected = talib.MIDPOINT(data, timeperiod=period)
                result = midpoint(data, timeperiod=period)

                # Compare where both have values
                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-7,
                        err_msg=f"MIDPOINT mismatch for period {period}",
                    )

    def test_midpoint_polars(self, random_data):
        """Test MIDPOINT with Polars expressions."""
        df = pl.DataFrame({"values": random_data})

        result = df.with_columns(
            midpoint("values", timeperiod=14).alias("midpoint"),
        )

        expected = talib.MIDPOINT(random_data, timeperiod=14)
        result_np = result["midpoint"].to_numpy()

        valid_idx = ~(np.isnan(expected) | np.isnan(result_np))
        if np.any(valid_idx):
            assert_allclose(result_np[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_midpoint_simple_cases(self):
        """Test MIDPOINT with simple cases."""
        # Constant values
        constant = np.array([5.0] * 10)
        result = midpoint(constant, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert_allclose(valid, 5.0, atol=1e-10)

        # Linear increasing
        linear = np.arange(1, 11, dtype=float)
        result = midpoint(linear, timeperiod=5)
        # For linear data, midpoint should be at the center
        # e.g., for values [1,2,3,4,5], midpoint = (5+1)/2 = 3
        assert_allclose(result[4], 3.0, atol=1e-10)  # Values 1-5: (5+1)/2 = 3
        assert_allclose(result[5], 4.0, atol=1e-10)  # Values 2-6: (6+2)/2 = 4
        assert_allclose(result[6], 5.0, atol=1e-10)  # Values 3-7: (7+3)/2 = 5

    def test_midpoint_vs_mean(self):
        """Test relationship between midpoint and mean."""
        # For uniformly distributed data, midpoint ≈ mean
        uniform_data = np.random.uniform(0, 100, 1000)

        midpoint_values = midpoint(uniform_data, timeperiod=50)
        # Calculate simple moving average for comparison
        sma_values = sma(uniform_data, period=50)

        valid_idx = ~(np.isnan(midpoint_values) | np.isnan(sma_values))
        if np.any(valid_idx):
            # They should be close but not identical
            diff = np.abs(midpoint_values[valid_idx] - sma_values[valid_idx])
            avg_diff = np.mean(diff)
            # For uniform distribution, difference should be small
            assert avg_diff < 5.0, f"Midpoint and SMA differ too much: {avg_diff}"

    def test_midpoint_edge_cases(self):
        """Test MIDPOINT with edge cases."""
        # Minimum period
        values = np.array([1, 5, 3, 7, 2, 8, 4, 6], dtype=float)
        result = midpoint(values, timeperiod=2)
        expected = talib.MIDPOINT(values, timeperiod=2)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

        # Check specific values
        assert_allclose(result[1], (5 + 1) / 2, atol=1e-10)  # (max=5, min=1)
        assert_allclose(result[2], (5 + 3) / 2, atol=1e-10)  # (max=5, min=3)

    def test_midpoint_parameter_validation(self):
        """Test parameter validation."""
        values = np.random.randn(100)

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            midpoint(values, timeperiod=1)

    def test_midpoint_nan_pattern(self):
        """Test MIDPOINT NaN pattern."""
        values = np.random.randn(100)

        for period in [5, 10, 20]:
            result = midpoint(values, timeperiod=period)
            expected = talib.MIDPOINT(values, timeperiod=period)

            # First period-1 values should be NaN
            assert np.all(np.isnan(result[: period - 1]))
            assert ~np.isnan(result[period - 1])

            # NaN patterns should match
            assert np.array_equal(np.isnan(result), np.isnan(expected))

    def test_midpoint_range_behavior(self):
        """Test MIDPOINT behavior with different ranges."""
        # Create data with expanding range
        n = 100
        base = 100.0
        expanding = np.zeros(n)
        for i in range(n):
            expanding[i] = base + (i % 20) * (i / 20)  # Expanding oscillation

        midpoint_short = midpoint(expanding, timeperiod=5)
        midpoint_long = midpoint(expanding, timeperiod=20)

        # Longer period should smooth more (lower midpoint variance)
        valid_idx = ~(np.isnan(midpoint_short) | np.isnan(midpoint_long))
        if np.any(valid_idx):
            short_var = np.var(midpoint_short[valid_idx])
            long_var = np.var(midpoint_long[valid_idx])
            assert long_var < short_var, "Longer period should smooth more (less variance)"

    def test_midpoint_crypto_accuracy(self, crypto_data_small):
        """Test MIDPOINT accuracy on real crypto data."""
        prices = crypto_data_small["close"].to_numpy()

        for period in [5, 10, 14, 20]:
            expected = talib.MIDPOINT(prices, timeperiod=period)
            result = midpoint(prices, timeperiod=period)

            valid_idx = ~(np.isnan(expected) | np.isnan(result))
            if np.any(valid_idx):
                assert_allclose(
                    result[valid_idx],
                    expected[valid_idx],
                    rtol=1e-7,
                    err_msg=f"MIDPOINT mismatch on crypto data for period {period}",
                )

    @pytest.mark.benchmark
    def test_midpoint_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark MIDPOINT performance."""
        prices = crypto_data["close"].to_numpy()

        # Warmup JIT
        warmup_jit(midpoint, prices[:100])

        import time

        # Our implementation
        start = time.perf_counter()
        _ = midpoint(prices, timeperiod=14)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.MIDPOINT(prices, timeperiod=14)
        talib_time = time.perf_counter() - start

        print(f"\nMIDPOINT Performance ({len(prices):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # MIDPOINT is a simple calculation but uses rolling window
        # Use rolling threshold (2x) with margin for CI variability
        threshold = performance_threshold("rolling") * 1.5  # Allow 3x for CI environments
        assert our_time < talib_time * threshold
