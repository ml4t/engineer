"""
Test AVGDEV (Average Deviation) indicator.
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

from ml4t.engineer.features.statistics import avgdev, stddev


class TestAVGDEV:
    """Test AVGDEV indicator."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    @pytest.fixture
    def constant_data(self):
        """Generate constant data."""
        return np.array([5.0] * 20)

    @pytest.fixture
    def random_data(self):
        """Generate random test data."""
        np.random.seed(42)
        return np.random.randn(1000) * 10 + 100

    def test_avgdev_accuracy(self, simple_data, random_data):
        """Test AVGDEV matches TA-Lib exactly."""
        for data in [simple_data, random_data]:
            for period in [2, 5, 10, 14, 20]:
                if len(data) < period:
                    continue

                expected = talib.AVGDEV(data, timeperiod=period)
                result = avgdev(data, timeperiod=period)

                # Compare where both have values
                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-7,
                        err_msg=f"AVGDEV mismatch for period {period}",
                    )

    def test_avgdev_polars(self, random_data):
        """Test AVGDEV with Polars expressions."""
        df = pl.DataFrame({"values": random_data})

        result = df.with_columns(avgdev("values", timeperiod=10).alias("avgdev"))

        expected = talib.AVGDEV(random_data, timeperiod=10)
        result_np = result["avgdev"].to_numpy()

        valid_idx = ~(np.isnan(expected) | np.isnan(result_np))
        if np.any(valid_idx):
            assert_allclose(result_np[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_avgdev_mathematical_properties(self, constant_data):
        """Test mathematical properties of average deviation."""
        # Constant values should have zero average deviation
        avgdev_const = avgdev(constant_data, timeperiod=5)
        valid = avgdev_const[~np.isnan(avgdev_const)]
        assert_allclose(valid, 0.0, atol=1e-10)

        # Average deviation should be non-negative
        random_data = np.random.randn(100)
        avgdev_random = avgdev(random_data, timeperiod=10)
        valid = avgdev_random[~np.isnan(avgdev_random)]
        assert np.all(valid >= 0), "Average deviation should be non-negative"

        # For symmetric distributions, avgdev ≈ 0.8 * stddev
        # This is approximate and depends on distribution
        normal_data = np.random.normal(100, 10, 1000)
        avgdev_values = avgdev(normal_data, timeperiod=50)
        stddev_values = stddev(normal_data, period=50)

        valid_idx = ~(np.isnan(avgdev_values) | np.isnan(stddev_values))
        if np.any(valid_idx):
            ratio = np.mean(avgdev_values[valid_idx] / stddev_values[valid_idx])
            # For normal distribution, theoretical ratio is sqrt(2/π) ≈ 0.7979
            assert 0.75 < ratio < 0.85, f"AVGDEV/STDDEV ratio {ratio} out of expected range"

    def test_avgdev_edge_cases(self):
        """Test AVGDEV with edge cases."""
        # Linear sequence
        linear = np.arange(1, 11, dtype=float)
        result = avgdev(linear, timeperiod=5)
        expected = talib.AVGDEV(linear, timeperiod=5)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

        # Minimum period
        values = np.random.randn(10)
        result = avgdev(values, timeperiod=2)
        expected = talib.AVGDEV(values, timeperiod=2)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_avgdev_parameter_validation(self):
        """Test parameter validation."""
        values = np.random.randn(100)

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            avgdev(values, timeperiod=1)

    def test_avgdev_nan_pattern(self):
        """Test AVGDEV NaN pattern."""
        values = np.random.randn(100)

        for period in [5, 10, 20]:
            result = avgdev(values, timeperiod=period)
            expected = talib.AVGDEV(values, timeperiod=period)

            # First period-1 values should be NaN
            assert np.all(np.isnan(result[: period - 1]))
            assert ~np.isnan(result[period - 1])

            # NaN patterns should match
            assert np.array_equal(np.isnan(result), np.isnan(expected))

    def test_avgdev_outlier_sensitivity(self):
        """Test that AVGDEV is less sensitive to outliers than stddev."""
        # Create data with outliers
        normal_data = np.random.normal(100, 2, 100)
        outlier_data = normal_data.copy()
        outlier_data[50] = 200  # Add a large outlier

        # Calculate avgdev for both
        avgdev_normal = avgdev(normal_data, timeperiod=20)
        avgdev_outlier = avgdev(outlier_data, timeperiod=20)

        # Calculate stddev for both
        stddev_normal = stddev(normal_data, period=20)
        stddev_outlier = stddev(outlier_data, period=20)

        # Find indices affected by the outlier (around position 50)
        affected_idx = list(
            range(31, 51),
        )  # Positions where the outlier is in the window

        # Average deviation should be less affected than standard deviation
        avgdev_change = np.mean(
            avgdev_outlier[affected_idx] / avgdev_normal[affected_idx],
        )
        stddev_change = np.mean(
            stddev_outlier[affected_idx] / stddev_normal[affected_idx],
        )

        assert avgdev_change < stddev_change, "AVGDEV should be less sensitive to outliers"

    def test_avgdev_crypto_accuracy(self, crypto_data_small):
        """Test AVGDEV accuracy on real crypto data."""
        prices = crypto_data_small["close"].to_numpy()

        for period in [5, 10, 14, 20]:
            expected = talib.AVGDEV(prices, timeperiod=period)
            result = avgdev(prices, timeperiod=period)

            valid_idx = ~(np.isnan(expected) | np.isnan(result))
            if np.any(valid_idx):
                assert_allclose(
                    result[valid_idx],
                    expected[valid_idx],
                    rtol=1e-7,
                    err_msg=f"AVGDEV mismatch on crypto data for period {period}",
                )

    @pytest.mark.benchmark
    def test_avgdev_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark AVGDEV performance."""
        prices = crypto_data["close"].to_numpy()

        # Warmup JIT
        warmup_jit(avgdev, prices[:100])

        import time

        # Our implementation
        start = time.perf_counter()
        _ = avgdev(prices, timeperiod=14)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.AVGDEV(prices, timeperiod=14)
        talib_time = time.perf_counter() - start

        print(f"\nAVGDEV Performance ({len(prices):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # AVGDEV is a simple calculation
        threshold = performance_threshold("simple")
        assert our_time < talib_time * threshold
