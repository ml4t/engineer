"""
Test VAR (Variance) indicator.
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

from ml4t.engineer.features.statistics import stddev, var


class TestVAR:
    """Test VAR indicator."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    @pytest.fixture
    def random_data(self):
        """Generate random test data."""
        np.random.seed(42)
        return np.random.randn(1000) * 10 + 100

    @pytest.fixture
    def volatile_data(self):
        """Generate volatile data."""
        np.random.seed(42)
        # Mix of trends and volatility
        n = 200
        base = np.linspace(100, 120, n)
        noise = np.random.normal(0, 5, n)
        shocks = np.zeros(n)
        shocks[50:60] = np.random.normal(0, 20, 10)  # Volatility spike
        return base + noise + shocks

    def test_var_accuracy(self, simple_data, random_data, volatile_data):
        """Test VAR matches TA-Lib exactly."""
        for data in [simple_data, random_data, volatile_data]:
            for period in [2, 5, 10, 20]:
                if len(data) < period:
                    continue

                expected = talib.VAR(data, timeperiod=period)
                result = var(data, timeperiod=period)

                # Compare where both have values
                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-5,
                        atol=1e-10,
                        err_msg=f"VAR mismatch for period {period}",
                    )

    def test_var_polars(self, random_data):
        """Test VAR with Polars expressions."""
        df = pl.DataFrame({"values": random_data})

        result = df.with_columns(var("values", timeperiod=10).alias("var"))

        expected = talib.VAR(random_data, timeperiod=10)
        result_np = result["var"].to_numpy()

        valid_idx = ~(np.isnan(expected) | np.isnan(result_np))
        if np.any(valid_idx):
            assert_allclose(result_np[valid_idx], expected[valid_idx], rtol=1e-6)

    def test_var_mathematical_properties(self):
        """Test mathematical properties of variance."""
        # Constant values should have zero variance
        constant = np.full(20, 5.0)
        var_const = var(constant, timeperiod=5)
        valid = var_const[~np.isnan(var_const)]
        assert_allclose(valid, 0.0, atol=1e-10)

        # Variance should be non-negative
        random_data = np.random.randn(100)
        var_random = var(random_data, timeperiod=10)
        valid = var_random[~np.isnan(var_random)]
        assert np.all(valid >= 0), "Variance should be non-negative"

        # Variance relationship with standard deviation
        std_random = stddev(random_data, period=10)
        valid_idx = ~(np.isnan(var_random) | np.isnan(std_random))
        if np.any(valid_idx):
            # Variance should equal stddev squared
            assert_allclose(
                var_random[valid_idx],
                std_random[valid_idx] ** 2,
                rtol=1e-6,
            )

    def test_var_nbdev_parameter(self, random_data):
        """Test that nbdev parameter doesn't affect VAR calculation."""
        # VAR should be the same regardless of nbdev value
        var1 = var(random_data, timeperiod=10, nbdev=1.0)
        var2 = var(random_data, timeperiod=10, nbdev=2.0)
        var3 = var(random_data, timeperiod=10, nbdev=0.5)

        assert_allclose(var1, var2, rtol=1e-10)
        assert_allclose(var1, var3, rtol=1e-10)

    def test_var_edge_cases(self):
        """Test VAR with edge cases."""
        # Single value repeated
        values = np.array([5.0] * 10)
        result = var(values, timeperiod=5)
        expected = talib.VAR(values, timeperiod=5)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-6, atol=1e-14)
            # Should be zero for constant values (allow for floating point precision)
            assert_allclose(result[valid_idx], 0.0, atol=1e-14)

        # Minimum period
        values = np.random.randn(10)
        result = var(values, timeperiod=1)
        expected = talib.VAR(values, timeperiod=1)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            # Variance with period=1 should be close to 0 (within floating point precision)
            assert_allclose(result[valid_idx], 0.0, atol=1e-14)
            assert_allclose(expected[valid_idx], 0.0, atol=1e-14)

    def test_var_parameter_validation(self):
        """Test parameter validation."""
        values = np.random.randn(100)

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be >= 1"):
            var(values, timeperiod=0)

    def test_var_nan_pattern(self):
        """Test VAR NaN pattern."""
        values = np.random.randn(100)

        for period in [5, 10, 20]:
            result = var(values, timeperiod=period)
            expected = talib.VAR(values, timeperiod=period)

            # First period-1 values should be NaN
            assert np.all(np.isnan(result[: period - 1]))
            assert ~np.isnan(result[period - 1])

            # NaN patterns should match
            assert np.array_equal(np.isnan(result), np.isnan(expected))

    def test_var_volatility_detection(self):
        """Test that VAR detects volatility changes."""
        # Create data with volatility regime change
        n = 200
        low_vol = np.random.normal(100, 2, n // 2)
        high_vol = np.random.normal(100, 10, n // 2)
        data = np.concatenate([low_vol, high_vol])

        var_values = var(data, timeperiod=20)

        # Variance in high volatility period should be higher
        low_vol_var = np.nanmean(var_values[30:90])
        high_vol_var = np.nanmean(var_values[120:180])

        assert high_vol_var > low_vol_var * 5, "VAR should detect volatility increase"

    def test_var_crypto_accuracy(self, crypto_data_small):
        """Test VAR accuracy on real crypto data."""
        prices = crypto_data_small["close"].to_numpy()

        for period in [5, 10, 20]:
            expected = talib.VAR(prices, timeperiod=period)
            result = var(prices, timeperiod=period)

            valid_idx = ~(np.isnan(expected) | np.isnan(result))
            if np.any(valid_idx):
                assert_allclose(
                    result[valid_idx],
                    expected[valid_idx],
                    rtol=2e-3,  # More lenient for crypto data with small variance values
                    atol=1e-9,
                    err_msg=f"VAR mismatch on crypto data for period {period}",
                )

    @pytest.mark.benchmark
    def test_var_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark VAR performance."""
        prices = crypto_data["close"].to_numpy()

        # Warmup JIT
        warmup_jit(var, prices[:100])

        import time

        # Our implementation
        start = time.perf_counter()
        _ = var(prices, timeperiod=10)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.VAR(prices, timeperiod=10)
        talib_time = time.perf_counter() - start

        print(f"\nVAR Performance ({len(prices):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # VAR is simple calculation
        threshold = performance_threshold("simple")
        assert our_time < talib_time * threshold
