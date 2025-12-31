"""
Test LINEARREG (Linear Regression) indicator.
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

from ml4t.engineer.features.statistics import linearreg
from ml4t.engineer.features.trend import sma


class TestLINEARREG:
    """Test LINEARREG indicator."""

    @pytest.fixture
    def linear_data(self):
        """Generate perfectly linear test data."""
        # y = 2x + 10
        return np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28], dtype=float)

    @pytest.fixture
    def trending_data(self):
        """Generate trending data with some noise."""
        np.random.seed(42)
        x = np.arange(50)
        trend = 100 + 2 * x
        noise = np.random.randn(50) * 2
        return trend + noise

    @pytest.fixture
    def random_data(self):
        """Generate random test data."""
        np.random.seed(42)
        return np.random.randn(1000) * 10 + 100

    def test_linearreg_accuracy(self, linear_data, trending_data, random_data):
        """Test LINEARREG matches TA-Lib exactly."""
        for data in [linear_data, trending_data, random_data]:
            for period in [2, 5, 10, 14, 20]:
                if len(data) < period:
                    continue

                expected = talib.LINEARREG(data, timeperiod=period)
                result = linearreg(data, timeperiod=period)

                # Compare where both have values
                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-7,
                        err_msg=f"LINEARREG mismatch for period {period}",
                    )

    def test_linearreg_polars(self, random_data):
        """Test LINEARREG with Polars expressions."""
        df = pl.DataFrame({"values": random_data})

        result = df.with_columns(
            linearreg("values", timeperiod=14).alias("linearreg"),
        )

        expected = talib.LINEARREG(random_data, timeperiod=14)
        result_np = result["linearreg"].to_numpy()

        valid_idx = ~(np.isnan(expected) | np.isnan(result_np))
        if np.any(valid_idx):
            assert_allclose(result_np[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_linearreg_perfect_linear(self):
        """Test LINEARREG with perfectly linear data."""
        # For perfectly linear data, LINEARREG should equal the actual values
        x = np.arange(20, dtype=float)
        y = 2 * x + 10  # y = 2x + 10

        result = linearreg(y, timeperiod=5)

        # For perfect linear data, regression line matches the data exactly
        valid_idx = ~np.isnan(result)
        assert_allclose(result[valid_idx], y[valid_idx], rtol=1e-10)

    def test_linearreg_constant_values(self):
        """Test LINEARREG with constant values."""
        # For constant values, regression should return the constant
        constant = np.array([50.0] * 20)

        result = linearreg(constant, timeperiod=10)
        valid = result[~np.isnan(result)]

        # All values should be the constant
        assert_allclose(valid, 50.0, atol=1e-10)

    def test_linearreg_vs_sma(self):
        """Test LINEARREG relationship with SMA."""
        # For symmetric data around a point, LINEARREG ≈ SMA
        np.random.seed(42)
        # Generate data that oscillates around a mean
        t = np.linspace(0, 4 * np.pi, 100)
        data = 100 + 10 * np.sin(t)

        linearreg_values = linearreg(data, timeperiod=20)
        sma_values = sma(data, period=20)

        # They should be close but not identical
        valid_idx = ~(np.isnan(linearreg_values) | np.isnan(sma_values))
        if np.any(valid_idx):
            diff = np.abs(linearreg_values[valid_idx] - sma_values[valid_idx])
            avg_diff = np.mean(diff)
            # For oscillating data, they should be somewhat close
            # LINEARREG can differ more from SMA especially at turning points
            assert avg_diff < 10.0

    def test_linearreg_slope_direction(self):
        """Test LINEARREG correctly identifies trend direction."""
        # Upward trend
        up_trend = np.array([10, 11, 12, 11, 13, 14, 13, 15, 16, 17], dtype=float)
        result_up = linearreg(up_trend, timeperiod=5)

        # Check that regression values are increasing (positive slope)
        valid_up = result_up[~np.isnan(result_up)]
        assert np.mean(np.diff(valid_up)) > 0  # Overall upward

        # Downward trend
        down_trend = np.array([20, 19, 18, 19, 17, 16, 17, 15, 14, 13], dtype=float)
        result_down = linearreg(down_trend, timeperiod=5)

        # Check that regression values are decreasing (negative slope)
        valid_down = result_down[~np.isnan(result_down)]
        assert np.mean(np.diff(valid_down)) < 0  # Overall downward

    def test_linearreg_parameter_validation(self):
        """Test parameter validation."""
        values = np.random.randn(100)

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            linearreg(values, timeperiod=1)

    def test_linearreg_nan_pattern(self):
        """Test LINEARREG NaN pattern."""
        values = np.random.randn(100)

        for period in [5, 10, 20]:
            result = linearreg(values, timeperiod=period)
            expected = talib.LINEARREG(values, timeperiod=period)

            # First period-1 values should be NaN
            assert np.all(np.isnan(result[: period - 1]))
            assert ~np.isnan(result[period - 1])

            # NaN patterns should match
            assert np.array_equal(np.isnan(result), np.isnan(expected))

    def test_linearreg_edge_cases(self):
        """Test LINEARREG with edge cases."""
        # Minimum period
        values = np.array([1, 3, 2, 4, 3, 5, 4, 6], dtype=float)
        result = linearreg(values, timeperiod=2)
        expected = talib.LINEARREG(values, timeperiod=2)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

        # Test specific regression calculation
        # For [1, 3] with period=2: slope = 2, intercept = 1
        # At x=1: y = 1 + 2*1 = 3
        assert_allclose(result[1], 3.0, atol=1e-10)

    def test_linearreg_mathematical_properties(self):
        """Test mathematical properties of linear regression."""
        # Generate data with known properties
        n = 50
        x = np.arange(n, dtype=float)

        # Test 1: For y = mx + b, LINEARREG should return exact values
        m, b = 1.5, 20.0
        y = m * x + b
        result = linearreg(y, timeperiod=10)
        valid_idx = ~np.isnan(result)
        assert_allclose(result[valid_idx], y[valid_idx], rtol=1e-10)

        # Test 2: Adding noise should smooth it out
        noise = np.random.RandomState(42).randn(n) * 5
        y_noisy = y + noise
        result_noisy = linearreg(y_noisy, timeperiod=10)

        # Regression should be smoother than original noisy data
        valid_idx = ~np.isnan(result_noisy)
        if len(valid_idx) > 2:
            # Variance of differences should be less for regression
            orig_diff_var = np.var(np.diff(y_noisy[valid_idx]))
            reg_diff_var = np.var(np.diff(result_noisy[valid_idx]))
            assert reg_diff_var < orig_diff_var

    def test_linearreg_crypto_accuracy(self, crypto_data_small):
        """Test LINEARREG accuracy on real crypto data."""
        prices = crypto_data_small["close"].to_numpy()

        for period in [5, 10, 14, 20]:
            expected = talib.LINEARREG(prices, timeperiod=period)
            result = linearreg(prices, timeperiod=period)

            valid_idx = ~(np.isnan(expected) | np.isnan(result))
            if np.any(valid_idx):
                assert_allclose(
                    result[valid_idx],
                    expected[valid_idx],
                    rtol=1e-7,
                    err_msg=f"LINEARREG mismatch on crypto data for period {period}",
                )

    @pytest.mark.benchmark
    def test_linearreg_performance(
        self,
        crypto_data,
        performance_threshold,
        warmup_jit,
    ):
        """Benchmark LINEARREG performance."""
        prices = crypto_data["close"].to_numpy()

        # Warmup JIT
        warmup_jit(linearreg, prices[:100])

        import time

        # Our implementation
        start = time.perf_counter()
        _ = linearreg(prices, timeperiod=14)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.LINEARREG(prices, timeperiod=14)
        talib_time = time.perf_counter() - start

        print(f"\nLINEARREG Performance ({len(prices):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # LINEARREG is more complex due to regression calculations
        # Use complex threshold (3x) with margin for CI variability
        threshold = performance_threshold("complex") * 1.5  # 4.5x for CI
        assert our_time < talib_time * threshold
