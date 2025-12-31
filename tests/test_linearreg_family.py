"""
Test Linear Regression family of indicators (ANGLE, INTERCEPT, SLOPE).
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

from ml4t.engineer.features.statistics import (
    linearreg,
    linearreg_angle,
    linearreg_intercept,
    linearreg_slope,
)


class TestLinearRegFamily:
    """Test Linear Regression family indicators."""

    @pytest.fixture
    def linear_data(self):
        """Generate perfectly linear test data."""
        # y = 2x + 10
        return np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28], dtype=float)

    @pytest.fixture
    def constant_data(self):
        """Generate constant test data."""
        return np.array([50.0] * 20)

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

    def test_linearreg_slope_accuracy(self, linear_data, trending_data, random_data):
        """Test LINEARREG_SLOPE matches TA-Lib exactly."""
        for data in [linear_data, trending_data, random_data]:
            for period in [2, 5, 10, 14, 20]:
                if len(data) < period:
                    continue

                expected = talib.LINEARREG_SLOPE(data, timeperiod=period)
                result = linearreg_slope(data, timeperiod=period)

                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-7,
                        err_msg=f"LINEARREG_SLOPE mismatch for period {period}",
                    )

    def test_linearreg_intercept_accuracy(
        self,
        linear_data,
        trending_data,
        random_data,
    ):
        """Test LINEARREG_INTERCEPT matches TA-Lib exactly."""
        for data in [linear_data, trending_data, random_data]:
            for period in [2, 5, 10, 14, 20]:
                if len(data) < period:
                    continue

                expected = talib.LINEARREG_INTERCEPT(data, timeperiod=period)
                result = linearreg_intercept(data, timeperiod=period)

                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-7,
                        err_msg=f"LINEARREG_INTERCEPT mismatch for period {period}",
                    )

    def test_linearreg_angle_accuracy(self, linear_data, trending_data, random_data):
        """Test LINEARREG_ANGLE matches TA-Lib exactly."""
        for data in [linear_data, trending_data, random_data]:
            for period in [2, 5, 10, 14, 20]:
                if len(data) < period:
                    continue

                expected = talib.LINEARREG_ANGLE(data, timeperiod=period)
                result = linearreg_angle(data, timeperiod=period)

                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-7,
                        err_msg=f"LINEARREG_ANGLE mismatch for period {period}",
                    )

    def test_linearreg_family_polars(self, random_data):
        """Test Linear Regression family with Polars expressions."""
        df = pl.DataFrame({"values": random_data})

        result = df.with_columns(
            [
                linearreg_slope("values", timeperiod=14).alias("slope"),
                linearreg_intercept("values", timeperiod=14).alias("intercept"),
                linearreg_angle("values", timeperiod=14).alias("angle"),
            ],
        )

        # Compare with TA-Lib
        expected_slope = talib.LINEARREG_SLOPE(random_data, timeperiod=14)
        expected_intercept = talib.LINEARREG_INTERCEPT(random_data, timeperiod=14)
        expected_angle = talib.LINEARREG_ANGLE(random_data, timeperiod=14)

        result_slope = result["slope"].to_numpy()
        result_intercept = result["intercept"].to_numpy()
        result_angle = result["angle"].to_numpy()

        valid_idx = ~np.isnan(expected_slope)
        if np.any(valid_idx):
            assert_allclose(
                result_slope[valid_idx],
                expected_slope[valid_idx],
                rtol=1e-7,
            )
            assert_allclose(
                result_intercept[valid_idx],
                expected_intercept[valid_idx],
                rtol=1e-7,
            )
            assert_allclose(
                result_angle[valid_idx],
                expected_angle[valid_idx],
                rtol=1e-7,
            )

    def test_linearreg_family_relationships(self):
        """Test relationships between slope, intercept, angle, and linearreg."""
        # For linear data, verify the mathematical relationships
        x = np.arange(20, dtype=float)
        m, b = 1.5, 20.0  # y = 1.5x + 20
        y = m * x + b

        period = 10

        slope = linearreg_slope(y, timeperiod=period)
        intercept = linearreg_intercept(y, timeperiod=period)
        angle = linearreg_angle(y, timeperiod=period)
        linearreg_result = linearreg(y, timeperiod=period)

        valid_idx = ~np.isnan(slope)

        # For perfect linear data:
        # 1. Slope should be constant and equal to m
        assert_allclose(slope[valid_idx], m, rtol=1e-10)

        # 2. Intercept won't be exactly b due to reversed x-axis
        # Just verify the relationship holds

        # 3. Angle should be arctan(slope) * 180/pi
        expected_angle = np.arctan(m) * 180 / np.pi
        assert_allclose(angle[valid_idx], expected_angle, rtol=1e-10)

        # 4. LINEARREG = intercept + slope * (period-1)
        expected_linearreg = intercept + slope * (period - 1)
        assert_allclose(linearreg_result[valid_idx], expected_linearreg[valid_idx], rtol=1e-10)

    def test_linearreg_slope_properties(self, linear_data, constant_data):
        """Test LINEARREG_SLOPE mathematical properties."""
        # For perfectly linear data, slope should be constant
        result = linearreg_slope(linear_data, timeperiod=5)
        valid = result[~np.isnan(result)]
        # y = 2x + 10, slope = 2
        assert_allclose(valid, 2.0, rtol=1e-10)

        # For constant data, slope should be 0
        result_const = linearreg_slope(constant_data, timeperiod=10)
        valid_const = result_const[~np.isnan(result_const)]
        assert_allclose(valid_const, 0.0, atol=1e-10)

    def test_linearreg_intercept_properties(self, linear_data, constant_data):
        """Test LINEARREG_INTERCEPT mathematical properties."""
        # For perfectly linear data y = 2x + 10
        # But with TA-Lib's reversed x-axis, need to calculate correctly
        result = linearreg_intercept(linear_data, timeperiod=5)
        expected = talib.LINEARREG_INTERCEPT(linear_data, timeperiod=5)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

        # For constant data, intercept should equal the constant
        result_const = linearreg_intercept(constant_data, timeperiod=10)
        valid_const = result_const[~np.isnan(result_const)]
        assert_allclose(valid_const, 50.0, atol=1e-10)

    def test_linearreg_angle_properties(self):
        """Test LINEARREG_ANGLE mathematical properties."""
        # Test specific angles
        # Slope of 0 -> angle of 0 degrees
        flat = np.array([100.0] * 20)
        angle_flat = linearreg_angle(flat, timeperiod=10)
        valid_flat = angle_flat[~np.isnan(angle_flat)]
        assert_allclose(valid_flat, 0.0, atol=1e-10)

        # Normal coordinate system:
        # Increasing trend -> positive slope -> positive angle
        x = np.arange(20, dtype=float)
        y = x + 100  # increasing trend
        angle_pos = linearreg_angle(y, timeperiod=10)
        valid_pos = angle_pos[~np.isnan(angle_pos)]
        assert_allclose(valid_pos, 45.0, rtol=1e-7)

        # Decreasing trend -> negative slope -> negative angle
        y_dec = -x + 100  # decreasing trend
        angle_neg = linearreg_angle(y_dec, timeperiod=10)
        valid_neg = angle_neg[~np.isnan(angle_neg)]
        assert_allclose(valid_neg, -45.0, rtol=1e-7)

    def test_linearreg_family_parameter_validation(self):
        """Test parameter validation for all indicators."""
        values = np.random.randn(100)

        # Invalid period for slope
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            linearreg_slope(values, timeperiod=1)

        # Invalid period for intercept
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            linearreg_intercept(values, timeperiod=1)

        # Invalid period for angle
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            linearreg_angle(values, timeperiod=1)

    def test_linearreg_family_nan_patterns(self):
        """Test NaN patterns for all indicators."""
        values = np.random.randn(100)
        period = 14

        # Test slope
        slope = linearreg_slope(values, timeperiod=period)
        expected_slope = talib.LINEARREG_SLOPE(values, timeperiod=period)
        assert np.array_equal(np.isnan(slope), np.isnan(expected_slope))

        # Test intercept
        intercept = linearreg_intercept(values, timeperiod=period)
        expected_intercept = talib.LINEARREG_INTERCEPT(values, timeperiod=period)
        assert np.array_equal(np.isnan(intercept), np.isnan(expected_intercept))

        # Test angle
        angle = linearreg_angle(values, timeperiod=period)
        expected_angle = talib.LINEARREG_ANGLE(values, timeperiod=period)
        assert np.array_equal(np.isnan(angle), np.isnan(expected_angle))

        # All should have first period-1 values as NaN
        assert np.all(np.isnan(slope[: period - 1]))
        assert np.all(np.isnan(intercept[: period - 1]))
        assert np.all(np.isnan(angle[: period - 1]))

    def test_linearreg_family_crypto_accuracy(self, crypto_data_small):
        """Test accuracy on real crypto data."""
        prices = crypto_data_small["close"].to_numpy()

        for period in [5, 10, 14, 20]:
            # Test slope
            expected_slope = talib.LINEARREG_SLOPE(prices, timeperiod=period)
            result_slope = linearreg_slope(prices, timeperiod=period)
            valid_idx = ~(np.isnan(expected_slope) | np.isnan(result_slope))
            if np.any(valid_idx):
                assert_allclose(
                    result_slope[valid_idx],
                    expected_slope[valid_idx],
                    rtol=1e-7,
                    err_msg=f"LINEARREG_SLOPE mismatch on crypto data for period {period}",
                )

            # Test intercept
            expected_intercept = talib.LINEARREG_INTERCEPT(prices, timeperiod=period)
            result_intercept = linearreg_intercept(prices, timeperiod=period)
            valid_idx = ~(np.isnan(expected_intercept) | np.isnan(result_intercept))
            if np.any(valid_idx):
                assert_allclose(
                    result_intercept[valid_idx],
                    expected_intercept[valid_idx],
                    rtol=1e-7,
                    err_msg=f"LINEARREG_INTERCEPT mismatch on crypto data for period {period}",
                )

            # Test angle
            expected_angle = talib.LINEARREG_ANGLE(prices, timeperiod=period)
            result_angle = linearreg_angle(prices, timeperiod=period)
            valid_idx = ~(np.isnan(expected_angle) | np.isnan(result_angle))
            if np.any(valid_idx):
                assert_allclose(
                    result_angle[valid_idx],
                    expected_angle[valid_idx],
                    rtol=1e-7,
                    err_msg=f"LINEARREG_ANGLE mismatch on crypto data for period {period}",
                )

    @pytest.mark.benchmark
    def test_linearreg_family_performance(
        self,
        crypto_data,
        performance_threshold,
        warmup_jit,
    ):
        """Benchmark Linear Regression family performance."""
        prices = crypto_data["close"].to_numpy()

        # Warmup JIT
        warmup_jit(linearreg_slope, prices[:100])
        warmup_jit(linearreg_intercept, prices[:100])
        warmup_jit(linearreg_angle, prices[:100])

        import time

        # Test LINEARREG_SLOPE
        start = time.perf_counter()
        _ = linearreg_slope(prices, timeperiod=14)
        our_slope_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = talib.LINEARREG_SLOPE(prices, timeperiod=14)
        talib_slope_time = time.perf_counter() - start

        print(f"\nLINEARREG_SLOPE Performance ({len(prices):,} rows):")
        print(f"Our implementation: {our_slope_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_slope_time * 1000:.2f}ms")
        print(f"Ratio: {our_slope_time / talib_slope_time:.2f}x")

        # Test LINEARREG_INTERCEPT
        start = time.perf_counter()
        _ = linearreg_intercept(prices, timeperiod=14)
        our_intercept_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = talib.LINEARREG_INTERCEPT(prices, timeperiod=14)
        talib_intercept_time = time.perf_counter() - start

        print(f"\nLINEARREG_INTERCEPT Performance ({len(prices):,} rows):")
        print(f"Our implementation: {our_intercept_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_intercept_time * 1000:.2f}ms")
        print(f"Ratio: {our_intercept_time / talib_intercept_time:.2f}x")

        # Test LINEARREG_ANGLE
        start = time.perf_counter()
        _ = linearreg_angle(prices, timeperiod=14)
        our_angle_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = talib.LINEARREG_ANGLE(prices, timeperiod=14)
        talib_angle_time = time.perf_counter() - start

        print(f"\nLINEARREG_ANGLE Performance ({len(prices):,} rows):")
        print(f"Our implementation: {our_angle_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_angle_time * 1000:.2f}ms")
        print(f"Ratio: {our_angle_time / talib_angle_time:.2f}x")

        # All should be reasonably fast
        # Linear regression is complex - use 3x threshold with 1.5 margin for CI
        threshold = performance_threshold("complex") * 1.5  # 4.5x for CI variability
        assert our_slope_time < talib_slope_time * threshold
        assert our_intercept_time < talib_intercept_time * threshold
        assert our_angle_time < talib_angle_time * threshold
