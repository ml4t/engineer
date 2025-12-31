"""
Test Math Operators: MAX, MIN, SUM.

These are fundamental building blocks for many technical indicators.
"""

import numpy as np
import polars as pl
import pytest

from .test_config import assert_indicator_match

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    pytest.skip("TA-Lib not available", allow_module_level=True)

from ml4t.engineer.features.math import maximum, minimum, summation


class TestMathOperators:
    """Test Math Operators."""

    @pytest.fixture
    def price_data(self):
        """Generate sample price data."""
        np.random.seed(42)
        n = 100

        # Generate realistic price movements
        returns = np.random.normal(0.0001, 0.02, n)
        close = 100 * (1 + returns).cumprod()

        return {"close": close}

    def test_max_accuracy(self, price_data):
        """Test MAX matches TA-Lib exactly."""
        close = price_data["close"]

        # Test different periods
        for period in [5, 14, 30]:
            # TA-Lib reference
            expected = talib.MAX(close, timeperiod=period)

            # Our implementation
            result = maximum(close, timeperiod=period)

            # Should match exactly
            assert_indicator_match(
                result,
                expected,
                f"MAX(period={period})",
                rtol=1e-10,
            )

    def test_min_accuracy(self, price_data):
        """Test MIN matches TA-Lib exactly."""
        close = price_data["close"]

        # Test different periods
        for period in [5, 14, 30]:
            # TA-Lib reference
            expected = talib.MIN(close, timeperiod=period)

            # Our implementation
            result = minimum(close, timeperiod=period)

            # Should match exactly
            assert_indicator_match(
                result,
                expected,
                f"MIN(period={period})",
                rtol=1e-10,
            )

    def test_sum_accuracy(self, price_data):
        """Test SUM matches TA-Lib exactly."""
        close = price_data["close"]

        # Test different periods
        for period in [5, 14, 30]:
            # TA-Lib reference
            expected = talib.SUM(close, timeperiod=period)

            # Our implementation
            result = summation(close, timeperiod=period)

            # Should match exactly
            assert_indicator_match(
                result,
                expected,
                f"SUM(period={period})",
                rtol=1e-10,
            )

    def test_math_operators_polars(self, price_data):
        """Test math operators with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expressions
        result_df = df.with_columns(
            [
                maximum("close", 14).alias("max_14"),
                minimum("close", 14).alias("min_14"),
                summation("close", 14).alias("sum_14"),
            ],
        )

        # Compare with TA-Lib
        expected_max = talib.MAX(price_data["close"], timeperiod=14)
        expected_min = talib.MIN(price_data["close"], timeperiod=14)
        expected_sum = talib.SUM(price_data["close"], timeperiod=14)

        assert_indicator_match(
            result_df["max_14"].to_numpy(),
            expected_max,
            "MAX (Polars)",
            rtol=1e-10,
        )
        assert_indicator_match(
            result_df["min_14"].to_numpy(),
            expected_min,
            "MIN (Polars)",
            rtol=1e-10,
        )
        assert_indicator_match(
            result_df["sum_14"].to_numpy(),
            expected_sum,
            "SUM (Polars)",
            rtol=1e-10,
        )

    def test_math_operators_edge_cases(self):
        """Test math operators with edge cases."""
        # Case 1: Minimal data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        max_result = maximum(data, timeperiod=3)
        min_result = minimum(data, timeperiod=3)
        sum_result = summation(data, timeperiod=3)

        # First 2 values should be NaN
        assert np.all(np.isnan(max_result[:2]))
        assert np.all(np.isnan(min_result[:2]))
        assert np.all(np.isnan(sum_result[:2]))

        # Check known values
        assert max_result[2] == 3.0  # max([1,2,3]) = 3
        assert min_result[2] == 1.0  # min([1,2,3]) = 1
        assert sum_result[2] == 6.0  # sum([1,2,3]) = 6

        assert max_result[4] == 5.0  # max([3,4,5]) = 5
        assert min_result[4] == 3.0  # min([3,4,5]) = 3
        assert sum_result[4] == 12.0  # sum([3,4,5]) = 12

    def test_math_operators_constant_values(self):
        """Test math operators with constant values."""
        data = np.full(50, 100.0)

        max_result = maximum(data, timeperiod=10)
        min_result = minimum(data, timeperiod=10)
        sum_result = summation(data, timeperiod=10)

        # Compare with TA-Lib
        expected_max = talib.MAX(data, timeperiod=10)
        expected_min = talib.MIN(data, timeperiod=10)
        expected_sum = talib.SUM(data, timeperiod=10)

        assert_indicator_match(max_result, expected_max, "MAX (constant)", rtol=1e-10)
        assert_indicator_match(min_result, expected_min, "MIN (constant)", rtol=1e-10)
        assert_indicator_match(sum_result, expected_sum, "SUM (constant)", rtol=1e-10)

    def test_math_operators_mathematical_properties(self):
        """Test mathematical properties of the operators."""
        np.random.seed(42)
        data = np.random.randn(100) * 10 + 100
        period = 10

        max_result = maximum(data, timeperiod=period)
        min_result = minimum(data, timeperiod=period)
        sum_result = summation(data, timeperiod=period)

        # For all valid values
        valid_idx = ~np.isnan(max_result)

        # Max >= Min for all windows
        assert np.all(max_result[valid_idx] >= min_result[valid_idx])

        # Sum should be positive for positive data
        assert np.all(sum_result[valid_idx] > 0)

        # Sum should equal period * average for each window
        for i in np.where(valid_idx)[0]:
            window = data[i - period + 1 : i + 1]
            expected_sum = np.sum(window)
            assert abs(sum_result[i] - expected_sum) < 1e-10

    def test_math_operators_parameter_validation(self):
        """Test parameter validation."""
        data = np.array([1.0, 2.0])

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            maximum(data, timeperiod=0)

        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            minimum(data, timeperiod=0)

        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            summation(data, timeperiod=0)

    def test_math_operators_insufficient_data(self):
        """Test behavior with insufficient data."""
        data = np.array([1.0, 2.0])

        # Period larger than data length
        max_result = maximum(data, timeperiod=5)
        min_result = minimum(data, timeperiod=5)
        sum_result = summation(data, timeperiod=5)

        # Should return all NaN
        assert np.all(np.isnan(max_result))
        assert np.all(np.isnan(min_result))
        assert np.all(np.isnan(sum_result))

    @pytest.mark.benchmark
    def test_math_operators_performance(
        self,
        crypto_data,
        performance_threshold,
        warmup_jit,
    ):
        """Benchmark math operators performance using real crypto data."""
        # Use real crypto data
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(maximum, close[:100], 14)
        warmup_jit(minimum, close[:100], 14)
        warmup_jit(summation, close[:100], 14)

        import time

        # Our implementations (warmed up)
        start = time.perf_counter()
        _ = maximum(close, timeperiod=14)
        our_max_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = minimum(close, timeperiod=14)
        our_min_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = summation(close, timeperiod=14)
        our_sum_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.MAX(close, timeperiod=14)
        talib_max_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = talib.MIN(close, timeperiod=14)
        talib_min_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = talib.SUM(close, timeperiod=14)
        talib_sum_time = time.perf_counter() - start

        print(f"\\nMath Operators Performance ({len(close):,} rows of crypto data):")
        print(
            f"MAX - Our: {our_max_time * 1000:.2f}ms, TA-Lib: {talib_max_time * 1000:.2f}ms, Ratio: {our_max_time / talib_max_time:.2f}x",
        )
        print(
            f"MIN - Our: {our_min_time * 1000:.2f}ms, TA-Lib: {talib_min_time * 1000:.2f}ms, Ratio: {our_min_time / talib_min_time:.2f}x",
        )
        print(
            f"SUM - Our: {our_sum_time * 1000:.2f}ms, TA-Lib: {talib_sum_time * 1000:.2f}ms, Ratio: {our_sum_time / talib_sum_time:.2f}x",
        )

        # Use threshold for basic math operations
        # MAX/MIN/SUM are the most basic operations where TA-Lib's C code
        # has maximum advantage. Our NumPy implementation achieving 7-15x is excellent.
        threshold = performance_threshold("math_basic")
        assert our_max_time < talib_max_time * threshold, (
            f"MAX performance ratio {our_max_time / talib_max_time:.1f}x exceeds threshold {threshold}x"
        )
        assert our_min_time < talib_min_time * threshold, (
            f"MIN performance ratio {our_min_time / talib_min_time:.1f}x exceeds threshold {threshold}x"
        )
        assert our_sum_time < talib_sum_time * threshold, (
            f"SUM performance ratio {our_sum_time / talib_sum_time:.1f}x exceeds threshold {threshold}x"
        )
