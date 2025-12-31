"""
Test TRIX indicator implementation.

TRIX is a momentum oscillator that shows the percent rate of change
of a triple exponentially smoothed moving average.
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

from ml4t.engineer.features.momentum import trix


class TestTRIX:
    """Test TRIX indicator."""

    @pytest.fixture
    def price_data(self):
        """Generate sample price data."""
        np.random.seed(42)
        n = 200  # Need more data for triple smoothing

        # Generate trending data with some noise
        trend = np.linspace(100, 120, n)
        noise = np.random.normal(0, 0.5, n)
        values = trend + noise

        return values

    def test_trix_accuracy(self, price_data):
        """Test TRIX matches TA-Lib exactly."""
        # Test with default period
        expected = talib.TRIX(price_data)
        result = trix(price_data)

        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

        # Test with custom periods
        for period in [10, 20, 30, 50]:
            expected = talib.TRIX(price_data, timeperiod=period)
            result = trix(price_data, timeperiod=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-10,
                equal_nan=True,
                err_msg=f"TRIX mismatch for period {period}",
            )

    def test_trix_polars(self, price_data):
        """Test TRIX with Polars expressions."""
        df = pl.DataFrame({"close": price_data})

        # Using Polars expression
        result_df = df.with_columns(trix("close").alias("trix"))

        # Compare with TA-Lib
        expected = talib.TRIX(price_data)

        assert_allclose(
            result_df["trix"].to_numpy(),
            expected,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_trix_edge_cases(self):
        """Test TRIX with edge cases."""
        # Case 1: Small dataset
        values = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        result = trix(values, timeperiod=2)
        expected = talib.TRIX(values, timeperiod=2)

        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

        # Case 2: Constant values
        values = np.full(100, 100.0)
        result = trix(values)
        expected = talib.TRIX(values)

        # For constant values, TRIX should be 0 (no change) or NaN
        # Use slightly higher tolerance for floating point precision
        assert_allclose(result, expected, rtol=1e-9, atol=1e-12, equal_nan=True)

        # Case 3: Linear increase
        values = np.arange(1, 101, dtype=float)
        result = trix(values, timeperiod=10)
        expected = talib.TRIX(values, timeperiod=10)

        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    def test_trix_nan_pattern(self):
        """Test TRIX NaN pattern matches TA-Lib."""
        # Create data with known pattern
        n = 100
        values = np.random.randn(n) + 100

        for period in [10, 20, 30]:
            result = trix(values, timeperiod=period)
            expected = talib.TRIX(values, timeperiod=period)

            # Check NaN pattern
            result_nan = np.isnan(result)
            expected_nan = np.isnan(expected)

            assert np.array_equal(
                result_nan,
                expected_nan,
            ), f"NaN pattern mismatch for period {period}"

    def test_trix_parameter_validation(self):
        """Test parameter validation."""
        values = np.random.randn(50) + 100

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be >= 1"):
            trix(values, timeperiod=0)

    def test_trix_crypto_accuracy(self, crypto_data_small):
        """Test TRIX accuracy on real crypto data."""
        # Use close prices from crypto data
        close = crypto_data_small["close"].to_numpy()

        # Test with different periods
        for period in [10, 20, 30, 50]:
            expected = talib.TRIX(close, timeperiod=period)
            result = trix(close, timeperiod=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-7,
                equal_nan=True,
                err_msg=f"TRIX mismatch on crypto data for period {period}",
            )

    @pytest.mark.benchmark
    def test_trix_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark TRIX performance using real crypto data."""
        # Use real crypto data
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(trix, close[:100])

        import time

        # Our implementation (warmed up)
        start = time.perf_counter()
        _ = trix(close)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.TRIX(close)
        talib_time = time.perf_counter() - start

        print(f"\nTRIX Performance ({len(close):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use dynamic threshold based on complexity (complex - triple EMA)
        threshold = performance_threshold("complex")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
