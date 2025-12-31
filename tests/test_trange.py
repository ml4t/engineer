"""
Test True Range (TRANGE) indicator implementation.

True Range is the greatest of:
- Current High minus current Low
- Absolute value of current High minus previous Close
- Absolute value of current Low minus previous Close
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

from ml4t.engineer.features.volatility import trange


class TestTrueRange:
    """Test True Range indicator."""

    @pytest.fixture
    def price_data(self):
        """Generate sample OHLC data."""
        np.random.seed(42)
        n = 100

        # Generate realistic price movements
        returns = np.random.normal(0.0001, 0.02, n)
        close = 100 * (1 + returns).cumprod()

        # Generate OHLC data
        high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_ = np.roll(close, 1)
        open_[0] = 100

        return {"high": high, "low": low, "close": close}

    def test_trange_accuracy(self, price_data):
        """Test TRANGE matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.TRANGE(high, low, close)

        # Our implementation
        result = trange(high, low, close)

        # Should match exactly
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    def test_trange_polars(self, price_data):
        """Test TRANGE with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expression
        result_df = df.with_columns(trange("high", "low", "close").alias("trange"))

        # Compare with TA-Lib
        expected = talib.TRANGE(
            price_data["high"],
            price_data["low"],
            price_data["close"],
        )

        assert_allclose(
            result_df["trange"].to_numpy(),
            expected,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_trange_edge_cases(self):
        """Test TRANGE with edge cases."""
        # Case 1: Single value
        high = np.array([100.0])
        low = np.array([99.0])
        close = np.array([99.5])

        result = trange(high, low, close)
        # First value should be NaN (no previous close)
        assert np.isnan(result[0])

        # Case 2: Constant prices
        high = np.full(10, 100.0)
        low = np.full(10, 100.0)
        close = np.full(10, 100.0)

        expected = talib.TRANGE(high, low, close)
        result = trange(high, low, close)

        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

        # Case 3: Large gap (test abs value logic)
        high = np.array([100.0, 105.0, 103.0])
        low = np.array([98.0, 102.0, 101.0])
        close = np.array([99.0, 104.0, 102.0])

        expected = talib.TRANGE(high, low, close)
        result = trange(high, low, close)

        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    def test_trange_calculation_details(self):
        """Test specific TRANGE calculations."""
        # Manual calculation test
        high = np.array([10.0, 11.0, 12.0, 11.5])
        low = np.array([9.0, 9.5, 10.5, 10.0])
        close = np.array([9.5, 10.5, 11.0, 10.5])

        result = trange(high, low, close)

        # First value is NaN (no previous close)
        assert np.isnan(result[0])

        # Second value: max(11-9.5, |11-9.5|, |9.5-9.5|) = max(1.5, 1.5, 0) = 1.5
        assert_allclose(result[1], 1.5, rtol=1e-10)

        # Third value: max(12-10.5, |12-10.5|, |10.5-10.5|) = max(1.5, 1.5, 0) = 1.5
        assert_allclose(result[2], 1.5, rtol=1e-10)

        # Fourth value: max(11.5-10, |11.5-11|, |10-11|) = max(1.5, 0.5, 1.0) = 1.5
        assert_allclose(result[3], 1.5, rtol=1e-10)

    def test_trange_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([100.0, 101.0])
        low = np.array([99.0])  # Wrong length
        close = np.array([100.0, 100.5])

        with pytest.raises(ValueError, match="must have the same length"):
            trange(high, low, close)

    def test_trange_crypto_accuracy(self, crypto_data_small):
        """Test TRANGE accuracy on real crypto data."""
        # Extract columns from 10K rows of real crypto data
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        # TA-Lib reference
        expected = talib.TRANGE(high, low, close)

        # Our implementation
        result = trange(high, low, close)

        # Should match exactly on real data
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    @pytest.mark.benchmark
    def test_trange_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark TRANGE performance using real crypto data."""
        # Use real crypto data
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(trange, high[:100], low[:100], close[:100])

        import time

        # Our implementation (warmed up)
        start = time.perf_counter()
        _ = trange(high, low, close)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.TRANGE(high, low, close)
        talib_time = time.perf_counter() - start

        print(f"\nTRANGE Performance ({len(high):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use dynamic threshold based on complexity (rolling calculation)
        threshold = performance_threshold("rolling")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
