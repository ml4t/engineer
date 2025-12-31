"""
Test Normalized Average True Range (NATR) indicator implementation.

NATR = (ATR / Close) * 100
Expresses ATR as a percentage of close price for better comparability.
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

from ml4t.engineer.features.volatility import atr, natr


class TestNATR:
    """Test Normalized ATR indicator."""

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

    def test_natr_accuracy(self, price_data):
        """Test NATR matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [7, 14, 21]:
            # TA-Lib reference
            expected = talib.NATR(high, low, close, timeperiod=period)

            # Our implementation
            result = natr(high, low, close, period=period)

            # Should match exactly
            assert_allclose(
                result,
                expected,
                rtol=1e-10,
                equal_nan=True,
                err_msg=f"NATR mismatch for period={period}",
            )

    def test_natr_polars(self, price_data):
        """Test NATR with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expression
        result_df = df.with_columns(
            natr("high", "low", "close", period=14).alias("natr"),
        )

        # Compare with TA-Lib
        expected = talib.NATR(
            price_data["high"],
            price_data["low"],
            price_data["close"],
            timeperiod=14,
        )

        assert_allclose(
            result_df["natr"].to_numpy(),
            expected,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_natr_edge_cases(self):
        """Test NATR with edge cases."""
        # Case 1: Constant prices
        high = np.full(20, 100.0)
        low = np.full(20, 100.0)
        close = np.full(20, 100.0)

        expected = talib.NATR(high, low, close, timeperiod=14)
        result = natr(high, low, close, period=14)

        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

        # Case 2: Zero close price (should handle division)
        high = np.array([10.0, 11.0, 12.0, 11.5, 10.0])
        low = np.array([9.0, 9.5, 10.5, 10.0, 9.0])
        close = np.array([9.5, 10.5, 11.0, 10.5, 0.0])  # Last close is 0

        # Our implementation should handle this gracefully
        result = natr(high, low, close, period=3)
        # Last value should be inf or very large since we're dividing by 0
        assert np.isnan(result[-1]) or np.isinf(result[-1])

    def test_natr_calculation_relationship(self):
        """Test that NATR = (ATR / Close) * 100."""
        high = np.random.randn(100) + 100
        low = high - np.abs(np.random.randn(100))
        close = (high + low) / 2 + np.random.randn(100) * 0.1

        # Ensure high >= low
        high = np.maximum(high, low)

        period = 14

        # Calculate ATR and NATR
        atr_values = atr(high, low, close, period=period)
        natr_values = natr(high, low, close, period=period)

        # NATR should equal (ATR / Close) * 100
        expected_natr = (
            np.divide(
                atr_values,
                close,
                out=np.full_like(atr_values, np.nan),
                where=close != 0,
            )
            * 100
        )

        # Compare where both are not NaN
        valid_mask = ~(np.isnan(natr_values) | np.isnan(expected_natr))
        assert_allclose(natr_values[valid_mask], expected_natr[valid_mask], rtol=1e-10)

    def test_natr_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([100.0, 101.0])
        low = np.array([99.0, 100.0])
        close = np.array([100.0, 100.5])

        # Period too small
        with pytest.raises(ValueError, match="period must be >= 1"):
            natr(high, low, close, period=0)

        # Arrays of different lengths
        with pytest.raises(ValueError, match="must have the same length"):
            natr(high[:-1], low, close, period=1)

    def test_natr_crypto_accuracy(self, crypto_data_small):
        """Test NATR accuracy on real crypto data."""
        # Extract columns from 10K rows of real crypto data
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        # Test with different periods
        for period in [7, 14, 21]:
            # TA-Lib reference
            expected = talib.NATR(high, low, close, timeperiod=period)

            # Our implementation
            result = natr(high, low, close, period=period)

            # Should match closely on real data
            assert_allclose(
                result,
                expected,
                rtol=1e-10,
                equal_nan=True,
                err_msg=f"NATR mismatch for period={period} on real crypto data",
            )

    @pytest.mark.benchmark
    def test_natr_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark NATR performance using real crypto data."""
        # Use real crypto data
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(natr, high[:100], low[:100], close[:100], 14)

        import time

        # Our implementation (warmed up)
        start = time.perf_counter()
        _ = natr(high, low, close, period=14)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.NATR(high, low, close, timeperiod=14)
        talib_time = time.perf_counter() - start

        print(f"\nNATR Performance ({len(high):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use dynamic threshold based on complexity (Wilder's smoothing)
        threshold = performance_threshold("wilders")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
