"""
Test Ultimate Oscillator (ULTOSC) indicator implementation.

The Ultimate Oscillator combines short, medium, and long-term momentum
to reduce false signals.
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

from ml4t.engineer.features.momentum import ultosc


class TestUltimateOscillator:
    """Test Ultimate Oscillator indicator."""

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

        return {"high": high, "low": low, "close": close}

    def test_ultosc_accuracy(self, price_data):
        """Test ULTOSC matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # Test with default parameters
        expected = talib.ULTOSC(high, low, close)
        result = ultosc(high, low, close)

        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

        # Test with custom parameters
        for periods in [(5, 10, 20), (10, 20, 40), (3, 7, 14)]:
            expected = talib.ULTOSC(
                high,
                low,
                close,
                timeperiod1=periods[0],
                timeperiod2=periods[1],
                timeperiod3=periods[2],
            )
            result = ultosc(
                high,
                low,
                close,
                timeperiod1=periods[0],
                timeperiod2=periods[1],
                timeperiod3=periods[2],
            )

            assert_allclose(
                result,
                expected,
                rtol=1e-10,
                equal_nan=True,
                err_msg=f"ULTOSC mismatch for periods {periods}",
            )

    def test_ultosc_polars(self, price_data):
        """Test ULTOSC with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expression
        result_df = df.with_columns(ultosc("high", "low", "close").alias("ultosc"))

        # Compare with TA-Lib
        expected = talib.ULTOSC(
            price_data["high"],
            price_data["low"],
            price_data["close"],
        )

        assert_allclose(
            result_df["ultosc"].to_numpy(),
            expected,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_ultosc_edge_cases(self):
        """Test ULTOSC with edge cases."""
        # Case 1: Small dataset (less than longest period)
        n = 20
        high = np.random.randn(n) + 100
        low = high - np.abs(np.random.randn(n))
        close = (high + low) / 2

        result = ultosc(high, low, close)
        expected = talib.ULTOSC(high, low, close)

        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

        # Case 2: Constant prices
        high = np.full(50, 100.0)
        low = np.full(50, 100.0)
        close = np.full(50, 100.0)

        result = ultosc(high, low, close)
        # With constant prices, BP and TR will be 0, leading to NaN
        assert np.all(np.isnan(result[1:]))  # First value is always NaN

    def test_ultosc_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([100.0, 101.0])
        low = np.array([99.0])  # Wrong length
        close = np.array([100.0, 100.5])

        with pytest.raises(ValueError, match="must have the same length"):
            ultosc(high, low, close)

        # Invalid time periods
        high = np.random.randn(50) + 100
        low = high - 1
        close = (high + low) / 2

        with pytest.raises(ValueError, match="must be positive"):
            ultosc(high, low, close, timeperiod1=0)

        with pytest.raises(ValueError, match="must be in ascending order"):
            ultosc(high, low, close, timeperiod1=20, timeperiod2=14, timeperiod3=28)

    def test_ultosc_crypto_accuracy(self, crypto_data_small):
        """Test ULTOSC accuracy on real crypto data."""
        # Use real crypto data
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        # Test with different parameter sets
        test_params = [
            (7, 14, 28),  # Default
            (5, 10, 20),  # Shorter periods
            (10, 20, 40),  # Longer periods
        ]

        for period1, period2, period3 in test_params:
            expected = talib.ULTOSC(
                high,
                low,
                close,
                timeperiod1=period1,
                timeperiod2=period2,
                timeperiod3=period3,
            )
            result = ultosc(
                high,
                low,
                close,
                timeperiod1=period1,
                timeperiod2=period2,
                timeperiod3=period3,
            )

            assert_allclose(
                result,
                expected,
                rtol=1e-10,
                equal_nan=True,
                err_msg=f"ULTOSC mismatch on crypto data for periods ({period1}, {period2}, {period3})",
            )

    @pytest.mark.benchmark
    def test_ultosc_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark ULTOSC performance using real crypto data."""
        # Use real crypto data
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(ultosc, high[:100], low[:100], close[:100])

        import time

        # Our implementation (warmed up)
        start = time.perf_counter()
        _ = ultosc(high, low, close)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.ULTOSC(high, low, close)
        talib_time = time.perf_counter() - start

        print(f"\nULTOSC Performance ({len(high):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # ULTOSC calculates 3 separate rolling windows (7, 14, 28 periods)
        # TA-Lib's pure C has significant advantage for multiple passes
        threshold = performance_threshold("multi_window")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
