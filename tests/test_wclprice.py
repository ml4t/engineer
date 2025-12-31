"""
Test Weighted Close Price (WCLPRICE) indicator implementation.

WCLPRICE = (High + Low + 2*Close) / 4
Gives more weight to the closing price.
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

from ml4t.engineer.features.price_transform import typprice, wclprice


class TestWeightedClosePrice:
    """Test Weighted Close Price indicator."""

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

        return {"open": open_, "high": high, "low": low, "close": close}

    def test_wclprice_accuracy(self, price_data):
        """Test WCLPRICE matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.WCLPRICE(high, low, close)

        # Our implementation
        result = wclprice(high, low, close)

        # Should match exactly
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    def test_wclprice_polars(self, price_data):
        """Test WCLPRICE with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expression
        result_df = df.with_columns(
            wclprice("high", "low", "close").alias("wclprice"),
        )

        # Compare with TA-Lib
        expected = talib.WCLPRICE(
            price_data["high"],
            price_data["low"],
            price_data["close"],
        )

        assert_allclose(
            result_df["wclprice"].to_numpy(),
            expected,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_wclprice_edge_cases(self):
        """Test WCLPRICE with edge cases."""
        # Case 1: Single value
        high = np.array([101.0])
        low = np.array([99.0])
        close = np.array([100.5])

        result = wclprice(high, low, close)
        expected = (101.0 + 99.0 + 2 * 100.5) / 4.0
        assert_allclose(result, expected, rtol=1e-10)

        # Case 2: All same prices
        high = np.full(10, 100.0)
        low = np.full(10, 100.0)
        close = np.full(10, 100.0)

        result = wclprice(high, low, close)
        expected = np.full(10, 100.0)
        assert_allclose(result, expected, rtol=1e-10)

        # Case 3: With NaN values
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([100.5, np.nan, 102.5])

        result = wclprice(high, low, close)
        # Second value should be NaN due to NaN in close
        assert ~np.isnan(result[0])
        assert np.isnan(result[1])
        assert ~np.isnan(result[2])

    def test_wclprice_calculation_manual(self):
        """Test WCLPRICE calculation manually."""
        high = np.array([102.0, 103.0, 104.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([101.0, 102.0, 103.0])

        result = wclprice(high, low, close)

        # Manual calculation: (H + L + 2*C) / 4
        expected = np.array(
            [
                (102.0 + 99.0 + 2 * 101.0) / 4.0,  # 100.75
                (103.0 + 100.0 + 2 * 102.0) / 4.0,  # 101.75
                (104.0 + 101.0 + 2 * 103.0) / 4.0,  # 102.75
            ],
        )

        assert_allclose(result, expected, rtol=1e-10)

    def test_wclprice_vs_typprice(self):
        """Test that WCLPRICE gives more weight to close than TYPPRICE."""
        # When close is significantly different from high/low
        high = np.array([110.0])
        low = np.array([90.0])
        close = np.array([95.0])  # Close is near low

        wclprice_result = wclprice(high, low, close)
        typprice_result = typprice(high, low, close)

        # WCLPRICE should be closer to close than TYPPRICE
        wclprice_diff = abs(wclprice_result[0] - close[0])
        typprice_diff = abs(typprice_result[0] - close[0])

        assert wclprice_diff < typprice_diff

    def test_wclprice_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([101.0, 102.0])
        low = np.array([99.0])  # Wrong length
        close = np.array([100.0, 101.0])

        with pytest.raises(ValueError, match="must have the same length"):
            wclprice(high, low, close)

    def test_wclprice_crypto_accuracy(self, crypto_data_small):
        """Test WCLPRICE accuracy on real crypto data."""
        # Extract columns from 10K rows of real crypto data
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        # TA-Lib reference
        expected = talib.WCLPRICE(high, low, close)

        # Our implementation
        result = wclprice(high, low, close)

        # Should match exactly on real data
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    @pytest.mark.benchmark
    def test_wclprice_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark WCLPRICE performance using real crypto data."""
        # Use real crypto data
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(wclprice, high[:100], low[:100], close[:100])

        import time

        # Our implementation (warmed up)
        start = time.perf_counter()
        _ = wclprice(high, low, close)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.WCLPRICE(high, low, close)
        talib_time = time.perf_counter() - start

        print(f"\nWCLPRICE Performance ({len(high):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use dynamic threshold based on complexity
        threshold = performance_threshold("simple")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
