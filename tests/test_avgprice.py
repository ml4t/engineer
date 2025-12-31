"""
Test Average Price (AVGPRICE) indicator implementation.

AVGPRICE = (Open + High + Low + Close) / 4
Simple average of OHLC prices.
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

from ml4t.engineer.features.price_transform import avgprice


class TestAveragePrice:
    """Test Average Price indicator."""

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

    def test_avgprice_accuracy(self, price_data):
        """Test AVGPRICE matches TA-Lib exactly."""
        open_ = price_data["open"]
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.AVGPRICE(open_, high, low, close)

        # Our implementation
        result = avgprice(open_, high, low, close)

        # Should match exactly
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    def test_avgprice_polars(self, price_data):
        """Test AVGPRICE with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expression
        result_df = df.with_columns(
            avgprice("open", "high", "low", "close").alias("avgprice"),
        )

        # Compare with TA-Lib
        expected = talib.AVGPRICE(
            price_data["open"],
            price_data["high"],
            price_data["low"],
            price_data["close"],
        )

        assert_allclose(
            result_df["avgprice"].to_numpy(),
            expected,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_avgprice_edge_cases(self):
        """Test AVGPRICE with edge cases."""
        # Case 1: Single value
        open_ = np.array([100.0])
        high = np.array([101.0])
        low = np.array([99.0])
        close = np.array([100.5])

        result = avgprice(open_, high, low, close)
        expected = (100.0 + 101.0 + 99.0 + 100.5) / 4.0
        assert_allclose(result, expected, rtol=1e-10)

        # Case 2: All same prices
        open_ = np.full(10, 100.0)
        high = np.full(10, 100.0)
        low = np.full(10, 100.0)
        close = np.full(10, 100.0)

        result = avgprice(open_, high, low, close)
        expected = np.full(10, 100.0)
        assert_allclose(result, expected, rtol=1e-10)

        # Case 3: With NaN values
        open_ = np.array([100.0, np.nan, 102.0])
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([100.5, 101.0, 102.5])

        result = avgprice(open_, high, low, close)
        # Second value should be NaN due to NaN in open
        assert ~np.isnan(result[0])
        assert np.isnan(result[1])
        assert ~np.isnan(result[2])

    def test_avgprice_calculation_manual(self):
        """Test AVGPRICE calculation manually."""
        open_ = np.array([100.0, 101.0, 102.0])
        high = np.array([102.0, 103.0, 104.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([101.0, 102.0, 103.0])

        result = avgprice(open_, high, low, close)

        # Manual calculation
        expected = np.array(
            [
                (100.0 + 102.0 + 99.0 + 101.0) / 4.0,  # 100.5
                (101.0 + 103.0 + 100.0 + 102.0) / 4.0,  # 101.5
                (102.0 + 104.0 + 101.0 + 103.0) / 4.0,  # 102.5
            ],
        )

        assert_allclose(result, expected, rtol=1e-10)

    def test_avgprice_parameter_validation(self):
        """Test parameter validation."""
        open_ = np.array([100.0, 101.0])
        high = np.array([101.0, 102.0])
        low = np.array([99.0])  # Wrong length
        close = np.array([100.0, 101.0])

        with pytest.raises(ValueError, match="must have the same length"):
            avgprice(open_, high, low, close)

    def test_avgprice_crypto_accuracy(self, crypto_data_small):
        """Test AVGPRICE accuracy on real crypto data."""
        # Extract columns from 10K rows of real crypto data
        open_ = crypto_data_small["open"].to_numpy()
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        # TA-Lib reference
        expected = talib.AVGPRICE(open_, high, low, close)

        # Our implementation
        result = avgprice(open_, high, low, close)

        # Should match exactly on real data
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    @pytest.mark.benchmark
    def test_avgprice_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark AVGPRICE performance using real crypto data."""
        # Use real crypto data
        open_ = crypto_data["open"].to_numpy()
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(avgprice, open_[:100], high[:100], low[:100], close[:100])

        import time

        # Our implementation (warmed up)
        start = time.perf_counter()
        _ = avgprice(open_, high, low, close)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.AVGPRICE(open_, high, low, close)
        talib_time = time.perf_counter() - start

        print(f"\nAVGPRICE Performance ({len(open_):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use dynamic threshold based on complexity (simple arithmetic)
        threshold = performance_threshold("simple")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
