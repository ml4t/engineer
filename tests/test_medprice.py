"""
Test Median Price (MEDPRICE) indicator implementation.

MEDPRICE = (High + Low) / 2
The midpoint between high and low prices.
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

from ml4t.engineer.features.price_transform import medprice


class TestMedianPrice:
    """Test Median Price indicator."""

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

    def test_medprice_accuracy(self, price_data):
        """Test MEDPRICE matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]

        # TA-Lib reference
        expected = talib.MEDPRICE(high, low)

        # Our implementation
        result = medprice(high, low)

        # Should match exactly
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    def test_medprice_polars(self, price_data):
        """Test MEDPRICE with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expression
        result_df = df.with_columns(medprice("high", "low").alias("medprice"))

        # Compare with TA-Lib
        expected = talib.MEDPRICE(price_data["high"], price_data["low"])

        assert_allclose(
            result_df["medprice"].to_numpy(),
            expected,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_medprice_edge_cases(self):
        """Test MEDPRICE with edge cases."""
        # Case 1: Single value
        high = np.array([101.0])
        low = np.array([99.0])

        result = medprice(high, low)
        expected = (101.0 + 99.0) / 2.0
        assert_allclose(result, expected, rtol=1e-10)

        # Case 2: High equals low
        high = np.full(10, 100.0)
        low = np.full(10, 100.0)

        result = medprice(high, low)
        expected = np.full(10, 100.0)
        assert_allclose(result, expected, rtol=1e-10)

        # Case 3: With NaN values
        high = np.array([101.0, np.nan, 103.0])
        low = np.array([99.0, 100.0, 101.0])

        result = medprice(high, low)
        # Second value should be NaN due to NaN in high
        assert ~np.isnan(result[0])
        assert np.isnan(result[1])
        assert ~np.isnan(result[2])

    def test_medprice_calculation_manual(self):
        """Test MEDPRICE calculation manually."""
        high = np.array([102.0, 103.0, 104.0])
        low = np.array([98.0, 99.0, 100.0])

        result = medprice(high, low)

        # Manual calculation: (H + L) / 2
        expected = np.array(
            [
                (102.0 + 98.0) / 2.0,  # 100.0
                (103.0 + 99.0) / 2.0,  # 101.0
                (104.0 + 100.0) / 2.0,  # 102.0
            ],
        )

        assert_allclose(result, expected, rtol=1e-10)

    def test_medprice_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([101.0, 102.0])
        low = np.array([99.0])  # Wrong length

        with pytest.raises(ValueError, match="must have the same length"):
            medprice(high, low)

    def test_medprice_crypto_accuracy(self, crypto_data_small):
        """Test MEDPRICE accuracy on real crypto data."""
        # Extract columns from 10K rows of real crypto data
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()

        # TA-Lib reference
        expected = talib.MEDPRICE(high, low)

        # Our implementation
        result = medprice(high, low)

        # Should match exactly on real data
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    @pytest.mark.benchmark
    def test_medprice_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark MEDPRICE performance using real crypto data."""
        # Use real crypto data
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(medprice, high[:100], low[:100])

        import time

        # Our implementation (warmed up)
        start = time.perf_counter()
        _ = medprice(high, low)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.MEDPRICE(high, low)
        talib_time = time.perf_counter() - start

        print(f"\nMEDPRICE Performance ({len(high):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use dynamic threshold based on complexity (simple arithmetic)
        threshold = performance_threshold("simple")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
