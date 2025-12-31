"""
Test Chaikin A/D Line (AD) indicator implementation.

AD = Cumulative sum of Money Flow Volume
Money Flow Volume = ((Close - Low) - (High - Close)) / (High - Low) * Volume

When High == Low, Money Flow Multiplier = 0
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

from ml4t.engineer.features.volume import ad


class TestAD:
    """Test Chaikin A/D Line indicator."""

    @pytest.fixture
    def price_volume_data(self):
        """Generate sample OHLCV data."""
        np.random.seed(42)
        n = 100

        # Generate realistic price movements
        returns = np.random.normal(0.0001, 0.02, n)
        close = 100 * (1 + returns).cumprod()

        # Generate OHLC data
        high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open = close + np.random.normal(0, 0.1, n)

        # Generate volume
        volume = np.abs(np.random.normal(1000000, 200000, n))

        return {
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    def test_ad_accuracy(self, price_volume_data):
        """Test AD matches TA-Lib exactly."""
        high = price_volume_data["high"]
        low = price_volume_data["low"]
        close = price_volume_data["close"]
        volume = price_volume_data["volume"]

        # TA-Lib reference
        expected = talib.AD(high, low, close, volume)

        # Our implementation
        result = ad(high, low, close, volume)

        # Should match exactly
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    def test_ad_polars(self, price_volume_data):
        """Test AD with Polars expressions."""
        df = pl.DataFrame(price_volume_data)

        # Using Polars expression
        result_df = df.with_columns(
            ad("high", "low", "close", "volume").alias("ad"),
        )

        # Compare with TA-Lib
        expected = talib.AD(
            price_volume_data["high"],
            price_volume_data["low"],
            price_volume_data["close"],
            price_volume_data["volume"],
        )

        assert_allclose(
            result_df["ad"].to_numpy(),
            expected,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_ad_edge_cases(self):
        """Test AD with edge cases."""
        # Case 1: High == Low (avoid division by zero)
        high = np.array([100.0, 100.0, 100.0])
        low = np.array([100.0, 100.0, 100.0])
        close = np.array([100.0, 100.0, 100.0])
        volume = np.array([1000.0, 2000.0, 1500.0])

        result = ad(high, low, close, volume)
        # When high == low, money flow multiplier should be 0
        expected = np.array([0.0, 0.0, 0.0])
        assert_allclose(result, expected, rtol=1e-10)

        # Case 2: Zero volume
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([0.0, 0.0, 0.0])

        result = ad(high, low, close, volume)
        expected = np.array([0.0, 0.0, 0.0])
        assert_allclose(result, expected, rtol=1e-10)

        # Case 3: With NaN values
        high = np.array([101.0, np.nan, 103.0, 104.0])
        low = np.array([99.0, 100.0, 101.0, 102.0])
        close = np.array([100.0, 101.0, 102.0, 103.0])
        volume = np.array([1000.0, 2000.0, 1500.0, 1800.0])

        result = ad(high, low, close, volume)
        # Should propagate NaN forward
        assert ~np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert np.isnan(result[3])

    def test_ad_calculation_manual(self):
        """Test AD calculation manually."""
        high = np.array([102.0, 103.0, 104.0])
        low = np.array([98.0, 99.0, 100.0])
        close = np.array([101.0, 100.0, 103.0])
        volume = np.array([1000.0, 2000.0, 1500.0])

        result = ad(high, low, close, volume)

        # Manual calculation
        # Day 1: MFM = ((101-98) - (102-101)) / (102-98) = (3-1)/4 = 0.5
        #        MFV = 0.5 * 1000 = 500
        #        AD = 500
        # Day 2: MFM = ((100-99) - (103-100)) / (103-99) = (1-3)/4 = -0.5
        #        MFV = -0.5 * 2000 = -1000
        #        AD = 500 + (-1000) = -500
        # Day 3: MFM = ((103-100) - (104-103)) / (104-100) = (3-1)/4 = 0.5
        #        MFV = 0.5 * 1500 = 750
        #        AD = -500 + 750 = 250

        expected = np.array([500.0, -500.0, 250.0])
        assert_allclose(result, expected, rtol=1e-10)

    def test_ad_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([101.0, 102.0])
        low = np.array([99.0])  # Wrong length
        close = np.array([100.0, 101.0])
        volume = np.array([1000.0, 2000.0])

        with pytest.raises(ValueError, match="must have the same length"):
            ad(high, low, close, volume)

    def test_ad_crypto_accuracy(self, crypto_data_small):
        """Test AD accuracy on real crypto data."""
        # Extract columns from 10K rows of real crypto data
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()
        volume = crypto_data_small["volume"].to_numpy()

        # TA-Lib reference
        expected = talib.AD(high, low, close, volume)

        # Our implementation
        result = ad(high, low, close, volume)

        # Should match closely on real data
        assert_allclose(result, expected, rtol=1e-10, equal_nan=True)

    @pytest.mark.benchmark
    def test_ad_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark AD performance using real crypto data."""
        # Use real crypto data
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()
        close = crypto_data["close"].to_numpy()
        volume = crypto_data["volume"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(ad, high[:100], low[:100], close[:100], volume[:100])

        import time

        # Our implementation (warmed up)
        start = time.perf_counter()
        _ = ad(high, low, close, volume)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.AD(high, low, close, volume)
        talib_time = time.perf_counter() - start

        print(f"\nAD Performance ({len(high):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use dynamic threshold based on complexity (rolling calculation)
        threshold = performance_threshold("rolling")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
