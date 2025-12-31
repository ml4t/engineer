"""
Test Chaikin A/D Oscillator (ADOSC) indicator implementation.

ADOSC = EMA(AD, fastperiod) - EMA(AD, slowperiod)

Where AD is the Chaikin A/D Line.
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

from ml4t.engineer.features.volume import adosc


class TestADOSC:
    """Test Chaikin A/D Oscillator indicator."""

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

    def test_adosc_accuracy(self, price_volume_data):
        """Test ADOSC matches TA-Lib exactly."""
        high = price_volume_data["high"]
        low = price_volume_data["low"]
        close = price_volume_data["close"]
        volume = price_volume_data["volume"]

        # TA-Lib reference
        expected = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

        # Our implementation
        result = adosc(high, low, close, volume, fastperiod=3, slowperiod=10)

        # Should match exactly - our implementation is precise
        assert_indicator_match(result, expected, "ADOSC")

    def test_adosc_polars(self, price_volume_data):
        """Test ADOSC with Polars expressions."""
        df = pl.DataFrame(price_volume_data)

        # Using Polars expression
        result_df = df.with_columns(
            adosc("high", "low", "close", "volume", 3, 10).alias("adosc"),
        )

        # Compare with TA-Lib
        expected = talib.ADOSC(
            price_volume_data["high"],
            price_volume_data["low"],
            price_volume_data["close"],
            price_volume_data["volume"],
            fastperiod=3,
            slowperiod=10,
        )

        assert_indicator_match(
            result_df["adosc"].to_numpy(),
            expected,
            "ADOSC (Polars)",
        )

    def test_adosc_edge_cases(self):
        """Test ADOSC with edge cases."""
        # Case 1: Minimal data
        high = np.array([101.0, 102.0, 103.0, 104.0, 105.0])
        low = np.array([99.0, 100.0, 101.0, 102.0, 103.0])
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        volume = np.array([1000.0, 2000.0, 1500.0, 1800.0, 2200.0])

        result = adosc(high, low, close, volume, fastperiod=2, slowperiod=3)
        # First slowperiod-1 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])  # From index 2 we should have values
        assert not np.isnan(result[3])

        # Case 2: High == Low
        high = np.full(20, 100.0)
        low = np.full(20, 100.0)
        close = np.full(20, 100.0)
        volume = np.random.rand(20) * 1000

        result = adosc(high, low, close, volume)
        # When high == low, AD should be 0, so ADOSC should be 0 after initial NaN
        talib_result = talib.ADOSC(high, low, close, volume)
        assert_indicator_match(result, talib_result, "ADOSC (high==low)")

    def test_adosc_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([101.0, 102.0])
        low = np.array([99.0])  # Wrong length
        close = np.array([100.0, 101.0])
        volume = np.array([1000.0, 2000.0])

        with pytest.raises(ValueError, match="must have the same length"):
            adosc(high, low, close, volume)

        # Invalid periods
        high = np.random.randn(50) + 100
        low = high - 2
        close = (high + low) / 2
        volume = np.random.rand(50) * 1000

        with pytest.raises(ValueError, match="fastperiod must be > 0"):
            adosc(high, low, close, volume, fastperiod=0)

        with pytest.raises(ValueError, match="slowperiod must be > 0"):
            adosc(high, low, close, volume, slowperiod=0)

    def test_adosc_different_periods(self):
        """Test ADOSC with various period combinations."""
        np.random.seed(42)
        n = 100
        high = np.random.randn(n) + 100 + np.abs(np.random.randn(n))
        low = high - np.abs(np.random.randn(n) * 2)
        close = (high + low) / 2 + np.random.randn(n) * 0.5
        volume = np.abs(np.random.randn(n) * 1000000)

        # Test different period combinations
        test_params = [
            (3, 10),  # Default
            (2, 5),  # Minimum values
            (12, 26),  # MACD-like periods
            (5, 35),  # Larger spread
        ]

        for fast, slow in test_params:
            result = adosc(high, low, close, volume, fast, slow)
            expected = talib.ADOSC(high, low, close, volume, fast, slow)
            assert_indicator_match(
                result,
                expected,
                f"ADOSC (fast={fast}, slow={slow})",
            )

    def test_adosc_crypto_accuracy(self, crypto_data_small):
        """Test ADOSC accuracy on real crypto data."""
        # Extract columns from 10K rows of real crypto data
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()
        volume = crypto_data_small["volume"].to_numpy()

        # Test with different parameter combinations
        test_params = [
            (3, 10),  # Default
            (12, 26),  # MACD-like periods
            (5, 35),  # Larger spread
        ]

        for fast, slow in test_params:
            # TA-Lib reference
            expected = talib.ADOSC(
                high,
                low,
                close,
                volume,
                fastperiod=fast,
                slowperiod=slow,
            )

            # Our implementation
            result = adosc(
                high,
                low,
                close,
                volume,
                fastperiod=fast,
                slowperiod=slow,
            )

            # Should match exactly on real data too
            assert_indicator_match(
                result,
                expected,
                f"ADOSC (crypto, fast={fast}, slow={slow})",
            )

    @pytest.mark.benchmark
    def test_adosc_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark ADOSC performance using real crypto data."""
        # Use real crypto data
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()
        close = crypto_data["close"].to_numpy()
        volume = crypto_data["volume"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(adosc, high[:100], low[:100], close[:100], volume[:100])

        import time

        # Our implementation (warmed up)
        start = time.perf_counter()
        _ = adosc(high, low, close, volume)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.ADOSC(high, low, close, volume)
        talib_time = time.perf_counter() - start

        print(f"\nADOSC Performance ({len(high):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use dynamic threshold based on complexity (very complex - multiple EMAs)
        threshold = performance_threshold("very_complex")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
