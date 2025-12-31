"""
Test ADXR (Average Directional Movement Index Rating) indicator.

ADXR is a smoothed version of ADX, calculated as the average of current ADX
and ADX from n periods ago.
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

from ml4t.engineer.features.momentum import adx, adxr


class TestADXR:
    """Test ADXR indicator."""

    @pytest.fixture
    def price_data(self):
        """Generate sample OHLC data."""
        np.random.seed(42)
        n = 200  # Need more data for ADXR

        # Generate trending price data
        trend = np.linspace(0, 10, n)
        noise = np.random.randn(n) * 0.5
        base = 100 + trend + noise

        # Create OHLC data
        high = base + np.abs(np.random.randn(n) * 0.5)
        low = base - np.abs(np.random.randn(n) * 0.5)
        close = (high + low) / 2 + np.random.randn(n) * 0.1

        return {"high": high, "low": low, "close": close}

    def test_adxr_accuracy_default(self, price_data):
        """Test ADXR matches TA-Lib with default parameters."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.ADXR(high, low, close, timeperiod=14)

        # Our implementation
        result = adxr(high, low, close, timeperiod=14)

        # Should match exactly
        assert_indicator_match(result, expected, "ADXR", rtol=1e-6)

    def test_adxr_custom_periods(self, price_data):
        """Test ADXR with custom periods."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # Test different periods
        for period in [7, 14, 21, 30]:
            # TA-Lib reference
            expected = talib.ADXR(high, low, close, timeperiod=period)

            # Our implementation
            result = adxr(high, low, close, timeperiod=period)

            assert_indicator_match(
                result,
                expected,
                f"ADXR (period={period})",
                rtol=1e-6,
            )

    def test_adxr_polars(self, price_data):
        """Test ADXR with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expressions
        result_df = df.with_columns(adxr("high", "low", "close", 14).alias("adxr"))

        # Compare with TA-Lib
        expected = talib.ADXR(
            price_data["high"],
            price_data["low"],
            price_data["close"],
            timeperiod=14,
        )

        assert_indicator_match(
            result_df["adxr"].to_numpy(),
            expected,
            "ADXR (Polars)",
            rtol=1e-6,
        )

    def test_adxr_relationship_to_adx(self, price_data):
        """Test ADXR relationship to ADX."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]
        period = 14

        # Calculate ADX and ADXR
        adx_result = adx(high, low, close, period=period)
        adxr_result = adxr(high, low, close, timeperiod=period)

        # ADXR should start producing values period bars after ADX
        first_adx = np.where(~np.isnan(adx_result))[0][0]
        first_adxr = np.where(~np.isnan(adxr_result))[0][0]
        assert first_adxr >= first_adx + period - 1

        # ADXR should be smoother than ADX (lower standard deviation)
        valid_idx = ~np.isnan(adxr_result)
        if np.sum(valid_idx) > 10:
            adx_std = np.std(adx_result[valid_idx])
            adxr_std = np.std(adxr_result[valid_idx])
            assert adxr_std <= adx_std, "ADXR should be smoother than ADX"

    def test_adxr_edge_cases(self):
        """Test ADXR with edge cases."""
        # Case 1: Minimal data for period=3
        high = np.array([10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0])
        low = np.array([9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])
        close = np.array([9.5, 10.5, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0])

        result = adxr(high, low, close, timeperiod=3)

        # ADXR needs at least 2*period + some warmup for ADX
        # First several values should be NaN
        assert np.sum(np.isnan(result)) >= 6

        # Case 2: Flat market (no directional movement)
        n = 50
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        close = np.full(n, 100.0)

        result = adxr(high, low, close, timeperiod=14)

        # In a flat market, ADX and ADXR should converge to low values
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.all(valid_values <= 20), "ADXR should be low in flat market"

    def test_adxr_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])
        close = np.array([9.5, 10.5, 11.5])

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            adxr(high, low, close, timeperiod=0)

    def test_adxr_crypto_accuracy(self, crypto_data_small):
        """Test ADXR accuracy on real crypto data."""
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        # Test with different periods
        for period in [7, 14, 21]:
            # TA-Lib reference
            expected = talib.ADXR(high, low, close, timeperiod=period)

            # Our implementation
            result = adxr(high, low, close, timeperiod=period)

            # Should match exactly even on crypto data
            assert_indicator_match(
                result,
                expected,
                f"ADXR (crypto, period={period})",
                rtol=1e-6,
            )

    def test_adxr_trending_vs_ranging(self):
        """Test ADXR behavior in trending vs ranging markets."""
        n = 100

        # Create strong trending market
        trend = np.linspace(100, 150, n)
        high_trend = trend + np.random.randn(n) * 0.5
        low_trend = trend - np.random.randn(n) * 0.5
        close_trend = (high_trend + low_trend) / 2

        # Create ranging market
        range_base = 100 + np.sin(np.linspace(0, 4 * np.pi, n)) * 5
        high_range = range_base + np.random.randn(n) * 0.5
        low_range = range_base - np.random.randn(n) * 0.5
        close_range = (high_range + low_range) / 2

        # Calculate ADXR for both
        adxr_trend = adxr(high_trend, low_trend, close_trend, timeperiod=14)
        adxr_range = adxr(high_range, low_range, close_range, timeperiod=14)

        # Average ADXR should be higher in trending market
        avg_trend = np.nanmean(adxr_trend)
        avg_range = np.nanmean(adxr_range)

        assert avg_trend > avg_range, "ADXR should be higher in trending markets"

    @pytest.mark.benchmark
    def test_adxr_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark ADXR performance."""
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation
        warmup_jit(adxr, high[:200], low[:200], close[:200], 14)

        import time

        # Our implementation
        start = time.perf_counter()
        _ = adxr(high, low, close, timeperiod=14)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.ADXR(high, low, close, timeperiod=14)
        talib_time = time.perf_counter() - start

        print(f"\nADXR Performance ({len(high):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use complex threshold for Wilder's smoothing indicators
        threshold = performance_threshold("complex")
        assert our_time < talib_time * threshold, (
            f"ADXR performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
