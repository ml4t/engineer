"""
Test Directional Movement indicators: PLUS_DM and MINUS_DM.

These indicators measure directional price movements.
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

from ml4t.engineer.features.momentum import minus_dm, plus_dm


class TestDMIndicators:
    """Test Directional Movement indicators."""

    @pytest.fixture
    def price_data(self):
        """Generate sample OHLC data with clear trends."""
        np.random.seed(42)
        n = 100

        # Create trending data
        trend = np.linspace(0, 20, n)
        noise = np.random.randn(n) * 0.5

        # Base price with trend
        base = 100 + trend + noise

        # Create OHLC
        high = base + np.abs(np.random.randn(n) * 0.8)
        low = base - np.abs(np.random.randn(n) * 0.8)
        close = (high + low) / 2 + np.random.randn(n) * 0.2

        return {"high": high, "low": low, "close": close}

    @pytest.mark.parametrize("period", [7, 14, 21, 30])
    def test_plus_dm_accuracy(self, price_data, period):
        """Test PLUS_DM matches TA-Lib exactly with various periods."""
        high = price_data["high"]
        low = price_data["low"]

        # TA-Lib reference
        expected = talib.PLUS_DM(high, low, timeperiod=period)

        # Our implementation
        result = plus_dm(high, low, timeperiod=period)

        # DM indicators may have some differences due to Wilder's smoothing
        # Ultra-high precision - DM indicators achieve machine-level accuracy
        assert_indicator_match(
            result,
            expected,
            f"PLUS_DM (period={period})",
            rtol=1e-10,
        )

    @pytest.mark.parametrize("period", [7, 14, 21, 30])
    def test_minus_dm_accuracy(self, price_data, period):
        """Test MINUS_DM matches TA-Lib exactly with various periods."""
        high = price_data["high"]
        low = price_data["low"]

        # TA-Lib reference
        expected = talib.MINUS_DM(high, low, timeperiod=period)

        # Our implementation
        result = minus_dm(high, low, timeperiod=period)

        # DM indicators may have some differences due to Wilder's smoothing
        # Ultra-high precision - DM indicators achieve machine-level accuracy
        assert_indicator_match(
            result,
            expected,
            f"MINUS_DM (period={period})",
            rtol=1e-10,
        )

    def test_dm_polars(self, price_data):
        """Test DM indicators with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expressions
        result_df = df.with_columns(
            [
                plus_dm("high", "low", 14).alias("plus_dm"),
                minus_dm("high", "low", 14).alias("minus_dm"),
            ],
        )

        # Compare with TA-Lib
        expected_plus = talib.PLUS_DM(
            price_data["high"],
            price_data["low"],
            timeperiod=14,
        )
        expected_minus = talib.MINUS_DM(
            price_data["high"],
            price_data["low"],
            timeperiod=14,
        )

        assert_indicator_match(
            result_df["plus_dm"].to_numpy(),
            expected_plus,
            "PLUS_DM (Polars)",
            rtol=1e-10,
        )
        assert_indicator_match(
            result_df["minus_dm"].to_numpy(),
            expected_minus,
            "MINUS_DM (Polars)",
            rtol=1e-10,
        )

    def test_dm_relationships(self, price_data):
        """Test relationships between DM indicators."""
        high = price_data["high"]
        low = price_data["low"]

        plus_dm_result = plus_dm(high, low, timeperiod=14)
        minus_dm_result = minus_dm(high, low, timeperiod=14)

        # DM values should be non-negative
        assert np.all(plus_dm_result[~np.isnan(plus_dm_result)] >= 0)
        assert np.all(minus_dm_result[~np.isnan(minus_dm_result)] >= 0)

        # In a strong uptrend, PLUS_DM should dominate
        # Create strong uptrend data
        n = 50
        up_trend = np.linspace(100, 150, n)
        high_up = up_trend + np.abs(np.random.randn(n) * 0.2)
        low_up = up_trend - np.abs(np.random.randn(n) * 0.2)

        plus_up = plus_dm(high_up, low_up, timeperiod=14)
        minus_up = minus_dm(high_up, low_up, timeperiod=14)

        # Average PLUS_DM should be higher in uptrend
        avg_plus = np.nanmean(plus_up)
        avg_minus = np.nanmean(minus_up)
        assert avg_plus > avg_minus

    def test_dm_edge_cases(self):
        """Test DM indicators with edge cases."""
        # Case 1: Minimal data
        high = np.array([10.0, 11.0, 10.5, 12.0, 11.5])
        low = np.array([9.0, 9.5, 9.2, 10.0, 10.5])

        plus_dm_result = plus_dm(high, low, timeperiod=3)
        minus_dm_result = minus_dm(high, low, timeperiod=3)

        # First timeperiod-1 values should be NaN
        assert np.all(np.isnan(plus_dm_result[:2]))
        assert np.all(np.isnan(minus_dm_result[:2]))

        # Case 2: No directional movement (sideways)
        high_flat = np.array([100.5, 100.5, 100.5, 100.5, 100.5])
        low_flat = np.array([99.5, 99.5, 99.5, 99.5, 99.5])

        plus_flat = plus_dm(high_flat, low_flat, timeperiod=3)
        minus_flat = minus_dm(high_flat, low_flat, timeperiod=3)

        # Should converge to zero (after initial values)
        assert np.allclose(plus_flat[3:], 0, atol=1e-10)
        assert np.allclose(minus_flat[3:], 0, atol=1e-10)

    def test_dm_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 10.0, 11.0])

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            plus_dm(high, low, timeperiod=0)

        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            minus_dm(high, low, timeperiod=0)

        # Mismatched lengths
        with pytest.raises(ValueError, match="high and low must have the same length"):
            plus_dm(high, low[:-1], timeperiod=14)

    @pytest.mark.parametrize("period", [5, 10, 14, 21])
    def test_dm_crypto_accuracy(self, crypto_data_small, period):
        """Test DM indicators accuracy on real crypto data."""
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()

        # PLUS_DM
        expected_plus = talib.PLUS_DM(high, low, timeperiod=period)
        result_plus = plus_dm(high, low, timeperiod=period)
        assert_indicator_match(
            result_plus,
            expected_plus,
            f"PLUS_DM (crypto, period={period})",
            rtol=1e-10,
        )

        # MINUS_DM
        expected_minus = talib.MINUS_DM(high, low, timeperiod=period)
        result_minus = minus_dm(high, low, timeperiod=period)
        assert_indicator_match(
            result_minus,
            expected_minus,
            f"MINUS_DM (crypto, period={period})",
            rtol=1e-10,
        )

    def test_dm_calculation_details(self):
        """Test specific DM calculation scenarios."""
        # Scenario 1: Clear upward movement
        high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 11.0, 12.0, 13.0])

        # Raw directional movements (before smoothing)
        # +DM[1] = 11-10 = 1, -DM[1] = 0 (low[1] > low[0])
        # +DM[2] = 12-11 = 1, -DM[2] = 0
        # etc.

        plus_dm_result = plus_dm(high, low, timeperiod=2)
        minus_dm_result = minus_dm(high, low, timeperiod=2)

        # Verify TA-Lib match
        expected_plus = talib.PLUS_DM(high, low, timeperiod=2)
        expected_minus = talib.MINUS_DM(high, low, timeperiod=2)

        assert_indicator_match(plus_dm_result, expected_plus, "PLUS_DM calc", rtol=1e-10)
        assert_indicator_match(minus_dm_result, expected_minus, "MINUS_DM calc", rtol=1e-10)

    @pytest.mark.benchmark
    def test_dm_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark DM indicators performance."""
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()

        # Warmup JIT compilation
        warmup_jit(plus_dm, high[:100], low[:100], 14)
        warmup_jit(minus_dm, high[:100], low[:100], 14)

        import time

        # Test PLUS_DM
        start = time.perf_counter()
        _ = plus_dm(high, low, timeperiod=14)
        our_plus_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = talib.PLUS_DM(high, low, timeperiod=14)
        talib_plus_time = time.perf_counter() - start

        print(f"\nPLUS_DM Performance ({len(high):,} rows):")
        print(f"Our implementation: {our_plus_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_plus_time * 1000:.2f}ms")
        print(f"Ratio: {our_plus_time / talib_plus_time:.2f}x")

        # Test MINUS_DM
        start = time.perf_counter()
        _ = minus_dm(high, low, timeperiod=14)
        our_minus_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = talib.MINUS_DM(high, low, timeperiod=14)
        talib_minus_time = time.perf_counter() - start

        print(f"\nMINUS_DM Performance ({len(high):,} rows):")
        print(f"Our implementation: {our_minus_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_minus_time * 1000:.2f}ms")
        print(f"Ratio: {our_minus_time / talib_minus_time:.2f}x")

        # Use standard threshold
        threshold = performance_threshold("standard")
        assert our_plus_time < talib_plus_time * threshold, (
            f"PLUS_DM performance ratio {our_plus_time / talib_plus_time:.1f}x exceeds threshold {threshold}x"
        )
        assert our_minus_time < talib_minus_time * threshold, (
            f"MINUS_DM performance ratio {our_minus_time / talib_minus_time:.1f}x exceeds threshold {threshold}x"
        )
