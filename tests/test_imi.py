"""
Test IMI (Intraday Momentum Index) indicator.

IMI is similar to RSI but uses the relationship between open and close prices
rather than consecutive closes.
"""

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from .test_config import assert_indicator_match

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    pytest.skip("TA-Lib not available", allow_module_level=True)

from ml4t.engineer.features.momentum import imi


class TestIMI:
    """Test Intraday Momentum Index indicator."""

    @pytest.fixture
    def ohlc_data(self):
        """Generate sample OHLC data."""
        np.random.seed(42)
        n = 100

        # Generate realistic price movements
        returns = np.random.normal(0.0001, 0.02, n)
        close = 100 * (1 + returns).cumprod()

        # Create OHLC data with realistic relationships
        # Open is typically close to previous close
        open_prices = np.concatenate([[100], close[:-1]]) + np.random.randn(n) * 0.1

        # High/Low based on open/close range
        high = np.maximum(open_prices, close) + np.abs(np.random.randn(n) * 0.3)
        low = np.minimum(open_prices, close) - np.abs(np.random.randn(n) * 0.3)

        return {"open": open_prices, "high": high, "low": low, "close": close}

    @pytest.mark.parametrize("period", [7, 14, 21, 28])
    def test_imi_accuracy(self, ohlc_data, period):
        """Test IMI matches TA-Lib exactly with various periods."""
        open_prices = ohlc_data["open"]
        close = ohlc_data["close"]

        # TA-Lib reference
        expected = talib.IMI(open_prices, close, timeperiod=period)

        # Our implementation
        result = imi(open_prices, close, timeperiod=period)

        # Should match exactly
        assert_indicator_match(result, expected, f"IMI (period={period})", rtol=1e-6)

    def test_imi_polars(self, ohlc_data):
        """Test IMI with Polars expressions."""
        df = pl.DataFrame(ohlc_data)

        # Using Polars expressions
        result_df = df.with_columns(imi("open", "close", 14).alias("imi"))

        # Compare with TA-Lib
        expected = talib.IMI(ohlc_data["open"], ohlc_data["close"], timeperiod=14)

        assert_indicator_match(
            result_df["imi"].to_numpy(),
            expected,
            "IMI (Polars)",
            rtol=1e-6,
        )

    def test_imi_bounds(self, ohlc_data):
        """Test IMI stays within expected bounds."""
        open_prices = ohlc_data["open"]
        close = ohlc_data["close"]

        imi_result = imi(open_prices, close, timeperiod=14)

        # IMI should be between 0 and 100
        valid_values = imi_result[~np.isnan(imi_result)]
        assert np.all(valid_values >= 0), "IMI should be >= 0"
        assert np.all(valid_values <= 100), "IMI should be <= 100"

    def test_imi_vs_rsi_concept(self):
        """Test IMI concept vs RSI - different but related."""
        # Create data with clear up/down days
        n = 50
        open_prices = np.full(n, 100.0)
        close = np.zeros(n)

        # First 25 days: closing higher than open (bullish)
        close[:25] = open_prices[:25] + 1.0

        # Next 25 days: closing lower than open (bearish)
        close[25:] = open_prices[25:] - 1.0

        imi_result = imi(open_prices, close, timeperiod=14)

        # During bullish period, IMI should be high
        bullish_imi = np.nanmean(imi_result[14:25])
        assert bullish_imi > 70, f"Bullish IMI should be high, got {bullish_imi}"

        # During bearish period, IMI should be low
        bearish_imi = np.nanmean(imi_result[35:])
        assert bearish_imi < 30, f"Bearish IMI should be low, got {bearish_imi}"

    def test_imi_edge_cases(self):
        """Test IMI with edge cases."""
        # Case 1: All gains (close > open)
        open_all_gains = np.array([100.0] * 20)
        close_all_gains = np.array([101.0] * 20)

        imi_all_gains = imi(open_all_gains, close_all_gains, timeperiod=14)
        # Should be 100 after warmup
        assert_allclose(imi_all_gains[13:], 100.0, rtol=1e-10)

        # Case 2: All losses (close < open)
        open_all_losses = np.array([100.0] * 20)
        close_all_losses = np.array([99.0] * 20)

        imi_all_losses = imi(open_all_losses, close_all_losses, timeperiod=14)
        # Should be 0 after warmup
        assert_allclose(imi_all_losses[13:], 0.0, rtol=1e-10)

        # Case 3: No change (close = open)
        open_no_change = np.array([100.0] * 20)
        close_no_change = np.array([100.0] * 20)

        imi_no_change = imi(open_no_change, close_no_change, timeperiod=14)
        # When there's no movement, IMI is undefined (NaN)
        assert np.all(np.isnan(imi_no_change[13:]))

    def test_imi_parameter_validation(self):
        """Test parameter validation."""
        open_prices = np.array([100.0, 101.0, 102.0])
        close = np.array([101.0, 100.0, 103.0])

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            imi(open_prices, close, timeperiod=0)

        # Mismatched lengths
        with pytest.raises(
            ValueError,
            match="open and close must have the same length",
        ):
            imi(open_prices, close[:-1], timeperiod=14)

    def test_imi_crypto_accuracy(self, crypto_data_small):
        """Test IMI accuracy on real crypto data."""
        # Need to ensure we have open prices
        open_prices = crypto_data_small["open"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        # Test with different periods
        for period in [7, 14, 21, 28]:
            # TA-Lib reference
            expected = talib.IMI(open_prices, close, timeperiod=period)

            # Our implementation
            result = imi(open_prices, close, timeperiod=period)

            assert_indicator_match(
                result,
                expected,
                f"IMI (crypto, period={period})",
                rtol=1e-6,
            )

    def test_imi_calculation_details(self):
        """Test specific IMI calculation scenarios."""
        # Simple test case
        open_prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        close = np.array([101.0, 99.0, 102.0, 98.0, 101.0])

        # Manual calculation for period=3
        # Gains: 1, 0, 2, 0, 1
        # Losses: 0, 1, 0, 2, 0
        # For index 2 (using indices 0,1,2):
        # Sum of gains = 1 + 0 + 2 = 3
        # Sum of losses = 0 + 1 + 0 = 1
        # IMI = 100 * 3 / (3 + 1) = 75

        imi_result = imi(open_prices, close, timeperiod=3)
        expected = talib.IMI(open_prices, close, timeperiod=3)

        assert_indicator_match(imi_result, expected, "IMI calculation", rtol=1e-6)

    @pytest.mark.benchmark
    def test_imi_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark IMI performance."""
        open_prices = crypto_data["open"].to_numpy()
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation
        warmup_jit(imi, open_prices[:100], close[:100], 14)

        import time

        # Our implementation
        start = time.perf_counter()
        _ = imi(open_prices, close, timeperiod=14)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.IMI(open_prices, close, timeperiod=14)
        talib_time = time.perf_counter() - start

        print(f"\nIMI Performance ({len(close):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use standard threshold
        threshold = performance_threshold("standard")
        assert our_time < talib_time * threshold, (
            f"IMI performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
