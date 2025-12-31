"""
Test STOCHF (Fast Stochastic) indicator.

STOCHF calculates the fast stochastic oscillator without additional smoothing.
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

from ml4t.engineer.features.momentum import stochf


class TestSTOCHF:
    """Test Fast Stochastic indicator."""

    @pytest.fixture
    def price_data(self):
        """Generate sample OHLC data."""
        np.random.seed(42)
        n = 100

        # Generate realistic OHLC data
        open_prices = 100 + np.random.randn(n).cumsum()
        close_prices = open_prices + np.random.randn(n) * 0.5
        high = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n) * 0.3)
        low = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n) * 0.3)

        return {"high": high, "low": low, "close": close_prices}

    def test_stochf_accuracy_default(self, price_data):
        """Test STOCHF matches TA-Lib with default parameters."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference - returns (fastk, fastd)
        expected_k, expected_d = talib.STOCHF(high, low, close)

        # Our implementation - single value by default
        result = stochf(high, low, close)

        # Should match %K exactly
        assert_indicator_match(result, expected_k, "STOCHF %K", rtol=1e-10)

    def test_stochf_accuracy_pair(self, price_data):
        """Test STOCHF matches TA-Lib when returning pair."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected_k, expected_d = talib.STOCHF(high, low, close)

        # Our implementation - return pair
        result_k, result_d = stochf(high, low, close, return_pair=True)

        # Should match exactly
        assert_indicator_match(result_k, expected_k, "STOCHF %K (pair)", rtol=1e-10)
        assert_indicator_match(result_d, expected_d, "STOCHF %D (pair)", rtol=1e-10)

    def test_stochf_custom_periods(self, price_data):
        """Test STOCHF with custom periods."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # Test different period combinations
        test_cases = [
            (3, 3),  # Very fast
            (14, 3),  # Standard
            (21, 5),  # Slower
        ]

        for k_period, d_period in test_cases:
            # TA-Lib reference
            expected_k, expected_d = talib.STOCHF(
                high,
                low,
                close,
                fastk_period=k_period,
                fastd_period=d_period,
            )

            # Our implementation
            result_k, result_d = stochf(
                high,
                low,
                close,
                fastk_period=k_period,
                fastd_period=d_period,
                return_pair=True,
            )

            assert_indicator_match(
                result_k,
                expected_k,
                f"STOCHF %K (k={k_period}, d={d_period})",
                rtol=1e-10,
            )
            assert_indicator_match(
                result_d,
                expected_d,
                f"STOCHF %D (k={k_period}, d={d_period})",
                rtol=1e-10,
            )

    def test_stochf_polars(self, price_data):
        """Test STOCHF with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expressions - single value
        result_df = df.with_columns(
            stochf("high", "low", "close").alias("stochf_k"),
        )

        # Compare with TA-Lib
        expected_k, _ = talib.STOCHF(
            price_data["high"],
            price_data["low"],
            price_data["close"],
        )

        assert_indicator_match(
            result_df["stochf_k"].to_numpy(),
            expected_k,
            "STOCHF %K (Polars)",
            rtol=1e-10,
        )

    def test_stochf_edge_cases(self):
        """Test STOCHF with edge cases."""
        # Case 1: Minimal data
        high = np.array([10.0, 15.0, 12.0, 18.0, 16.0])
        low = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
        close = np.array([9.0, 14.0, 11.0, 17.0, 15.0])

        result = stochf(high, low, close, fastk_period=3)

        # First 4 values should be NaN (fastk=3, fastd=3 default, so starts at 3+3-2=4)
        assert np.all(np.isnan(result[:4]))

        # Check specific calculation for index 4
        # With fastk=3, fastd=3, output starts at index 4
        # For index 4, we use the intermediate %K values from indices 2,3,4
        # This matches TA-Lib behavior
        expected = talib.STOCHF(high, low, close, 3, 3)[0]
        assert_allclose(result, expected, rtol=1e-10)

        # Case 2: All prices the same (flat market)
        flat_price = 100.0
        high = np.full(10, flat_price)
        low = np.full(10, flat_price)
        close = np.full(10, flat_price)

        result = stochf(high, low, close, fastk_period=5)

        # When high = low, the indicator is undefined (0/0)
        # With fastk=5, fastd=3, output starts at index 5+3-2=6
        # TA-Lib returns 0 in this case for flat markets
        assert np.all(np.isnan(result[:6]))  # Indices 0-5 should be NaN
        assert np.all(result[6:] == 0.0)  # Indices 6-9 should be 0.0

    def test_stochf_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([10.0, 15.0, 12.0])
        low = np.array([8.0, 9.0, 10.0])
        close = np.array([9.0, 14.0, 11.0])

        # Invalid k period
        with pytest.raises(ValueError, match="fastk_period must be > 0"):
            stochf(high, low, close, fastk_period=0)

        # Invalid d period
        with pytest.raises(ValueError, match="fastd_period must be > 0"):
            stochf(high, low, close, fastd_period=0, return_pair=True)

    def test_stochf_crypto_accuracy(self, crypto_data_small):
        """Test STOCHF accuracy on real crypto data."""
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        # Test with different periods
        for k_period in [5, 14, 21]:
            # TA-Lib reference
            expected_k, expected_d = talib.STOCHF(
                high,
                low,
                close,
                fastk_period=k_period,
            )

            # Our implementation
            result_k = stochf(high, low, close, fastk_period=k_period)
            result_k_pair, result_d_pair = stochf(
                high,
                low,
                close,
                fastk_period=k_period,
                return_pair=True,
            )

            assert_indicator_match(
                result_k,
                expected_k,
                f"STOCHF %K (crypto, k={k_period})",
                rtol=1e-10,
            )
            assert_indicator_match(
                result_k_pair,
                expected_k,
                f"STOCHF %K pair (crypto, k={k_period})",
                rtol=1e-10,
            )
            assert_indicator_match(
                result_d_pair,
                expected_d,
                f"STOCHF %D pair (crypto, k={k_period})",
                rtol=1e-10,
            )

    def test_stochf_comparison_with_stoch(self, price_data):
        """Test that STOCHF is faster version of STOCH."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # STOCHF with d_period=1 should be same as raw %K
        stochf_k = stochf(high, low, close, fastd_period=1)

        # Regular STOCH with slowk=1, slowd=1 should match STOCHF
        stoch_k, _ = talib.STOCH(high, low, close, slowk_period=1, slowd_period=1)

        # STOCHF %K should match
        assert_indicator_match(stochf_k, stoch_k, "STOCHF vs STOCH", rtol=1e-10)

    @pytest.mark.benchmark
    def test_stochf_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark STOCHF performance."""
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation
        warmup_jit(stochf, high[:100], low[:100], close[:100])

        import time

        # Test single output
        start = time.perf_counter()
        _ = stochf(high, low, close)
        our_time = time.perf_counter() - start

        start = time.perf_counter()
        _, _ = talib.STOCHF(high, low, close)
        talib_time = time.perf_counter() - start

        print(f"\nSTOCHF Performance ({len(high):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use standard threshold for momentum indicators
        threshold = performance_threshold("standard")
        assert our_time < talib_time * threshold, (
            f"STOCHF performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
