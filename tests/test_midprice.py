"""
Test MIDPRICE (Midpoint Price over period) indicator.
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

from ml4t.engineer.features.price_transform import medprice, midprice
from ml4t.engineer.features.trend import midpoint


class TestMIDPRICE:
    """Test MIDPRICE indicator."""

    @pytest.fixture
    def ohlc_data(self):
        """Generate simple OHLC test data."""
        return {
            "high": np.array([10, 12, 11, 13, 14, 12, 11, 15, 13, 12], dtype=float),
            "low": np.array([9, 10, 10, 11, 12, 10, 9, 13, 11, 10], dtype=float),
        }

    @pytest.fixture
    def trending_data(self):
        """Generate trending OHLC data."""
        n = 50
        trend = np.linspace(100, 150, n)
        noise = np.random.RandomState(42).randn(n) * 2
        high = trend + np.abs(noise) + 1
        low = trend - np.abs(noise) - 1
        return {"high": high, "low": low}

    @pytest.fixture
    def random_ohlc_data(self):
        """Generate random OHLC data."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5) + 0.1
        low = close - np.abs(np.random.randn(n) * 0.5) - 0.1
        return {"high": high, "low": low}

    def test_midprice_accuracy(self, ohlc_data, trending_data, random_ohlc_data):
        """Test MIDPRICE matches TA-Lib exactly."""
        datasets = [ohlc_data, trending_data, random_ohlc_data]

        for data in datasets:
            high = data["high"]
            low = data["low"]

            for period in [2, 5, 10, 14, 20]:
                if len(high) < period:
                    continue

                expected = talib.MIDPRICE(high, low, timeperiod=period)
                result = midprice(high, low, timeperiod=period)

                # Compare where both have values
                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-7,
                        err_msg=f"MIDPRICE mismatch for period {period}",
                    )

    def test_midprice_polars(self, random_ohlc_data):
        """Test MIDPRICE with Polars expressions."""
        df = pl.DataFrame(
            {
                "high": random_ohlc_data["high"],
                "low": random_ohlc_data["low"],
            },
        )

        result = df.with_columns(
            midprice("high", "low", timeperiod=14).alias("midprice"),
        )

        expected = talib.MIDPRICE(
            random_ohlc_data["high"],
            random_ohlc_data["low"],
            timeperiod=14,
        )
        result_np = result["midprice"].to_numpy()

        valid_idx = ~(np.isnan(expected) | np.isnan(result_np))
        if np.any(valid_idx):
            assert_allclose(result_np[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_midprice_simple_cases(self, ohlc_data):
        """Test MIDPRICE with simple cases."""
        high = ohlc_data["high"]
        low = ohlc_data["low"]

        # Test with period=2
        result = midprice(high, low, timeperiod=2)

        # Manual calculation for first few points
        # Index 1: highest_high = max(12, 10) = 12, lowest_low = min(10, 9) = 9
        # midprice = (12 + 9) / 2 = 10.5
        assert_allclose(result[1], 10.5, atol=1e-10)

        # Index 2: highest_high = max(11, 12) = 12, lowest_low = min(10, 10) = 10
        # midprice = (12 + 10) / 2 = 11.0
        assert_allclose(result[2], 11.0, atol=1e-10)

    def test_midprice_vs_medprice(self):
        """Test MIDPRICE relationship with MEDPRICE."""
        # For flat/constant data, MIDPRICE should converge to MEDPRICE
        high_flat = np.array([10.0] * 20)
        low_flat = np.array([8.0] * 20)

        midprice_5 = midprice(high_flat, low_flat, timeperiod=5)
        medprice_result = medprice(high_flat, low_flat)

        # For constant data, both should give the same result
        expected = (10.0 + 8.0) / 2  # 9.0
        valid_idx = ~np.isnan(midprice_5)
        assert_allclose(midprice_5[valid_idx], expected, atol=1e-10)
        assert_allclose(medprice_result, expected, atol=1e-10)

    def test_midprice_constant_range(self):
        """Test MIDPRICE with constant high-low range."""
        # Create data with constant range but shifting values
        n = 20
        base = np.linspace(100, 120, n)
        high = base + 5  # Constant range of 10
        low = base - 5

        result = midprice(high, low, timeperiod=5)
        expected = talib.MIDPRICE(high, low, timeperiod=5)

        # Compare with TA-Lib
        valid_idx = ~(np.isnan(result) | np.isnan(expected))
        assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

        # For linearly increasing data with constant spread,
        # midprice will be the average of the window's extremes
        # e.g., for window [100, 101, 102, 103, 104] with ±5 spread:
        # highest = 104+5 = 109, lowest = 100-5 = 95
        # midprice = (109 + 95) / 2 = 102

    def test_midprice_expanding_range(self):
        """Test MIDPRICE with expanding price range."""
        # Create data with expanding range
        n = 50
        center = 100.0
        high = np.zeros(n)
        low = np.zeros(n)

        for i in range(n):
            expansion = i * 0.5
            high[i] = center + expansion
            low[i] = center - expansion

        result = midprice(high, low, timeperiod=10)

        # The midprice should remain at the center
        valid = result[~np.isnan(result)]
        assert_allclose(valid, center, atol=1e-10)

    def test_midprice_vs_midpoint(self):
        """Test relationship between MIDPRICE and MIDPOINT."""
        # When high and low follow the same pattern, MIDPRICE should be related to MIDPOINT
        np.random.seed(42)
        n = 100
        base_values = 100 + np.cumsum(np.random.randn(n) * 0.5)

        # Create high/low with fixed spread
        spread = 2.0
        high = base_values + spread / 2
        low = base_values - spread / 2

        midprice_result = midprice(high, low, timeperiod=14)
        midpoint_result = midpoint(base_values, timeperiod=14)

        # They should be very close since high/low are symmetric around base
        valid_idx = ~(np.isnan(midprice_result) | np.isnan(midpoint_result))
        if np.any(valid_idx):
            assert_allclose(midprice_result[valid_idx], midpoint_result[valid_idx], rtol=1e-6)

    def test_midprice_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([10, 11, 12, 13, 14], dtype=float)
        low = np.array([9, 10, 11, 12, 13], dtype=float)

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            midprice(high, low, timeperiod=1)

        # Mismatched array lengths
        with pytest.raises(
            ValueError,
            match="high and low arrays must have the same length",
        ):
            midprice(high, low[:-1], timeperiod=2)

    def test_midprice_nan_pattern(self):
        """Test MIDPRICE NaN pattern."""
        high = np.random.uniform(100, 110, 100)
        low = np.random.uniform(90, 100, 100)

        for period in [5, 10, 20]:
            result = midprice(high, low, timeperiod=period)
            expected = talib.MIDPRICE(high, low, timeperiod=period)

            # First period-1 values should be NaN
            assert np.all(np.isnan(result[: period - 1]))
            assert ~np.isnan(result[period - 1])

            # NaN patterns should match
            assert np.array_equal(np.isnan(result), np.isnan(expected))

    def test_midprice_edge_cases(self):
        """Test MIDPRICE with edge cases."""
        # When high == low (single price point)
        single_price = np.array([100] * 10, dtype=float)
        result = midprice(single_price, single_price, timeperiod=5)
        valid = result[~np.isnan(result)]
        assert_allclose(valid, 100.0, atol=1e-10)

        # Large period relative to data size
        high = np.array([10, 15, 12, 18, 14], dtype=float)
        low = np.array([8, 12, 10, 15, 12], dtype=float)
        result = midprice(high, low, timeperiod=5)

        # Only one value should be non-NaN
        assert np.sum(~np.isnan(result)) == 1
        # It should be (highest high + lowest low) / 2 = (18 + 8) / 2 = 13
        assert_allclose(result[4], 13.0, atol=1e-10)

    def test_midprice_crypto_accuracy(self, crypto_data_small):
        """Test MIDPRICE accuracy on real crypto data."""
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()

        for period in [5, 10, 14, 20]:
            expected = talib.MIDPRICE(high, low, timeperiod=period)
            result = midprice(high, low, timeperiod=period)

            valid_idx = ~(np.isnan(expected) | np.isnan(result))
            if np.any(valid_idx):
                assert_allclose(
                    result[valid_idx],
                    expected[valid_idx],
                    rtol=1e-7,
                    err_msg=f"MIDPRICE mismatch on crypto data for period {period}",
                )

    @pytest.mark.benchmark
    def test_midprice_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark MIDPRICE performance."""
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()

        # Warmup JIT
        warmup_jit(lambda h, low_val: midprice(h, low_val), high[:100], low[:100])

        import time

        # Our implementation
        start = time.perf_counter()
        _ = midprice(high, low, timeperiod=14)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.MIDPRICE(high, low, timeperiod=14)
        talib_time = time.perf_counter() - start

        print(f"\nMIDPRICE Performance ({len(high):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # MIDPRICE requires two rolling operations (MAX and MIN)
        # Use same threshold as math_basic operators
        threshold = performance_threshold("math_basic")
        assert our_time < talib_time * threshold, (
            f"MIDPRICE performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
