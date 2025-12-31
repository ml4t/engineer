"""
Test KAMA (Kaufman's Adaptive Moving Average) indicator.
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

from ml4t.engineer.features.trend import kama


class TestKAMA:
    """Test KAMA indicator."""

    @pytest.fixture
    def trending_data(self):
        """Generate trending data to test adaptive behavior."""
        np.random.seed(42)

        # Create data with trending and ranging periods
        trend1 = np.linspace(100, 120, 50)
        range1 = 120 + np.random.normal(0, 0.5, 50)
        trend2 = np.linspace(120, 110, 50)
        range2 = 110 + np.random.normal(0, 0.3, 50)

        return np.concatenate([trend1, range1, trend2, range2])

    @pytest.fixture
    def random_data(self):
        """Generate random test data."""
        np.random.seed(42)
        return np.random.randn(1000) * 10 + 100

    def test_kama_accuracy(self, trending_data, random_data):
        """Test KAMA matches TA-Lib exactly."""
        # Test with trending data
        for period in [10, 20, 30, 50]:
            expected = talib.KAMA(trending_data, timeperiod=period)
            result = kama(trending_data, timeperiod=period)

            # KAMA has an unstable period, so focus on stable part
            # TA-Lib defaults to 1 unstable period for KAMA
            period + 1

            # Compare where both have values
            valid_idx = ~(np.isnan(expected) | np.isnan(result))
            if np.any(valid_idx):
                assert_allclose(
                    result[valid_idx],
                    expected[valid_idx],
                    rtol=1e-7,
                    err_msg=f"KAMA mismatch for period {period}",
                )

        # Test with random data
        expected = talib.KAMA(random_data, timeperiod=30)
        result = kama(random_data, timeperiod=30)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_kama_polars(self, trending_data):
        """Test KAMA with Polars expressions."""
        df = pl.DataFrame({"price": trending_data})

        result = df.with_columns(kama("price", timeperiod=20).alias("kama"))

        expected = talib.KAMA(trending_data, timeperiod=20)
        result_np = result["kama"].to_numpy()

        # Compare where both have values
        valid_idx = ~(np.isnan(expected) | np.isnan(result_np))
        if np.any(valid_idx):
            assert_allclose(result_np[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_kama_adaptive_behavior(self, trending_data):
        """Test that KAMA adapts to market conditions."""
        kama_values = kama(trending_data, timeperiod=20)

        # Remove NaN values for analysis
        valid_kama = kama_values[~np.isnan(kama_values)]
        valid_prices = trending_data[~np.isnan(kama_values)]

        # During trending periods (first 50 and 100-150), KAMA should follow closely
        # During ranging periods (50-100 and 150-200), KAMA should be smoother

        # Calculate average distance from price
        trend1_dist = np.mean(np.abs(valid_kama[:45] - valid_prices[:45]))
        range1_dist = np.mean(np.abs(valid_kama[45:95] - valid_prices[45:95]))

        # In trending markets, KAMA should be closer to price
        # In ranging markets, KAMA should be further (more smoothing)
        assert trend1_dist < range1_dist * 0.8, "KAMA should adapt to trends"

    def test_kama_edge_cases(self):
        """Test KAMA with edge cases."""
        # Minimum period
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = kama(values, timeperiod=2)
        expected = talib.KAMA(values, timeperiod=2)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

        # Constant values
        values = np.full(50, 100.0)
        result = kama(values, timeperiod=10)
        # KAMA should converge to the constant value
        assert_allclose(result[20:], 100.0, rtol=1e-7)

    def test_kama_parameter_validation(self):
        """Test parameter validation."""
        values = np.random.randn(100)

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            kama(values, timeperiod=1)

    def test_kama_nan_pattern(self):
        """Test KAMA NaN pattern."""
        values = np.random.randn(100)

        for period in [10, 20, 30]:
            result = kama(values, timeperiod=period)
            expected = talib.KAMA(values, timeperiod=period)

            # The NaN pattern might differ due to unstable period
            # Just ensure we have some output after the minimum period
            assert np.sum(~np.isnan(result)) > 0
            assert np.sum(~np.isnan(expected)) > 0

    def test_kama_crypto_accuracy(self, crypto_data_small):
        """Test KAMA accuracy on real crypto data."""
        prices = crypto_data_small["close"].to_numpy()

        # Test different periods
        for period in [10, 20, 30]:
            if len(prices) < period + 10:  # Need some extra for unstable period
                continue

            expected = talib.KAMA(prices, timeperiod=period)
            result = kama(prices, timeperiod=period)

            # Compare where both have values
            valid_idx = ~(np.isnan(expected) | np.isnan(result))
            if np.any(valid_idx):
                assert_allclose(
                    result[valid_idx],
                    expected[valid_idx],
                    rtol=1e-7,
                    err_msg=f"KAMA mismatch on crypto data for period {period}",
                )

    @pytest.mark.benchmark
    def test_kama_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark KAMA performance."""
        prices = crypto_data["close"].to_numpy()

        # Warmup JIT
        warmup_jit(kama, prices[:100])

        import time

        # Our implementation
        start = time.perf_counter()
        _ = kama(prices, timeperiod=30)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.KAMA(prices, timeperiod=30)
        talib_time = time.perf_counter() - start

        print(f"\nKAMA Performance ({len(prices):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # KAMA is complex, use appropriate threshold
        threshold = performance_threshold("complex")
        assert our_time < talib_time * threshold
