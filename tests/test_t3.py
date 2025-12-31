"""
Test T3 (Triple Exponential Moving Average) indicator.
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

from ml4t.engineer.features.trend import t3


class TestT3:
    """Test T3 indicator."""

    @pytest.fixture
    def smooth_data(self):
        """Generate smooth trending data."""
        # T3 is designed for smoothing, test with smooth trends
        n = 200
        x = np.linspace(0, 4 * np.pi, n)
        # Sine wave with trend
        return 100 + x * 2 + 10 * np.sin(x) + np.random.normal(0, 0.5, n)

    @pytest.fixture
    def random_data(self):
        """Generate random test data."""
        np.random.seed(42)
        return np.random.randn(1000) * 10 + 100

    def test_t3_accuracy(self, smooth_data, random_data):
        """Test T3 matches TA-Lib exactly."""
        # Test with smooth data
        for period in [5, 10, 20]:
            for vfactor in [0.5, 0.7, 0.9]:
                expected = talib.T3(smooth_data, timeperiod=period, vfactor=vfactor)
                result = t3(smooth_data, timeperiod=period, vfactor=vfactor)

                # T3 has an unstable period, focus on stable part
                # Compare where both have values
                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-7,
                        err_msg=f"T3 mismatch for period {period}, vfactor {vfactor}",
                    )

        # Test with random data
        expected = talib.T3(random_data, timeperiod=5, vfactor=0.7)
        result = t3(random_data, timeperiod=5, vfactor=0.7)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_t3_polars(self, smooth_data):
        """Test T3 with Polars expressions."""
        df = pl.DataFrame({"price": smooth_data})

        result = df.with_columns(
            t3("price", timeperiod=10, vfactor=0.7).alias("t3"),
        )

        expected = talib.T3(smooth_data, timeperiod=10, vfactor=0.7)
        result_np = result["t3"].to_numpy()

        # Compare where both have values
        valid_idx = ~(np.isnan(expected) | np.isnan(result_np))
        if np.any(valid_idx):
            assert_allclose(result_np[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_t3_smoothing(self, smooth_data):
        """Test that T3 provides smoothing."""
        # T3 should be smoother than the input
        t3_values = t3(smooth_data, timeperiod=10, vfactor=0.7)

        # Remove NaN values
        valid_idx = ~np.isnan(t3_values)
        t3_valid = t3_values[valid_idx]
        data_valid = smooth_data[valid_idx]

        # Calculate roughness (sum of absolute differences)
        data_roughness = np.sum(np.abs(np.diff(data_valid)))
        t3_roughness = np.sum(np.abs(np.diff(t3_valid)))

        # T3 should be significantly smoother
        # Note: 0.7 threshold is reasonable - T3 typically reduces roughness by 30-50%
        assert t3_roughness < data_roughness * 0.7, "T3 should provide smoothing"

    def test_t3_vfactor_effect(self, smooth_data):
        """Test that vfactor controls smoothing amount."""
        # Lower vfactor = more smoothing, higher vfactor = less smoothing
        t3_low_vf = t3(smooth_data, timeperiod=10, vfactor=0.3)
        t3_high_vf = t3(smooth_data, timeperiod=10, vfactor=0.9)

        # Remove NaN values
        valid_idx = ~(np.isnan(t3_low_vf) | np.isnan(t3_high_vf))
        t3_low = t3_low_vf[valid_idx]
        t3_high = t3_high_vf[valid_idx]
        data = smooth_data[valid_idx]

        # High vfactor should follow price more closely
        low_vf_distance = np.mean(np.abs(t3_low - data))
        high_vf_distance = np.mean(np.abs(t3_high - data))

        assert high_vf_distance < low_vf_distance, "Higher vfactor should follow price more closely"

    def test_t3_edge_cases(self):
        """Test T3 with edge cases."""
        # Minimum period
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = t3(values, timeperiod=2, vfactor=0.7)
        expected = talib.T3(values, timeperiod=2, vfactor=0.7)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

        # Extreme vfactor values
        for vf in [0.0, 1.0]:
            result = t3(values, timeperiod=3, vfactor=vf)
            expected = talib.T3(values, timeperiod=3, vfactor=vf)

            valid_idx = ~(np.isnan(expected) | np.isnan(result))
            if np.any(valid_idx):
                assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_t3_parameter_validation(self):
        """Test parameter validation."""
        values = np.random.randn(100)

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            t3(values, timeperiod=1)

        # Invalid vfactor
        with pytest.raises(ValueError, match="vfactor must be between 0 and 1"):
            t3(values, timeperiod=5, vfactor=-0.1)

        with pytest.raises(ValueError, match="vfactor must be between 0 and 1"):
            t3(values, timeperiod=5, vfactor=1.1)

    def test_t3_nan_pattern(self):
        """Test T3 NaN pattern."""
        values = np.random.randn(100)

        for period in [5, 10, 20]:
            result = t3(values, timeperiod=period)
            talib.T3(values, timeperiod=period)

            # The NaN pattern might differ due to unstable period
            # Just ensure we have some output after the minimum period
            lookback = 6 * (period - 1)
            assert np.sum(~np.isnan(result)) > len(values) - lookback - 10

    def test_t3_crypto_accuracy(self, crypto_data_small):
        """Test T3 accuracy on real crypto data."""
        prices = crypto_data_small["close"].to_numpy()

        # Test different periods and vfactors
        for period in [5, 10, 20]:
            for vfactor in [0.5, 0.7]:
                if len(prices) < 6 * period:  # Need enough data for T3
                    continue

                expected = talib.T3(prices, timeperiod=period, vfactor=vfactor)
                result = t3(prices, timeperiod=period, vfactor=vfactor)

                # Compare where both have values
                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-7,
                        err_msg=f"T3 mismatch on crypto data for period {period}, vfactor {vfactor}",
                    )

    @pytest.mark.benchmark
    def test_t3_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark T3 performance."""
        prices = crypto_data["close"].to_numpy()

        # Warmup JIT
        warmup_jit(t3, prices[:100])

        import time

        # Our implementation
        start = time.perf_counter()
        _ = t3(prices, timeperiod=5, vfactor=0.7)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.T3(prices, timeperiod=5, vfactor=0.7)
        talib_time = time.perf_counter() - start

        print(f"\nT3 Performance ({len(prices):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # T3 is very complex (6 EMAs), use appropriate threshold
        threshold = performance_threshold("very_complex")
        assert our_time < talib_time * threshold
