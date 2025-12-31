"""
Test CMO (Chande Momentum Oscillator) indicator.
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

from ml4t.engineer.features.momentum import cmo, rsi


class TestCMO:
    """Test CMO indicator."""

    @pytest.fixture
    def trending_data(self):
        """Generate trending data for CMO testing."""
        # CMO responds well to trending data
        n = 100
        trend = np.linspace(100, 150, n)
        noise = np.random.normal(0, 1, n)
        return trend + noise

    @pytest.fixture
    def oscillating_data(self):
        """Generate oscillating data."""
        n = 200
        x = np.linspace(0, 4 * np.pi, n)
        return 100 + 10 * np.sin(x) + np.random.normal(0, 0.5, n)

    @pytest.fixture
    def random_data(self):
        """Generate random test data."""
        np.random.seed(42)
        return np.random.randn(1000) * 10 + 100

    def test_cmo_accuracy(self, trending_data, oscillating_data, random_data):
        """Test CMO matches TA-Lib exactly."""
        # Test with different data patterns
        for data in [trending_data, oscillating_data, random_data]:
            for period in [5, 14, 20, 30]:
                expected = talib.CMO(data, timeperiod=period)
                result = cmo(data, timeperiod=period)

                # Compare where both have values
                valid_idx = ~(np.isnan(expected) | np.isnan(result))
                if np.any(valid_idx):
                    assert_allclose(
                        result[valid_idx],
                        expected[valid_idx],
                        rtol=1e-7,
                        err_msg=f"CMO mismatch for period {period}",
                    )

    def test_cmo_polars(self, random_data):
        """Test CMO with Polars expressions."""
        df = pl.DataFrame({"price": random_data})

        result = df.with_columns(cmo("price", timeperiod=14).alias("cmo"))

        expected = talib.CMO(random_data, timeperiod=14)
        result_np = result["cmo"].to_numpy()

        valid_idx = ~(np.isnan(expected) | np.isnan(result_np))
        if np.any(valid_idx):
            assert_allclose(result_np[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_cmo_range(self, random_data):
        """Test that CMO values are within [-100, 100]."""
        result = cmo(random_data, timeperiod=14)
        valid_values = result[~np.isnan(result)]

        assert np.all(valid_values >= -100), "CMO values should be >= -100"
        assert np.all(valid_values <= 100), "CMO values should be <= 100"

    def test_cmo_trending_behavior(self):
        """Test CMO behavior with strong trends."""
        # Strong uptrend
        uptrend = np.linspace(100, 200, 50)
        cmo_up = cmo(uptrend, timeperiod=10)

        # Remove NaN values
        valid_up = cmo_up[~np.isnan(cmo_up)]

        # CMO should be mostly positive in an uptrend
        assert np.mean(valid_up) > 50, "CMO should be strongly positive in uptrend"

        # Strong downtrend
        downtrend = np.linspace(200, 100, 50)
        cmo_down = cmo(downtrend, timeperiod=10)

        valid_down = cmo_down[~np.isnan(cmo_down)]

        # CMO should be mostly negative in a downtrend
        assert np.mean(valid_down) < -50, "CMO should be strongly negative in downtrend"

    def test_cmo_vs_rsi_relationship(self, random_data):
        """Test relationship between CMO and RSI."""
        # CMO and RSI are related:
        # RSI = 50 * (CMO/100 + 1)
        # or CMO = 2 * RSI - 100

        cmo_values = cmo(random_data, timeperiod=14)
        rsi_values = rsi(random_data, period=14)

        # Compare where both have values
        valid_idx = ~(np.isnan(cmo_values) | np.isnan(rsi_values))

        if np.any(valid_idx):
            # Calculate expected CMO from RSI
            expected_cmo = 2 * rsi_values[valid_idx] - 100

            # They should match within numerical precision
            assert_allclose(
                cmo_values[valid_idx],
                expected_cmo,
                rtol=1e-5,  # Slightly looser due to different smoothing
                err_msg="CMO and RSI relationship doesn't hold",
            )

    def test_cmo_edge_cases(self):
        """Test CMO with edge cases."""
        # Constant values
        constant = np.full(50, 100.0)
        result = cmo(constant, timeperiod=14)

        # CMO should be 0 for constant values (no momentum)
        valid = result[~np.isnan(result)]
        assert_allclose(valid, 0.0, atol=1e-10)

        # Minimum period
        values = np.random.randn(10) + 100
        result = cmo(values, timeperiod=2)
        expected = talib.CMO(values, timeperiod=2)

        valid_idx = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_idx):
            assert_allclose(result[valid_idx], expected[valid_idx], rtol=1e-7)

    def test_cmo_parameter_validation(self):
        """Test parameter validation."""
        values = np.random.randn(100)

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            cmo(values, timeperiod=1)

    def test_cmo_nan_pattern(self):
        """Test CMO NaN pattern."""
        values = np.random.randn(100)

        for period in [5, 14, 30]:
            result = cmo(values, timeperiod=period)
            expected = talib.CMO(values, timeperiod=period)

            # First 'period' values should be NaN
            assert np.all(np.isnan(result[:period]))
            assert ~np.isnan(result[period])

            # NaN patterns should match
            assert np.array_equal(np.isnan(result), np.isnan(expected))

    def test_cmo_crypto_accuracy(self, crypto_data_small):
        """Test CMO accuracy on real crypto data."""
        prices = crypto_data_small["close"].to_numpy()

        for period in [7, 14, 21]:
            expected = talib.CMO(prices, timeperiod=period)
            result = cmo(prices, timeperiod=period)

            valid_idx = ~(np.isnan(expected) | np.isnan(result))
            if np.any(valid_idx):
                assert_allclose(
                    result[valid_idx],
                    expected[valid_idx],
                    rtol=1e-7,
                    err_msg=f"CMO mismatch on crypto data for period {period}",
                )

    @pytest.mark.benchmark
    def test_cmo_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark CMO performance."""
        prices = crypto_data["close"].to_numpy()

        # Warmup JIT
        warmup_jit(cmo, prices[:100])

        import time

        # Our implementation
        start = time.perf_counter()
        _ = cmo(prices, timeperiod=14)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.CMO(prices, timeperiod=14)
        talib_time = time.perf_counter() - start

        print(f"\nCMO Performance ({len(prices):,} rows):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # CMO is similar complexity to RSI
        threshold = performance_threshold("simple")
        assert our_time < talib_time * threshold
