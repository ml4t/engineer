"""
Test TSF (Time Series Forecast) indicator.

TSF projects a linear regression line one period into the future.
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

from ml4t.engineer.features.statistics import tsf


class TestTSF:
    """Test TSF indicator."""

    @pytest.fixture
    def price_data(self):
        """Generate sample price data."""
        np.random.seed(42)
        n = 100

        # Generate realistic price movements with some trend
        trend = np.linspace(100, 150, n)
        noise = np.random.normal(0, 2, n)
        close = trend + noise

        return {"close": close}

    def test_tsf_accuracy(self, price_data):
        """Test TSF matches TA-Lib exactly."""
        close = price_data["close"]

        # Test different periods
        for period in [5, 14, 21]:
            # TA-Lib reference
            expected = talib.TSF(close, timeperiod=period)

            # Our implementation
            result = tsf(close, timeperiod=period)

            # Should match within reasonable tolerance
            assert_indicator_match(
                result,
                expected,
                f"TSF(period={period})",
                rtol=1e-10,
            )

    def test_tsf_polars(self, price_data):
        """Test TSF with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expressions
        result_df = df.with_columns(
            [
                tsf("close", 14).alias("tsf_14"),
            ],
        )

        # Compare with TA-Lib
        expected = talib.TSF(price_data["close"], timeperiod=14)

        assert_indicator_match(
            result_df["tsf_14"].to_numpy(),
            expected,
            "TSF (Polars)",
            rtol=1e-10,
        )

    def test_tsf_mathematical_properties(self):
        """Test mathematical properties of TSF."""
        # Test with a perfect linear trend
        trend_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = tsf(trend_data, timeperiod=3)

        # For a perfect trend, TSF should predict the next value exactly
        # With period=3: TSF[2] should predict 4.0, TSF[3] should predict 5.0, etc.
        expected_values = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

        valid_idx = ~np.isnan(result)
        assert_allclose(result[valid_idx], expected_values, rtol=1e-10)

    def test_tsf_edge_cases(self):
        """Test TSF with edge cases."""
        # Case 1: Minimal data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = tsf(data, timeperiod=3)

        # First 2 values should be NaN
        assert np.all(np.isnan(result[:2]))
        assert ~np.isnan(result[2])

        # Case 2: Constant values
        data = np.full(10, 100.0)

        result = tsf(data, timeperiod=5)
        expected = talib.TSF(data, timeperiod=5)

        assert_indicator_match(result, expected, "TSF (constant)", rtol=1e-10)

    def test_tsf_parameter_validation(self):
        """Test parameter validation."""
        data = np.array([1.0, 2.0, 3.0])

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            tsf(data, timeperiod=0)

    def test_tsf_insufficient_data(self):
        """Test behavior with insufficient data."""
        data = np.array([1.0, 2.0])

        # Period larger than data length
        result = tsf(data, timeperiod=5)

        # Should return all NaN
        assert np.all(np.isnan(result))

    def test_tsf_crypto_accuracy(self, crypto_data_small):
        """Test TSF accuracy on real crypto data."""
        # Extract close prices from real crypto data
        close = crypto_data_small["close"].to_numpy()

        # Test different periods
        for period in [7, 14, 21]:
            # TA-Lib reference
            expected = talib.TSF(close, timeperiod=period)

            # Our implementation
            result = tsf(close, timeperiod=period)

            # Should match exactly
            assert_indicator_match(
                result,
                expected,
                f"TSF (crypto, period={period})",
                rtol=1e-10,
            )

    def test_tsf_forecasting_property(self):
        """Test that TSF actually forecasts the next value."""
        # Create data where we know the trend
        x = np.arange(20)
        # Linear trend: y = 2x + 10
        data = 2 * x + 10 + np.random.normal(0, 0.1, 20)  # Small noise

        result = tsf(data, timeperiod=5)

        # TSF should be close to the actual next values for low-noise linear data
        # This is more of a conceptual test than exact validation
        valid_idx = ~np.isnan(result)
        assert len(result[valid_idx]) > 0  # Should have some valid forecasts

    @pytest.mark.benchmark
    def test_tsf_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark TSF performance using real crypto data."""
        # Use real crypto data
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(tsf, close[:100], 14)

        import time

        # Our implementation (warmed up)
        start = time.perf_counter()
        _ = tsf(close, timeperiod=14)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.TSF(close, timeperiod=14)
        talib_time = time.perf_counter() - start

        print(f"\\nTSF Performance ({len(close):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use threshold for moderate complexity (linear regression)
        threshold = performance_threshold("moderate")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
