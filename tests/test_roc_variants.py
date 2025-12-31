"""
Test ROC variant indicators: ROCP, ROCR, ROCR100.

These indicators calculate rate of change in different formats.
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

from ml4t.engineer.features.momentum import rocp, rocr, rocr100


class TestROCVariants:
    """Test ROC variant indicators."""

    @pytest.fixture
    def price_data(self):
        """Generate sample price data."""
        np.random.seed(42)
        n = 100

        # Generate realistic price movements
        returns = np.random.normal(0.0001, 0.02, n)
        close = 100 * (1 + returns).cumprod()

        return {"close": close}

    def test_rocp_accuracy(self, price_data):
        """Test ROCP matches TA-Lib exactly."""
        close = price_data["close"]

        # Test different periods
        for period in [5, 10, 20]:
            # TA-Lib reference
            expected = talib.ROCP(close, timeperiod=period)

            # Our implementation
            result = rocp(close, timeperiod=period)

            # Should match exactly
            assert_indicator_match(
                result,
                expected,
                f"ROCP(period={period})",
                rtol=1e-10,
            )

    def test_rocr_accuracy(self, price_data):
        """Test ROCR matches TA-Lib exactly."""
        close = price_data["close"]

        # Test different periods
        for period in [5, 10, 20]:
            # TA-Lib reference
            expected = talib.ROCR(close, timeperiod=period)

            # Our implementation
            result = rocr(close, timeperiod=period)

            # Should match exactly
            assert_indicator_match(
                result,
                expected,
                f"ROCR(period={period})",
                rtol=1e-10,
            )

    def test_rocr100_accuracy(self, price_data):
        """Test ROCR100 matches TA-Lib exactly."""
        close = price_data["close"]

        # Test different periods
        for period in [5, 10, 20]:
            # TA-Lib reference
            expected = talib.ROCR100(close, timeperiod=period)

            # Our implementation
            result = rocr100(close, timeperiod=period)

            # Should match exactly
            assert_indicator_match(
                result,
                expected,
                f"ROCR100(period={period})",
                rtol=1e-10,
            )

    def test_roc_variants_polars(self, price_data):
        """Test ROC variants with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expressions
        result_df = df.with_columns(
            [
                rocp("close", 10).alias("rocp"),
                rocr("close", 10).alias("rocr"),
                rocr100("close", 10).alias("rocr100"),
            ],
        )

        # Compare with TA-Lib
        expected_rocp = talib.ROCP(price_data["close"], timeperiod=10)
        expected_rocr = talib.ROCR(price_data["close"], timeperiod=10)
        expected_rocr100 = talib.ROCR100(price_data["close"], timeperiod=10)

        assert_indicator_match(
            result_df["rocp"].to_numpy(),
            expected_rocp,
            "ROCP (Polars)",
            rtol=1e-10,
        )
        assert_indicator_match(
            result_df["rocr"].to_numpy(),
            expected_rocr,
            "ROCR (Polars)",
            rtol=1e-10,
        )
        assert_indicator_match(
            result_df["rocr100"].to_numpy(),
            expected_rocr100,
            "ROCR100 (Polars)",
            rtol=1e-10,
        )

    def test_roc_variants_relationships(self):
        """Test mathematical relationships between ROC variants."""
        np.random.seed(42)
        close = 100 + np.random.randn(50).cumsum()
        period = 10

        rocp_result = rocp(close, period)
        rocr_result = rocr(close, period)
        rocr100_result = rocr100(close, period)

        # Verify relationships
        valid_idx = ~np.isnan(rocp_result)

        # ROCP = ROCR - 1 (both are ratios, not percentages)
        expected_rocp = rocr_result[valid_idx] - 1
        assert_allclose(rocp_result[valid_idx], expected_rocp, rtol=1e-10)

        # ROCR100 = ROCR * 100
        expected_rocr100 = rocr_result[valid_idx] * 100
        assert_allclose(rocr100_result[valid_idx], expected_rocr100, rtol=1e-10)

    def test_roc_variants_edge_cases(self):
        """Test ROC variants with edge cases."""
        # Case 1: Minimal data
        data = np.array([100.0, 102.0, 104.0, 103.0, 105.0])

        rocp_result = rocp(data, timeperiod=3)
        rocr_result = rocr(data, timeperiod=3)
        rocr100_result = rocr100(data, timeperiod=3)

        # First 3 values should be NaN
        assert np.all(np.isnan(rocp_result[:3]))
        assert np.all(np.isnan(rocr_result[:3]))
        assert np.all(np.isnan(rocr100_result[:3]))

        # Check specific values
        assert_allclose(rocp_result[3], 0.03, rtol=1e-10)  # (103-100)/100 = 0.03
        assert_allclose(rocr_result[3], 1.03, rtol=1e-10)  # 103/100 = 1.03
        assert_allclose(rocr100_result[3], 103.0, rtol=1e-10)  # 103/100*100 = 103.0

        # Case 2: Zero price (should handle division by zero)
        data_with_zero = np.array([100.0, 0.0, 50.0, 75.0, 100.0])

        # These should handle zero gracefully
        rocp_zero = rocp(data_with_zero, timeperiod=2)
        rocr(data_with_zero, timeperiod=2)
        rocr100(data_with_zero, timeperiod=2)

        # Test specific values with zero in data
        # data_with_zero = [100.0, 0.0, 50.0, 75.0, 100.0], timeperiod=2
        # ROCP[2] = (50 - 100) / 100 = -0.5 (not division by zero)
        # ROCP[3] = (75 - 0) / 0 = 0.0 (TA-Lib special case)
        # ROCP[4] = (100 - 50) / 50 = 1.0
        assert_allclose(rocp_zero[2], -0.5, rtol=1e-10)
        assert_allclose(rocp_zero[3], 0.0, rtol=1e-10)
        assert_allclose(rocp_zero[4], 1.0, rtol=1e-10)

    def test_roc_variants_parameter_validation(self):
        """Test parameter validation."""
        data = np.array([1.0, 2.0, 3.0])

        # Invalid period
        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            rocp(data, timeperiod=0)

        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            rocr(data, timeperiod=0)

        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            rocr100(data, timeperiod=0)

    def test_roc_variants_crypto_accuracy(self, crypto_data_small):
        """Test ROC variants accuracy on real crypto data."""
        close = crypto_data_small["close"].to_numpy()

        # Test different periods
        for period in [5, 14, 30]:
            # ROCP
            expected_rocp = talib.ROCP(close, timeperiod=period)
            result_rocp = rocp(close, timeperiod=period)
            assert_indicator_match(
                result_rocp,
                expected_rocp,
                f"ROCP (crypto, period={period})",
                rtol=1e-10,
            )

            # ROCR
            expected_rocr = talib.ROCR(close, timeperiod=period)
            result_rocr = rocr(close, timeperiod=period)
            assert_indicator_match(
                result_rocr,
                expected_rocr,
                f"ROCR (crypto, period={period})",
                rtol=1e-10,
            )

            # ROCR100
            expected_rocr100 = talib.ROCR100(close, timeperiod=period)
            result_rocr100 = rocr100(close, timeperiod=period)
            assert_indicator_match(
                result_rocr100,
                expected_rocr100,
                f"ROCR100 (crypto, period={period})",
                rtol=1e-10,
            )

    @pytest.mark.benchmark
    def test_roc_variants_performance(
        self,
        crypto_data,
        performance_threshold,
        warmup_jit,
    ):
        """Benchmark ROC variants performance using real crypto data."""
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation
        warmup_jit(rocp, close[:100], 10)
        warmup_jit(rocr, close[:100], 10)
        warmup_jit(rocr100, close[:100], 10)

        import time

        # Test ROCP
        start = time.perf_counter()
        _ = rocp(close, timeperiod=10)
        our_rocp_time = time.perf_counter() - start

        start = time.perf_counter()
        _ = talib.ROCP(close, timeperiod=10)
        talib_rocp_time = time.perf_counter() - start

        print(f"\nROCP Performance ({len(close):,} rows):")
        print(f"Our implementation: {our_rocp_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_rocp_time * 1000:.2f}ms")
        print(f"Ratio: {our_rocp_time / talib_rocp_time:.2f}x")

        # Use threshold for simple operations
        threshold = performance_threshold("simple")
        assert our_rocp_time < talib_rocp_time * threshold, (
            f"ROCP performance ratio {our_rocp_time / talib_rocp_time:.1f}x exceeds threshold {threshold}x"
        )
