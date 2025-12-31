"""
Test Directional Movement indicators: PLUS_DI, MINUS_DI, DX.

These are components of the ADX (Average Directional Index) system:
- PLUS_DI: Positive Directional Indicator
- MINUS_DI: Negative Directional Indicator
- DX: Directional Movement Index
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

from ml4t.engineer.features.momentum import dx, minus_di, plus_di


class TestDirectionalIndicators:
    """Test Directional Movement indicators."""

    @pytest.fixture
    def price_data(self):
        """Generate sample OHLC data."""
        np.random.seed(42)
        n = 100

        # Generate realistic price movements
        returns = np.random.normal(0.0001, 0.02, n)
        close = 100 * (1 + returns).cumprod()

        # Generate OHLC data
        high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))

        return {"high": high, "low": low, "close": close}

    def test_plus_di_accuracy(self, price_data):
        """Test PLUS_DI matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.PLUS_DI(high, low, close, timeperiod=14)

        # Our implementation
        result = plus_di(high, low, close, timeperiod=14)

        # Ultra-high precision - directional indicators achieve machine-level accuracy
        assert_indicator_match(result, expected, "PLUS_DI", rtol=1e-10)

    def test_minus_di_accuracy(self, price_data):
        """Test MINUS_DI matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.MINUS_DI(high, low, close, timeperiod=14)

        # Our implementation
        result = minus_di(high, low, close, timeperiod=14)

        # Ultra-high precision - directional indicators achieve machine-level accuracy
        assert_indicator_match(result, expected, "MINUS_DI", rtol=1e-10)

    def test_dx_accuracy(self, price_data):
        """Test DX matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.DX(high, low, close, timeperiod=14)

        # Our implementation
        result = dx(high, low, close, timeperiod=14)

        # Ultra-high precision - DX achieves near floating-point accuracy
        assert_indicator_match(result, expected, "DX", rtol=1e-10)

    def test_directional_polars(self, price_data):
        """Test directional indicators with Polars expressions."""
        df = pl.DataFrame(price_data)

        # Using Polars expressions
        result_df = df.with_columns(
            [
                plus_di("high", "low", "close", 14).alias("plus_di"),
                minus_di("high", "low", "close", 14).alias("minus_di"),
                dx("high", "low", "close", 14).alias("dx"),
            ],
        )

        # Compare with TA-Lib
        expected_plus = talib.PLUS_DI(
            price_data["high"],
            price_data["low"],
            price_data["close"],
            timeperiod=14,
        )
        expected_minus = talib.MINUS_DI(
            price_data["high"],
            price_data["low"],
            price_data["close"],
            timeperiod=14,
        )
        expected_dx = talib.DX(
            price_data["high"],
            price_data["low"],
            price_data["close"],
            timeperiod=14,
        )

        assert_indicator_match(
            result_df["plus_di"].to_numpy(),
            expected_plus,
            "PLUS_DI (Polars)",
            rtol=1e-10,
        )
        assert_indicator_match(
            result_df["minus_di"].to_numpy(),
            expected_minus,
            "MINUS_DI (Polars)",
            rtol=1e-10,
        )
        assert_indicator_match(
            result_df["dx"].to_numpy(),
            expected_dx,
            "DX (Polars)",
            rtol=1e-10,
        )

    def test_directional_edge_cases(self):
        """Test directional indicators with edge cases."""
        # Case 1: Minimal data
        high = np.array([101.0, 102.0, 103.0] * 10)  # Need at least 2*period values
        low = np.array([99.0, 100.0, 101.0] * 10)
        close = np.array([100.0, 101.0, 102.0] * 10)

        plus_di_result = plus_di(high, low, close, timeperiod=14)
        minus_di_result = minus_di(high, low, close, timeperiod=14)
        dx_result = dx(high, low, close, timeperiod=14)

        # First period values should be NaN
        assert np.all(np.isnan(plus_di_result[:14]))
        assert np.all(np.isnan(minus_di_result[:14]))
        assert np.all(np.isnan(dx_result[:14]))
        assert ~np.isnan(plus_di_result[14])
        assert ~np.isnan(minus_di_result[14])
        assert ~np.isnan(dx_result[14])

        # Case 2: Constant prices
        high = np.full(50, 100.0)
        low = np.full(50, 100.0)
        close = np.full(50, 100.0)

        plus_di_result = plus_di(high, low, close)
        minus_di_result = minus_di(high, low, close)
        dx_result = dx(high, low, close)

        # Compare with TA-Lib
        expected_plus = talib.PLUS_DI(high, low, close)
        expected_minus = talib.MINUS_DI(high, low, close)
        expected_dx = talib.DX(high, low, close)

        assert_indicator_match(plus_di_result, expected_plus, "PLUS_DI (constant)", rtol=0.035)
        assert_indicator_match(
            minus_di_result,
            expected_minus,
            "MINUS_DI (constant)",
            rtol=0.035,
        )
        assert_indicator_match(dx_result, expected_dx, "DX (constant)", rtol=1e-9)

    def test_directional_relationships(self):
        """Test relationships between directional indicators."""
        np.random.seed(42)
        n = 100

        # Create trending data
        trend = np.linspace(100, 150, n)
        noise = np.random.randn(n) * 0.5
        close = trend + noise
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))

        plus_di_result = plus_di(high, low, close)
        minus_di_result = minus_di(high, low, close)
        dx_result = dx(high, low, close)

        # DX should be between 0 and 100
        valid_dx = dx_result[~np.isnan(dx_result)]
        assert np.all(valid_dx >= 0)
        assert np.all(valid_dx <= 100)

        # Plus_DI and Minus_DI should be non-negative
        valid_plus = plus_di_result[~np.isnan(plus_di_result)]
        valid_minus = minus_di_result[~np.isnan(minus_di_result)]
        assert np.all(valid_plus >= 0)
        assert np.all(valid_minus >= 0)

    def test_directional_parameter_validation(self):
        """Test parameter validation."""
        high = np.array([101.0, 102.0])
        low = np.array([99.0])  # Wrong length
        close = np.array([100.0, 101.0])

        with pytest.raises(ValueError, match="must have the same length"):
            plus_di(high, low, close)

        with pytest.raises(ValueError, match="must have the same length"):
            minus_di(high, low, close)

        with pytest.raises(ValueError, match="must have the same length"):
            dx(high, low, close)

        # Invalid period
        high = np.random.randn(50) + 100
        low = high - 2
        close = (high + low) / 2

        with pytest.raises(ValueError, match="timeperiod must be > 0"):
            plus_di(high, low, close, timeperiod=0)

    def test_directional_crypto_accuracy(self, crypto_data_small):
        """Test directional indicators accuracy on real crypto data."""
        # Extract columns from 10K rows of real crypto data
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        # Test different periods
        for period in [7, 14, 21]:
            # TA-Lib reference
            expected_plus = talib.PLUS_DI(high, low, close, timeperiod=period)
            expected_minus = talib.MINUS_DI(high, low, close, timeperiod=period)
            expected_dx = talib.DX(high, low, close, timeperiod=period)

            # Our implementation
            result_plus = plus_di(high, low, close, timeperiod=period)
            result_minus = minus_di(high, low, close, timeperiod=period)
            result_dx = dx(high, low, close, timeperiod=period)

            # Should match with reasonable tolerance on real data
            # Ultra-high precision - directional indicators achieve machine-level accuracy
            assert_indicator_match(
                result_plus,
                expected_plus,
                f"PLUS_DI (crypto, period={period})",
                rtol=1e-10,
            )
            assert_indicator_match(
                result_minus,
                expected_minus,
                f"MINUS_DI (crypto, period={period})",
                rtol=1e-10,
            )
            assert_indicator_match(
                result_dx,
                expected_dx,
                f"DX (crypto, period={period})",
                rtol=1e-10,
            )

    @pytest.mark.benchmark
    def test_directional_performance(
        self,
        crypto_data,
        performance_threshold,
        warmup_jit,
    ):
        """Benchmark directional indicators performance using real crypto data."""
        # Use real crypto data
        high = crypto_data["high"].to_numpy()
        low = crypto_data["low"].to_numpy()
        close = crypto_data["close"].to_numpy()

        # Warmup JIT compilation with small sample
        warmup_jit(plus_di, high[:100], low[:100], close[:100])

        import time

        # Our implementation - PLUS_DI (warmed up)
        start = time.perf_counter()
        _ = plus_di(high, low, close)
        our_time = time.perf_counter() - start

        # TA-Lib
        start = time.perf_counter()
        _ = talib.PLUS_DI(high, low, close)
        talib_time = time.perf_counter() - start

        print(f"\nPLUS_DI Performance ({len(high):,} rows of crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Ratio: {our_time / talib_time:.2f}x")

        # Use dynamic threshold based on complexity (Wilder's smoothing)
        threshold = performance_threshold("wilders")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )
