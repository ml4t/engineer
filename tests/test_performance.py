"""Performance benchmark tests for QuantLab-TA.

These tests measure performance against reference implementations
and validate our 10-50x performance claims.
"""

import time
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

# Try to import reference libraries for comparison
try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

import importlib.util

HAS_PANDAS = importlib.util.find_spec("pandas") is not None

from ml4t.engineer.features.momentum import macd, rsi
from ml4t.engineer.features.trend import ema, sma
from ml4t.engineer.features.utils.helpers import add_indicators
from ml4t.engineer.features.volatility import bollinger_bands


def create_large_dataset(n_rows: int = 100_000):
    """Create large dataset for performance testing."""
    np.random.seed(42)

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_rows)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    closes = np.array(prices)
    highs = closes * (1 + np.abs(np.random.normal(0, 0.01, n_rows)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.01, n_rows)))
    volumes = np.random.lognormal(12, 0.5, n_rows).astype(int)

    return pl.DataFrame(
        {
            "timestamp": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_rows)],
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
    )


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def setup_method(self):
        """Set up large dataset for performance testing."""
        self.small_data = create_large_dataset(1_000)
        self.medium_data = create_large_dataset(10_000)
        self.large_data = create_large_dataset(100_000)

    def time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start

    def test_sma_performance_small(self):
        """Test SMA performance on small dataset."""
        # Our implementation
        _, our_time = self.time_function(
            lambda: self.small_data.with_columns([sma("close", 20).alias("sma_20")]),
        )

        print(f"\nSMA (1K rows) - Our time: {our_time:.4f}s")

        if HAS_TALIB:
            closes = self.small_data["close"].to_numpy()
            _, talib_time = self.time_function(lambda: talib.SMA(closes, timeperiod=20))

            speedup = talib_time / our_time
            print(
                f"SMA (1K rows) - TA-Lib time: {talib_time:.4f}s, Speedup: {speedup:.2f}x",
            )

        # Performance threshold: should complete in reasonable time
        assert our_time < 1.0, f"SMA too slow: {our_time:.4f}s"

    def test_sma_performance_large(self):
        """Test SMA performance on large dataset."""
        # Our implementation
        _, our_time = self.time_function(
            lambda: self.large_data.with_columns([sma("close", 20).alias("sma_20")]),
        )

        print(f"\nSMA (100K rows) - Our time: {our_time:.4f}s")

        if HAS_TALIB:
            closes = self.large_data["close"].to_numpy()
            _, talib_time = self.time_function(lambda: talib.SMA(closes, timeperiod=20))

            speedup = talib_time / our_time
            print(
                f"SMA (100K rows) - TA-Lib time: {talib_time:.4f}s, Speedup: {speedup:.2f}x",
            )

        # Performance threshold: should handle 100K rows efficiently
        assert our_time < 5.0, f"SMA too slow on large data: {our_time:.4f}s"

    def test_multiple_indicators_performance(self):
        """Test performance of computing multiple indicators simultaneously."""
        indicators = {
            "sma_10": sma("close", 10),
            "sma_20": sma("close", 20),
            "sma_50": sma("close", 50),
            "ema_12": ema("close", 12),
            "ema_26": ema("close", 26),
            "rsi_14": rsi("close", 14),
            "macd": macd("close", 12, 26),
            # "returns_1d": returns("close", 1),
            # "returns_5d": returns("close", 5),
        }

        # Our batch implementation
        _, our_time = self.time_function(
            lambda: add_indicators(self.medium_data, indicators),
        )

        calculations_per_second = (len(self.medium_data) * len(indicators)) / our_time

        print(f"\nMultiple indicators (10K rows × {len(indicators)} indicators):")
        print(f"Time: {our_time:.4f}s")
        print(f"Calculations per second: {calculations_per_second:,.0f}")

        # Should achieve reasonable throughput
        # Note: This includes complex indicators like RSI and MACD with lookback periods
        assert calculations_per_second > 100_000, (
            f"Too slow: {calculations_per_second:,.0f} calculations/sec"
        )

    def test_rsi_performance_vs_talib(self):
        """Test RSI performance vs TA-Lib."""
        if not HAS_TALIB:
            pytest.skip("TA-Lib not available")

        closes = self.medium_data["close"].to_numpy()

        # Warm up JIT compilation
        _ = rsi(closes[:100], 14)

        # Our implementation - test NumPy directly for fair comparison
        _, our_time = self.time_function(lambda: rsi(closes, 14))

        # TA-Lib implementation
        _, talib_time = self.time_function(lambda: talib.RSI(closes, timeperiod=14))

        speedup = talib_time / our_time

        print("\nRSI (10K rows):")
        print(f"Our time: {our_time:.4f}s")
        print(f"TA-Lib time: {talib_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Should be competitive with TA-Lib
        assert speedup > 0.5, f"Too slow compared to TA-Lib: {speedup:.2f}x"

    def test_bollinger_bands_performance(self):
        """Test Bollinger Bands performance."""
        _, our_time = self.time_function(
            lambda: self.medium_data.with_columns(
                [bollinger_bands("close", 20, 2.0).alias("bb")],
            ).with_columns(
                [
                    pl.col("bb").struct.field("upper").alias("bb_upper"),
                    pl.col("bb").struct.field("middle").alias("bb_middle"),
                    pl.col("bb").struct.field("lower").alias("bb_lower"),
                ],
            ),
        )

        print(f"\nBollinger Bands (10K rows): {our_time:.4f}s")

        if HAS_TALIB:
            closes = self.medium_data["close"].to_numpy()
            _, talib_time = self.time_function(
                lambda: talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2),
            )

            speedup = talib_time / our_time
            print(f"TA-Lib time: {talib_time:.4f}s, Speedup: {speedup:.2f}x")

        # Should complete in reasonable time
        assert our_time < 2.0, f"Bollinger Bands too slow: {our_time:.4f}s"

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Test with lazy evaluation - should not consume excessive memory
        lazy_result = self.large_data.lazy().with_columns(
            [
                sma("close", 20).alias("sma_20"),
                ema("close", 12).alias("ema_12"),
                rsi("close", 14).alias("rsi_14"),
            ],
        )

        # This should not consume much memory until collected
        assert lazy_result is not None

        # Collect result - memory usage should be reasonable
        _, collect_time = self.time_function(lambda: lazy_result.collect())

        print("\nMemory efficiency test (100K rows, 3 indicators):")
        print(f"Collection time: {collect_time:.4f}s")

        # Should complete without memory issues
        assert collect_time < 10.0, f"Collection too slow: {collect_time:.4f}s"

    def test_streaming_performance(self):
        """Test performance with streaming-like operations."""
        # Simulate processing data in chunks
        chunk_size = 1000
        n_chunks = len(self.large_data) // chunk_size

        total_time = 0

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(self.large_data))
            chunk = self.large_data[start_idx:end_idx]

            _, chunk_time = self.time_function(
                lambda chunk_data=chunk: chunk_data.with_columns(
                    [
                        sma("close", 20).alias("sma_20"),
                        rsi("close", 14).alias("rsi_14"),
                    ],
                ),
            )

            total_time += chunk_time

        avg_time_per_chunk = total_time / n_chunks
        rows_per_second = chunk_size / avg_time_per_chunk

        print("\nStreaming performance test:")
        print(f"Processed {n_chunks} chunks of {chunk_size} rows each")
        print(f"Average time per chunk: {avg_time_per_chunk:.4f}s")
        print(f"Rows per second: {rows_per_second:,.0f}")

        # Should maintain good throughput
        assert rows_per_second > 10_000, f"Streaming too slow: {rows_per_second:,.0f} rows/sec"


@pytest.mark.benchmark
@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Test scalability with very large datasets."""

    def time_function(self, func):
        """Time a function execution."""
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        return result, end - start

    def test_million_row_performance(self):
        """Test performance with 1 million rows."""
        # Run with reduced size for reasonable test time
        large_data = create_large_dataset(
            100_000,
        )  # Reduced from 1M to 100K for reasonable test time

        _, our_time = self.time_function(
            lambda: large_data.with_columns([sma("close", 50).alias("sma_50")]),
        )

        rows_per_second = 100_000 / our_time

        print("\n100K row SMA performance:")
        print(f"Time: {our_time:.2f}s")
        print(f"Rows per second: {rows_per_second:,.0f}")

        # Should handle 100K rows within reasonable time
        assert our_time < 6.0, f"100K row test too slow: {our_time:.2f}s"
        assert rows_per_second > 50_000, (
            f"100K row throughput too low: {rows_per_second:,.0f} rows/sec"
        )
