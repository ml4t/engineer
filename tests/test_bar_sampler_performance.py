"""Performance benchmarks for bar sampler optimizations."""

import time
from datetime import datetime, timedelta

import numpy as np
import polars as pl


def generate_test_data(n_rows: int = 100_000) -> pl.DataFrame:
    """Generate synthetic test data for bar sampling."""
    np.random.seed(42)

    # Generate timestamps
    timestamps = [datetime(2024, 1, 1) + timedelta(seconds=i) for i in range(n_rows)]

    # Generate prices with random walk
    price_changes = np.random.randn(n_rows) * 0.01
    prices = 100 * np.exp(np.cumsum(price_changes))

    # Generate volumes with some variance
    volumes = np.random.lognormal(mean=np.log(1000), sigma=0.5, size=n_rows)

    # Generate random sides
    sides = np.random.choice([1, -1], size=n_rows, p=[0.52, 0.48])

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


def benchmark_volume_bar_sampler():
    """Benchmark VolumeBarSamplerVectorized performance."""
    from ml4t.engineer.bars.vectorized import VolumeBarSamplerVectorized

    print("\n=== Volume Bar Sampler Performance ===")

    # Test different data sizes
    sizes = [10_000, 50_000, 100_000, 500_000]

    for size in sizes:
        data = generate_test_data(size)
        sampler = VolumeBarSamplerVectorized(volume_per_bar=10_000)

        # Warm up JIT compilation
        _ = sampler.sample(data[:1000])

        # Benchmark
        start_time = time.time()
        result = sampler.sample(data)
        elapsed = time.time() - start_time

        rows_per_second = size / elapsed
        n_bars = len(result)

        print(f"  {size:,} rows -> {n_bars:,} bars in {elapsed:.3f}s")
        print(f"  Performance: {rows_per_second:,.0f} rows/second")


def benchmark_dollar_bar_sampler():
    """Benchmark DollarBarSamplerVectorized performance."""
    from ml4t.engineer.bars.vectorized import DollarBarSamplerVectorized

    print("\n=== Dollar Bar Sampler Performance ===")

    # Test different data sizes
    sizes = [10_000, 50_000, 100_000, 500_000]

    for size in sizes:
        data = generate_test_data(size)
        sampler = DollarBarSamplerVectorized(dollars_per_bar=1_000_000)

        # Warm up JIT compilation
        _ = sampler.sample(data[:1000])

        # Benchmark
        start_time = time.time()
        result = sampler.sample(data)
        elapsed = time.time() - start_time

        rows_per_second = size / elapsed
        n_bars = len(result)

        print(f"  {size:,} rows -> {n_bars:,} bars in {elapsed:.3f}s")
        print(f"  Performance: {rows_per_second:,.0f} rows/second")


def compare_implementations():
    """Compare old vs new implementations if available."""
    print("\n=== Performance Comparison ===")

    data = generate_test_data(100_000)

    # Test Volume Bar Sampler
    from ml4t.engineer.bars.vectorized import VolumeBarSamplerVectorized

    sampler = VolumeBarSamplerVectorized(volume_per_bar=10_000)

    # Warm up
    _ = sampler.sample(data[:1000])

    # Benchmark optimized version
    start_time = time.time()
    result = sampler.sample(data)
    optimized_time = time.time() - start_time

    print("Volume Bar Sampler (100k rows):")
    print(f"  Numba-optimized: {optimized_time:.3f}s ({100_000 / optimized_time:,.0f} rows/sec)")
    print(f"  Number of bars created: {len(result)}")

    # Test Dollar Bar Sampler
    from ml4t.engineer.bars.vectorized import DollarBarSamplerVectorized

    sampler = DollarBarSamplerVectorized(dollars_per_bar=1_000_000)

    # Warm up
    _ = sampler.sample(data[:1000])

    # Benchmark optimized version
    start_time = time.time()
    result = sampler.sample(data)
    optimized_time = time.time() - start_time

    print("\nDollar Bar Sampler (100k rows):")
    print(f"  Numba-optimized: {optimized_time:.3f}s ({100_000 / optimized_time:,.0f} rows/sec)")
    print(f"  Number of bars created: {len(result)}")


if __name__ == "__main__":
    print("🚀 Bar Sampler Performance Benchmarks")
    print("=" * 50)

    benchmark_volume_bar_sampler()
    benchmark_dollar_bar_sampler()
    compare_implementations()

    print("\n✅ Performance benchmarks complete!")
    print("\nExpected Performance Targets:")
    print("  - Small datasets (10k rows): > 50,000 rows/second")
    print("  - Medium datasets (100k rows): > 100,000 rows/second")
    print("  - Large datasets (500k rows): > 100,000 rows/second")
