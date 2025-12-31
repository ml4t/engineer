"""
Comprehensive accuracy validation to ensure all indicators are within 1e-6 tolerance.
"""

import timeit

import numpy as np
import pytest

talib = pytest.importorskip("talib")

from ml4t.engineer.features.momentum.rsi import rsi  # noqa: E402
from ml4t.engineer.features.statistics.stddev import stddev  # noqa: E402
from ml4t.engineer.features.trend.ema import ema  # noqa: E402
from ml4t.engineer.features.trend.sma import sma  # noqa: E402
from ml4t.engineer.features.volatility.atr import atr  # noqa: E402


def test_rsi_accuracy_detailed():
    """Investigate RSI accuracy issues."""
    np.random.seed(42)
    n = 1000

    # Test with different data patterns
    test_cases = {
        "random_walk": np.random.randn(n).cumsum() + 100,
        "trending_up": np.linspace(100, 200, n),
        "trending_down": np.linspace(200, 100, n),
        "sine_wave": 100 + 10 * np.sin(np.linspace(0, 4 * np.pi, n)),
        "constant": np.full(n, 100.0),
    }

    results = []
    for name, data in test_cases.items():
        # TA-Lib RSI
        talib_result = talib.RSI(data, timeperiod=14)

        # Our RSI
        our_result = rsi(data, period=14)

        # Compare where both have values
        mask = ~(np.isnan(talib_result) | np.isnan(our_result))
        if np.any(mask):
            diff = np.abs(talib_result[mask] - our_result[mask])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            # Find where differences occur
            if max_diff > 1e-6:
                worst_idx = np.argmax(diff)
                actual_idx = np.where(mask)[0][worst_idx]

                results.append(
                    {
                        "case": name,
                        "max_diff": max_diff,
                        "mean_diff": mean_diff,
                        "worst_idx": actual_idx,
                        "talib_val": talib_result[actual_idx],
                        "our_val": our_result[actual_idx],
                        "status": "❌" if max_diff > 1e-6 else "✅",
                    },
                )
            else:
                results.append(
                    {
                        "case": name,
                        "max_diff": max_diff,
                        "mean_diff": mean_diff,
                        "status": "✅",
                    },
                )

    # Assert all tests pass
    for result in results:
        assert result["status"] == "✅", (
            f"RSI accuracy failed for {result['case']}: max_diff={result['max_diff']}"
        )


def validate_all_indicators():
    """Validate all indicators meet 1e-6 tolerance."""
    np.random.seed(42)
    n = 10000

    # Generate test data
    returns = np.random.normal(0.0001, 0.02, n)
    close = 100 * (1 + returns).cumprod()
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))

    validations = []

    # SMA
    talib_sma = talib.SMA(close, timeperiod=20)
    our_sma = sma(close, period=20)
    mask = ~(np.isnan(talib_sma) | np.isnan(our_sma))
    max_diff = np.max(np.abs(talib_sma[mask] - our_sma[mask]))
    validations.append(("SMA", max_diff, max_diff < 1e-6))

    # EMA
    talib_ema = talib.EMA(close, timeperiod=20)
    our_ema = ema(close, period=20, initialization="sma", adjust=False)
    mask = ~(np.isnan(talib_ema) | np.isnan(our_ema))
    max_diff = np.max(np.abs(talib_ema[mask] - our_ema[mask]))
    validations.append(("EMA", max_diff, max_diff < 1e-6))

    # STDDEV
    talib_std = talib.STDDEV(close, timeperiod=20, nbdev=1)
    our_std = stddev(close, period=20, nbdev=1.0, ddof=0)
    mask = ~(np.isnan(talib_std) | np.isnan(our_std))
    max_diff = np.max(np.abs(talib_std[mask] - our_std[mask]))
    validations.append(("STDDEV", max_diff, max_diff < 1e-6))

    # RSI
    talib_rsi = talib.RSI(close, timeperiod=14)
    our_rsi = rsi(close, period=14)
    mask = ~(np.isnan(talib_rsi) | np.isnan(our_rsi))
    max_diff = np.max(np.abs(talib_rsi[mask] - our_rsi[mask]))
    validations.append(("RSI", max_diff, max_diff < 1e-6))

    # ATR
    talib_atr = talib.ATR(high, low, close, timeperiod=14)
    our_atr = atr(high, low, close, period=14)
    mask = ~(np.isnan(talib_atr) | np.isnan(our_atr))
    max_diff = np.max(np.abs(talib_atr[mask] - our_atr[mask]))
    validations.append(("ATR", max_diff, max_diff < 1e-6))

    return validations


def benchmark_with_timeit():
    """Use timeit for accurate performance measurement."""
    np.random.seed(42)
    n = 100000

    # Generate test data
    returns = np.random.normal(0.0001, 0.02, n)
    close = 100 * (1 + returns).cumprod()
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))

    # Warm up Numba functions
    _ = sma(close[:100], period=20)
    _ = ema(close[:100], period=20)
    _ = stddev(close[:100], period=20)
    _ = rsi(close[:100], period=14)
    _ = atr(high[:100], low[:100], close[:100], period=14)

    benchmarks = []

    # SMA
    talib_time = timeit.timeit(lambda: talib.SMA(close, timeperiod=20), number=100) / 100
    our_time = timeit.timeit(lambda: sma(close, period=20), number=100) / 100
    benchmarks.append(
        ("SMA", talib_time * 1000, our_time * 1000, talib_time / our_time),
    )

    # EMA
    talib_time = timeit.timeit(lambda: talib.EMA(close, timeperiod=20), number=100) / 100
    our_time = (
        timeit.timeit(
            lambda: ema(close, period=20, initialization="sma", adjust=False),
            number=100,
        )
        / 100
    )
    benchmarks.append(
        ("EMA", talib_time * 1000, our_time * 1000, talib_time / our_time),
    )

    # STDDEV
    talib_time = (
        timeit.timeit(lambda: talib.STDDEV(close, timeperiod=20, nbdev=1), number=100) / 100
    )
    our_time = timeit.timeit(lambda: stddev(close, period=20, nbdev=1.0, ddof=0), number=100) / 100
    benchmarks.append(
        ("STDDEV", talib_time * 1000, our_time * 1000, talib_time / our_time),
    )

    # RSI
    talib_time = timeit.timeit(lambda: talib.RSI(close, timeperiod=14), number=100) / 100
    our_time = timeit.timeit(lambda: rsi(close, period=14), number=100) / 100
    benchmarks.append(
        ("RSI", talib_time * 1000, our_time * 1000, talib_time / our_time),
    )

    # ATR
    talib_time = timeit.timeit(lambda: talib.ATR(high, low, close, timeperiod=14), number=100) / 100
    our_time = timeit.timeit(lambda: atr(high, low, close, period=14), number=100) / 100
    benchmarks.append(
        ("ATR", talib_time * 1000, our_time * 1000, talib_time / our_time),
    )

    return benchmarks


if __name__ == "__main__":
    print("=" * 80)
    print("ACCURACY VALIDATION REPORT")
    print("=" * 80)

    # Check RSI issues
    print("\n1. RSI Detailed Analysis:")
    print("-" * 40)
    rsi_results = test_rsi_accuracy_detailed()
    for result in rsi_results:
        print(
            f"{result['case']:<15} Max Diff: {result['max_diff']:.2e} {result['status']}",
        )
        if "worst_idx" in result:
            print(
                f"  Worst at idx {result['worst_idx']}: TA-Lib={result['talib_val']:.6f}, Ours={result['our_val']:.6f}",
            )

    # Validate all indicators
    print("\n2. All Indicators Validation (1e-6 tolerance):")
    print("-" * 40)
    validations = validate_all_indicators()
    all_pass = True
    for name, max_diff, passes in validations:
        status = "✅" if passes else "❌"
        print(f"{name:<10} Max Diff: {max_diff:.2e} {status}")
        if not passes:
            all_pass = False

    print(
        f"\nOverall: {'✅ All indicators pass' if all_pass else '❌ Some indicators fail'}",
    )

    # Performance with timeit
    print("\n3. Performance Benchmarks (using timeit, 100 runs):")
    print("-" * 40)
    benchmarks = benchmark_with_timeit()
    for name, talib_ms, our_ms, speedup in benchmarks:
        print(
            f"{name:<10} TA-Lib: {talib_ms:6.3f}ms  QuantLab: {our_ms:6.3f}ms  Speedup: {speedup:5.2f}x",
        )
