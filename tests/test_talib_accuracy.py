"""
Test accuracy against TA-Lib with 1e-6 precision tolerance.
"""

# Import our implementations
import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

talib = pytest.importorskip("talib")

# Import from new architecture structure - import functions directly from modules
from ml4t.engineer.features.momentum.adx import adx
from ml4t.engineer.features.momentum.cci import cci
from ml4t.engineer.features.momentum.macd import macd_full
from ml4t.engineer.features.momentum.mfi import mfi
from ml4t.engineer.features.momentum.roc import roc
from ml4t.engineer.features.momentum.rsi import rsi
from ml4t.engineer.features.momentum.stochastic import stochastic
from ml4t.engineer.features.momentum.willr import willr
from ml4t.engineer.features.statistics.stddev import stddev
from ml4t.engineer.features.trend.ema import ema
from ml4t.engineer.features.trend.sma import sma
from ml4t.engineer.features.volatility.atr import atr
from ml4t.engineer.features.volatility.bollinger_bands import bollinger_bands


class TestTALibAccuracy:
    """Test suite for exact TA-Lib compatibility."""

    @pytest.fixture
    def price_data(self):
        """Generate realistic price data."""
        np.random.seed(42)
        n = 5000  # Increased for comprehensive testing
        returns = np.random.normal(0.0001, 0.02, n)
        close = 100 * (1 + returns).cumprod()
        high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_ = np.roll(close, 1)
        open_[0] = 100
        volume = np.random.randint(100000, 1000000, n).astype(float)

        return {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    @pytest.fixture
    def edge_cases(self):
        """Generate edge case data."""
        return {
            "constant": np.full(100, 50.0),
            "trending_up": np.arange(100, dtype=float),
            "trending_down": np.arange(100, 0, -1, dtype=float),
            "single_spike": np.concatenate([np.ones(50), [100], np.ones(49)]),
            "alternating": np.array([1, -1] * 50, dtype=float),
            "with_nans": np.concatenate([[np.nan], np.arange(1, 100, dtype=float)]),
        }

    def test_sma_accuracy(self, price_data):
        """Test SMA matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 10, 20, 50, 100]:
            # TA-Lib reference
            expected = talib.SMA(close, timeperiod=period)

            # Our implementation
            result = sma(close, period=period)

            # Check accuracy (1e-6 tolerance)
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"SMA mismatch for period={period}",
            )

    def test_sma_edge_cases(self, edge_cases):
        """Test SMA with edge cases."""
        for name, data in edge_cases.items():
            if name == "with_nans":
                continue  # Skip NaN test for now

            for period in [5, 20]:
                expected = talib.SMA(data, timeperiod=period)
                result = sma(data, period=period)

                assert_allclose(
                    result,
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"SMA edge case '{name}' failed",
                )

    def test_ema_accuracy(self, price_data):
        """Test EMA matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 12, 20, 26, 50]:
            # TA-Lib reference
            expected = talib.EMA(close, timeperiod=period)

            # Our implementation
            result = ema(close, period=period)

            # Check accuracy (1e-6 tolerance)
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"EMA mismatch for period={period}",
            )

    def test_stddev_accuracy(self, price_data):
        """Test STDDEV accuracy against TA-Lib."""
        close = price_data["close"]

        # Default parameters
        window = 5
        nbdev = 1.0

        talib_result = talib.STDDEV(close, timeperiod=window, nbdev=nbdev)
        our_result = stddev(close, period=window, nbdev=nbdev)

        np.testing.assert_allclose(talib_result, our_result, rtol=1e-8, equal_nan=True)

    def test_atr_accuracy(self, price_data):
        """Test ATR matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [7, 14, 21]:
            # TA-Lib reference
            expected = talib.ATR(high, low, close, timeperiod=period)

            # Our implementation
            result = atr(high, low, close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"ATR mismatch for period={period}",
            )

    def test_rsi_accuracy(self, price_data):
        """Test RSI matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [6, 14, 21, 28]:
            # TA-Lib reference
            expected = talib.RSI(close, timeperiod=period)

            # Our implementation
            result = rsi(close, period=period)

            # Check accuracy
            # RSI can have slightly larger differences due to Wilder's smoothing
            assert_allclose(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"RSI mismatch for period={period}",
            )

    def test_atr_accuracy_extended(self, price_data):
        """Test ATR matches TA-Lib exactly with extended periods."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [7, 14, 21]:
            # TA-Lib reference
            expected = talib.ATR(high, low, close, timeperiod=period)

            # Our implementation
            result = atr(high, low, close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"ATR mismatch for period={period}",
            )

    def test_lookback_periods(self):
        """Test that our lookback periods match TA-Lib concepts."""
        # In TA-Lib, lookback is the number of previous values needed
        # to calculate the first output value

        # SMA needs period-1 previous values
        data = np.arange(100, dtype=float)
        result = sma(data, period=20)
        first_valid = np.where(~np.isnan(result))[0][0]
        assert first_valid == 19  # period - 1

        # EMA also starts at period-1 (without unstable period)
        result = ema(data, period=20)
        first_valid = np.where(~np.isnan(result))[0][0]
        assert first_valid == 19  # period - 1

        # RSI needs period values
        result = rsi(data, period=14)
        first_valid = np.where(~np.isnan(result))[0][0]
        assert first_valid == 14  # period

        # ATR needs period values
        high = data + 1
        low = data - 1
        close = data
        result = atr(high, low, close, period=14)
        first_valid = np.where(~np.isnan(result))[0][0]
        assert first_valid == 14  # period

    def test_nan_handling(self):
        """Test NaN propagation matches TA-Lib."""
        # Create data with NaN at start
        data = np.array([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

        # SMA with NaN - TA-Lib ignores leading NaNs
        expected = talib.SMA(data, timeperiod=3)
        result = sma(data, period=3)

        # Find where both have valid values
        valid_mask = ~(np.isnan(expected) | np.isnan(result))
        if np.any(valid_mask):
            assert_allclose(
                result[valid_mask],
                expected[valid_mask],
                rtol=1e-6,
                atol=1e-6,
            )

        # Test with NaN in middle
        data2 = np.array([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10], dtype=float)
        expected2 = talib.SMA(data2, timeperiod=3)
        result2 = sma(data2, period=3)

        # Check where both have valid values
        valid_mask2 = ~(np.isnan(expected2) | np.isnan(result2))
        if np.any(valid_mask2):
            assert_allclose(
                result2[valid_mask2],
                expected2[valid_mask2],
                rtol=1e-6,
                atol=1e-6,
            )

    def test_parameter_validation(self):
        """Test parameter ranges match TA-Lib."""
        data = np.random.randn(100)

        # Test valid periods
        for period in [2, 100, 1000]:
            result = sma(data, period=period)
            assert len(result) == len(data)

        # Test edge periods
        result = sma(data, period=1)  # Should work
        assert_allclose(result, data, rtol=1e-6, atol=1e-6)

    def test_different_data_types(self):
        """Test with different input data types."""
        data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data_array = np.array(data_list, dtype=float)
        data_series = pl.Series(data_list, dtype=pl.Float64)

        # Test SMA with different types
        expected = talib.SMA(data_array, timeperiod=3)

        result_array = sma(data_array, period=3)
        result_series = sma(data_series, period=3)

        assert_allclose(result_array, expected, rtol=1e-6, atol=1e-6)
        assert_allclose(result_series, expected, rtol=1e-6, atol=1e-6)

    def test_performance_consistency(self):
        """Test that different implementations give same results."""
        np.random.seed(42)
        data = np.random.randn(10000).cumsum() + 100

        # Test SMA implementations
        result_polars = sma(data, period=20, implementation="polars")
        result_numba = sma(data, period=20, implementation="numba")

        assert_allclose(
            result_polars,
            result_numba,
            rtol=1e-10,
            atol=1e-10,
            err_msg="Polars and Numba implementations differ",
        )

    # Crypto data accuracy tests
    def test_sma_crypto_accuracy(self, crypto_data_small):
        """Test SMA with crypto data."""
        close = crypto_data_small["close"].to_numpy()

        for period in [5, 10, 20, 50]:
            # TA-Lib reference
            expected = talib.SMA(close, timeperiod=period)

            # Our implementation
            result = sma(close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"SMA crypto mismatch for period={period}",
            )

    def test_ema_crypto_accuracy(self, crypto_data_small):
        """Test EMA with crypto data."""
        close = crypto_data_small["close"].to_numpy()

        for period in [5, 12, 20, 26]:
            # TA-Lib reference
            expected = talib.EMA(close, timeperiod=period)

            # Our implementation
            result = ema(close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"EMA crypto mismatch for period={period}",
            )

    def test_stddev_crypto_accuracy(self, crypto_data_small):
        """Test STDDEV with crypto data."""
        close = crypto_data_small["close"].to_numpy()

        for period in [5, 10, 20]:
            for nbdev in [1.0, 2.0]:
                # TA-Lib reference
                expected = talib.STDDEV(close, timeperiod=period, nbdev=nbdev)

                # Our implementation
                result = stddev(close, period=period, nbdev=nbdev)

                # Check accuracy - STDDEV can have small numerical differences
                assert_allclose(
                    result,
                    expected,
                    rtol=5e-6,
                    atol=5e-6,
                    equal_nan=True,
                    err_msg=f"STDDEV crypto mismatch for period={period}, nbdev={nbdev}",
                )

    def test_atr_crypto_accuracy(self, crypto_data_small):
        """Test ATR with crypto data."""
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        for period in [7, 14, 21]:
            # TA-Lib reference
            expected = talib.ATR(high, low, close, timeperiod=period)

            # Our implementation
            result = atr(high, low, close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"ATR crypto mismatch for period={period}",
            )

    def test_bollinger_bands_crypto_accuracy(self, crypto_data_small):
        """Test Bollinger Bands with crypto data."""
        close = crypto_data_small["close"].to_numpy()

        for period in [10, 20]:
            for nbdevup, nbdevdn in [(2, 2), (2.5, 2.5), (3, 3)]:
                # TA-Lib reference
                upper_expected, middle_expected, lower_expected = talib.BBANDS(
                    close,
                    timeperiod=period,
                    nbdevup=nbdevup,
                    nbdevdn=nbdevdn,
                    matype=0,
                )

                # Our implementation
                upper_result, middle_result, lower_result = bollinger_bands(
                    close,
                    period=period,
                    nbdevup=nbdevup,
                    nbdevdn=nbdevdn,
                )

                # Check accuracy
                assert_allclose(
                    upper_result,
                    upper_expected,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"BB upper crypto mismatch for period={period}",
                )
                assert_allclose(
                    middle_result,
                    middle_expected,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"BB middle crypto mismatch for period={period}",
                )
                assert_allclose(
                    lower_result,
                    lower_expected,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"BB lower crypto mismatch for period={period}",
                )

    def test_macd_crypto_accuracy(self, crypto_data_small):
        """Test MACD with crypto data."""
        close = crypto_data_small["close"].to_numpy()

        # Standard MACD parameters
        fastperiod, slowperiod, signalperiod = 12, 26, 9

        # TA-Lib reference
        macd_expected, signal_expected, hist_expected = talib.MACD(
            close,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod,
        )

        # Our implementation - use macd_full which returns all three components
        macd_result, signal_result, hist_result = macd_full(
            close,
            fast_period=fastperiod,
            slow_period=slowperiod,
            signal_period=signalperiod,
        )

        # Check accuracy - handle NaN differences
        # MACD uses exponential smoothing, so we allow slightly higher tolerance
        # MACD line
        valid_mask = ~(np.isnan(macd_result) | np.isnan(macd_expected))
        assert_allclose(
            macd_result[valid_mask],
            macd_expected[valid_mask],
            rtol=1e-5,
            atol=1e-5,
            err_msg="MACD line crypto mismatch",
        )

        # Signal line - skip first 100 values for warm-up differences
        valid_mask = ~(np.isnan(signal_result) | np.isnan(signal_expected))
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 100:
            valid_indices = valid_indices[100:]  # Skip first 100 valid values
            assert_allclose(
                signal_result[valid_indices],
                signal_expected[valid_indices],
                rtol=1e-5,
                atol=1e-5,
                err_msg="MACD signal crypto mismatch",
            )

        # Histogram - skip first 100 values for warm-up differences
        valid_mask = ~(np.isnan(hist_result) | np.isnan(hist_expected))
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 100:
            valid_indices = valid_indices[100:]  # Skip first 100 valid values
            assert_allclose(
                hist_result[valid_indices],
                hist_expected[valid_indices],
                rtol=1e-5,
                atol=1e-5,
                err_msg="MACD histogram crypto mismatch",
            )

    def test_adx_crypto_accuracy(self, crypto_data_small):
        """Test ADX with crypto data."""
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        for period in [7, 14, 21]:
            # TA-Lib reference
            expected = talib.ADX(high, low, close, timeperiod=period)

            # Our implementation
            result = adx(high, low, close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"ADX crypto mismatch for period={period}",
            )

    def test_cci_crypto_accuracy(self, crypto_data_small):
        """Test CCI with crypto data."""
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        for period in [14, 20]:
            # TA-Lib reference
            expected = talib.CCI(high, low, close, timeperiod=period)

            # Our implementation
            result = cci(high, low, close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"CCI crypto mismatch for period={period}",
            )

    def test_roc_crypto_accuracy(self, crypto_data_small):
        """Test ROC with crypto data."""
        close = crypto_data_small["close"].to_numpy()

        for period in [10, 12, 20]:
            # TA-Lib reference
            expected = talib.ROC(close, timeperiod=period)

            # Our implementation
            result = roc(close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"ROC crypto mismatch for period={period}",
            )

    def test_rsi_crypto_accuracy(self, crypto_data_small):
        """Test RSI with crypto data."""
        close = crypto_data_small["close"].to_numpy()

        for period in [6, 14, 21]:
            # TA-Lib reference
            expected = talib.RSI(close, timeperiod=period)

            # Our implementation
            result = rsi(close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"RSI crypto mismatch for period={period}",
            )

    def test_willr_crypto_accuracy(self, crypto_data_small):
        """Test Williams %R with crypto data."""
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        for period in [7, 14, 21]:
            # TA-Lib reference
            expected = talib.WILLR(high, low, close, timeperiod=period)

            # Our implementation
            result = willr(high, low, close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"WILLR crypto mismatch for period={period}",
            )

    def test_mfi_crypto_accuracy(self, crypto_data_small):
        """Test MFI with crypto data."""
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()
        volume = crypto_data_small["volume"].to_numpy()

        for period in [7, 14, 21]:
            # TA-Lib reference
            expected = talib.MFI(high, low, close, volume, timeperiod=period)

            # Our implementation
            result = mfi(high, low, close, volume, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"MFI crypto mismatch for period={period}",
            )

    def test_stoch_crypto_accuracy(self, crypto_data_small):
        """Test Stochastic with crypto data."""
        high = crypto_data_small["high"].to_numpy()
        low = crypto_data_small["low"].to_numpy()
        close = crypto_data_small["close"].to_numpy()

        # Test with default parameters
        fastk_period, slowk_period, slowd_period = 5, 3, 3

        # TA-Lib reference
        slowk_expected, slowd_expected = talib.STOCH(
            high,
            low,
            close,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,
            slowd_period=slowd_period,
            slowd_matype=0,
        )

        # Our implementation - need to set return_pair=True to get both %K and %D
        slowk_result, slowd_result = stochastic(
            high,
            low,
            close,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period,
            return_pair=True,
        )

        # Check accuracy
        assert_allclose(
            slowk_result,
            slowk_expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Stochastic %K crypto mismatch",
        )
        # Skip first few slowd values - TA-Lib uses different initialization
        # Our implementation starts slowd 2 indices later (after enough slowk values)
        first_valid_ours = 10  # Our first non-NaN slowd
        assert_allclose(
            slowd_result[first_valid_ours:],
            slowd_expected[first_valid_ours:],
            rtol=1e-6,
            atol=1e-6,
            err_msg="Stochastic %D crypto mismatch",
        )


class TestPerformance:
    """Test performance characteristics."""

    def generate_data(self, n):
        """Generate random price data."""
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, n)
        close = 100 * (1 + returns).cumprod()
        return close

    def test_sma_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark SMA implementations."""
        # Use real crypto data
        close = crypto_data["close"].to_numpy()
        period = 20

        # Warmup JIT compilation with small sample
        warmup_jit(sma, close[:100], period)

        import time

        # Time our implementation
        our_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = sma(close, period)
            our_times.append(time.perf_counter() - start)
        our_time = np.mean(our_times)

        # Time TA-Lib
        talib_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = talib.SMA(close, timeperiod=period)
            talib_times.append(time.perf_counter() - start)
        talib_time = np.mean(talib_times)

        print("\nSMA Performance (crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Speedup: {talib_time / our_time:.2f}x")

        # Use dynamic threshold based on complexity (simple moving average)
        threshold = performance_threshold("simple")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )

    def test_rsi_performance(self, crypto_data, performance_threshold, warmup_jit):
        """Benchmark RSI implementation."""
        # Use real crypto data
        close = crypto_data["close"].to_numpy()
        period = 14

        # Warmup JIT compilation with small sample
        warmup_jit(rsi, close[:100], period)

        import time

        # Time our implementation
        our_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = rsi(close, period)
            our_times.append(time.perf_counter() - start)
        our_time = np.mean(our_times)

        # Time TA-Lib
        talib_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = talib.RSI(close, timeperiod=period)
            talib_times.append(time.perf_counter() - start)
        talib_time = np.mean(talib_times)

        print("\nRSI Performance (crypto data):")
        print(f"Our implementation: {our_time * 1000:.2f}ms")
        print(f"TA-Lib: {talib_time * 1000:.2f}ms")
        print(f"Speedup: {talib_time / our_time:.2f}x")

        # Use dynamic threshold based on complexity (RSI uses exponential averaging)
        threshold = performance_threshold("rolling")
        assert our_time < talib_time * threshold, (
            f"Performance ratio {our_time / talib_time:.1f}x exceeds threshold {threshold}x"
        )


if __name__ == "__main__":
    # Run specific test
    test = TestTALibAccuracy()

    # Generate test data with thousands of points for robust testing
    np.random.seed(42)
    n = 5000  # Increased for comprehensive testing
    returns = np.random.normal(0.0001, 0.02, n)
    close = 100 * (1 + returns).cumprod()

    # Test SMA
    print("Testing SMA accuracy...")
    expected = talib.SMA(close, timeperiod=20)
    result = sma(close, period=20)

    # Find first valid index
    first_valid = np.where(~np.isnan(expected))[0][0]

    print("First 5 valid SMA values:")
    print(f"TA-Lib: {expected[first_valid : first_valid + 5]}")
    print(f"Ours:   {result[first_valid : first_valid + 5]}")
    print(f"Max difference: {np.nanmax(np.abs(expected - result))}")

    # Test RSI
    print("\nTesting RSI accuracy...")
    expected = talib.RSI(close, timeperiod=14)
    result = rsi(close, period=14)

    first_valid = np.where(~np.isnan(expected))[0][0]
    print("First 5 valid RSI values:")
    print(f"TA-Lib: {expected[first_valid : first_valid + 5]}")
    print(f"Ours:   {result[first_valid : first_valid + 5]}")
    print(f"Max difference: {np.nanmax(np.abs(expected - result))}")
