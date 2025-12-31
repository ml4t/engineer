"""
TA-Lib Accuracy Tests for P0 Critical Indicators.

These are the most commonly used indicators that MUST match TA-Lib exactly.
Tests validate: WMA, DEMA, TEMA, TRIMA, KAMA, OBV, MOM, SAR, BOP
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

talib = pytest.importorskip("talib")

from ml4t.engineer.features.momentum.bop import bop
from ml4t.engineer.features.momentum.mom import mom
from ml4t.engineer.features.momentum.sar import sar
from ml4t.engineer.features.trend.dema import dema
from ml4t.engineer.features.trend.kama import kama
from ml4t.engineer.features.trend.tema import tema
from ml4t.engineer.features.trend.trima import trima

# Import our implementations
from ml4t.engineer.features.trend.wma import wma
from ml4t.engineer.features.volume.obv import obv


class TestP0IndicatorAccuracy:
    """Test suite for P0 critical indicators - must match TA-Lib exactly."""

    @pytest.fixture
    def price_data(self):
        """Generate realistic price data for testing."""
        np.random.seed(42)
        n = 5000
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
            "trending_up": np.arange(100, dtype=float) + 100,
            "trending_down": np.arange(100, 0, -1, dtype=float) + 100,
            "volatile": 100 + 10 * np.sin(np.linspace(0, 10 * np.pi, 100)),
        }

    # ==================== WMA Tests ====================

    def test_wma_accuracy(self, price_data):
        """Test Weighted Moving Average matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 10, 20, 50]:
            # TA-Lib reference
            expected = talib.WMA(close, timeperiod=period)

            # Our implementation
            result = wma(close, period=period)

            # Check accuracy (1e-8 for exact match)
            assert_allclose(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                err_msg=f"WMA mismatch for period={period}",
            )

    def test_wma_edge_cases(self, edge_cases):
        """Test WMA with edge cases."""
        for name, data in edge_cases.items():
            for period in [5, 20]:
                expected = talib.WMA(data, timeperiod=period)
                result = wma(data, period=period)

                assert_allclose(
                    result,
                    expected,
                    rtol=1e-8,
                    atol=1e-8,
                    err_msg=f"WMA edge case '{name}' failed for period={period}",
                )

    # ==================== DEMA Tests ====================

    def test_dema_accuracy(self, price_data):
        """Test Double Exponential Moving Average matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 10, 20, 30, 50]:
            # TA-Lib reference
            expected = talib.DEMA(close, timeperiod=period)

            # Our implementation
            result = dema(close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"DEMA mismatch for period={period}",
            )

    def test_dema_edge_cases(self, edge_cases):
        """Test DEMA with edge cases."""
        for name, data in edge_cases.items():
            expected = talib.DEMA(data, timeperiod=10)
            result = dema(data, period=10)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"DEMA edge case '{name}' failed",
            )

    # ==================== TEMA Tests ====================

    def test_tema_accuracy(self, price_data):
        """Test Triple Exponential Moving Average matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 10, 20, 30]:
            # TA-Lib reference
            expected = talib.TEMA(close, timeperiod=period)

            # Our implementation
            result = tema(close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"TEMA mismatch for period={period}",
            )

    def test_tema_edge_cases(self, edge_cases):
        """Test TEMA with edge cases."""
        for name, data in edge_cases.items():
            expected = talib.TEMA(data, timeperiod=10)
            result = tema(data, period=10)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"TEMA edge case '{name}' failed",
            )

    # ==================== TRIMA Tests ====================

    def test_trima_accuracy(self, price_data):
        """Test Triangular Moving Average matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 10, 20, 30]:
            # TA-Lib reference
            expected = talib.TRIMA(close, timeperiod=period)

            # Our implementation
            result = trima(close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                err_msg=f"TRIMA mismatch for period={period}",
            )

    # ==================== KAMA Tests ====================

    def test_kama_accuracy(self, price_data):
        """Test Kaufman Adaptive Moving Average matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [10, 20, 30]:
            # TA-Lib reference
            expected = talib.KAMA(close, timeperiod=period)

            # Our implementation
            result = kama(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"KAMA mismatch for period={period}",
            )

    # ==================== OBV Tests ====================

    def test_obv_accuracy(self, price_data):
        """Test On Balance Volume matches TA-Lib exactly."""
        close = price_data["close"]
        volume = price_data["volume"]

        # TA-Lib reference
        expected = talib.OBV(close, volume)

        # Our implementation
        result = obv(close, volume)

        # Check accuracy
        assert_allclose(
            result,
            expected,
            rtol=1e-8,
            atol=1e-8,
            err_msg="OBV mismatch",
        )

    def test_obv_edge_cases(self, edge_cases):
        """Test OBV with edge cases."""
        for name, data in edge_cases.items():
            volume = np.random.randint(1000, 10000, len(data)).astype(float)
            expected = talib.OBV(data, volume)
            result = obv(data, volume)

            assert_allclose(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                err_msg=f"OBV edge case '{name}' failed",
            )

    # ==================== MOM Tests ====================

    def test_mom_accuracy(self, price_data):
        """Test Momentum matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 10, 14, 20]:
            # TA-Lib reference
            expected = talib.MOM(close, timeperiod=period)

            # Our implementation
            result = mom(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                err_msg=f"MOM mismatch for period={period}",
            )

    def test_mom_edge_cases(self, edge_cases):
        """Test MOM with edge cases."""
        for name, data in edge_cases.items():
            expected = talib.MOM(data, timeperiod=10)
            result = mom(data, timeperiod=10)

            assert_allclose(
                result,
                expected,
                rtol=1e-8,
                atol=1e-8,
                err_msg=f"MOM edge case '{name}' failed",
            )

    # ==================== SAR Tests ====================

    def test_sar_accuracy(self, price_data):
        """Test Parabolic SAR matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]

        # Test default parameters
        expected = talib.SAR(high, low)
        result = sar(high, low)

        # Check accuracy
        assert_allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg="SAR mismatch (default params)",
        )

    def test_sar_parameters(self, price_data):
        """Test SAR with different acceleration parameters."""
        high = price_data["high"]
        low = price_data["low"]

        # Test various acceleration parameters
        for acceleration in [0.02, 0.05]:
            for maximum in [0.2, 0.3]:
                expected = talib.SAR(high, low, acceleration=acceleration, maximum=maximum)
                result = sar(high, low, acceleration=acceleration, maximum=maximum)

                assert_allclose(
                    result,
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"SAR mismatch for accel={acceleration}, max={maximum}",
                )

    # ==================== BOP Tests ====================

    def test_bop_accuracy(self, price_data):
        """Test Balance of Power matches TA-Lib exactly."""
        open_ = price_data["open"]
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.BOP(open_, high, low, close)

        # Our implementation
        result = bop(open_, high, low, close)

        # Check accuracy
        assert_allclose(
            result,
            expected,
            rtol=1e-8,
            atol=1e-8,
            err_msg="BOP mismatch",
        )

    def test_bop_edge_cases(self):
        """Test BOP with edge cases."""
        # Constant prices (should give BOP = 0)
        n = 100
        constant = np.full(n, 50.0)
        expected = talib.BOP(constant, constant, constant, constant)
        result = bop(constant, constant, constant, constant)

        assert_allclose(result, expected, rtol=1e-8, atol=1e-8)

        # All closes at high (bullish - BOP should be 1)
        high = np.arange(100, 200, dtype=float)
        low = high - 10
        open_ = high - 5
        close = high.copy()

        expected = talib.BOP(open_, high, low, close)
        result = bop(open_, high, low, close)

        assert_allclose(result, expected, rtol=1e-8, atol=1e-8)

        # All closes at low (bearish - BOP should be -1)
        close = low.copy()
        expected = talib.BOP(open_, high, low, close)
        result = bop(open_, high, low, close)

        assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


class TestP0PerformanceBenchmarks:
    """Performance benchmarks for P0 indicators vs TA-Lib."""

    @pytest.fixture
    def large_dataset(self):
        """Generate large dataset for performance testing."""
        np.random.seed(42)
        n = 100000
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

    @pytest.mark.benchmark
    def test_wma_performance(self, large_dataset, benchmark):
        """Benchmark WMA performance."""
        close = large_dataset["close"]
        period = 20

        # Warm up JIT
        _ = wma(close[:1000], period=period)

        # Benchmark our implementation
        result = benchmark(wma, close, period=period)

        # Verify correctness
        expected = talib.WMA(close, timeperiod=period)
        assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.benchmark
    def test_obv_performance(self, large_dataset, benchmark):
        """Benchmark OBV performance."""
        close = large_dataset["close"]
        volume = large_dataset["volume"]

        # Warm up JIT
        _ = obv(close[:1000], volume[:1000])

        # Benchmark our implementation
        result = benchmark(obv, close, volume)

        # Verify correctness
        expected = talib.OBV(close, volume)
        assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.benchmark
    def test_mom_performance(self, large_dataset, benchmark):
        """Benchmark MOM performance."""
        close = large_dataset["close"]
        period = 10

        # Warm up JIT
        _ = mom(close[:1000], timeperiod=period)

        # Benchmark our implementation
        result = benchmark(mom, close, timeperiod=period)

        # Verify correctness
        expected = talib.MOM(close, timeperiod=period)
        assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
