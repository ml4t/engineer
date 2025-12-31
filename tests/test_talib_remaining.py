"""
TA-Lib Remaining Indicator Validation Tests.

High Priority (P0): RSI, EMA, SMA, ATR, ADX
Medium Priority (P1-P2): MFI, CCI, WILLR, STDDEV, ROC

Tests verify exact numerical accuracy against TA-Lib reference implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

talib = pytest.importorskip("talib")

from ml4t.engineer.features.momentum.adx import adx
from ml4t.engineer.features.momentum.cci import cci
from ml4t.engineer.features.momentum.mfi import mfi
from ml4t.engineer.features.momentum.roc import roc
from ml4t.engineer.features.momentum.rsi import rsi
from ml4t.engineer.features.momentum.willr import willr
from ml4t.engineer.features.statistics.stddev import stddev
from ml4t.engineer.features.trend.ema import ema
from ml4t.engineer.features.trend.sma import sma
from ml4t.engineer.features.volatility.atr import atr


@pytest.fixture
def price_data():
    """Generate random OHLCV price data for testing."""
    np.random.seed(42)
    n = 5000

    close = np.cumsum(np.random.randn(n) * 2) + 100
    high = close + np.abs(np.random.randn(n) * 2)
    low = close - np.abs(np.random.randn(n) * 2)
    open_price = close + np.random.randn(n)
    volume = np.abs(np.random.randn(n) * 1000000)

    return {
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


class TestP0RemainingIndicators:
    """Test high-priority remaining TA-Lib indicators."""

    def test_rsi_accuracy(self, price_data):
        """Test RSI matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [7, 14, 21]:
            expected = talib.RSI(close, timeperiod=period)
            result = rsi(close, period=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"RSI mismatch for period={period}",
            )

    def test_ema_accuracy(self, price_data):
        """Test EMA matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 12, 20, 50]:
            expected = talib.EMA(close, timeperiod=period)
            result = ema(close, period=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"EMA mismatch for period={period}",
            )

    def test_sma_accuracy(self, price_data):
        """Test SMA matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 20, 50, 200]:
            expected = talib.SMA(close, timeperiod=period)
            result = sma(close, period=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"SMA mismatch for period={period}",
            )

    def test_atr_accuracy(self, price_data):
        """Test ATR matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [7, 14, 21]:
            expected = talib.ATR(high, low, close, timeperiod=period)
            result = atr(high, low, close, period=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"ATR mismatch for period={period}",
            )

    def test_adx_accuracy(self, price_data):
        """Test ADX matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [7, 14, 21]:
            expected = talib.ADX(high, low, close, timeperiod=period)
            result = adx(high, low, close, period=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"ADX mismatch for period={period}",
            )


class TestP1P2RemainingIndicators:
    """Test medium-priority remaining TA-Lib indicators."""

    def test_mfi_accuracy(self, price_data):
        """Test MFI matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]
        volume = price_data["volume"]

        for period in [7, 14, 21]:
            expected = talib.MFI(high, low, close, volume, timeperiod=period)
            result = mfi(high, low, close, volume, period=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"MFI mismatch for period={period}",
            )

    def test_cci_accuracy(self, price_data):
        """Test CCI matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [14, 20]:
            expected = talib.CCI(high, low, close, timeperiod=period)
            result = cci(high, low, close, period=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"CCI mismatch for period={period}",
            )

    def test_willr_accuracy(self, price_data):
        """Test WILLR matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [7, 14, 21]:
            expected = talib.WILLR(high, low, close, timeperiod=period)
            result = willr(high, low, close, period=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"WILLR mismatch for period={period}",
            )

    def test_stddev_accuracy(self, price_data):
        """Test STDDEV matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 20]:
            expected = talib.STDDEV(close, timeperiod=period, nbdev=1)
            result = stddev(close, period=period, nbdev=1)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"STDDEV mismatch for period={period}",
            )

    def test_roc_accuracy(self, price_data):
        """Test ROC matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [10, 12, 20]:
            expected = talib.ROC(close, timeperiod=period)
            result = roc(close, period=period)

            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"ROC mismatch for period={period}",
            )
