"""
TA-Lib P1 Indicator Validation Tests.

P1 Priority: High-priority directional and momentum indicators
- ADX family (6 indicators)
- Aroon family (2 indicators)
- Stochastic family (2 indicators)
- Oscillators (2 indicators)

Total: 12 indicators

Tests verify exact numerical accuracy against TA-Lib reference implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

talib = pytest.importorskip("talib")

from ml4t.engineer.features.momentum.adxr import adxr
from ml4t.engineer.features.momentum.apo import apo
from ml4t.engineer.features.momentum.aroon import aroon, aroonosc
from ml4t.engineer.features.momentum.directional import dx, minus_di, plus_di
from ml4t.engineer.features.momentum.minus_dm import minus_dm
from ml4t.engineer.features.momentum.plus_dm import plus_dm
from ml4t.engineer.features.momentum.ppo import ppo
from ml4t.engineer.features.momentum.stochf import stochf
from ml4t.engineer.features.momentum.stochrsi import stochrsi


@pytest.fixture
def price_data():
    """Generate realistic price data for testing."""
    np.random.seed(42)
    n = 5000

    # Generate correlated OHLCV data
    close = 100 + np.random.randn(n).cumsum()
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_ = close + np.random.randn(n) * 0.5
    volume = np.abs(np.random.randn(n)) * 1000000 + 1000000

    return {
        "high": high,
        "low": low,
        "close": close,
        "open": open_,
        "volume": volume,
    }


class TestP1IndicatorAccuracy:
    """Test P1 indicators match TA-Lib exactly."""

    # ==================== DX Tests ====================

    def test_dx_accuracy(self, price_data):
        """Test Directional Movement Index matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [10, 14, 20]:
            # TA-Lib reference
            expected = talib.DX(high, low, close, timeperiod=period)

            # Our implementation
            result = dx(high, low, close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"DX mismatch for period={period}",
            )

    # ==================== ADXR Tests ====================

    def test_adxr_accuracy(self, price_data):
        """Test Average Directional Movement Index Rating matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [10, 14, 20]:
            # TA-Lib reference
            expected = talib.ADXR(high, low, close, timeperiod=period)

            # Our implementation
            result = adxr(high, low, close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"ADXR mismatch for period={period}",
            )

    # ==================== Plus DI Tests ====================

    def test_plus_di_accuracy(self, price_data):
        """Test Plus Directional Indicator matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [10, 14, 20]:
            # TA-Lib reference
            expected = talib.PLUS_DI(high, low, close, timeperiod=period)

            # Our implementation
            result = plus_di(high, low, close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"PLUS_DI mismatch for period={period}",
            )

    # ==================== Minus DI Tests ====================

    def test_minus_di_accuracy(self, price_data):
        """Test Minus Directional Indicator matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [10, 14, 20]:
            # TA-Lib reference
            expected = talib.MINUS_DI(high, low, close, timeperiod=period)

            # Our implementation
            result = minus_di(high, low, close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"MINUS_DI mismatch for period={period}",
            )

    # ==================== Plus DM Tests ====================

    def test_plus_dm_accuracy(self, price_data):
        """Test Plus Directional Movement matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]

        for period in [10, 14, 20]:
            # TA-Lib reference
            expected = talib.PLUS_DM(high, low, timeperiod=period)

            # Our implementation
            result = plus_dm(high, low, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"PLUS_DM mismatch for period={period}",
            )

    # ==================== Minus DM Tests ====================

    def test_minus_dm_accuracy(self, price_data):
        """Test Minus Directional Movement matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]

        for period in [10, 14, 20]:
            # TA-Lib reference
            expected = talib.MINUS_DM(high, low, timeperiod=period)

            # Our implementation
            result = minus_dm(high, low, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"MINUS_DM mismatch for period={period}",
            )

    # ==================== Aroon Tests ====================

    def test_aroon_accuracy(self, price_data):
        """Test Aroon matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]

        for period in [14, 25]:
            # TA-Lib reference
            expected_down, expected_up = talib.AROON(high, low, timeperiod=period)

            # Our implementation
            result_down, result_up = aroon(high, low, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result_down,
                expected_down,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"AROON down mismatch for period={period}",
            )
            assert_allclose(
                result_up,
                expected_up,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"AROON up mismatch for period={period}",
            )

    # ==================== Aroon Oscillator Tests ====================

    def test_aroonosc_accuracy(self, price_data):
        """Test Aroon Oscillator matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]

        for period in [14, 25]:
            # TA-Lib reference
            expected = talib.AROONOSC(high, low, timeperiod=period)

            # Our implementation
            result = aroonosc(high, low, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"AROONOSC mismatch for period={period}",
            )

    # ==================== Stochastic Fast Tests ====================

    def test_stochf_accuracy(self, price_data):
        """Test Stochastic Fast matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for fastk_period in [5, 14]:
            for fastd_period in [3, 5]:
                # TA-Lib reference
                expected_k, expected_d = talib.STOCHF(
                    high,
                    low,
                    close,
                    fastk_period=fastk_period,
                    fastd_period=fastd_period,
                    fastd_matype=0,  # SMA
                )

                # Our implementation
                result_k, result_d = stochf(
                    high,
                    low,
                    close,
                    fastk_period=fastk_period,
                    fastd_period=fastd_period,
                    fastd_matype=0,  # SMA
                    return_pair=True,
                )

                # Check accuracy
                assert_allclose(
                    result_k,
                    expected_k,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"STOCHF %K mismatch for fastk_period={fastk_period}, fastd_period={fastd_period}",
                )
                assert_allclose(
                    result_d,
                    expected_d,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"STOCHF %D mismatch for fastk_period={fastk_period}, fastd_period={fastd_period}",
                )

    # ==================== Stochastic RSI Tests ====================

    def test_stochrsi_accuracy(self, price_data):
        """Test Stochastic RSI matches TA-Lib exactly."""
        close = price_data["close"]

        for timeperiod in [14, 21]:
            for fastk_period in [5, 14]:
                # TA-Lib reference
                expected_k, expected_d = talib.STOCHRSI(
                    close,
                    timeperiod=timeperiod,
                    fastk_period=fastk_period,
                    fastd_period=3,
                    fastd_matype=0,  # SMA
                )

                # Our implementation
                result_k, result_d = stochrsi(
                    close,
                    timeperiod=timeperiod,
                    fastk_period=fastk_period,
                    fastd_period=3,
                    fastd_matype=0,  # SMA
                    return_pair=True,
                )

                # Check accuracy
                assert_allclose(
                    result_k,
                    expected_k,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"STOCHRSI %K mismatch for timeperiod={timeperiod}, fastk_period={fastk_period}",
                )
                assert_allclose(
                    result_d,
                    expected_d,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"STOCHRSI %D mismatch for timeperiod={timeperiod}, fastk_period={fastk_period}",
                )

    # ==================== APO Tests ====================

    def test_apo_accuracy(self, price_data):
        """Test Absolute Price Oscillator matches TA-Lib exactly."""
        close = price_data["close"]

        for fast in [12, 10]:
            for slow in [26, 30]:
                # TA-Lib reference
                expected = talib.APO(close, fastperiod=fast, slowperiod=slow, matype=0)

                # Our implementation
                result = apo(close, fast_period=fast, slow_period=slow, matype=0)

                # Check accuracy
                assert_allclose(
                    result,
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"APO mismatch for fastperiod={fast}, slowperiod={slow}",
                )

    # ==================== PPO Tests ====================

    def test_ppo_accuracy(self, price_data):
        """Test Percentage Price Oscillator matches TA-Lib exactly."""
        close = price_data["close"]

        for fast in [12, 10]:
            for slow in [26, 30]:
                # TA-Lib reference
                expected = talib.PPO(close, fastperiod=fast, slowperiod=slow, matype=0)

                # Our implementation
                result = ppo(close, fast_period=fast, slow_period=slow, matype=0)

                # Check accuracy
                assert_allclose(
                    result,
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"PPO mismatch for fastperiod={fast}, slowperiod={slow}",
                )
