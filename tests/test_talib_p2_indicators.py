"""
TA-Lib P2 Indicator Validation Tests.

P2 Priority: Medium-priority price transforms, volatility, volume, and statistics
- Price transforms (4 indicators)
- Volatility (2 indicators)
- Volume (2 indicators)
- Statistics/Regression (5 indicators)

Total: 13 indicators

Tests verify exact numerical accuracy against TA-Lib reference implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

talib = pytest.importorskip("talib")

from ml4t.engineer.features.price_transform.avgprice import avgprice
from ml4t.engineer.features.price_transform.medprice import medprice
from ml4t.engineer.features.price_transform.typprice import typprice
from ml4t.engineer.features.price_transform.wclprice import wclprice
from ml4t.engineer.features.statistics.linearreg import linearreg
from ml4t.engineer.features.statistics.linearreg_angle import linearreg_angle
from ml4t.engineer.features.statistics.linearreg_intercept import linearreg_intercept
from ml4t.engineer.features.statistics.linearreg_slope import linearreg_slope
from ml4t.engineer.features.statistics.tsf import tsf
from ml4t.engineer.features.volatility.natr import natr
from ml4t.engineer.features.volatility.trange import trange
from ml4t.engineer.features.volume.ad import ad
from ml4t.engineer.features.volume.adosc import adosc


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


class TestP2IndicatorAccuracy:
    """Test P2 indicators match TA-Lib exactly."""

    # ==================== Price Transform Tests ====================

    def test_avgprice_accuracy(self, price_data):
        """Test Average Price matches TA-Lib exactly."""
        open_ = price_data["open"]
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.AVGPRICE(open_, high, low, close)

        # Our implementation
        result = avgprice(open_, high, low, close)

        # Check accuracy
        assert_allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg="AVGPRICE mismatch",
        )

    def test_medprice_accuracy(self, price_data):
        """Test Median Price matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]

        # TA-Lib reference
        expected = talib.MEDPRICE(high, low)

        # Our implementation
        result = medprice(high, low)

        # Check accuracy
        assert_allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg="MEDPRICE mismatch",
        )

    def test_typprice_accuracy(self, price_data):
        """Test Typical Price matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.TYPPRICE(high, low, close)

        # Our implementation
        result = typprice(high, low, close)

        # Check accuracy
        assert_allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg="TYPPRICE mismatch",
        )

    def test_wclprice_accuracy(self, price_data):
        """Test Weighted Close Price matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.WCLPRICE(high, low, close)

        # Our implementation
        result = wclprice(high, low, close)

        # Check accuracy
        assert_allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg="WCLPRICE mismatch",
        )

    # ==================== Volatility Tests ====================

    def test_trange_accuracy(self, price_data):
        """Test True Range matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib reference
        expected = talib.TRANGE(high, low, close)

        # Our implementation
        result = trange(high, low, close)

        # Check accuracy
        assert_allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg="TRANGE mismatch",
        )

    def test_natr_accuracy(self, price_data):
        """Test Normalized Average True Range matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        for period in [14, 20]:
            # TA-Lib reference
            expected = talib.NATR(high, low, close, timeperiod=period)

            # Our implementation
            result = natr(high, low, close, period=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"NATR mismatch for period={period}",
            )

    # ==================== Volume Tests ====================

    def test_ad_accuracy(self, price_data):
        """Test Chaikin A/D Line matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]
        volume = price_data["volume"]

        # TA-Lib reference
        expected = talib.AD(high, low, close, volume)

        # Our implementation
        result = ad(high, low, close, volume)

        # Check accuracy
        assert_allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg="AD mismatch",
        )

    def test_adosc_accuracy(self, price_data):
        """Test Chaikin A/D Oscillator matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]
        volume = price_data["volume"]

        for fast in [3, 5]:
            for slow in [10, 14]:
                # TA-Lib reference
                expected = talib.ADOSC(high, low, close, volume, fastperiod=fast, slowperiod=slow)

                # Our implementation
                result = adosc(high, low, close, volume, fastperiod=fast, slowperiod=slow)

                # Check accuracy
                assert_allclose(
                    result,
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"ADOSC mismatch for fastperiod={fast}, slowperiod={slow}",
                )

    # ==================== Statistics/Regression Tests ====================

    def test_linearreg_accuracy(self, price_data):
        """Test Linear Regression matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [14, 20]:
            # TA-Lib reference
            expected = talib.LINEARREG(close, timeperiod=period)

            # Our implementation
            result = linearreg(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"LINEARREG mismatch for period={period}",
            )

    def test_linearreg_slope_accuracy(self, price_data):
        """Test Linear Regression Slope matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [14, 20]:
            # TA-Lib reference
            expected = talib.LINEARREG_SLOPE(close, timeperiod=period)

            # Our implementation
            result = linearreg_slope(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"LINEARREG_SLOPE mismatch for period={period}",
            )

    def test_linearreg_intercept_accuracy(self, price_data):
        """Test Linear Regression Intercept matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [14, 20]:
            # TA-Lib reference
            expected = talib.LINEARREG_INTERCEPT(close, timeperiod=period)

            # Our implementation
            result = linearreg_intercept(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"LINEARREG_INTERCEPT mismatch for period={period}",
            )

    def test_linearreg_angle_accuracy(self, price_data):
        """Test Linear Regression Angle matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [14, 20]:
            # TA-Lib reference
            expected = talib.LINEARREG_ANGLE(close, timeperiod=period)

            # Our implementation
            result = linearreg_angle(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"LINEARREG_ANGLE mismatch for period={period}",
            )

    def test_tsf_accuracy(self, price_data):
        """Test Time Series Forecast matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [14, 20]:
            # TA-Lib reference
            expected = talib.TSF(close, timeperiod=period)

            # Our implementation
            result = tsf(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"TSF mismatch for period={period}",
            )
