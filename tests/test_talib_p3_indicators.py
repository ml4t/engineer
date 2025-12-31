"""
TA-Lib P3 Indicator Validation Tests.

P3 Priority: Specialized momentum, ROC variations, and MA variations
- Specialized momentum (3 indicators)
- ROC variations (3 indicators)
- MACD variant (1 indicator)
- Variance (1 indicator)
- Math functions (2 indicators)
- Specialized MA (3 indicators)

Total: 13 indicators

Tests verify exact numerical accuracy against TA-Lib reference implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

talib = pytest.importorskip("talib")

from ml4t.engineer.features.math.max import maximum
from ml4t.engineer.features.math.min import minimum
from ml4t.engineer.features.momentum.cmo import cmo
from ml4t.engineer.features.momentum.macd import macd
from ml4t.engineer.features.momentum.macdfix import macdfix
from ml4t.engineer.features.momentum.rocp import rocp
from ml4t.engineer.features.momentum.rocr import rocr
from ml4t.engineer.features.momentum.rocr100 import rocr100
from ml4t.engineer.features.momentum.trix import trix
from ml4t.engineer.features.momentum.ultosc import ultosc
from ml4t.engineer.features.price_transform.midprice import midprice
from ml4t.engineer.features.statistics.var import var
from ml4t.engineer.features.trend.midpoint import midpoint
from ml4t.engineer.features.trend.t3 import t3


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


class TestP3IndicatorAccuracy:
    """Test P3 indicators match TA-Lib exactly."""

    # ==================== Specialized Momentum Tests ====================

    def test_cmo_accuracy(self, price_data):
        """Test Chande Momentum Oscillator matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [14, 20]:
            # TA-Lib reference
            expected = talib.CMO(close, timeperiod=period)

            # Our implementation
            result = cmo(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"CMO mismatch for period={period}",
            )

    def test_trix_accuracy(self, price_data):
        """Test TRIX matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [30, 20]:
            # TA-Lib reference
            expected = talib.TRIX(close, timeperiod=period)

            # Our implementation
            result = trix(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"TRIX mismatch for period={period}",
            )

    def test_ultosc_accuracy(self, price_data):
        """Test Ultimate Oscillator matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]
        close = price_data["close"]

        # TA-Lib default periods
        # TA-Lib reference
        expected = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        # Our implementation
        result = ultosc(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        # Check accuracy
        assert_allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg="ULTOSC mismatch",
        )

    # ==================== ROC Variations Tests ====================

    def test_rocp_accuracy(self, price_data):
        """Test Rate of Change Percentage matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [10, 12]:
            # TA-Lib reference
            expected = talib.ROCP(close, timeperiod=period)

            # Our implementation
            result = rocp(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"ROCP mismatch for period={period}",
            )

    def test_rocr_accuracy(self, price_data):
        """Test Rate of Change Ratio matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [10, 12]:
            # TA-Lib reference
            expected = talib.ROCR(close, timeperiod=period)

            # Our implementation
            result = rocr(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"ROCR mismatch for period={period}",
            )

    def test_rocr100_accuracy(self, price_data):
        """Test Rate of Change Ratio 100 scale matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [10, 12]:
            # TA-Lib reference
            expected = talib.ROCR100(close, timeperiod=period)

            # Our implementation
            result = rocr100(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"ROCR100 mismatch for period={period}",
            )

    # ==================== MACD Variant Tests ====================

    def test_macd_accuracy(self, price_data):
        """Test MACD (12, 26, 9) matches TA-Lib exactly."""
        close = price_data["close"]

        # TA-Lib reference
        expected_macd, _, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        # Our implementation
        result_macd = macd(close, fast_period=12, slow_period=26)

        # Check exact match
        assert_allclose(
            result_macd,
            expected_macd,
            rtol=1e-6,
            atol=1e-6,
            err_msg="MACD mismatch",
        )

    def test_macdfix_accuracy(self, price_data):
        """Test MACD Fix 12/26 matches TA-Lib exactly.

        IMPORTANT: MACDFIX uses FIXED k constants, not standard k = 2/(period+1)!

        From TA-Lib source code (ta_MACD.c lines 420-435):
        - Fast period 12: k = 0.15 (NOT 2/(12+1) = 0.1538...)
        - Slow period 26: k = 0.075 (NOT 2/(26+1) = 0.0740...)

        This is why it's called "FIX" - fixed decimal constants for backwards
        compatibility with legacy systems that used these approximations.

        Our implementation now correctly uses these fixed constants.
        """
        close = price_data["close"]

        # TA-Lib reference
        expected_macd, expected_signal, expected_hist = talib.MACDFIX(close, signalperiod=9)

        # Our implementation (uses fixed k=0.15 and k=0.075)
        result_macd = macdfix(close, signalperiod=9)

        # Should match exactly
        assert_allclose(
            result_macd,
            expected_macd,
            rtol=1e-6,
            atol=1e-6,
            err_msg="MACDFIX macd mismatch",
        )

    # ==================== Variance Tests ====================

    def test_var_accuracy(self, price_data):
        """Test Variance matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 10]:
            # TA-Lib reference
            expected = talib.VAR(close, timeperiod=period, nbdev=1)

            # Our implementation
            result = var(close, timeperiod=period, nbdev=1)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"VAR mismatch for period={period}",
            )

    # ==================== Math Functions Tests ====================

    def test_maximum_accuracy(self, price_data):
        """Test Maximum (highest value) matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [14, 30]:
            # TA-Lib reference (uses MAX function)
            expected = talib.MAX(close, timeperiod=period)

            # Our implementation
            result = maximum(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"MAXIMUM mismatch for period={period}",
            )

    def test_minimum_accuracy(self, price_data):
        """Test Minimum (lowest value) matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [14, 30]:
            # TA-Lib reference (uses MIN function)
            expected = talib.MIN(close, timeperiod=period)

            # Our implementation
            result = minimum(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"MINIMUM mismatch for period={period}",
            )

    # ==================== Specialized MA Tests ====================

    def test_t3_accuracy(self, price_data):
        """Test T3 Moving Average matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [5, 10]:
            # TA-Lib reference
            expected = talib.T3(close, timeperiod=period, vfactor=0)

            # Our implementation
            result = t3(close, timeperiod=period, vfactor=0)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"T3 mismatch for period={period}",
            )

    def test_midpoint_accuracy(self, price_data):
        """Test Midpoint matches TA-Lib exactly."""
        close = price_data["close"]

        for period in [14, 20]:
            # TA-Lib reference
            expected = talib.MIDPOINT(close, timeperiod=period)

            # Our implementation
            result = midpoint(close, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"MIDPOINT mismatch for period={period}",
            )

    def test_midprice_accuracy(self, price_data):
        """Test Midpoint Price matches TA-Lib exactly."""
        high = price_data["high"]
        low = price_data["low"]

        for period in [14, 20]:
            # TA-Lib reference
            expected = talib.MIDPRICE(high, low, timeperiod=period)

            # Our implementation
            result = midprice(high, low, timeperiod=period)

            # Check accuracy
            assert_allclose(
                result,
                expected,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"MIDPRICE mismatch for period={period}",
            )
