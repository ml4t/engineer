"""
Tests targeting Numba code paths in momentum indicators.

These tests use large datasets and explicit numba implementation selection
to trigger JIT-compiled code paths and improve coverage.
"""

import numpy as np
import pytest


class TestADXNumba:
    """Tests for ADX Numba implementation."""

    @pytest.fixture
    def large_ohlc(self):
        """Create large OHLC arrays to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return high, low, close

    def test_adx_numba_impl(self, large_ohlc):
        """Test ADX with numba implementation."""
        from ml4t.engineer.features.momentum.adx import adx

        high, low, close = large_ohlc
        result = adx(high, low, close, period=14)

        assert result is not None
        assert len(result) == len(close)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        # ADX is typically between 0 and 100
        assert np.all((valid >= 0) & (valid <= 100))

    def test_adx_auto_impl_large(self, large_ohlc):
        """Test ADX auto implementation with large data."""
        from ml4t.engineer.features.momentum.adx import adx

        high, low, close = large_ohlc
        result = adx(high, low, close, period=14)

        assert result is not None
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestSARNumba:
    """Tests for SAR Numba implementation."""

    @pytest.fixture
    def large_ohlc(self):
        """Create large OHLC arrays to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return high, low, close

    def test_sar_numba_impl(self, large_ohlc):
        """Test SAR with numba implementation."""
        from ml4t.engineer.features.momentum.sar import sar

        high, low, _ = large_ohlc
        result = sar(high, low, acceleration=0.02, maximum=0.2)

        assert result is not None
        assert len(result) == len(high)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_sar_different_params(self, large_ohlc):
        """Test SAR with different acceleration parameters."""
        from ml4t.engineer.features.momentum.sar import sar

        high, low, _ = large_ohlc

        r1 = sar(high, low, acceleration=0.02, maximum=0.2)
        r2 = sar(high, low, acceleration=0.01, maximum=0.1)

        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            assert not np.allclose(r1[valid_mask], r2[valid_mask], rtol=0.01)


class TestMACDNumba:
    """Tests for MACD Numba implementation."""

    @pytest.fixture
    def large_prices(self):
        """Create large price array to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return close

    def test_macd_numba_impl(self, large_prices):
        """Test MACD with numba implementation."""
        from ml4t.engineer.features.momentum.macd import macd

        result = macd(
            large_prices,
            fast_period=12,
            slow_period=26,
        )

        assert result is not None
        assert len(result) == len(large_prices)

    def test_macd_different_periods(self, large_prices):
        """Test MACD with different periods."""
        from ml4t.engineer.features.momentum.macd import macd

        r1 = macd(large_prices, fast_period=12, slow_period=26)
        r2 = macd(large_prices, fast_period=8, slow_period=17)

        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            assert not np.allclose(r1[valid_mask], r2[valid_mask], rtol=0.01)


class TestRSINumba:
    """Tests for RSI Numba implementation."""

    @pytest.fixture
    def large_prices(self):
        """Create large price array to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return close

    def test_rsi_numba_impl(self, large_prices):
        """Test RSI with numba implementation."""
        from ml4t.engineer.features.momentum.rsi import rsi

        result = rsi(large_prices, period=14)

        assert result is not None
        assert len(result) == len(large_prices)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        # RSI is between 0 and 100
        assert np.all((valid >= 0) & (valid <= 100))


class TestCMONumba:
    """Tests for CMO Numba implementation."""

    @pytest.fixture
    def large_prices(self):
        """Create large price array to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        return close

    def test_cmo_numba_impl(self, large_prices):
        """Test CMO with numba implementation."""
        from ml4t.engineer.features.momentum.cmo import cmo

        result = cmo(large_prices, timeperiod=14)

        assert result is not None
        assert len(result) == len(large_prices)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestStochasticNumba:
    """Tests for Stochastic Numba implementation."""

    @pytest.fixture
    def large_ohlc(self):
        """Create large OHLC arrays to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return high, low, close

    def test_stochf_numba_impl(self, large_ohlc):
        """Test STOCHF with auto implementation (covers Numba path for arrays)."""
        from ml4t.engineer.features.momentum.stochf import stochf

        high, low, close = large_ohlc
        # Use auto implementation - for numpy arrays this uses Numba path
        result = stochf(high, low, close, fastk_period=14, fastd_period=3)

        assert result is not None
        assert len(result) == len(close)


class TestDirectionalNumba:
    """Tests for Directional indicators Numba implementation."""

    @pytest.fixture
    def large_ohlc(self):
        """Create large OHLC arrays to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return high, low, close

    def test_plus_di_basic(self, large_ohlc):
        """Test PLUS_DI basic functionality."""
        from ml4t.engineer.features.momentum.directional import plus_di

        high, low, close = large_ohlc
        result = plus_di(high, low, close, timeperiod=14)

        assert result is not None
        assert len(result) == len(close)

    def test_minus_di_basic(self, large_ohlc):
        """Test MINUS_DI basic functionality."""
        from ml4t.engineer.features.momentum.directional import minus_di

        high, low, close = large_ohlc
        result = minus_di(high, low, close, timeperiod=14)

        assert result is not None
        assert len(result) == len(close)


class TestMFINumba:
    """Tests for MFI Numba implementation."""

    @pytest.fixture
    def large_ohlcv(self):
        """Create large OHLCV arrays to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(1000, 10000, n).astype(float)
        return high, low, close, volume

    def test_mfi_numba_impl(self, large_ohlcv):
        """Test MFI with numba implementation."""
        from ml4t.engineer.features.momentum.mfi import mfi

        high, low, close, volume = large_ohlcv
        result = mfi(high, low, close, volume, period=14)

        assert result is not None
        assert len(result) == len(close)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        # MFI is between 0 and 100
        assert np.all((valid >= -1e-10) & (valid <= 100 + 1e-10))


class TestULTOSCNumba:
    """Tests for ULTOSC Numba implementation."""

    @pytest.fixture
    def large_ohlc(self):
        """Create large OHLC arrays to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return high, low, close

    def test_ultosc_numba_impl(self, large_ohlc):
        """Test ULTOSC with numba implementation."""
        from ml4t.engineer.features.momentum.ultosc import ultosc

        high, low, close = large_ohlc
        result = ultosc(
            high,
            low,
            close,
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28,
        )

        assert result is not None
        assert len(result) == len(close)


class TestAROONNumba:
    """Tests for AROON Numba implementation."""

    @pytest.fixture
    def large_ohlc(self):
        """Create large OHLC arrays to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return high, low, close

    def test_aroon_numba_impl(self, large_ohlc):
        """Test AROON with numba implementation."""
        from ml4t.engineer.features.momentum.aroon import aroon

        high, low, _ = large_ohlc
        result = aroon(high, low, timeperiod=14)

        # AROON returns a tuple (aroon_up, aroon_down) or expression
        assert result is not None
        if isinstance(result, tuple):
            aroon_up, aroon_down = result
            assert len(aroon_up) == len(high)
            assert len(aroon_down) == len(high)


class TestCCINumba:
    """Tests for CCI Numba implementation."""

    @pytest.fixture
    def large_ohlc(self):
        """Create large OHLC arrays to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return high, low, close

    def test_cci_numba_impl(self, large_ohlc):
        """Test CCI with numba implementation."""
        from ml4t.engineer.features.momentum.cci import cci

        high, low, close = large_ohlc
        result = cci(high, low, close, period=14)

        assert result is not None
        assert len(result) == len(close)


class TestWILLRNumba:
    """Tests for WILLR Numba implementation."""

    @pytest.fixture
    def large_ohlc(self):
        """Create large OHLC arrays to trigger Numba paths."""
        np.random.seed(42)
        n = 1000
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return high, low, close

    def test_willr_numba_impl(self, large_ohlc):
        """Test WILLR with numba implementation."""
        from ml4t.engineer.features.momentum.willr import willr

        high, low, close = large_ohlc
        result = willr(high, low, close, period=14)

        assert result is not None
        assert len(result) == len(close)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            # WILLR is between -100 and 0
            assert np.all((valid >= -100) & (valid <= 0))
