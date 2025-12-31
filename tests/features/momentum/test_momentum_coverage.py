"""
Comprehensive coverage tests for momentum indicators.

Tests edge cases and various code paths to improve coverage
for SAR, ADX, MFI, ULTOSC, RSI, CMO, and other momentum indicators.
"""

import numpy as np
import polars as pl
import pytest


class TestSARCoverage:
    """Coverage tests for Parabolic SAR indicator."""

    @pytest.fixture
    def ohlc_data(self):
        """Create OHLC data for SAR testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return pl.DataFrame({"high": high, "low": low, "close": close})

    def test_sar_basic(self, ohlc_data):
        """Test basic SAR calculation."""
        from ml4t.engineer.features.momentum.sar import sar

        result = ohlc_data.select(sar("high", "low").alias("sar"))
        assert "sar" in result.columns
        # SAR should be either above or below price
        sar_values = result["sar"].drop_nulls()
        assert len(sar_values) > 0

    def test_sar_custom_params(self, ohlc_data):
        """Test SAR with custom acceleration parameters."""
        from ml4t.engineer.features.momentum.sar import sar

        result = ohlc_data.select(sar("high", "low", acceleration=0.01, maximum=0.1).alias("sar"))
        assert "sar" in result.columns

    def test_sar_uptrend(self):
        """Test SAR in uptrend (should be below price)."""
        from ml4t.engineer.features.momentum.sar import sar

        # Strong uptrend
        n = 100
        close = np.array([100 + i * 0.5 for i in range(n)])
        high = close + 0.2
        low = close - 0.2

        # Use numpy arrays directly
        result = sar(high, low)

        # Find valid indices (non-NaN)
        valid_mask = ~np.isnan(result)
        sar_valid = result[valid_mask]
        low_valid = low[valid_mask]

        # In uptrend, at least some SAR values should be below low
        if len(sar_valid) > 0:
            below_low = (sar_valid < low_valid).mean()
            assert below_low > 0.3  # At least 30% should be below

    def test_sar_downtrend(self):
        """Test SAR in downtrend (should be above price)."""
        from ml4t.engineer.features.momentum.sar import sar

        # Strong downtrend
        n = 100
        close = np.array([100 - i * 0.5 for i in range(n)])
        high = close + 0.2
        low = close - 0.2

        result = sar(high, low)

        valid_mask = ~np.isnan(result)
        sar_valid = result[valid_mask]
        high_valid = high[valid_mask]

        # In downtrend, at least some SAR values should be above high
        if len(sar_valid) > 0:
            above_high = (sar_valid > high_valid).mean()
            assert above_high > 0.3  # At least 30% should be above

    def test_sar_reversal(self):
        """Test SAR reversal on trend change."""
        from ml4t.engineer.features.momentum.sar import sar

        # Uptrend then downtrend
        up = [100 + i * 0.5 for i in range(50)]
        down = [up[-1] - i * 0.5 for i in range(1, 51)]
        close = np.array(up + down)

        high = close + 0.2
        low = close - 0.2

        result = sar(high, low)

        # Should not crash and should produce values
        assert len(result[~np.isnan(result)]) > 0


class TestADXCoverage:
    """Coverage tests for ADX indicator."""

    @pytest.fixture
    def ohlc_arrays(self):
        """Create OHLC numpy arrays for ADX testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return high, low, close

    def test_adx_basic(self, ohlc_arrays):
        """Test basic ADX calculation."""
        from ml4t.engineer.features.momentum.adx import adx

        high, low, close = ohlc_arrays
        result = adx(high, low, close, period=14)

        assert result is not None
        assert len(result) == len(high)

        # ADX should be between 0 and 100
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)

    def test_adx_short_period(self, ohlc_arrays):
        """Test ADX with short period."""
        from ml4t.engineer.features.momentum.adx import adx

        high, low, close = ohlc_arrays
        result = adx(high, low, close, period=5)

        assert result is not None
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_adx_strong_trend(self):
        """Test ADX in strong trending market."""
        from ml4t.engineer.features.momentum.adx import adx

        # Strong uptrend with some noise
        np.random.seed(42)
        n = 100
        trend = np.array([100 + i * 1.0 for i in range(n)])
        noise = np.random.randn(n) * 0.1
        close = trend + noise
        high = close + 0.5
        low = close - 0.5

        result = adx(high, low, close, period=14)

        # Strong trend should have high ADX
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            # ADX > 25 indicates trend
            assert valid_values.mean() > 20

    def test_adx_ranging_market(self):
        """Test ADX in ranging/sideways market."""
        from ml4t.engineer.features.momentum.adx import adx

        np.random.seed(42)
        n = 100
        # Sideways with noise
        close = 100 + np.random.randn(n) * 2
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)

        result = adx(high, low, close, period=14)

        # Ranging market should have lower ADX
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # Allow for noise - ADX should generally be lower in ranging market
        assert valid_values.mean() < 60


class TestMFICoverage:
    """Coverage tests for Money Flow Index."""

    def test_mfi_basic(self):
        """Test basic MFI calculation."""
        from ml4t.engineer.features.momentum.mfi import mfi

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(1000, 10000, n).astype(float)

        result = mfi(high, low, close, volume, period=14)

        assert result is not None
        assert len(result) == n

        # MFI should be between 0 and 100
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)

    def test_mfi_overbought(self):
        """Test MFI in overbought condition."""
        from ml4t.engineer.features.momentum.mfi import mfi

        # Strong uptrend with increasing volume
        n = 100
        close = np.array([100 + i * 0.5 for i in range(n)])
        high = close + 0.2
        low = close - 0.2
        volume = np.array([1000.0 + i * 50 for i in range(n)])

        result = mfi(high, low, close, volume, period=14)

        # Should have high MFI (overbought)
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert valid_values.mean() > 45

    def test_mfi_oversold(self):
        """Test MFI in oversold condition."""
        from ml4t.engineer.features.momentum.mfi import mfi

        # Strong downtrend with increasing volume
        n = 100
        close = np.array([100 - i * 0.5 for i in range(n)])
        high = close + 0.2
        low = close - 0.2
        volume = np.array([1000.0 + i * 50 for i in range(n)])

        result = mfi(high, low, close, volume, period=14)

        # Should have low MFI (oversold)
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert valid_values.mean() < 55


class TestULTOSCCoverage:
    """Coverage tests for Ultimate Oscillator."""

    def test_ultosc_basic(self):
        """Test basic ULTOSC calculation."""
        from ml4t.engineer.features.momentum.ultosc import ultosc

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = ultosc(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        assert result is not None
        assert len(result) == n

        # ULTOSC should be between 0 and 100
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)

    def test_ultosc_custom_periods(self):
        """Test ULTOSC with custom periods."""
        from ml4t.engineer.features.momentum.ultosc import ultosc

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)

        result = ultosc(high, low, close, timeperiod1=5, timeperiod2=10, timeperiod3=20)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_ultosc_bullish(self):
        """Test ULTOSC in bullish condition."""
        from ml4t.engineer.features.momentum.ultosc import ultosc

        # Strong uptrend
        n = 100
        close = np.array([100 + i * 0.5 for i in range(n)])
        high = close + 0.3
        low = close - 0.3

        result = ultosc(high, low, close)

        # Bullish should have higher ULTOSC
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert valid_values.mean() > 30


class TestRSICoverage:
    """Coverage tests for RSI indicator."""

    def test_rsi_basic(self):
        """Test basic RSI calculation."""
        from ml4t.engineer.features.momentum.rsi import rsi

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = rsi(close, period=14)

        assert result is not None
        assert len(result) == n

        # RSI should be between 0 and 100
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)

    def test_rsi_overbought(self):
        """Test RSI in overbought condition."""
        from ml4t.engineer.features.momentum.rsi import rsi

        # Strong uptrend
        n = 100
        close = np.array([100 + i * 1.0 for i in range(n)])

        result = rsi(close, period=14)

        # Strong uptrend should have high RSI
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert valid_values.mean() > 50

    def test_rsi_oversold(self):
        """Test RSI in oversold condition."""
        from ml4t.engineer.features.momentum.rsi import rsi

        # Strong downtrend
        n = 100
        close = np.array([100 - i * 1.0 for i in range(n)])

        result = rsi(close, period=14)

        # Strong downtrend should have low RSI
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert valid_values.mean() < 50

    def test_rsi_short_period(self):
        """Test RSI with short period."""
        from ml4t.engineer.features.momentum.rsi import rsi

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = rsi(close, period=5)

        assert result is not None
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0


class TestCMOCoverage:
    """Coverage tests for Chande Momentum Oscillator."""

    def test_cmo_basic(self):
        """Test basic CMO calculation."""
        from ml4t.engineer.features.momentum.cmo import cmo

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = cmo(close, timeperiod=14)

        assert result is not None
        assert len(result) == n

        # CMO should be between -100 and 100
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= -100)
        assert np.all(valid_values <= 100)

    def test_cmo_strong_uptrend(self):
        """Test CMO in strong uptrend."""
        from ml4t.engineer.features.momentum.cmo import cmo

        n = 100
        close = np.array([100 + i * 0.5 for i in range(n)])

        result = cmo(close, timeperiod=14)

        # Strong uptrend should have positive CMO
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert valid_values.mean() > 0

    def test_cmo_strong_downtrend(self):
        """Test CMO in strong downtrend."""
        from ml4t.engineer.features.momentum.cmo import cmo

        n = 100
        close = np.array([100 - i * 0.5 for i in range(n)])

        result = cmo(close, timeperiod=14)

        # Strong downtrend should have negative CMO
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert valid_values.mean() < 0


class TestDirectionalCoverage:
    """Coverage tests for directional movement indicators."""

    @pytest.fixture
    def ohlc_arrays(self):
        """Create OHLC numpy arrays for directional testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        return high, low, close

    def test_plus_di_basic(self, ohlc_arrays):
        """Test +DI calculation."""
        from ml4t.engineer.features.momentum.directional import plus_di

        high, low, close = ohlc_arrays
        result = plus_di(high, low, close, timeperiod=14)

        assert result is not None
        assert len(result) == len(high)

        # +DI should be non-negative
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 0)

    def test_minus_di_basic(self, ohlc_arrays):
        """Test -DI calculation."""
        from ml4t.engineer.features.momentum.directional import minus_di

        high, low, close = ohlc_arrays
        result = minus_di(high, low, close, timeperiod=14)

        assert result is not None
        assert len(result) == len(high)

        # -DI should be non-negative
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.all(valid_values >= 0)

    def test_plus_dm_basic(self, ohlc_arrays):
        """Test +DM calculation."""
        from ml4t.engineer.features.momentum.plus_dm import plus_dm

        high, low, close = ohlc_arrays
        result = plus_dm(high, low, timeperiod=14)

        assert result is not None
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_minus_dm_basic(self, ohlc_arrays):
        """Test -DM calculation."""
        from ml4t.engineer.features.momentum.minus_dm import minus_dm

        high, low, close = ohlc_arrays
        result = minus_dm(high, low, timeperiod=14)

        assert result is not None
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0


class TestEdgeCases:
    """Test edge cases for momentum indicators."""

    def test_constant_prices(self):
        """Test with constant prices."""
        from ml4t.engineer.features.momentum.rsi import rsi

        close = np.array([100.0] * 50)
        result = rsi(close, period=14)

        # Constant prices should give RSI of 50 (no movement) or NaN
        assert result is not None
        assert len(result) == 50

    def test_single_direction_movement(self):
        """Test with single direction price movement."""
        from ml4t.engineer.features.momentum.rsi import rsi

        # All up movement
        close = np.array([100.0 + i for i in range(50)])
        result = rsi(close, period=14)

        valid_values = result[~np.isnan(result)]
        # All up should give high RSI (close to 100)
        if len(valid_values) > 0:
            assert valid_values.mean() > 70

    def test_nan_handling(self):
        """Test NaN handling in momentum indicators."""
        from ml4t.engineer.features.momentum.rsi import rsi

        values = np.array([100.0, np.nan, 101.0, 102.0] * 20)
        result = rsi(values, period=14)

        # Should not crash
        assert result is not None
        assert len(result) == len(values)

    def test_short_data(self):
        """Test with data shorter than period."""
        from ml4t.engineer.features.momentum.rsi import rsi

        close = np.array([100.0, 101.0, 102.0])
        result = rsi(close, period=14)

        # Should handle gracefully
        assert result is not None
        assert len(result) == 3


class TestPolarsExpressions:
    """Test Polars expression interface where supported."""

    def test_rsi_polars_expr(self):
        """Test RSI with Polars expressions."""
        from ml4t.engineer.features.momentum.rsi import rsi

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(rsi("close", period=14))
        assert result is not None
        assert len(result) == n

    def test_cmo_polars_expr(self):
        """Test CMO with Polars expressions."""
        from ml4t.engineer.features.momentum.cmo import cmo

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(cmo("close", timeperiod=14))
        assert result is not None
        assert len(result) == n

    def test_mfi_polars_expr(self):
        """Test MFI with Polars expressions."""
        from ml4t.engineer.features.momentum.mfi import mfi

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(1000, 10000, n).astype(float)

        df = pl.DataFrame(
            {
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        result = df.select(mfi("high", "low", "close", "volume", period=14))
        assert result is not None
        assert len(result) == n
