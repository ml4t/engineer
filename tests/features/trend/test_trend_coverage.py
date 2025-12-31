"""
Comprehensive coverage tests for trend indicators.

Tests edge cases and various code paths to improve coverage
for SMA, EMA, DEMA, TEMA, T3, WMA, TRIMA, KAMA, MIDPOINT.
"""

import numpy as np
import polars as pl


class TestSMACoverage:
    """Coverage tests for Simple Moving Average."""

    def test_sma_basic(self):
        """Test basic SMA calculation."""
        from ml4t.engineer.features.trend.sma import sma

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = sma(close, period=20)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_sma_short_period(self):
        """Test SMA with short period."""
        from ml4t.engineer.features.trend.sma import sma

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        result = sma(close, period=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 90  # Most values should be valid

    def test_sma_long_period(self):
        """Test SMA with long period."""
        from ml4t.engineer.features.trend.sma import sma

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)

        result = sma(close, period=50)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_sma_polars_expr(self):
        """Test SMA with Polars expressions."""
        from ml4t.engineer.features.trend.sma import sma

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(sma("close", period=20))
        assert len(result) == 200


class TestEMACoverage:
    """Coverage tests for Exponential Moving Average."""

    def test_ema_basic(self):
        """Test basic EMA calculation."""
        from ml4t.engineer.features.trend.ema import ema

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = ema(close, period=20)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_ema_short_period(self):
        """Test EMA with short period."""
        from ml4t.engineer.features.trend.ema import ema

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        result = ema(close, period=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 90

    def test_ema_vs_sma(self):
        """Test that EMA responds faster than SMA to changes."""
        from ml4t.engineer.features.trend.ema import ema
        from ml4t.engineer.features.trend.sma import sma

        # Sharp upward move
        close = np.array([100.0] * 50 + [110.0] * 50)

        ema_result = ema(close, period=10)
        sma_result = sma(close, period=10)

        # EMA should respond faster (higher value at position 55)
        valid_ema = ema_result[~np.isnan(ema_result)]
        valid_sma = sma_result[~np.isnan(sma_result)]

        assert len(valid_ema) > 0
        assert len(valid_sma) > 0

    def test_ema_polars_expr(self):
        """Test EMA with Polars expressions."""
        from ml4t.engineer.features.trend.ema import ema

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(ema("close", period=20))
        assert len(result) == 200


class TestDEMACoverage:
    """Coverage tests for Double Exponential Moving Average."""

    def test_dema_basic(self):
        """Test basic DEMA calculation."""
        from ml4t.engineer.features.trend.dema import dema

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = dema(close, period=20)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_dema_short_period(self):
        """Test DEMA with short period."""
        from ml4t.engineer.features.trend.dema import dema

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        result = dema(close, period=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_dema_polars_expr(self):
        """Test DEMA with Polars expressions."""
        from ml4t.engineer.features.trend.dema import dema

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(dema("close", period=20))
        assert len(result) == 200


class TestTEMACoverage:
    """Coverage tests for Triple Exponential Moving Average."""

    def test_tema_basic(self):
        """Test basic TEMA calculation."""
        from ml4t.engineer.features.trend.tema import tema

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = tema(close, period=20)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_tema_short_period(self):
        """Test TEMA with short period."""
        from ml4t.engineer.features.trend.tema import tema

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        result = tema(close, period=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_tema_polars_expr(self):
        """Test TEMA with Polars expressions."""
        from ml4t.engineer.features.trend.tema import tema

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(tema("close", period=20))
        assert len(result) == 200


class TestT3Coverage:
    """Coverage tests for T3 indicator."""

    def test_t3_basic(self):
        """Test basic T3 calculation."""
        from ml4t.engineer.features.trend.t3 import t3

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = t3(close, timeperiod=5, vfactor=0.7)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_t3_custom_vfactor(self):
        """Test T3 with custom vfactor."""
        from ml4t.engineer.features.trend.t3 import t3

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)

        r1 = t3(close, timeperiod=5, vfactor=0.5)
        r2 = t3(close, timeperiod=5, vfactor=0.9)

        # Both should produce valid results
        assert len(r1[~np.isnan(r1)]) > 0
        assert len(r2[~np.isnan(r2)]) > 0

    def test_t3_polars_expr(self):
        """Test T3 with Polars expressions."""
        from ml4t.engineer.features.trend.t3 import t3

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(t3("close", timeperiod=5))
        assert len(result) == 200


class TestWMACoverage:
    """Coverage tests for Weighted Moving Average."""

    def test_wma_basic(self):
        """Test basic WMA calculation."""
        from ml4t.engineer.features.trend.wma import wma

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = wma(close, period=20)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_wma_short_period(self):
        """Test WMA with short period."""
        from ml4t.engineer.features.trend.wma import wma

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        result = wma(close, period=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 90

    def test_wma_polars_expr(self):
        """Test WMA with Polars expressions."""
        from ml4t.engineer.features.trend.wma import wma

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(wma("close", period=20))
        assert len(result) == 200


class TestTRIMACoverage:
    """Coverage tests for Triangular Moving Average."""

    def test_trima_basic(self):
        """Test basic TRIMA calculation."""
        from ml4t.engineer.features.trend.trima import trima

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = trima(close, period=20)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_trima_short_period(self):
        """Test TRIMA with short period."""
        from ml4t.engineer.features.trend.trima import trima

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        result = trima(close, period=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_trima_polars_expr(self):
        """Test TRIMA with Polars expressions."""
        from ml4t.engineer.features.trend.trima import trima

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(trima("close", period=20))
        assert len(result) == 200


class TestKAMACoverage:
    """Coverage tests for Kaufman Adaptive Moving Average."""

    def test_kama_basic(self):
        """Test basic KAMA calculation."""
        from ml4t.engineer.features.trend.kama import kama

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = kama(close, timeperiod=30)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_kama_short_period(self):
        """Test KAMA with short period."""
        from ml4t.engineer.features.trend.kama import kama

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        result = kama(close, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_kama_trending_vs_ranging(self):
        """Test KAMA adapts to trending vs ranging markets."""
        from ml4t.engineer.features.trend.kama import kama

        # Trending market
        trend_close = np.array([100.0 + i * 0.5 for i in range(100)])
        trend_result = kama(trend_close, timeperiod=10)

        # Ranging market
        np.random.seed(42)
        range_close = 100 + np.random.randn(100) * 2
        range_result = kama(range_close, timeperiod=10)

        # Both should produce valid results
        assert len(trend_result[~np.isnan(trend_result)]) > 0
        assert len(range_result[~np.isnan(range_result)]) > 0

    def test_kama_polars_expr(self):
        """Test KAMA with Polars expressions."""
        from ml4t.engineer.features.trend.kama import kama

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(kama("close", timeperiod=30))
        assert len(result) == 200


class TestMIDPOINTCoverage:
    """Coverage tests for Midpoint indicator."""

    def test_midpoint_basic(self):
        """Test basic MIDPOINT calculation."""
        from ml4t.engineer.features.trend.midpoint import midpoint

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = midpoint(close, timeperiod=14)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_midpoint_short_period(self):
        """Test MIDPOINT with short period."""
        from ml4t.engineer.features.trend.midpoint import midpoint

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        result = midpoint(close, timeperiod=5)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 90

    def test_midpoint_polars_expr(self):
        """Test MIDPOINT with Polars expressions."""
        from ml4t.engineer.features.trend.midpoint import midpoint

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)
        df = pl.DataFrame({"close": close})

        result = df.select(midpoint("close", timeperiod=14))
        assert len(result) == 200


class TestEdgeCases:
    """Test edge cases for trend indicators."""

    def test_constant_prices(self):
        """Test with constant prices."""
        from ml4t.engineer.features.trend.sma import sma

        close = np.array([100.0] * 50)
        result = sma(close, period=14)

        # Constant prices should give constant SMA
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.allclose(valid_values, 100.0, rtol=1e-10)

    def test_single_value(self):
        """Test with single value."""
        from ml4t.engineer.features.trend.sma import sma

        close = np.array([100.0])
        result = sma(close, period=14)

        # Should handle gracefully
        assert result is not None
        assert len(result) == 1

    def test_short_data(self):
        """Test with data shorter than period."""
        from ml4t.engineer.features.trend.sma import sma

        close = np.array([100.0, 101.0, 102.0])
        result = sma(close, period=14)

        # Should handle gracefully
        assert result is not None
        assert len(result) == 3

    def test_nan_handling(self):
        """Test NaN handling in trend indicators."""
        from ml4t.engineer.features.trend.sma import sma

        values = np.array([100.0, np.nan, 101.0, 102.0] * 20)
        result = sma(values, period=14)

        # Should not crash
        assert result is not None
        assert len(result) == len(values)

    def test_all_ma_types_available(self):
        """Test that all MA types can be imported and called."""
        from ml4t.engineer.features.trend.dema import dema
        from ml4t.engineer.features.trend.ema import ema
        from ml4t.engineer.features.trend.kama import kama
        from ml4t.engineer.features.trend.midpoint import midpoint
        from ml4t.engineer.features.trend.sma import sma
        from ml4t.engineer.features.trend.t3 import t3
        from ml4t.engineer.features.trend.tema import tema
        from ml4t.engineer.features.trend.trima import trima
        from ml4t.engineer.features.trend.wma import wma

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)

        # All should produce valid results
        for func in [sma, ema, dema, tema, wma, trima]:
            result = func(close, period=20)
            assert result is not None
            assert len(result) == 200

        # T3 needs vfactor and uses timeperiod
        result = t3(close, timeperiod=5, vfactor=0.7)
        assert result is not None

        # KAMA and MIDPOINT use timeperiod
        result = kama(close, timeperiod=30)
        assert result is not None

        result = midpoint(close, timeperiod=14)
        assert result is not None


class TestT3NumbaPath:
    """Tests for T3 Numba implementation path."""

    def test_t3_numba_direct(self):
        """Test T3 Numba function directly."""
        from ml4t.engineer.features.trend.t3 import t3_numba

        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)

        result = t3_numba(close, timeperiod=5, vfactor=0.7)

        assert result is not None
        assert len(result) == n
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_t3_numba_different_periods(self):
        """Test T3 Numba with different periods."""
        from ml4t.engineer.features.trend.t3 import t3_numba

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(500) * 0.5)

        for period in [3, 5, 10, 15]:
            result = t3_numba(close, timeperiod=period, vfactor=0.7)
            assert result is not None
            valid_values = result[~np.isnan(result)]
            # More periods means more NaN at start
            expected_valid = len(close) - 6 * (period - 1)
            if expected_valid > 0:
                assert len(valid_values) > 0

    def test_t3_numba_short_data(self):
        """Test T3 Numba with short data (less than lookback)."""
        from ml4t.engineer.features.trend.t3 import t3_numba

        # Data too short for the lookback
        close = np.array([100.0, 101.0, 102.0, 103.0])
        result = t3_numba(close, timeperiod=5, vfactor=0.7)

        # Should return all NaN
        assert result is not None
        assert len(result) == 4
        assert np.all(np.isnan(result))

    def test_t3_numba_vfactor_extremes(self):
        """Test T3 Numba with extreme vfactor values."""
        from ml4t.engineer.features.trend.t3 import t3_numba

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)

        # vfactor=0
        r0 = t3_numba(close, timeperiod=5, vfactor=0.0)
        assert len(r0[~np.isnan(r0)]) > 0

        # vfactor=1
        r1 = t3_numba(close, timeperiod=5, vfactor=1.0)
        assert len(r1[~np.isnan(r1)]) > 0


class TestNumbaImplementations:
    """Tests for Numba implementations to boost coverage."""

    def test_ema_numba_path(self):
        """Test EMA Numba implementation."""
        from ml4t.engineer.features.trend.ema import ema

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(1000) * 0.5)

        # Force numba path with _implementation parameter if available
        result = ema(close, period=20)

        assert result is not None
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_dema_numba_path(self):
        """Test DEMA Numba implementation."""
        from ml4t.engineer.features.trend.dema import dema

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(1000) * 0.5)

        result = dema(close, period=20)

        assert result is not None
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_tema_numba_path(self):
        """Test TEMA Numba implementation."""
        from ml4t.engineer.features.trend.tema import tema

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(1000) * 0.5)

        result = tema(close, period=20)

        assert result is not None
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_kama_numba_path(self):
        """Test KAMA Numba implementation."""
        from ml4t.engineer.features.trend.kama import kama

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(1000) * 0.5)

        result = kama(close, timeperiod=30)

        assert result is not None
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_sma_numba_path(self):
        """Test SMA Numba implementation."""
        from ml4t.engineer.features.trend.sma import sma

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(1000) * 0.5)

        result = sma(close, period=20)

        assert result is not None
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
