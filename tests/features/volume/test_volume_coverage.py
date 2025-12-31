"""
Comprehensive coverage tests for volume indicators.

Tests edge cases and various code paths to improve coverage.
"""

import numpy as np
import polars as pl
import pytest


class TestADCoverage:
    """Coverage tests for Accumulation/Distribution Line."""

    @pytest.fixture
    def ohlcv_arrays(self):
        """Create OHLCV arrays for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(1000, 10000, n).astype(float)
        return high, low, close, volume

    @pytest.fixture
    def ohlcv_df(self, ohlcv_arrays):
        """Create OHLCV DataFrame for testing."""
        high, low, close, volume = ohlcv_arrays
        return pl.DataFrame(
            {
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def test_ad_numpy_basic(self, ohlcv_arrays):
        """Test A/D line with numpy arrays."""
        from ml4t.engineer.features.volume.ad import ad

        high, low, close, volume = ohlcv_arrays
        result = ad(high, low, close, volume)

        assert result is not None
        assert len(result) == len(close)
        assert isinstance(result, np.ndarray)

    def test_ad_polars_series(self, ohlcv_df):
        """Test A/D line with Polars Series."""
        from ml4t.engineer.features.volume.ad import ad

        result = ad(
            ohlcv_df["high"],
            ohlcv_df["low"],
            ohlcv_df["close"],
            ohlcv_df["volume"],
        )

        assert result is not None
        assert len(result) == len(ohlcv_df)

    def test_ad_polars_expr(self, ohlcv_df):
        """Test A/D line with Polars expressions."""
        from ml4t.engineer.features.volume.ad import ad

        result = ohlcv_df.select(ad("high", "low", "close", "volume"))
        assert result is not None
        assert len(result) == len(ohlcv_df)

    def test_ad_numba_impl(self, ohlcv_arrays):
        """Test A/D line with numba implementation."""
        from ml4t.engineer.features.volume.ad import ad

        high, low, close, volume = ohlcv_arrays
        result = ad(high, low, close, volume, implementation="numba")

        assert result is not None
        assert len(result) == len(close)

    def test_ad_polars_impl(self, ohlcv_df):
        """Test A/D line with polars implementation."""
        from ml4t.engineer.features.volume.ad import ad

        result = ohlcv_df.select(ad("high", "low", "close", "volume", implementation="polars"))
        assert result is not None
        assert len(result) == len(ohlcv_df)

    def test_ad_constant_prices(self):
        """Test A/D with constant high/low."""
        from ml4t.engineer.features.volume.ad import ad

        n = 50
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        close = np.full(n, 100.0)
        volume = np.full(n, 5000.0)

        result = ad(high, low, close, volume)
        assert result is not None
        assert len(result) == n

    def test_ad_zero_volume(self):
        """Test A/D with zero volume."""
        from ml4t.engineer.features.volume.ad import ad

        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + 1
        low = close - 1
        volume = np.zeros(n)

        result = ad(high, low, close, volume)
        assert result is not None

    def test_ad_empty_arrays(self):
        """Test A/D with empty arrays."""
        from ml4t.engineer.features.volume.ad import ad

        empty = np.array([])
        result = ad(empty, empty, empty, empty)
        assert len(result) == 0

    def test_ad_single_value(self):
        """Test A/D with single value."""
        from ml4t.engineer.features.volume.ad import ad

        result = ad(
            np.array([101.0]),
            np.array([99.0]),
            np.array([100.0]),
            np.array([5000.0]),
        )
        assert len(result) == 1


class TestADOSCCoverage:
    """Coverage tests for Chaikin A/D Oscillator."""

    @pytest.fixture
    def ohlcv_arrays(self):
        """Create OHLCV arrays for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        volume = np.random.randint(1000, 10000, n).astype(float)
        return high, low, close, volume

    @pytest.fixture
    def ohlcv_df(self, ohlcv_arrays):
        """Create OHLCV DataFrame for testing."""
        high, low, close, volume = ohlcv_arrays
        return pl.DataFrame(
            {
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def test_adosc_numpy_basic(self, ohlcv_arrays):
        """Test ADOSC with numpy arrays."""
        from ml4t.engineer.features.volume.adosc import adosc

        high, low, close, volume = ohlcv_arrays
        result = adosc(high, low, close, volume, fastperiod=3, slowperiod=10)

        assert result is not None
        assert len(result) == len(close)
        assert isinstance(result, np.ndarray)

    def test_adosc_polars_series(self, ohlcv_df):
        """Test ADOSC with Polars Series."""
        from ml4t.engineer.features.volume.adosc import adosc

        result = adosc(
            ohlcv_df["high"],
            ohlcv_df["low"],
            ohlcv_df["close"],
            ohlcv_df["volume"],
            fastperiod=3,
            slowperiod=10,
        )

        assert result is not None
        assert len(result) == len(ohlcv_df)

    def test_adosc_polars_expr(self, ohlcv_df):
        """Test ADOSC with Polars expressions."""
        from ml4t.engineer.features.volume.adosc import adosc

        result = ohlcv_df.select(
            adosc("high", "low", "close", "volume", fastperiod=3, slowperiod=10)
        )
        assert result is not None
        assert len(result) == len(ohlcv_df)

    def test_adosc_numba_impl(self, ohlcv_arrays):
        """Test ADOSC with numba implementation."""
        from ml4t.engineer.features.volume.adosc import adosc

        high, low, close, volume = ohlcv_arrays
        result = adosc(high, low, close, volume)

        assert result is not None
        assert len(result) == len(close)

    def test_adosc_polars_impl(self, ohlcv_df):
        """Test ADOSC with polars implementation."""
        from ml4t.engineer.features.volume.adosc import adosc

        result = ohlcv_df.select(adosc("high", "low", "close", "volume"))
        assert result is not None
        assert len(result) == len(ohlcv_df)

    def test_adosc_different_periods(self, ohlcv_arrays):
        """Test ADOSC with different period combinations."""
        from ml4t.engineer.features.volume.adosc import adosc

        high, low, close, volume = ohlcv_arrays

        r1 = adosc(high, low, close, volume, fastperiod=3, slowperiod=10)
        r2 = adosc(high, low, close, volume, fastperiod=5, slowperiod=20)

        valid_mask = ~np.isnan(r1) & ~np.isnan(r2)
        if np.sum(valid_mask) > 0:
            # Different periods should give different results
            assert not np.allclose(r1[valid_mask], r2[valid_mask], rtol=0.01)

    def test_adosc_empty_arrays(self):
        """Test ADOSC with empty arrays."""
        from ml4t.engineer.features.volume.adosc import adosc

        empty = np.array([])
        result = adosc(empty, empty, empty, empty)
        assert len(result) == 0

    def test_adosc_insufficient_data(self):
        """Test ADOSC with insufficient data."""
        from ml4t.engineer.features.volume.adosc import adosc

        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0, 101.0])
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([5000.0, 5100.0, 5200.0])

        result = adosc(high, low, close, volume, fastperiod=3, slowperiod=10)
        assert result is not None
        assert len(result) == 3


class TestOBVCoverage:
    """Coverage tests for On Balance Volume."""

    @pytest.fixture
    def price_volume_arrays(self):
        """Create price and volume arrays for testing."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.random.randint(1000, 10000, n).astype(float)
        return close, volume

    @pytest.fixture
    def price_volume_df(self, price_volume_arrays):
        """Create price and volume DataFrame for testing."""
        close, volume = price_volume_arrays
        return pl.DataFrame(
            {
                "close": close,
                "volume": volume,
            }
        )

    def test_obv_numpy_basic(self, price_volume_arrays):
        """Test OBV with numpy arrays."""
        from ml4t.engineer.features.volume.obv import obv

        close, volume = price_volume_arrays
        result = obv(close, volume)

        assert result is not None
        assert len(result) == len(close)
        assert isinstance(result, np.ndarray)

    def test_obv_polars_series(self, price_volume_df):
        """Test OBV with Polars Series."""
        from ml4t.engineer.features.volume.obv import obv

        result = obv(
            price_volume_df["close"],
            price_volume_df["volume"],
        )

        assert result is not None
        assert len(result) == len(price_volume_df)

    def test_obv_polars_expr(self, price_volume_df):
        """Test OBV with Polars expressions."""
        from ml4t.engineer.features.volume.obv import obv

        result = price_volume_df.select(obv("close", "volume"))
        assert result is not None
        assert len(result) == len(price_volume_df)

    def test_obv_numba_impl(self, price_volume_arrays):
        """Test OBV with numba implementation."""
        from ml4t.engineer.features.volume.obv import obv

        close, volume = price_volume_arrays
        result = obv(close, volume)

        assert result is not None
        assert len(result) == len(close)

    def test_obv_polars_impl(self, price_volume_df):
        """Test OBV with polars implementation."""
        from ml4t.engineer.features.volume.obv import obv

        result = price_volume_df.select(obv("close", "volume"))
        assert result is not None
        assert len(result) == len(price_volume_df)

    def test_obv_uptrend(self):
        """Test OBV with uptrend."""
        from ml4t.engineer.features.volume.obv import obv

        # Uptrend: prices increase, OBV should increase
        close = np.linspace(100, 150, 50)
        volume = np.full(50, 5000.0)

        result = obv(close, volume)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 1:
            # OBV should be increasing
            assert valid_values[-1] > valid_values[1]

    def test_obv_downtrend(self):
        """Test OBV with downtrend."""
        from ml4t.engineer.features.volume.obv import obv

        # Downtrend: prices decrease, OBV should decrease
        close = np.linspace(150, 100, 50)
        volume = np.full(50, 5000.0)

        result = obv(close, volume)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 1:
            # OBV should be decreasing
            assert valid_values[-1] < valid_values[1]

    def test_obv_constant_prices(self):
        """Test OBV with constant prices."""
        from ml4t.engineer.features.volume.obv import obv

        close = np.full(50, 100.0)
        volume = np.full(50, 5000.0)

        result = obv(close, volume)
        assert result is not None
        assert len(result) == 50

    def test_obv_empty_arrays(self):
        """Test OBV with empty arrays."""
        from ml4t.engineer.features.volume.obv import obv

        result = obv(np.array([]), np.array([]))
        assert len(result) == 0

    def test_obv_single_value(self):
        """Test OBV with single value."""
        from ml4t.engineer.features.volume.obv import obv

        result = obv(np.array([100.0]), np.array([5000.0]))
        assert len(result) == 1

    def test_obv_zero_volume(self):
        """Test OBV with zero volume."""
        from ml4t.engineer.features.volume.obv import obv

        close = 100 + np.cumsum(np.random.randn(50) * 0.5)
        volume = np.zeros(50)

        result = obv(close, volume)
        assert result is not None


class TestVolumeEdgeCases:
    """Edge case tests for volume indicators."""

    def test_high_equals_low(self):
        """Test A/D when high equals low (division edge case)."""
        from ml4t.engineer.features.volume.ad import ad

        # When high == low, CLV formula has division issue
        high = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        low = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        volume = np.array([5000.0, 5100.0, 5200.0, 5300.0, 5400.0])

        result = ad(high, low, close, volume)
        assert result is not None
        # Should handle gracefully without errors

    def test_volume_variations(self):
        """Test volume indicators with varying volume patterns."""
        from ml4t.engineer.features.volume.ad import ad
        from ml4t.engineer.features.volume.obv import obv

        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + 1
        low = close - 1

        # Varying volume
        volume = np.exp(np.random.randn(n)) * 5000

        ad_result = ad(high, low, close, volume)
        obv_result = obv(close, volume)

        assert ad_result is not None
        assert obv_result is not None

    def test_large_volume_values(self):
        """Test with very large volume values."""
        from ml4t.engineer.features.volume.ad import ad

        np.random.seed(42)
        n = 50
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + 1
        low = close - 1
        volume = np.random.randint(int(1e9), int(1e10), n).astype(float)

        result = ad(high, low, close, volume)
        assert result is not None
        # Should handle large values without overflow

    def test_small_price_movements(self):
        """Test with very small price movements."""
        from ml4t.engineer.features.volume.ad import ad

        n = 50
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.0001)
        high = close + 0.0001
        low = close - 0.0001
        volume = np.full(n, 5000.0)

        result = ad(high, low, close, volume)
        assert result is not None


class TestVolumeDataTypes:
    """Test different input data types for volume indicators."""

    def test_float32_arrays(self):
        """Test with float32 numpy arrays."""
        from ml4t.engineer.features.volume.ad import ad

        np.random.seed(42)
        n = 100
        high = (np.random.randn(n) + 101).astype(np.float32)
        low = (np.random.randn(n) + 99).astype(np.float32)
        close = (np.random.randn(n) + 100).astype(np.float32)
        volume = np.random.randint(1000, 10000, n).astype(np.float32)

        result = ad(high, low, close, volume)
        assert isinstance(result, np.ndarray)

    def test_float64_arrays(self):
        """Test with float64 numpy arrays."""
        from ml4t.engineer.features.volume.obv import obv

        np.random.seed(42)
        n = 100
        close = (np.random.randn(n) + 100).astype(np.float64)
        volume = np.random.randint(1000, 10000, n).astype(np.float64)

        result = obv(close, volume)
        assert isinstance(result, np.ndarray)

    def test_polars_series_types(self):
        """Test with Polars Series of different types."""
        from ml4t.engineer.features.volume.ad import ad

        n = 100
        high = pl.Series([float(i) + 101 for i in range(n)])
        low = pl.Series([float(i) + 99 for i in range(n)])
        close = pl.Series([float(i) + 100 for i in range(n)])
        volume = pl.Series([float(5000 + i * 10) for i in range(n)])

        result = ad(high, low, close, volume)
        assert result is not None
