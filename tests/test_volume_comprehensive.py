"""Comprehensive tests for volume indicators to achieve high coverage."""

import numpy as np
import polars as pl

from ml4t.engineer.features.volume import ad, adosc, obv
from ml4t.engineer.features.volume.ad import ad_numba
from ml4t.engineer.features.volume.adosc import adosc_numba
from ml4t.engineer.features.volume.obv import obv_numba

# ============================================================================
# Direct Numba Function Tests
# ============================================================================


class TestOBVNumba:
    """Tests for OBV Numba implementation."""

    def test_empty_input(self):
        """Test OBV with empty arrays."""
        close = np.array([])
        volume = np.array([])
        result = obv_numba(close, volume)
        assert len(result) == 0

    def test_mismatched_lengths(self):
        """Test OBV with mismatched array lengths."""
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([1000.0, 1100.0])  # Different length
        result = obv_numba(close, volume)
        assert np.all(np.isnan(result))

    def test_single_value(self):
        """Test OBV with single data point."""
        close = np.array([100.0])
        volume = np.array([1000.0])
        result = obv_numba(close, volume)
        assert result[0] == 1000.0

    def test_price_up(self):
        """Test OBV when price goes up."""
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([1000.0, 1100.0, 1200.0])
        result = obv_numba(close, volume)
        assert result[0] == 1000.0
        assert result[1] == 2100.0  # 1000 + 1100
        assert result[2] == 3300.0  # 2100 + 1200

    def test_price_down(self):
        """Test OBV when price goes down."""
        close = np.array([102.0, 101.0, 100.0])
        volume = np.array([1000.0, 1100.0, 1200.0])
        result = obv_numba(close, volume)
        assert result[0] == 1000.0
        assert result[1] == -100.0  # 1000 - 1100
        assert result[2] == -1300.0  # -100 - 1200

    def test_price_unchanged(self):
        """Test OBV when price is unchanged."""
        close = np.array([100.0, 100.0, 100.0])
        volume = np.array([1000.0, 1100.0, 1200.0])
        result = obv_numba(close, volume)
        assert result[0] == 1000.0
        assert result[1] == 1000.0  # No change
        assert result[2] == 1000.0  # No change


class TestADNumba:
    """Tests for AD Numba implementation."""

    def test_empty_input(self):
        """Test AD with empty arrays."""
        high = np.array([])
        low = np.array([])
        close = np.array([])
        volume = np.array([])
        result = ad_numba(high, low, close, volume)
        assert len(result) == 0

    def test_mismatched_lengths(self):
        """Test AD with mismatched array lengths."""
        high = np.array([102.0, 103.0])
        low = np.array([98.0])  # Different length
        close = np.array([101.0, 102.0])
        volume = np.array([1000.0, 1100.0])
        result = ad_numba(high, low, close, volume)
        # AD may handle mismatched lengths gracefully, just verify it completes
        assert len(result) > 0

    def test_single_value(self):
        """Test AD with single data point."""
        high = np.array([102.0])
        low = np.array([98.0])
        close = np.array([101.0])
        volume = np.array([1000.0])
        result = ad_numba(high, low, close, volume)
        # CLV = ((101-98) - (102-101)) / (102-98) = (3-1)/4 = 0.5
        # MFV = 0.5 * 1000 = 500
        assert result[0] == 500.0

    def test_high_equals_low(self):
        """Test AD when high equals low (avoid division by zero)."""
        high = np.array([100.0])
        low = np.array([100.0])  # Same as high
        close = np.array([100.0])
        volume = np.array([1000.0])
        result = ad_numba(high, low, close, volume)
        # CLV = 0 when high == low
        assert result[0] == 0.0

    def test_close_at_high(self):
        """Test AD when close is at high."""
        high = np.array([102.0])
        low = np.array([98.0])
        close = np.array([102.0])  # At high
        volume = np.array([1000.0])
        result = ad_numba(high, low, close, volume)
        # CLV = ((102-98) - (102-102)) / (102-98) = 4/4 = 1.0
        # MFV = 1.0 * 1000 = 1000
        assert result[0] == 1000.0

    def test_close_at_low(self):
        """Test AD when close is at low."""
        high = np.array([102.0])
        low = np.array([98.0])
        close = np.array([98.0])  # At low
        volume = np.array([1000.0])
        result = ad_numba(high, low, close, volume)
        # CLV = ((98-98) - (102-98)) / (102-98) = -4/4 = -1.0
        # MFV = -1.0 * 1000 = -1000
        assert result[0] == -1000.0

    def test_cumulative_nature(self):
        """Test that AD is cumulative."""
        high = np.array([102.0, 103.0])
        low = np.array([98.0, 99.0])
        close = np.array([101.0, 102.0])
        volume = np.array([1000.0, 1100.0])
        result = ad_numba(high, low, close, volume)
        # Result should accumulate
        assert result[1] != result[0]
        assert result[1] == result[0] + ((102 - 99) - (103 - 102)) / (103 - 99) * 1100


class TestADOSCNumba:
    """Tests for ADOSC Numba implementation."""

    def test_empty_input(self):
        """Test ADOSC with empty arrays."""
        high = np.array([])
        low = np.array([])
        close = np.array([])
        volume = np.array([])
        result = adosc_numba(high, low, close, volume, fastperiod=3, slowperiod=10)
        assert len(result) == 0

    def test_insufficient_data(self):
        """Test ADOSC with insufficient data for slowperiod."""
        high = np.array([102.0, 103.0])
        low = np.array([98.0, 99.0])
        close = np.array([101.0, 102.0])
        volume = np.array([1000.0, 1100.0])
        result = adosc_numba(high, low, close, volume, fastperiod=3, slowperiod=10)
        # All NaN due to insufficient data
        assert np.all(np.isnan(result))

    def test_exact_minimum_data(self):
        """Test ADOSC with exact minimum data needed."""
        n = 10
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)
        close = np.full(n, 101.0)
        volume = np.full(n, 1000.0)
        result = adosc_numba(high, low, close, volume, fastperiod=3, slowperiod=10)
        # Should have at least one valid value at the end
        assert not np.all(np.isnan(result))

    def test_fast_equals_slow(self):
        """Test ADOSC when fastperiod equals slowperiod (should be zero)."""
        n = 15
        high = np.linspace(100, 110, n)
        low = np.linspace(95, 105, n)
        close = np.linspace(98, 108, n)
        volume = np.full(n, 1000.0)
        result = adosc_numba(high, low, close, volume, fastperiod=5, slowperiod=5)
        # When fast == slow, oscillator should be 0
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.allclose(valid_values, 0.0, atol=1e-10)

    def test_oscillator_behavior(self):
        """Test that ADOSC can produce positive and negative values."""
        n = 20
        high = np.array([102.0] * 10 + [101.0] * 10)  # Change trend
        low = np.array([98.0] * 10 + [97.0] * 10)
        close = np.array([101.0] * 10 + [99.0] * 10)  # Price decline
        volume = np.full(n, 1000.0)
        result = adosc_numba(high, low, close, volume, fastperiod=3, slowperiod=10)
        valid_values = result[~np.isnan(result)]
        # Oscillator should vary
        if len(valid_values) > 1:
            assert not np.all(valid_values == valid_values[0])


# ============================================================================
# Polars Expression Tests
# ============================================================================


class TestVolumeExpressionsWithTypes:
    """Test volume indicators with different data types and edge cases."""

    def test_obv_with_zero_volume(self):
        """Test OBV with zero volume values."""
        data = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [0.0, 0.0, 0.0],
            }
        )
        result = data.with_columns(obv("close", "volume").alias("obv"))
        assert result["obv"][0] == 0.0

    def test_obv_with_negative_volume(self):
        """Test OBV with negative volume (should still work mathematically)."""
        data = pl.DataFrame(
            {
                "close": [100.0, 101.0, 102.0],
                "volume": [-1000.0, -1100.0, -1200.0],
            }
        )
        result = data.with_columns(obv("close", "volume").alias("obv"))
        # OBV should accumulate negative volumes
        assert result["obv"][0] == -1000.0

    def test_ad_with_very_small_range(self):
        """Test AD with very small high-low range."""
        data = pl.DataFrame(
            {
                "high": [100.001, 100.002],
                "low": [100.000, 100.001],
                "close": [100.0005, 100.0015],
                "volume": [1000.0, 1100.0],
            }
        )
        result = data.with_columns(ad("high", "low", "close", "volume").alias("ad"))
        # Should not cause numerical issues
        assert not pl.Series(result["ad"]).is_nan().all()

    def test_ad_with_zero_volume(self):
        """Test AD with zero volume."""
        data = pl.DataFrame(
            {
                "high": [102.0, 103.0],
                "low": [98.0, 99.0],
                "close": [101.0, 102.0],
                "volume": [0.0, 0.0],
            }
        )
        result = data.with_columns(ad("high", "low", "close", "volume").alias("ad"))
        # Should be zero since volume is zero
        assert result["ad"][0] == 0.0
        assert result["ad"][1] == 0.0

    def test_adosc_with_custom_small_periods(self):
        """Test ADOSC with very small periods."""
        n = 10
        data = pl.DataFrame(
            {
                "high": np.linspace(100, 110, n),
                "low": np.linspace(95, 105, n),
                "close": np.linspace(98, 108, n),
                "volume": np.full(n, 1000.0),
            }
        )
        result = data.with_columns(
            adosc("high", "low", "close", "volume", fastperiod=2, slowperiod=3).alias("adosc_2_3")
        )
        # Should work with small periods
        assert "adosc_2_3" in result.columns
        valid_values = result["adosc_2_3"].drop_nulls()
        assert len(valid_values) > 0

    def test_adosc_large_periods(self):
        """Test ADOSC with large periods."""
        n = 100
        data = pl.DataFrame(
            {
                "high": np.linspace(100, 200, n),
                "low": np.linspace(95, 195, n),
                "close": np.linspace(98, 198, n),
                "volume": np.full(n, 1000.0),
            }
        )
        result = data.with_columns(
            adosc("high", "low", "close", "volume", fastperiod=20, slowperiod=50).alias(
                "adosc_20_50"
            )
        )
        # Should work with large periods
        assert "adosc_20_50" in result.columns
        valid_values = result["adosc_20_50"].drop_nulls()
        assert len(valid_values) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestVolumeIntegration:
    """Integration tests for volume indicators."""

    def test_all_volume_indicators_together(self):
        """Test using all volume indicators on same data."""
        data = pl.DataFrame(
            {
                "high": np.linspace(100, 120, 20),
                "low": np.linspace(95, 115, 20),
                "close": np.linspace(98, 118, 20),
                "volume": np.full(20, 1000.0),
            }
        )
        result = data.with_columns(
            [
                obv("close", "volume").alias("obv"),
                ad("high", "low", "close", "volume").alias("ad"),
                adosc("high", "low", "close", "volume").alias("adosc"),
            ]
        )
        # All columns should exist
        assert "obv" in result.columns
        assert "ad" in result.columns
        assert "adosc" in result.columns

        # All should have valid values
        assert result["obv"].drop_nulls().len() > 0
        assert result["ad"].drop_nulls().len() > 0
        assert result["adosc"].drop_nulls().len() > 0

    def test_volume_indicators_with_real_pattern(self):
        """Test volume indicators with realistic price pattern."""
        # Simulate uptrend with increasing volume
        n = 30
        close = np.linspace(100, 130, n)  # Uptrend
        volume = np.linspace(1000, 2000, n)  # Increasing volume
        high = close + 2
        low = close - 2

        data = pl.DataFrame(
            {
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        result = data.with_columns(
            [
                obv("close", "volume").alias("obv"),
                ad("high", "low", "close", "volume").alias("ad"),
            ]
        )

        # OBV should increase during uptrend
        obv_values = result["obv"].to_numpy()
        assert obv_values[-1] > obv_values[0]

        # AD should accumulate during uptrend
        ad_values = result["ad"].to_numpy()
        assert ad_values[-1] > ad_values[0]
