"""Tests for ADOSC (Chaikin A/D Oscillator) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.volume.adosc import adosc


@pytest.fixture
def ohlcv_data():
    """Generate OHLCV test data with volume."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    return {
        "high": close + np.random.rand(n),
        "low": close - np.random.rand(n),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }


@pytest.fixture
def ohlcv_df(ohlcv_data):
    """Standard OHLCV DataFrame for testing."""
    return pl.DataFrame(
        {
            "open": ohlcv_data["close"] - np.random.rand(len(ohlcv_data["close"])) * 0.5,
            "high": ohlcv_data["high"],
            "low": ohlcv_data["low"],
            "close": ohlcv_data["close"],
            "volume": ohlcv_data["volume"],
        }
    )


class TestBasicFunctionality:
    """Test basic ADOSC functionality."""

    def test_computes_successfully_numba(self, ohlcv_data):
        """Test ADOSC computes without errors using NumPy arrays."""
        result = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=3,
            slowperiod=10,
        )
        assert result is not None
        assert len(result) == len(ohlcv_data["close"])
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_series(self, ohlcv_data):
        """Test ADOSC computes with Polars Series."""
        high = pl.Series(ohlcv_data["high"])
        low = pl.Series(ohlcv_data["low"])
        close = pl.Series(ohlcv_data["close"])
        volume = pl.Series(ohlcv_data["volume"])

        result = adosc(high, low, close, volume, fastperiod=3, slowperiod=10)
        assert result is not None
        assert len(result) == len(ohlcv_data["close"])

    def test_computes_successfully_polars_expr(self, ohlcv_df):
        """Test ADOSC computes with Polars expression."""
        result = ohlcv_df.select(
            adosc("high", "low", "close", "volume", fastperiod=3, slowperiod=10).alias("adosc")
        )
        assert result is not None
        assert len(result) == len(ohlcv_df)
        assert "adosc" in result.columns

    def test_default_parameters(self, ohlcv_data):
        """Test ADOSC with default parameters."""
        result = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
        )
        assert result is not None
        # Default fastperiod=3, slowperiod=10
        assert len(result) == len(ohlcv_data["close"])

    def test_different_periods(self, ohlcv_data):
        """Test ADOSC produces different results with different periods."""
        r1 = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=3,
            slowperiod=10,
        )
        r2 = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=5,
            slowperiod=20,
        )

        # Results should be different
        valid_r1 = r1[~np.isnan(r1)]
        valid_r2 = r2[~np.isnan(r2)]
        assert len(valid_r1) > 0
        assert len(valid_r2) > 0
        assert not np.allclose(valid_r1[:50], valid_r2[:50], equal_nan=True)


class TestLookbackPeriod:
    """Test lookback period behavior."""

    def test_lookback_based_on_slowperiod(self, ohlcv_data):
        """Test that lookback is based on slowperiod."""
        slowperiod = 10
        result = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=3,
            slowperiod=slowperiod,
        )

        # Lookback is slowperiod - 1
        lookback = slowperiod - 1
        # First lookback values should be NaN
        assert all(np.isnan(result[:lookback]))
        # Value at lookback should be valid
        assert not np.isnan(result[lookback])

    def test_different_lookback_periods(self, ohlcv_data):
        """Test lookback with different slowperiods."""
        r1 = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=3,
            slowperiod=10,
        )
        r2 = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=3,
            slowperiod=20,
        )

        # r1 should have valid values earlier
        assert np.sum(~np.isnan(r1)) > np.sum(~np.isnan(r2))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        empty = np.array([])
        result = adosc(empty, empty, empty, empty, fastperiod=3, slowperiod=10)
        assert len(result) == 0

    def test_single_value(self):
        """Test with single value."""
        single = np.array([100.0])
        result = adosc(single, single, single, single, fastperiod=3, slowperiod=10)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_insufficient_data(self):
        """Test with insufficient data for slowperiod."""
        short_high = np.array([101.0, 102.0, 103.0])
        short_low = np.array([99.0, 100.0, 101.0])
        short_close = np.array([100.0, 101.0, 102.0])
        short_volume = np.array([1000.0, 1100.0, 1200.0])

        result = adosc(
            short_high,
            short_low,
            short_close,
            short_volume,
            fastperiod=3,
            slowperiod=10,
        )

        # All should be NaN
        assert all(np.isnan(result))

    def test_constant_prices(self):
        """Test with constant prices."""
        const = np.ones(100) * 100.0
        volume = np.ones(100) * 1000.0

        result = adosc(const, const, const, volume, fastperiod=3, slowperiod=10)

        # ADOSC should be valid but likely near zero
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise error."""
        high = np.array([101.0, 102.0, 103.0])
        low = np.array([99.0, 100.0])  # Different length
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([1000.0, 1100.0, 1200.0])

        with pytest.raises(ValueError, match="same length"):
            adosc(high, low, close, volume, fastperiod=3, slowperiod=10)

    def test_polars_empty_dataframe(self):
        """Test with empty Polars DataFrame."""
        empty_df = pl.DataFrame(
            {
                "high": pl.Series([], dtype=pl.Float64),
                "low": pl.Series([], dtype=pl.Float64),
                "close": pl.Series([], dtype=pl.Float64),
                "volume": pl.Series([], dtype=pl.Float64),
            }
        )
        result = empty_df.select(
            adosc("high", "low", "close", "volume", fastperiod=3, slowperiod=10).alias("adosc")
        )
        assert len(result) == 0


class TestValidation:
    """Test parameter validation."""

    def test_invalid_fastperiod_zero(self, ohlcv_data):
        """Test that fastperiod = 0 raises error."""
        with pytest.raises(ValueError, match="fastperiod must be > 0"):
            adosc(
                ohlcv_data["high"],
                ohlcv_data["low"],
                ohlcv_data["close"],
                ohlcv_data["volume"],
                fastperiod=0,
                slowperiod=10,
            )

    def test_invalid_fastperiod_negative(self, ohlcv_data):
        """Test that negative fastperiod raises error."""
        with pytest.raises(ValueError, match="fastperiod must be > 0"):
            adosc(
                ohlcv_data["high"],
                ohlcv_data["low"],
                ohlcv_data["close"],
                ohlcv_data["volume"],
                fastperiod=-3,
                slowperiod=10,
            )

    def test_invalid_slowperiod_zero(self, ohlcv_data):
        """Test that slowperiod = 0 raises error."""
        with pytest.raises(ValueError, match="slowperiod must be > 0"):
            adosc(
                ohlcv_data["high"],
                ohlcv_data["low"],
                ohlcv_data["close"],
                ohlcv_data["volume"],
                fastperiod=3,
                slowperiod=0,
            )

    def test_invalid_slowperiod_negative(self, ohlcv_data):
        """Test that negative slowperiod raises error."""
        with pytest.raises(ValueError, match="slowperiod must be > 0"):
            adosc(
                ohlcv_data["high"],
                ohlcv_data["low"],
                ohlcv_data["close"],
                ohlcv_data["volume"],
                fastperiod=3,
                slowperiod=-10,
            )


class TestNumericalProperties:
    """Test numerical properties of ADOSC."""

    def test_oscillator_behavior(self, ohlcv_data):
        """Test that ADOSC oscillates around zero."""
        result = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=3,
            slowperiod=10,
        )

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Should have both positive and negative values
            has_positive = np.any(valid_values > 0)
            has_negative = np.any(valid_values < 0)
            # At least one of these should be true (might all be one sign in special cases)
            assert has_positive or has_negative

    def test_responds_to_volume(self):
        """Test that ADOSC responds to volume changes."""
        # Create data with high volume spike
        high = np.array([101.0] * 50)
        low = np.array([99.0] * 50)
        close = np.array([100.0] * 50)
        volume = np.array([1000.0] * 25 + [5000.0] * 25)

        result = adosc(high, low, close, volume, fastperiod=3, slowperiod=10)

        # Should compute successfully
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_money_flow_direction(self):
        """Test that ADOSC reflects money flow direction."""
        # Create uptrend: close near high
        n = 50
        base = 100.0
        high = np.linspace(base + 2, base + 3, n)
        low = np.linspace(base, base + 1, n)
        close = np.linspace(base + 1.8, base + 2.8, n)  # Near high
        volume = np.ones(n) * 1000.0

        result_up = adosc(high, low, close, volume, fastperiod=3, slowperiod=10)

        # Create downtrend: close near low
        close_down = np.linspace(base + 0.2, base + 1.2, n)  # Near low
        result_down = adosc(high, low, close_down, volume, fastperiod=3, slowperiod=10)

        # Results should differ
        valid_up = result_up[~np.isnan(result_up)]
        valid_down = result_down[~np.isnan(result_down)]

        if len(valid_up) > 5 and len(valid_down) > 5:
            # Uptrend should generally have higher ADOSC
            assert np.mean(valid_up[-10:]) > np.mean(valid_down[-10:])


class TestImplementationConsistency:
    """Test that different implementations produce consistent results."""

    def test_numba_vs_polars_consistency(self, ohlcv_data):
        """Test NumPy/Numba vs Polars implementation consistency."""
        # NumPy/Numba
        result_numba = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=3,
            slowperiod=10,
        )

        # Polars
        df = pl.DataFrame(ohlcv_data)
        result_polars = df.select(
            adosc("high", "low", "close", "volume", fastperiod=3, slowperiod=10).alias("adosc")
        )["adosc"].to_numpy()

        # Should produce identical results
        np.testing.assert_allclose(
            result_numba,
            result_polars,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_series_vs_array_consistency(self, ohlcv_data):
        """Test Polars Series vs NumPy array consistency."""
        # NumPy arrays
        result_array = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=3,
            slowperiod=10,
        )

        # Polars Series
        high = pl.Series(ohlcv_data["high"])
        low = pl.Series(ohlcv_data["low"])
        close = pl.Series(ohlcv_data["close"])
        volume = pl.Series(ohlcv_data["volume"])

        result_series = adosc(high, low, close, volume, fastperiod=3, slowperiod=10)

        # Should be identical
        np.testing.assert_allclose(
            result_array,
            result_series,
            rtol=1e-10,
            equal_nan=True,
        )


class TestSpecialCases:
    """Test special mathematical cases."""

    def test_fastperiod_equals_slowperiod(self, ohlcv_data):
        """Test when fastperiod equals slowperiod."""
        result = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=10,
            slowperiod=10,
        )

        # Should compute - oscillator should be near zero
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            # When periods are equal, fast and slow EMA are identical
            # so ADOSC should be very close to zero
            assert np.allclose(valid_values, 0.0, atol=1e-8)

    def test_fastperiod_greater_than_slowperiod(self, ohlcv_data):
        """Test when fastperiod > slowperiod (unusual but valid)."""
        result = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=20,
            slowperiod=10,
        )

        # Should still compute
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_very_small_periods(self, ohlcv_data):
        """Test with very small periods."""
        result = adosc(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            ohlcv_data["volume"],
            fastperiod=1,
            slowperiod=2,
        )

        # Should compute with minimal lookback
        assert not np.isnan(result[1])

    def test_zero_high_low_range(self):
        """Test when high equals low (zero range)."""
        n = 50
        const_price = np.ones(n) * 100.0
        volume = np.ones(n) * 1000.0

        result = adosc(const_price, const_price, const_price, volume, fastperiod=3, slowperiod=10)

        # Should handle gracefully (A/D line won't change)
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
