"""Tests for OBV (On Balance Volume) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.volume.obv import obv


@pytest.fixture
def price_volume_data():
    """Generate price and volume test data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volume = np.random.randint(1000, 10000, n).astype(float)
    return {"close": close, "volume": volume}


@pytest.fixture
def ohlcv_df(price_volume_data):
    """Standard OHLCV DataFrame for testing."""
    close = price_volume_data["close"]
    volume = price_volume_data["volume"]
    n = len(close)

    return pl.DataFrame(
        {
            "open": close - np.random.rand(n) * 0.5,
            "high": close + np.random.rand(n),
            "low": close - np.random.rand(n),
            "close": close,
            "volume": volume,
        }
    )


class TestBasicFunctionality:
    """Test basic OBV functionality."""

    def test_computes_successfully_numba(self, price_volume_data):
        """Test OBV computes without errors using NumPy arrays."""
        result = obv(price_volume_data["close"], price_volume_data["volume"])
        assert result is not None
        assert len(result) == len(price_volume_data["close"])
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_series(self, price_volume_data):
        """Test OBV computes with Polars Series."""
        close = pl.Series(price_volume_data["close"])
        volume = pl.Series(price_volume_data["volume"])

        result = obv(close, volume)
        assert result is not None
        assert len(result) == len(price_volume_data["close"])

    def test_computes_successfully_polars_expr(self, ohlcv_df):
        """Test OBV computes with Polars expression."""
        result = ohlcv_df.select(obv("close", "volume").alias("obv"))
        assert result is not None
        assert len(result) == len(ohlcv_df)
        assert "obv" in result.columns

    def test_no_default_parameters(self, price_volume_data):
        """Test that OBV requires close and volume parameters."""
        # OBV doesn't have optional parameters like periods
        result = obv(price_volume_data["close"], price_volume_data["volume"])
        assert result is not None


class TestOBVLogic:
    """Test OBV calculation logic."""

    def test_initialization(self):
        """Test that OBV starts with first volume value."""
        close = np.array([100.0, 101.0, 102.0, 101.0, 100.0])
        volume = np.array([1000.0, 1100.0, 1200.0, 900.0, 800.0])

        result = obv(close, volume)

        # First value should be first volume
        assert result[0] == volume[0]

    def test_price_increase_adds_volume(self):
        """Test that price increase adds volume to OBV."""
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([1000.0, 1100.0, 1200.0])

        result = obv(close, volume)

        # Each increase should add volume
        assert result[0] == 1000.0
        assert result[1] == 1000.0 + 1100.0
        assert result[2] == 1000.0 + 1100.0 + 1200.0

    def test_price_decrease_subtracts_volume(self):
        """Test that price decrease subtracts volume from OBV."""
        close = np.array([100.0, 99.0, 98.0])
        volume = np.array([1000.0, 1100.0, 1200.0])

        result = obv(close, volume)

        # Each decrease should subtract volume
        assert result[0] == 1000.0
        assert result[1] == 1000.0 - 1100.0
        assert result[2] == 1000.0 - 1100.0 - 1200.0

    def test_price_unchanged_maintains_obv(self):
        """Test that unchanged price maintains OBV."""
        close = np.array([100.0, 100.0, 100.0])
        volume = np.array([1000.0, 1100.0, 1200.0])

        result = obv(close, volume)

        # Unchanged price should not change OBV
        assert result[0] == 1000.0
        assert result[1] == 1000.0  # No change
        assert result[2] == 1000.0  # No change

    def test_mixed_price_movement(self):
        """Test OBV with mixed price movements."""
        close = np.array([100.0, 101.0, 101.0, 100.0, 102.0])
        volume = np.array([1000.0, 1100.0, 1200.0, 900.0, 1300.0])

        result = obv(close, volume)

        expected = [
            1000.0,  # Initial
            1000.0 + 1100.0,  # Up
            1000.0 + 1100.0,  # Unchanged
            1000.0 + 1100.0 - 900.0,  # Down
            1000.0 + 1100.0 - 900.0 + 1300.0,  # Up
        ]

        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        empty = np.array([])
        result = obv(empty, empty)
        assert len(result) == 0

    def test_single_value(self):
        """Test with single value."""
        close = np.array([100.0])
        volume = np.array([1000.0])
        result = obv(close, volume)

        assert len(result) == 1
        assert result[0] == volume[0]

    def test_two_values(self):
        """Test with two values."""
        close = np.array([100.0, 101.0])
        volume = np.array([1000.0, 1100.0])
        result = obv(close, volume)

        assert len(result) == 2
        assert result[0] == 1000.0
        assert result[1] == 2100.0  # 1000 + 1100

    def test_constant_prices(self):
        """Test with constant prices."""
        close = np.ones(50) * 100.0
        volume = np.ones(50) * 1000.0

        result = obv(close, volume)

        # OBV should remain constant (first volume value)
        assert all(result == 1000.0)

    def test_mismatched_lengths(self):
        """Test that mismatched array lengths raise error."""
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([1000.0, 1100.0])  # Different length

        with pytest.raises(ValueError, match="same length"):
            obv(close, volume)

    def test_polars_empty_dataframe(self):
        """Test with empty Polars DataFrame."""
        empty_df = pl.DataFrame(
            {
                "close": pl.Series([], dtype=pl.Float64),
                "volume": pl.Series([], dtype=pl.Float64),
            }
        )
        result = empty_df.select(obv("close", "volume").alias("obv"))
        assert len(result) == 0

    def test_polars_single_row(self):
        """Test with single row DataFrame."""
        single_df = pl.DataFrame(
            {
                "close": [100.0],
                "volume": [1000.0],
            }
        )
        result = single_df.select(obv("close", "volume").alias("obv"))
        assert len(result) == 1
        assert result["obv"][0] == 1000.0


class TestNumericalProperties:
    """Test numerical properties of OBV."""

    def test_cumulative_nature(self):
        """Test that OBV is cumulative."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        volume = np.array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0])

        result = obv(close, volume)

        # Each value should be based on cumulative volume
        # All prices increasing, so OBV should be strictly increasing
        assert all(np.diff(result) > 0)

    def test_trend_confirmation(self):
        """Test that OBV confirms price trends."""
        # Uptrend
        up_close = np.linspace(100, 120, 20)
        up_volume = np.ones(20) * 1000.0

        result_up = obv(up_close, up_volume)

        # OBV should be increasing
        assert result_up[-1] > result_up[0]

        # Downtrend
        down_close = np.linspace(120, 100, 20)
        down_volume = np.ones(20) * 1000.0

        result_down = obv(down_close, down_volume)

        # OBV should be decreasing
        assert result_down[-1] < result_down[0]

    def test_volume_sensitivity(self):
        """Test that OBV is sensitive to volume changes."""
        # Low volume
        close = np.array([100.0, 101.0, 102.0, 103.0])
        low_volume = np.array([100.0, 100.0, 100.0, 100.0])

        result_low = obv(close, low_volume)

        # High volume
        high_volume = np.array([1000.0, 1000.0, 1000.0, 1000.0])

        result_high = obv(close, high_volume)

        # High volume should produce larger OBV values
        assert abs(result_high[-1] - result_high[0]) > abs(result_low[-1] - result_low[0])

    def test_divergence_detection(self):
        """Test OBV can detect divergences."""
        # Price making new high, but OBV not confirming (bearish divergence)
        close = np.array([100.0, 110.0, 105.0, 115.0])  # New high at end
        # But volume decreasing on rallies
        volume = np.array([1000.0, 500.0, 600.0, 300.0])

        result = obv(close, volume)

        # OBV might not make new high with price due to lower volume
        # This is just testing the calculation works correctly
        assert result is not None
        assert len(result) == len(close)


class TestImplementationConsistency:
    """Test that different implementations produce consistent results."""

    def test_numba_vs_polars_consistency(self, price_volume_data):
        """Test NumPy/Numba vs Polars implementation consistency."""
        # NumPy/Numba
        result_numba = obv(price_volume_data["close"], price_volume_data["volume"])

        # Polars
        df = pl.DataFrame(price_volume_data)
        result_polars = df.select(obv("close", "volume").alias("obv"))["obv"].to_numpy()

        # Should produce identical results
        np.testing.assert_allclose(
            result_numba,
            result_polars,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_series_vs_array_consistency(self, price_volume_data):
        """Test Polars Series vs NumPy array consistency."""
        # NumPy arrays
        result_array = obv(price_volume_data["close"], price_volume_data["volume"])

        # Polars Series
        close = pl.Series(price_volume_data["close"])
        volume = pl.Series(price_volume_data["volume"])
        result_series = obv(close, volume)

        # Should be identical
        np.testing.assert_allclose(
            result_array,
            result_series,
            rtol=1e-10,
            equal_nan=True,
        )


class TestSpecialCases:
    """Test special mathematical cases."""

    def test_all_price_increases(self):
        """Test with all price increases."""
        close = np.arange(1, 51, dtype=float)
        volume = np.ones(50) * 1000.0

        result = obv(close, volume)

        # Should be strictly increasing
        assert all(np.diff(result) == 1000.0)

    def test_all_price_decreases(self):
        """Test with all price decreases."""
        close = np.arange(50, 0, -1, dtype=float)
        volume = np.ones(50) * 1000.0

        result = obv(close, volume)

        # Should be strictly decreasing
        assert all(np.diff(result) == -1000.0)

    def test_alternating_prices(self):
        """Test with alternating price movements."""
        close = np.array([100.0, 101.0, 100.0, 101.0, 100.0])
        volume = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

        result = obv(close, volume)

        # Should oscillate around initial value
        assert result[0] == 1000.0
        assert result[1] == 2000.0  # +1000
        assert result[2] == 1000.0  # -1000
        assert result[3] == 2000.0  # +1000
        assert result[4] == 1000.0  # -1000

    def test_zero_volume(self):
        """Test with zero volume."""
        close = np.array([100.0, 101.0, 102.0, 101.0])
        volume = np.array([0.0, 0.0, 0.0, 0.0])

        result = obv(close, volume)

        # OBV should remain at 0
        assert all(result == 0.0)

    def test_large_volume_values(self):
        """Test with large volume values."""
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([1e9, 2e9, 3e9])

        result = obv(close, volume)

        # Should handle large values
        assert result[0] == 1e9
        assert result[1] == 1e9 + 2e9
        assert result[2] == 1e9 + 2e9 + 3e9

    def test_small_price_changes(self):
        """Test with very small price changes."""
        close = np.array([100.0, 100.0001, 100.0002, 99.9999])
        volume = np.array([1000.0, 1100.0, 1200.0, 900.0])

        result = obv(close, volume)

        # Should still detect tiny changes
        assert result[0] == 1000.0
        assert result[1] == 1000.0 + 1100.0  # Tiny increase
        assert result[2] == 1000.0 + 1100.0 + 1200.0  # Tiny increase
        assert result[3] == 1000.0 + 1100.0 + 1200.0 - 900.0  # Tiny decrease


class TestInterpretation:
    """Test OBV interpretation and usage."""

    def test_accumulation_distribution(self):
        """Test that OBV reflects accumulation/distribution."""
        # Accumulation: price up with volume
        close_acc = np.array([100.0, 102.0, 104.0, 106.0])
        volume_acc = np.array([1000.0, 1500.0, 2000.0, 2500.0])

        result_acc = obv(close_acc, volume_acc)

        # Distribution: price down with volume
        close_dist = np.array([106.0, 104.0, 102.0, 100.0])
        volume_dist = np.array([1000.0, 1500.0, 2000.0, 2500.0])

        result_dist = obv(close_dist, volume_dist)

        # Accumulation should have rising OBV
        assert result_acc[-1] > result_acc[0]

        # Distribution should have falling OBV
        assert result_dist[-1] < result_dist[0]

    def test_no_lookback_period(self):
        """Test that OBV has no lookback (starts immediately)."""
        close = np.array([100.0, 101.0, 102.0])
        volume = np.array([1000.0, 1100.0, 1200.0])

        result = obv(close, volume)

        # All values should be valid (no NaN)
        assert not any(np.isnan(result))
        # First value should be valid
        assert result[0] == volume[0]
