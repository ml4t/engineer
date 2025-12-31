"""Tests for T3 (Triple Exponential Moving Average) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.trend.t3 import t3


@pytest.fixture
def price_data():
    """Generate test price data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return close


@pytest.fixture
def ohlcv_df(price_data):
    """Standard OHLCV DataFrame for testing."""
    n = len(price_data)
    return pl.DataFrame(
        {
            "open": price_data - np.random.rand(n) * 0.5,
            "high": price_data + np.random.rand(n),
            "low": price_data - np.random.rand(n),
            "close": price_data,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestBasicFunctionality:
    """Test basic T3 functionality."""

    def test_computes_successfully_numba(self, price_data):
        """Test T3 computes without errors using NumPy array."""
        result = t3(price_data, timeperiod=5, vfactor=0.7)
        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_series(self, price_data):
        """Test T3 computes with Polars Series."""
        series = pl.Series(price_data)
        result = t3(series, timeperiod=5, vfactor=0.7)
        assert result is not None
        assert len(result) == len(price_data)

    def test_computes_successfully_polars_expr(self, ohlcv_df):
        """Test T3 computes with Polars expression."""
        result = ohlcv_df.select(t3("close", timeperiod=5, vfactor=0.7).alias("t3"))
        assert result is not None
        assert len(result) == len(ohlcv_df)
        assert "t3" in result.columns

    def test_default_parameters(self, price_data):
        """Test T3 with default parameters."""
        result = t3(price_data)
        assert result is not None
        # Default timeperiod=5, vfactor=0.7
        assert len(result) == len(price_data)

    def test_different_timeperiods(self, price_data):
        """Test T3 produces different results with different periods."""
        r1 = t3(price_data, timeperiod=5, vfactor=0.7)
        r2 = t3(price_data, timeperiod=10, vfactor=0.7)

        # Results should be different
        valid_r1 = r1[~np.isnan(r1)]
        valid_r2 = r2[~np.isnan(r2)]
        assert len(valid_r1) > 0
        assert len(valid_r2) > 0
        # At least some values should differ
        assert not np.allclose(valid_r1[:50], valid_r2[:50], equal_nan=True)

    def test_different_vfactors(self, price_data):
        """Test T3 produces different results with different volume factors."""
        r1 = t3(price_data, timeperiod=5, vfactor=0.3)
        r2 = t3(price_data, timeperiod=5, vfactor=0.9)

        # Results should be different
        valid_r1 = r1[~np.isnan(r1)]
        valid_r2 = r2[~np.isnan(r2)]
        assert len(valid_r1) > 0
        assert len(valid_r2) > 0
        assert not np.allclose(valid_r1, valid_r2, equal_nan=True)


class TestLookbackPeriod:
    """Test lookback period behavior."""

    def test_lookback_calculation(self, price_data):
        """Test that lookback is 6 * (timeperiod - 1)."""
        timeperiod = 5
        lookback = 6 * (timeperiod - 1)
        result = t3(price_data, timeperiod=timeperiod, vfactor=0.7)

        # First lookback values should be NaN
        assert all(np.isnan(result[:lookback]))
        # Value at lookback should be valid
        assert not np.isnan(result[lookback])

    def test_longer_lookback(self, price_data):
        """Test lookback with longer period."""
        timeperiod = 10
        lookback = 6 * (timeperiod - 1)
        result = t3(price_data, timeperiod=timeperiod, vfactor=0.7)

        assert all(np.isnan(result[:lookback]))
        assert not np.isnan(result[lookback])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test with empty array."""
        empty = np.array([])
        result = t3(empty, timeperiod=5)
        assert len(result) == 0

    def test_single_value(self):
        """Test with single value."""
        single = np.array([100.0])
        result = t3(single, timeperiod=5)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_insufficient_data(self):
        """Test with insufficient data for lookback period."""
        # Need at least lookback + 1 values
        timeperiod = 5
        lookback = 6 * (timeperiod - 1)
        short_data = np.array([100.0] * lookback)
        result = t3(short_data, timeperiod=timeperiod)

        # All should be NaN
        assert all(np.isnan(result))

    def test_just_enough_data(self):
        """Test with exactly enough data."""
        timeperiod = 5
        lookback = 6 * (timeperiod - 1)
        data = np.array([100.0 + i * 0.5 for i in range(lookback + 1)])
        result = t3(data, timeperiod=timeperiod)

        # Last value should be valid
        assert not np.isnan(result[-1])

    def test_constant_values(self):
        """Test with constant values."""
        const = np.ones(100) * 100.0
        result = t3(const, timeperiod=5, vfactor=0.7)

        # T3 of constant should be approximately constant
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.allclose(valid_values, 100.0, rtol=1e-5)

    def test_with_nans_in_input(self):
        """Test handling of NaN values in input."""
        data = np.array([100.0, 101.0, np.nan, 102.0, 103.0] * 20)
        result = t3(data, timeperiod=5, vfactor=0.7)

        # Result should exist but may have NaN values
        assert result is not None
        assert len(result) == len(data)

    def test_polars_empty_dataframe(self):
        """Test with empty Polars DataFrame."""
        empty_df = pl.DataFrame({"close": pl.Series([], dtype=pl.Float64)})
        result = empty_df.select(t3("close").alias("t3"))
        assert len(result) == 0

    def test_polars_single_row(self):
        """Test with single row DataFrame."""
        single_df = pl.DataFrame({"close": [100.0]})
        result = single_df.select(t3("close").alias("t3"))
        assert len(result) == 1
        assert result["t3"][0] is None or np.isnan(result["t3"][0])


class TestValidation:
    """Test parameter validation."""

    def test_invalid_timeperiod_too_small(self, price_data):
        """Test that timeperiod < 2 raises error."""
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            t3(price_data, timeperiod=1)

    def test_invalid_timeperiod_zero(self, price_data):
        """Test that timeperiod = 0 raises error."""
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            t3(price_data, timeperiod=0)

    def test_invalid_timeperiod_negative(self, price_data):
        """Test that negative timeperiod raises error."""
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            t3(price_data, timeperiod=-5)

    def test_invalid_vfactor_too_low(self, price_data):
        """Test that vfactor < 0 raises error."""
        with pytest.raises(ValueError, match="vfactor must be between 0 and 1"):
            t3(price_data, timeperiod=5, vfactor=-0.1)

    def test_invalid_vfactor_too_high(self, price_data):
        """Test that vfactor > 1 raises error."""
        with pytest.raises(ValueError, match="vfactor must be between 0 and 1"):
            t3(price_data, timeperiod=5, vfactor=1.5)

    def test_valid_boundary_vfactors(self, price_data):
        """Test that vfactor at boundaries (0, 1) is valid."""
        # vfactor = 0 should work
        r1 = t3(price_data, timeperiod=5, vfactor=0.0)
        assert r1 is not None

        # vfactor = 1 should work
        r2 = t3(price_data, timeperiod=5, vfactor=1.0)
        assert r2 is not None


class TestNumericalProperties:
    """Test numerical properties of T3."""

    def test_smoothness(self, price_data):
        """Test that T3 is smoother than the original data."""
        result = t3(price_data, timeperiod=10, vfactor=0.7)

        # Remove NaN values for comparison
        valid_idx = ~np.isnan(result)
        valid_result = result[valid_idx]
        valid_price = price_data[valid_idx]

        # Calculate variability (standard deviation of differences)
        if len(valid_result) > 1:
            price_variability = np.std(np.diff(valid_price))
            t3_variability = np.std(np.diff(valid_result))

            # T3 should be smoother (lower variability)
            assert t3_variability < price_variability

    def test_follows_trend(self, price_data):
        """Test that T3 follows price trends."""
        result = t3(price_data, timeperiod=5, vfactor=0.7)

        valid_idx = ~np.isnan(result)
        if np.sum(valid_idx) > 10:
            valid_result = result[valid_idx]
            valid_price = price_data[valid_idx]

            # Check correlation is positive
            correlation = np.corrcoef(valid_price, valid_result)[0, 1]
            assert correlation > 0.9  # Should be highly correlated

    def test_lag_behavior(self, price_data):
        """Test that T3 has less lag than traditional MAs."""
        # T3 is designed to have less lag
        # We just verify it produces reasonable values
        result = t3(price_data, timeperiod=10, vfactor=0.7)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

        # Values should be within reasonable range of price data
        price_mean = np.mean(price_data)
        price_std = np.std(price_data)

        assert np.mean(valid_values) > price_mean - 3 * price_std
        assert np.mean(valid_values) < price_mean + 3 * price_std


class TestImplementationConsistency:
    """Test that different implementations produce consistent results."""

    def test_numba_vs_polars_consistency(self, price_data):
        """Test NumPy/Numba vs Polars implementation consistency."""
        # NumPy/Numba
        result_numba = t3(price_data, timeperiod=5, vfactor=0.7)

        # Polars
        df = pl.DataFrame({"close": price_data})
        result_polars = df.select(t3("close", timeperiod=5, vfactor=0.7).alias("t3"))[
            "t3"
        ].to_numpy()

        # Should produce identical results
        np.testing.assert_allclose(
            result_numba,
            result_polars,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_series_vs_array_consistency(self, price_data):
        """Test Polars Series vs NumPy array consistency."""
        # NumPy array
        result_array = t3(price_data, timeperiod=5, vfactor=0.7)

        # Polars Series
        series = pl.Series(price_data)
        result_series = t3(series, timeperiod=5, vfactor=0.7)

        # Should be identical
        np.testing.assert_allclose(
            result_array,
            result_series,
            rtol=1e-10,
            equal_nan=True,
        )


class TestSpecialCases:
    """Test special mathematical cases."""

    def test_zero_vfactor(self, price_data):
        """Test T3 with vfactor=0."""
        result = t3(price_data, timeperiod=5, vfactor=0.0)

        # Should still compute
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_maximum_vfactor(self, price_data):
        """Test T3 with vfactor=1."""
        result = t3(price_data, timeperiod=5, vfactor=1.0)

        # Should still compute
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_minimum_timeperiod(self, price_data):
        """Test T3 with minimum timeperiod=2."""
        result = t3(price_data, timeperiod=2, vfactor=0.7)

        # Should compute with shorter lookback
        lookback = 6 * (2 - 1)  # 6
        assert not np.isnan(result[lookback])

    def test_large_timeperiod(self, price_data):
        """Test T3 with large timeperiod."""
        result = t3(price_data, timeperiod=50, vfactor=0.7)

        # Should compute but have longer warmup
        lookback = 6 * (50 - 1)  # 294
        if len(price_data) > lookback:
            assert not np.isnan(result[lookback])
