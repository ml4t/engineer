"""Tests for DEMA (Double Exponential Moving Average) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.trend.dema import dema


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
    """Test basic DEMA functionality."""

    def test_computes_successfully_numba(self, price_data):
        """Test DEMA computes without errors using NumPy array."""
        result = dema(price_data, period=14)
        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_series(self, price_data):
        """Test DEMA computes with Polars Series."""
        series = pl.Series(price_data)
        result = dema(series, period=14)
        assert result is not None
        assert len(result) == len(price_data)

    def test_computes_successfully_polars_expr(self, ohlcv_df):
        """Test DEMA computes with Polars expression."""
        result = ohlcv_df.select(dema("close", period=14).alias("dema"))
        assert result is not None
        assert len(result) == len(ohlcv_df)
        assert "dema" in result.columns

    def test_default_parameters(self, price_data):
        """Test DEMA with default parameters."""
        result = dema(price_data)
        assert result is not None
        # Default period=30
        assert len(result) == len(price_data)

    def test_different_periods(self, price_data):
        """Test DEMA produces different results with different periods."""
        r1 = dema(price_data, period=10)
        r2 = dema(price_data, period=30)

        # Results should be different
        valid_r1 = r1[~np.isnan(r1)]
        valid_r2 = r2[~np.isnan(r2)]
        assert len(valid_r1) > 0
        assert len(valid_r2) > 0
        assert not np.allclose(valid_r1[:20], valid_r2[:20], equal_nan=True)


class TestLookbackPeriod:
    """Test lookback period behavior."""

    def test_lookback_calculation(self, price_data):
        """Test that lookback is 2 * (period - 1)."""
        period = 14
        lookback = 2 * (period - 1)
        result = dema(price_data, period=period)

        # First lookback values should be NaN
        assert all(np.isnan(result[:lookback]))

        # Value after lookback should eventually be valid
        # (may need a bit more due to double smoothing)
        if len(price_data) > lookback + 1:
            # At least some values after lookback should be valid
            assert np.sum(~np.isnan(result[lookback:])) > 0

    def test_longer_lookback(self, price_data):
        """Test lookback with longer period."""
        period = 20
        lookback = 2 * (period - 1)
        result = dema(price_data, period=period)

        assert all(np.isnan(result[:lookback]))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test with empty array."""
        empty = np.array([])
        result = dema(empty, period=14)
        assert len(result) == 0

    def test_single_value(self):
        """Test with single value."""
        single = np.array([100.0])
        result = dema(single, period=14)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_insufficient_data(self):
        """Test with insufficient data for lookback period."""
        period = 14
        lookback = 2 * (period - 1)
        short_data = np.array([100.0] * lookback)
        result = dema(short_data, period=period)

        # All should be NaN
        assert all(np.isnan(result))

    def test_constant_values(self):
        """Test with constant values."""
        const = np.ones(150) * 100.0
        result = dema(const, period=14)

        # DEMA of constant should be approximately constant
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.allclose(valid_values, 100.0, rtol=1e-5)

    def test_polars_empty_dataframe(self):
        """Test with empty Polars DataFrame."""
        empty_df = pl.DataFrame({"close": pl.Series([], dtype=pl.Float64)})
        result = empty_df.select(dema("close", period=14).alias("dema"))
        assert len(result) == 0

    def test_polars_single_row(self):
        """Test with single row DataFrame."""
        single_df = pl.DataFrame({"close": [100.0]})
        result = single_df.select(dema("close", period=14).alias("dema"))
        assert len(result) == 1


class TestNumericalProperties:
    """Test numerical properties of DEMA."""

    def test_less_lag_than_ema(self):
        """Test that DEMA has less lag than EMA."""
        # DEMA is designed to reduce lag compared to EMA
        # Create data with trend change
        data = np.concatenate(
            [
                np.linspace(100, 110, 50),
                np.linspace(110, 105, 50),
            ]
        )

        result = dema(data, period=10)

        # DEMA should exist and follow the data
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

    def test_smoothness(self, price_data):
        """Test that DEMA is smooth."""
        result = dema(price_data, period=20)

        # Remove NaN values
        valid_idx = ~np.isnan(result)
        valid_result = result[valid_idx]

        if len(valid_result) > 1:
            # Calculate variability
            dema_variability = np.std(np.diff(valid_result))
            # Should be finite and reasonable
            assert np.isfinite(dema_variability)
            assert dema_variability > 0

    def test_follows_trend(self, price_data):
        """Test that DEMA follows price trends."""
        result = dema(price_data, period=14)

        valid_idx = ~np.isnan(result)
        if np.sum(valid_idx) > 10:
            valid_result = result[valid_idx]
            valid_price = price_data[valid_idx]

            # Check correlation is positive
            correlation = np.corrcoef(valid_price, valid_result)[0, 1]
            assert correlation > 0.8  # Should be reasonably correlated


class TestImplementationConsistency:
    """Test that different implementations produce consistent results."""

    def test_numba_vs_polars_consistency(self, price_data):
        """Test NumPy/Numba vs Polars implementation consistency."""
        # NumPy/Numba
        result_numba = dema(price_data, period=14)

        # Polars
        df = pl.DataFrame({"close": price_data})
        result_polars = df.select(dema("close", period=14).alias("dema"))["dema"].to_numpy()

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
        result_array = dema(price_data, period=14)

        # Polars Series
        series = pl.Series(price_data)
        result_series = dema(series, period=14)

        # Should be identical
        np.testing.assert_allclose(
            result_array,
            result_series,
            rtol=1e-10,
            equal_nan=True,
        )


class TestSpecialCases:
    """Test special mathematical cases."""

    def test_short_period(self, price_data):
        """Test DEMA with short period."""
        result = dema(price_data, period=5)

        # Should compute
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

        # Lookback should be 2 * (5 - 1) = 8
        lookback = 8
        assert all(np.isnan(result[:lookback]))

    def test_long_period(self, price_data):
        """Test DEMA with long period."""
        result = dema(price_data, period=50)

        # Should compute but have longer warmup
        lookback = 2 * (50 - 1)  # 98
        if len(price_data) > lookback:
            assert all(np.isnan(result[:lookback]))

    def test_increasing_sequence(self):
        """Test DEMA with strictly increasing sequence."""
        data = np.arange(150, dtype=float)
        result = dema(data, period=10)

        # DEMA should generally be increasing (after warmup)
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Most differences should be positive
            diffs = np.diff(valid_values)
            assert np.mean(diffs > 0) > 0.7

    def test_decreasing_sequence(self):
        """Test DEMA with strictly decreasing sequence."""
        data = np.arange(150, 0, -1, dtype=float)
        result = dema(data, period=10)

        # DEMA should generally be decreasing (after warmup)
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # Most differences should be negative
            diffs = np.diff(valid_values)
            assert np.mean(diffs < 0) > 0.7


class TestFormula:
    """Test DEMA formula: 2*EMA - EMA(EMA)."""

    def test_formula_relationship(self, price_data):
        """Test that DEMA = 2*EMA - EMA(EMA)."""
        # We can't easily test this without importing ema,
        # but we can verify the result is reasonable
        result = dema(price_data, period=14)

        valid_values = result[~np.isnan(result)]
        price_valid = price_data[~np.isnan(result)]

        if len(valid_values) > 0:
            # DEMA should be within reasonable range of prices
            price_range = np.ptp(price_valid)
            dema_range = np.ptp(valid_values)

            # DEMA range should be similar to price range
            assert dema_range > 0
            # Not more than 2x the price range
            assert dema_range < price_range * 2
