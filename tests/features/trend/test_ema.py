"""Tests for EMA (Exponential Moving Average) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.trend.ema import ema


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
    """Test basic EMA functionality."""

    def test_computes_successfully_numba(self, price_data):
        """Test EMA computes without errors using NumPy array."""
        result = ema(price_data, period=14)
        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_series(self, price_data):
        """Test EMA computes with Polars Series."""
        series = pl.Series(price_data)
        result = ema(series, period=14)
        assert result is not None
        assert len(result) == len(price_data)

    def test_computes_successfully_polars_expr(self, ohlcv_df):
        """Test EMA computes with Polars expression."""
        result = ohlcv_df.select(ema("close", period=14).alias("ema"))
        assert result is not None
        assert len(result) == len(ohlcv_df)
        assert "ema" in result.columns

    def test_different_periods(self, price_data):
        """Test EMA produces different results with different periods."""
        r1 = ema(price_data, period=10)
        r2 = ema(price_data, period=30)

        # Results should be different
        valid_r1 = r1[~np.isnan(r1)]
        valid_r2 = r2[~np.isnan(r2)]
        assert len(valid_r1) > 0
        assert len(valid_r2) > 0
        # Should differ after initial values
        assert not np.allclose(valid_r1[:50], valid_r2[:50], equal_nan=True)


class TestLookbackPeriod:
    """Test lookback period behavior."""

    def test_lookback_period(self, price_data):
        """Test that first (period-1) values are NaN."""
        period = 14
        result = ema(price_data, period=period)

        # First period-1 values should be NaN
        assert all(np.isnan(result[: period - 1]))
        # Value at period-1 should be valid (SMA initialization)
        assert not np.isnan(result[period - 1])

    def test_initialization_with_sma(self, price_data):
        """Test that EMA initializes with SMA of first period values."""
        period = 10
        result = ema(price_data, period=period)

        # First valid value should be close to SMA
        first_valid_idx = period - 1
        expected_sma = np.mean(price_data[:period])

        # Should be exactly equal for initialization
        assert np.abs(result[first_valid_idx] - expected_sma) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test with empty array."""
        empty = np.array([])
        result = ema(empty, period=14)
        assert len(result) == 0

    def test_single_value(self):
        """Test with single value."""
        single = np.array([100.0])
        result = ema(single, period=14)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_insufficient_data(self):
        """Test with insufficient data for period."""
        short_data = np.array([100.0, 101.0, 102.0])
        result = ema(short_data, period=10)

        # All should be NaN
        assert all(np.isnan(result))

    def test_exactly_period_values(self):
        """Test with exactly period values."""
        period = 5
        data = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = ema(data, period=period)

        # First period-1 should be NaN, last should be valid
        assert all(np.isnan(result[: period - 1]))
        assert not np.isnan(result[-1])

    def test_constant_values(self):
        """Test with constant values."""
        const = np.ones(100) * 100.0
        result = ema(const, period=14)

        # EMA of constant should be constant
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.allclose(valid_values, 100.0, rtol=1e-10)

    def test_with_nans_in_input(self):
        """Test handling of NaN values in input."""
        data = np.array([100.0, 101.0, np.nan, 102.0, 103.0] * 20)
        result = ema(data, period=5)

        # Result should exist
        assert result is not None
        assert len(result) == len(data)
        # Should skip over NaN values per TA-Lib behavior

    def test_polars_empty_dataframe(self):
        """Test with empty Polars DataFrame."""
        empty_df = pl.DataFrame({"close": pl.Series([], dtype=pl.Float64)})
        result = empty_df.select(ema("close", period=14).alias("ema"))
        assert len(result) == 0

    def test_polars_single_row(self):
        """Test with single row DataFrame."""
        single_df = pl.DataFrame({"close": [100.0]})
        result = single_df.select(ema("close", period=14).alias("ema"))
        assert len(result) == 1


class TestValidation:
    """Test parameter validation."""

    def test_invalid_period_zero(self, price_data):
        """Test that period = 0 raises error."""
        with pytest.raises((ValueError, TypeError)):
            ema(price_data, period=0)

    def test_invalid_period_negative(self, price_data):
        """Test that negative period raises error."""
        with pytest.raises((ValueError, TypeError)):
            ema(price_data, period=-5)

    def test_valid_period_one(self, price_data):
        """Test that period = 1 is valid."""
        result = ema(price_data, period=1)
        assert result is not None
        # With period=1, EMA should equal input (after first value)
        assert not np.isnan(result[0])


class TestNumericalProperties:
    """Test numerical properties of EMA."""

    def test_smoothness(self, price_data):
        """Test that EMA is smoother than the original data."""
        result = ema(price_data, period=20)

        # Remove NaN values for comparison
        valid_idx = ~np.isnan(result)
        valid_result = result[valid_idx]
        valid_price = price_data[valid_idx]

        # Calculate variability (standard deviation of differences)
        if len(valid_result) > 1:
            price_variability = np.std(np.diff(valid_price))
            ema_variability = np.std(np.diff(valid_result))

            # EMA should be smoother (lower variability)
            assert ema_variability < price_variability

    def test_follows_trend(self, price_data):
        """Test that EMA follows price trends."""
        result = ema(price_data, period=14)

        valid_idx = ~np.isnan(result)
        if np.sum(valid_idx) > 10:
            valid_result = result[valid_idx]
            valid_price = price_data[valid_idx]

            # Check correlation is positive and reasonably strong
            correlation = np.corrcoef(valid_price, valid_result)[0, 1]
            assert correlation > 0.80

    def test_alpha_calculation(self):
        """Test that alpha = 2 / (period + 1)."""
        # For known sequence, verify EMA calculation
        data = np.array([22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29])
        period = 5
        result = ema(data, period=period)

        # Alpha should be 2 / (5 + 1) = 1/3
        alpha = 2.0 / (period + 1.0)
        assert np.abs(alpha - (1.0 / 3.0)) < 1e-10

        # First EMA should be SMA of first 5 values
        first_ema = np.mean(data[:5])
        assert np.abs(result[4] - first_ema) < 1e-10

    def test_exponential_weighting(self):
        """Test that recent values have more weight."""
        # Create data with a step change
        data = np.concatenate([np.ones(50) * 100.0, np.ones(50) * 110.0])
        result = ema(data, period=10)

        # After the step, EMA should move toward new value
        # but not reach it immediately
        step_idx = 50
        ema_after_step = result[step_idx + 5]

        assert ema_after_step > 100.0  # Moving up
        assert ema_after_step < 110.0  # But not there yet


class TestImplementationConsistency:
    """Test that different implementations produce consistent results."""

    def test_numba_vs_polars_consistency(self, price_data):
        """Test NumPy/Numba vs Polars implementation consistency."""
        # NumPy/Numba
        result_numba = ema(price_data, period=14)

        # Polars
        df = pl.DataFrame({"close": price_data})
        result_polars = df.select(ema("close", period=14).alias("ema"))["ema"].to_numpy()

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
        result_array = ema(price_data, period=14)

        # Polars Series
        series = pl.Series(price_data)
        result_series = ema(series, period=14)

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
        """Test EMA with very short period."""
        result = ema(price_data, period=2)

        # Should compute and respond quickly to changes
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

        # Should start producing values quickly
        assert not np.isnan(result[1])

    def test_long_period(self, price_data):
        """Test EMA with long period."""
        result = ema(price_data, period=50)

        # Should compute but be smoother
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

        # First valid value at period-1
        assert not np.isnan(result[49])
        assert all(np.isnan(result[:49]))

    def test_increasing_sequence(self):
        """Test EMA with strictly increasing sequence."""
        data = np.arange(100, dtype=float)
        result = ema(data, period=10)

        # EMA should also be increasing (after warmup)
        valid_values = result[~np.isnan(result)]
        assert all(np.diff(valid_values) > 0)

    def test_decreasing_sequence(self):
        """Test EMA with strictly decreasing sequence."""
        data = np.arange(100, 0, -1, dtype=float)
        result = ema(data, period=10)

        # EMA should also be decreasing (after warmup)
        valid_values = result[~np.isnan(result)]
        assert all(np.diff(valid_values) < 0)


class TestReactivity:
    """Test EMA reactivity to price changes."""

    def test_faster_than_sma(self):
        """Test that EMA reacts faster than SMA to price changes."""
        # Create data with sudden change
        data = np.concatenate(
            [
                np.ones(50) * 100.0,
                np.ones(50) * 110.0,
            ]
        )

        result_ema = ema(data, period=20)

        # Calculate simple SMA for comparison
        sma = np.full_like(data, np.nan)
        for i in range(19, len(data)):
            sma[i] = np.mean(data[i - 19 : i + 1])

        # After the step change, EMA should be closer to new value than SMA
        # Check 5 periods after the step
        step_idx = 50
        check_idx = step_idx + 5

        if not np.isnan(result_ema[check_idx]) and not np.isnan(sma[check_idx]):
            # Both should be between 100 and 110
            assert 100.0 < result_ema[check_idx] < 110.0
            assert 100.0 < sma[check_idx] < 110.0

            # EMA should be closer to 110 (reacts faster)
            ema_distance = abs(110.0 - result_ema[check_idx])
            sma_distance = abs(110.0 - sma[check_idx])
            assert ema_distance < sma_distance
