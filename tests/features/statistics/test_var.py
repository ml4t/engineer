"""Tests for VAR (Variance) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.statistics.var import var


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
    """Test basic VAR functionality."""

    def test_computes_successfully_numba(self, price_data):
        """Test VAR computes without errors using NumPy array."""
        result = var(price_data, timeperiod=5)
        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_series(self, price_data):
        """Test VAR computes with Polars Series."""
        series = pl.Series(price_data)
        result = var(series, timeperiod=5)
        assert result is not None
        assert len(result) == len(price_data)

    def test_computes_successfully_polars_expr(self, ohlcv_df):
        """Test VAR computes with Polars expression."""
        result = ohlcv_df.select(var("close", timeperiod=5).alias("var"))
        assert result is not None
        assert len(result) == len(ohlcv_df)
        assert "var" in result.columns

    def test_default_parameters(self, price_data):
        """Test VAR with default parameters."""
        result = var(price_data)
        assert result is not None
        # Default timeperiod=5
        assert len(result) == len(price_data)

    def test_different_periods(self, price_data):
        """Test VAR produces different results with different periods."""
        r1 = var(price_data, timeperiod=5)
        r2 = var(price_data, timeperiod=20)

        # Results should be different
        valid_r1 = r1[~np.isnan(r1)]
        valid_r2 = r2[~np.isnan(r2)]
        assert len(valid_r1) > 0
        assert len(valid_r2) > 0
        assert not np.allclose(valid_r1[:50], valid_r2[:50], equal_nan=True)


class TestLookbackPeriod:
    """Test lookback period behavior."""

    def test_lookback_period(self, price_data):
        """Test that first (timeperiod-1) values are NaN."""
        timeperiod = 10
        result = var(price_data, timeperiod=timeperiod)

        # First timeperiod-1 values should be NaN
        assert all(np.isnan(result[: timeperiod - 1]))
        # Value at timeperiod-1 should be valid
        assert not np.isnan(result[timeperiod - 1])

    def test_output_starts_at_period_minus_one(self, price_data):
        """Test that output starts at timeperiod-1."""
        timeperiod = 5
        result = var(price_data, timeperiod=timeperiod)

        # First 4 values should be NaN
        assert np.sum(np.isnan(result[: timeperiod - 1])) == timeperiod - 1
        # Value at index 4 should be valid
        assert not np.isnan(result[timeperiod - 1])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test with empty array."""
        empty = np.array([])
        result = var(empty, timeperiod=5)
        assert len(result) == 0

    def test_single_value(self):
        """Test with single value."""
        single = np.array([100.0])
        result = var(single, timeperiod=5)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_insufficient_data(self):
        """Test with insufficient data for timeperiod."""
        short_data = np.array([100.0, 101.0, 102.0])
        result = var(short_data, timeperiod=10)

        # All should be NaN
        assert all(np.isnan(result))

    def test_exactly_period_values(self):
        """Test with exactly timeperiod values."""
        timeperiod = 5
        data = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = var(data, timeperiod=timeperiod)

        # First timeperiod-1 should be NaN, last should be valid
        assert all(np.isnan(result[: timeperiod - 1]))
        assert not np.isnan(result[-1])

    def test_constant_values(self):
        """Test with constant values."""
        const = np.ones(100) * 100.0
        result = var(const, timeperiod=10)

        # Variance of constant should be zero
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.allclose(valid_values, 0.0, atol=1e-10)

    def test_polars_empty_dataframe(self):
        """Test with empty Polars DataFrame."""
        empty_df = pl.DataFrame({"close": pl.Series([], dtype=pl.Float64)})
        result = empty_df.select(var("close", timeperiod=5).alias("var"))
        assert len(result) == 0

    def test_polars_single_row(self):
        """Test with single row DataFrame."""
        single_df = pl.DataFrame({"close": [100.0]})
        result = single_df.select(var("close", timeperiod=5).alias("var"))
        assert len(result) == 1


class TestValidation:
    """Test parameter validation."""

    def test_invalid_timeperiod_zero(self, price_data):
        """Test that timeperiod = 0 raises error."""
        with pytest.raises(ValueError, match="timeperiod must be >= 1"):
            var(price_data, timeperiod=0)

    def test_invalid_timeperiod_negative(self, price_data):
        """Test that negative timeperiod raises error."""
        with pytest.raises(ValueError, match="timeperiod must be >= 1"):
            var(price_data, timeperiod=-5)

    def test_valid_timeperiod_one(self, price_data):
        """Test that timeperiod = 1 is valid."""
        result = var(price_data, timeperiod=1)
        assert result is not None


class TestNumericalProperties:
    """Test numerical properties of VAR."""

    def test_always_non_negative(self, price_data):
        """Test that variance is always non-negative."""
        result = var(price_data, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        assert all(valid_values >= -1e-10)  # Allow for tiny numerical errors

    def test_zero_for_constant(self):
        """Test that variance is zero for constant values."""
        const = np.ones(50) * 42.0
        result = var(const, timeperiod=5)

        valid_values = result[~np.isnan(result)]
        assert all(np.abs(valid_values) < 1e-10)

    def test_increases_with_volatility(self):
        """Test that variance increases with price volatility."""
        # Low volatility data
        np.random.seed(42)
        low_vol = 100 + np.cumsum(np.random.randn(100) * 0.1)
        r_low = var(low_vol, timeperiod=10)

        # High volatility data
        np.random.seed(42)
        high_vol = 100 + np.cumsum(np.random.randn(100) * 2.0)
        r_high = var(high_vol, timeperiod=10)

        # Average variance should be higher for high volatility
        avg_low = np.nanmean(r_low)
        avg_high = np.nanmean(r_high)

        assert avg_high > avg_low

    def test_relationship_to_stddev(self, price_data):
        """Test that variance is the square of standard deviation."""
        from ml4t.engineer.features.statistics.stddev import stddev

        result_var = var(price_data, timeperiod=10)
        result_std = stddev(price_data, period=10, ddof=0)

        # VAR should equal STDDEV^2
        np.testing.assert_allclose(
            result_var,
            result_std**2,
            rtol=1e-8,
            equal_nan=True,
        )


class TestImplementationConsistency:
    """Test that different implementations produce consistent results."""

    def test_numba_vs_polars_consistency(self, price_data):
        """Test NumPy/Numba vs Polars implementation consistency."""
        # NumPy/Numba
        result_numba = var(price_data, timeperiod=10)

        # Polars
        df = pl.DataFrame({"close": price_data})
        result_polars = df.select(var("close", timeperiod=10).alias("var"))["var"].to_numpy()

        # Should produce very close results
        np.testing.assert_allclose(
            result_numba,
            result_polars,
            rtol=1e-8,
            equal_nan=True,
        )

    def test_series_vs_array_consistency(self, price_data):
        """Test Polars Series vs NumPy array consistency."""
        # NumPy array
        result_array = var(price_data, timeperiod=10)

        # Polars Series
        series = pl.Series(price_data)
        result_series = var(series, timeperiod=10)

        # Should be identical
        np.testing.assert_allclose(
            result_array,
            result_series,
            rtol=1e-10,
            equal_nan=True,
        )


class TestSpecialCases:
    """Test special mathematical cases."""

    def test_period_one(self, price_data):
        """Test VAR with period=1."""
        result = var(price_data, timeperiod=1)

        # Should compute - variance of single value is 0
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # All should be zero (single value has no variance)
        assert all(np.abs(valid_values) < 1e-10)

    def test_period_two(self, price_data):
        """Test VAR with period=2."""
        result = var(price_data, timeperiod=2)

        # Should compute
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

        # First value should be NaN, second should be valid
        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_large_period(self, price_data):
        """Test VAR with large period."""
        result = var(price_data, timeperiod=50)

        # Should compute
        if len(price_data) >= 50:
            assert not np.isnan(result[49])

    def test_alternating_values(self):
        """Test VAR with alternating values."""
        # Alternating between 100 and 110
        data = np.array([100.0, 110.0] * 50)
        result = var(data, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            # Should have consistent non-zero variance
            assert all(valid_values > 0)
            # Should be relatively constant since pattern is regular
            assert np.std(valid_values) < np.mean(valid_values) * 0.1

    def test_increasing_sequence(self):
        """Test VAR with increasing sequence."""
        data = np.arange(100, dtype=float)
        result = var(data, timeperiod=10)

        # Should have relatively constant variance
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # For linearly increasing data, variance should be relatively stable
            assert np.std(valid_values) < np.mean(valid_values) * 0.5


class TestComparisonWithNumPy:
    """Test consistency with NumPy var calculation."""

    def test_matches_numpy_rolling(self):
        """Test that VAR matches NumPy rolling calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        timeperiod = 5
        result = var(data, timeperiod=timeperiod)

        # Manually calculate for comparison
        for i in range(timeperiod - 1, len(data)):
            window = data[i - timeperiod + 1 : i + 1]
            expected = np.var(window, ddof=0)
            assert np.abs(result[i] - expected) < 1e-10

    def test_formula_e_x_squared_minus_e_x_squared(self):
        """Test VAR = E[X²] - (E[X])²."""
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        timeperiod = 5
        result = var(data, timeperiod=timeperiod)

        # Manually calculate using formula for last window
        window = data[-5:]
        e_x = np.mean(window)
        e_x_squared = np.mean(window**2)
        expected_var = e_x_squared - e_x**2

        assert np.abs(result[-1] - expected_var) < 1e-10
