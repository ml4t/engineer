"""Tests for STDDEV (Standard Deviation) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.statistics.stddev import stddev


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
    """Test basic STDDEV functionality."""

    def test_computes_successfully_numba(self, price_data):
        """Test STDDEV computes without errors using NumPy array."""
        result = stddev(price_data, period=5)
        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_series(self, price_data):
        """Test STDDEV computes with Polars Series."""
        series = pl.Series(price_data)
        result = stddev(series, period=5)
        assert result is not None
        assert len(result) == len(price_data)

    def test_computes_successfully_polars_expr(self, ohlcv_df):
        """Test STDDEV computes with Polars expression."""
        result = ohlcv_df.select(stddev("close", period=5).alias("stddev"))
        assert result is not None
        assert len(result) == len(ohlcv_df)
        assert "stddev" in result.columns

    def test_default_parameters(self, price_data):
        """Test STDDEV with default parameters."""
        result = stddev(price_data)
        assert result is not None
        # Default period=5, nbdev=1.0, ddof=0
        assert len(result) == len(price_data)

    def test_different_periods(self, price_data):
        """Test STDDEV produces different results with different periods."""
        r1 = stddev(price_data, period=5)
        r2 = stddev(price_data, period=20)

        # Results should be different
        valid_r1 = r1[~np.isnan(r1)]
        valid_r2 = r2[~np.isnan(r2)]
        assert len(valid_r1) > 0
        assert len(valid_r2) > 0
        assert not np.allclose(valid_r1[:50], valid_r2[:50], equal_nan=True)

    def test_different_nbdev(self, price_data):
        """Test STDDEV with different nbdev scaling factors."""
        r1 = stddev(price_data, period=5, nbdev=1.0)
        r2 = stddev(price_data, period=5, nbdev=2.0)

        # r2 should be exactly 2x r1
        np.testing.assert_allclose(r2, r1 * 2.0, rtol=1e-10, equal_nan=True)


class TestLookbackPeriod:
    """Test lookback period behavior."""

    def test_lookback_period(self, price_data):
        """Test that first (period-1) values are NaN."""
        period = 10
        result = stddev(price_data, period=period)

        # First period-1 values should be NaN
        assert all(np.isnan(result[: period - 1]))
        # Value at period-1 should be valid
        assert not np.isnan(result[period - 1])

    def test_output_starts_at_period_minus_one(self, price_data):
        """Test that output starts at period-1."""
        period = 5
        result = stddev(price_data, period=period)

        # First 4 values should be NaN
        assert np.sum(np.isnan(result[: period - 1])) == period - 1
        # Value at index 4 should be valid
        assert not np.isnan(result[period - 1])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test with empty array."""
        empty = np.array([])
        result = stddev(empty, period=5)
        assert len(result) == 0

    def test_single_value(self):
        """Test with single value."""
        single = np.array([100.0])
        result = stddev(single, period=5)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_insufficient_data(self):
        """Test with insufficient data for period."""
        short_data = np.array([100.0, 101.0, 102.0])
        result = stddev(short_data, period=10)

        # All should be NaN
        assert all(np.isnan(result))

    def test_exactly_period_values(self):
        """Test with exactly period values."""
        period = 5
        data = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = stddev(data, period=period)

        # First period-1 should be NaN, last should be valid
        assert all(np.isnan(result[: period - 1]))
        assert not np.isnan(result[-1])

    def test_constant_values(self):
        """Test with constant values."""
        const = np.ones(100) * 100.0
        result = stddev(const, period=10)

        # Stddev of constant should be zero
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.allclose(valid_values, 0.0, atol=1e-10)

    def test_with_nans_in_input(self):
        """Test handling of NaN values in input."""
        data = np.array([100.0, 101.0, np.nan, 102.0, 103.0] * 20)
        result = stddev(data, period=5)

        # Result should exist
        assert result is not None
        assert len(result) == len(data)

    def test_polars_empty_dataframe(self):
        """Test with empty Polars DataFrame."""
        empty_df = pl.DataFrame({"close": pl.Series([], dtype=pl.Float64)})
        result = empty_df.select(stddev("close", period=5).alias("stddev"))
        assert len(result) == 0

    def test_polars_single_row(self):
        """Test with single row DataFrame."""
        single_df = pl.DataFrame({"close": [100.0]})
        result = single_df.select(stddev("close", period=5).alias("stddev"))
        assert len(result) == 1


class TestDDOF:
    """Test degrees of freedom parameter."""

    def test_population_stddev(self, price_data):
        """Test population standard deviation (ddof=0)."""
        result = stddev(price_data, period=10, ddof=0)

        # Should compute successfully
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # All values should be non-negative
        assert all(valid_values >= 0)

    def test_sample_stddev(self, price_data):
        """Test sample standard deviation (ddof=1)."""
        result = stddev(price_data, period=10, ddof=1)

        # Should compute successfully
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # All values should be non-negative
        assert all(valid_values >= 0)

    def test_ddof_difference(self, price_data):
        """Test that ddof=1 gives slightly larger values than ddof=0."""
        r_pop = stddev(price_data, period=10, ddof=0)
        r_sample = stddev(price_data, period=10, ddof=1)

        # Remove NaN
        valid_idx = ~np.isnan(r_pop) & ~np.isnan(r_sample)
        valid_pop = r_pop[valid_idx]
        valid_sample = r_sample[valid_idx]

        if len(valid_pop) > 0:
            # Sample stddev should be >= population stddev
            assert all(valid_sample >= valid_pop - 1e-10)


class TestNumericalProperties:
    """Test numerical properties of STDDEV."""

    def test_always_non_negative(self, price_data):
        """Test that standard deviation is always non-negative."""
        result = stddev(price_data, period=10)

        valid_values = result[~np.isnan(result)]
        assert all(valid_values >= 0)

    def test_zero_for_constant(self):
        """Test that stddev is zero for constant values."""
        const = np.ones(50) * 42.0
        result = stddev(const, period=5)

        valid_values = result[~np.isnan(result)]
        assert all(np.abs(valid_values) < 1e-10)

    def test_increases_with_volatility(self):
        """Test that stddev increases with price volatility."""
        # Low volatility data
        np.random.seed(42)
        low_vol = 100 + np.cumsum(np.random.randn(100) * 0.1)
        r_low = stddev(low_vol, period=10)

        # High volatility data
        np.random.seed(42)
        high_vol = 100 + np.cumsum(np.random.randn(100) * 2.0)
        r_high = stddev(high_vol, period=10)

        # Average stddev should be higher for high volatility
        avg_low = np.nanmean(r_low)
        avg_high = np.nanmean(r_high)

        assert avg_high > avg_low

    def test_nbdev_scaling(self, price_data):
        """Test that nbdev parameter scales results correctly."""
        r1 = stddev(price_data, period=10, nbdev=1.0)
        r3 = stddev(price_data, period=10, nbdev=3.0)

        # r3 should be exactly 3x r1
        np.testing.assert_allclose(r3, r1 * 3.0, rtol=1e-10, equal_nan=True)


class TestImplementationConsistency:
    """Test that different implementations produce consistent results."""

    def test_numba_vs_polars_consistency(self, price_data):
        """Test NumPy/Numba vs Polars implementation consistency."""
        # NumPy/Numba
        result_numba = stddev(price_data, period=10, nbdev=1.0, ddof=0)

        # Polars
        df = pl.DataFrame({"close": price_data})
        result_polars = df.select(stddev("close", period=10, nbdev=1.0, ddof=0).alias("stddev"))[
            "stddev"
        ].to_numpy()

        # Should produce very close results
        # (may differ slightly due to Polars native implementation)
        np.testing.assert_allclose(
            result_numba,
            result_polars,
            rtol=1e-8,
            equal_nan=True,
        )

    def test_series_vs_array_consistency(self, price_data):
        """Test Polars Series vs NumPy array consistency."""
        # NumPy array
        result_array = stddev(price_data, period=10)

        # Polars Series
        series = pl.Series(price_data)
        result_series = stddev(series, period=10)

        # Should be identical
        np.testing.assert_allclose(
            result_array,
            result_series,
            rtol=1e-10,
            equal_nan=True,
        )


class TestSpecialCases:
    """Test special mathematical cases."""

    def test_period_two(self, price_data):
        """Test STDDEV with minimum period=2."""
        result = stddev(price_data, period=2)

        # Should compute
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

        # First value should be NaN, second should be valid
        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_large_period(self, price_data):
        """Test STDDEV with large period."""
        result = stddev(price_data, period=50)

        # Should compute
        if len(price_data) >= 50:
            assert not np.isnan(result[49])

    def test_alternating_values(self):
        """Test STDDEV with alternating values."""
        # Alternating between 100 and 110
        data = np.array([100.0, 110.0] * 50)
        result = stddev(data, period=10)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            # Should have consistent non-zero stddev
            assert all(valid_values > 0)
            # Should be relatively constant since pattern is regular
            assert np.std(valid_values) < np.mean(valid_values) * 0.1

    def test_increasing_sequence(self):
        """Test STDDEV with increasing sequence."""
        data = np.arange(100, dtype=float)
        result = stddev(data, period=10)

        # Should have relatively constant stddev
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # For linearly increasing data with constant step,
            # stddev should be relatively stable
            assert np.std(valid_values) < np.mean(valid_values) * 0.5


class TestComparisonWithNumPy:
    """Test consistency with NumPy std calculation."""

    def test_matches_numpy_rolling(self):
        """Test that STDDEV matches NumPy rolling calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 5
        result = stddev(data, period=period, ddof=0)

        # Manually calculate for comparison
        for i in range(period - 1, len(data)):
            window = data[i - period + 1 : i + 1]
            expected = np.std(window, ddof=0)
            assert np.abs(result[i] - expected) < 1e-10
