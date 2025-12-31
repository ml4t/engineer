"""Tests for KAMA (Kaufman's Adaptive Moving Average) indicator."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.trend.kama import kama


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
    """Test basic KAMA functionality."""

    def test_computes_successfully_numba(self, price_data):
        """Test KAMA computes without errors using NumPy array."""
        result = kama(price_data, timeperiod=30)
        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_computes_successfully_polars_series(self, price_data):
        """Test KAMA computes with Polars Series."""
        series = pl.Series(price_data)
        result = kama(series, timeperiod=30)
        assert result is not None
        assert len(result) == len(price_data)

    def test_computes_successfully_polars_expr(self, ohlcv_df):
        """Test KAMA computes with Polars expression."""
        result = ohlcv_df.select(kama("close", timeperiod=30).alias("kama"))
        assert result is not None
        assert len(result) == len(ohlcv_df)
        assert "kama" in result.columns

    def test_default_parameters(self, price_data):
        """Test KAMA with default parameters."""
        result = kama(price_data)
        assert result is not None
        # Default timeperiod=30
        assert len(result) == len(price_data)

    def test_different_periods(self, price_data):
        """Test KAMA produces different results with different periods."""
        r1 = kama(price_data, timeperiod=10)
        r2 = kama(price_data, timeperiod=50)

        # Results should be different
        valid_r1 = r1[~np.isnan(r1)]
        valid_r2 = r2[~np.isnan(r2)]
        assert len(valid_r1) > 0
        assert len(valid_r2) > 0
        assert not np.allclose(valid_r1[:30], valid_r2[:30], equal_nan=True)


class TestLookbackPeriod:
    """Test lookback period behavior."""

    def test_lookback_period(self, price_data):
        """Test that first timeperiod values produce NaN."""
        timeperiod = 30
        result = kama(price_data, timeperiod=timeperiod)

        # First timeperiod values should be NaN
        assert all(np.isnan(result[:timeperiod]))

        # Value at timeperiod should be valid
        if len(price_data) > timeperiod:
            assert not np.isnan(result[timeperiod])

    def test_output_starts_after_period(self, price_data):
        """Test that output starts after the period."""
        timeperiod = 20
        result = kama(price_data, timeperiod=timeperiod)

        # Should have NaN for first timeperiod values
        assert np.sum(np.isnan(result[:timeperiod])) == timeperiod


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test with empty array."""
        empty = np.array([])
        result = kama(empty, timeperiod=30)
        assert len(result) == 0

    def test_single_value(self):
        """Test with single value."""
        single = np.array([100.0])
        result = kama(single, timeperiod=30)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_insufficient_data(self):
        """Test with insufficient data for timeperiod."""
        short_data = np.array([100.0] * 20)
        result = kama(short_data, timeperiod=30)

        # All should be NaN
        assert all(np.isnan(result))

    def test_exactly_period_values(self):
        """Test with exactly timeperiod values."""
        timeperiod = 30
        data = np.arange(timeperiod, dtype=float) + 100
        result = kama(data, timeperiod=timeperiod)

        # First timeperiod values should be NaN
        assert all(np.isnan(result[:timeperiod]))

    def test_constant_values(self):
        """Test with constant values."""
        const = np.ones(100) * 100.0
        result = kama(const, timeperiod=30)

        # KAMA of constant should be constant
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            assert np.allclose(valid_values, 100.0, rtol=1e-10)

    def test_polars_empty_dataframe(self):
        """Test with empty Polars DataFrame."""
        empty_df = pl.DataFrame({"close": pl.Series([], dtype=pl.Float64)})
        result = empty_df.select(kama("close", timeperiod=30).alias("kama"))
        assert len(result) == 0

    def test_polars_single_row(self):
        """Test with single row DataFrame."""
        single_df = pl.DataFrame({"close": [100.0]})
        result = single_df.select(kama("close", timeperiod=30).alias("kama"))
        assert len(result) == 1


class TestValidation:
    """Test parameter validation."""

    def test_invalid_timeperiod_too_small(self, price_data):
        """Test that timeperiod < 2 raises error."""
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            kama(price_data, timeperiod=1)

    def test_invalid_timeperiod_zero(self, price_data):
        """Test that timeperiod = 0 raises error."""
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            kama(price_data, timeperiod=0)

    def test_invalid_timeperiod_negative(self, price_data):
        """Test that negative timeperiod raises error."""
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            kama(price_data, timeperiod=-10)

    def test_valid_minimum_timeperiod(self, price_data):
        """Test that timeperiod = 2 is valid."""
        result = kama(price_data, timeperiod=2)
        assert result is not None


class TestAdaptiveNature:
    """Test the adaptive nature of KAMA."""

    def test_adapts_to_trending_market(self):
        """Test that KAMA adapts faster in trending markets."""
        # Create strongly trending data
        trending_data = np.linspace(100, 150, 100)
        result = kama(trending_data, timeperiod=20)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # In trending market, KAMA should follow trend closely
            # Check that it's increasing
            assert np.mean(np.diff(valid_values) > 0) > 0.7

    def test_slows_in_choppy_market(self):
        """Test that KAMA slows down in choppy markets."""
        # Create choppy (oscillating) data
        choppy_data = 100 + 5 * np.sin(np.arange(100) * 0.5)
        result = kama(choppy_data, timeperiod=20)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 10:
            # KAMA should be smoother than raw data in choppy market
            data_variability = np.std(np.diff(choppy_data))
            kama_variability = np.std(np.diff(valid_values))
            assert kama_variability < data_variability

    def test_efficiency_ratio_concept(self):
        """Test that KAMA behavior is consistent with efficiency ratio."""
        # High efficiency ratio = trending market = faster adaptation
        # Low efficiency ratio = choppy market = slower adaptation

        # Create combined data: trending then choppy
        trending = np.linspace(100, 120, 50)
        choppy = 120 + 2 * np.sin(np.arange(50) * 0.8)
        combined = np.concatenate([trending, choppy])

        result = kama(combined, timeperiod=20)

        # Just verify it computes successfully
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0


class TestNumericalProperties:
    """Test numerical properties of KAMA."""

    def test_follows_trend(self, price_data):
        """Test that KAMA follows price trends."""
        result = kama(price_data, timeperiod=30)

        valid_idx = ~np.isnan(result)
        if np.sum(valid_idx) > 10:
            valid_result = result[valid_idx]
            valid_price = price_data[valid_idx]

            # Check correlation is positive (KAMA adapts to volatility, so correlation may be moderate)
            correlation = np.corrcoef(valid_price, valid_result)[0, 1]
            assert correlation > 0.5

    def test_smoothness(self, price_data):
        """Test that KAMA is smoother than raw data."""
        result = kama(price_data, timeperiod=30)

        # Remove NaN values
        valid_idx = ~np.isnan(result)
        valid_result = result[valid_idx]
        valid_price = price_data[valid_idx]

        if len(valid_result) > 10:
            # Calculate variability
            price_variability = np.std(np.diff(valid_price))
            kama_variability = np.std(np.diff(valid_result))

            # KAMA should be smoother
            assert kama_variability < price_variability

    def test_bounded_by_price_range(self, price_data):
        """Test that KAMA stays within reasonable price range."""
        result = kama(price_data, timeperiod=30)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            price_min = np.min(price_data)
            price_max = np.max(price_data)

            # KAMA should be within or very close to price range
            # Allow for small overshoot due to smoothing
            assert np.min(valid_values) >= price_min - abs(price_min) * 0.1
            assert np.max(valid_values) <= price_max + abs(price_max) * 0.1


class TestImplementationConsistency:
    """Test that different implementations produce consistent results."""

    def test_numba_vs_polars_consistency(self, price_data):
        """Test NumPy/Numba vs Polars implementation consistency."""
        # NumPy/Numba
        result_numba = kama(price_data, timeperiod=30)

        # Polars
        df = pl.DataFrame({"close": price_data})
        result_polars = df.select(kama("close", timeperiod=30).alias("kama"))["kama"].to_numpy()

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
        result_array = kama(price_data, timeperiod=30)

        # Polars Series
        series = pl.Series(price_data)
        result_series = kama(series, timeperiod=30)

        # Should be identical
        np.testing.assert_allclose(
            result_array,
            result_series,
            rtol=1e-10,
            equal_nan=True,
        )


class TestSpecialCases:
    """Test special mathematical cases."""

    def test_short_timeperiod(self, price_data):
        """Test KAMA with short timeperiod."""
        result = kama(price_data, timeperiod=5)

        # Should compute with shorter warmup
        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0

        # First 5 values should be NaN
        assert all(np.isnan(result[:5]))

    def test_long_timeperiod(self, price_data):
        """Test KAMA with long timeperiod."""
        result = kama(price_data, timeperiod=50)

        # Should compute but have longer warmup
        if len(price_data) > 50:
            assert all(np.isnan(result[:50]))
            # Some values after warmup should be valid
            assert np.sum(~np.isnan(result[50:])) > 0

    def test_increasing_sequence(self):
        """Test KAMA with strictly increasing sequence."""
        data = np.arange(150, dtype=float)
        result = kama(data, timeperiod=20)

        # KAMA should also be increasing (after warmup)
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 2:
            # Should be generally increasing
            assert np.mean(np.diff(valid_values) > 0) > 0.8

    def test_decreasing_sequence(self):
        """Test KAMA with strictly decreasing sequence."""
        data = np.arange(150, 0, -1, dtype=float)
        result = kama(data, timeperiod=20)

        # KAMA should also be decreasing (after warmup)
        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 2:
            # Should be generally decreasing
            assert np.mean(np.diff(valid_values) < 0) > 0.8
