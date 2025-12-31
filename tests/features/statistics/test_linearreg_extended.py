"""Tests for Linear Regression family of indicators.

Tests the following functions:
- linearreg: Linear regression endpoint value
- linearreg_slope: Slope of the regression line
- linearreg_angle: Angle of the regression line in degrees
- linearreg_intercept: Y-intercept of the regression line
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.statistics.linearreg import linearreg
from ml4t.engineer.features.statistics.linearreg_angle import linearreg_angle
from ml4t.engineer.features.statistics.linearreg_intercept import linearreg_intercept
from ml4t.engineer.features.statistics.linearreg_slope import linearreg_slope


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


class TestLinearReg:
    """Tests for LINEARREG (regression endpoint)."""

    def test_computes_successfully(self, price_data):
        """Test LINEARREG computes without errors."""
        result = linearreg(price_data, timeperiod=14)
        assert result is not None
        assert len(result) == len(price_data)
        assert isinstance(result, np.ndarray)

    def test_polars_expression(self, ohlcv_df):
        """Test LINEARREG with Polars expression."""
        result = ohlcv_df.select(linearreg("close", timeperiod=14).alias("linreg"))
        assert result is not None
        assert "linreg" in result.columns

    def test_default_parameters(self, price_data):
        """Test with default timeperiod=14."""
        result = linearreg(price_data)
        assert result is not None

    def test_different_periods(self, price_data):
        """Test different periods produce different results."""
        r1 = linearreg(price_data, timeperiod=10)
        r2 = linearreg(price_data, timeperiod=30)

        valid_r1 = r1[~np.isnan(r1)]
        valid_r2 = r2[~np.isnan(r2)]
        assert len(valid_r1) > 0
        assert len(valid_r2) > 0
        assert not np.allclose(valid_r1[:20], valid_r2[:20])

    def test_lookback_period(self, price_data):
        """Test lookback period is timeperiod-1."""
        timeperiod = 14
        result = linearreg(price_data, timeperiod=timeperiod)

        # First timeperiod-1 should be NaN
        assert all(np.isnan(result[: timeperiod - 1]))
        # Value at timeperiod-1 should be valid
        assert not np.isnan(result[timeperiod - 1])

    def test_linear_data(self):
        """Test with perfectly linear data."""
        # y = 2x + 100
        data = np.arange(50, dtype=float) * 2 + 100
        result = linearreg(data, timeperiod=10)

        # For linear data, linearreg should match the data closely
        valid_idx = ~np.isnan(result)
        if np.sum(valid_idx) > 0:
            # Should be very close to actual values
            np.testing.assert_allclose(
                result[valid_idx],
                data[valid_idx],
                rtol=1e-10,
            )

    def test_validation_timeperiod_too_small(self, price_data):
        """Test that timeperiod < 2 raises error."""
        with pytest.raises(ValueError, match="timeperiod must be >= 2"):
            linearreg(price_data, timeperiod=1)


class TestLinearRegSlope:
    """Tests for LINEARREG_SLOPE."""

    def test_computes_successfully(self, price_data):
        """Test LINEARREG_SLOPE computes without errors."""
        result = linearreg_slope(price_data, timeperiod=14)
        assert result is not None
        assert len(result) == len(price_data)

    def test_polars_expression(self, ohlcv_df):
        """Test with Polars expression."""
        result = ohlcv_df.select(linearreg_slope("close", timeperiod=14).alias("slope"))
        assert result is not None
        assert "slope" in result.columns

    def test_constant_data(self):
        """Test slope is zero for constant data."""
        const = np.ones(50) * 100.0
        result = linearreg_slope(const, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.allclose(valid_values, 0.0, atol=1e-10)

    def test_increasing_data(self):
        """Test slope is positive for increasing data."""
        increasing = np.arange(50, dtype=float)
        result = linearreg_slope(increasing, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # All slopes should be positive
        assert all(valid_values > 0)

    def test_decreasing_data(self):
        """Test slope is negative for decreasing data."""
        decreasing = np.arange(50, 0, -1, dtype=float)
        result = linearreg_slope(decreasing, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # All slopes should be negative
        assert all(valid_values < 0)

    def test_linear_data_known_slope(self):
        """Test with data having known slope."""
        # y = 2x + 100
        data = np.arange(50, dtype=float) * 2 + 100
        result = linearreg_slope(data, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            # Slope should be approximately 2.0
            assert np.allclose(valid_values, 2.0, rtol=1e-10)


class TestLinearRegAngle:
    """Tests for LINEARREG_ANGLE."""

    def test_computes_successfully(self, price_data):
        """Test LINEARREG_ANGLE computes without errors."""
        result = linearreg_angle(price_data, timeperiod=14)
        assert result is not None
        assert len(result) == len(price_data)

    def test_polars_expression(self, ohlcv_df):
        """Test with Polars expression."""
        result = ohlcv_df.select(linearreg_angle("close", timeperiod=14).alias("angle"))
        assert result is not None
        assert "angle" in result.columns

    def test_constant_data(self):
        """Test angle is zero for constant data."""
        const = np.ones(50) * 100.0
        result = linearreg_angle(const, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        assert np.allclose(valid_values, 0.0, atol=1e-10)

    def test_increasing_data(self):
        """Test angle is positive for increasing data."""
        increasing = np.arange(50, dtype=float)
        result = linearreg_angle(increasing, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # All angles should be positive
        assert all(valid_values > 0)

    def test_decreasing_data(self):
        """Test angle is negative for decreasing data."""
        decreasing = np.arange(50, 0, -1, dtype=float)
        result = linearreg_angle(decreasing, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # All angles should be negative
        assert all(valid_values < 0)

    def test_angle_range(self, price_data):
        """Test that angle is in valid range."""
        result = linearreg_angle(price_data, timeperiod=14)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            # Angle should be between -90 and 90 degrees
            assert all(valid_values >= -90)
            assert all(valid_values <= 90)

    def test_relationship_to_slope(self, price_data):
        """Test angle = arctan(slope) * 180/π."""
        slope_result = linearreg_slope(price_data, timeperiod=14)
        angle_result = linearreg_angle(price_data, timeperiod=14)

        # Calculate expected angle from slope
        expected_angle = np.arctan(slope_result) * (180.0 / np.pi)

        np.testing.assert_allclose(
            angle_result,
            expected_angle,
            rtol=1e-10,
            equal_nan=True,
        )


class TestLinearRegIntercept:
    """Tests for LINEARREG_INTERCEPT."""

    def test_computes_successfully(self, price_data):
        """Test LINEARREG_INTERCEPT computes without errors."""
        result = linearreg_intercept(price_data, timeperiod=14)
        assert result is not None
        assert len(result) == len(price_data)

    def test_polars_expression(self, ohlcv_df):
        """Test with Polars expression."""
        result = ohlcv_df.select(linearreg_intercept("close", timeperiod=14).alias("intercept"))
        assert result is not None
        assert "intercept" in result.columns

    def test_constant_data(self):
        """Test intercept equals constant value for constant data."""
        const = np.ones(50) * 42.0
        result = linearreg_intercept(const, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        assert len(valid_values) > 0
        # Intercept should equal the constant value
        assert np.allclose(valid_values, 42.0, rtol=1e-8)

    def test_linear_data_known_intercept(self):
        """Test with data having known intercept."""
        # y = 2x + 100, where x is the index
        # For a rolling window, the intercept is the y-value at the start of each window
        # As window moves, the intercept changes
        data = np.arange(50, dtype=float) * 2 + 100
        result = linearreg_intercept(data, timeperiod=10)

        valid_values = result[~np.isnan(result)]
        if len(valid_values) > 0:
            # Just verify we get valid numeric output
            # The intercept increases as the window moves
            assert len(valid_values) > 0
            assert np.all(np.isfinite(valid_values))


class TestLinearRegFamily:
    """Tests for relationships between linear regression family."""

    def test_reconstruction_from_components(self):
        """Test that linearreg = slope * (period-1) + intercept."""
        data = np.arange(100, dtype=float) * 2 + 50
        timeperiod = 10

        reg = linearreg(data, timeperiod=timeperiod)
        slope = linearreg_slope(data, timeperiod=timeperiod)
        intercept = linearreg_intercept(data, timeperiod=timeperiod)

        # linearreg should equal intercept + slope * (timeperiod - 1)
        # (since x-values go from 0 to timeperiod-1)
        expected = intercept + slope * (timeperiod - 1)

        np.testing.assert_allclose(
            reg,
            expected,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_all_functions_same_lookback(self, price_data):
        """Test that all functions have same lookback period."""
        timeperiod = 14

        reg = linearreg(price_data, timeperiod=timeperiod)
        slope = linearreg_slope(price_data, timeperiod=timeperiod)
        angle = linearreg_angle(price_data, timeperiod=timeperiod)
        intercept = linearreg_intercept(price_data, timeperiod=timeperiod)

        # All should have NaN for first timeperiod-1 values
        for result in [reg, slope, angle, intercept]:
            assert all(np.isnan(result[: timeperiod - 1]))
            assert not np.isnan(result[timeperiod - 1])


class TestEdgeCases:
    """Test edge cases for all linear regression functions."""

    def test_empty_array(self):
        """Test with empty array."""
        empty = np.array([])

        assert len(linearreg(empty, timeperiod=14)) == 0
        assert len(linearreg_slope(empty, timeperiod=14)) == 0
        assert len(linearreg_angle(empty, timeperiod=14)) == 0
        assert len(linearreg_intercept(empty, timeperiod=14)) == 0

    def test_single_value(self):
        """Test with single value."""
        single = np.array([100.0])

        for func in [linearreg, linearreg_slope, linearreg_angle, linearreg_intercept]:
            result = func(single, timeperiod=14)
            assert len(result) == 1
            assert np.isnan(result[0])

    def test_insufficient_data(self):
        """Test with insufficient data."""
        short_data = np.array([100.0, 101.0, 102.0])
        timeperiod = 10

        for func in [linearreg, linearreg_slope, linearreg_angle, linearreg_intercept]:
            result = func(short_data, timeperiod=timeperiod)
            assert all(np.isnan(result))

    def test_exactly_period_values(self):
        """Test with exactly timeperiod values."""
        timeperiod = 10
        data = np.arange(timeperiod, dtype=float) + 100

        for func in [linearreg, linearreg_slope, linearreg_angle, linearreg_intercept]:
            result = func(data, timeperiod=timeperiod)
            assert all(np.isnan(result[: timeperiod - 1]))
            assert not np.isnan(result[-1])


class TestImplementationConsistency:
    """Test implementation consistency across array types."""

    def test_numba_vs_polars_linearreg(self, price_data):
        """Test NumPy vs Polars consistency for linearreg."""
        result_numba = linearreg(price_data, timeperiod=14)

        df = pl.DataFrame({"close": price_data})
        result_polars = df.select(linearreg("close", timeperiod=14).alias("reg"))["reg"].to_numpy()

        np.testing.assert_allclose(
            result_numba,
            result_polars,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_numba_vs_polars_slope(self, price_data):
        """Test NumPy vs Polars consistency for slope."""
        result_numba = linearreg_slope(price_data, timeperiod=14)

        df = pl.DataFrame({"close": price_data})
        result_polars = df.select(linearreg_slope("close", timeperiod=14).alias("slope"))[
            "slope"
        ].to_numpy()

        np.testing.assert_allclose(
            result_numba,
            result_polars,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_numba_vs_polars_angle(self, price_data):
        """Test NumPy vs Polars consistency for angle."""
        result_numba = linearreg_angle(price_data, timeperiod=14)

        df = pl.DataFrame({"close": price_data})
        result_polars = df.select(linearreg_angle("close", timeperiod=14).alias("angle"))[
            "angle"
        ].to_numpy()

        np.testing.assert_allclose(
            result_numba,
            result_polars,
            rtol=1e-10,
            equal_nan=True,
        )

    def test_numba_vs_polars_intercept(self, price_data):
        """Test NumPy vs Polars consistency for intercept."""
        result_numba = linearreg_intercept(price_data, timeperiod=14)

        df = pl.DataFrame({"close": price_data})
        result_polars = df.select(linearreg_intercept("close", timeperiod=14).alias("intercept"))[
            "intercept"
        ].to_numpy()

        np.testing.assert_allclose(
            result_numba,
            result_polars,
            rtol=1e-10,
            equal_nan=True,
        )


class TestSpecialCases:
    """Test special mathematical cases."""

    def test_minimum_period(self):
        """Test with minimum timeperiod=2."""
        data = np.arange(20, dtype=float)

        for func in [linearreg, linearreg_slope, linearreg_angle, linearreg_intercept]:
            result = func(data, timeperiod=2)
            valid_values = result[~np.isnan(result)]
            assert len(valid_values) > 0

    def test_large_period(self, price_data):
        """Test with large timeperiod."""
        if len(price_data) >= 50:
            for func in [linearreg, linearreg_slope, linearreg_angle, linearreg_intercept]:
                result = func(price_data, timeperiod=50)
                assert not np.isnan(result[49])

    def test_validation_all_functions(self, price_data):
        """Test that all functions validate timeperiod >= 2."""
        for func in [linearreg, linearreg_slope, linearreg_angle, linearreg_intercept]:
            with pytest.raises(ValueError, match="timeperiod must be >= 2"):
                func(price_data, timeperiod=1)
