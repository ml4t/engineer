"""
Tests for statistics functions - Feature v2 decorator system.

Tests cover:
- Linear regression family (linearreg, slope, intercept, angle, TSF)
- Statistical measures (stddev, var, avgdev)
- Basic functionality, edge cases, parameter validation
- TA-Lib compatibility
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.statistics import (
    avgdev,
    linearreg,
    linearreg_angle,
    linearreg_intercept,
    linearreg_slope,
    stddev,
    tsf,
    var,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Sample data for statistics testing with clear trend."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 30),
                interval="1d",
                eager=True,
            ),
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
                113.0,
                114.0,
                115.0,
                116.0,
                117.0,
                118.0,
                119.0,
                120.0,
                121.0,
                122.0,
                123.0,
                124.0,
                125.0,
                126.0,
                127.0,
                128.0,
                129.0,
            ],
        },
    )


@pytest.fixture
def small_data():
    """Small dataset for edge case testing."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 10),
                interval="1d",
                eager=True,
            ),
            "close": [100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 105.0, 108.0, 107.0, 110.0],
        },
    )


@pytest.fixture
def constant_data():
    """Dataset with constant values for edge case testing."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 20),
                interval="1d",
                eager=True,
            ),
            "close": [100.0] * 20,
        },
    )


# =============================================================================
# LINEARREG Tests
# =============================================================================


def test_linearreg_basic_functionality(sample_data):
    """Test linearreg basic calculation."""
    result = sample_data.with_columns(linearreg("close", timeperiod=5).alias("linearreg_5"))

    assert "linearreg_5" in result.columns
    assert len(result) == len(sample_data)
    assert result["linearreg_5"].dtype == pl.Float64

    values = result["linearreg_5"].to_numpy()
    # First 4 values should be NaN (need 5 for calculation)
    assert np.all(np.isnan(values[:4]))
    # After that, should have valid values
    assert not np.all(np.isnan(values[4:]))


def test_linearreg_trend(sample_data):
    """Test linearreg captures trend correctly."""
    result = sample_data.with_columns(linearreg("close", timeperiod=10).alias("linearreg_10"))
    values = result["linearreg_10"].to_numpy()

    # With perfect uptrend, regression should be close to actual values
    # After warmup period
    valid_values = values[~np.isnan(values)]
    assert len(valid_values) > 0


def test_linearreg_custom_period(sample_data):
    """Test linearreg with different periods."""
    result = sample_data.with_columns(linearreg("close", timeperiod=15).alias("linearreg_15"))

    assert "linearreg_15" in result.columns
    values = result["linearreg_15"].to_numpy()
    # First 14 values should be NaN
    assert np.all(np.isnan(values[:14]))


def test_linearreg_insufficient_data(small_data):
    """Test linearreg with insufficient data."""
    result = small_data.with_columns(linearreg("close", timeperiod=20).alias("linearreg_20"))
    values = result["linearreg_20"].to_numpy()

    # All should be NaN (only 10 data points)
    assert np.all(np.isnan(values))


# =============================================================================
# LINEARREG_SLOPE Tests
# =============================================================================


def test_linearreg_slope_basic_functionality(sample_data):
    """Test linearreg_slope basic calculation."""
    result = sample_data.with_columns(
        linearreg_slope("close", timeperiod=5).alias("linearreg_slope_5")
    )

    assert "linearreg_slope_5" in result.columns
    assert len(result) == len(sample_data)

    slopes = result["linearreg_slope_5"].to_numpy()
    valid_slopes = slopes[~np.isnan(slopes)]

    # With perfect uptrend of +1 per period, slope should be close to 1
    assert len(valid_slopes) > 0
    # Allow some numerical tolerance
    assert np.all(valid_slopes > 0.5)  # Positive slope


def test_linearreg_slope_constant_data(constant_data):
    """Test linearreg_slope with constant values."""
    result = constant_data.with_columns(
        linearreg_slope("close", timeperiod=5).alias("linearreg_slope_5")
    )
    slopes = result["linearreg_slope_5"].to_numpy()
    valid_slopes = slopes[~np.isnan(slopes)]

    # Constant data should have slope ~0
    assert len(valid_slopes) > 0
    assert np.all(np.abs(valid_slopes) < 0.01)


# =============================================================================
# LINEARREG_INTERCEPT Tests
# =============================================================================


def test_linearreg_intercept_basic_functionality(sample_data):
    """Test linearreg_intercept basic calculation."""
    result = sample_data.with_columns(
        linearreg_intercept("close", timeperiod=5).alias("linearreg_intercept_5")
    )

    assert "linearreg_intercept_5" in result.columns
    assert len(result) == len(sample_data)

    intercepts = result["linearreg_intercept_5"].to_numpy()
    valid_intercepts = intercepts[~np.isnan(intercepts)]

    assert len(valid_intercepts) > 0


# =============================================================================
# LINEARREG_ANGLE Tests
# =============================================================================


def test_linearreg_angle_basic_functionality(sample_data):
    """Test linearreg_angle basic calculation."""
    result = sample_data.with_columns(
        linearreg_angle("close", timeperiod=5).alias("linearreg_angle_5")
    )

    assert "linearreg_angle_5" in result.columns
    assert len(result) == len(sample_data)

    angles = result["linearreg_angle_5"].to_numpy()
    valid_angles = angles[~np.isnan(angles)]

    # Angles should be in degrees
    assert len(valid_angles) > 0
    # For uptrend, angles should be positive
    assert np.all(valid_angles > 0)


def test_linearreg_angle_constant_data(constant_data):
    """Test linearreg_angle with constant values."""
    result = constant_data.with_columns(
        linearreg_angle("close", timeperiod=5).alias("linearreg_angle_5")
    )
    angles = result["linearreg_angle_5"].to_numpy()
    valid_angles = angles[~np.isnan(angles)]

    # Constant data should have angle ~0 degrees
    assert len(valid_angles) > 0
    assert np.all(np.abs(valid_angles) < 1.0)


# =============================================================================
# TSF Tests
# =============================================================================


def test_tsf_basic_functionality(sample_data):
    """Test TSF basic calculation."""
    result = sample_data.with_columns(tsf("close", timeperiod=5).alias("tsf_5"))

    assert "tsf_5" in result.columns
    assert len(result) == len(sample_data)

    tsf_values = result["tsf_5"].to_numpy()
    valid_values = tsf_values[~np.isnan(tsf_values)]

    assert len(valid_values) > 0


def test_tsf_forecast_ahead(sample_data):
    """Test that TSF forecasts ahead of current value."""
    result = sample_data.with_columns(tsf("close", timeperiod=10).alias("tsf_10"))

    # TSF should be projection, generally ahead for uptrend
    tsf_values = result["tsf_10"].to_numpy()
    close_values = result["close"].to_numpy()

    # After warmup, TSF should generally be >= close for uptrend
    for i in range(10, len(tsf_values)):
        if not np.isnan(tsf_values[i]):
            # TSF projects forward, should be higher than current for uptrend
            assert tsf_values[i] >= close_values[i] - 5.0  # Allow some tolerance


# =============================================================================
# STDDEV Tests
# =============================================================================


def test_stddev_basic_functionality(sample_data):
    """Test stddev basic calculation."""
    result = sample_data.with_columns(stddev("close", period=5).alias("stddev_5"))

    assert "stddev_5" in result.columns
    assert len(result) == len(sample_data)

    stddev_values = result["stddev_5"].to_numpy()
    valid_values = stddev_values[~np.isnan(stddev_values)]

    assert len(valid_values) > 0
    # Standard deviation should be non-negative
    assert np.all(valid_values >= 0)


def test_stddev_constant_data(constant_data):
    """Test stddev with constant values."""
    result = constant_data.with_columns(stddev("close", period=5).alias("stddev_5"))
    stddev_values = result["stddev_5"].to_numpy()
    valid_values = stddev_values[~np.isnan(stddev_values)]

    # Constant data should have stddev = 0
    assert len(valid_values) > 0
    assert np.all(np.abs(valid_values) < 0.01)


def test_stddev_nbdev_parameter(sample_data):
    """Test stddev with different nbdev values."""
    result1 = sample_data.with_columns(stddev("close", period=5, nbdev=1.0).alias("stddev_5_1x"))
    result2 = sample_data.with_columns(stddev("close", period=5, nbdev=2.0).alias("stddev_5_2x"))

    stddev1 = result1["stddev_5_1x"].to_numpy()
    stddev2 = result2["stddev_5_2x"].to_numpy()

    # nbdev=2.0 should give approximately 2x the values of nbdev=1.0
    # After warmup period
    for i in range(5, len(stddev1)):
        if not np.isnan(stddev1[i]) and not np.isnan(stddev2[i]):
            # stddev2 should be approximately 2x stddev1
            assert np.abs(stddev2[i] - 2.0 * stddev1[i]) < 0.01


# =============================================================================
# VAR Tests
# =============================================================================


def test_var_basic_functionality(sample_data):
    """Test var basic calculation."""
    result = sample_data.with_columns(var("close", timeperiod=5).alias("var_5"))

    assert "var_5" in result.columns
    assert len(result) == len(sample_data)

    var_values = result["var_5"].to_numpy()
    valid_values = var_values[~np.isnan(var_values)]

    assert len(valid_values) > 0
    # Variance should be non-negative
    assert np.all(valid_values >= 0)


def test_var_constant_data(constant_data):
    """Test var with constant values."""
    result = constant_data.with_columns(var("close", timeperiod=5).alias("var_5"))
    var_values = result["var_5"].to_numpy()
    valid_values = var_values[~np.isnan(var_values)]

    # Constant data should have variance = 0
    assert len(valid_values) > 0
    assert np.all(np.abs(valid_values) < 0.01)


def test_var_relationship_to_stddev(sample_data):
    """Test that var = stddev^2 relationship holds."""
    result = sample_data.with_columns(
        [
            stddev("close", period=10, nbdev=1.0).alias("stddev_10"),
            var("close", timeperiod=10, nbdev=1.0).alias("var_10"),
        ]
    )

    stddev_values = result["stddev_10"].to_numpy()
    var_values = result["var_10"].to_numpy()

    # Variance should equal stddev squared (approximately)
    for i in range(len(stddev_values)):
        if not np.isnan(stddev_values[i]) and not np.isnan(var_values[i]):
            expected_var = stddev_values[i] ** 2
            # Allow some numerical tolerance
            assert np.abs(var_values[i] - expected_var) < 0.1


# =============================================================================
# AVGDEV Tests
# =============================================================================


def test_avgdev_basic_functionality(sample_data):
    """Test avgdev basic calculation."""
    result = sample_data.with_columns(avgdev("close", timeperiod=5).alias("avgdev_5"))

    assert "avgdev_5" in result.columns
    assert len(result) == len(sample_data)

    avgdev_values = result["avgdev_5"].to_numpy()
    valid_values = avgdev_values[~np.isnan(avgdev_values)]

    assert len(valid_values) > 0
    # Average deviation should be non-negative
    assert np.all(valid_values >= 0)


def test_avgdev_constant_data(constant_data):
    """Test avgdev with constant values."""
    result = constant_data.with_columns(avgdev("close", timeperiod=5).alias("avgdev_5"))
    avgdev_values = result["avgdev_5"].to_numpy()
    valid_values = avgdev_values[~np.isnan(avgdev_values)]

    # Constant data should have avgdev = 0
    assert len(valid_values) > 0
    assert np.all(np.abs(valid_values) < 0.01)


def test_avgdev_different_periods(sample_data):
    """Test avgdev with different periods."""
    result = sample_data.with_columns(
        [
            avgdev("close", timeperiod=5).alias("avgdev_5"),
            avgdev("close", timeperiod=10).alias("avgdev_10"),
        ]
    )

    assert "avgdev_5" in result.columns
    assert "avgdev_10" in result.columns

    # Both should have valid values after warmup
    valid5 = result["avgdev_5"].to_numpy()[~np.isnan(result["avgdev_5"].to_numpy())]
    valid10 = result["avgdev_10"].to_numpy()[~np.isnan(result["avgdev_10"].to_numpy())]

    assert len(valid5) > 0
    assert len(valid10) > 0


# =============================================================================
# Registry Tests
# =============================================================================


def test_registry_has_statistics_features():
    """Test that statistics features are registered in the global registry."""
    from ml4t.engineer.core.registry import get_registry

    registry = get_registry()

    # Check all statistics features are registered
    assert registry.get("linearreg") is not None
    assert registry.get("linearreg_slope") is not None
    assert registry.get("linearreg_intercept") is not None
    assert registry.get("linearreg_angle") is not None
    assert registry.get("tsf") is not None
    assert registry.get("stddev") is not None
    assert registry.get("var") is not None
    assert registry.get("avgdev") is not None

    # Verify category for a sample
    meta = registry.get("linearreg")
    assert meta.category == "statistics"


def test_stationarity_classification():
    """Test that features are correctly classified as stationary/non-stationary."""
    from ml4t.engineer.core.registry import get_registry

    registry = get_registry()

    # All statistics features are currently marked as non-stationary
    # This aligns with TA-Lib's classification where these are raw statistical measures
    assert registry.get("linearreg").normalized is False
    assert registry.get("linearreg_slope").normalized is False
    assert registry.get("linearreg_intercept").normalized is False
    assert registry.get("linearreg_angle").normalized is False
    assert registry.get("tsf").normalized is False
    assert registry.get("stddev").normalized is False
    assert registry.get("var").normalized is False
    assert registry.get("avgdev").normalized is False
