"""
Tests for ML features - Decorator-based feature system.

Tests cover:
- Basic functionality for all 3 implemented features
- Edge cases (insufficient data, NaN handling)
- Parameter validation
- Output structure
- Mathematical correctness
- Registry integration
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.ml import (
    create_lag_features,
    cyclical_encode,
    rolling_entropy,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_ml_data():
    """Sample data for ML feature testing."""
    np.random.seed(42)
    dates = pl.datetime_range(
        start=pl.datetime(2024, 1, 1),
        end=pl.datetime(2024, 2, 29),
        interval="1d",
        eager=True,
    )

    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 2)

    return pl.DataFrame(
        {
            "timestamp": dates,
            "close": prices,
            "returns": np.concatenate([[np.nan], np.diff(np.log(prices))]),
            "hour": [i % 24 for i in range(len(dates))],
            "dayofweek": [i % 7 for i in range(len(dates))],
            "volume": np.random.randint(1000, 10000, len(dates)),
        },
    )


@pytest.fixture
def small_ml_data():
    """Small dataset for edge case testing."""
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 10),
                interval="1d",
                eager=True,
            ),
            "close": [100.0, 101.0, 99.5, 102.0, 101.5, 103.0, 102.0, 104.5, 103.5, 105.0],
            "returns": [np.nan, 0.01, -0.015, 0.025, -0.005, 0.015, -0.01, 0.024, -0.01, 0.014],
            "hour": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
    )


# =============================================================================
# Rolling Entropy Tests
# =============================================================================


def test_entropy_basic_functionality(sample_ml_data):
    """Test rolling entropy basic calculation."""
    result = sample_ml_data.with_columns(
        rolling_entropy("returns", window=20, n_bins=10).alias("entropy_returns_20")
    )

    assert "entropy_returns_20" in result.columns
    assert len(result) == len(sample_ml_data)
    assert result["entropy_returns_20"].dtype == pl.Float64


def test_entropy_with_custom_parameters(sample_ml_data):
    """Test entropy with different window and bin parameters."""
    result = sample_ml_data.with_columns(
        rolling_entropy("returns", window=10, n_bins=5).alias("entropy_returns_10")
    )

    assert "entropy_returns_10" in result.columns
    entropy_values = result["entropy_returns_10"].drop_nulls().drop_nans()
    assert len(entropy_values) > 0


def test_entropy_bounded(sample_ml_data):
    """Test that entropy values are non-negative and bounded."""
    result = sample_ml_data.with_columns(
        rolling_entropy("returns", window=15, n_bins=10).alias("entropy_returns_15")
    )

    entropy_values = result["entropy_returns_15"].drop_nulls().drop_nans().to_numpy()

    # Entropy should be non-negative
    assert np.all(entropy_values >= 0)

    # Entropy is bounded by log2(n_bins)
    max_entropy = np.log2(10)  # n_bins = 10
    assert np.all(entropy_values <= max_entropy * 1.01)  # Small tolerance


# =============================================================================
# Cyclical Encoding Tests
# =============================================================================


def test_cyclical_encode_basic_functionality(sample_ml_data):
    """Test cyclical encoding basic calculation."""
    encodings = cyclical_encode("hour", period=24, name_prefix="hour")
    result = sample_ml_data.with_columns(
        [
            encodings["hour_sin"].alias("hour_sin"),
            encodings["hour_cos"].alias("hour_cos"),
        ]
    )

    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    assert len(result) == len(sample_ml_data)
    assert result["hour_sin"].dtype == pl.Float64
    assert result["hour_cos"].dtype == pl.Float64


def test_cyclical_encode_bounded(sample_ml_data):
    """Test that cyclical encoding produces bounded values."""
    encodings = cyclical_encode("hour", period=24, name_prefix="hour")
    result = sample_ml_data.with_columns(
        [
            encodings["hour_sin"].alias("hour_sin"),
            encodings["hour_cos"].alias("hour_cos"),
        ]
    )

    sin_values = result["hour_sin"].to_numpy()
    cos_values = result["hour_cos"].to_numpy()

    # Should be bounded between -1 and 1
    assert np.all(sin_values >= -1)
    assert np.all(sin_values <= 1)
    assert np.all(cos_values >= -1)
    assert np.all(cos_values <= 1)


def test_cyclical_encode_preserves_cyclical_relationship(small_ml_data):
    """Test that cyclical encoding preserves cyclical relationships."""
    encodings = cyclical_encode("hour", period=24, name_prefix="hour")
    result = small_ml_data.with_columns(
        [
            encodings["hour_sin"].alias("hour_sin"),
            encodings["hour_cos"].alias("hour_cos"),
        ]
    )

    # Hour 0 should be similar to hour 23 (if we had it)
    # Test that hour 0 and hour 1 produce expected sin/cos
    hour_0_sin = result.filter(pl.col("hour") == 0)["hour_sin"][0]
    hour_0_cos = result.filter(pl.col("hour") == 0)["hour_cos"][0]

    # Hour 0: 2π * 0 / 24 = 0
    # sin(0) = 0, cos(0) = 1
    assert abs(hour_0_sin - 0.0) < 0.01
    assert abs(hour_0_cos - 1.0) < 0.01


def test_cyclical_encode_different_periods(sample_ml_data):
    """Test cyclical encoding with different periods."""
    # Day of week encoding (period = 7)
    encodings = cyclical_encode("dayofweek", period=7, name_prefix="dayofweek")
    result = sample_ml_data.with_columns(
        [
            encodings["dayofweek_sin"].alias("dayofweek_sin"),
            encodings["dayofweek_cos"].alias("dayofweek_cos"),
        ]
    )

    assert "dayofweek_sin" in result.columns
    assert "dayofweek_cos" in result.columns

    sin_values = result["dayofweek_sin"].to_numpy()
    cos_values = result["dayofweek_cos"].to_numpy()

    assert np.all(sin_values >= -1)
    assert np.all(sin_values <= 1)
    assert np.all(cos_values >= -1)
    assert np.all(cos_values <= 1)


# =============================================================================
# Lag Features Tests
# =============================================================================


def test_lag_features_basic_functionality(sample_ml_data):
    """Test lag features basic calculation."""
    lag_dict = create_lag_features(
        "returns", lags=[1, 2, 5], include_diff=False, include_ratio=False
    )
    result = sample_ml_data.with_columns(
        [
            lag_dict["lag_1"].alias("returns_lag_1"),
            lag_dict["lag_2"].alias("returns_lag_2"),
            lag_dict["lag_5"].alias("returns_lag_5"),
        ]
    )

    assert "returns_lag_1" in result.columns
    assert "returns_lag_2" in result.columns
    assert "returns_lag_5" in result.columns
    assert len(result) == len(sample_ml_data)


def test_lag_features_correctness(small_ml_data):
    """Test that lag features produce correct lagged values."""
    lag_dict = create_lag_features("returns", lags=[1, 2], include_diff=False, include_ratio=False)
    result = small_ml_data.with_columns(
        [
            lag_dict["lag_1"].alias("returns_lag_1"),
            lag_dict["lag_2"].alias("returns_lag_2"),
        ]
    )

    # Check lag_1: should match previous row
    for i in range(1, len(result)):
        expected_lag_1 = small_ml_data["returns"][i - 1]
        actual_lag_1 = result["returns_lag_1"][i]

        if expected_lag_1 is not None and not np.isnan(expected_lag_1):
            assert abs(actual_lag_1 - expected_lag_1) < 1e-10

    # Check lag_2: should match 2 rows back
    for i in range(2, len(result)):
        expected_lag_2 = small_ml_data["returns"][i - 2]
        actual_lag_2 = result["returns_lag_2"][i]

        if expected_lag_2 is not None and not np.isnan(expected_lag_2):
            assert abs(actual_lag_2 - expected_lag_2) < 1e-10


def test_lag_features_with_diff(sample_ml_data):
    """Test lag features with differences."""
    lag_dict = create_lag_features("returns", lags=[1, 3], include_diff=True, include_ratio=False)
    result = sample_ml_data.with_columns(
        [
            lag_dict["lag_1"].alias("returns_lag_1"),
            lag_dict["diff_1"].alias("returns_diff_1"),
            lag_dict["lag_3"].alias("returns_lag_3"),
            lag_dict["diff_3"].alias("returns_diff_3"),
        ]
    )

    assert "returns_lag_1" in result.columns
    assert "returns_diff_1" in result.columns
    assert "returns_lag_3" in result.columns
    assert "returns_diff_3" in result.columns


def test_lag_features_with_ratio(sample_ml_data):
    """Test lag features with ratios."""
    lag_dict = create_lag_features("close", lags=[1, 2], include_diff=False, include_ratio=True)
    result = sample_ml_data.with_columns(
        [
            lag_dict["lag_1"].alias("close_lag_1"),
            lag_dict["ratio_1"].alias("close_ratio_1"),
            lag_dict["lag_2"].alias("close_lag_2"),
            lag_dict["ratio_2"].alias("close_ratio_2"),
        ]
    )

    assert "close_lag_1" in result.columns
    assert "close_ratio_1" in result.columns
    assert "close_lag_2" in result.columns
    assert "close_ratio_2" in result.columns


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_insufficient_data():
    """Test all features handle insufficient data gracefully."""
    tiny_df = pl.DataFrame(
        {
            "close": [100.0, 101.0],
            "returns": [np.nan, 0.01],
            "hour": [0, 1],
        },
    )

    # All features should handle insufficient data without crashing
    result_entropy = tiny_df.with_columns(
        rolling_entropy("returns", window=10, n_bins=10).alias("entropy_returns_10")
    )

    encodings = cyclical_encode("hour", period=24, name_prefix="hour")
    result_cyclical = tiny_df.with_columns(
        [
            encodings["hour_sin"].alias("hour_sin"),
            encodings["hour_cos"].alias("hour_cos"),
        ]
    )

    lag_dict = create_lag_features("returns", lags=[1, 2], include_diff=False, include_ratio=False)
    result_lag = tiny_df.with_columns(
        [
            lag_dict["lag_1"].alias("returns_lag_1"),
            lag_dict["lag_2"].alias("returns_lag_2"),
        ]
    )

    assert len(result_entropy) == 2
    assert len(result_cyclical) == 2
    assert len(result_lag) == 2


def test_nan_data():
    """Test features handle NaN values in data."""
    data_with_nan = pl.DataFrame(
        {
            "close": [100.0, np.nan, 102.0, 101.0, np.nan, 103.0, 102.0, 104.0, np.nan, 105.0],
            "returns": [np.nan, 0.01, np.nan, -0.01, 0.02, np.nan, -0.01, 0.02, -0.01, np.nan],
            "hour": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
    )

    # All features should handle NaN gracefully
    result_entropy = data_with_nan.with_columns(
        rolling_entropy("returns", window=5, n_bins=10).alias("entropy_returns_5")
    )

    encodings = cyclical_encode("hour", period=24, name_prefix="hour")
    result_cyclical = data_with_nan.with_columns(
        [
            encodings["hour_sin"].alias("hour_sin"),
            encodings["hour_cos"].alias("hour_cos"),
        ]
    )

    lag_dict = create_lag_features("returns", lags=[1], include_diff=False, include_ratio=False)
    result_lag = data_with_nan.with_columns(lag_dict["lag_1"].alias("returns_lag_1"))

    assert len(result_entropy) == 10
    assert len(result_cyclical) == 10
    assert len(result_lag) == 10


# =============================================================================
# Registry Tests
# =============================================================================


def test_registry_has_ml_features():
    """Test that ML features are registered in the global registry."""
    from ml4t.engineer.core.registry import get_registry

    registry = get_registry()

    # Check all implemented ML features are registered
    assert registry.get("rolling_entropy") is not None
    assert registry.get("cyclical_encode") is not None
    assert registry.get("create_lag_features") is not None

    # Verify category
    meta = registry.get("rolling_entropy")
    assert meta.category == "ml"

    meta = registry.get("cyclical_encode")
    assert meta.category == "ml"

    meta = registry.get("create_lag_features")
    assert meta.category == "ml"


def test_registry_metadata_completeness():
    """Test that registered features have complete metadata."""
    from ml4t.engineer.core.registry import get_registry

    registry = get_registry()

    for feature_name in ["rolling_entropy", "cyclical_encode", "create_lag_features"]:
        meta = registry.get(feature_name)
        assert meta is not None
        assert meta.name == feature_name
        assert meta.category == "ml"
        assert meta.description
        assert meta.lookback is not None


# =============================================================================
# Parameter Validation Tests
# =============================================================================


def test_entropy_parameter_validation(sample_ml_data):
    """Test that rolling_entropy validates parameters."""
    # Invalid window
    with pytest.raises(ValueError):
        sample_ml_data.with_columns(
            rolling_entropy("returns", window=0, n_bins=10).alias("entropy")
        )

    # Invalid n_bins
    with pytest.raises(ValueError):
        sample_ml_data.with_columns(
            rolling_entropy("returns", window=20, n_bins=1).alias("entropy")
        )


def test_cyclical_encode_parameter_validation(sample_ml_data):  # noqa: ARG001
    """Test that cyclical_encode validates parameters."""
    # Invalid period
    with pytest.raises(ValueError):
        cyclical_encode("hour", period=0, name_prefix="hour")

    with pytest.raises(ValueError):
        cyclical_encode("hour", period=-1, name_prefix="hour")

    # Invalid name_prefix
    with pytest.raises(TypeError):
        cyclical_encode("hour", period=24, name_prefix=123)  # type: ignore


def test_lag_features_parameter_validation(sample_ml_data):  # noqa: ARG001
    """Test that create_lag_features validates parameters."""
    # Empty lags list
    with pytest.raises(ValueError):
        create_lag_features("returns", lags=[], include_diff=False, include_ratio=False)

    # Non-positive lag
    with pytest.raises(ValueError):
        create_lag_features("returns", lags=[1, 0, 2], include_diff=False, include_ratio=False)

    with pytest.raises(ValueError):
        create_lag_features("returns", lags=[1, -1, 2], include_diff=False, include_ratio=False)

    # Non-integer lag
    with pytest.raises(TypeError):
        create_lag_features("returns", lags=[1, 2.5, 3], include_diff=False, include_ratio=False)  # type: ignore

    # Invalid boolean parameters
    with pytest.raises(TypeError):
        create_lag_features("returns", lags=[1, 2], include_diff="yes", include_ratio=False)  # type: ignore

    with pytest.raises(TypeError):
        create_lag_features("returns", lags=[1, 2], include_diff=True, include_ratio="no")  # type: ignore
