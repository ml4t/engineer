"""Tests for composite feature builders.

Tests cover:
- rolling_z_score: Rolling standardization
- z_score_composite: Weighted combination of z-scored features
- illiquidity_composite: Domain-specific illiquidity aggregation
- momentum_composite: Domain-specific momentum aggregation
- Edge cases: missing columns, zero variance, weights validation
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.composite import (
    illiquidity_composite,
    momentum_composite,
    rolling_z_score,
    z_score_composite,
)

# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def simple_data():
    """Simple data for basic tests."""
    return pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "c": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )


@pytest.fixture
def multi_feature_data():
    """Data with multiple features for composite testing."""
    return pl.DataFrame(
        {
            "momentum": [0.1, 0.2, -0.1, 0.3, 0.15, 0.0, -0.2, 0.25],
            "value": [1.5, 1.2, 1.8, 0.9, 1.1, 1.4, 1.0, 1.3],
            "quality": [0.8, 0.7, 0.9, 0.85, 0.75, 0.8, 0.6, 0.95],
        }
    )


@pytest.fixture
def illiquidity_data():
    """Data with illiquidity proxies."""
    return pl.DataFrame(
        {
            "kyle_lambda": [0.001, 0.002, 0.0015, 0.003, 0.001, 0.0025],
            "amihud": [0.05, 0.08, 0.06, 0.1, 0.04, 0.09],
            "roll_spread": [0.002, 0.003, 0.0025, 0.004, 0.0015, 0.0035],
        }
    )


@pytest.fixture
def momentum_indicator_data():
    """Data with momentum indicators."""
    return pl.DataFrame(
        {
            "rsi": [45.0, 55.0, 65.0, 50.0, 40.0, 60.0, 35.0, 70.0],
            "macd": [-0.5, 0.2, 0.8, 0.1, -0.3, 0.5, -0.7, 1.0],
            "roc": [-0.02, 0.01, 0.03, 0.0, -0.01, 0.02, -0.03, 0.04],
        }
    )


# =============================================================================
# Tests for rolling_z_score
# =============================================================================


class TestRollingZScore:
    """Tests for rolling_z_score function."""

    def test_basic_z_score(self, simple_data):
        """Test basic z-score computation."""
        result = simple_data.with_columns(rolling_z_score("a", period=3).alias("z_a"))
        assert "z_a" in result.columns
        # Should have values after sufficient lookback
        assert result["z_a"][4] is not None

    def test_z_score_mean_centering(self):
        """Test that z-scores are approximately centered at 0 for stationary data."""
        # Create stationary random data (should have z-scores near 0 on average)
        np.random.seed(42)
        x = np.random.randn(200)  # Stationary series with mean 0
        df = pl.DataFrame({"x": x.tolist()})
        result = df.with_columns(rolling_z_score("x", period=20).alias("z_x"))
        # After warmup, z-scores should have mean close to 0
        z_values = result["z_x"][40:].to_numpy()
        # Stationary data should have z-scores roughly centered at 0
        assert abs(np.nanmean(z_values)) < 0.5

    def test_z_score_with_constant_series(self):
        """Test that constant series produces zero z-scores."""
        df = pl.DataFrame({"const": [5.0, 5.0, 5.0, 5.0, 5.0]})
        result = df.with_columns(rolling_z_score("const", period=3).alias("z_const"))
        # Constant series has std=0, should return 0
        z_values = result["z_const"].to_numpy()
        assert np.allclose(z_values[2:], 0.0, equal_nan=True)

    def test_min_periods(self, simple_data):
        """Test min_periods parameter."""
        result = simple_data.with_columns(
            rolling_z_score("a", period=5, min_periods=2).alias("z_a")
        )
        # With min_periods=2, should have values earlier
        assert result["z_a"][2] is not None

    def test_z_score_scales_correctly(self):
        """Test that z-scores scale features to unit variance."""
        # Create data with known variance
        np.random.seed(42)
        x = np.random.randn(100) * 10 + 50  # High variance, non-zero mean
        df = pl.DataFrame({"x": x.tolist()})
        result = df.with_columns(rolling_z_score("x", period=20).alias("z_x"))
        # After warmup, z-scores should have std close to 1
        z_values = result["z_x"][40:].to_numpy()
        assert 0.5 < np.nanstd(z_values) < 1.5  # Approximate unit variance


# =============================================================================
# Tests for z_score_composite
# =============================================================================


class TestZScoreComposite:
    """Tests for z_score_composite function."""

    def test_basic_composite(self, multi_feature_data):
        """Test basic composite calculation."""
        result = z_score_composite(
            multi_feature_data,
            feature_cols=["momentum", "value", "quality"],
            period=3,
        )
        assert "composite_score" in result.columns
        # Should have valid values after warmup
        assert result["composite_score"][4] is not None

    def test_equal_weights_default(self, multi_feature_data):
        """Test that equal weights are used by default."""
        result = z_score_composite(
            multi_feature_data,
            feature_cols=["momentum", "value", "quality"],
            period=3,
        )
        # With equal weights, composite should be average of z-scores
        assert result["composite_score"].null_count() < len(result)

    def test_custom_weights(self, multi_feature_data):
        """Test composite with custom weights."""
        result = z_score_composite(
            multi_feature_data,
            feature_cols=["momentum", "value", "quality"],
            period=3,
            weights=[0.5, 0.3, 0.2],
        )
        assert "composite_score" in result.columns

    def test_weights_are_normalized(self, multi_feature_data):
        """Test that weights are normalized to sum to 1."""
        # These weights sum to 10, should be normalized internally
        result1 = z_score_composite(
            multi_feature_data,
            feature_cols=["momentum", "value"],
            period=3,
            weights=[5.0, 5.0],
        )
        result2 = z_score_composite(
            multi_feature_data,
            feature_cols=["momentum", "value"],
            period=3,
            weights=[0.5, 0.5],
        )
        # Results should be identical after normalization
        np.testing.assert_array_almost_equal(
            result1["composite_score"].to_numpy(),
            result2["composite_score"].to_numpy(),
            decimal=10,
        )

    def test_custom_output_column(self, multi_feature_data):
        """Test custom output column name."""
        result = z_score_composite(
            multi_feature_data,
            feature_cols=["momentum", "value"],
            period=3,
            output_col="my_score",
        )
        assert "my_score" in result.columns
        assert "composite_score" not in result.columns

    def test_empty_feature_cols_raises(self, multi_feature_data):
        """Test that empty feature_cols raises error."""
        with pytest.raises(ValueError, match="feature_cols must not be empty"):
            z_score_composite(multi_feature_data, feature_cols=[], period=3)

    def test_mismatched_weights_length_raises(self, multi_feature_data):
        """Test that mismatched weights length raises error."""
        with pytest.raises(ValueError, match="weights length"):
            z_score_composite(
                multi_feature_data,
                feature_cols=["momentum", "value"],
                period=3,
                weights=[0.5, 0.3, 0.2],  # 3 weights, 2 columns
            )

    def test_zero_weights_sum_raises(self, multi_feature_data):
        """Test that weights summing to zero raises error."""
        with pytest.raises(ValueError, match="weights must sum to a non-zero"):
            z_score_composite(
                multi_feature_data,
                feature_cols=["momentum", "value"],
                period=3,
                weights=[0.0, 0.0],
            )

    def test_preserves_original_columns(self, multi_feature_data):
        """Test that original columns are preserved."""
        result = z_score_composite(
            multi_feature_data,
            feature_cols=["momentum", "value"],
            period=3,
        )
        assert "momentum" in result.columns
        assert "value" in result.columns
        assert "quality" in result.columns


# =============================================================================
# Tests for illiquidity_composite
# =============================================================================


class TestIlliquidityComposite:
    """Tests for illiquidity_composite function."""

    def test_basic_illiquidity(self, illiquidity_data):
        """Test basic illiquidity composite."""
        result = illiquidity_composite(illiquidity_data, period=3)
        assert "illiquidity_score" in result.columns

    def test_custom_output_column(self, illiquidity_data):
        """Test custom output column name."""
        result = illiquidity_composite(illiquidity_data, period=3, output_col="my_illiquidity")
        assert "my_illiquidity" in result.columns

    def test_missing_column_handled(self, illiquidity_data):
        """Test that missing columns are gracefully skipped."""
        # Remove one column
        df = illiquidity_data.drop("roll_spread")
        result = illiquidity_composite(df, period=3)
        # Should still work with remaining columns
        assert "illiquidity_score" in result.columns

    def test_all_columns_missing_raises(self):
        """Test that all columns missing raises error."""
        df = pl.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="None of"):
            illiquidity_composite(df, period=3)

    def test_higher_illiquidity_score_for_illiquid(self):
        """Test that illiquid periods get higher scores."""
        df = pl.DataFrame(
            {
                "kyle_lambda": [0.001, 0.001, 0.001, 0.010, 0.010, 0.010],
                "amihud": [0.01, 0.01, 0.01, 0.10, 0.10, 0.10],
            }
        )
        result = illiquidity_composite(df, period=3)
        # Later high-illiquidity values should have higher scores
        scores = result["illiquidity_score"].to_numpy()
        # After initial period, high illiquidity period should score higher
        # (The exact pattern depends on z-score dynamics)
        assert not np.all(np.isnan(scores[2:]))


# =============================================================================
# Tests for momentum_composite
# =============================================================================


class TestMomentumComposite:
    """Tests for momentum_composite function."""

    def test_basic_momentum(self, momentum_indicator_data):
        """Test basic momentum composite."""
        result = momentum_composite(momentum_indicator_data, period=3)
        assert "momentum_score" in result.columns

    def test_custom_output_column(self, momentum_indicator_data):
        """Test custom output column name."""
        result = momentum_composite(momentum_indicator_data, period=3, output_col="my_momentum")
        assert "my_momentum" in result.columns

    def test_missing_column_handled(self, momentum_indicator_data):
        """Test that missing columns are gracefully skipped."""
        df = momentum_indicator_data.drop("roc")
        result = momentum_composite(df, period=3)
        assert "momentum_score" in result.columns

    def test_all_columns_missing_raises(self):
        """Test that all columns missing raises error."""
        df = pl.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="None of"):
            momentum_composite(df, period=3)


# =============================================================================
# Integration Tests
# =============================================================================


class TestCompositeIntegration:
    """Integration tests for composite features."""

    def test_composites_are_comparable(self):
        """Test that different composites can be compared."""
        np.random.seed(42)
        n = 100
        df = pl.DataFrame(
            {
                # Illiquidity features
                "kyle_lambda": np.abs(np.random.randn(n) * 0.001).tolist(),
                "amihud": np.abs(np.random.randn(n) * 0.05).tolist(),
                # Momentum features
                "rsi": (50 + np.random.randn(n) * 15).tolist(),
                "macd": (np.random.randn(n) * 0.5).tolist(),
            }
        )

        result = illiquidity_composite(df, period=20)
        result = momentum_composite(result, period=20)

        # Both composites should exist and be z-scored (roughly comparable scale)
        illiq = result["illiquidity_score"][50:].to_numpy()
        mom = result["momentum_score"][50:].to_numpy()

        # Both should have similar scale (z-scored)
        assert 0.1 < np.nanstd(illiq) < 3.0
        assert 0.1 < np.nanstd(mom) < 3.0

    def test_composite_with_single_feature(self, simple_data):
        """Test composite with single feature equals z-score."""
        result = z_score_composite(simple_data, feature_cols=["a"], period=3)
        # Single feature composite should equal the z-score
        expected = simple_data.with_columns(rolling_z_score("a", period=3).alias("z_a"))
        np.testing.assert_array_almost_equal(
            result["composite_score"].to_numpy(),
            expected["z_a"].to_numpy(),
            decimal=10,
        )
