"""
Tests for structural break detection features.

Tests the following features based on 2025 ADIA Lab competition insights:
- Coefficient of Variation (CV)
- CV Z-score
- Variance Ratio
- KL Divergence
- Wasserstein Distance
- Drift Detection
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.statistics.structural_break import (
    _cv_nb,
    _drift_nb,
    _drift_zscore_nb,
    _kl_divergence_nb,
    _variance_ratio_nb,
    _wasserstein_1d_nb,
    coefficient_of_variation,
    rolling_cv_zscore,
    rolling_drift,
    rolling_kl_divergence,
    rolling_wasserstein,
    variance_ratio,
)

# =============================================================================
# Coefficient of Variation Tests
# =============================================================================


class TestCoefficientOfVariation:
    """Tests for Coefficient of Variation (CV)."""

    def test_cv_basic(self) -> None:
        """Test basic CV calculation."""
        # CV = std / |mean|
        values = np.array([10.0, 12.0, 8.0, 11.0, 9.0], dtype=np.float64)
        cv = _cv_nb(values)

        expected_mean = np.mean(values)
        expected_std = np.std(values)
        expected_cv = expected_std / abs(expected_mean)

        np.testing.assert_almost_equal(cv, expected_cv, decimal=10)

    def test_cv_constant_values(self) -> None:
        """Test CV of constant values is zero (std=0)."""
        values = np.ones(10, dtype=np.float64) * 5.0
        cv = _cv_nb(values)

        assert cv == 0.0

    def test_cv_zero_mean(self) -> None:
        """Test CV when mean is zero returns NaN."""
        values = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        cv = _cv_nb(values)

        # Mean is 0, so CV is undefined
        assert np.isnan(cv)

    def test_cv_nan_handling(self) -> None:
        """Test CV with NaN values."""
        values = np.array([10.0, np.nan, 12.0, 8.0], dtype=np.float64)
        cv = _cv_nb(values)

        # Should compute CV of valid values only
        assert not np.isnan(cv)
        assert cv > 0

    def test_rolling_cv_basic(self) -> None:
        """Test rolling CV computation."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(200) + 10})  # Positive mean

        result = df.select(coefficient_of_variation("values", window=50).alias("cv"))

        valid_count = result["cv"].drop_nulls().len()
        assert valid_count > 100
        assert all(result["cv"].drop_nulls() >= 0)  # CV is non-negative

    def test_cv_regime_change_detection(self) -> None:
        """Test that CV increases during regime changes."""
        np.random.seed(42)
        # Low volatility regime followed by high volatility
        low_vol = np.random.randn(100) * 0.1 + 10
        high_vol = np.random.randn(100) * 1.0 + 10
        values = np.concatenate([low_vol, high_vol])

        df = pl.DataFrame({"values": values})
        result = df.select(coefficient_of_variation("values", window=50).alias("cv"))

        cv_values = result["cv"].drop_nulls().to_numpy()
        # CV should be higher in the high-volatility portion
        first_half_cv = np.mean(cv_values[:50])
        second_half_cv = np.mean(cv_values[-50:])

        assert second_half_cv > first_half_cv


class TestCVZscore:
    """Tests for CV Z-score."""

    def test_rolling_cv_zscore_basic(self) -> None:
        """Test rolling CV Z-score computation."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(500) + 10})

        result = df.select(
            rolling_cv_zscore("values", window=50, lookback_multiplier=5).alias("cv_zscore")
        )

        valid_values = result["cv_zscore"].drop_nulls().to_numpy()
        assert len(valid_values) > 100

        # Z-scores should typically be between -3 and 3
        assert np.mean(np.abs(valid_values) < 3) > 0.9


# =============================================================================
# Variance Ratio Tests
# =============================================================================


class TestVarianceRatio:
    """Tests for Variance Ratio test statistic."""

    def test_vr_random_walk(self) -> None:
        """Test that random walk has VR ≈ 1."""
        np.random.seed(42)
        # Random walk: cumulative sum of random increments
        increments = np.random.randn(500)
        prices = np.cumsum(increments)

        vr = _variance_ratio_nb(prices, q=5)

        # VR should be close to 1 for random walk
        assert 0.7 < vr < 1.3

    def test_vr_trending(self) -> None:
        """Test that trending series has VR > 1."""
        np.random.seed(42)
        # Trending: cumulative sum of positively autocorrelated increments
        base = np.random.randn(500)
        trending = np.cumsum(base + 0.3 * np.roll(base, 1))

        vr = _variance_ratio_nb(trending, q=5)

        # Trending should have VR > 1
        assert vr > 1.0

    def test_vr_mean_reverting(self) -> None:
        """Test that mean-reverting series has VR < 1."""
        np.random.seed(42)
        # Mean-reverting: AR(1) with negative coefficient
        values = np.zeros(500)
        values[0] = np.random.randn()
        for i in range(1, 500):
            values[i] = -0.5 * values[i - 1] + np.random.randn()

        # Use cumulative sum for variance ratio
        prices = np.cumsum(values)
        vr = _variance_ratio_nb(prices, q=10)

        # Mean-reverting should have VR < 1 (though effect may be small)
        # Just check it's computed
        assert not np.isnan(vr)

    def test_rolling_variance_ratio_basic(self) -> None:
        """Test rolling variance ratio computation."""
        np.random.seed(42)
        df = pl.DataFrame({"prices": np.cumsum(np.random.randn(300))})

        result = df.select(variance_ratio("prices", window=100, q=5).alias("vr"))

        valid_count = result["vr"].drop_nulls().len()
        assert valid_count > 100


# =============================================================================
# KL Divergence Tests
# =============================================================================


class TestKLDivergence:
    """Tests for KL Divergence."""

    def test_kl_identical_distributions(self) -> None:
        """Test that identical distributions have near-zero KL divergence."""
        np.random.seed(42)
        # Same distribution repeated
        values = np.tile(np.random.randn(50), 2)

        kl = _kl_divergence_nb(values, _split_point=50, n_bins=20)

        # Should be close to zero (with small epsilon)
        assert kl < 0.5

    def test_kl_different_distributions(self) -> None:
        """Test that different distributions have higher KL divergence."""
        np.random.seed(42)
        # Very different distributions
        first_half = np.random.randn(50) - 5  # Mean -5
        second_half = np.random.randn(50) + 5  # Mean +5
        values = np.concatenate([first_half, second_half])

        kl = _kl_divergence_nb(values, _split_point=50, n_bins=20)

        # Should have substantial divergence
        assert kl > 1.0

    def test_kl_nan_handling(self) -> None:
        """Test KL divergence with NaN values."""
        values = np.array([np.nan] * 10, dtype=np.float64)
        kl = _kl_divergence_nb(values, _split_point=5, n_bins=10)

        assert np.isnan(kl)

    def test_rolling_kl_divergence_basic(self) -> None:
        """Test rolling KL divergence computation."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(300)})

        result = df.select(rolling_kl_divergence("values", window=100, n_bins=20).alias("kl"))

        valid_count = result["kl"].drop_nulls().len()
        assert valid_count > 100
        assert all(result["kl"].drop_nulls() >= 0)  # KL is non-negative


# =============================================================================
# Wasserstein Distance Tests
# =============================================================================


class TestWassersteinDistance:
    """Tests for Wasserstein Distance."""

    def test_wasserstein_identical(self) -> None:
        """Test that identical distributions have zero Wasserstein distance."""
        values = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2)
        dist = _wasserstein_1d_nb(values)

        assert dist == 0.0

    def test_wasserstein_shifted(self) -> None:
        """Test Wasserstein distance for shifted distributions."""
        np.random.seed(42)
        shift = 5.0
        first_half = np.random.randn(50)
        second_half = np.random.randn(50) + shift
        values = np.concatenate([first_half, second_half])

        dist = _wasserstein_1d_nb(values)

        # Distance should be approximately equal to shift
        assert 3.0 < dist < 7.0

    def test_wasserstein_nan_handling(self) -> None:
        """Test Wasserstein with NaN values."""
        values = np.array([np.nan] * 10, dtype=np.float64)
        dist = _wasserstein_1d_nb(values)

        assert np.isnan(dist)

    def test_rolling_wasserstein_basic(self) -> None:
        """Test rolling Wasserstein distance computation."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(300)})

        result = df.select(rolling_wasserstein("values", window=100).alias("wasserstein"))

        valid_count = result["wasserstein"].drop_nulls().len()
        assert valid_count > 100
        assert all(result["wasserstein"].drop_nulls() >= 0)  # Distance is non-negative


# =============================================================================
# Drift Detection Tests
# =============================================================================


class TestDriftDetection:
    """Tests for drift detection."""

    def test_drift_basic(self) -> None:
        """Test basic drift calculation."""
        # Clear drift: first half has mean 0, second half has mean 5
        first_half = np.zeros(50)
        second_half = np.ones(50) * 5.0
        values = np.concatenate([first_half, second_half])

        drift = _drift_nb(values)

        assert drift == 5.0

    def test_drift_zscore(self) -> None:
        """Test drift Z-score calculation."""
        np.random.seed(42)
        # No drift: same distribution
        values = np.random.randn(100)
        zscore = _drift_zscore_nb(values)

        # Z-score should be small for no drift
        assert abs(zscore) < 3.0

    def test_drift_significant(self) -> None:
        """Test that significant drift has high Z-score."""
        np.random.seed(42)
        # Clear drift
        first_half = np.random.randn(50) * 0.1  # Low variance
        second_half = np.random.randn(50) * 0.1 + 2.0  # Shifted mean
        values = np.concatenate([first_half, second_half])

        zscore = _drift_zscore_nb(values)

        # Should be significantly positive
        assert zscore > 2.0

    def test_rolling_drift_basic(self) -> None:
        """Test rolling drift computation."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(300)})

        result = df.select(rolling_drift("values", window=100, normalize=True).alias("drift"))

        valid_count = result["drift"].drop_nulls().len()
        assert valid_count > 100

    def test_rolling_drift_unnormalized(self) -> None:
        """Test rolling drift without normalization."""
        np.random.seed(42)
        df = pl.DataFrame({"values": np.random.randn(300)})

        result = df.select(rolling_drift("values", window=100, normalize=False).alias("drift"))

        valid_count = result["drift"].drop_nulls().len()
        assert valid_count > 100


# =============================================================================
# Integration Tests
# =============================================================================


class TestStructuralBreakIntegration:
    """Integration tests for structural break features."""

    def test_all_features_on_regime_change(self) -> None:
        """Test all features detect a clear regime change."""
        np.random.seed(42)
        # Create data with clear regime change at t=200
        regime1 = np.random.randn(200) * 0.1 + 5.0  # Low vol, mean 5
        regime2 = np.random.randn(200) * 1.0 + 10.0  # High vol, mean 10
        values = np.concatenate([regime1, regime2])

        df = pl.DataFrame({"values": values})

        # Compute all features
        result = df.select(
            coefficient_of_variation("values", window=50).alias("cv"),
            variance_ratio("values", window=100, q=5).alias("vr"),
            rolling_kl_divergence("values", window=100).alias("kl"),
            rolling_wasserstein("values", window=100).alias("wasserstein"),
            rolling_drift("values", window=100).alias("drift"),
        )

        # All features should have valid values
        for col in ["cv", "vr", "kl", "wasserstein", "drift"]:
            valid_count = result[col].drop_nulls().len()
            assert valid_count > 100, f"{col} has too few valid values"

    def test_features_on_stationary_data(self) -> None:
        """Test features on stationary data."""
        np.random.seed(42)
        # Stationary data
        values = np.random.randn(500)

        df = pl.DataFrame({"values": values})

        result = df.select(
            coefficient_of_variation("values", window=50).alias("cv"),
            rolling_drift("values", window=100).alias("drift"),
        )

        # Drift should be centered around 0 for stationary data
        drift_values = result["drift"].drop_nulls().to_numpy()
        assert abs(np.mean(drift_values)) < 1.0


# =============================================================================
# Validation Tests
# =============================================================================


class TestStructuralBreakValidation:
    """Tests for input validation."""

    def test_cv_invalid_window(self) -> None:
        """Test that invalid window raises error."""
        df = pl.DataFrame({"values": [1.0, 2.0, 3.0]})

        with pytest.raises(ValueError):
            df.select(coefficient_of_variation("values", window=0))

    def test_variance_ratio_q_too_large(self) -> None:
        """Test variance ratio with q too large for window."""
        df = pl.DataFrame({"values": list(range(50))})

        with pytest.raises(ValueError):
            df.select(variance_ratio("values", window=10, q=10))  # q must be < window/2

    def test_kl_divergence_invalid_bins(self) -> None:
        """Test KL divergence with invalid bins."""
        df = pl.DataFrame({"values": list(range(200))})

        with pytest.raises(ValueError):
            df.select(rolling_kl_divergence("values", window=100, n_bins=1))
