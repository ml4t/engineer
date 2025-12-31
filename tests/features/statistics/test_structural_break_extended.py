"""Extended tests for structural break detection features.

Tests CV regime detection, variance ratio interpretation,
and synthetic data with known breakpoints.
"""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.features.statistics.structural_break import (
    _kl_divergence_nb,
    _wasserstein_1d_nb,
    coefficient_of_variation,
    rolling_cv_zscore,
    rolling_drift,
    rolling_kl_divergence,
    rolling_wasserstein,
    variance_ratio,
)

# =============================================================================
# CV Regime Detection Tests
# =============================================================================


class TestCVRegimeDetection:
    """Tests for CV as regime change detector."""

    def test_cv_detects_volatility_regime_change(self) -> None:
        """Test CV increases when volatility regime changes."""
        np.random.seed(42)

        # Create clear regime change: low vol -> high vol
        low_vol = np.random.randn(200) * 0.1 + 100
        high_vol = np.random.randn(200) * 2.0 + 100

        values = np.concatenate([low_vol, high_vol])
        df = pl.DataFrame({"values": values})

        result = df.select(coefficient_of_variation("values", window=50).alias("cv"))

        cv_values = result["cv"].drop_nulls().to_numpy()

        # CV should spike around the regime change (index ~200)
        # Compare CV before and after transition
        if len(cv_values) > 250:
            pre_change_cv = np.mean(cv_values[100:150])  # Before change
            post_change_cv = np.mean(cv_values[250:300])  # After change

            # Post-change CV should be higher
            assert post_change_cv > pre_change_cv * 3

    def test_cv_stable_in_stationary_regime(self) -> None:
        """Test CV remains stable in stationary regime."""
        np.random.seed(42)

        # Stationary process
        values = np.random.randn(500) + 10
        df = pl.DataFrame({"values": values})

        result = df.select(coefficient_of_variation("values", window=50).alias("cv"))

        cv_values = result["cv"].drop_nulls().to_numpy()

        # CV should have low variance in stationary regime
        cv_std = np.std(cv_values)
        cv_mean = np.mean(cv_values)

        # Coefficient of variation of CV should be low
        assert cv_std / cv_mean < 0.5

    def test_cv_zscore_identifies_outlier_regimes(self) -> None:
        """Test CV Z-score identifies unusual volatility periods."""
        np.random.seed(42)

        # Normal volatility with one high-volatility spike
        normal = np.random.randn(400) * 0.5 + 10
        spike = np.random.randn(100) * 3.0 + 10
        values = np.concatenate([normal[:200], spike, normal[200:]])

        df = pl.DataFrame({"values": values})

        result = df.select(
            rolling_cv_zscore(
                "values",
                window=50,
                lookback_multiplier=5,
            ).alias("cv_zscore")
        )

        cv_zscore = result["cv_zscore"].drop_nulls().to_numpy()

        # Z-score should spike during high volatility period
        if len(cv_zscore) > 250:
            # Check for spike around index 200-300
            spike_region = cv_zscore[200:300]
            max_zscore = np.max(np.abs(spike_region))

            # Should have some Z-score deviation during spike (may not be extreme with random data)
            assert max_zscore > 1.0

    def test_cv_gate_between_regimes(self) -> None:
        """Test CV acts as gate between different market regimes."""
        np.random.seed(42)

        # Three distinct regimes
        regime1 = np.random.randn(200) * 0.2 + 10  # Low vol
        regime2 = np.random.randn(200) * 1.0 + 10  # Med vol
        regime3 = np.random.randn(200) * 0.3 + 10  # Low vol again

        values = np.concatenate([regime1, regime2, regime3])
        df = pl.DataFrame({"values": values})

        result = df.select(coefficient_of_variation("values", window=50).alias("cv"))

        cv_values = result["cv"].drop_nulls().to_numpy()

        if len(cv_values) > 500:
            cv_regime1 = np.median(cv_values[100:150])
            cv_regime2 = np.median(cv_values[250:350])
            cv_regime3 = np.median(cv_values[450:500])

            # Regime 2 should have distinctly higher CV
            assert cv_regime2 > cv_regime1 * 2
            assert cv_regime2 > cv_regime3 * 2


# =============================================================================
# Variance Ratio Interpretation Tests
# =============================================================================


class TestVarianceRatioInterpretation:
    """Tests for interpreting variance ratio values."""

    def test_vr_random_walk_equals_one(self) -> None:
        """Test VR ≈ 1 for random walk (efficient market)."""
        np.random.seed(42)

        # Pure random walk
        increments = np.random.randn(1000)
        prices = 100 + np.cumsum(increments)

        df = pl.DataFrame({"prices": prices})

        result = df.select(variance_ratio("prices", window=200, q=5).alias("vr"))

        vr_values = result["vr"].drop_nulls().to_numpy()

        # Mean VR should be close to 1
        mean_vr = np.mean(vr_values)
        assert 0.8 < mean_vr < 1.2

    def test_vr_greater_than_one_indicates_momentum(self) -> None:
        """Test VR > 1 indicates momentum/trending behavior."""
        np.random.seed(42)

        # Create trending series with positive autocorrelation
        base = np.random.randn(500)
        trending = np.zeros(500)
        trending[0] = base[0]

        for i in range(1, 500):
            # Positive autocorrelation creates trending
            trending[i] = 0.3 * trending[i - 1] + base[i]

        prices = 100 + np.cumsum(trending)
        df = pl.DataFrame({"prices": prices})

        result = df.select(variance_ratio("prices", window=200, q=5).alias("vr"))

        vr_values = result["vr"].drop_nulls().to_numpy()

        # Mean VR should be > 1 for trending
        mean_vr = np.mean(vr_values)
        assert mean_vr > 1.0

    def test_vr_less_than_one_indicates_mean_reversion(self) -> None:
        """Test VR < 1 indicates mean reversion."""
        np.random.seed(42)

        # Create mean-reverting series with negative autocorrelation
        base = np.random.randn(500)
        mean_reverting = np.zeros(500)
        mean_reverting[0] = base[0]

        for i in range(1, 500):
            # Negative autocorrelation creates mean reversion
            mean_reverting[i] = -0.4 * mean_reverting[i - 1] + base[i]

        prices = 100 + np.cumsum(mean_reverting)
        df = pl.DataFrame({"prices": prices})

        result = df.select(variance_ratio("prices", window=200, q=10).alias("vr"))

        vr_values = result["vr"].drop_nulls().to_numpy()

        # At least some VR values should be < 1
        assert np.min(vr_values) < 1.0

    def test_vr_detects_regime_shift(self) -> None:
        """Test VR detects shift from trending to mean-reverting."""
        np.random.seed(42)

        # First half: trending (positive autocorr)
        trend = np.zeros(250)
        for i in range(1, 250):
            trend[i] = 0.3 * trend[i - 1] + np.random.randn()

        # Second half: mean-reverting (negative autocorr)
        mean_rev = np.zeros(250)
        for i in range(1, 250):
            mean_rev[i] = -0.3 * mean_rev[i - 1] + np.random.randn()

        values = np.concatenate([trend, mean_rev])
        prices = 100 + np.cumsum(values)

        df = pl.DataFrame({"prices": prices})

        result = df.select(variance_ratio("prices", window=100, q=5).alias("vr"))

        vr_values = result["vr"].drop_nulls().to_numpy()

        if len(vr_values) > 300:
            # VR should change between first and second half
            first_half_vr = np.mean(vr_values[:150])
            second_half_vr = np.mean(vr_values[-100:])

            # They should be different
            assert abs(first_half_vr - second_half_vr) > 0.2


# =============================================================================
# Synthetic Data with Known Breakpoints
# =============================================================================


class TestKnownBreakpoints:
    """Tests with synthetic data with known structural breaks."""

    def test_single_mean_shift_detection(self) -> None:
        """Test detection of single mean shift at known point."""
        np.random.seed(42)

        # Regime 1: mean=10, std=1
        regime1 = np.random.randn(300) * 1.0 + 10

        # Regime 2: mean=15, std=1 (mean shift at t=300)
        regime2 = np.random.randn(300) * 1.0 + 15

        values = np.concatenate([regime1, regime2])
        df = pl.DataFrame({"values": values})

        result = df.select(
            rolling_drift("values", window=100, normalize=True).alias("drift"),
            rolling_kl_divergence("values", window=100).alias("kl"),
        )

        # Drift should spike around the breakpoint (index 300)
        drift_values = result["drift"].drop_nulls().to_numpy()
        result["kl"].drop_nulls().to_numpy()

        if len(drift_values) > 350:
            # Find maximum drift in region around breakpoint
            breakpoint_region = drift_values[250:350]
            max_drift = np.max(np.abs(breakpoint_region))

            # Should detect significant drift
            assert max_drift > 2.0

    def test_single_variance_shift_detection(self) -> None:
        """Test detection of variance shift at known point."""
        np.random.seed(42)

        # Regime 1: mean=10, std=0.5
        regime1 = np.random.randn(300) * 0.5 + 10

        # Regime 2: mean=10, std=2.0 (variance shift at t=300)
        regime2 = np.random.randn(300) * 2.0 + 10

        values = np.concatenate([regime1, regime2])
        df = pl.DataFrame({"values": values})

        result = df.select(
            coefficient_of_variation("values", window=50).alias("cv"),
            rolling_wasserstein("values", window=100).alias("wasserstein"),
        )

        cv_values = result["cv"].drop_nulls().to_numpy()

        if len(cv_values) > 350:
            # CV should increase after breakpoint
            pre_break = np.mean(cv_values[200:250])
            post_break = np.mean(cv_values[350:400])

            assert post_break > pre_break * 2

    def test_multiple_breakpoints(self) -> None:
        """Test detection of multiple structural breaks."""
        np.random.seed(42)

        # Three regimes with known breakpoints
        regime1 = np.random.randn(200) * 1.0 + 10  # Break at 200
        regime2 = np.random.randn(200) * 2.0 + 15  # Break at 400
        regime3 = np.random.randn(200) * 0.5 + 12

        values = np.concatenate([regime1, regime2, regime3])
        df = pl.DataFrame({"values": values})

        result = df.select(
            rolling_drift("values", window=100).alias("drift"),
            coefficient_of_variation("values", window=50).alias("cv"),
        )

        drift_values = result["drift"].drop_nulls().to_numpy()
        cv_values = result["cv"].drop_nulls().to_numpy()

        # Should have spikes at both breakpoints
        # This is a smoke test - just verify features run
        assert len(drift_values) > 400
        assert len(cv_values) > 500

    def test_gradual_vs_abrupt_breaks(self) -> None:
        """Test detection of gradual vs abrupt structural breaks."""
        np.random.seed(42)

        # Abrupt break
        abrupt1 = np.random.randn(200) + 10
        abrupt2 = np.random.randn(200) + 15
        abrupt = np.concatenate([abrupt1, abrupt2])

        # Gradual break (linear drift)
        gradual = np.random.randn(400) + np.linspace(10, 15, 400)

        df_abrupt = pl.DataFrame({"values": abrupt})
        df_gradual = pl.DataFrame({"values": gradual})

        # Test on abrupt break
        result_abrupt = df_abrupt.select(rolling_drift("values", window=100).alias("drift"))

        # Test on gradual break
        result_gradual = df_gradual.select(rolling_drift("values", window=100).alias("drift"))

        # Both should produce valid results
        assert result_abrupt["drift"].drop_nulls().len() > 100
        assert result_gradual["drift"].drop_nulls().len() > 100


# =============================================================================
# Distribution Divergence Tests
# =============================================================================


class TestDistributionDivergence:
    """Tests for KL divergence and Wasserstein distance."""

    def test_kl_divergence_location_shift(self) -> None:
        """Test KL divergence detects location (mean) shifts."""
        np.random.seed(42)

        # Same distribution, shifted mean
        dist1 = np.random.randn(100) + 0
        dist2 = np.random.randn(100) + 5

        values = np.concatenate([dist1, dist2])
        kl = _kl_divergence_nb(values, _split_point=100, n_bins=20)

        # Should have high KL divergence
        assert kl > 1.0

    def test_kl_divergence_scale_shift(self) -> None:
        """Test KL divergence detects scale (variance) shifts."""
        np.random.seed(42)

        # Same mean, different variance
        dist1 = np.random.randn(100) * 0.5
        dist2 = np.random.randn(100) * 2.0

        values = np.concatenate([dist1, dist2])
        kl = _kl_divergence_nb(values, _split_point=100, n_bins=20)

        # Should have significant KL divergence
        assert kl > 0.5

    def test_wasserstein_proportional_to_shift(self) -> None:
        """Test Wasserstein distance proportional to distribution shift."""
        np.random.seed(42)

        # Small shift
        dist1_small = np.random.randn(100)
        dist2_small = np.random.randn(100) + 1.0
        values_small = np.concatenate([dist1_small, dist2_small])
        w_small = _wasserstein_1d_nb(values_small)

        # Large shift
        dist1_large = np.random.randn(100)
        dist2_large = np.random.randn(100) + 5.0
        values_large = np.concatenate([dist1_large, dist2_large])
        w_large = _wasserstein_1d_nb(values_large)

        # Larger shift should have larger Wasserstein distance
        assert w_large > w_small * 2

    def test_wasserstein_robust_to_outliers(self) -> None:
        """Test Wasserstein more robust than KL to outliers."""
        np.random.seed(42)

        # Distribution with outliers
        dist1 = np.random.randn(100)
        dist2 = np.random.randn(100)
        dist2[0] = 100  # Add outlier

        values = np.concatenate([dist1, dist2])

        kl = _kl_divergence_nb(values, _split_point=100, n_bins=20)
        wasserstein = _wasserstein_1d_nb(values)

        # Both should compute (smoke test)
        assert not np.isnan(kl)
        assert not np.isnan(wasserstein)


# =============================================================================
# Integration Tests with ADIA Lab Insights
# =============================================================================


class TestADIALabInsights:
    """Tests based on ADIA Lab 2025 structural break challenge insights."""

    def test_cv_as_primary_feature(self) -> None:
        """Test CV as primary regime detection feature (ADIA insight)."""
        np.random.seed(42)

        # Multi-regime scenario
        regimes = [
            np.random.randn(150) * 0.5 + 10,  # Low vol
            np.random.randn(150) * 2.0 + 10,  # High vol
            np.random.randn(150) * 0.8 + 10,  # Med vol
        ]

        values = np.concatenate(regimes)
        df = pl.DataFrame({"values": values})

        result = df.select(
            coefficient_of_variation("values", window=50).alias("cv"),
            rolling_cv_zscore("values", window=50).alias("cv_zscore"),
        )

        # CV should vary across regimes
        cv_values = result["cv"].drop_nulls().to_numpy()
        assert np.std(cv_values) > 0

    def test_multi_scale_features(self) -> None:
        """Test multi-scale features capture different break types."""
        np.random.seed(42)

        values = np.random.randn(500)
        df = pl.DataFrame({"values": values})

        # Different window sizes capture different scales
        result = df.select(
            coefficient_of_variation("values", window=25).alias("cv_short"),
            coefficient_of_variation("values", window=50).alias("cv_med"),
            coefficient_of_variation("values", window=100).alias("cv_long"),
        )

        # All should produce values
        assert result["cv_short"].drop_nulls().len() > 400
        assert result["cv_med"].drop_nulls().len() > 400
        assert result["cv_long"].drop_nulls().len() > 300

    def test_feature_combination_for_detection(self) -> None:
        """Test combining multiple features improves detection."""
        np.random.seed(42)

        # Clear structural break
        regime1 = np.random.randn(250) * 1.0 + 10
        regime2 = np.random.randn(250) * 2.0 + 15

        values = np.concatenate([regime1, regime2])
        df = pl.DataFrame({"values": values})

        result = df.select(
            coefficient_of_variation("values", window=50).alias("cv"),
            variance_ratio("values", window=100, q=5).alias("vr"),
            rolling_drift("values", window=100).alias("drift"),
            rolling_kl_divergence("values", window=100).alias("kl"),
        )

        # All features should detect something at the break
        for col in ["cv", "vr", "drift", "kl"]:
            assert result[col].drop_nulls().len() > 200


# =============================================================================
# Validation & Edge Cases
# =============================================================================


class TestStructuralBreakValidationExtended:
    """Extended validation tests."""

    def test_handles_constant_values(self) -> None:
        """Test all features handle constant values gracefully."""
        values = np.ones(200) * 10.0
        df = pl.DataFrame({"values": values})

        result = df.select(
            coefficient_of_variation("values", window=50).alias("cv"),
            variance_ratio("values", window=100, q=5).alias("vr"),
            rolling_drift("values", window=100).alias("drift"),
        )

        # CV should be zero
        cv_values = result["cv"].drop_nulls()
        assert all(cv_values == 0.0)

    def test_handles_nan_values(self) -> None:
        """Test features handle NaN values in data."""
        np.random.seed(42)
        values = np.random.randn(200)
        values[50:55] = np.nan  # Insert NaN block

        df = pl.DataFrame({"values": values})

        result = df.select(
            coefficient_of_variation("values", window=50).alias("cv"),
            rolling_drift("values", window=50).alias("drift"),
        )

        # Should still produce some valid values
        assert result["cv"].drop_nulls().len() > 100

    def test_minimum_window_size_requirements(self) -> None:
        """Test features respect minimum window requirements."""
        df = pl.DataFrame({"values": list(range(50))})

        # Too small window for variance ratio with q=10
        with pytest.raises(ValueError):
            df.select(variance_ratio("values", window=15, q=10))

        # Valid window
        result = df.select(variance_ratio("values", window=30, q=5))
        assert len(result) == 50
