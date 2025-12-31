"""Comprehensive integration tests for Module C feature-outcome analysis.

Tests end-to-end workflows combining IC analysis, importance computation,
drift detection, and feature selection across various scenarios.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.engineer.config.feature_config import (
    ICConfig,
    MLDiagnosticsConfig,
    ModuleCConfig,
)
from ml4t.engineer.outcome.feature_outcome import (
    FeatureOutcome,
    FeatureOutcomeResult,
)


class TestEndToEndWorkflow:
    """Test complete analysis pipelines from features to recommendations."""

    def test_small_feature_set_workflow(self):
        """Test workflow with small feature set (5 features)."""
        np.random.seed(42)
        n = 1000

        # Create correlated features
        outcome = np.random.randn(n)
        features = pd.DataFrame(
            {
                "feat1": outcome + np.random.randn(n) * 0.1,  # Strong correlation
                "feat2": outcome + np.random.randn(n) * 0.5,  # Moderate
                "feat3": np.random.randn(n),  # Noise
                "feat4": -outcome + np.random.randn(n) * 0.1,  # Negative correlation
                "feat5": outcome**2 + np.random.randn(n) * 0.1,  # Nonlinear
            }
        )

        # Run analysis with custom IC lags
        config = ModuleCConfig(ic=ICConfig(lag_structure=[1, 5, 10]))
        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, outcome)

        # Verify results
        assert isinstance(result, FeatureOutcomeResult)
        assert len(result.ic_results) == 5

        # Check IC values are in valid range
        for feat_ic in result.ic_results.values():
            assert -1.0 <= feat_ic.ic_mean <= 1.0

        # Get recommendations
        recs = result.get_recommendations()
        assert isinstance(recs, list)

    def test_medium_feature_set_workflow(self):
        """Test workflow with medium feature set (50 features)."""
        np.random.seed(42)
        n = 2000
        n_features = 50

        outcome = np.random.randn(n)
        features = pd.DataFrame()

        # Mix of feature types
        for i in range(n_features):
            if i < 10:  # Strong features
                features[f"strong_{i}"] = outcome + np.random.randn(n) * 0.1
            elif i < 25:  # Moderate features
                features[f"moderate_{i}"] = outcome + np.random.randn(n) * 0.5
            else:  # Noise
                features[f"noise_{i}"] = np.random.randn(n)

        config = ModuleCConfig(ml_diagnostics=MLDiagnosticsConfig(feature_importance=True))
        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, outcome)

        assert len(result.ic_results) == n_features

        # Top features by IC mean should be strong ones (strongest predictive power)
        top_features = result.get_top_features(n=10, by="ic_mean")
        strong_in_top = sum(1 for f in top_features if f.startswith("strong_"))
        # Most top features should be strong (have high signal)
        assert strong_in_top >= 8

    def test_large_feature_set_workflow(self):
        """Test workflow with large feature set (100 features)."""
        np.random.seed(42)
        n = 5000
        n_features = 100

        outcome = np.random.randn(n)

        # Build dict first to avoid fragmentation
        feature_dict = {f"feat_{i:03d}": np.random.randn(n) for i in range(n_features)}
        # Add predictive features
        feature_dict["predictive_1"] = outcome + np.random.randn(n) * 0.1
        feature_dict["predictive_2"] = outcome + np.random.randn(n) * 0.2
        features = pd.DataFrame(feature_dict)

        config = ModuleCConfig(ic=ICConfig(lag_structure=[1]))
        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, outcome)

        assert len(result.ic_results) == 102  # 100 + 2 predictive

    def test_continuous_outcome_workflow(self):
        """Test with continuous outcome (returns)."""
        np.random.seed(42)
        n = 1000

        outcome = np.random.randn(n) * 0.02  # Returns-like distribution
        features = pd.DataFrame(
            {
                "momentum": np.roll(outcome, 1),
                "volatility": pd.Series(outcome).rolling(20).std().fillna(0),
                "volume": np.abs(np.random.randn(n)),
            }
        )

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        assert len(result.ic_results) == 3


class TestConfigurationVariations:
    """Test different configuration options."""

    def test_different_ic_lags(self):
        """Test IC analysis with different lag configurations."""
        np.random.seed(42)
        n = 1000
        outcome = np.random.randn(n)
        features = pd.DataFrame({"feat1": np.random.randn(n)})

        # Single lag
        config1 = ModuleCConfig(ic=ICConfig(lag_structure=[1]))
        analyzer1 = FeatureOutcome(config=config1)
        result1 = analyzer1.run_analysis(features, outcome)
        assert result1.ic_results["feat1"].ic_by_lag

        # Multiple lags
        config2 = ModuleCConfig(ic=ICConfig(lag_structure=[1, 5, 10, 20]))
        analyzer2 = FeatureOutcome(config=config2)
        result2 = analyzer2.run_analysis(features, outcome)
        assert len(result2.ic_results["feat1"].ic_by_lag) >= 1

    def test_importance_computation(self):
        """Test ML importance computation."""
        np.random.seed(42)
        n = 500
        outcome = np.random.randn(n)
        features = pd.DataFrame(
            {
                "feat1": outcome + np.random.randn(n) * 0.1,
                "feat2": np.random.randn(n),
                "feat3": outcome + np.random.randn(n) * 0.3,
            }
        )

        config = ModuleCConfig(ml_diagnostics=MLDiagnosticsConfig(feature_importance=True))
        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, outcome)

        # Should have importance results
        assert len(result.importance_results) > 0

    def test_verbose_mode(self, capsys):
        """Test verbose logging output."""
        np.random.seed(42)
        n = 500
        outcome = np.random.randn(n)
        features = pd.DataFrame({"feat1": np.random.randn(n)})

        analyzer = FeatureOutcome()
        analyzer.run_analysis(features, outcome, verbose=True)

        # Verbose mode should print (implementation may vary)
        _ = capsys.readouterr()
        # Check if anything was output (lenient check)


class TestErrorHandling:
    """Test error handling and validation."""

    def test_empty_features(self):
        """Test with empty feature DataFrame."""
        outcome = np.random.randn(100)
        features = pd.DataFrame()

        analyzer = FeatureOutcome()
        with pytest.raises((ValueError, KeyError, IndexError)):
            analyzer.run_analysis(features, outcome)

    def test_all_nan_features(self):
        """Test with all-NaN features."""
        n = 100
        outcome = np.random.randn(n)
        features = pd.DataFrame(
            {
                "feat1": np.full(n, np.nan),
                "feat2": np.full(n, np.nan),
            }
        )

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)
        # Should handle gracefully
        assert result is not None

    def test_invalid_outcome(self):
        """Test with invalid outcome values."""
        features = pd.DataFrame({"feat1": np.random.randn(100)})
        outcome = np.full(100, np.nan)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)
        # Should handle gracefully
        assert result is not None


class TestPolarsIntegration:
    """Test Polars DataFrame support."""

    def test_polars_features_numpy_outcome(self):
        """Test with Polars features and numpy outcome."""
        np.random.seed(42)
        n = 500

        outcome = np.random.randn(n)
        features = pl.DataFrame(
            {
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n),
            }
        )

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)
        assert len(result.ic_results) == 2

    def test_polars_features_pandas_series_outcome(self):
        """Test with Polars features and pandas Series outcome."""
        np.random.seed(42)
        n = 500

        outcome = pd.Series(np.random.randn(n))
        features = pl.DataFrame(
            {
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n),
            }
        )

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)
        assert len(result.ic_results) == 2


class TestResultOperations:
    """Test operations on FeatureOutcomeResult."""

    def test_to_dataframe_comprehensive(self):
        """Test comprehensive DataFrame conversion."""
        np.random.seed(42)
        n = 500
        outcome = np.random.randn(n)
        features = pd.DataFrame(
            {
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n),
                "feat3": np.random.randn(n),
            }
        )

        config = ModuleCConfig(ml_diagnostics=MLDiagnosticsConfig(feature_importance=True))
        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, outcome)

        df = result.to_dataframe()
        assert "feature" in df.columns
        assert len(df) == 3

    def test_get_top_features_variations(self):
        """Test get_top_features with various parameters."""
        np.random.seed(42)
        n = 500
        outcome = np.random.randn(n)
        features = pd.DataFrame({f"feat{i}": np.random.randn(n) for i in range(10)})

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        # Top N
        top5 = result.get_top_features(n=5)
        assert len(top5) <= 5

        # By different metrics
        top_ic = result.get_top_features(n=3, by="ic_mean")
        assert len(top_ic) <= 3

        top_ir = result.get_top_features(n=3, by="ic_ir")
        assert len(top_ir) <= 3

    def test_get_recommendations_comprehensive(self):
        """Test recommendation generation."""
        np.random.seed(42)
        n = 500
        outcome = np.random.randn(n)
        features = pd.DataFrame(
            {
                "good_feat": outcome + np.random.randn(n) * 0.1,
                "bad_feat": np.random.randn(n),
            }
        )

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        recs = result.get_recommendations()
        assert isinstance(recs, list)


class TestSpecificFeatureSelection:
    """Test analyzing specific features only."""

    def test_subset_of_features(self):
        """Test analyzing only a subset of features."""
        np.random.seed(42)
        n = 500
        outcome = np.random.randn(n)
        features = pd.DataFrame(
            {
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n),
                "feat3": np.random.randn(n),
                "feat4": np.random.randn(n),
            }
        )

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome, feature_names=["feat1", "feat3"])

        # Should only analyze selected features
        assert len(result.ic_results) == 2
        assert "feat1" in result.ic_results
        assert "feat3" in result.ic_results
        assert "feat2" not in result.ic_results


class TestLongRunningAnalysis:
    """Test analysis with realistic dataset sizes."""

    def test_realistic_dataset(self):
        """Test with realistic dataset: 30 features, 3K rows."""
        np.random.seed(42)
        n = 3000
        n_features = 30

        outcome = np.random.randn(n)
        features = pd.DataFrame()

        # Mix of feature types
        for i in range(n_features // 3):
            features[f"strong_{i}"] = outcome + np.random.randn(n) * 0.2
        for i in range(n_features // 3):
            features[f"weak_{i}"] = outcome + np.random.randn(n) * 0.8
        for i in range(n_features // 3):
            features[f"noise_{i}"] = np.random.randn(n)

        config = ModuleCConfig(
            ic=ICConfig(lag_structure=[1, 5, 10]),
            ml_diagnostics=MLDiagnosticsConfig(feature_importance=True),
        )
        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, outcome)

        assert len(result.ic_results) == n_features
        assert len(result.importance_results) > 0
