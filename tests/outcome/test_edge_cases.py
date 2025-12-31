"""Edge case tests for Module C feature-outcome analysis.

Tests unusual, extreme, or boundary conditions using the actual API.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ml4t.engineer.outcome.drift import compute_psi, compute_wasserstein_distance
from ml4t.engineer.outcome.feature_outcome import FeatureOutcome


class TestEmptyAndMinimalData:
    """Test edge cases with empty or minimal data."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        features = pd.DataFrame()
        outcome = np.array([])

        analyzer = FeatureOutcome()

        # Should handle gracefully or raise error
        with pytest.raises((ValueError, KeyError, IndexError)):
            analyzer.run_analysis(features, outcome)

    def test_single_feature(self):
        """Test with single feature."""
        n = 100
        features = pd.DataFrame({"only_feat": np.random.randn(n)})
        outcome = np.random.randn(n)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        assert len(result.ic_results) == 1
        assert "only_feat" in result.ic_results


class TestNaNAndMissingData:
    """Test handling of NaN and missing values."""

    def test_all_nan_feature(self):
        """Test feature that is all NaN."""
        n = 100
        features = pd.DataFrame(
            {
                "all_nan": np.full(n, np.nan),
                "valid": np.random.randn(n),
            }
        )
        outcome = np.random.randn(n)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        # Should handle gracefully
        assert result.ic_results is not None

    def test_some_nan_features(self):
        """Test features with some NaN values."""
        n = 100
        features = pd.DataFrame(
            {
                "partial_nan": np.where(np.arange(n) % 5 == 0, np.nan, np.random.randn(n)),
                "valid": np.random.randn(n),
            }
        )
        outcome = np.random.randn(n)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        assert result.ic_results is not None


class TestConstantAndZeroVariance:
    """Test features with no variance."""

    def test_constant_feature(self):
        """Test feature with constant value."""
        n = 100
        features = pd.DataFrame(
            {
                "constant": np.full(n, 5.0),
                "variable": np.random.randn(n),
            }
        )
        outcome = np.random.randn(n)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        assert result.ic_results is not None


class TestPerfectCorrelation:
    """Test perfectly correlated features."""

    def test_perfect_positive_correlation(self):
        """Test feature perfectly correlated with outcome."""
        n = 100
        outcome = np.random.randn(n)
        features = pd.DataFrame(
            {
                "perfect": outcome,  # Exact copy
                "noise": np.random.randn(n),
            }
        )

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        # Perfect contemporaneous correlation should have IC near 1.0 at lag 0
        assert abs(result.ic_results["perfect"].ic_by_lag[0]) > 0.99

    def test_perfect_negative_correlation(self):
        """Test feature perfectly negatively correlated with outcome."""
        n = 100
        outcome = np.random.randn(n)
        features = pd.DataFrame(
            {
                "negative": -outcome,  # Negative copy
                "noise": np.random.randn(n),
            }
        )

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        # Perfect negative correlation should have IC near -1.0 at lag 0
        assert abs(result.ic_results["negative"].ic_by_lag[0]) > 0.99


class TestExtremeValues:
    """Test handling of extreme values."""

    def test_very_large_values(self):
        """Test with very large feature values."""
        n = 100
        features = pd.DataFrame(
            {
                "large": np.random.randn(n) * 1e10,
                "normal": np.random.randn(n),
            }
        )
        outcome = np.random.randn(n)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        assert result.ic_results is not None

    def test_very_small_values(self):
        """Test with very small feature values."""
        n = 100
        features = pd.DataFrame(
            {
                "small": np.random.randn(n) * 1e-10,
                "normal": np.random.randn(n),
            }
        )
        outcome = np.random.randn(n)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        assert result.ic_results is not None


class TestDriftEdgeCases:
    """Edge cases for drift detection functions."""

    def test_drift_identical_distributions(self):
        """Test drift detection with identical distributions."""
        n = 100
        baseline = np.random.randn(n)
        current = baseline.copy()

        result = compute_psi(baseline, current)

        assert result.psi < 0.01  # Should be near zero

    def test_drift_completely_different(self):
        """Test drift with completely different distributions."""
        n = 100
        baseline = np.random.randn(n)
        current = np.random.randn(n) * 10 + 100  # Very different

        result = compute_psi(baseline, current)

        assert result.psi > 0.5  # Should indicate drift

    def test_wasserstein_identical(self):
        """Test Wasserstein distance for identical distributions."""
        n = 100
        baseline = np.random.randn(n)
        current = baseline.copy()

        result = compute_wasserstein_distance(baseline, current)

        assert result.distance < 0.01  # Should be near zero


class TestResultOperations:
    """Edge cases for result operations."""

    def test_get_top_features_more_than_exist(self):
        """Test getting more top features than exist."""
        n = 100
        features = pd.DataFrame({"feat1": np.random.randn(n), "feat2": np.random.randn(n)})
        outcome = np.random.randn(n)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        # Request more features than available
        top = result.get_top_features(n=10)
        assert len(top) <= 2

    def test_get_top_features_with_zero(self):
        """Test getting zero top features."""
        n = 100
        features = pd.DataFrame({"feat1": np.random.randn(n)})
        outcome = np.random.randn(n)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        top = result.get_top_features(n=0)
        assert len(top) == 0


class TestDataTypeMixing:
    """Test mixing different data types."""

    def test_pandas_features_numpy_outcome(self):
        """Test pandas features with numpy outcome."""
        n = 100
        features = pd.DataFrame({"feat1": np.random.randn(n)})
        outcome = np.random.randn(n)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        assert result.ic_results is not None

    def test_pandas_features_series_outcome(self):
        """Test pandas features with pandas Series outcome."""
        n = 100
        features = pd.DataFrame({"feat1": np.random.randn(n)})
        outcome = pd.Series(np.random.randn(n))

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        assert result.ic_results is not None

    def test_polars_features_numpy_outcome(self):
        """Test Polars features with numpy outcome."""
        n = 100
        features = pl.DataFrame({"feat1": np.random.randn(n)})
        outcome = np.random.randn(n)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcome)

        assert result.ic_results is not None
