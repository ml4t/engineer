"""Tests for correlation matrix computation."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy import stats

from ml4t.engineer.relationships.correlation import compute_correlation_matrix


class TestCorrelationBasic:
    """Basic tests for correlation computation."""

    def test_pearson_perfect_positive(self):
        """Test perfect positive correlation."""
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})

        corr = compute_correlation_matrix(df, method="pearson")

        assert isinstance(corr, pl.DataFrame)
        assert "feature" in corr.columns
        assert "x" in corr.columns
        assert "y" in corr.columns

        # Convert to pandas for easier testing
        corr_pd = corr.to_pandas().set_index("feature")

        # Perfect positive correlation
        assert corr_pd.loc["x", "y"] == pytest.approx(1.0, abs=1e-10)
        assert corr_pd.loc["y", "x"] == pytest.approx(1.0, abs=1e-10)

        # Diagonal should be 1
        assert corr_pd.loc["x", "x"] == pytest.approx(1.0)
        assert corr_pd.loc["y", "y"] == pytest.approx(1.0)

    def test_pearson_perfect_negative(self):
        """Test perfect negative correlation."""
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 8, 6, 4, 2]})

        corr = compute_correlation_matrix(df, method="pearson")
        corr_pd = corr.to_pandas().set_index("feature")

        # Perfect negative correlation
        assert corr_pd.loc["x", "y"] == pytest.approx(-1.0, abs=1e-10)
        assert corr_pd.loc["y", "x"] == pytest.approx(-1.0, abs=1e-10)

    def test_pearson_no_correlation(self):
        """Test zero correlation."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "x": np.random.randn(100),
                "y": np.random.randn(100),  # Independent random variables
            }
        )

        corr = compute_correlation_matrix(df, method="pearson")
        corr_pd = corr.to_pandas().set_index("feature")

        # Should be close to 0 (not exact due to randomness)
        assert abs(corr_pd.loc["x", "y"]) < 0.2  # Loose bound for random data

    def test_spearman_monotonic(self):
        """Test Spearman correlation on monotonic non-linear relationship."""
        # Non-linear but monotonic relationship
        x = np.array([1, 2, 3, 4, 5])
        y = x**2  # Quadratic relationship
        df = pl.DataFrame({"x": x, "y": y})

        corr = compute_correlation_matrix(df, method="spearman")
        corr_pd = corr.to_pandas().set_index("feature")

        # Spearman should be perfect for monotonic relationship
        assert corr_pd.loc["x", "y"] == pytest.approx(1.0, abs=1e-10)

    def test_kendall_ordinal(self):
        """Test Kendall correlation."""
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 3, 2, 4, 5]})

        corr = compute_correlation_matrix(df, method="kendall")
        corr_pd = corr.to_pandas().set_index("feature")

        # Kendall should give reasonable correlation
        assert 0.5 < corr_pd.loc["x", "y"] < 1.0


class TestCorrelationMethods:
    """Test all three correlation methods."""

    def test_all_methods_available(self):
        """Test that all three methods work."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2, 4, 6, 8, 10],
                "c": [5, 4, 3, 2, 1],
            }
        )

        for method in ["pearson", "spearman", "kendall"]:
            corr = compute_correlation_matrix(df, method=method)
            assert isinstance(corr, pl.DataFrame)
            assert len(corr) == 3  # 3 features
            assert len(corr.columns) == 4  # feature + 3 correlations

    def test_invalid_method(self):
        """Test error on invalid method."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        with pytest.raises(ValueError, match="Invalid method"):
            compute_correlation_matrix(df, method="invalid")


class TestMissingDataHandling:
    """Test missing data handling."""

    def test_with_nans_pairwise_deletion(self):
        """Test correlation with missing values (pairwise deletion)."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [2.0, None, 6.0, 8.0, 10.0],  # One missing value
                "z": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        corr = compute_correlation_matrix(df, method="pearson")
        corr_pd = corr.to_pandas().set_index("feature")

        # Should compute correlation using available pairs
        assert not np.isnan(corr_pd.loc["x", "y"])
        assert not np.isnan(corr_pd.loc["x", "z"])

    def test_min_periods_constraint(self):
        """Test min_periods parameter."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y": [2.0, None, None, None, 10.0],  # Only 2 valid pairs
            }
        )

        # Should compute with 2 observations
        corr1 = compute_correlation_matrix(df, method="pearson", min_periods=2)
        corr1_pd = corr1.to_pandas().set_index("feature")
        assert not np.isnan(corr1_pd.loc["x", "y"])

        # Should return NaN with min_periods=3
        corr2 = compute_correlation_matrix(df, method="pearson", min_periods=3)
        corr2_pd = corr2.to_pandas().set_index("feature")
        assert np.isnan(corr2_pd.loc["x", "y"])

    def test_all_nans_column(self):
        """Test handling of all-NaN column."""
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [None, None, None],  # All NaN
                "z": [4.0, 5.0, 6.0],
            }
        )

        corr = compute_correlation_matrix(df, method="pearson")
        corr_pd = corr.to_pandas().set_index("feature")

        # Correlation with all-NaN column should be NaN
        assert np.isnan(corr_pd.loc["x", "y"])
        assert np.isnan(corr_pd.loc["y", "z"])


class TestFeatureSelection:
    """Test feature selection."""

    def test_specific_features(self):
        """Test correlation with specific features."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2, 4, 6, 8, 10],
                "c": [5, 4, 3, 2, 1],
                "d": [1, 1, 1, 1, 1],
            }
        )

        # Compute correlation for subset
        corr = compute_correlation_matrix(df, features=["a", "b"])
        corr_pd = corr.to_pandas().set_index("feature")

        assert len(corr_pd) == 2
        assert "a" in corr_pd.index
        assert "b" in corr_pd.index
        assert "c" not in corr_pd.index

    def test_missing_features_error(self):
        """Test error when specified features don't exist."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        with pytest.raises(ValueError, match="Features not found"):
            compute_correlation_matrix(df, features=["x", "z"])

    def test_non_numeric_feature_error(self):
        """Test error when specified feature is non-numeric."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": ["a", "b", "c"],  # String column
            }
        )

        with pytest.raises(ValueError, match="Non-numeric features"):
            compute_correlation_matrix(df, features=["x", "y"])

    def test_auto_select_numeric(self):
        """Test automatic selection of numeric columns."""
        df = pl.DataFrame(
            {
                "numeric1": [1, 2, 3],
                "text": ["a", "b", "c"],
                "numeric2": [4, 5, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        corr_pd = corr.to_pandas().set_index("feature")

        # Should only include numeric columns
        assert len(corr_pd) == 2
        assert "numeric1" in corr_pd.index
        assert "numeric2" in corr_pd.index
        assert "text" not in corr_pd.index


class TestInputFormats:
    """Test different input formats."""

    def test_pandas_dataframe_input(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})

        corr = compute_correlation_matrix(df, method="pearson")

        assert isinstance(corr, pl.DataFrame)
        corr_pd = corr.to_pandas().set_index("feature")
        assert corr_pd.loc["x", "y"] == pytest.approx(1.0, abs=1e-10)

    def test_polars_dataframe_input(self):
        """Test with Polars DataFrame input."""
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})

        corr = compute_correlation_matrix(df, method="pearson")

        assert isinstance(corr, pl.DataFrame)
        corr_pd = corr.to_pandas().set_index("feature")
        assert corr_pd.loc["x", "y"] == pytest.approx(1.0, abs=1e-10)


class TestRealWorldScenarios:
    """Test real-world scenarios."""

    def test_financial_data(self):
        """Test with financial returns data."""
        np.random.seed(42)
        n = 100

        # Simulate correlated returns
        market = np.random.randn(n)
        stock1 = 0.8 * market + 0.2 * np.random.randn(n)  # High correlation
        stock2 = 0.3 * market + 0.7 * np.random.randn(n)  # Low correlation

        df = pl.DataFrame(
            {
                "market": market,
                "stock1": stock1,
                "stock2": stock2,
            }
        )

        corr = compute_correlation_matrix(df, method="pearson")
        corr_pd = corr.to_pandas().set_index("feature")

        # Stock1 should be more correlated with market than stock2
        assert corr_pd.loc["market", "stock1"] > corr_pd.loc["market", "stock2"]
        assert corr_pd.loc["market", "stock1"] > 0.7  # Should be fairly high

    def test_multi_feature_matrix(self):
        """Test with larger feature set."""
        np.random.seed(42)
        n = 50

        df = pl.DataFrame({f"feature_{i}": np.random.randn(n) for i in range(10)})

        corr = compute_correlation_matrix(df, method="pearson")

        # Should have 10 features + 1 feature column
        assert len(corr) == 10
        assert len(corr.columns) == 11

        # All diagonal elements should be 1
        corr_pd = corr.to_pandas().set_index("feature")
        for feat in corr_pd.index:
            assert corr_pd.loc[feat, feat] == pytest.approx(1.0)

    def test_comparison_with_scipy(self):
        """Test that results match scipy.stats."""
        np.random.seed(42)
        x = np.random.randn(30)
        y = 0.5 * x + 0.5 * np.random.randn(30)

        df = pl.DataFrame({"x": x, "y": y})

        # Our implementation
        corr = compute_correlation_matrix(df, method="spearman")
        our_corr = corr.to_pandas().set_index("feature").loc["x", "y"]

        # Scipy reference
        scipy_corr, _ = stats.spearmanr(x, y)

        assert our_corr == pytest.approx(scipy_corr, abs=1e-10)


class TestEdgeCases:
    """Test edge cases."""

    def test_single_feature(self):
        """Test with single feature."""
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})

        corr = compute_correlation_matrix(df, method="pearson")
        corr_pd = corr.to_pandas().set_index("feature")

        assert len(corr_pd) == 1
        assert corr_pd.loc["x", "x"] == pytest.approx(1.0)

    def test_constant_feature(self):
        """Test with constant feature."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "const": [5, 5, 5, 5, 5],  # Constant
            }
        )

        corr = compute_correlation_matrix(df, method="pearson")
        corr_pd = corr.to_pandas().set_index("feature")

        # Correlation with constant should be NaN
        assert np.isnan(corr_pd.loc["x", "const"])

    def test_two_observations(self):
        """Test with only two observations."""
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})

        corr = compute_correlation_matrix(df, method="pearson")
        corr_pd = corr.to_pandas().set_index("feature")

        # With 2 points, correlation is always perfect (or undefined)
        assert corr_pd.loc["x", "y"] == pytest.approx(1.0, abs=1e-10)

    def test_no_numeric_columns(self):
        """Test error with no numeric columns."""
        df = pl.DataFrame({"text1": ["a", "b"], "text2": ["c", "d"]})

        with pytest.raises(ValueError, match="No numeric columns found"):
            compute_correlation_matrix(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
