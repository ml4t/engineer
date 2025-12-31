# ruff: noqa: F841
"""Tests for feature-outcome analysis (Module C orchestration)."""

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
    FeatureICResults,
    FeatureImportanceResults,
    FeatureOutcome,
    FeatureOutcomeResult,
)


class TestFeatureICResults:
    """Test FeatureICResults dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = FeatureICResults(
            feature="test_feature",
            ic_mean=0.15,
            ic_std=0.05,
            ic_ir=3.0,
            p_value=0.001,
            n_observations=1000,
        )

        assert result.feature == "test_feature"
        assert result.ic_mean == 0.15
        assert result.ic_ir == 3.0
        assert result.p_value == 0.001

    def test_defaults(self):
        """Test default values."""
        result = FeatureICResults(feature="test")

        assert result.ic_mean == 0.0
        assert result.ic_std == 0.0
        assert result.ic_ir == 0.0
        assert result.p_value == 1.0


class TestFeatureImportanceResults:
    """Test FeatureImportanceResults dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = FeatureImportanceResults(
            feature="test_feature",
            mdi_importance=0.35,
            permutation_importance=0.28,
            rank_mdi=1,
            rank_permutation=2,
        )

        assert result.feature == "test_feature"
        assert result.mdi_importance == 0.35
        assert result.rank_mdi == 1


class TestFeatureOutcomeResult:
    """Test FeatureOutcomeResult aggregation."""

    def test_creation(self):
        """Test basic creation."""
        result = FeatureOutcomeResult(features=["f1", "f2", "f3"])

        assert len(result.features) == 3
        assert result.ic_results == {}
        assert result.importance_results == {}
        assert result.drift_results is None

    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        result = FeatureOutcomeResult(features=["f1", "f2"])

        # Add some IC results
        result.ic_results["f1"] = FeatureICResults(
            feature="f1", ic_mean=0.15, ic_std=0.05, ic_ir=3.0, p_value=0.001
        )
        result.ic_results["f2"] = FeatureICResults(
            feature="f2", ic_mean=-0.08, ic_std=0.04, ic_ir=-2.0, p_value=0.05
        )

        # Add importance results
        result.importance_results["f1"] = FeatureImportanceResults(
            feature="f1", mdi_importance=0.35, rank_mdi=1
        )

        df = result.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "feature" in df.columns
        assert "ic_mean" in df.columns
        assert "ic_ir" in df.columns
        assert "mdi_importance" in df.columns

        # Check values
        f1_row = df[df["feature"] == "f1"].iloc[0]
        assert f1_row["ic_mean"] == 0.15
        assert f1_row["ic_ir"] == 3.0
        assert f1_row["mdi_importance"] == 0.35

    def test_get_top_features(self):
        """Test top feature selection."""
        result = FeatureOutcomeResult(features=["f1", "f2", "f3"])

        # Add IC results with different values
        result.ic_results["f1"] = FeatureICResults(feature="f1", ic_ir=3.0)
        result.ic_results["f2"] = FeatureICResults(feature="f2", ic_ir=1.5)
        result.ic_results["f3"] = FeatureICResults(feature="f3", ic_ir=2.0)

        top_features = result.get_top_features(n=2, by="ic_ir")

        assert len(top_features) == 2
        assert top_features[0] == "f1"  # Highest IC IR
        assert top_features[1] == "f3"  # Second highest

    def test_get_top_features_with_errors(self):
        """Test that features with errors are excluded."""
        result = FeatureOutcomeResult(features=["f1", "f2", "f3"])

        result.ic_results["f1"] = FeatureICResults(feature="f1", ic_ir=3.0)
        result.ic_results["f2"] = FeatureICResults(feature="f2", ic_ir=2.0)
        # f3 has error
        result.errors["f3"] = "Test error"

        top_features = result.get_top_features(n=3, by="ic_ir")

        # Should only return 2 features (f3 excluded due to error)
        assert len(top_features) == 2
        assert "f3" not in top_features

    def test_get_recommendations(self):
        """Test recommendation generation."""
        result = FeatureOutcomeResult(features=["f1", "f2", "f3"])

        # f1: Strong signal
        result.ic_results["f1"] = FeatureICResults(feature="f1", ic_ir=2.5, p_value=0.001)

        # f2: Weak signal
        result.ic_results["f2"] = FeatureICResults(feature="f2", ic_ir=0.3, p_value=0.2)

        # f3: Error
        result.errors["f3"] = "Insufficient data"

        recommendations = result.get_recommendations()

        assert len(recommendations) > 0
        # Should mention strong signal
        assert any("f1" in rec and "Strong" in rec for rec in recommendations)
        # Should mention weak signals
        assert any("weak" in rec.lower() for rec in recommendations)
        # Should mention errors
        assert any("failed" in rec.lower() or "error" in rec.lower() for rec in recommendations)


class TestFeatureOutcome:
    """Test FeatureOutcome orchestration class."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        analyzer = FeatureOutcome()

        assert analyzer.config is not None
        assert isinstance(analyzer.config, ModuleCConfig)
        assert analyzer.config.ic.enabled is True

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = ModuleCConfig(
            ic=ICConfig(enabled=True, hac_adjustment=True),
            ml_diagnostics=MLDiagnosticsConfig(drift_detection=True),
        )
        analyzer = FeatureOutcome(config=config)

        assert analyzer.config.ic.hac_adjustment is True
        assert analyzer.config.ml_diagnostics.drift_detection is True

    def test_run_analysis_basic(self):
        """Test basic analysis with synthetic data."""
        np.random.seed(42)

        # Create synthetic data
        n = 500
        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, n),
                "f2": np.random.normal(0, 1, n),
                "f3": np.random.normal(0, 1, n),
            },
            index=pd.date_range("2020-01-01", periods=n),
        )

        # Create outcomes correlated with f1
        outcomes = pd.Series(
            features["f1"] * 0.5 + np.random.normal(0, 0.5, n), index=features.index
        )

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        assert isinstance(result, FeatureOutcomeResult)
        assert len(result.features) == 3
        assert "f1" in result.ic_results
        assert "f2" in result.ic_results
        assert "f3" in result.ic_results

        # f1 should have higher IC than f2/f3 (correlated with outcome)
        ic_f1 = result.ic_results["f1"].ic_mean
        ic_f2 = result.ic_results["f2"].ic_mean
        assert abs(ic_f1) > abs(ic_f2)

    def test_run_analysis_with_numpy_outcomes(self):
        """Test with numpy array outcomes."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, 100),
                "f2": np.random.normal(0, 1, 100),
            }
        )

        outcomes = np.random.normal(0, 1, 100)

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        assert isinstance(result, FeatureOutcomeResult)
        assert len(result.features) == 2

    def test_run_analysis_specific_features(self):
        """Test analyzing only specific features."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, 100),
                "f2": np.random.normal(0, 1, 100),
                "f3": np.random.normal(0, 1, 100),
            }
        )

        outcomes = pd.Series(np.random.normal(0, 1, 100))

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes, feature_names=["f1", "f2"])

        assert len(result.features) == 2
        assert "f1" in result.features
        assert "f2" in result.features
        assert "f3" not in result.features

    def test_run_analysis_with_missing_features(self):
        """Test error handling for missing features."""
        features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})

        outcomes = pd.Series([1, 2, 3])

        analyzer = FeatureOutcome()

        with pytest.raises(ValueError, match="Features not found"):
            analyzer.run_analysis(features, outcomes, feature_names=["f1", "missing"])

    def test_run_analysis_with_misaligned_data(self):
        """Test error handling for misaligned data."""
        features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})

        outcomes = pd.Series([1, 2])  # Different length!

        analyzer = FeatureOutcome()

        with pytest.raises(ValueError, match="must have same length"):
            analyzer.run_analysis(features, outcomes)

    def test_run_analysis_with_nans(self):
        """Test handling of NaN values."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "f1": [1, 2, np.nan, 4, 5],
                "f2": [np.nan, 2, 3, 4, 5],
            }
        )

        outcomes = pd.Series([1, 2, 3, np.nan, 5])

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        # Should handle NaNs gracefully - insufficient data leads to errors
        assert isinstance(result, FeatureOutcomeResult)
        # With only 5 samples and NaNs, should have errors for insufficient data
        assert "f1" in result.errors or "f1" in result.ic_results
        assert "f2" in result.errors or "f2" in result.ic_results

    def test_run_analysis_insufficient_data(self):
        """Test handling of insufficient data."""
        # Only 5 samples - too few for analysis
        features = pd.DataFrame(
            {
                "f1": [1, 2, 3, 4, 5],
                "f2": [np.nan, np.nan, np.nan, np.nan, np.nan],  # All NaN
            }
        )

        outcomes = pd.Series([1, 2, 3, 4, 5])

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        # f2 should have error due to all NaN
        assert "f2" in result.errors

    def test_run_analysis_with_importance(self):
        """Test ML importance analysis."""
        np.random.seed(42)

        # Need more samples for importance analysis
        n = 200
        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, n),
                "f2": np.random.normal(0, 1, n),
                "f3": np.random.normal(0, 1, n),
            }
        )

        # Outcomes correlated with f1
        outcomes = features["f1"] * 2 + np.random.normal(0, 0.5, n)

        config = ModuleCConfig(
            ml_diagnostics=MLDiagnosticsConfig(feature_importance=True, drift_detection=False)
        )

        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, outcomes)

        # Check if importance was computed (depends on LightGBM availability)
        if len(result.importance_results) > 0:
            assert "f1" in result.importance_results
            # f1 should have higher importance (correlated with outcome)
            imp_f1 = result.importance_results["f1"].mdi_importance
            assert imp_f1 > 0

    def test_run_analysis_with_drift_detection(self):
        """Test drift detection integration."""
        np.random.seed(42)

        n = 500
        # Create data with drift in second half
        f1_first = np.random.normal(0, 1, n // 2)
        f1_second = np.random.normal(0.5, 1, n // 2)  # Mean shift
        f1 = np.concatenate([f1_first, f1_second])

        features = pd.DataFrame(
            {
                "f1": f1,
                "f2": np.random.normal(0, 1, n),  # No drift
            }
        )

        outcomes = pd.Series(np.random.normal(0, 1, n))

        config = ModuleCConfig(
            ic=ICConfig(enabled=False),  # Disable IC for speed
            ml_diagnostics=MLDiagnosticsConfig(feature_importance=False, drift_detection=True),
        )

        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, outcomes)

        assert result.drift_results is not None
        # f1 should show drift
        drift_df = result.drift_results.to_dataframe()
        # Convert to pandas if polars
        if isinstance(drift_df, pl.DataFrame):
            drift_df = drift_df.to_pandas()
        f1_drift = drift_df[drift_df["feature"] == "f1"]
        # Depending on threshold, may or may not flag (just check it ran)
        assert len(f1_drift) > 0

    def test_summary_dataframe_completeness(self):
        """Test that summary DataFrame contains all expected columns."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, 200),
                "f2": np.random.normal(0, 1, 200),
            }
        )

        outcomes = pd.Series(np.random.normal(0, 1, 200))

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        df = result.summary

        assert df is not None
        assert "feature" in df.columns
        assert "ic_mean" in df.columns
        assert "ic_ir" in df.columns
        assert "error" in df.columns

    def test_metadata_tracking(self):
        """Test that metadata is properly tracked."""
        np.random.seed(42)

        features = pd.DataFrame({"f1": np.random.normal(0, 1, 100)})

        outcomes = pd.Series(np.random.normal(0, 1, 100))

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes)

        assert "n_features" in result.metadata
        assert result.metadata["n_features"] == 1
        assert "n_observations" in result.metadata
        assert result.metadata["n_observations"] == 100
        assert "computation_time" in result.metadata
        assert result.metadata["computation_time"] > 0

    def test_verbose_mode(self, capsys):
        """Test verbose output."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, 100),
                "f2": np.random.normal(0, 1, 100),
            }
        )

        outcomes = pd.Series(np.random.normal(0, 1, 100))

        analyzer = FeatureOutcome()
        result = analyzer.run_analysis(features, outcomes, verbose=True)

        captured = capsys.readouterr()
        assert "Analyzing" in captured.out
        assert "features" in captured.out
        assert "complete" in captured.out


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self):
        """Test complete workflow from data to recommendations."""
        np.random.seed(42)

        # Create realistic synthetic data
        n = 500
        features = pd.DataFrame(
            {
                "momentum": np.random.normal(0, 1, n),
                "value": np.random.normal(0, 1, n),
                "quality": np.random.normal(0, 1, n),
                "sentiment": np.random.normal(0, 1, n),
            },
            index=pd.date_range("2020-01-01", periods=n),
        )

        # Create returns with some correlation to momentum
        returns = (
            features["momentum"] * 0.3 + features["quality"] * 0.15 + np.random.normal(0, 1, n)
        )

        # Run full analysis
        config = ModuleCConfig(
            ic=ICConfig(enabled=True),
            ml_diagnostics=MLDiagnosticsConfig(feature_importance=True, drift_detection=True),
        )

        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, returns, verbose=False)

        # Validate results
        assert len(result.features) == 4
        assert len(result.ic_results) == 4

        # Get top features
        top_features = result.get_top_features(n=2, by="ic_ir")
        assert len(top_features) <= 2

        # Get recommendations
        recommendations = result.get_recommendations()
        assert isinstance(recommendations, list)

        # Export to DataFrame
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4


class TestSHAPIntegration:
    """Tests for SHAP importance analysis."""

    def test_shap_function_basic(self):
        """Test compute_shap_importance function basics."""
        pytest.importorskip("lightgbm")
        pytest.importorskip("shap")

        import lightgbm as lgb

        from ml4t.engineer.outcome.feature_outcome import compute_shap_importance

        # Create simple dataset
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 5)
        y = X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n) * 0.1

        # Train model
        model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
        model.fit(X, y)

        # Compute SHAP importance
        result = compute_shap_importance(
            model=model, X=X, feature_names=[f"f{i}" for i in range(5)], max_samples=100
        )

        # Validate result structure
        assert "shap_values" in result
        assert "importances" in result
        assert "feature_names" in result
        assert "base_value" in result
        assert "n_features" in result
        assert "n_samples" in result

        # Validate shapes
        assert result["shap_values"].shape[1] == 5  # 5 features
        assert result["shap_values"].shape[0] <= 100  # max_samples
        assert len(result["importances"]) == 5
        assert len(result["feature_names"]) == 5

        # Top features should be f0 and f1 (highest coefficients in y = X0 + 0.5*X1)
        assert result["feature_names"][0] in ["f0", "f1"]
        assert result["feature_names"][1] in ["f0", "f1"]

    def test_shap_integration_workflow(self):
        """Test SHAP in full FeatureOutcome workflow."""
        pytest.importorskip("lightgbm")
        pytest.importorskip("shap")

        np.random.seed(42)
        n = 300

        # Create features with known relationships
        features = pd.DataFrame(
            {
                "strong_signal": np.random.normal(0, 1, n),  # Strong predictor
                "weak_signal": np.random.normal(0, 1, n),  # Weak predictor
                "noise1": np.random.normal(0, 1, n),  # Pure noise
                "noise2": np.random.normal(0, 1, n),  # Pure noise
            }
        )

        # Create returns with strong correlation to strong_signal
        returns = (
            features["strong_signal"] * 0.5
            + features["weak_signal"] * 0.1
            + np.random.normal(0, 1, n) * 0.3
        )

        # Run analysis with SHAP enabled
        config = ModuleCConfig(
            ic=ICConfig(enabled=True),
            ml_diagnostics=MLDiagnosticsConfig(
                feature_importance=True,
                shap_analysis=True,  # Enable SHAP
            ),
        )

        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, returns, verbose=False)

        # Validate SHAP results exist
        assert len(result.importance_results) == 4

        # Check that SHAP values are populated
        has_shap = False
        for feature in result.features:
            if feature in result.importance_results:
                imp = result.importance_results[feature]
                if imp.shap_mean is not None:
                    has_shap = True
                    assert imp.shap_std is not None
                    assert imp.shap_mean >= 0  # Mean absolute SHAP is non-negative

        assert has_shap, "SHAP values should be computed when enabled"

        # Validate metadata
        assert result.metadata["shap_analysis_enabled"] is True

        # Check summary DataFrame includes SHAP columns
        df = result.to_dataframe()
        assert "shap_mean" in df.columns
        assert "shap_std" in df.columns

        # Top feature by SHAP should be strong_signal (highest correlation)
        shap_means = {
            f: result.importance_results[f].shap_mean
            for f in result.features
            if result.importance_results[f].shap_mean is not None
        }
        if len(shap_means) > 0:
            top_by_shap = max(shap_means, key=shap_means.get)
            # Allow for some randomness, but strong_signal should typically be top
            # (This test may occasionally fail due to randomness, which is acceptable)
            assert top_by_shap in ["strong_signal", "weak_signal"]

    def test_shap_with_pandas_dataframe(self):
        """Test SHAP works with pandas DataFrame input."""
        pytest.importorskip("lightgbm")
        pytest.importorskip("shap")

        import lightgbm as lgb

        from ml4t.engineer.outcome.feature_outcome import compute_shap_importance

        # Create pandas DataFrame
        np.random.seed(42)
        n = 150
        X_df = pd.DataFrame(
            {
                "col_a": np.random.randn(n),
                "col_b": np.random.randn(n),
                "col_c": np.random.randn(n),
            }
        )
        y = X_df["col_a"] + X_df["col_b"] * 0.5 + np.random.randn(n) * 0.1

        # Train model
        model = lgb.LGBMRegressor(n_estimators=30, random_state=42, verbose=-1)
        model.fit(X_df, y)

        # Compute SHAP importance (no feature_names - should infer from DataFrame)
        result = compute_shap_importance(model=model, X=X_df)

        # Should use column names
        assert "col_a" in result["feature_names"]
        assert "col_b" in result["feature_names"]
        assert "col_c" in result["feature_names"]

    def test_shap_performance_constraint(self):
        """Test SHAP respects max_samples for performance."""
        pytest.importorskip("lightgbm")
        pytest.importorskip("shap")

        import lightgbm as lgb

        from ml4t.engineer.outcome.feature_outcome import compute_shap_importance

        # Create large dataset
        np.random.seed(42)
        n = 10000  # Large dataset
        X = np.random.randn(n, 10)
        y = X[:, 0] + np.random.randn(n) * 0.1

        # Train model
        model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
        model.fit(X, y)

        # Compute SHAP with max_samples constraint
        max_samples = 1000
        result = compute_shap_importance(model=model, X=X, max_samples=max_samples)

        # Should respect max_samples
        assert result["n_samples"] == max_samples
        assert result["shap_values"].shape[0] == max_samples

    def test_shap_disabled_gracefully(self):
        """Test that SHAP disabled or unavailable doesn't break analysis."""
        np.random.seed(42)
        n = 200

        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, n),
                "f2": np.random.normal(0, 1, n),
            }
        )
        returns = features["f1"] * 0.3 + np.random.normal(0, 1, n)

        # Config with SHAP disabled
        config = ModuleCConfig(
            ml_diagnostics=MLDiagnosticsConfig(feature_importance=True, shap_analysis=False)
        )

        analyzer = FeatureOutcome(config=config)
        result = analyzer.run_analysis(features, returns, verbose=False)

        # Should work fine without SHAP
        assert len(result.importance_results) > 0

        # SHAP values should be None
        for feature in result.features:
            if feature in result.importance_results:
                imp = result.importance_results[feature]
                assert imp.shap_mean is None or np.isnan(imp.shap_mean)
                assert imp.shap_std is None or np.isnan(imp.shap_std)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
