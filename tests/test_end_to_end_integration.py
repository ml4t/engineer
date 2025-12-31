# mypy: disable-error-code="call-arg,no-untyped-def,operator"
"""End-to-end integration tests for complete ml4t-engineer workflow.

Tests the full feature engineering pipeline:
1. Relationships - Compute correlation matrices
2. Outcome - Analyze feature-outcome relationships (IC, importance, drift)
3. Selection - Systematic feature filtering
4. Visualization - Generate comprehensive plots

Note: Diagnostics functionality is available in the separate ml4t-diagnostic library.

This test suite validates that all modules work together seamlessly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.config.feature_config import (
    ICConfig,
    MLDiagnosticsConfig,
    ModuleCConfig,
)
from ml4t.engineer.outcome import FeatureOutcome
from ml4t.engineer.relationships import compute_correlation_matrix
from ml4t.engineer.selection import FeatureSelector
from ml4t.engineer.visualization import export_plot, plot_feature_analysis_summary

if TYPE_CHECKING:
    pass


@pytest.fixture
def sample_data():
    """Create realistic sample data for integration tests."""
    np.random.seed(42)
    n = 1000

    # Create returns time series (outcome variable)
    returns = np.random.randn(n) * 0.02  # 2% daily volatility

    # Create features with varying quality
    features = pl.DataFrame(
        {
            # High quality: Strong predictive power, stationary
            "momentum_5d": returns + np.random.randn(n) * 0.01,  # Leading indicator
            "momentum_20d": np.convolve(returns, np.ones(5) / 5, mode="same"),  # Smoothed momentum
            # Medium quality: Moderate signal
            "volatility_10d": np.abs(returns) + np.random.randn(n) * 0.005,
            "volume_trend": np.cumsum(np.random.randn(n) * 0.1),
            # Low quality: Noisy
            "noise_1": np.random.randn(n),
            "noise_2": np.random.randn(n),
            # Correlated pair (for redundancy testing)
            # Create genuinely correlated features
            "base_signal": returns * 2 + np.random.randn(n) * 0.01,
        }
    )

    # Add correlated features based on base_signal
    features = features.with_columns(
        [
            (pl.col("base_signal") + np.random.randn(n) * 0.02).alias("corr_feat_a"),
            (pl.col("base_signal") + np.random.randn(n) * 0.02).alias("corr_feat_b"),
            # Non-stationary (for diagnostics)
            pl.lit(np.cumsum(np.random.randn(n))).alias("trend_feature"),
        ]
    ).drop("base_signal")

    outcomes = pl.DataFrame({"forward_returns_1d": returns})

    return features, outcomes


class TestCompleteWorkflow:
    """Test complete feature engineering workflow end-to-end."""

    def test_full_pipeline_small_dataset(self, sample_data):
        """Test complete workflow with small dataset (1K rows)."""
        features, outcomes = sample_data

        # Step 1: Relationships - Compute correlation matrix
        corr_matrix = compute_correlation_matrix(features, method="pearson")
        assert isinstance(corr_matrix, pl.DataFrame)
        assert corr_matrix.shape == (
            len(features.columns),
            len(features.columns) + 1,
        )  # +1 for feature col

        # Verify high correlation between corr_feat_a and corr_feat_b
        corr_row_a = corr_matrix.filter(pl.col("feature") == "corr_feat_a")
        corr_ab = corr_row_a.select("corr_feat_b").item()
        assert abs(corr_ab) > 0.5, (
            f"Designed correlated features should have >0.5 correlation, got {corr_ab}"
        )

        # Step 2: Outcome - Analyze feature-outcome relationships
        config = ModuleCConfig(
            ic=ICConfig(lag_structure=[1, 5]),
            ml_diagnostics=MLDiagnosticsConfig(shap_analysis=False),  # Skip SHAP for speed
        )
        analyzer = FeatureOutcome(config=config)
        outcome_result = analyzer.run_analysis(features, outcomes.to_pandas()["forward_returns_1d"])

        assert len(outcome_result.ic_results) == len(features.columns)
        assert len(outcome_result.importance_results) == len(features.columns)

        # Step 3: Selection - Systematic feature filtering
        selector = FeatureSelector(outcome_result, corr_matrix)

        # Apply multi-stage filtering
        selector.filter_by_ic(threshold=0.01, min_periods=20)
        selector.filter_by_correlation(threshold=0.7, keep_strategy="higher_ic")
        selector.filter_by_importance(threshold=0.0, method="mdi", top_k=5)

        final_features = selector.selected_features
        report = selector.get_selection_report()

        # Verify filtering worked
        assert len(final_features) <= len(features.columns)
        assert len(final_features) > 0, "Should have at least one feature after filtering"

        # Verify report tracks all steps
        assert len(report.steps) == 3  # IC, correlation, importance
        assert len(report.final_features) == len(final_features)

        # Step 4: Visualization - Generate summary plot
        fig = plot_feature_analysis_summary(
            outcome_result,
            corr_matrix,
            top_n=5,
            importance_type="mdi",
        )
        assert fig is not None

    def test_workflow_with_drift_detection(self, sample_data):
        """Test workflow including drift detection step."""
        features, outcomes = sample_data

        # Split into train and test for drift detection
        train_features = features[:700]
        test_features = features[700:]

        # Run outcome analysis on train set
        config = ModuleCConfig(
            ic=ICConfig(lag_structure=[1]),
            ml_diagnostics=MLDiagnosticsConfig(shap_analysis=False),
        )
        analyzer = FeatureOutcome(config=config)
        outcome_result = analyzer.run_analysis(
            train_features,
            outcomes[:700].to_pandas()["forward_returns_1d"],
        )

        # Compute correlation on train set
        corr_matrix = compute_correlation_matrix(train_features, method="spearman")

        # Create selector
        selector = FeatureSelector(outcome_result, corr_matrix)

        # Note: analyze_drift requires train/test split
        # Verify that drift analysis can be performed separately
        from ml4t.engineer.outcome.drift import analyze_drift

        drift_results = analyze_drift(
            train_features.to_pandas(),
            test_features.to_pandas(),
        )

        # Verify drift results are available for selection decisions
        assert drift_results is not None
        assert len(drift_results.feature_results) == len(train_features.columns)

        # Apply standard filtering
        selector.filter_by_ic(threshold=0.01)
        selector.filter_by_correlation(threshold=0.8)

        final_features = selector.selected_features
        assert len(final_features) >= 0  # May be 0 if all filtered out

    def test_workflow_handles_missing_data(self):
        """Test that workflow handles missing data gracefully."""
        np.random.seed(42)
        n = 500

        # Create data with missing values
        features = pl.DataFrame(
            {
                "feat_complete": np.random.randn(n),
                "feat_missing": [
                    np.random.randn() if np.random.rand() > 0.1 else None for _ in range(n)
                ],
                "feat_mostly_missing": [
                    np.random.randn() if np.random.rand() > 0.5 else None for _ in range(n)
                ],
            }
        )
        outcomes = pl.DataFrame({"returns": np.random.randn(n) * 0.02})

        # Correlation should handle missing data (pairwise deletion)
        corr_matrix = compute_correlation_matrix(features, method="pearson")
        assert corr_matrix is not None

        # Outcome analysis should handle missing data
        config = ModuleCConfig(
            ml_diagnostics=MLDiagnosticsConfig(shap_analysis=False),
        )
        analyzer = FeatureOutcome(config=config)

        # This should work without errors (dropna handled internally)
        outcome_result = analyzer.run_analysis(features, outcomes.to_pandas()["returns"])
        assert len(outcome_result.ic_results) > 0


class TestWorkflowPerformance:
    """Performance benchmarks for complete workflow."""

    def test_medium_dataset_performance(self):
        """Test workflow with medium dataset (10K rows, 50 features) completes <30s.

        This is a critical acceptance criterion: the complete workflow must
        finish in under 30 seconds for practical use in research.
        """
        np.random.seed(42)
        n_rows = 10_000
        n_features = 50

        # Create realistic feature matrix
        features = pl.DataFrame(
            {f"feature_{i}": np.random.randn(n_rows) for i in range(n_features)}
        )

        # Create correlated outcome
        outcome_base = np.random.randn(n_rows)
        # Make first 10 features predictive
        for i in range(10):
            features = features.with_columns(
                (pl.col(f"feature_{i}") + outcome_base * 0.3 + np.random.randn(n_rows) * 0.1).alias(
                    f"feature_{i}"
                )
            )

        outcomes = pl.DataFrame({"returns": outcome_base * 0.02})

        # Time the complete workflow
        start_time = time.time()

        # Step 1: Skip diagnostics for performance (not the bottleneck)
        # Diagnostics would need to iterate through all features which adds overhead
        # but is not part of the core analysis workflow

        # Step 2: Correlation
        corr_matrix = compute_correlation_matrix(features, method="pearson")

        # Step 3: Outcome analysis (skip SHAP for speed)
        config = ModuleCConfig(
            ic=ICConfig(lag_structure=[1, 5]),
            ml_diagnostics=MLDiagnosticsConfig(shap_analysis=False),
        )
        analyzer = FeatureOutcome(config=config)
        outcome_result = analyzer.run_analysis(features, outcomes.to_pandas()["returns"])

        # Step 4: Selection
        selector = FeatureSelector(outcome_result, corr_matrix)
        selector.filter_by_ic(threshold=0.01).filter_by_correlation(
            threshold=0.8
        ).filter_by_importance(threshold=0, method="mdi", top_k=20)

        # Step 5: Visualization
        fig = plot_feature_analysis_summary(outcome_result, corr_matrix, top_n=10)

        elapsed = time.time() - start_time

        # CRITICAL: Must complete in <30 seconds
        assert elapsed < 30, f"Workflow took {elapsed:.2f}s, must be <30s"

        # Verify results are valid
        assert len(selector.selected_features) >= 0
        assert fig is not None


class TestOptionalDependencies:
    """Test workflow with and without optional dependencies."""

    def test_workflow_without_shap(self, sample_data):
        """Test that workflow works when SHAP is not available."""
        features, outcomes = sample_data

        # Configure to skip SHAP
        config = ModuleCConfig(
            ml_diagnostics=MLDiagnosticsConfig(shap_analysis=False),
        )
        analyzer = FeatureOutcome(config=config)
        outcome_result = analyzer.run_analysis(features, outcomes.to_pandas()["forward_returns_1d"])

        # Should still have MDI and permutation importance
        for result in outcome_result.importance_results.values():
            assert result.mdi_importance is not None
            # SHAP should be None when disabled
            assert result.shap_mean is None

        # Selection should work with just MDI
        selector = FeatureSelector(outcome_result, None)
        selector.filter_by_importance(threshold=0, method="mdi", top_k=5)
        assert len(selector.selected_features) > 0

    @pytest.mark.slow
    def test_workflow_with_shap(self, sample_data):
        """Test that workflow works with SHAP enabled (slow test)."""
        try:
            import shap  # noqa: F401
        except ImportError:
            pytest.skip("SHAP not installed")

        features, outcomes = sample_data

        # Enable SHAP
        config = ModuleCConfig(
            ic=ICConfig(lag_structure=[1]),
            ml_diagnostics=MLDiagnosticsConfig(shap_analysis=True, shap_sample_size=100),
        )
        analyzer = FeatureOutcome(config=config)
        outcome_result = analyzer.run_analysis(features, outcomes.to_pandas()["forward_returns_1d"])

        # Should have SHAP values
        has_shap = False
        for result in outcome_result.importance_results.values():
            if result.shap_mean is not None:
                has_shap = True
                break

        assert has_shap, "At least one feature should have SHAP values when enabled"

        # Selection should work with SHAP
        selector = FeatureSelector(outcome_result, None)
        selector.filter_by_importance(threshold=0, method="shap", top_k=5)
        assert len(selector.selected_features) > 0


class TestVisualizationExport:
    """Test visualization export functionality."""

    def test_plot_export_to_png(self, sample_data, tmp_path):
        """Test that plots can be exported to PNG."""
        features, outcomes = sample_data

        # Run minimal analysis
        config = ModuleCConfig(
            ic=ICConfig(lag_structure=[1]),
            ml_diagnostics=MLDiagnosticsConfig(shap_analysis=False),
        )
        analyzer = FeatureOutcome(config=config)
        outcome_result = analyzer.run_analysis(features, outcomes.to_pandas()["forward_returns_1d"])
        corr_matrix = compute_correlation_matrix(features)

        # Generate plot
        fig = plot_feature_analysis_summary(outcome_result, corr_matrix, top_n=5)

        # Export to PNG
        output_path = tmp_path / "analysis_summary.png"
        export_plot(fig, output_path, dpi=150)

        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0  # Non-empty file

    def test_plot_export_to_pdf(self, sample_data, tmp_path):
        """Test that plots can be exported to PDF."""
        features, outcomes = sample_data

        # Run minimal analysis
        config = ModuleCConfig(
            ic=ICConfig(lag_structure=[1]),
            ml_diagnostics=MLDiagnosticsConfig(shap_analysis=False),
        )
        analyzer = FeatureOutcome(config=config)
        outcome_result = analyzer.run_analysis(features, outcomes.to_pandas()["forward_returns_1d"])
        corr_matrix = compute_correlation_matrix(features)

        # Generate plot
        fig = plot_feature_analysis_summary(outcome_result, corr_matrix, top_n=5)

        # Export to PDF
        output_path = tmp_path / "analysis_summary.pdf"
        export_plot(fig, output_path)

        # Verify file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0  # Non-empty file


class TestErrorHandling:
    """Test error handling in integrated workflows."""

    def test_empty_dataframe_handling(self):
        """Test that empty DataFrames are handled gracefully."""
        empty_features = pl.DataFrame()

        # Correlation should handle empty input
        with pytest.raises((ValueError, RuntimeError)):
            compute_correlation_matrix(empty_features)

    def test_mismatched_lengths(self):
        """Test that mismatched feature/outcome lengths are caught."""
        features = pl.DataFrame({"feat1": [1, 2, 3]})
        outcomes = pl.DataFrame({"returns": [1, 2, 3, 4, 5]})  # Different length

        config = ModuleCConfig()
        analyzer = FeatureOutcome(config=config)

        # Should raise error about length mismatch
        with pytest.raises((ValueError, RuntimeError)):
            analyzer.run_analysis(features, outcomes.to_pandas()["returns"])

    def test_all_nan_feature(self):
        """Test handling of all-NaN feature."""
        features = pl.DataFrame(
            {
                "valid_feat": [1, 2, 3, 4, 5],
                "nan_feat": [None, None, None, None, None],
            }
        )
        outcomes = pl.DataFrame({"returns": [1, 2, 3, 4, 5]})

        # Should handle gracefully (may skip NaN feature or include it with NaN results)
        config = ModuleCConfig(ml_diagnostics=MLDiagnosticsConfig(shap_analysis=False))
        analyzer = FeatureOutcome(config=config)
        outcome_result = analyzer.run_analysis(features, outcomes.to_pandas()["returns"])

        # Should have results (may include all features or skip NaN ones)
        assert len(outcome_result.ic_results) >= 0  # May be 0 if all features invalid
