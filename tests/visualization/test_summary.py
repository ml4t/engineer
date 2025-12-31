"""Tests for unified visualization module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from matplotlib.figure import Figure

from ml4t.engineer.outcome.feature_outcome import (
    FeatureICResults,
    FeatureImportanceResults,
    FeatureOutcomeResult,
)
from ml4t.engineer.visualization import export_plot, plot_feature_analysis_summary


@pytest.fixture
def mock_feature_results():
    """Create mock FeatureOutcomeResult for testing."""
    # Create 20 features with varying importance and IC values
    features = [f"feature_{i}" for i in range(20)]

    # IC results
    ic_results = {}
    for i, feat in enumerate(features):
        # Create varying IC values
        ic_mean = 0.5 - i * 0.02  # Decreasing from 0.5 to 0.12
        ic_std = 0.1 + i * 0.005  # Increasing from 0.1 to 0.195
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

        ic_results[feat] = FeatureICResults(
            feature=feat,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            t_stat=ic_mean / ic_std * np.sqrt(100),
            p_value=0.01 if abs(ic_mean) > 0.2 else 0.1,
            ic_by_lag={0: ic_mean, 1: ic_mean * 0.9, 5: ic_mean * 0.7},
            n_observations=100,
        )

    # Importance results
    importance_results = {}
    for i, feat in enumerate(features):
        # Create varying importance values (different ranking than IC)
        mdi = 1.0 - i * 0.04  # Decreasing from 1.0 to 0.24
        perm = 0.8 - i * 0.03  # Decreasing from 0.8 to 0.23
        shap = 0.6 - i * 0.025  # Decreasing from 0.6 to 0.125

        importance_results[feat] = FeatureImportanceResults(
            feature=feat,
            mdi_importance=mdi,
            permutation_importance=perm,
            permutation_std=0.05,
            shap_mean=shap,
            shap_std=0.03,
            rank_mdi=i + 1,
            rank_permutation=i + 1,
        )

    return FeatureOutcomeResult(
        features=features,
        ic_results=ic_results,
        importance_results=importance_results,
    )


@pytest.fixture
def mock_correlation_matrix():
    """Create mock correlation matrix for testing."""
    features = [f"feature_{i}" for i in range(20)]

    # Create correlation matrix with some structure
    n = len(features)
    corr_values = np.eye(n)  # Start with identity

    # Add some correlations
    for i in range(n):
        for j in range(i + 1, n):
            # Correlation decreases with distance
            corr = 0.8 * np.exp(-abs(i - j) / 5.0)
            corr_values[i, j] = corr
            corr_values[j, i] = corr

    # Convert to DataFrame
    data = {"feature": features}
    for i, feat in enumerate(features):
        data[feat] = corr_values[:, i]

    return pl.DataFrame(data)


class TestPlotFeatureAnalysisSummary:
    """Test suite for plot_feature_analysis_summary."""

    def test_basic_3panel_plot(self, mock_feature_results, mock_correlation_matrix):
        """Test creating basic 3-panel summary plot."""
        fig = plot_feature_analysis_summary(mock_feature_results, mock_correlation_matrix, top_n=10)

        assert isinstance(fig, Figure)
        # Note: correlation heatmap adds a colorbar, so we get 4 axes (3 panels + 1 colorbar)
        assert len(fig.axes) >= 3  # At least 3 panels

        # Check main axes have titles (colorbar may not)
        main_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        assert len(main_axes) == 3
        for ax in main_axes:
            title = ax.get_title()
            assert len(title) > 0

    def test_2panel_plot_without_correlation(self, mock_feature_results):
        """Test creating 2-panel plot when correlation is None."""
        fig = plot_feature_analysis_summary(mock_feature_results, correlation_matrix=None, top_n=10)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2  # Only importance and IC panels

    def test_custom_top_n(self, mock_feature_results, mock_correlation_matrix):
        """Test with different top_n values."""
        for top_n in [5, 10, 15, 20]:
            fig = plot_feature_analysis_summary(
                mock_feature_results, mock_correlation_matrix, top_n=top_n
            )
            assert isinstance(fig, Figure)

    def test_custom_figsize(self, mock_feature_results, mock_correlation_matrix):
        """Test with custom figure size."""
        fig = plot_feature_analysis_summary(
            mock_feature_results, mock_correlation_matrix, figsize=(20, 16), top_n=10
        )

        assert isinstance(fig, Figure)
        # Check figure size (allow small tolerance)
        width, height = fig.get_size_inches()
        assert abs(width - 20) < 0.5
        assert abs(height - 16) < 0.5

    def test_custom_title(self, mock_feature_results):
        """Test with custom title."""
        custom_title = "My Custom Analysis"
        fig = plot_feature_analysis_summary(mock_feature_results, title=custom_title, top_n=10)

        assert isinstance(fig, Figure)
        assert fig._suptitle is not None
        assert custom_title in fig._suptitle.get_text()

    def test_different_importance_types(self, mock_feature_results, mock_correlation_matrix):
        """Test with different importance types."""
        for imp_type in ["mdi", "permutation", "shap"]:
            fig = plot_feature_analysis_summary(
                mock_feature_results,
                mock_correlation_matrix,
                importance_type=imp_type,
                top_n=10,
            )
            assert isinstance(fig, Figure)

    def test_empty_results_graceful_handling(self):
        """Test handling of empty results."""
        # Create result with empty dicts
        empty_result = FeatureOutcomeResult(
            features=[],
            ic_results={},
            importance_results={},
        )

        # Should not raise, but show error messages in panels
        fig = plot_feature_analysis_summary(empty_result, top_n=10)
        assert isinstance(fig, Figure)

    def test_missing_importance_data(self, mock_feature_results):
        """Test handling when importance data is missing."""
        # Remove importance results
        mock_feature_results.importance_results = {}

        fig = plot_feature_analysis_summary(mock_feature_results, top_n=10)
        assert isinstance(fig, Figure)

    def test_missing_ic_data(self, mock_feature_results):
        """Test handling when IC data is missing."""
        # Remove IC results
        mock_feature_results.ic_results = {}

        fig = plot_feature_analysis_summary(mock_feature_results, top_n=10)
        assert isinstance(fig, Figure)

    def test_partial_correlation_features(self, mock_feature_results):
        """Test when correlation matrix has only subset of features."""
        # Create correlation matrix with only 5 features
        features = [f"feature_{i}" for i in range(5)]
        n = len(features)
        corr_values = np.eye(n)

        data = {"feature": features}
        for i, feat in enumerate(features):
            data[feat] = corr_values[:, i]

        partial_corr = pl.DataFrame(data)

        fig = plot_feature_analysis_summary(mock_feature_results, partial_corr, top_n=10)
        assert isinstance(fig, Figure)


class TestExportPlot:
    """Test suite for export_plot function."""

    def test_export_png(self, mock_feature_results, tmp_path):
        """Test exporting to PNG."""
        fig = plot_feature_analysis_summary(mock_feature_results, top_n=5)

        output_path = tmp_path / "test_plot.png"
        export_plot(fig, output_path, dpi=150)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_pdf(self, mock_feature_results, tmp_path):
        """Test exporting to PDF."""
        fig = plot_feature_analysis_summary(mock_feature_results, top_n=5)

        output_path = tmp_path / "test_plot.pdf"
        export_plot(fig, output_path, dpi=300)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_export_different_dpi(self, mock_feature_results, tmp_path):
        """Test exporting with different DPI values."""
        fig = plot_feature_analysis_summary(mock_feature_results, top_n=5)

        for dpi in [150, 300, 600]:
            output_path = tmp_path / f"test_plot_{dpi}.png"
            export_plot(fig, output_path, dpi=dpi)
            assert output_path.exists()

    def test_export_creates_parent_dirs(self, mock_feature_results, tmp_path):
        """Test that export creates parent directories if needed."""
        fig = plot_feature_analysis_summary(mock_feature_results, top_n=5)

        output_path = tmp_path / "subdir1" / "subdir2" / "test_plot.png"
        export_plot(fig, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_export_str_path(self, mock_feature_results, tmp_path):
        """Test that export works with string paths."""
        fig = plot_feature_analysis_summary(mock_feature_results, top_n=5)

        output_path = str(tmp_path / "test_plot.png")
        export_plot(fig, output_path)

        assert Path(output_path).exists()

    def test_export_invalid_extension(self, mock_feature_results, tmp_path):
        """Test error handling for invalid file extension."""
        fig = plot_feature_analysis_summary(mock_feature_results, top_n=5)

        output_path = tmp_path / "test_plot.invalid"

        with pytest.raises(ValueError, match="Unsupported format"):
            export_plot(fig, output_path)

    def test_export_custom_kwargs(self, mock_feature_results, tmp_path):
        """Test passing custom kwargs to savefig."""
        fig = plot_feature_analysis_summary(mock_feature_results, top_n=5)

        output_path = tmp_path / "test_plot.png"
        export_plot(
            fig,
            output_path,
            dpi=150,
            facecolor="white",
            edgecolor="none",
        )

        assert output_path.exists()


class TestPerformance:
    """Performance tests for visualization."""

    def test_summary_plot_performance(self, mock_feature_results, mock_correlation_matrix):
        """Test that summary plot generation is fast (<5 seconds)."""
        import time

        start = time.time()

        # Create plot with 50 features (only showing top 10)
        # Add more features to the result
        for i in range(20, 50):
            feat = f"feature_{i}"
            mock_feature_results.features.append(feat)

            # Add IC
            ic_mean = 0.5 - i * 0.01
            ic_std = 0.1
            mock_feature_results.ic_results[feat] = FeatureICResults(
                feature=feat,
                ic_mean=ic_mean,
                ic_std=ic_std,
                ic_ir=ic_mean / ic_std,
                t_stat=ic_mean / ic_std * np.sqrt(100),
                p_value=0.05,
                ic_by_lag={0: ic_mean},
                n_observations=100,
            )

            # Add importance
            mock_feature_results.importance_results[feat] = FeatureImportanceResults(
                feature=feat,
                mdi_importance=1.0 - i * 0.02,
                permutation_importance=0.8 - i * 0.015,
                permutation_std=0.05,
                shap_mean=0.6 - i * 0.012,
                shap_std=0.03,
                rank_mdi=i + 1,
                rank_permutation=i + 1,
            )

        fig = plot_feature_analysis_summary(mock_feature_results, mock_correlation_matrix, top_n=10)

        elapsed = time.time() - start

        assert isinstance(fig, Figure)
        assert elapsed < 5.0, f"Plot generation took {elapsed:.2f}s, expected <5s"


class TestIntegration:
    """Integration tests with real-world scenarios."""

    def test_end_to_end_workflow(self, mock_feature_results, mock_correlation_matrix, tmp_path):
        """Test complete workflow: create plot and export."""
        # Create plot
        fig = plot_feature_analysis_summary(
            mock_feature_results,
            mock_correlation_matrix,
            top_n=15,
            title="End-to-End Test",
        )

        # Export to multiple formats
        for fmt in ["png", "pdf"]:
            output_path = tmp_path / f"summary.{fmt}"
            export_plot(fig, output_path, dpi=300)
            assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
