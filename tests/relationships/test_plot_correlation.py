"""Tests for correlation heatmap plotting."""

import numpy as np
import polars as pl
import pytest

from ml4t.engineer.relationships.correlation import compute_correlation_matrix
from ml4t.engineer.relationships.plot_correlation import plot_correlation_heatmap


class TestHeatmapBasic:
    """Basic heatmap plotting tests."""

    def test_basic_heatmap(self):
        """Test basic heatmap generation."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10],
                "z": [5, 4, 3, 2, 1],
            }
        )

        corr = compute_correlation_matrix(df, method="pearson")
        fig = plot_correlation_heatmap(corr)

        assert fig is not None
        assert len(fig.axes) == 2  # Main plot + colorbar
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_custom_figsize(self):
        """Test custom figure size."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, figsize=(6, 4))

        assert fig.get_size_inches()[0] == pytest.approx(6.0, abs=0.1)
        assert fig.get_size_inches()[1] == pytest.approx(4.0, abs=0.1)

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_custom_title(self):
        """Test custom plot title."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 3, 4],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, title="Test Correlation")

        ax = fig.axes[0]
        assert ax.get_title() == "Test Correlation"

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_default_title(self):
        """Test default title."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 3, 4],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr)

        ax = fig.axes[0]
        assert ax.get_title() == "Correlation Matrix"

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestHeatmapAnnotations:
    """Test value annotations."""

    def test_annotations_enabled(self):
        """Test that annotations are present by default."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, annot=True)

        ax = fig.axes[0]
        texts = ax.texts
        assert len(texts) > 0  # Should have text annotations

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_annotations_disabled(self):
        """Test that annotations can be disabled."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, annot=False)

        ax = fig.axes[0]
        texts = ax.texts
        assert len(texts) == 0  # Should have no text annotations

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_annotation_format(self):
        """Test custom annotation format."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, annot=True, fmt=".3f")

        ax = fig.axes[0]
        # Check that some text has 3 decimal places
        texts = [t.get_text() for t in ax.texts]
        assert any("." in t and len(t.split(".")[-1]) == 3 for t in texts if t != "NaN")

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestThresholdFiltering:
    """Test threshold filtering."""

    def test_threshold_applied(self):
        """Test that threshold filtering works."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10],  # Perfect correlation
                "z": [1, 3, 2, 5, 4],  # Weak correlation
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, threshold=0.5)

        # Should create plot without error
        assert fig is not None

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_threshold_none(self):
        """Test that threshold can be None (no filtering)."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, threshold=None)

        assert fig is not None

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestColorbar:
    """Test colorbar functionality."""

    def test_colorbar_present(self):
        """Test that colorbar is added."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr)

        # Should have 2 axes: main plot + colorbar
        assert len(fig.axes) == 2

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_colorbar_range(self):
        """Test that colorbar has correct range."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr)

        ax = fig.axes[0]
        images = ax.get_images()
        assert len(images) == 1

        # Check color limits are -1 to 1
        im = images[0]
        assert im.get_clim() == (-1, 1)

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestLabels:
    """Test axis labels."""

    def test_labels_present(self):
        """Test that feature labels are present."""
        df = pl.DataFrame(
            {
                "feature_a": [1, 2, 3],
                "feature_b": [2, 4, 6],
                "feature_c": [3, 2, 1],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr)

        ax = fig.axes[0]
        xticklabels = [label.get_text() for label in ax.get_xticklabels()]
        yticklabels = [label.get_text() for label in ax.get_yticklabels()]

        assert "feature_a" in xticklabels
        assert "feature_b" in xticklabels
        assert "feature_c" in xticklabels

        assert "feature_a" in yticklabels
        assert "feature_b" in yticklabels
        assert "feature_c" in yticklabels

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_labels_rotated(self):
        """Test that x-axis labels are rotated."""
        df = pl.DataFrame(
            {
                "long_feature_name_a": [1, 2, 3],
                "long_feature_name_b": [2, 4, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr)

        ax = fig.axes[0]
        # Check that x-labels are rotated
        for label in ax.get_xticklabels():
            rotation = label.get_rotation()
            assert rotation == 45

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestColormaps:
    """Test different colormaps."""

    def test_default_colormap(self):
        """Test default colormap (RdBu_r)."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr)

        ax = fig.axes[0]
        im = ax.get_images()[0]
        assert im.get_cmap().name == "RdBu_r"

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_custom_colormap(self):
        """Test custom colormap."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, cmap="coolwarm")

        ax = fig.axes[0]
        im = ax.get_images()[0]
        assert im.get_cmap().name == "coolwarm"

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestEdgeCases:
    """Test edge cases."""

    def test_single_feature(self):
        """Test with single feature."""
        df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr)

        assert fig is not None

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_with_nans(self):
        """Test with NaN correlations."""
        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [4, 5, 6],
                "const": [1, 1, 1],  # Constant = NaN correlation
            }
        )

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr)

        assert fig is not None

        # Check that NaN is shown in annotations
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert "NaN" in texts or "nan" in texts

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_large_matrix(self):
        """Test with larger correlation matrix."""
        np.random.seed(42)
        n_features = 10
        data = {f"feature_{i}": np.random.randn(50) for i in range(n_features)}
        df = pl.DataFrame(data)

        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, figsize=(12, 10))

        assert fig is not None
        assert len(fig.axes) == 2  # Main + colorbar

        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestErrorHandling:
    """Test error handling."""

    def test_missing_feature_column(self):
        """Test error when feature column is missing."""
        # Create invalid correlation matrix
        df = pl.DataFrame(
            {
                "x": [1.0, 0.5],
                "y": [0.5, 1.0],
            }
        )

        with pytest.raises(ValueError, match="must have 'feature' column"):
            plot_correlation_heatmap(df)

    def test_non_square_matrix(self):
        """Test error with non-square matrix."""
        # Create non-square matrix
        df = pl.DataFrame(
            {
                "feature": ["x", "y"],
                "x": [1.0, 0.5],
                "y": [0.5, 1.0],
                "z": [0.3, 0.4],  # Extra column
            }
        )

        with pytest.raises(ValueError, match="must be square"):
            plot_correlation_heatmap(df)


class TestCustomAxes:
    """Test custom axes functionality."""

    def test_custom_axes(self):
        """Test plotting on custom axes."""
        plt = pytest.importorskip("matplotlib.pyplot")

        df = pl.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )

        corr = compute_correlation_matrix(df)

        # Create custom figure and axes
        custom_fig, custom_ax = plt.subplots(figsize=(8, 6))

        # Plot on custom axes
        fig = plot_correlation_heatmap(corr, ax=custom_ax)

        # Should return the same figure
        assert fig is custom_fig

        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
