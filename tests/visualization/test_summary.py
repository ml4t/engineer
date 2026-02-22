"""Tests for unified visualization module.

.. deprecated:: 0.1.0a8
    Feature outcome visualization has moved to ml4t-diagnostic.
    Use ml4t.diagnostic.visualization instead.

These tests require ml4t-diagnostic to be installed for the mock fixtures.
"""

from __future__ import annotations

import pytest

# These tests require ml4t-diagnostic for the data classes
try:
    from ml4t.diagnostic.evaluation import FeatureOutcome  # noqa: F401

    HAS_DIAGNOSTIC = True
except ImportError:
    HAS_DIAGNOSTIC = False

from ml4t.engineer.visualization import export_plot

pytestmark = pytest.mark.skipif(
    not HAS_DIAGNOSTIC,
    reason="Visualization tests require ml4t-diagnostic. Install with: pip install ml4t-diagnostic",
)


class TestExportPlot:
    """Test export_plot utility function (always available)."""

    def test_export_requires_figure(self, tmp_path):
        """Test that export_plot validates figure argument."""
        output_path = tmp_path / "test.png"
        # Should raise error with invalid figure
        with pytest.raises((AttributeError, TypeError)):
            export_plot(None, output_path)

    def test_export_invalid_extension(self, tmp_path):
        """Test that export_plot validates file extension."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "test.xyz"
        with pytest.raises(ValueError, match="Unsupported format"):
            export_plot(fig, output_path)

        plt.close(fig)

    def test_export_creates_directory(self, tmp_path):
        """Test that export_plot creates parent directories."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "subdir" / "test.png"
        export_plot(fig, output_path, dpi=72)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        plt.close(fig)


# Note: Feature analysis summary visualization tests have moved to ml4t-diagnostic.
# The plot_feature_analysis_summary function is deprecated and will raise
# NotImplementedError directing users to ml4t.diagnostic.visualization.
