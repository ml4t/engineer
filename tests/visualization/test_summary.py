"""Tests for plot export utility."""

from __future__ import annotations

import pytest

from ml4t.engineer.visualization import export_plot


class TestExportPlot:
    """Test export_plot utility function."""

    def test_export_requires_figure(self, tmp_path):
        """Test that export_plot validates figure argument."""
        output_path = tmp_path / "test.png"
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
