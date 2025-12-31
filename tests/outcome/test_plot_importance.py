"""Tests for feature importance plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import matplotlib types directly (needed for isinstance checks)
from matplotlib.figure import Figure

from ml4t.engineer.outcome.feature_outcome import FeatureImportanceResults
from ml4t.engineer.outcome.plot_importance import (
    _importance_dict_to_dataframe,
    plot_feature_importance,
    plot_importance_comparison,
)


@pytest.fixture
def sample_importance_dict() -> dict[str, FeatureImportanceResults]:
    """Create sample importance results as dictionary."""
    return {
        "feature_1": FeatureImportanceResults(
            feature="feature_1",
            mdi_importance=0.25,
            permutation_importance=0.30,
            permutation_std=0.05,
            shap_mean=0.22,
            shap_std=0.03,
            rank_mdi=1,
            rank_permutation=1,
        ),
        "feature_2": FeatureImportanceResults(
            feature="feature_2",
            mdi_importance=0.20,
            permutation_importance=0.25,
            permutation_std=0.04,
            shap_mean=0.18,
            shap_std=0.02,
            rank_mdi=2,
            rank_permutation=2,
        ),
        "feature_3": FeatureImportanceResults(
            feature="feature_3",
            mdi_importance=0.15,
            permutation_importance=0.20,
            permutation_std=0.06,
            shap_mean=0.15,
            shap_std=0.04,
            rank_mdi=3,
            rank_permutation=3,
        ),
        "feature_4": FeatureImportanceResults(
            feature="feature_4",
            mdi_importance=0.10,
            permutation_importance=0.15,
            permutation_std=0.03,
            shap_mean=0.12,
            shap_std=0.02,
            rank_mdi=4,
            rank_permutation=4,
        ),
        "feature_5": FeatureImportanceResults(
            feature="feature_5",
            mdi_importance=0.05,
            permutation_importance=0.10,
            permutation_std=0.02,
            shap_mean=0.08,
            shap_std=0.01,
            rank_mdi=5,
            rank_permutation=5,
        ),
    }


@pytest.fixture
def sample_importance_df() -> pd.DataFrame:
    """Create sample importance results as DataFrame."""
    return pd.DataFrame(
        {
            "feature": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            "mdi_importance": [0.25, 0.20, 0.15, 0.10, 0.05],
            "permutation_importance": [0.30, 0.25, 0.20, 0.15, 0.10],
            "permutation_std": [0.05, 0.04, 0.06, 0.03, 0.02],
            "shap_mean": [0.22, 0.18, 0.15, 0.12, 0.08],
            "shap_std": [0.03, 0.02, 0.04, 0.02, 0.01],
            "rank_mdi": [1, 2, 3, 4, 5],
            "rank_permutation": [1, 2, 3, 4, 5],
        }
    )


class TestPlotFeatureImportance:
    """Tests for plot_feature_importance function."""

    def test_basic_mdi_plot_dict(self, sample_importance_dict: dict) -> None:
        """Test basic MDI importance plot from dict."""
        fig = plot_feature_importance(sample_importance_dict, importance_type="mdi")

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "MDI Importance"
        assert "Feature Importance (MDI)" in ax.get_title()
        assert len(ax.patches) == 5  # 5 bars

    def test_basic_mdi_plot_df(self, sample_importance_df: pd.DataFrame) -> None:
        """Test basic MDI importance plot from DataFrame."""
        fig = plot_feature_importance(sample_importance_df, importance_type="mdi")

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "MDI Importance"
        assert len(ax.patches) == 5

    def test_permutation_plot(self, sample_importance_dict: dict) -> None:
        """Test permutation importance plot."""
        fig = plot_feature_importance(sample_importance_dict, importance_type="permutation")

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Permutation Importance"
        assert "Feature Importance (PERMUTATION)" in ax.get_title()

    def test_shap_plot(self, sample_importance_dict: dict) -> None:
        """Test SHAP importance plot."""
        fig = plot_feature_importance(sample_importance_dict, importance_type="shap")

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Mean |SHAP|"
        assert "Feature Importance (SHAP)" in ax.get_title()

    def test_top_n_filtering(self, sample_importance_dict: dict) -> None:
        """Test filtering to top N features."""
        fig = plot_feature_importance(sample_importance_dict, importance_type="mdi", top_n=3)

        ax = fig.axes[0]
        assert len(ax.patches) == 3  # Only 3 bars

    def test_custom_title(self, sample_importance_dict: dict) -> None:
        """Test custom title."""
        custom_title = "My Custom Importance Plot"
        fig = plot_feature_importance(
            sample_importance_dict, importance_type="mdi", title=custom_title
        )

        ax = fig.axes[0]
        assert ax.get_title() == custom_title

    def test_custom_color(self, sample_importance_dict: dict) -> None:
        """Test custom bar color."""
        fig = plot_feature_importance(sample_importance_dict, importance_type="mdi", color="red")

        ax = fig.axes[0]
        # Check that bars have the specified color
        assert ax.patches[0].get_facecolor()[:3] == (1.0, 0.0, 0.0)  # RGB for red

    def test_custom_figsize(self, sample_importance_dict: dict) -> None:
        """Test custom figure size."""
        figsize = (15, 10)
        fig = plot_feature_importance(
            sample_importance_dict, importance_type="mdi", figsize=figsize
        )

        # Figure size in inches
        assert fig.get_size_inches()[0] == pytest.approx(figsize[0], abs=0.1)
        assert fig.get_size_inches()[1] == pytest.approx(figsize[1], abs=0.1)

    def test_error_bars_permutation(self, sample_importance_dict: dict) -> None:
        """Test error bars for permutation importance."""
        fig = plot_feature_importance(
            sample_importance_dict, importance_type="permutation", error_bars=True
        )

        ax = fig.axes[0]
        # Check that error bars exist (via container)
        assert len(ax.containers) > 0

    def test_error_bars_shap(self, sample_importance_dict: dict) -> None:
        """Test error bars for SHAP importance."""
        fig = plot_feature_importance(
            sample_importance_dict, importance_type="shap", error_bars=True
        )

        ax = fig.axes[0]
        # Check that error bars exist
        assert len(ax.containers) > 0

    def test_no_error_bars_mdi(self, sample_importance_dict: dict) -> None:
        """Test no error bars for MDI (not applicable)."""
        fig = plot_feature_importance(
            sample_importance_dict, importance_type="mdi", error_bars=True
        )

        # Should still work, just no error bars
        assert isinstance(fig, Figure)

    def test_custom_axes(self, sample_importance_dict: dict) -> None:
        """Test plotting on custom axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_fig = plot_feature_importance(sample_importance_dict, importance_type="mdi", ax=ax)

        assert result_fig is fig
        assert len(ax.patches) == 5

    def test_invalid_importance_type(self, sample_importance_dict: dict) -> None:
        """Test error on invalid importance type."""
        with pytest.raises(ValueError, match="importance_type must be one of"):
            plot_feature_importance(sample_importance_dict, importance_type="invalid")  # type: ignore[arg-type]

    def test_missing_feature_column(self) -> None:
        """Test error when DataFrame missing 'feature' column."""
        df = pd.DataFrame({"mdi_importance": [0.1, 0.2]})

        with pytest.raises(ValueError, match="must have 'feature' column"):
            plot_feature_importance(df, importance_type="mdi")

    def test_missing_importance_column(self, sample_importance_df: pd.DataFrame) -> None:
        """Test error when importance column missing."""
        df = sample_importance_df.drop(columns=["mdi_importance"])

        with pytest.raises(ValueError, match="missing 'mdi_importance' column"):
            plot_feature_importance(df, importance_type="mdi")

    def test_all_nan_values(self) -> None:
        """Test error when all importance values are NaN."""
        df = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "mdi_importance": [np.nan, np.nan],
            }
        )

        with pytest.raises(ValueError, match="No valid mdi importance values"):
            plot_feature_importance(df, importance_type="mdi")

    def test_some_nan_values(self, sample_importance_df: pd.DataFrame) -> None:
        """Test handling of some NaN values."""
        df = sample_importance_df.copy()
        df.loc[2, "mdi_importance"] = np.nan

        fig = plot_feature_importance(df, importance_type="mdi")

        ax = fig.axes[0]
        # Should have 4 bars (5 - 1 NaN)
        assert len(ax.patches) == 4

    def test_sorting_by_importance(self, sample_importance_dict: dict) -> None:
        """Test that features are sorted by importance."""
        fig = plot_feature_importance(sample_importance_dict, importance_type="mdi")

        ax = fig.axes[0]
        y_labels = [label.get_text() for label in ax.get_yticklabels()]

        # Should be in reverse order (lowest to highest, bottom to top)
        assert y_labels == ["feature_5", "feature_4", "feature_3", "feature_2", "feature_1"]

    def test_grid_present(self, sample_importance_dict: dict) -> None:
        """Test that grid is present."""
        fig = plot_feature_importance(sample_importance_dict, importance_type="mdi")

        ax = fig.axes[0]
        assert ax.xaxis.get_gridlines()[0].get_visible()


class TestPlotImportanceComparison:
    """Tests for plot_importance_comparison function."""

    def test_basic_comparison_dict(self, sample_importance_dict: dict) -> None:
        """Test basic comparison plot from dict."""
        fig = plot_importance_comparison(sample_importance_dict)

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Normalized Importance"
        assert "Feature Importance Comparison" in ax.get_title()

    def test_basic_comparison_df(self, sample_importance_df: pd.DataFrame) -> None:
        """Test basic comparison plot from DataFrame."""
        fig = plot_importance_comparison(sample_importance_df)

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Normalized Importance"

    def test_top_n_filtering(self, sample_importance_dict: dict) -> None:
        """Test filtering to top N features."""
        fig = plot_importance_comparison(sample_importance_dict, top_n=3)

        ax = fig.axes[0]
        # Should have 3 features × 3 metrics = 9 bars
        assert len(ax.patches) == 9

    def test_custom_title(self, sample_importance_dict: dict) -> None:
        """Test custom title."""
        custom_title = "My Comparison Plot"
        fig = plot_importance_comparison(sample_importance_dict, title=custom_title)

        ax = fig.axes[0]
        assert ax.get_title() == custom_title

    def test_custom_figsize(self, sample_importance_dict: dict) -> None:
        """Test custom figure size."""
        figsize = (15, 10)
        fig = plot_importance_comparison(sample_importance_dict, figsize=figsize)

        assert fig.get_size_inches()[0] == pytest.approx(figsize[0], abs=0.1)
        assert fig.get_size_inches()[1] == pytest.approx(figsize[1], abs=0.1)

    def test_legend_present(self, sample_importance_dict: dict) -> None:
        """Test that legend is present."""
        fig = plot_importance_comparison(sample_importance_dict)

        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None

        # Should have 3 entries (MDI, Permutation, SHAP)
        legend_texts = [text.get_text() for text in legend.get_texts()]
        assert "MDI" in legend_texts
        assert "Permutation" in legend_texts
        assert "SHAP" in legend_texts

    def test_normalization(self, sample_importance_dict: dict) -> None:
        """Test that importance values are normalized to [0, 1]."""
        fig = plot_importance_comparison(sample_importance_dict)

        ax = fig.axes[0]
        # Get all bar heights
        bar_heights = [patch.get_width() for patch in ax.patches]

        # All heights should be <= 1.0
        assert all(h <= 1.0 for h in bar_heights)
        # At least one should be close to 1.0 (the maximum)
        assert any(abs(h - 1.0) < 0.01 for h in bar_heights)

    def test_missing_feature_column(self) -> None:
        """Test error when DataFrame missing 'feature' column."""
        df = pd.DataFrame({"mdi_importance": [0.1, 0.2]})

        with pytest.raises(ValueError, match="must have 'feature' column"):
            plot_importance_comparison(df)

    def test_no_valid_metrics(self) -> None:
        """Test error when no valid importance metrics found."""
        df = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "mdi_importance": [np.nan, np.nan],
                "permutation_importance": [np.nan, np.nan],
                "shap_mean": [np.nan, np.nan],
            }
        )

        with pytest.raises(ValueError, match="No valid importance metrics"):
            plot_importance_comparison(df)

    def test_only_mdi_available(self) -> None:
        """Test comparison with only MDI importance available."""
        df = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "mdi_importance": [0.3, 0.2, 0.1],
                "permutation_importance": [np.nan, np.nan, np.nan],
                "shap_mean": [np.nan, np.nan, np.nan],
                "rank_mdi": [1, 2, 3],
                "rank_permutation": [0, 0, 0],
            }
        )

        fig = plot_importance_comparison(df)

        ax = fig.axes[0]
        # Should have 3 features × 1 metric = 3 bars
        assert len(ax.patches) == 3

    def test_mixed_availability(self) -> None:
        """Test comparison with mixed metric availability."""
        df = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "mdi_importance": [0.3, 0.2, 0.1],
                "permutation_importance": [0.25, 0.15, np.nan],
                "shap_mean": [np.nan, np.nan, np.nan],
                "rank_mdi": [1, 2, 3],
                "rank_permutation": [1, 2, 0],
            }
        )

        fig = plot_importance_comparison(df)

        # Should handle gracefully
        assert isinstance(fig, Figure)

    def test_grid_present(self, sample_importance_dict: dict) -> None:
        """Test that grid is present."""
        fig = plot_importance_comparison(sample_importance_dict)

        ax = fig.axes[0]
        assert ax.xaxis.get_gridlines()[0].get_visible()


class TestImportanceDictToDataFrame:
    """Tests for _importance_dict_to_dataframe helper."""

    def test_basic_conversion(self, sample_importance_dict: dict) -> None:
        """Test basic dict to DataFrame conversion."""
        df = _importance_dict_to_dataframe(sample_importance_dict)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "feature" in df.columns
        assert "mdi_importance" in df.columns
        assert "permutation_importance" in df.columns
        assert "shap_mean" in df.columns

    def test_all_columns_present(self, sample_importance_dict: dict) -> None:
        """Test that all expected columns are present."""
        df = _importance_dict_to_dataframe(sample_importance_dict)

        expected_columns = {
            "feature",
            "mdi_importance",
            "permutation_importance",
            "permutation_std",
            "shap_mean",
            "shap_std",
            "rank_mdi",
            "rank_permutation",
        }
        assert set(df.columns) == expected_columns

    def test_values_match(self, sample_importance_dict: dict) -> None:
        """Test that values match the input dict."""
        df = _importance_dict_to_dataframe(sample_importance_dict)

        # Check first row
        row1 = df[df["feature"] == "feature_1"].iloc[0]
        assert row1["mdi_importance"] == 0.25
        assert row1["permutation_importance"] == 0.30
        assert row1["shap_mean"] == 0.22

    def test_empty_dict(self) -> None:
        """Test conversion of empty dict."""
        df = _importance_dict_to_dataframe({})

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_none_shap_values(self) -> None:
        """Test handling of None SHAP values."""
        results = {
            "f1": FeatureImportanceResults(
                feature="f1",
                mdi_importance=0.5,
                permutation_importance=0.4,
                shap_mean=None,  # Not computed
                shap_std=None,
            )
        }

        df = _importance_dict_to_dataframe(results)

        assert pd.isna(df.loc[0, "shap_mean"])
        assert pd.isna(df.loc[0, "shap_std"])


class TestIntegration:
    """Integration tests for importance plotting."""

    def test_save_to_file(self, sample_importance_dict: dict, tmp_path) -> None:
        """Test saving plot to file."""
        output_path = tmp_path / "importance.png"

        fig = plot_feature_importance(sample_importance_dict, importance_type="mdi")
        fig.savefig(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_multiple_plots_same_data(self, sample_importance_dict: dict) -> None:
        """Test creating multiple plots from same data."""
        fig1 = plot_feature_importance(sample_importance_dict, importance_type="mdi")
        fig2 = plot_feature_importance(sample_importance_dict, importance_type="permutation")
        fig3 = plot_importance_comparison(sample_importance_dict)

        assert fig1 is not fig2
        assert fig2 is not fig3
        assert isinstance(fig1, Figure)
        assert isinstance(fig2, Figure)
        assert isinstance(fig3, Figure)
