"""Tests for IC plotting functions."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from ml4t.engineer.outcome.feature_outcome import FeatureICResults
from ml4t.engineer.outcome.plot_ic import (
    plot_ic_comparison,
    plot_ic_time_series,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture
def sample_ic_results() -> dict[str, FeatureICResults]:
    """Create sample IC results for testing."""
    return {
        "feature_1": FeatureICResults(
            feature="feature_1",
            ic_mean=0.15,
            ic_std=0.05,
            ic_ir=3.0,
            t_stat=2.5,
            p_value=0.012,
            ic_by_lag={0: 0.12, 1: 0.15, 5: 0.18, 10: 0.14, 21: 0.13},
            n_observations=1000,
        ),
        "feature_2": FeatureICResults(
            feature="feature_2",
            ic_mean=-0.08,
            ic_std=0.06,
            ic_ir=-1.33,
            t_stat=-1.5,
            p_value=0.134,
            ic_by_lag={0: -0.07, 1: -0.08, 5: -0.09, 10: -0.08, 21: -0.07},
            n_observations=1000,
        ),
        "feature_3": FeatureICResults(
            feature="feature_3",
            ic_mean=0.22,
            ic_std=0.04,
            ic_ir=5.5,
            t_stat=4.2,
            p_value=0.001,
            ic_by_lag={0: 0.20, 1: 0.22, 5: 0.24, 10: 0.21, 21: 0.23},
            n_observations=1000,
        ),
        "feature_4": FeatureICResults(
            feature="feature_4",
            ic_mean=0.05,
            ic_std=0.10,
            ic_ir=0.5,
            t_stat=0.4,
            p_value=0.689,
            ic_by_lag={0: 0.03, 1: 0.05, 5: 0.07, 10: 0.04, 21: 0.06},
            n_observations=1000,
        ),
    }


@pytest.fixture
def sample_ic_dataframe() -> pd.DataFrame:
    """Create sample IC results as DataFrame."""
    return pd.DataFrame(
        [
            {
                "feature": "feature_1",
                "ic_mean": 0.15,
                "ic_std": 0.05,
                "ic_ir": 3.0,
                "t_stat": 2.5,
                "p_value": 0.012,
                "n_observations": 1000,
            },
            {
                "feature": "feature_2",
                "ic_mean": -0.08,
                "ic_std": 0.06,
                "ic_ir": -1.33,
                "t_stat": -1.5,
                "p_value": 0.134,
                "n_observations": 1000,
            },
            {
                "feature": "feature_3",
                "ic_mean": 0.22,
                "ic_std": 0.04,
                "ic_ir": 5.5,
                "t_stat": 4.2,
                "p_value": 0.001,
                "n_observations": 1000,
            },
        ]
    )


class TestPlotICTimeSeries:
    """Tests for plot_ic_time_series function."""

    def test_basic_plot_dict(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test basic IC time series plot with dict input."""
        fig = plot_ic_time_series(sample_ic_results, "feature_1")

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        ax = fig.axes[0]
        assert ax.get_xlabel() == "Forward Horizon (lag)"
        assert ax.get_ylabel() == "Information Coefficient"
        assert "IC Time Series: feature_1" in ax.get_title()

    def test_plot_with_dataframe(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test IC time series plot with DataFrame input."""
        # Convert to DataFrame
        rows = []
        for feature, result in sample_ic_results.items():
            rows.append(
                {
                    "feature": feature,
                    "ic_mean": result.ic_mean,
                    "ic_std": result.ic_std,
                    "ic_ir": result.ic_ir,
                    "ic_by_lag": result.ic_by_lag,
                }
            )
        df = pd.DataFrame(rows)

        fig = plot_ic_time_series(df, "feature_1")
        assert isinstance(fig, Figure)

    def test_custom_styling(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test custom styling options."""
        fig = plot_ic_time_series(
            sample_ic_results,
            "feature_1",
            figsize=(14, 7),
            title="Custom IC Plot",
            color="coral",
            show_mean=False,
            show_bands=False,
        )

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_title() == "Custom IC Plot"
        assert fig.get_figwidth() == 14
        assert fig.get_figheight() == 7

    def test_with_existing_axes(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test plotting on existing axes."""
        fig, ax = plt.subplots()
        result_fig = plot_ic_time_series(sample_ic_results, "feature_1", ax=ax)

        assert result_fig is fig
        assert len(fig.axes) == 1

    def test_missing_feature(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test error handling for missing feature."""
        with pytest.raises(ValueError, match="Feature 'nonexistent' not found"):
            plot_ic_time_series(sample_ic_results, "nonexistent")

    def test_missing_ic_by_lag(self) -> None:
        """Test error handling when ic_by_lag is empty."""
        # Create result without ic_by_lag
        bad_results = {
            "feature_1": FeatureICResults(
                feature="feature_1",
                ic_mean=0.15,
                ic_std=0.05,
                ic_ir=3.0,
                ic_by_lag={},  # Empty
            )
        }

        with pytest.raises(ValueError, match="No IC by lag data available"):
            plot_ic_time_series(bad_results, "feature_1")

    def test_mean_line_display(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test that mean line is displayed correctly."""
        fig = plot_ic_time_series(sample_ic_results, "feature_1", show_mean=True)
        ax = fig.axes[0]

        # Check for horizontal lines (mean line)
        hlines = [line for line in ax.get_lines() if hasattr(line, "get_linestyle")]
        assert any("--" in line.get_linestyle() for line in hlines)

    def test_confidence_bands(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test that confidence bands are displayed."""
        fig = plot_ic_time_series(sample_ic_results, "feature_1", show_bands=True)
        ax = fig.axes[0]

        # Check for filled areas (confidence bands)
        assert len(ax.collections) > 0  # PolyCollections for filled areas

    def test_negative_ic_values(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test plotting with negative IC values."""
        fig = plot_ic_time_series(sample_ic_results, "feature_2")
        assert isinstance(fig, Figure)

        # Check that plot was created successfully
        ax = fig.axes[0]
        assert len(ax.get_lines()) > 0

    def test_plot_data_values(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test that plotted data matches input values."""
        result = sample_ic_results["feature_1"]
        fig = plot_ic_time_series(sample_ic_results, "feature_1")
        ax = fig.axes[0]

        # Get the line data
        line = ax.get_lines()[0]
        xdata = line.get_xdata()
        ydata = line.get_ydata()

        # Check that values match
        expected_lags = sorted(result.ic_by_lag.keys())
        expected_values = [result.ic_by_lag[lag] for lag in expected_lags]

        np.testing.assert_array_equal(xdata, expected_lags)
        np.testing.assert_array_almost_equal(ydata, expected_values)


class TestPlotICComparison:
    """Tests for plot_ic_comparison function."""

    def test_basic_comparison_dict(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test basic IC comparison plot with dict input."""
        fig = plot_ic_comparison(sample_ic_results)

        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

        ax = fig.axes[0]
        assert "IC Comparison" in ax.get_title()

    def test_comparison_with_dataframe(self, sample_ic_dataframe: pd.DataFrame) -> None:
        """Test IC comparison with DataFrame input."""
        fig = plot_ic_comparison(sample_ic_dataframe)
        assert isinstance(fig, Figure)

    def test_top_n_selection(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test top N feature selection."""
        fig = plot_ic_comparison(sample_ic_results, top_n=2)
        ax = fig.axes[0]

        # Should have 2 bars
        assert len(ax.patches) == 2

    def test_specific_features(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test plotting specific features."""
        features = ["feature_1", "feature_3"]
        fig = plot_ic_comparison(sample_ic_results, features=features)
        ax = fig.axes[0]

        # Should have 2 bars
        assert len(ax.patches) == 2

    def test_sort_by_metric(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test sorting by different metrics."""
        # Sort by IC mean
        fig1 = plot_ic_comparison(sample_ic_results, sort_by="ic_mean")
        assert isinstance(fig1, Figure)

        # Sort by IC IR
        fig2 = plot_ic_comparison(sample_ic_results, sort_by="ic_ir")
        assert isinstance(fig2, Figure)

        # Sort by IC std
        fig3 = plot_ic_comparison(sample_ic_results, sort_by="ic_std")
        assert isinstance(fig3, Figure)

    def test_invalid_sort_by(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test error handling for invalid sort_by."""
        with pytest.raises(ValueError, match="sort_by must be one of"):
            plot_ic_comparison(sample_ic_results, sort_by="invalid_metric")

    def test_custom_styling(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test custom styling options."""
        fig = plot_ic_comparison(sample_ic_results, figsize=(14, 10), title="Custom IC Comparison")

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_title() == "Custom IC Comparison"
        assert fig.get_figwidth() == 14
        assert fig.get_figheight() == 10

    def test_color_by_sign(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test that bars are colored by IC sign."""
        fig = plot_ic_comparison(sample_ic_results)
        ax = fig.axes[0]

        # Get bar colors
        colors = [patch.get_facecolor() for patch in ax.patches]

        # We should have both blue and red bars (positive and negative IC)
        # Since we have features with positive and negative IC
        assert len(colors) > 0
        assert len({tuple(c) for c in colors}) > 1  # Multiple colors used

    def test_error_bars_present(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test that error bars are displayed."""
        fig = plot_ic_comparison(sample_ic_results)
        ax = fig.axes[0]

        # Check for error bars (LineCollection objects)
        assert len(ax.collections) > 0

    def test_missing_features(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test error handling for missing features."""
        with pytest.raises(ValueError, match="No matching features found"):
            plot_ic_comparison(sample_ic_results, features=["nonexistent"])

    def test_legend_present(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test that legend is displayed."""
        fig = plot_ic_comparison(sample_ic_results)
        ax = fig.axes[0]

        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2  # Positive and Negative IC

    def test_zero_line_present(self, sample_ic_results: dict[str, FeatureICResults]) -> None:
        """Test that zero reference line is present."""
        fig = plot_ic_comparison(sample_ic_results)
        ax = fig.axes[0]

        # Check for vertical lines (zero line)
        vlines = list(ax.get_lines())
        assert len(vlines) > 0


class TestICPlottingEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_results_time_series(self) -> None:
        """Test error handling for empty results."""
        empty_results: dict[str, FeatureICResults] = {}

        with pytest.raises(ValueError, match="not found"):
            plot_ic_time_series(empty_results, "feature_1")

    def test_empty_results_comparison(self) -> None:
        """Test error handling for empty results."""
        empty_df = pd.DataFrame(columns=["feature", "ic_mean", "ic_std", "ic_ir"])

        with pytest.raises(ValueError, match="No matching features found"):
            plot_ic_comparison(empty_df, features=["feature_1"])

    def test_single_feature_comparison(
        self, sample_ic_results: dict[str, FeatureICResults]
    ) -> None:
        """Test comparison with single feature."""
        single = {"feature_1": sample_ic_results["feature_1"]}
        fig = plot_ic_comparison(single)

        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert len(ax.patches) == 1

    def test_missing_columns_dataframe(self) -> None:
        """Test error handling for DataFrame with missing columns."""
        bad_df = pd.DataFrame([{"feature": "f1"}])

        with pytest.raises(ValueError, match="missing required columns"):
            plot_ic_comparison(bad_df)

    def test_close_figures_after_test(self) -> None:
        """Ensure figures are closed to prevent memory leaks."""
        # This test just ensures matplotlib cleanup works
        plt.close("all")
        assert len(plt.get_fignums()) == 0


class TestICDataConversion:
    """Tests for data conversion helper functions."""

    def test_dict_to_dataframe_conversion(
        self, sample_ic_results: dict[str, FeatureICResults]
    ) -> None:
        """Test conversion from dict to DataFrame."""
        from ml4t.engineer.outcome.plot_ic import _ic_dict_to_dataframe

        df = _ic_dict_to_dataframe(sample_ic_results)

        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert len(df) == len(sample_ic_results)
        assert all(col in df.columns for col in ["ic_mean", "ic_std", "ic_ir"])

    def test_dataframe_to_dict_conversion(self, sample_ic_dataframe: pd.DataFrame) -> None:
        """Test conversion from DataFrame to dict."""
        from ml4t.engineer.outcome.plot_ic import _ic_dataframe_to_dict

        # Add ic_by_lag column
        sample_ic_dataframe["ic_by_lag"] = [{0: 0.1, 1: 0.15}] * len(sample_ic_dataframe)

        results = _ic_dataframe_to_dict(sample_ic_dataframe)

        assert isinstance(results, dict)
        assert len(results) == len(sample_ic_dataframe)
        assert all(isinstance(v, FeatureICResults) for v in results.values())


# Cleanup after all tests
def teardown_module() -> None:
    """Clean up after all tests."""
    plt.close("all")
