"""Unified feature analysis visualization.

This module provides comprehensive summary visualizations combining:
- Feature importance analysis
- Information Coefficient (IC) analysis
- Feature correlation analysis

All in a single, publication-ready figure.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ml4t.engineer.outcome.feature_outcome import (
        FeatureImportanceResults,
        FeatureOutcomeResult,
    )


def plot_feature_analysis_summary(
    results: FeatureOutcomeResult,
    correlation_matrix: pl.DataFrame | None = None,
    top_n: int = 10,
    figsize: tuple[int, int] = (16, 12),
    title: str | None = None,
    importance_type: Literal["mdi", "permutation", "shap"] = "mdi",
) -> Figure:
    """Create unified 3-panel feature analysis summary.

    Combines feature importance, IC comparison, and correlation analysis into
    a single comprehensive visualization.

    Parameters
    ----------
    results : FeatureOutcomeResult
        Results from FeatureOutcome.run_analysis() containing IC and importance data.
    correlation_matrix : pl.DataFrame | None, default None
        Correlation matrix from compute_correlation_matrix(). If None, correlation
        panel is omitted.
    top_n : int, default 10
        Number of top features to show in each panel.
    figsize : tuple[int, int], default (16, 12)
        Figure size in inches (width, height).
    title : str | None, default None
        Overall figure title. If None, uses "Feature Analysis Summary".
    importance_type : {"mdi", "permutation", "shap"}, default "mdi"
        Type of feature importance to display in panel 1.

    Returns
    -------
    Figure
        Matplotlib figure with 3 subplots (or 2 if correlation_matrix is None).

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If results are missing required data.

    Examples
    --------
    >>> from ml4t.engineer.outcome import FeatureOutcome
    >>> from ml4t.engineer.relationships import compute_correlation_matrix
    >>> from ml4t.engineer.visualization import plot_feature_analysis_summary
    >>> import polars as pl
    >>>
    >>> # Run analysis
    >>> analyzer = FeatureOutcome()
    >>> results = analyzer.run_analysis(features_df, outcomes_df)
    >>>
    >>> # Compute correlation
    >>> corr = compute_correlation_matrix(features_df)
    >>>
    >>> # Create summary plot
    >>> fig = plot_feature_analysis_summary(results, corr, top_n=15)
    >>> fig.savefig("feature_analysis_summary.png", dpi=300)
    >>>
    >>> # Without correlation
    >>> fig = plot_feature_analysis_summary(results, top_n=20)

    Notes
    -----
    **Panel Layout**:

    - **Panel 1 (Top Left)**: Feature importance ranking (top N features)
    - **Panel 2 (Top Right)**: IC comparison (top N by |IC mean|)
    - **Panel 3 (Bottom)**: Correlation heatmap (top N features by importance)

    **Interpretation**:

    - Features appearing in both importance and IC panels are strong candidates
    - Highly correlated features (in panel 3) may be redundant
    - Low IC but high importance suggests overfitting risk

    **Performance**: Generates plot in <5 seconds for 50 features.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib is required for plotting. Install with: pip install matplotlib"
        raise ImportError(msg) from e

    # Import plotting functions from existing modules
    from ml4t.engineer.outcome.plot_ic import plot_ic_comparison
    from ml4t.engineer.outcome.plot_importance import plot_feature_importance
    from ml4t.engineer.relationships.plot_correlation import plot_correlation_heatmap

    # Determine subplot layout
    n_panels = 3 if correlation_matrix is not None else 2
    if n_panels == 3:
        # 2x2 grid with bottom row spanning full width
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_importance = fig.add_subplot(gs[0, 0])
        ax_ic = fig.add_subplot(gs[0, 1])
        ax_corr = fig.add_subplot(gs[1, :])
        axes = [ax_importance, ax_ic, ax_corr]
    else:
        # 1x2 grid
        fig, axes_array = plt.subplots(1, 2, figsize=(figsize[0], figsize[1] // 2))
        axes = axes_array.tolist()  # Convert to list for consistent indexing
        ax_importance = axes[0]
        ax_ic = axes[1]

    # Panel 1: Feature Importance
    try:
        plot_feature_importance(
            results.importance_results,
            importance_type=importance_type,
            top_n=top_n,
            ax=ax_importance,
            error_bars=True,
        )
    except (ValueError, KeyError) as e:
        # If importance data is missing, show empty panel with message
        ax_importance.text(
            0.5,
            0.5,
            f"Feature importance data not available\n({e})",
            ha="center",
            va="center",
            transform=ax_importance.transAxes,
        )
        ax_importance.set_title("Feature Importance")

    # Panel 2: IC Comparison
    try:
        plot_ic_comparison(
            results.ic_results,
            top_n=top_n,
            ax=ax_ic,
            sort_by="ic_mean",
        )
    except (ValueError, KeyError) as e:
        # If IC data is missing, show empty panel with message
        ax_ic.text(
            0.5,
            0.5,
            f"IC data not available\n({e})",
            ha="center",
            va="center",
            transform=ax_ic.transAxes,
        )
        ax_ic.set_title("IC Comparison")

    # Panel 3: Correlation Heatmap (if provided)
    if correlation_matrix is not None:
        try:
            # Select top N features by importance for correlation view
            top_features = _get_top_features_by_importance(
                results.importance_results, top_n, importance_type
            )

            # Filter correlation matrix to top features
            filtered_corr = _filter_correlation_matrix(correlation_matrix, top_features)

            plot_correlation_heatmap(
                filtered_corr,
                ax=ax_corr,
                figsize=(figsize[0], figsize[1] // 2),
                title=f"Correlation Heatmap (Top {len(top_features)} Features)",
            )
        except (ValueError, KeyError) as e:
            # If correlation data is missing, show empty panel with message
            ax_corr.text(
                0.5,
                0.5,
                f"Correlation data not available\n({e})",
                ha="center",
                va="center",
                transform=ax_corr.transAxes,
            )
            ax_corr.set_title("Correlation Heatmap")

    # Set overall title
    if title is None:
        title = "Feature Analysis Summary"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Adjust layout to prevent overlap (suppress warning for incompatible axes)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="This figure includes Axes that are not compatible"
        )
        fig.tight_layout()

    return fig


def export_plot(
    fig: Figure,
    output_path: str | Path,
    dpi: int = 300,
    bbox_inches: str = "tight",
    **kwargs: Any,
) -> None:
    """Export matplotlib figure to file.

    Saves figure to PNG or PDF with configurable quality settings.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    output_path : str | Path
        Output file path. Format determined by extension (.png, .pdf, etc.).
    dpi : int, default 300
        Resolution in dots per inch. Higher values = better quality but larger files.
        - 150: Draft quality
        - 300: Publication quality (default)
        - 600: High-resolution print
    bbox_inches : str, default "tight"
        Bounding box specification. "tight" removes extra whitespace.
    **kwargs
        Additional keyword arguments passed to fig.savefig().

    Raises
    ------
    ValueError
        If output_path has unsupported extension.
    OSError
        If file cannot be written (permissions, disk space, etc.).

    Examples
    --------
    >>> from ml4t.engineer.visualization import plot_feature_analysis_summary, export_plot
    >>>
    >>> # Create and export plot
    >>> fig = plot_feature_analysis_summary(results, corr)
    >>> export_plot(fig, "analysis.png", dpi=300)
    >>>
    >>> # High-quality PDF
    >>> export_plot(fig, "analysis.pdf", dpi=600)
    >>>
    >>> # Draft quality for quick review
    >>> export_plot(fig, "draft.png", dpi=150)
    >>>
    >>> # Custom options
    >>> export_plot(
    ...     fig,
    ...     "custom.png",
    ...     dpi=300,
    ...     facecolor="white",
    ...     edgecolor="none"
    ... )

    Notes
    -----
    **Supported Formats**:

    - PNG: Recommended for web/presentations (lossless compression)
    - PDF: Best for publications (vector graphics, scalable)
    - SVG: Web-friendly vector format
    - JPEG: Smaller files but lossy compression (not recommended for plots)

    **File Sizes** (approximate for typical 3-panel summary):

    - PNG @ 150 DPI: ~500 KB
    - PNG @ 300 DPI: ~1.5 MB
    - PNG @ 600 DPI: ~5 MB
    - PDF @ 300 DPI: ~200 KB (vector)
    """
    output_path = Path(output_path)

    # Validate extension
    valid_extensions = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps", ".ps"}
    if output_path.suffix.lower() not in valid_extensions:
        msg = f"Unsupported format: {output_path.suffix}. Use one of {valid_extensions}"
        raise ValueError(msg)

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    except OSError as e:
        msg = f"Failed to save figure to {output_path}: {e}"
        raise OSError(msg) from e


def _get_top_features_by_importance(
    importance_results: dict[str, FeatureImportanceResults],
    top_n: int,
    importance_type: str,
) -> list[str]:
    """Get top N features by importance metric.

    Parameters
    ----------
    importance_results : dict
        Feature importance results dictionary.
    top_n : int
        Number of top features to return.
    importance_type : str
        Type of importance: "mdi", "permutation", or "shap".

    Returns
    -------
    list[str]
        List of feature names, sorted by importance (descending).
    """

    from ml4t.engineer.outcome.plot_importance import _importance_dict_to_dataframe

    # Convert to DataFrame
    df = _importance_dict_to_dataframe(importance_results)

    # Get appropriate column
    if importance_type == "mdi":
        col = "mdi_importance"
    elif importance_type == "permutation":
        col = "permutation_importance"
    else:  # shap
        col = "shap_mean"

    # Filter out NaN and sort
    df = df[df[col].notna()].copy()
    df = df.sort_values(col, ascending=False)

    # Return top N feature names
    top_features: list[str] = df.head(top_n)["feature"].tolist()
    return top_features


def _filter_correlation_matrix(
    corr_matrix: pl.DataFrame,
    features: list[str],
) -> pl.DataFrame:
    """Filter correlation matrix to specified features.

    Parameters
    ----------
    corr_matrix : pl.DataFrame
        Full correlation matrix with "feature" column.
    features : list[str]
        Features to keep.

    Returns
    -------
    pl.DataFrame
        Filtered correlation matrix.
    """
    # Filter rows
    filtered = corr_matrix.filter(pl.col("feature").is_in(features))

    # Filter columns (keep "feature" column + feature columns)
    cols_to_keep = ["feature"] + [f for f in features if f in filtered.columns]
    filtered = filtered.select(cols_to_keep)

    return filtered
