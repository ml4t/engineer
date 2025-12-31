# mypy: disable-error-code="arg-type,assignment"
"""Correlation matrix visualization.

This module provides plotting functions for correlation matrices.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure


def plot_correlation_heatmap(
    corr_matrix: pl.DataFrame,
    threshold: float | None = None,
    cmap: str = "RdBu_r",
    figsize: tuple[int, int] = (10, 8),
    title: str | None = None,
    annot: bool = True,
    fmt: str = ".2f",
    ax: Axes | None = None,
) -> Figure | SubFigure:
    """Plot correlation matrix as heatmap.

    Creates a color-coded heatmap visualization of a correlation matrix with
    optional value annotations and threshold filtering.

    Parameters
    ----------
    corr_matrix : pl.DataFrame
        Correlation matrix from compute_correlation_matrix().
        Must have "feature" column plus correlation columns.
    threshold : float | None, default None
        Optional threshold to highlight strong correlations.
        If provided, correlations with |value| < threshold are dimmed.
    cmap : str, default "RdBu_r"
        Matplotlib colormap name. Default is diverging red-blue (reversed)
        where red = negative correlation, blue = positive correlation.
    figsize : tuple[int, int], default (10, 8)
        Figure size in inches (width, height).
    title : str | None, default None
        Plot title. If None, uses "Correlation Matrix".
    annot : bool, default True
        Whether to annotate cells with correlation values.
    fmt : str, default ".2f"
        Format string for annotations (e.g., ".2f" for 2 decimal places).
    ax : Axes | None, default None
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    Figure
        Matplotlib figure object.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If corr_matrix doesn't have expected structure.

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.relationships import compute_correlation_matrix, plot_correlation_heatmap
    >>>
    >>> # Create sample data
    >>> df = pl.DataFrame({
    ...     "returns": [0.01, -0.02, 0.03, -0.01, 0.02],
    ...     "volume": [1000, 1500, 1200, 1300, 1100],
    ...     "volatility": [0.15, 0.25, 0.18, 0.20, 0.16]
    ... })
    >>>
    >>> # Compute correlation and plot
    >>> corr = compute_correlation_matrix(df, method="pearson")
    >>> fig = plot_correlation_heatmap(corr, threshold=0.3)
    >>> fig.savefig("correlation_heatmap.png")
    >>>
    >>> # Custom styling
    >>> fig = plot_correlation_heatmap(
    ...     corr,
    ...     cmap="coolwarm",
    ...     figsize=(12, 10),
    ...     title="Feature Correlations",
    ...     annot=False  # No value annotations
    ... )

    Notes
    -----
    **Interpretation**:

    - **Color intensity**: Stronger correlation (closer to -1 or +1)
    - **Red**: Negative correlation (variables move in opposite directions)
    - **Blue**: Positive correlation (variables move together)
    - **White**: No correlation (independent variables)

    **Threshold Filtering**:

    When threshold is provided, correlations with |r| < threshold are shown
    with reduced alpha (transparency), making strong correlations stand out.

    **Recommended Colormaps**:

    - "RdBu_r": Red-blue diverging (default, good for correlations)
    - "coolwarm": Cool-warm diverging
    - "seismic": Seismic diverging
    - "bwr": Blue-white-red

    **Plot Customization**:

    The returned figure can be further customized:

    >>> fig = plot_correlation_heatmap(corr)
    >>> fig.suptitle("My Custom Title", fontsize=16)
    >>> fig.tight_layout()
    >>> fig.savefig("corr.png", dpi=300, bbox_inches="tight")

    References
    ----------
    - Matplotlib heatmaps: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    # Check matplotlib availability
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e

    # Validate input structure
    if "feature" not in corr_matrix.columns:
        raise ValueError("corr_matrix must have 'feature' column")

    # Convert to pandas for easier manipulation with matplotlib
    corr_pd = corr_matrix.to_pandas().set_index("feature")

    # Validate it's a square matrix
    if corr_pd.shape[0] != corr_pd.shape[1]:
        raise ValueError(f"Correlation matrix must be square. Got shape {corr_pd.shape}")

    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_result = ax.get_figure()
        if fig_result is None:
            raise ValueError("Axes must have an associated figure")
        fig = fig_result

    # Get correlation values as numpy array
    corr_values = corr_pd.values

    # Create heatmap using imshow
    im = ax.imshow(corr_values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Set ticks and labels
    feature_names = corr_pd.index.tolist()
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label="Correlation")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    # Add value annotations if requested
    if annot:
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                value = corr_values[i, j]

                # Skip NaN values
                if np.isnan(value):
                    text = "NaN"
                    color = "gray"
                else:
                    text = f"{value:{fmt}}"

                    # Apply threshold filtering if specified
                    if threshold is not None and abs(value) < threshold:
                        color = "lightgray"
                    else:
                        # Choose text color based on background
                        color = "white" if abs(value) > 0.5 else "black"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

    # Set title
    if title is None:
        title = "Correlation Matrix"
    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")

    # Improve layout (suppress warning for incompatible axes)
    # Note: SubFigure doesn't have tight_layout, but we only get SubFigure
    # when ax was passed in, and the parent figure handles layout
    from matplotlib.figure import Figure as MplFigure

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="This figure includes Axes that are not compatible"
        )
        if isinstance(fig, MplFigure):
            fig.tight_layout()

    return fig
