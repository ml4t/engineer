"""Cross-feature relationship analysis (Module B).

This module provides tools for analyzing relationships between features:
- Correlation matrices (Pearson, Spearman, Kendall)
- Correlation heatmap visualization
- Multicollinearity detection
- Feature clustering

Example:
    >>> from ml4t.engineer.relationships import compute_correlation_matrix, plot_correlation_heatmap
    >>> import polars as pl
    >>>
    >>> df = pl.DataFrame({
    ...     "feature1": [1, 2, 3, 4, 5],
    ...     "feature2": [2, 4, 6, 8, 10],
    ...     "feature3": [1, 3, 2, 5, 4]
    ... })
    >>>
    >>> # Compute correlation
    >>> corr = compute_correlation_matrix(df, method="pearson")
    >>> print(corr)
    >>>
    >>> # Visualize as heatmap
    >>> fig = plot_correlation_heatmap(corr, threshold=0.5)
    >>> fig.savefig("correlation_heatmap.png")
"""

from ml4t.engineer.relationships.correlation import compute_correlation_matrix
from ml4t.engineer.relationships.plot_correlation import plot_correlation_heatmap

__all__ = ["compute_correlation_matrix", "plot_correlation_heatmap"]
