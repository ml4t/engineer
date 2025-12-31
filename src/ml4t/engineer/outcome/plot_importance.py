"""Feature importance visualization.

This module provides plotting functions for feature importance analysis.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ml4t.engineer.outcome.feature_outcome import FeatureImportanceResults


def plot_feature_importance(
    importance_results: dict[str, FeatureImportanceResults] | pd.DataFrame,
    importance_type: Literal["mdi", "permutation", "shap"] = "mdi",
    top_n: int | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
    color: str = "steelblue",
    error_bars: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plot feature importance as horizontal bar chart.

    Creates a horizontal bar chart showing feature importance values with
    optional error bars (for permutation and SHAP methods).

    Parameters
    ----------
    importance_results : dict[str, FeatureImportanceResults] | pd.DataFrame
        Feature importance results. Can be either:
        - Dictionary mapping feature names to FeatureImportanceResults objects
        - DataFrame with 'feature' column and importance columns
    importance_type : {"mdi", "permutation", "shap"}, default "mdi"
        Type of importance to plot:
        - "mdi": Mean Decrease in Impurity (tree-based models)
        - "permutation": Permutation importance
        - "shap": Mean absolute SHAP values
    top_n : int | None, default None
        Number of top features to display. If None, shows all features.
    figsize : tuple[int, int], default (10, 6)
        Figure size in inches (width, height).
    title : str | None, default None
        Plot title. If None, uses "Feature Importance ({importance_type})".
    color : str, default "steelblue"
        Bar color.
    error_bars : bool, default True
        Whether to show error bars (only for permutation and SHAP).
        Uses permutation_std for permutation importance, shap_std for SHAP.
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
        If importance_type is invalid or required data is missing.

    Examples
    --------
    >>> from ml4t.engineer.outcome import FeatureOutcome, plot_feature_importance
    >>> import polars as pl
    >>>
    >>> # Create sample data
    >>> features = pl.DataFrame({
    ...     "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5],
    ...     "feature_2": [0.5, 0.4, 0.3, 0.2, 0.1],
    ... })
    >>> outcomes = pl.DataFrame({"returns": [0.01, -0.02, 0.03, -0.01, 0.02]})
    >>>
    >>> # Analyze features
    >>> analyzer = FeatureOutcome()
    >>> results = analyzer.run_analysis(features, outcomes)
    >>>
    >>> # Plot MDI importance
    >>> fig = plot_feature_importance(results.importance_results, importance_type="mdi")
    >>>
    >>> # Plot top 10 by permutation importance with error bars
    >>> fig = plot_feature_importance(
    ...     results.importance_results,
    ...     importance_type="permutation",
    ...     top_n=10,
    ...     error_bars=True
    ... )
    >>>
    >>> # Plot SHAP importance
    >>> fig = plot_feature_importance(
    ...     results.importance_results,
    ...     importance_type="shap",
    ...     color="coral"
    ... )

    Notes
    -----
    - Features are sorted by importance value (highest at top)
    - Error bars show ±1 standard deviation
    - For SHAP, requires shap_mean to be computed in importance_results
    - For permutation, shows permutation_std if available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib is required for plotting. Install with: pip install matplotlib"
        raise ImportError(msg) from e

    # Validate importance type
    valid_types = {"mdi", "permutation", "shap"}
    if importance_type not in valid_types:
        msg = f"importance_type must be one of {valid_types}, got {importance_type}"
        raise ValueError(msg)

    # Convert to DataFrame if dict
    if isinstance(importance_results, dict):
        df = _importance_dict_to_dataframe(importance_results)
    else:
        df = importance_results.copy()

    # Check required columns
    if "feature" not in df.columns:
        msg = "DataFrame must have 'feature' column"
        raise ValueError(msg)

    # Get importance column and std column (if applicable)
    if importance_type == "mdi":
        value_col = "mdi_importance"
        std_col = None
    elif importance_type == "permutation":
        value_col = "permutation_importance"
        std_col = "permutation_std" if error_bars else None
    else:  # shap
        value_col = "shap_mean"
        std_col = "shap_std" if error_bars else None

    if value_col not in df.columns:
        msg = f"DataFrame missing '{value_col}' column for {importance_type} importance"
        raise ValueError(msg)

    # Filter out NaN values
    df = df[df[value_col].notna()].copy()

    if len(df) == 0:
        msg = f"No valid {importance_type} importance values found"
        raise ValueError(msg)

    # Sort by importance (descending)
    df = df.sort_values(value_col, ascending=False)

    # Select top N if specified
    if top_n is not None:
        df = df.head(top_n)

    # Reverse order for horizontal bar chart (highest at top)
    df = df.iloc[::-1]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig_result = ax.get_figure()
        if fig_result is None:
            msg = "Axes must have an associated Figure"
            raise ValueError(msg)
        # Use cast since we know ax is a standard Axes with a Figure parent
        fig = cast("Figure", fig_result)

    # Get values
    features = df["feature"].values
    values = df[value_col].values

    # Get error bars if applicable
    yerr = None
    if std_col is not None and std_col in df.columns:
        yerr = df[std_col].values

    # Create horizontal bar chart
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=color, xerr=yerr, capsize=3)

    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)

    # Set labels and title
    importance_labels = {
        "mdi": "MDI Importance",
        "permutation": "Permutation Importance",
        "shap": "Mean |SHAP|",
    }
    ax.set_xlabel(importance_labels[importance_type])

    if title is None:
        title = f"Feature Importance ({importance_type.upper()})"
    ax.set_title(title)

    # Add grid for readability
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Tight layout (suppress warning for incompatible axes)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="This figure includes Axes that are not compatible"
        )
        fig.tight_layout()

    return fig


def plot_importance_comparison(
    importance_results: dict[str, FeatureImportanceResults] | pd.DataFrame,
    top_n: int = 10,
    figsize: tuple[int, int] = (12, 8),
    title: str | None = None,
) -> Figure:
    """Plot comparison of different importance metrics side-by-side.

    Creates a grouped bar chart comparing MDI, permutation, and SHAP importance
    for the same set of features.

    Parameters
    ----------
    importance_results : dict[str, FeatureImportanceResults] | pd.DataFrame
        Feature importance results.
    top_n : int, default 10
        Number of top features to display (by average rank across metrics).
    figsize : tuple[int, int], default (12, 8)
        Figure size in inches (width, height).
    title : str | None, default None
        Plot title. If None, uses "Feature Importance Comparison".

    Returns
    -------
    Figure
        Matplotlib figure object with comparison plot.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If required data is missing.

    Examples
    --------
    >>> from ml4t.engineer.outcome import FeatureOutcome, plot_importance_comparison
    >>> import polars as pl
    >>>
    >>> # Analyze features
    >>> analyzer = FeatureOutcome()
    >>> results = analyzer.run_analysis(features_df, outcomes_df)
    >>>
    >>> # Compare all importance types
    >>> fig = plot_importance_comparison(results.importance_results, top_n=15)

    Notes
    -----
    - Features are selected based on average rank across all available metrics
    - Missing values are shown as zero-height bars
    - All importance values are normalized to [0, 1] range for comparison
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib is required for plotting. Install with: pip install matplotlib"
        raise ImportError(msg) from e

    # Convert to DataFrame if dict
    if isinstance(importance_results, dict):
        df = _importance_dict_to_dataframe(importance_results)
    else:
        df = importance_results.copy()

    # Check required columns
    if "feature" not in df.columns:
        msg = "DataFrame must have 'feature' column"
        raise ValueError(msg)

    # Ensure we have at least one importance metric
    has_mdi = "mdi_importance" in df.columns and df["mdi_importance"].notna().any()
    has_perm = "permutation_importance" in df.columns and df["permutation_importance"].notna().any()
    has_shap = "shap_mean" in df.columns and df["shap_mean"].notna().any()

    if not (has_mdi or has_perm or has_shap):
        msg = "No valid importance metrics found"
        raise ValueError(msg)

    # Select top N features by average rank
    ranks = []
    if has_mdi:
        ranks.append(df["rank_mdi"])
    if has_perm:
        ranks.append(df["rank_permutation"])

    if ranks:
        # Average rank (lower is better)
        df["avg_rank"] = pd.concat(ranks, axis=1).mean(axis=1)
        df = df.sort_values("avg_rank").head(top_n)
    else:
        # Fallback: sort by first available metric
        if has_mdi:
            df = df.sort_values("mdi_importance", ascending=False).head(top_n)
        elif has_perm:
            df = df.sort_values("permutation_importance", ascending=False).head(top_n)
        else:
            df = df.sort_values("shap_mean", ascending=False).head(top_n)

    # Reverse order for plotting (highest at top)
    df = df.iloc[::-1]

    # Normalize importance values to [0, 1] for comparison
    metrics = []
    labels = []
    colors = []

    if has_mdi:
        mdi_values = df["mdi_importance"].fillna(0).values
        if mdi_values.max() > 0:
            mdi_values = mdi_values / mdi_values.max()
        metrics.append(mdi_values)
        labels.append("MDI")
        colors.append("steelblue")

    if has_perm:
        perm_values = df["permutation_importance"].fillna(0).values
        if perm_values.max() > 0:
            perm_values = perm_values / perm_values.max()
        metrics.append(perm_values)
        labels.append("Permutation")
        colors.append("coral")

    if has_shap:
        shap_values = df["shap_mean"].fillna(0).values
        if shap_values.max() > 0:
            shap_values = shap_values / shap_values.max()
        metrics.append(shap_values)
        labels.append("SHAP")
        colors.append("mediumseagreen")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set up bar positions
    features = df["feature"].values
    y_pos = np.arange(len(features))
    bar_height = 0.25
    n_metrics = len(metrics)

    # Plot grouped bars
    for i, (values, label, color) in enumerate(zip(metrics, labels, colors)):
        offset = (i - n_metrics / 2 + 0.5) * bar_height
        ax.barh(y_pos + offset, values, bar_height, label=label, color=color, alpha=0.8)

    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)

    # Set labels and title
    ax.set_xlabel("Normalized Importance")
    if title is None:
        title = "Feature Importance Comparison"
    ax.set_title(title)

    # Add legend
    ax.legend(loc="lower right")

    # Add grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Tight layout
    fig.tight_layout()

    return fig


def _importance_dict_to_dataframe(
    importance_results: dict[str, FeatureImportanceResults],
) -> pd.DataFrame:
    """Convert importance results dict to DataFrame.

    Parameters
    ----------
    importance_results : dict[str, FeatureImportanceResults]
        Dictionary mapping feature names to importance results.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per feature.
    """
    rows = []
    for feature, result in importance_results.items():
        row = {
            "feature": feature,
            "mdi_importance": result.mdi_importance,
            "permutation_importance": result.permutation_importance,
            "permutation_std": result.permutation_std,
            "shap_mean": result.shap_mean,
            "shap_std": result.shap_std,
            "rank_mdi": result.rank_mdi,
            "rank_permutation": result.rank_permutation,
        }
        rows.append(row)

    return pd.DataFrame(rows)
