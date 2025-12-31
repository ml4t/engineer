"""Information Coefficient (IC) visualization.

This module provides plotting functions for IC analysis.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from ml4t.engineer.outcome.feature_outcome import FeatureICResults


def plot_ic_time_series(
    ic_results: dict[str, FeatureICResults] | pd.DataFrame,
    feature: str,
    figsize: tuple[int, int] = (12, 6),
    title: str | None = None,
    color: str = "steelblue",
    show_mean: bool = True,
    show_bands: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plot IC time series for a single feature.

    Creates a line plot showing IC values over different forward horizons (lags)
    with optional mean line and confidence bands.

    Parameters
    ----------
    ic_results : dict[str, FeatureICResults] | pd.DataFrame
        IC analysis results. Can be either:
        - Dictionary mapping feature names to FeatureICResults objects
        - DataFrame with 'feature' column and IC columns
    feature : str
        Feature name to plot.
    figsize : tuple[int, int], default (12, 6)
        Figure size in inches (width, height).
    title : str | None, default None
        Plot title. If None, uses "IC Time Series: {feature}".
    color : str, default "steelblue"
        Line color.
    show_mean : bool, default True
        Whether to show horizontal line at mean IC.
    show_bands : bool, default True
        Whether to show ±1 std deviation bands.
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
        If feature is not found or IC data is missing.

    Examples
    --------
    >>> from ml4t.engineer.outcome import FeatureOutcome, plot_ic_time_series
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
    >>> # Plot IC time series
    >>> fig = plot_ic_time_series(results.ic_results, "feature_1")
    >>>
    >>> # Custom styling
    >>> fig = plot_ic_time_series(
    ...     results.ic_results,
    ...     "feature_2",
    ...     color="coral",
    ...     show_bands=False
    ... )

    Notes
    -----
    - IC values show predictive power at different forward horizons
    - Mean IC represents average predictive power across all horizons
    - Confidence bands show ±1 standard deviation
    - Values closer to ±1 indicate stronger predictive power
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib is required for plotting. Install with: pip install matplotlib"
        raise ImportError(msg) from e

    # Convert to dict if DataFrame
    if isinstance(ic_results, pd.DataFrame):
        ic_results = _ic_dataframe_to_dict(ic_results)

    # Check feature exists
    if feature not in ic_results:
        available = list(ic_results.keys())
        msg = f"Feature '{feature}' not found. Available: {available}"
        raise ValueError(msg)

    result = ic_results[feature]

    # Check if ic_by_lag exists and has data
    if not result.ic_by_lag or len(result.ic_by_lag) == 0:
        msg = f"No IC by lag data available for feature '{feature}'"
        raise ValueError(msg)

    # Extract lag and IC values
    lags = sorted(result.ic_by_lag.keys())
    ic_values = [result.ic_by_lag[lag] for lag in lags]

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

    # Plot IC time series
    ax.plot(lags, ic_values, marker="o", color=color, linewidth=2, markersize=6, label="IC")

    # Add mean line if requested
    if show_mean and result.ic_mean != 0.0:
        ax.axhline(
            result.ic_mean,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Mean IC: {result.ic_mean:.3f}",
        )

    # Add confidence bands if requested
    if show_bands and result.ic_std != 0.0:
        upper = result.ic_mean + result.ic_std
        lower = result.ic_mean - result.ic_std
        ax.fill_between(
            lags, lower, upper, color=color, alpha=0.2, label=f"±1 std ({result.ic_std:.3f})"
        )

    # Add zero line for reference
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    # Set labels and title
    ax.set_xlabel("Forward Horizon (lag)")
    ax.set_ylabel("Information Coefficient")

    if title is None:
        title = f"IC Time Series: {feature}"
    ax.set_title(title)

    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc="best")

    # Tight layout
    fig.tight_layout()

    return fig


def plot_ic_comparison(
    ic_results: dict[str, FeatureICResults] | pd.DataFrame,
    features: list[str] | None = None,
    top_n: int | None = None,
    figsize: tuple[int, int] = (12, 8),
    title: str | None = None,
    sort_by: str = "ic_mean",
    ax: Axes | None = None,
) -> Figure:
    """Plot IC comparison across multiple features.

    Creates a bar chart comparing IC metrics (mean, IR) across features,
    helping identify which features have the strongest predictive power.

    Parameters
    ----------
    ic_results : dict[str, FeatureICResults] | pd.DataFrame
        IC analysis results.
    features : list[str] | None, default None
        Specific features to compare. If None, uses all features or top_n.
    top_n : int | None, default None
        Number of top features to display (by sort_by metric).
        Only used if features is None.
    figsize : tuple[int, int], default (12, 8)
        Figure size in inches (width, height).
    title : str | None, default None
        Plot title. If None, uses "IC Comparison".
    sort_by : str, default "ic_mean"
        Metric to sort by: "ic_mean", "ic_ir", or "ic_std".
    ax : Axes | None, default None
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    Figure
        Matplotlib figure object with comparison plot.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If required data is missing or sort_by is invalid.

    Examples
    --------
    >>> from ml4t.engineer.outcome import FeatureOutcome, plot_ic_comparison
    >>> import polars as pl
    >>>
    >>> # Analyze features
    >>> analyzer = FeatureOutcome()
    >>> results = analyzer.run_analysis(features_df, outcomes_df)
    >>>
    >>> # Compare all features
    >>> fig = plot_ic_comparison(results.ic_results)
    >>>
    >>> # Compare top 10 by IC IR
    >>> fig = plot_ic_comparison(results.ic_results, top_n=10, sort_by="ic_ir")
    >>>
    >>> # Compare specific features
    >>> fig = plot_ic_comparison(
    ...     results.ic_results,
    ...     features=["feature_1", "feature_2", "feature_3"]
    ... )

    Notes
    -----
    - Features are sorted by the specified metric (default: ic_mean)
    - Bars show IC mean with error bars representing ±1 std deviation
    - IC IR (Information Ratio) = ic_mean / ic_std shows risk-adjusted predictive power
    - Higher |IC| indicates stronger predictive power
    - Positive IC: feature positively predicts outcome
    - Negative IC: feature negatively predicts outcome
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = "matplotlib is required for plotting. Install with: pip install matplotlib"
        raise ImportError(msg) from e

    # Validate sort_by
    valid_sort = {"ic_mean", "ic_ir", "ic_std"}
    if sort_by not in valid_sort:
        msg = f"sort_by must be one of {valid_sort}, got {sort_by}"
        raise ValueError(msg)

    # Convert to DataFrame if dict (type: ignore for type narrowing issue)
    df = _ic_dict_to_dataframe(ic_results) if isinstance(ic_results, dict) else ic_results.copy()

    # Check required columns
    if "feature" not in df.columns:
        msg = "DataFrame must have 'feature' column"
        raise ValueError(msg)

    required_cols = ["ic_mean", "ic_std", "ic_ir"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        msg = f"DataFrame missing required columns: {missing_cols}"
        raise ValueError(msg)

    # Filter to specific features if requested
    if features is not None:
        df = df[df["feature"].isin(features)].copy()
        if len(df) == 0:
            msg = f"No matching features found: {features}"
            raise ValueError(msg)
    # Otherwise select top N if specified
    elif top_n is not None:
        # Sort by absolute value of sort_by metric (descending)
        df["_abs_sort"] = df[sort_by].abs()
        df = df.sort_values("_abs_sort", ascending=False).head(top_n)
        df = df.drop(columns=["_abs_sort"])

    # Sort by metric for display (descending)
    df = df.sort_values(sort_by, ascending=False)

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
    feature_names = df["feature"].values
    ic_mean_values = df["ic_mean"].values
    ic_std_values = df["ic_std"].values

    # Create horizontal bar chart with error bars
    y_pos = np.arange(len(feature_names))

    # Color bars based on IC sign (positive = blue, negative = red)
    colors = ["steelblue" if ic >= 0 else "coral" for ic in ic_mean_values]

    ax.barh(y_pos, ic_mean_values, color=colors, xerr=ic_std_values, capsize=3, alpha=0.8)

    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)

    # Add vertical line at zero
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)

    # Set labels and title
    metric_labels = {
        "ic_mean": "Mean IC",
        "ic_ir": "IC Information Ratio",
        "ic_std": "IC Std Dev",
    }
    ax.set_xlabel(metric_labels.get(sort_by, sort_by))

    if title is None:
        title = "IC Comparison"
    ax.set_title(title)

    # Add grid for readability
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend for colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", label="Positive IC", alpha=0.8),
        Patch(facecolor="coral", label="Negative IC", alpha=0.8),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    # Tight layout (suppress warning for incompatible axes)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="This figure includes Axes that are not compatible"
        )
        fig.tight_layout()

    return fig


def _ic_dict_to_dataframe(ic_results: dict[str, FeatureICResults]) -> pd.DataFrame:
    """Convert IC results dict to DataFrame.

    Parameters
    ----------
    ic_results : dict[str, FeatureICResults]
        Dictionary mapping feature names to IC results.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per feature.
    """
    rows = []
    for feature, result in ic_results.items():
        row = {
            "feature": feature,
            "ic_mean": result.ic_mean,
            "ic_std": result.ic_std,
            "ic_ir": result.ic_ir,
            "t_stat": result.t_stat,
            "p_value": result.p_value,
            "n_observations": result.n_observations,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _ic_dataframe_to_dict(df: pd.DataFrame) -> dict[str, FeatureICResults]:
    """Convert IC DataFrame to results dict.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with IC results.

    Returns
    -------
    dict[str, FeatureICResults]
        Dictionary mapping feature names to IC results.
    """
    from ml4t.engineer.outcome.feature_outcome import FeatureICResults

    results = {}
    for _, row in df.iterrows():
        feature = row["feature"]
        results[feature] = FeatureICResults(
            feature=feature,
            ic_mean=row.get("ic_mean", 0.0),
            ic_std=row.get("ic_std", 0.0),
            ic_ir=row.get("ic_ir", 0.0),
            t_stat=row.get("t_stat", 0.0),
            p_value=row.get("p_value", 1.0),
            ic_by_lag=row.get("ic_by_lag", {}),
            n_observations=row.get("n_observations", 0),
        )

    return results
