"""Correlation matrix computation for feature analysis.

This module provides functions to compute correlation matrices between features
using different correlation methods (Pearson, Spearman, Kendall).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import polars as pl


def _kendall_tau_b(x: pd.Series, y: pd.Series, min_periods: int) -> float:
    """Compute pairwise Kendall tau-b without SciPy."""
    x_values = x.to_numpy(dtype=float, na_value=np.nan)
    y_values = y.to_numpy(dtype=float, na_value=np.nan)
    valid = ~(np.isnan(x_values) | np.isnan(y_values))
    x_values = x_values[valid]
    y_values = y_values[valid]

    if len(x_values) < min_periods:
        return np.nan

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for index in range(len(x_values) - 1):
        x_tail = x_values[index + 1 :]
        y_tail = y_values[index + 1 :]
        x_sign = (x_tail > x_values[index]).astype(np.int8) - (x_tail < x_values[index]).astype(
            np.int8
        )
        y_sign = (y_tail > y_values[index]).astype(np.int8) - (y_tail < y_values[index]).astype(
            np.int8
        )
        products = x_sign * y_sign

        concordant += int(np.count_nonzero(products > 0))
        discordant += int(np.count_nonzero(products < 0))
        ties_x += int(np.count_nonzero((x_sign == 0) & (y_sign != 0)))
        ties_y += int(np.count_nonzero((y_sign == 0) & (x_sign != 0)))

    ordered_pairs = concordant + discordant
    denominator = np.sqrt(float(ordered_pairs + ties_x) * float(ordered_pairs + ties_y))
    if denominator == 0:
        return np.nan
    return (concordant - discordant) / denominator


def _kendall_correlation(data: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    """Compute a symmetric Kendall tau-b correlation matrix."""
    columns = list(data.columns)
    values = np.full((len(columns), len(columns)), np.nan, dtype=float)
    for left_index, left_column in enumerate(columns):
        for right_index in range(left_index, len(columns)):
            correlation = _kendall_tau_b(data[left_column], data[columns[right_index]], min_periods)
            values[left_index, right_index] = correlation
            values[right_index, left_index] = correlation
    return pd.DataFrame(values, index=columns, columns=columns)


def compute_correlation_matrix(
    data: pd.DataFrame | pl.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    min_periods: int | None = None,
    features: list[str] | None = None,
) -> pl.DataFrame:
    """Compute correlation matrix between features.

    Calculates pairwise correlations between all numeric features using
    the specified correlation method. Handles missing data appropriately
    for each method.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Input data with features as columns
    method : {"pearson", "spearman", "kendall"}, default "pearson"
        Correlation method to use:
        - "pearson": Linear correlation (assumes normality)
        - "spearman": Rank correlation (non-parametric, monotonic relationships)
        - "kendall": Tau correlation (non-parametric, ordinal data)
    min_periods : int | None, default None
        Minimum number of observations required per pair.
        If None, uses all available pairwise observations.
    features : list[str] | None, default None
        Specific features to compute correlations for.
        If None, uses all numeric columns.

    Returns
    -------
    pl.DataFrame
        Correlation matrix as Polars DataFrame with feature names as index/columns.
        Values range from -1 (perfect negative correlation) to 1 (perfect positive).

    Raises
    ------
    ValueError
        If method is not one of the supported methods
        If specified features not found in data
        If data has insufficient numeric columns

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.relationships import compute_correlation_matrix
    >>>
    >>> # Create sample data
    >>> df = pl.DataFrame({
    ...     "returns": [0.01, -0.02, 0.03, -0.01, 0.02],
    ...     "volume": [1000, 1500, 1200, 1300, 1100],
    ...     "volatility": [0.15, 0.25, 0.18, 0.20, 0.16]
    ... })
    >>>
    >>> # Compute Pearson correlation
    >>> corr = compute_correlation_matrix(df, method="pearson")
    >>> print(corr)
    >>>
    >>> # Compute Spearman (rank) correlation
    >>> corr_spearman = compute_correlation_matrix(df, method="spearman")
    >>>
    >>> # Specify specific features
    >>> corr_subset = compute_correlation_matrix(
    ...     df, features=["returns", "volume"], method="pearson"
    ... )

    Notes
    -----
    **Correlation Methods**:

    - **Pearson**: Measures linear relationship. Assumes:
      - Variables are continuous
      - Relationship is linear
      - Variables are normally distributed (for significance testing)
      Formula: ρ = cov(X,Y) / (σ_X * σ_Y)

    - **Spearman**: Measures monotonic relationship. Non-parametric.
      - Converts data to ranks
      - Robust to outliers
      - Detects non-linear monotonic relationships
      Formula: Pearson correlation of ranked data

    - **Kendall**: Measures ordinal association. Non-parametric.
      - Based on concordant/discordant pairs
      - More robust than Spearman for small samples
      - Better for ordinal data
      Formula: τ = (concordant - discordant) / total_pairs

    **Missing Data Handling**:
    - Uses pairwise deletion by default
    - Each correlation computed using all available pairs
    - Set min_periods to require minimum observations per pair

    **Performance**:
    - Pearson: O(n*m²) where n=rows, m=features (fastest)
    - Spearman: O(n*m² * log(n)) (ranking overhead)
    - Kendall: O(n²*m²) (slowest, use for small datasets)

    References
    ----------
    - Pearson, K. (1895). "Notes on regression and inheritance in the case of
      two parents". Proceedings of the Royal Society of London, 58, 240-242.
    - Spearman, C. (1904). "The proof and measurement of association between
      two things". American Journal of Psychology, 15(1), 72-101.
    - Kendall, M. G. (1938). "A new measure of rank correlation". Biometrika,
      30(1/2), 81-93.
    """
    # Validate method
    valid_methods = ["pearson", "spearman", "kendall"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {', '.join(valid_methods)}")

    # Convert to pandas for computation (easier correlation handling)
    df = (
        pd.DataFrame(data.to_dict(as_series=False))
        if isinstance(data, pl.DataFrame)
        else data.copy()
    )

    # Select features
    if features is None:
        # Use all numeric columns (including all-null columns which pandas treats as object)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Also include columns that are all None (from Polars null type)
        # These become object dtype in pandas but should be included in correlation
        for col in df.columns:
            if col not in numeric_cols and df[col].dtype == object and df[col].isna().all():
                numeric_cols.append(col)

        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in data")
        features = numeric_cols
    else:
        # Validate specified features exist
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(f"Features not found in data: {sorted(missing)}")

        # Ensure features are numeric or all-null
        non_numeric = []
        for feat in features:
            is_numeric = pd.api.types.is_numeric_dtype(df[feat].dtype)
            is_all_null = df[feat].dtype == object and df[feat].isna().all()
            if not (is_numeric or is_all_null):
                non_numeric.append(feat)
        if non_numeric:
            raise ValueError(f"Non-numeric features specified: {sorted(non_numeric)}")

    # Extract feature data
    feature_data = df[features]

    # Set default min_periods if not specified
    # pandas requires an integer, default to 1 (use all available pairs)
    if min_periods is None:
        min_periods = 1

    # Compute correlation matrix based on method
    if method == "pearson":
        # Pearson correlation (linear)
        corr_matrix = feature_data.corr(method="pearson", min_periods=min_periods)

    elif method == "spearman":
        # Spearman rank correlation
        corr_matrix = feature_data.corr(method="spearman", min_periods=min_periods)

    elif method == "kendall":
        # Kendall tau correlation
        corr_matrix = _kendall_correlation(feature_data, min_periods)

    # Ensure all features are present in correlation matrix
    # (pandas drops all-NaN columns, but we want to preserve them)
    corr_matrix = corr_matrix.reindex(index=features, columns=features)

    # Convert to Polars DataFrame with proper schema
    # Reset index to make feature names a column
    corr_df = corr_matrix.reset_index()
    corr_df = corr_df.rename(columns={"index": "feature"})

    # Convert to Polars
    result = pl.DataFrame(corr_df.to_dict(orient="list"))

    return result
