"""Composite feature builders for combining multiple signals.

This module provides utilities for combining multiple features into composite
signals using various normalization and weighting schemes. Common use cases:

1. **Multi-factor scores**: Combine momentum, value, quality into single score
2. **Illiquidity composites**: Aggregate multiple liquidity proxies
3. **Risk aggregation**: Combine volatility, correlation, drawdown measures
4. **Sentiment composites**: Aggregate multiple sentiment signals

The z-score approach ensures features are on comparable scales before combining.

References
----------
.. [1] Cahan, R. & Luo, Y. (2013). "Breaking Bad Trends: The Role of Trend
       Reversals in Capital Market Anomalies". Journal of Portfolio Management.
"""

import polars as pl

__all__ = [
    "z_score_composite",
    "rolling_z_score",
    "illiquidity_composite",
    "momentum_composite",
]


def rolling_z_score(
    column: str | pl.Expr,
    period: int = 252,
    min_periods: int | None = None,
) -> pl.Expr:
    """Compute rolling z-score for a column.

    The rolling z-score standardizes a feature using its trailing mean and
    standard deviation, making it comparable across different scales.

    Parameters
    ----------
    column : str | pl.Expr
        Column to z-score.
    period : int, default 252
        Lookback period for rolling statistics.
    min_periods : int, optional
        Minimum observations required. Defaults to period // 2.

    Returns
    -------
    pl.Expr
        Rolling z-score expression.

    Notes
    -----
    The z-score is computed as:

    .. math::

        z_t = \\frac{x_t - \\bar{x}_{t-n:t}}{\\sigma_{t-n:t}}

    where n is the lookback period.

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.features.composite import rolling_z_score
    >>>
    >>> df = pl.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> df.with_columns(rolling_z_score("feature", period=3).alias("z_feature"))
    """
    col = pl.col(column) if isinstance(column, str) else column
    min_p = min_periods if min_periods is not None else max(1, period // 2)

    rolling_mean = col.rolling_mean(period, min_samples=min_p)
    rolling_std = col.rolling_std(period, min_samples=min_p)

    # Avoid division by zero: when std is 0, z-score is 0
    return pl.when(rolling_std > 0).then((col - rolling_mean) / rolling_std).otherwise(0.0)


def z_score_composite(
    data: pl.DataFrame,
    feature_cols: list[str],
    period: int = 252,
    weights: list[float] | None = None,
    output_col: str = "composite_score",
    min_periods: int | None = None,
) -> pl.DataFrame:
    """Combine multiple features via z-score normalization and weighting.

    This function standardizes each feature using rolling z-scores, then
    combines them with optional weights into a single composite score.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame containing feature columns.
    feature_cols : list[str]
        List of column names to combine.
    period : int, default 252
        Lookback period for z-score normalization.
    weights : list[float], optional
        Weights for each feature. If None, uses equal weights.
        Must sum to a non-zero value (will be normalized internally).
    output_col : str, default "composite_score"
        Name for the output composite column.
    min_periods : int, optional
        Minimum observations for rolling statistics. Defaults to period // 2.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added composite score column.

    Notes
    -----
    The composite is computed as:

    .. math::

        C_t = \\sum_{i=1}^{n} w_i \\cdot z_i(x_{i,t})

    where w_i are normalized weights and z_i is the rolling z-score.

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.features.composite import z_score_composite
    >>>
    >>> df = pl.DataFrame({
    ...     "momentum": [0.1, 0.2, -0.1, 0.3, 0.15],
    ...     "value": [1.5, 1.2, 1.8, 0.9, 1.1],
    ...     "quality": [0.8, 0.7, 0.9, 0.85, 0.75],
    ... })
    >>> result = z_score_composite(
    ...     df,
    ...     feature_cols=["momentum", "value", "quality"],
    ...     period=3,
    ...     weights=[0.4, 0.3, 0.3],
    ... )
    """
    if not feature_cols:
        msg = "feature_cols must not be empty"
        raise ValueError(msg)

    # Normalize weights or use equal weights
    if weights is None:
        normalized_weights = [1.0 / len(feature_cols)] * len(feature_cols)
    else:
        if len(weights) != len(feature_cols):
            msg = f"weights length ({len(weights)}) must match feature_cols ({len(feature_cols)})"
            raise ValueError(msg)
        weight_sum = sum(weights)
        if weight_sum == 0:
            msg = "weights must sum to a non-zero value"
            raise ValueError(msg)
        normalized_weights = [w / weight_sum for w in weights]

    # Compute weighted sum of z-scores
    composite_expr = pl.lit(0.0)
    for col_name, weight in zip(feature_cols, normalized_weights, strict=True):
        z_score = rolling_z_score(col_name, period=period, min_periods=min_periods)
        composite_expr = composite_expr + (z_score * weight)

    return data.with_columns(composite_expr.alias(output_col))


def illiquidity_composite(
    data: pl.DataFrame,
    kyle_col: str = "kyle_lambda",
    amihud_col: str = "amihud",
    roll_col: str = "roll_spread",
    period: int = 20,
    output_col: str = "illiquidity_score",
) -> pl.DataFrame:
    """Compute composite illiquidity score from multiple proxies.

    Combines Kyle's Lambda, Amihud illiquidity, and Roll spread estimator
    into a single z-score normalized composite. Higher scores indicate
    lower liquidity (harder to trade without price impact).

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame containing illiquidity columns.
    kyle_col : str, default "kyle_lambda"
        Column with Kyle's Lambda values.
    amihud_col : str, default "amihud"
        Column with Amihud illiquidity values.
    roll_col : str, default "roll_spread"
        Column with Roll spread estimator values.
    period : int, default 20
        Lookback period for z-score normalization.
    output_col : str, default "illiquidity_score"
        Name for the output column.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added illiquidity composite column.

    Notes
    -----
    Only includes columns that exist in the input DataFrame. Missing columns
    are skipped with a warning-free fallback to available columns.

    The composite uses equal weights since the z-score normalization
    already puts all measures on comparable scales.

    References
    ----------
    .. [1] Kyle, A.S. (1985). "Continuous Auctions and Insider Trading".
           Econometrica, 53(6), 1315-1335.
    .. [2] Amihud, Y. (2002). "Illiquidity and Stock Returns".
           Journal of Financial Markets, 5(1), 31-56.
    .. [3] Roll, R. (1984). "A Simple Implicit Measure of the Effective
           Bid-Ask Spread in an Efficient Market".
           Journal of Finance, 39(4), 1127-1139.

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.features.composite import illiquidity_composite
    >>>
    >>> df = pl.DataFrame({
    ...     "kyle_lambda": [0.001, 0.002, 0.0015, 0.003],
    ...     "amihud": [0.05, 0.08, 0.06, 0.1],
    ...     "roll_spread": [0.002, 0.003, 0.0025, 0.004],
    ... })
    >>> result = illiquidity_composite(df, period=2)
    """
    # Check which columns exist
    candidate_cols = [kyle_col, amihud_col, roll_col]
    available_cols = [c for c in candidate_cols if c in data.columns]

    if not available_cols:
        msg = f"None of {candidate_cols} found in DataFrame columns: {data.columns}"
        raise ValueError(msg)

    return z_score_composite(
        data,
        feature_cols=available_cols,
        period=period,
        weights=None,  # Equal weights
        output_col=output_col,
    )


def momentum_composite(
    data: pl.DataFrame,
    rsi_col: str = "rsi",
    macd_col: str = "macd",
    roc_col: str = "roc",
    period: int = 20,
    output_col: str = "momentum_score",
) -> pl.DataFrame:
    """Compute composite momentum score from multiple signals.

    Combines RSI, MACD, and Rate of Change into a single z-score
    normalized composite. Higher scores indicate stronger bullish momentum.

    Parameters
    ----------
    data : pl.DataFrame
        Input DataFrame containing momentum indicator columns.
    rsi_col : str, default "rsi"
        Column with RSI values (0-100 scale).
    macd_col : str, default "macd"
        Column with MACD histogram values.
    roc_col : str, default "roc"
        Column with Rate of Change values.
    period : int, default 20
        Lookback period for z-score normalization.
    output_col : str, default "momentum_score"
        Name for the output column.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added momentum composite column.

    Notes
    -----
    Only includes columns that exist in the input DataFrame.
    Uses equal weights by default.

    Examples
    --------
    >>> import polars as pl
    >>> from ml4t.engineer.features.composite import momentum_composite
    >>>
    >>> df = pl.DataFrame({
    ...     "rsi": [45, 55, 65, 50, 40],
    ...     "macd": [-0.5, 0.2, 0.8, 0.1, -0.3],
    ...     "roc": [-0.02, 0.01, 0.03, 0.0, -0.01],
    ... })
    >>> result = momentum_composite(df, period=3)
    """
    candidate_cols = [rsi_col, macd_col, roc_col]
    available_cols = [c for c in candidate_cols if c in data.columns]

    if not available_cols:
        msg = f"None of {candidate_cols} found in DataFrame columns: {data.columns}"
        raise ValueError(msg)

    return z_score_composite(
        data,
        feature_cols=available_cols,
        period=period,
        weights=None,
        output_col=output_col,
    )
