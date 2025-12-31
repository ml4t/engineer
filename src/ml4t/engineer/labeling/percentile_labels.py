"""Rolling percentile-based binary labels for fixed-horizon strategies.

This module implements adaptive binary labeling using rolling historical percentiles,
suitable for creating training labels that adapt to volatility regimes.

Key Concepts:
- Uses rolling window of historical returns to compute percentiles
- Thresholds adapt to market volatility (high vol → wider thresholds)
- Natural class balance control via percentile selection
- Session-aware: Respects session boundaries (e.g., CME futures gaps)

Example:
    >>> import polars as pl
    >>> from ml4t.engineer.labeling import rolling_percentile_binary_labels

    >>> # Load futures data with session_date column
    >>> df = pl.read_parquet("futures_data.parquet")

    >>> # Create long labels: Top 5% of historical returns
    >>> labels = rolling_percentile_binary_labels(
    ...     df,
    ...     horizon=30,
    ...     percentile=95,  # 95th percentile
    ...     direction="long",
    ...     lookback_window=252 * 24 * 12,  # ~1 year of hourly data
    ...     session_col="session_date"
    ... )

Reference:
    Based on methodology from Wyden Long-Short Trading System.
    See: .claude/reference/LABELING_AND_SCORING_METHODOLOGY.md
"""

from typing import Literal

import polars as pl


def rolling_percentile_binary_labels(
    data: pl.DataFrame,
    horizon: int,
    percentile: float,
    direction: Literal["long", "short"] = "long",
    lookback_window: int = 252 * 24 * 12,  # ~1 year hourly
    price_col: str = "close",
    session_col: str | None = None,
    min_samples: int | None = None,
) -> pl.DataFrame:
    """Create binary labels using rolling historical percentiles.

    Computes forward returns, then creates binary labels by comparing returns
    to rolling percentile thresholds. Thresholds adapt to volatility regimes.

    Algorithm:
    1. Compute forward returns over horizon (session-aware if session_col provided)
    2. Compute rolling percentile from lookback window
    3. For long: label = 1 if forward_return >= threshold, else 0
       For short: label = 1 if forward_return <= threshold, else 0

    Parameters
    ----------
    data : pl.DataFrame
        Input data with OHLCV and optionally session_date
    horizon : int
        Forward-looking horizon in bars
    percentile : float
        Percentile for thresholding (0-100)
        - Long: High percentiles (e.g., 95, 98) → top returns
        - Short: Low percentiles (e.g., 5, 10) → bottom returns
    direction : {"long", "short"}, default "long"
        Trading direction:
        - "long": Labels profitable long entries (high positive returns)
        - "short": Labels profitable short entries (high negative returns)
    lookback_window : int, default ~1 year
        Rolling window size for percentile computation (in bars)
    price_col : str, default "close"
        Price column for return computation
    session_col : str, optional
        Session column for session-aware forward returns (e.g., "session_date")
        If provided, forward returns won't cross session boundaries
    min_samples : int, optional
        Minimum samples for rolling calculation (default: 1008 = ~3.5 days of 5-min bars)
        Lower values allow earlier threshold computation but with less statistical confidence

    Returns
    -------
    pl.DataFrame
        Original data with added columns:
        - forward_return_{horizon}: Forward returns
        - threshold_p{percentile}_h{horizon}: Rolling percentile threshold
        - label_{direction}_p{percentile}_h{horizon}: Binary label (0 or 1)

    Examples
    --------
    >>> # Long labels: Top 5% of returns (95th percentile)
    >>> labels_long = rolling_percentile_binary_labels(
    ...     df,
    ...     horizon=30,
    ...     percentile=95,
    ...     direction="long",
    ...     session_col="session_date"
    ... )
    >>> print(labels_long["label_long_p95_h30"].mean())  # Should be ~0.05

    >>> # Short labels: Bottom 5% of returns (5th percentile)
    >>> labels_short = rolling_percentile_binary_labels(
    ...     df,
    ...     horizon=30,
    ...     percentile=5,
    ...     direction="short",
    ...     session_col="session_date"
    ... )
    >>> print(labels_short["label_short_p5_h30"].mean())  # Should be ~0.05

    Notes
    -----
    - First lookback_window bars will have null labels (insufficient history)
    - Last horizon bars will have null forward returns (insufficient future data)
    - Class balance approximately matches percentile (p95 → ~5% positives)
    - Adaptive: Thresholds widen in high volatility, tighten in low volatility
    - No lookahead bias: Only uses past data for percentile computation
    """
    if min_samples is None:
        min_samples = min(1008, lookback_window // 10)  # 1008 = ~3.5 days of 5-min bars

    result = data.clone()

    # Step 1: Compute forward returns (session-aware if session_col provided)
    if session_col is not None:
        # Session-aware forward returns (don't cross session boundaries)
        temp_df = pl.DataFrame(
            {
                session_col: data[session_col],
                "price": data[price_col],
            }
        )

        forward_returns = temp_df.with_columns(
            [
                (pl.col("price").shift(-horizon) / pl.col("price") - 1)
                .over(session_col)
                .alias("forward_return")
            ]
        )["forward_return"]
    else:
        # Simple forward returns (may cross any boundaries)
        forward_returns = data[price_col].shift(-horizon) / data[price_col] - 1

    result = result.with_columns(forward_returns.alias(f"forward_return_{horizon}"))

    # Step 2: Compute rolling percentile threshold
    # Use quantile (0-1 scale) not percentile (0-100 scale)
    quantile = percentile / 100.0

    # CRITICAL: Use result[f"forward_return_{horizon}"] to ensure rolling window
    # sees the data correctly (forward_returns variable might not be in right context)
    rolling_threshold = result[f"forward_return_{horizon}"].rolling_quantile(
        window_size=lookback_window,
        quantile=quantile,
        min_samples=min_samples,  # Updated from min_periods (deprecated)
        center=False,  # Only look backward (no lookahead bias)
    )

    threshold_col_name = f"threshold_p{int(percentile)}_h{horizon}"
    result = result.with_columns(rolling_threshold.alias(threshold_col_name))

    # Step 3: Create binary labels based on direction
    forward_ret_col = result[f"forward_return_{horizon}"]
    threshold_col = result[threshold_col_name]

    if direction == "long":
        # Long: 1 if forward_return >= threshold (top percentile)
        label = (forward_ret_col >= threshold_col).cast(pl.Int8)
    elif direction == "short":
        # Short: 1 if forward_return <= threshold (bottom percentile)
        label = (forward_ret_col <= threshold_col).cast(pl.Int8)
    else:
        msg = f"Invalid direction: {direction}. Must be 'long' or 'short'."
        raise ValueError(msg)

    label_col_name = f"label_{direction}_p{int(percentile)}_h{horizon}"
    result = result.with_columns(label.alias(label_col_name))

    return result


def rolling_percentile_multi_labels(
    data: pl.DataFrame,
    horizons: list[int],
    percentiles: list[float],
    direction: Literal["long", "short"] = "long",
    lookback_window: int = 252 * 24 * 12,
    price_col: str = "close",
    session_col: str | None = None,
) -> pl.DataFrame:
    """Create binary labels for multiple horizons and percentiles.

    Convenience function to generate labels for multiple configurations
    in a single call.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with OHLCV and optionally session_date
    horizons : list[int]
        List of forward-looking horizons (e.g., [15, 30, 60])
    percentiles : list[float]
        List of percentiles (e.g., [95, 98] for long, [5, 10] for short)
    direction : {"long", "short"}, default "long"
        Trading direction
    lookback_window : int, default ~1 year
        Rolling window size for percentile computation
    price_col : str, default "close"
        Price column
    session_col : str, optional
        Session column for session-aware returns

    Returns
    -------
    pl.DataFrame
        Original data with label columns for all combinations:
        - label_{direction}_p{percentile}_h{horizon}

    Examples
    --------
    >>> # Generate labels for multiple horizons and percentiles
    >>> labels = rolling_percentile_multi_labels(
    ...     df,
    ...     horizons=[15, 30, 60],
    ...     percentiles=[95, 98],
    ...     direction="long",
    ...     session_col="session_date"
    ... )
    >>> # Creates 6 label columns: 3 horizons × 2 percentiles
    >>> print([c for c in labels.columns if c.startswith("label_")])
    """
    result = data.clone()

    for horizon in horizons:
        # First call for this horizon - will add forward_return column
        first_percentile = percentiles[0]
        result = rolling_percentile_binary_labels(
            result,
            horizon=horizon,
            percentile=first_percentile,
            direction=direction,
            lookback_window=lookback_window,
            price_col=price_col,
            session_col=session_col,
        )

        # Subsequent calls for same horizon - skip if forward_return already exists
        for percentile in percentiles[1:]:
            # Call with the result that already has forward_return for this horizon
            result = rolling_percentile_binary_labels(
                result,
                horizon=horizon,
                percentile=percentile,
                direction=direction,
                lookback_window=lookback_window,
                price_col=price_col,
                session_col=session_col,
            )

    return result


def compute_label_statistics(
    data: pl.DataFrame,
    label_col: str,
) -> dict[str, float | int]:
    """Compute statistics for a binary label column.

    Useful for validating label quality and understanding class balance.

    Parameters
    ----------
    data : pl.DataFrame
        Data with label column
    label_col : str
        Name of binary label column

    Returns
    -------
    dict
        Statistics including:
        - total_bars: Total number of bars
        - positive_labels: Count of 1s
        - negative_labels: Count of 0s
        - null_labels: Count of nulls
        - positive_rate: Percentage of 1s (among non-null)
        - null_rate: Percentage of nulls

    Examples
    --------
    >>> stats = compute_label_statistics(df, "label_long_p95_h30")
    >>> print(f"Positive rate: {stats['positive_rate']:.2f}%")
    >>> print(f"Null rate: {stats['null_rate']:.2f}%")
    """
    labels = data[label_col]

    total = len(labels)
    nulls = labels.null_count()
    non_null = total - nulls

    if non_null > 0:
        positives = labels.filter(labels == 1).len()
        negatives = labels.filter(labels == 0).len()
        positive_rate = (positives / non_null) * 100
    else:
        positives = 0
        negatives = 0
        positive_rate = 0.0

    return {
        "total_bars": total,
        "positive_labels": positives,
        "negative_labels": negatives,
        "null_labels": nulls,
        "positive_rate": positive_rate,
        "null_rate": (nulls / total) * 100,
    }
