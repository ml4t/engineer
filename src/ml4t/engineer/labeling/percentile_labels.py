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

    >>> # Time-based horizon: 1-hour forward returns
    >>> labels = rolling_percentile_binary_labels(
    ...     df,
    ...     horizon="1h",
    ...     percentile=95,
    ...     direction="long",
    ...     lookback_window="5d",  # 5-day rolling window
    ... )

Reference:
    Based on methodology from Wyden Long-Short Trading System.
    See: .claude/reference/LABELING_AND_SCORING_METHODOLOGY.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

from ml4t.engineer.labeling.utils import (
    get_future_price_at_time,
    is_duration_string,
    parse_duration,
    resolve_labeling_columns,
)

if TYPE_CHECKING:
    from ml4t.engineer.config import DataContractConfig, LabelingConfig


def rolling_percentile_binary_labels(
    data: pl.DataFrame,
    horizon: int | str,
    percentile: float,
    direction: Literal["long", "short"] = "long",
    lookback_window: int | str = 252 * 24 * 12,  # ~1 year hourly
    price_col: str | None = None,
    session_col: str | None = None,
    min_samples: int | None = None,
    group_col: str | list[str] | None = None,
    timestamp_col: str | None = None,
    tolerance: str | None = None,
    *,
    config: LabelingConfig | None = None,
    contract: DataContractConfig | None = None,
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
    horizon : int | str
        Forward-looking horizon:
        - int: Number of bars
        - str: Duration string (e.g., '1h', '30m', '1d')
    percentile : float
        Percentile for thresholding (0-100)
        - Long: High percentiles (e.g., 95, 98) → top returns
        - Short: Low percentiles (e.g., 5, 10) → bottom returns
    direction : {"long", "short"}, default "long"
        Trading direction:
        - "long": Labels profitable long entries (high positive returns)
        - "short": Labels profitable short entries (high negative returns)
    lookback_window : int | str, default ~1 year
        Rolling window size for percentile computation:
        - int: Number of bars
        - str: Duration string (e.g., '5d', '1w'). Polars rolling supports duration strings.
    price_col : str | None, default None
        Price column for return computation
    session_col : str, optional
        Session column for session-aware forward returns (e.g., "session_date")
        If provided, forward returns won't cross session boundaries
    min_samples : int, optional
        Minimum samples for rolling calculation (default: 1008 = ~3.5 days of 5-min bars)
    group_col : str | list[str] | None, default None
        Column(s) to group by for panel-aware labeling. If None, auto-detects from
        common symbol columns when present.
    timestamp_col : str | None, default None
        Column to use for chronological sorting. If None, auto-detects from
        column dtype (pl.Datetime, pl.Date). Required for time-based horizons.
    tolerance : str | None, default None
        Maximum time gap allowed for time-based horizons (e.g., '2m').
        Only used when horizon is a duration string.
    config : LabelingConfig | None, default None
        Optional column contract source. If provided, `price_col`, `timestamp_col`,
        and `group_col` default to config values when omitted.
    contract : DataContractConfig | None, default None
        Optional shared dataframe contract. Used after config and before defaults.

    Returns
    -------
    pl.DataFrame
        Original data with added columns:
        - forward_return_{horizon}: Forward returns
        - threshold_p{percentile}_h{horizon}: Rolling percentile threshold
        - label_{direction}_p{percentile}_h{horizon}: Binary label (0 or 1)

    Examples
    --------
    >>> # Bar-based: Top 5% of 30-bar returns
    >>> labels_long = rolling_percentile_binary_labels(
    ...     df,
    ...     horizon=30,
    ...     percentile=95,
    ...     direction="long",
    ...     session_col="session_date"
    ... )
    >>> print(labels_long["label_long_p95_h30"].mean())  # Should be ~0.05

    >>> # Time-based: 1-hour forward returns with 5-day lookback
    >>> labels = rolling_percentile_binary_labels(
    ...     df,
    ...     horizon="1h",
    ...     percentile=95,
    ...     direction="long",
    ...     lookback_window="5d",
    ... )

    >>> # Short labels: Bottom 5% of returns (5th percentile)
    >>> labels_short = rolling_percentile_binary_labels(
    ...     df,
    ...     horizon=30,
    ...     percentile=5,
    ...     direction="short",
    ...     session_col="session_date"
    ... )

    Notes
    -----
    - First lookback_window bars will have null labels (insufficient history)
    - Last horizon bars will have null forward returns (insufficient future data)
    - Class balance approximately matches percentile (p95 → ~5% positives)
    - Adaptive: Thresholds widen in high volatility, tighten in low volatility
    - No lookahead bias: Only uses past data for percentile computation

    **Time-based horizons**: When horizon is a duration string, uses join_asof
    to get future prices. This is useful for irregular data like trade bars.

    **Time-based lookback**: Polars rolling functions natively support duration
    strings for the window parameter, allowing time-based rolling windows.

    **Important**: Data is automatically sorted by timestamp before labeling.
    This is required because Polars .over() and .shift() preserve row order.
    The result is returned sorted chronologically.
    """
    # Determine if time-based
    is_time_based_horizon = isinstance(horizon, str) and is_duration_string(horizon)
    is_time_based_lookback = isinstance(lookback_window, str) and is_duration_string(
        lookback_window
    )

    resolved_price_col, resolved_ts_col, resolved_group_cols = resolve_labeling_columns(
        data=data,
        price_col=price_col,
        timestamp_col=timestamp_col,
        group_col=group_col,
        config=config,
        contract=contract,
        require_timestamp=is_time_based_horizon or is_time_based_lookback,
    )

    sort_cols = resolved_group_cols + ([resolved_ts_col] if resolved_ts_col else [])
    if sort_cols:
        data = data.sort(sort_cols)

    result = data.clone()

    # Create label suffix for column naming
    if is_time_based_horizon:
        horizon_label = horizon.lower().replace(" ", "")  # type: ignore[union-attr]
    else:
        horizon_label = str(horizon)

    # Step 1: Compute forward returns
    if is_time_based_horizon:
        # Time-based forward returns using join_asof
        td = parse_duration(horizon)  # type: ignore[arg-type]
        future_prices, valid_mask = get_future_price_at_time(
            data=data,
            time_horizon=td,
            price_col=resolved_price_col,
            timestamp_col=resolved_ts_col,
            tolerance=tolerance,
            group_cols=resolved_group_cols if resolved_group_cols else None,
        )
        current_prices = data[resolved_price_col]
        forward_returns = (future_prices - current_prices) / current_prices

        # Mask invalid joins if tolerance specified
        if tolerance is not None:
            forward_returns = pl.when(valid_mask).then(forward_returns).otherwise(pl.lit(None))
    elif session_col is not None:
        if session_col not in data.columns:
            raise ValueError(f"Session column '{session_col}' not found in data")

        session_groups = (
            [*resolved_group_cols, session_col] if resolved_group_cols else [session_col]
        )

        # Session-aware forward returns (don't cross session boundaries)
        forward_returns = result.with_columns(
            (pl.col(resolved_price_col).shift(-horizon) / pl.col(resolved_price_col) - 1)
            .over(session_groups)
            .alias("forward_return")
        )["forward_return"]
    else:
        # Simple bar-based forward returns
        forward_expr = pl.col(resolved_price_col).shift(-horizon) / pl.col(resolved_price_col) - 1
        if resolved_group_cols:
            forward_expr = forward_expr.over(resolved_group_cols)
        forward_returns = result.with_columns(forward_expr.alias("forward_return"))[
            "forward_return"
        ]

    forward_return_col = f"forward_return_{horizon_label}"
    result = result.with_columns(forward_returns.alias(forward_return_col))

    # Step 2: Compute rolling percentile threshold
    quantile = percentile / 100.0

    # Determine min_samples default
    if min_samples is None:
        if isinstance(lookback_window, int):
            min_samples = min(1008, lookback_window // 10)
        else:
            # For time-based lookback, use a reasonable default
            min_samples = 100

    # Compute rolling threshold
    # Polars rolling_quantile supports duration strings for window_size when
    # using rolling() context with index_column
    if is_time_based_lookback and resolved_ts_col:
        # Use Polars native time-based rolling via rolling() context
        rolling_result = result.rolling(
            index_column=resolved_ts_col,
            period=lookback_window,  # type: ignore[arg-type]
            group_by=resolved_group_cols if resolved_group_cols else None,
        ).agg(pl.col(forward_return_col).quantile(quantile).alias("_rolling_threshold"))

        join_keys = (
            [*resolved_group_cols, resolved_ts_col] if resolved_group_cols else [resolved_ts_col]
        )
        # Join back to get the threshold column
        rolling_threshold = result.join(
            rolling_result,
            on=join_keys,
            how="left",
        )["_rolling_threshold"]
    else:
        # Bar-based rolling (original implementation)
        if not isinstance(lookback_window, int):
            raise ValueError(
                f"lookback_window must be an integer for bar-based rolling, "
                f"got '{lookback_window}'. For time-based, ensure timestamp_col is set."
            )
        rolling_threshold_expr = pl.col(forward_return_col).rolling_quantile(
            window_size=lookback_window,
            quantile=quantile,
            min_samples=min_samples,
            center=False,
        )
        if resolved_group_cols:
            rolling_threshold_expr = rolling_threshold_expr.over(resolved_group_cols)
        rolling_threshold = result.with_columns(rolling_threshold_expr.alias("_rolling_threshold"))[
            "_rolling_threshold"
        ]

    threshold_col_name = f"threshold_p{int(percentile)}_h{horizon_label}"
    result = result.with_columns(rolling_threshold.alias(threshold_col_name))

    # Step 3: Create binary labels based on direction
    forward_ret_col = result[forward_return_col]
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

    label_col_name = f"label_{direction}_p{int(percentile)}_h{horizon_label}"
    result = result.with_columns(label.alias(label_col_name))

    return result


def rolling_percentile_multi_labels(
    data: pl.DataFrame,
    horizons: list[int | str],
    percentiles: list[float],
    direction: Literal["long", "short"] = "long",
    lookback_window: int | str = 252 * 24 * 12,
    price_col: str | None = None,
    session_col: str | None = None,
    group_col: str | list[str] | None = None,
    timestamp_col: str | None = None,
    tolerance: str | None = None,
    *,
    config: LabelingConfig | None = None,
    contract: DataContractConfig | None = None,
) -> pl.DataFrame:
    """Create binary labels for multiple horizons and percentiles.

    Convenience function to generate labels for multiple configurations
    in a single call.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with OHLCV and optionally session_date
    horizons : list[int | str]
        List of forward-looking horizons (e.g., [15, 30, "1h"])
    percentiles : list[float]
        List of percentiles (e.g., [95, 98] for long, [5, 10] for short)
    direction : {"long", "short"}, default "long"
        Trading direction
    lookback_window : int | str, default ~1 year
        Rolling window size for percentile computation
    price_col : str | None, default None
        Price column
    session_col : str, optional
        Session column for session-aware returns
    group_col : str | list[str] | None, default None
        Column(s) to group by for panel-aware labeling.
    timestamp_col : str | None, default None
        Timestamp column for time-based horizons/lookbacks.
    tolerance : str | None, default None
        Maximum time gap for time-based horizons.
    config : LabelingConfig | None, default None
        Optional column contract source.
    contract : DataContractConfig | None, default None
        Optional shared dataframe contract. Used after config and before defaults.

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
            group_col=group_col,
            timestamp_col=timestamp_col,
            tolerance=tolerance,
            config=config,
            contract=contract,
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
                group_col=group_col,
                timestamp_col=timestamp_col,
                tolerance=tolerance,
                config=config,
                contract=contract,
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
