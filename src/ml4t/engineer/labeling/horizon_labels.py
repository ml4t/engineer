# mypy: disable-error-code="no-any-return,arg-type,call-arg,return-value,assignment"
"""Fixed horizon and trend scanning labeling methods.

Provides simpler labeling methods for supervised learning:
- Fixed time horizon labels (forward returns)
- Trend scanning labels (De Prado's method)

References
----------
.. [1] De Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
       Chapter 3: Labeling and Chapter 18: Entropy Features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ml4t.engineer.labeling.utils import (
    get_future_price_at_time,
    is_duration_string,
    parse_duration,
    resolve_labeling_columns,
)

if TYPE_CHECKING:
    from ml4t.engineer.config import DataContractConfig, LabelingConfig


def fixed_time_horizon_labels(
    data: pl.DataFrame,
    horizon: int | str = 1,
    method: str = "returns",
    price_col: str | None = None,
    group_col: str | list[str] | None = None,
    timestamp_col: str | None = None,
    tolerance: str | None = None,
    *,
    config: LabelingConfig | None = None,
    contract: DataContractConfig | None = None,
) -> pl.DataFrame:
    """Generate forward-looking labels based on fixed time horizon.

    Creates labels by looking ahead a fixed number of periods (bars) or a
    fixed time duration and computing the return or direction of price
    movement. Commonly used for supervised learning in financial forecasting.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with price information
    horizon : int | str, default 1
        Horizon for forward-looking labels:
        - int: Number of bars to look ahead
        - str: Duration string (e.g., '1h', '30m', '1d') for time-based horizon
    method : str, default "returns"
        Labeling method:
        - "returns": (price[t+h] - price[t]) / price[t]
        - "log_returns": log(price[t+h] / price[t])
        - "binary": 1 if price[t+h] > price[t] else -1
    price_col : str | None, default None
        Name of the price column to use
    group_col : str | list[str] | None, default None
        Column(s) to group by for per-asset labels. If None, auto-detects from
        common column names: 'symbol', 'product' (futures), or uses composite
        grouping if 'position' column exists (e.g., for futures contract months).
        Pass an empty list explicitly to disable grouping.
    timestamp_col : str | None, default None
        Column to use for chronological sorting. If None, auto-detects from
        column dtype (pl.Datetime, pl.Date). Required for time-based horizons.
    tolerance : str | None, default None
        Maximum time gap allowed for time-based horizons (e.g., '2m').
        Only used when horizon is a duration string. If the nearest future
        price is beyond this tolerance, the label will be null.
    config : LabelingConfig | None, default None
        Optional column contract source. If provided, `price_col`, `timestamp_col`,
        and `group_col` default to config values when omitted.
    contract : DataContractConfig | None, default None
        Optional shared dataframe contract. Used after config and before defaults.

    Returns
    -------
    pl.DataFrame
        Original data with additional label column.
        Last `horizon` values per group will be null (insufficient future data).

    Examples
    --------
    >>> # Bar-based: 5-period forward returns (unchanged API)
    >>> labeled = fixed_time_horizon_labels(df, horizon=5, method="returns")
    >>>
    >>> # Time-based: 1-hour forward returns
    >>> labeled = fixed_time_horizon_labels(df, horizon="1h", method="returns")
    >>>
    >>> # Time-based with tolerance for irregular data
    >>> labeled = fixed_time_horizon_labels(
    ...     df, horizon="15m", tolerance="2m", method="returns"
    ... )
    >>>
    >>> # Binary classification (up/down)
    >>> labeled = fixed_time_horizon_labels(df, horizon=1, method="binary")
    >>>
    >>> # Log returns for ML training
    >>> labeled = fixed_time_horizon_labels(df, horizon="1d", method="log_returns")

    Notes
    -----
    This is a simple labeling method that:
    - Uses future information (forward-looking)
    - Cannot be used for live prediction (requires future data)
    - Best for supervised learning model training
    - Last `horizon` rows will have null labels

    **Time-based horizons**: When horizon is a duration string (e.g., '1h'),
    the function uses ``join_asof`` to find the first available price at or
    after that time in the future. This is useful for:
    - Irregular data (trade bars) where you want time-based returns
    - Multi-frequency workflows where time semantics matter
    - Calendar-aware operations across trading breaks

    **Bar-based horizons**: When horizon is an integer, the function uses
    simple shift operations for maximum performance.

    **Important**: Data is automatically sorted by [group_cols, timestamp] before
    computing labels. This is required because Polars ``.over()`` preserves row
    order and does not sort within groups. The result is returned sorted
    chronologically within each group.

    References
    ----------
    .. [1] De Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 3: Labeling.

    See Also
    --------
    triple_barrier_labels : Path-dependent labeling with profit/loss targets
    trend_scanning_labels : De Prado's trend scanning method
    """
    if method not in ["returns", "log_returns", "binary"]:
        raise ValueError(f"Unknown method: {method}. Use 'returns', 'log_returns', or 'binary'")

    # Determine if time-based or bar-based
    is_time_based = isinstance(horizon, str) and is_duration_string(horizon)
    resolved_price_col, resolved_ts_col, resolved_group_cols = resolve_labeling_columns(
        data=data,
        price_col=price_col,
        timestamp_col=timestamp_col,
        group_col=group_col,
        config=config,
        contract=contract,
        require_timestamp=False,
    )
    if is_time_based and resolved_ts_col is None:
        raise ValueError(
            "Time-based horizon requires a timestamp column. "
            "Provide timestamp_col parameter or ensure data has a datetime column.",
        )

    if is_time_based:
        return _time_based_horizon_labels(
            data=data,
            horizon=horizon,  # type: ignore[arg-type]
            method=method,
            price_col=resolved_price_col,
            group_cols=resolved_group_cols,
            timestamp_col=resolved_ts_col,
            tolerance=tolerance,
        )
    else:
        # Bar-based: validate horizon is positive int
        if isinstance(horizon, str):
            raise ValueError(
                f"Invalid horizon: '{horizon}'. For bar-based labels use an integer, "
                f"for time-based labels use a duration string like '1h', '30m'."
            )
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        return _bar_based_horizon_labels(
            data=data,
            horizon=horizon,
            method=method,
            price_col=resolved_price_col,
            group_cols=resolved_group_cols,
            timestamp_col=resolved_ts_col,
        )


def _bar_based_horizon_labels(
    data: pl.DataFrame,
    horizon: int,
    method: str,
    price_col: str,
    group_cols: list[str],
    timestamp_col: str | None,
) -> pl.DataFrame:
    """Bar-based horizon labels using shift operations (original implementation)."""
    # Sort data chronologically within groups for correct shift operations
    if timestamp_col:
        sort_cols = group_cols + [timestamp_col] if group_cols else [timestamp_col]
        data = data.sort(sort_cols)

    # Get price column
    prices = pl.col(price_col)

    if group_cols:
        future_prices = prices.shift(-horizon).over(group_cols)
    else:
        future_prices = prices.shift(-horizon)

    # Compute label based on method
    if method == "returns":
        label = (future_prices - prices) / prices
        label_name = f"label_return_{horizon}p"
    elif method == "log_returns":
        label = (future_prices / prices).log()
        label_name = f"label_log_return_{horizon}p"
    elif method == "binary":
        # 1 if price goes up, -1 if down, 0 if no change, null if no future data
        label = (
            pl.when(future_prices.is_null())
            .then(pl.lit(None))
            .when(future_prices > prices)
            .then(1)
            .when(future_prices < prices)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
        )
        label_name = f"label_direction_{horizon}p"

    # Add label column to data
    return data.with_columns(label.alias(label_name))


def _time_based_horizon_labels(
    data: pl.DataFrame,
    horizon: str,
    method: str,
    price_col: str,
    group_cols: list[str],
    timestamp_col: str | None,
    tolerance: str | None,
) -> pl.DataFrame:
    """Time-based horizon labels using join_asof."""
    if timestamp_col is None:
        raise ValueError(
            "Time-based horizon requires a timestamp column. "
            "Provide timestamp_col parameter or ensure data has a datetime column."
        )

    # Sort data chronologically within groups
    sort_cols = group_cols + [timestamp_col] if group_cols else [timestamp_col]
    data = data.sort(sort_cols)

    # Parse duration for label naming
    td = parse_duration(horizon)
    # Create a clean label suffix (e.g., "1h" -> "1h", "1d2h" -> "1d2h")
    label_suffix = horizon.lower().replace(" ", "")

    # Get future prices using join_asof
    future_prices, valid_mask = get_future_price_at_time(
        data=data,
        time_horizon=td,
        price_col=price_col,
        timestamp_col=timestamp_col,
        tolerance=tolerance,
        group_cols=group_cols if group_cols else None,
    )

    # Current prices
    current_prices = data[price_col]

    # Compute label based on method
    if method == "returns":
        label = (future_prices - current_prices) / current_prices
        label_name = f"label_return_{label_suffix}"
    elif method == "log_returns":
        label = (future_prices / current_prices).log()
        label_name = f"label_log_return_{label_suffix}"
    elif method == "binary":
        # 1 if price goes up, -1 if down, 0 if no change, null if no future data
        label = (
            pl.when(future_prices.is_null())
            .then(pl.lit(None))
            .when(future_prices > current_prices)
            .then(pl.lit(1))
            .when(future_prices < current_prices)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
        )
        label_name = f"label_direction_{label_suffix}"

    # Mask invalid joins (beyond tolerance)
    if tolerance is not None:
        label = pl.when(valid_mask).then(label).otherwise(pl.lit(None))

    # Add label column to data
    return data.with_columns(label.alias(label_name))


def trend_scanning_labels(
    data: pl.DataFrame,
    min_window: int = 5,
    max_window: int = 50,
    step: int = 1,
    price_col: str | None = None,
    timestamp_col: str | None = None,
    *,
    config: LabelingConfig | None = None,
    contract: DataContractConfig | None = None,
) -> pl.DataFrame:
    """Generate labels using De Prado's trend scanning method.

    For each observation, fits linear trends over windows of varying lengths
    and selects the window with the highest absolute t-statistic. The label
    is assigned based on the trend direction (sign of the t-statistic).

    This method is more robust than fixed-horizon labeling as it adapts to
    the local trend structure in the data.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with price information
    min_window : int, default 5
        Minimum window size to scan
    max_window : int, default 50
        Maximum window size to scan
    step : int, default 1
        Step size for window scanning
    price_col : str | None, default None
        Name of the price column to use
    timestamp_col : str | None, default None
        Column to use for chronological sorting. If None, auto-detects from
        column dtype (pl.Datetime, pl.Date). Required for correct scanning.
    config : LabelingConfig | None, default None
        Optional column contract source. If provided, `price_col` and
        `timestamp_col` default to config values when omitted.
    contract : DataContractConfig | None, default None
        Optional shared dataframe contract. Used after config and before defaults.

    Returns
    -------
    pl.DataFrame
        Original data with additional columns:
        - label: ±1 based on trend direction
        - t_value: t-statistic of the selected trend
        - optimal_window: window size with highest |t-value|

    Examples
    --------
    >>> # Scan windows from 5 to 50 bars
    >>> labeled = trend_scanning_labels(df, min_window=5, max_window=50)
    >>>
    >>> # Fast scanning with larger steps
    >>> labeled = trend_scanning_labels(df, min_window=10, max_window=100, step=5)

    Notes
    -----
    The trend scanning method:
    1. For each observation, scans forward with windows of varying lengths
    2. Fits a linear regression to each window
    3. Computes t-statistic for the slope coefficient
    4. Selects the window with highest absolute t-statistic
    5. Assigns label = sign(t-statistic)

    This approach:
    - Adapts to local trend structure
    - More robust than fixed horizons
    - Computationally expensive (O(n * m) where m = window range)

    References
    ----------
    .. [1] De Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 18: Entropy Features (Section on Trend Scanning).

    See Also
    --------
    fixed_time_horizon_labels : Simple fixed-horizon labeling
    triple_barrier_labels : Path-dependent labeling with barriers

    Notes
    -----
    **Important**: Data is automatically sorted by timestamp before scanning.
    This is required because the algorithm scans forward in row order.
    The result is returned sorted chronologically.
    """
    from scipy import stats

    if min_window < 2:
        raise ValueError("min_window must be at least 2")
    if max_window <= min_window:
        raise ValueError("max_window must be greater than min_window")
    if step < 1:
        raise ValueError("step must be at least 1")
    resolved_price_col, resolved_ts_col, _ = resolve_labeling_columns(
        data=data,
        price_col=price_col,
        timestamp_col=timestamp_col,
        group_col=[],
        config=config,
        contract=contract,
    )

    # Sort data chronologically for correct forward scanning
    if resolved_ts_col:
        data = data.sort(resolved_ts_col)

    # Extract prices as numpy array for faster computation
    prices = data[resolved_price_col].to_numpy()
    n = len(prices)

    # Initialize result arrays
    labels = np.full(n, np.nan)
    t_values = np.full(n, np.nan)
    windows = np.full(n, np.nan)

    # Scan each observation
    for i in range(n - min_window):
        best_t = 0.0
        best_window = min_window

        # Scan windows of different lengths
        for window in range(min_window, min(max_window, n - i), step):
            # Extract window
            window_prices = prices[i : i + window]
            x = np.arange(window)
            y = window_prices

            # Fit linear regression
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # Compute t-statistic
                t_stat = slope / std_err if std_err > 0 else 0.0

                # Keep window with highest |t|
                if abs(t_stat) > abs(best_t):
                    best_t = t_stat
                    best_window = window
            except (ValueError, RuntimeError):
                # Handle numerical issues
                continue

        # Assign label based on trend direction
        labels[i] = 1 if best_t > 0 else -1
        t_values[i] = best_t
        windows[i] = best_window

    # Add results to dataframe
    # Convert NaN to None for Polars compatibility
    label_series = pl.Series("label", labels)
    label_series = label_series.fill_nan(None).cast(pl.Int8)

    t_value_series = pl.Series("t_value", t_values)
    window_series = pl.Series("optimal_window", windows).fill_nan(None).cast(pl.Int32)

    return data.with_columns([label_series, t_value_series, window_series])


__all__ = [
    "fixed_time_horizon_labels",
    "trend_scanning_labels",
]
