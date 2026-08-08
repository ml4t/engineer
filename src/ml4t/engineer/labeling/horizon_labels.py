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

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from ml4t.engineer.core.exceptions import DataValidationError
from ml4t.engineer.labeling.utils import (
    get_future_price_at_time,
    is_duration_string,
    parse_duration,
    resolve_labeling_columns,
    validate_price_no_nans,
)

if TYPE_CHECKING:  # pragma: no cover - imports used only by static analysis
    from ml4t.engineer.config import DataContractConfig, LabelingConfig


def _linear_regression_slope_stderr(
    x: npt.NDArray[np.int64],
    y: npt.NDArray[np.float64],
) -> tuple[float, float]:
    if len(x) != len(y) or len(x) < 2 or not np.isfinite(y).all():
        raise ValueError("linear regression requires equal finite arrays with at least two values")

    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    sum_squared_x = float(np.dot(centered_x, centered_x))
    if sum_squared_x == 0:
        raise ValueError("linear regression requires varying x values")

    slope = float(np.dot(centered_x, centered_y) / sum_squared_x)
    if len(x) == 2:
        return slope, 0.0

    residuals = centered_y - slope * centered_x
    residual_variance = float(np.dot(residuals, residuals) / (len(x) - 2))
    return slope, float(np.sqrt(residual_variance / sum_squared_x))


def fixed_time_horizon_labels(
    data: pl.DataFrame,
    horizon: int | str | None = None,
    method: Literal["returns", "log_returns", "binary"] | None = None,
    price_col: str | None = None,
    group_col: str | list[str] | None = None,
    timestamp_col: str | None = None,
    tolerance: str | None = None,
    threshold: float | None = None,
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
    threshold : float | None, default None
        Nonnegative no-change band for binary labels. Returns above the threshold
        receive +1, returns below its negative receive -1, and returns inside the
        band receive 0.
    config : LabelingConfig | None, default None
        Fixed-horizon configuration. Its horizon, return method, threshold, and
        column mapping control execution. Conflicting explicit values are rejected.
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
    if config is not None:
        if config.method != "fixed_horizon":
            raise DataValidationError(
                "fixed_time_horizon_labels requires LabelingConfig.method='fixed_horizon'."
            )
        if horizon is not None and horizon != config.horizon:
            raise DataValidationError("Explicit horizon conflicts with config.horizon.")
        if method is not None and method != config.return_method:
            raise DataValidationError("Explicit method conflicts with config.return_method.")
        if threshold is not None and threshold != config.threshold:
            raise DataValidationError("Explicit threshold conflicts with config.threshold.")
        horizon = config.horizon
        method = config.return_method
        threshold = config.threshold

    horizon = 1 if horizon is None else horizon
    method = "returns" if method is None else method
    if method not in ["returns", "log_returns", "binary"]:
        raise ValueError(f"Unknown method: {method}. Use 'returns', 'log_returns', or 'binary'")
    if threshold is not None and (
        isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0
    ):
        raise ValueError("threshold must be a finite nonnegative number")
    if method != "binary" and threshold is not None:
        raise ValueError("threshold is supported only when method='binary'")
    binary_threshold = 0.0 if threshold is None else threshold

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

    validate_price_no_nans(data, resolved_price_col)

    if is_time_based:
        return _time_based_horizon_labels(
            data=data,
            horizon=horizon,  # type: ignore[arg-type]
            method=method,
            price_col=resolved_price_col,
            group_cols=resolved_group_cols,
            timestamp_col=resolved_ts_col,
            tolerance=tolerance,
            threshold=binary_threshold,
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
            threshold=binary_threshold,
        )


def _bar_based_horizon_labels(
    data: pl.DataFrame,
    horizon: int,
    method: str,
    price_col: str,
    group_cols: list[str],
    timestamp_col: str | None,
    threshold: float,
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
        returns = (future_prices - prices) / prices
        label = (
            pl.when(future_prices.is_null())
            .then(pl.lit(None))
            .when(returns > threshold)
            .then(1)
            .when(returns < -threshold)
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
    threshold: float,
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
        returns = (future_prices - current_prices) / current_prices
        label = (
            pl.when(future_prices.is_null())
            .then(pl.lit(None))
            .when(returns > threshold)
            .then(pl.lit(1))
            .when(returns < -threshold)
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


def _trend_scanning_single_group(
    data: pl.DataFrame,
    min_window: int,
    max_window: int,
    step: int,
    price_col: str,
    timestamp_col: str | None,
    t_value_threshold: float,
) -> pl.DataFrame:
    """Apply trend scanning to a single asset/group."""
    # Sort data chronologically for correct forward scanning
    if timestamp_col:
        data = data.sort(timestamp_col)

    # Extract prices as numpy array for faster computation
    prices = data[price_col].to_numpy()
    n = len(prices)

    # Initialize result arrays
    labels = np.full(n, np.nan)
    t_values = np.full(n, np.nan)
    windows = np.full(n, np.nan)
    perfect_fit_t = np.finfo(np.float64).max

    # Scan each observation
    for i in range(n - min_window + 1):
        best_t = 0.0
        best_window = min_window

        # Scan windows of different lengths
        for window in range(min_window, min(max_window, n - i) + 1, step):
            # Extract window
            window_prices = prices[i : i + window]
            x = np.arange(window)
            y = window_prices

            # Fit linear regression
            try:
                slope, std_err = _linear_regression_slope_stderr(x, y)

                # Compute t-statistic
                if std_err > 0:
                    t_stat = slope / std_err
                elif slope > 0:
                    t_stat = perfect_fit_t
                elif slope < 0:
                    t_stat = -perfect_fit_t
                else:
                    t_stat = 0.0

                # Keep window with highest |t|
                if abs(t_stat) > abs(best_t):
                    best_t = t_stat
                    best_window = window
            except (ValueError, RuntimeError):
                # Handle numerical issues
                continue

        if best_t == 0:
            continue
        t_values[i] = best_t
        windows[i] = best_window

        # Assign a label only when the selected trend reaches the requested significance.
        if abs(best_t) < t_value_threshold:
            continue
        if best_t > 0:
            labels[i] = 1
        else:
            labels[i] = -1

    # Add results to dataframe
    label_series = pl.Series("label", labels).fill_nan(None).cast(pl.Int8)
    t_value_series = pl.Series("t_value", t_values).fill_nan(None)
    window_series = pl.Series("optimal_window", windows).fill_nan(None).cast(pl.Int32)

    return data.with_columns([label_series, t_value_series, window_series])


def trend_scanning_labels(
    data: pl.DataFrame,
    min_window: int | None = None,
    max_window: int | None = None,
    step: int | None = None,
    price_col: str | None = None,
    timestamp_col: str | None = None,
    group_col: str | list[str] | None = None,
    *,
    t_value_threshold: float | None = None,
    config: LabelingConfig | None = None,
    contract: DataContractConfig | None = None,
) -> pl.DataFrame:
    """Generate labels using De Prado's trend scanning method.

    For each observation, fits linear trends over windows of varying lengths
    and selects the window with the highest absolute t-statistic. The label
    is assigned based on the trend direction when a non-zero t-statistic is
    found. Observations with no directional trend remain null.

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
    group_col : str | list[str] | None, default None
        Column(s) to group by for per-asset labels. If None, auto-detects from
        common column names: 'symbol', 'product', 'ticker'.
        Pass an empty list explicitly to disable grouping.
    t_value_threshold : float | None, default None
        Minimum absolute t-value required for a directional label. The selected
        t-value and window remain available when the label is null.
    config : LabelingConfig | None, default None
        Trend-scanning configuration. Its windows, step, significance threshold,
        and column mapping control execution. Conflicting explicit values are rejected.
    contract : DataContractConfig | None, default None
        Optional shared dataframe contract. Used after config and before defaults.

    Returns
    -------
    pl.DataFrame
        Original data with additional columns:
        - label: +1, -1, or null based on trend direction
        - t_value: t-statistic of the selected trend, or null
        - optimal_window: window size with highest |t-value|, or null

        Exact nonconstant linear fits use the largest finite float as the signed
        t-value. Constant windows have null outputs.

    Examples
    --------
    >>> # Scan windows from 5 to 50 bars
    >>> labeled = trend_scanning_labels(df, min_window=5, max_window=50)
    >>>
    >>> # Fast scanning with larger steps
    >>> labeled = trend_scanning_labels(df, min_window=10, max_window=100, step=5)
    >>>
    >>> # Panel data: per-asset scanning
    >>> labeled = trend_scanning_labels(df, group_col="symbol")

    Notes
    -----
    The trend scanning method:
    1. For each observation, scans forward with windows of varying lengths
    2. Fits a linear regression to each window
    3. Computes t-statistic for the slope coefficient
    4. Selects the window with highest absolute t-statistic
    5. Assigns label = sign(t-statistic), or null if no directional trend is found

    **Important**: Data is automatically sorted by [group_col, timestamp] before
    scanning. This is required because the algorithm scans forward in row order.

    References
    ----------
    .. [1] De Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 18: Entropy Features (Section on Trend Scanning).

    See Also
    --------
    fixed_time_horizon_labels : Simple fixed-horizon labeling
    triple_barrier_labels : Path-dependent labeling with barriers
    """
    if config is not None:
        if config.method != "trend_scanning":
            raise DataValidationError(
                "trend_scanning_labels requires LabelingConfig.method='trend_scanning'."
            )
        conflicts = (
            (min_window is not None and min_window != config.min_horizon),
            (max_window is not None and max_window != config.max_horizon),
            (step is not None and step != config.step),
            (t_value_threshold is not None and t_value_threshold != config.t_value_threshold),
        )
        if any(conflicts):
            raise DataValidationError("Explicit trend-scanning arguments conflict with config.")
        min_window = config.min_horizon
        max_window = config.max_horizon
        step = config.step
        t_value_threshold = config.t_value_threshold

    min_window = 5 if min_window is None else min_window
    max_window = 50 if max_window is None else max_window
    step = 1 if step is None else step
    t_value_threshold = 0.0 if t_value_threshold is None else t_value_threshold

    if min_window < 2:
        raise ValueError("min_window must be at least 2")
    if max_window < min_window:
        raise ValueError("max_window must be greater than or equal to min_window")
    if step < 1:
        raise ValueError("step must be at least 1")
    if not math.isfinite(t_value_threshold) or t_value_threshold < 0:
        raise ValueError("t_value_threshold must be a finite nonnegative number")
    resolved_price_col, resolved_ts_col, group_cols = resolve_labeling_columns(
        data=data,
        price_col=price_col,
        timestamp_col=timestamp_col,
        group_col=group_col,
        config=config,
        contract=contract,
    )

    validate_price_no_nans(data, resolved_price_col)

    if group_cols:
        sort_cols = group_cols + ([resolved_ts_col] if resolved_ts_col else [])
        sorted_data = data.sort(sort_cols)
        grouped_frames = sorted_data.partition_by(group_cols, maintain_order=True)

        grouped_results = [
            _trend_scanning_single_group(
                data=group_df,
                min_window=min_window,
                max_window=max_window,
                step=step,
                price_col=resolved_price_col,
                timestamp_col=resolved_ts_col,
                t_value_threshold=t_value_threshold,
            )
            for group_df in grouped_frames
        ]
        return pl.concat(grouped_results, how="vertical")

    return _trend_scanning_single_group(
        data=data,
        min_window=min_window,
        max_window=max_window,
        step=step,
        price_col=resolved_price_col,
        timestamp_col=resolved_ts_col,
        t_value_threshold=t_value_threshold,
    )


__all__ = [
    "fixed_time_horizon_labels",
    "trend_scanning_labels",
]
