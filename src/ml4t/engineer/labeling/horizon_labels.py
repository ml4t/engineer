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

import numpy as np
import polars as pl

from ml4t.engineer.core.exceptions import DataValidationError

# Common grouping column names for different asset classes
_DEFAULT_GROUP_COLS = ["symbol", "product", "ticker"]


def _resolve_group_cols(
    data: pl.DataFrame,
    group_col: str | list[str] | None,
) -> list[str]:
    """Resolve grouping columns for label computation.

    Auto-detects grouping columns if not explicitly specified, handling:
    - Equities: 'symbol' column
    - Futures: 'product' column, plus 'position' for contract months
    - Explicit specification via group_col parameter

    Args:
        data: Input DataFrame
        group_col: User-specified grouping column(s), or None for auto-detection

    Returns:
        List of column names to group by (empty list means no grouping)
    """
    # Explicit specification
    if group_col is not None:
        if isinstance(group_col, str):
            return [group_col] if group_col in data.columns else []
        elif isinstance(group_col, list):
            # Return only columns that exist
            return [c for c in group_col if c in data.columns]

    # Auto-detection: Find first matching default column
    detected_cols = []
    for col in _DEFAULT_GROUP_COLS:
        if col in data.columns:
            detected_cols.append(col)
            break  # Use first match only

    # Check for position column (futures contract months)
    # If product + position exist, group by both to handle continuous futures correctly
    if "position" in data.columns:
        # Add position to grouping if not already covered
        if detected_cols and detected_cols[0] in ["product", "symbol"]:
            detected_cols.append("position")
        elif not detected_cols:
            # If only position exists, use it alone
            detected_cols.append("position")

    return detected_cols


def fixed_time_horizon_labels(
    data: pl.DataFrame,
    horizon: int = 1,
    method: str = "returns",
    price_col: str = "close",
    group_col: str | list[str] | None = None,
) -> pl.DataFrame:
    """Generate forward-looking labels based on fixed time horizon.

    Creates labels by looking ahead a fixed number of periods and computing
    the return or direction of price movement. Commonly used for supervised
    learning in financial forecasting.

    Parameters
    ----------
    data : pl.DataFrame
        Input data with price information
    horizon : int, default 1
        Number of periods to look ahead
    method : str, default "returns"
        Labeling method:
        - "returns": (price[t+h] - price[t]) / price[t]
        - "log_returns": log(price[t+h] / price[t])
        - "binary": 1 if price[t+h] > price[t] else -1
    price_col : str, default "close"
        Name of the price column to use
    group_col : str | list[str] | None, default None
        Column(s) to group by for per-asset labels. If None, auto-detects from
        common column names: 'symbol', 'product' (futures), or uses composite
        grouping if 'position' column exists (e.g., for futures contract months).
        Pass an empty list explicitly to disable grouping.

    Returns
    -------
    pl.DataFrame
        Original data with additional label column.
        Last `horizon` values per group will be null (insufficient future data).

    Examples
    --------
    >>> # Simple returns over 5-period horizon
    >>> labeled = fixed_time_horizon_labels(df, horizon=5, method="returns")
    >>>
    >>> # Binary classification (up/down)
    >>> labeled = fixed_time_horizon_labels(df, horizon=1, method="binary")
    >>>
    >>> # Log returns for ML training
    >>> labeled = fixed_time_horizon_labels(df, horizon=10, method="log_returns")

    Notes
    -----
    This is a simple labeling method that:
    - Uses future information (forward-looking)
    - Cannot be used for live prediction (requires future data)
    - Best for supervised learning model training
    - Last `horizon` rows will have null labels

    References
    ----------
    .. [1] De Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
           Chapter 3: Labeling.

    See Also
    --------
    triple_barrier_labels : Path-dependent labeling with profit/loss targets
    trend_scanning_labels : De Prado's trend scanning method
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if method not in ["returns", "log_returns", "binary"]:
        raise ValueError(f"Unknown method: {method}. Use 'returns', 'log_returns', or 'binary'")

    if price_col not in data.columns:
        raise DataValidationError(f"Column '{price_col}' not found in data")

    # Get price column
    prices = pl.col(price_col)

    # Determine grouping columns for multi-asset/multi-position data
    # This prevents labels from leaking across asset/contract boundaries
    group_cols = _resolve_group_cols(data, group_col)

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
        # 1 if price goes up, -1 if down, null if no change or no data
        label = (
            pl.when(future_prices > prices)
            .then(1)
            .when(future_prices < prices)
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8)
        )
        label_name = f"label_direction_{horizon}p"

    # Add label column to data
    return data.with_columns(label.alias(label_name))


def trend_scanning_labels(
    data: pl.DataFrame,
    min_window: int = 5,
    max_window: int = 50,
    step: int = 1,
    price_col: str = "close",
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
    price_col : str, default "close"
        Name of the price column to use

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
    """
    from scipy import stats

    if min_window < 2:
        raise ValueError("min_window must be at least 2")
    if max_window <= min_window:
        raise ValueError("max_window must be greater than min_window")
    if step < 1:
        raise ValueError("step must be at least 1")
    if price_col not in data.columns:
        raise DataValidationError(f"Column '{price_col}' not found in data")

    # Extract prices as numpy array for faster computation
    prices = data[price_col].to_numpy()
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
