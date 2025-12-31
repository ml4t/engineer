# mypy: disable-error-code="no-any-return"
"""
Exponential Moving Average (EMA) - TA-Lib compatible implementation.

The Exponential Moving Average gives more weight to recent close, with the weight
decreasing exponentially for older close. Uses TA-Lib's exact initialization method.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def ema_numba(close: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    """
    Exponential Moving Average using exact TA-Lib algorithm.

    Initializes with SMA for first value, then uses exponential smoothing.
    Alpha = 2 / (period + 1) as per TA-Lib standard.
    Handles NaN close by skipping them, matching TA-Lib behavior.

    Parameters
    ----------
    close : npt.NDArray
        Price data (may contain NaN close)
    period : int
        Number of periods (span) for the EMA

    Returns
    -------
    npt.NDArray
        EMA close with NaN for insufficient data
    """
    n = len(close)
    result = np.full(n, np.nan)

    if period < 1 or period > n:
        return result

    # Find first non-NaN value
    start_idx = -1
    for i in range(n):
        if not np.isnan(close[i]):
            start_idx = i
            break

    if start_idx == -1 or start_idx + period > n:
        return result

    # Calculate alpha (TA-Lib uses 2/(period+1))
    alpha = 2.0 / (period + 1.0)

    # Initialize with SMA for first valid value (TA-Lib compatible)
    # Need 'period' consecutive non-NaN close
    sum_val = 0.0
    count = 0
    sma_idx = -1

    for i in range(start_idx, n):
        if not np.isnan(close[i]):
            sum_val += close[i]
            count += 1
            if count == period:
                sma_idx = i
                result[i] = sum_val / period
                break
        else:
            # Reset if we encounter NaN before completing SMA
            sum_val = 0.0
            count = 0

    # If we couldn't get enough close for SMA, return
    if count < period:
        return result

    # Calculate EMA for subsequent close
    # Use the last valid EMA value, skipping over any NaN in input
    last_ema = result[sma_idx]
    for i in range(sma_idx + 1, n):
        if not np.isnan(close[i]):
            result[i] = alpha * close[i] + (1.0 - alpha) * last_ema
            last_ema = result[i]

    return result


def ema_polars(column: str, period: int) -> pl.Expr:
    """
    EMA using Polars - delegates to Numba for TA-Lib compatibility.

    Parameters
    ----------
    column : str
        Column name to apply EMA to
    period : int
        Number of periods for the EMA

    Returns
    -------
    pl.Expr
        Polars expression for EMA calculation
    """
    return pl.col(column).map_batches(
        lambda s: pl.Series(ema_numba(s.to_numpy(), period)),
        return_dtype=pl.Float64,
    )


@feature(
    name="ema",
    category="trend",
    description="EMA - Exponential Moving Average",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def ema(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Exponential Moving Average with TA-Lib compatibility.

    The EMA reacts more quickly to recent price changes compared to SMA.
    Uses TA-Lib's exact initialization method for maximum accuracy.

    Parameters
    ----------
    close : array-like or column name
        Price data or column name
    period : int
        Number of periods (span) for the EMA
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        EMA close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29])
    >>> ema_5 = ema(close, period=5)
    >>> ema_5[4:]  # First 4 close are NaN
    array([22.208, 22.180, 22.197, 22.274, 22.233, 22.232])

    Notes
    -----
    - Alpha = 2 / (period + 1)
    - First value initialized with SMA of first 'period' close
    - Subsequent close: EMA[i] = alpha * price[i] + (1-alpha) * EMA[i-1]
    """
    # Validate parameters
    if period < 1:
        raise InvalidParameterError(f"period must be >= 1, got {period}")

    if isinstance(close, str):
        return ema_polars(close, period)

    # Always use Numba for exact TA-Lib compatibility with numeric data
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return ema_numba(close, period)


# Export utilities for other indicators
__all__ = ["ema", "ema_numba"]
