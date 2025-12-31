# mypy: disable-error-code="no-any-return,return-value"
"""
Simple Moving Average (SMA) - TA-Lib compatible implementation.

The Simple Moving Average is the most basic moving average, calculated as the
arithmetic mean of close over a specified period.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=False, fastmath=False)  # type: ignore[misc]
def sma_numba(close: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    """
    Simple Moving Average using Numba JIT compilation.

    Optimized with running sum for O(n) complexity instead of O(n*period).

    Parameters
    ----------
    close : npt.NDArray
        Price data
    period : int
        Number of periods for the moving average

    Returns
    -------
    npt.NDArray
        SMA close with NaN for insufficient data
    """
    n = len(close)
    result = np.full(n, np.nan)

    if period < 1 or period > n:
        return result

    # Calculate first SMA
    sum_val = 0.0
    count = 0
    for i in range(period):
        if not np.isnan(close[i]):
            sum_val += close[i]
            count += 1

    if count == period:
        result[period - 1] = sum_val / period

    # Use running sum for efficiency
    for i in range(period, n):
        # Skip if current or previous value is NaN
        if np.isnan(close[i]) or np.isnan(close[i - period]):
            # Recalculate sum for this window
            sum_val = 0.0
            count = 0
            for j in range(i - period + 1, i + 1):
                if not np.isnan(close[j]):
                    sum_val += close[j]
                    count += 1
            if count == period:
                result[i] = sum_val / period
        else:
            sum_val = sum_val - close[i - period] + close[i]
            result[i] = sum_val / period

    return result


def sma_polars(column: str, period: int) -> pl.Expr:
    """
    Simple Moving Average using Polars native operations.

    Parameters
    ----------
    column : str
        Column name to apply SMA to
    period : int
        Number of periods for the moving average

    Returns
    -------
    pl.Expr
        Polars expression for SMA calculation
    """
    return pl.col(column).rolling_mean(window_size=period)


@feature(
    name="sma",
    category="trend",
    description="SMA - Simple Moving Average",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def sma(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int,
    implementation: Literal["auto", "numba", "polars"] = "auto",
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Simple Moving Average with automatic implementation selection.

    Automatically chooses the most appropriate implementation based on input type:
    - String input → Polars expression (for DataFrame operations)
    - Numeric input → Numba implementation (for performance)

    Parameters
    ----------
    close : array-like or column name
        Price data or column name (for Polars expressions)
    period : int
        Number of periods for the moving average
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        SMA close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> sma_5 = sma(close, period=5)
    >>> sma_5[4:]  # First 4 close are NaN
    array([3., 4., 5., 6., 7., 8.])

    >>> import polars as pl
    >>> df = pl.DataFrame({'close': close})
    >>> df.with_columns(sma_5=sma('close', period=5))
    """
    # Validate parameters
    if period < 1:
        raise InvalidParameterError(f"period must be >= 1, got {period}")

    if isinstance(close, str):
        return sma_polars(close, period)

    if implementation == "polars" or (implementation == "auto" and isinstance(close, pl.Series)):
        if isinstance(close, np.ndarray):
            series = pl.Series(close)
            return series.rolling_mean(window_size=period).to_numpy()
        return close.rolling_mean(window_size=period)
    # Use Numba (default for numpy arrays and best performance)
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return sma_numba(close, period)


# Export the main function
__all__ = ["sma"]
