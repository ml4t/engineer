# mypy: disable-error-code="no-any-return"
"""
Normalized Average True Range (NATR) - TA-Lib compatible implementation.

NATR = (ATR / Close) * 100

Normalizes ATR by expressing it as a percentage of the closing price,
making it easier to compare volatility across different price levels.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature

from .atr import atr_numba


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def natr_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Normalized ATR calculation using Numba.

    NATR = (ATR / Close) * 100

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    close : npt.NDArray
        Close close
    period : int, default 14
        Number of periods for ATR calculation

    Returns
    -------
    npt.NDArray
        NATR close as percentage with NaN for insufficient data
    """
    # Calculate ATR first
    atr_values = atr_numba(high, low, close, period)

    # Normalize by close price and convert to percentage
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(n):
        if not np.isnan(atr_values[i]) and close[i] != 0:
            result[i] = (atr_values[i] / close[i]) * 100.0
        elif not np.isnan(atr_values[i]) and close[i] == 0:
            # Handle division by zero - TA-Lib returns inf
            result[i] = np.inf

    return result


def natr_polars(
    high: str,
    low_col: str,
    close_col: str,
    period: int = 14,
) -> pl.Expr:
    """
    Normalized ATR using Polars expressions.

    Parameters
    ----------
    high : str
        Name of high price column
    low_col : str
        Name of low price column
    close_col : str
        Name of close price column
    period : int, default 14
        Number of periods for ATR calculation

    Returns
    -------
    pl.Expr
        Polars expression for NATR calculation
    """
    # Import atr_polars here to avoid circular import
    from .atr import atr_polars

    # Calculate ATR
    atr_expr = atr_polars(high, low_col, close_col, period)

    # Normalize by close price and convert to percentage
    # Handle division by zero gracefully
    return (atr_expr / pl.col(close_col)) * 100.0


@feature(
    name="natr",
    category="volatility",
    description="NATR - Normalized ATR (percentage)",
    lookback="period",
    value_range=(0.0, 100.0),
    normalized=True,
    formula="NATR = (ATR / close) * 100",
    ta_lib_compatible=True,
)
def natr(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Normalized Average True Range (NATR).

    NATR normalizes ATR by expressing it as a percentage of the closing price.
    This makes it easier to compare volatility across instruments with different
    price levels.

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    close : array-like or column name
        Close close
    period : int, default 14
        Number of periods for ATR calculation
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        NATR close as percentage

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([10.0, 11.0, 12.0, 11.5, 10.5, 11.0])
    >>> low = np.array([9.0, 9.5, 10.5, 10.0, 9.5, 10.0])
    >>> close = np.array([9.5, 10.5, 11.0, 10.5, 10.0, 10.5])
    >>> natr_14 = natr(high, low, close, period=3)
    >>> # Returns ATR as percentage of close price

    Notes
    -----
    - NATR = (ATR / Close) * 100
    - Useful for comparing volatility across different price levels
    - Division by zero returns inf (matching TA-Lib behavior)
    - Standard period is 14 days
    """
    # Validate parameters
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")

    # Handle string inputs (Polars column names)
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        return natr_polars(high, low, close, period)

    # Convert to numpy arrays
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, and close must have the same length")

    return natr_numba(high, low, close, period)


# Export the main function
__all__ = ["natr"]
