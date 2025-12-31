# mypy: disable-error-code="no-any-return"
"""
MIDPOINT - Midpoint over period.

The midpoint is the average of the highest and lowest close over a given period.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def midpoint_numba(close: npt.NDArray[np.float64], timeperiod: int = 14) -> npt.NDArray[np.float64]:
    """
    MIDPOINT calculation using optimized sliding window.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    npt.NDArray
        Midpoint close
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Need at least timeperiod close
    if n < timeperiod:
        return result

    # Optimized sliding window with vectorized min/max
    for i in range(timeperiod - 1, n):
        start_idx = i - timeperiod + 1
        window = close[start_idx : i + 1]
        highest: float = float(np.max(window))
        lowest: float = float(np.min(window))
        result[i] = (highest + lowest) * 0.5

    return result


def midpoint_polars(col: str, timeperiod: int = 14) -> pl.Expr:
    """
    MIDPOINT using Polars expressions.

    Parameters
    ----------
    col : str
        Name of the column containing close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    pl.Expr
        Polars expression for MIDPOINT calculation
    """
    return pl.col(col).map_batches(
        lambda x: pl.Series(midpoint_numba(x.to_numpy(), timeperiod)),
        return_dtype=pl.Float64,
    )


@feature(
    name="midpoint",
    category="trend",
    description="Midpoint - average of highest and lowest close",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def midpoint(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    MIDPOINT - Midpoint over period.

    The midpoint indicator calculates the middle point between the highest
    and lowest close over a specified period. It's a simple way to identify
    the center of the trading range.

    Formula:
    MIDPOINT = (Highest Value + Lowest Value) / 2

    Parameters
    ----------
    close : array-like or column name
        Input close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    array or Polars expression
        Midpoint close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([10, 12, 11, 13, 14, 12, 11, 15, 13, 12])
    >>> midpoint_values = midpoint(close, timeperiod=5)

    Notes
    -----
    - The midpoint represents the center of the price range
    - Can be used as a simple support/resistance indicator
    - Also known as the "range midpoint" or "price channel midpoint"
    - Similar to a moving average but based on extremes rather than all close
    """
    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return midpoint_polars(close, timeperiod)

    # Convert to numpy array
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if timeperiod < 2:
        raise ValueError(f"timeperiod must be >= 2, got {timeperiod}")

    return midpoint_numba(close, timeperiod)


# Export the main function
__all__ = ["midpoint"]
