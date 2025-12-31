# mypy: disable-error-code="no-any-return"
"""
Average Price (AVGPRICE) - TA-Lib compatible implementation.

AVGPRICE = (Open + High + Low + Close) / 4

Simple average of the four main price points (OHLC).
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def avgprice_numba(
    open: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Average Price calculation using optimized Numba.

    AVGPRICE = (Open + High + Low + Close) / 4

    Parameters
    ----------
    open : npt.NDArray
        Open close
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    close : npt.NDArray
        Close close

    Returns
    -------
    npt.NDArray
        Average price close
    """
    # Vectorized calculation - much faster than loop
    return (open + high + low + close) * 0.25


def avgprice_polars(
    open_col: str,
    high: str,
    low_col: str,
    close_col: str,
) -> pl.Expr:
    """
    Average Price using Polars expressions.

    Parameters
    ----------
    open_col : str
        Name of open price column
    high : str
        Name of high price column
    low_col : str
        Name of low price column
    close_col : str
        Name of close price column

    Returns
    -------
    pl.Expr
        Polars expression for average price calculation
    """
    # Simple average of OHLC
    return (pl.col(open_col) + pl.col(high) + pl.col(low_col) + pl.col(close_col)) / 4.0


@feature(
    name="avgprice",
    category="price_transform",
    description="AVGPRICE - Average Price",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def avgprice(
    open: npt.NDArray[np.float64] | pl.Series | str,
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Average Price (AVGPRICE).

    Calculates the average of the four main price points: Open, High, Low, Close.
    This provides a simple representation of the overall price level for a period.

    Parameters
    ----------
    open : array-like or column name
        Open close
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    close : array-like or column name
        Close close

    Returns
    -------
    array or Polars expression
        Average price close

    Examples
    --------
    >>> import numpy as np
    >>> open = np.array([100.0, 101.0, 102.0])
    >>> high = np.array([102.0, 103.0, 104.0])
    >>> low = np.array([99.0, 100.0, 101.0])
    >>> close = np.array([101.0, 102.0, 103.0])
    >>> avg = avgprice(open, high, low, close)
    >>> avg
    array([100.5, 101.5, 102.5])

    Notes
    -----
    - Simple arithmetic mean of OHLC close
    - Returns NaN if any input price is NaN
    - Useful as a simple price level indicator
    - Often used as input to other indicators
    """
    # Handle string inputs (Polars column names)
    if (
        isinstance(open, str)
        and isinstance(high, str)
        and isinstance(low, str)
        and isinstance(close, str)
    ):
        return avgprice_polars(open, high, low, close)

    # Convert to numpy arrays
    if isinstance(open, pl.Series):
        open = open.to_numpy()
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if len(open) != len(high) or len(open) != len(low) or len(open) != len(close):
        raise ValueError("open, high, low, and close must have the same length")

    return avgprice_numba(open, high, low, close)


# Export the main function
__all__ = ["avgprice"]
