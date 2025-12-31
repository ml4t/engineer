# mypy: disable-error-code="no-any-return"
"""
Weighted Close Price (WCLPRICE) - TA-Lib compatible implementation.

WCLPRICE = (High + Low + 2*Close) / 4

Gives double weight to the closing price compared to high and low.
This emphasizes the importance of the closing price in the calculation.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def wclprice_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Weighted Close Price calculation using optimized Numba.

    WCLPRICE = (High + Low + 2*Close) / 4

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    close : npt.NDArray
        Close close

    Returns
    -------
    npt.NDArray
        Weighted close price close
    """
    # Vectorized calculation within Numba - fastest approach
    return (high + low + 2.0 * close) / 4.0


def wclprice_polars(high: str, low_col: str, close_col: str) -> pl.Expr:
    """
    Weighted Close Price using Polars expressions.

    Parameters
    ----------
    high : str
        Name of high price column
    low_col : str
        Name of low price column
    close_col : str
        Name of close price column

    Returns
    -------
    pl.Expr
        Polars expression for weighted close price calculation
    """
    # (High + Low + 2*Close) / 4
    return (pl.col(high) + pl.col(low_col) + 2.0 * pl.col(close_col)) / 4.0


@feature(
    name="wclprice",
    category="price_transform",
    description="WCLPRICE - Weighted Close Price",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def wclprice(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    implementation: Literal["auto", "numba", "polars"] = "auto",
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Weighted Close Price (WCLPRICE).

    Calculates a weighted average of high, low, and close close, giving
    double weight to the closing price. This reflects the common belief that
    the closing price is the most important price of the period.

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    close : array-like or column name
        Close close
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        Weighted close price close

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([102.0, 103.0, 104.0])
    >>> low = np.array([99.0, 100.0, 101.0])
    >>> close = np.array([101.0, 102.0, 103.0])
    >>> wcl = wclprice(high, low, close)
    >>> wcl
    array([100.75, 101.75, 102.75])

    Notes
    -----
    - Gives double weight to closing price
    - Formula: (High + Low + 2*Close) / 4
    - Returns NaN if any input price is NaN
    - Often used in momentum and trend indicators
    - Emphasizes the importance of the closing price
    """
    # Handle string inputs (Polars column names) or explicit polars request
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        return wclprice_polars(high, low, close)
    if implementation == "polars":
        raise ValueError(
            "Polars implementation requires all inputs to be column names (strings)",
        )

    # Convert to numpy arrays
    high = high.to_numpy() if isinstance(high, pl.Series) else np.asarray(high, dtype=np.float64)
    low = low.to_numpy() if isinstance(low, pl.Series) else np.asarray(low, dtype=np.float64)
    if isinstance(close, pl.Series):
        close = close.to_numpy()
    else:
        close = np.asarray(close, dtype=np.float64)

    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, and close must have the same length")

    # Choose implementation
    if implementation in {"numba", "auto"}:
        # Use Numba with proper caching
        return wclprice_numba(high, low, close)
    # Pure NumPy fallback
    return (high + low + 2.0 * close) / 4.0


# Export the main function
__all__ = ["wclprice"]
