# mypy: disable-error-code="no-any-return"
"""
Typical Price (TYPPRICE) - TA-Lib compatible implementation.

TYPPRICE = (High + Low + Close) / 3

The typical price is the average of the high, low, and closing close.
It's often used as a single value to represent the price action for a period.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def typprice_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Typical Price calculation using Numba.

    TYPPRICE = (High + Low + Close) / 3

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
        Typical price close
    """
    # Vectorized calculation is faster in Numba
    return (high + low + close) / 3.0


def typprice_polars(high: str, low_col: str, close_col: str) -> pl.Expr:
    """
    Typical Price using Polars expressions.

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
        Polars expression for typical price calculation
    """
    # Average of high, low, and close
    return (pl.col(high) + pl.col(low_col) + pl.col(close_col)) / 3.0


@feature(
    name="typprice",
    category="price_transform",
    description="TYPPRICE - Typical Price",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def typprice(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    implementation: Literal["auto", "numba", "polars"] = "auto",
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Typical Price (TYPPRICE).

    Calculates the average of the high, low, and closing close. This provides
    a single value that represents the "typical" price for a period, often used
    as input to other indicators or as a simplified price representation.

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    close : array-like or column name
        Close close

    Returns
    -------
    array or Polars expression
        Typical price close

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([102.0, 103.0, 104.0])
    >>> low = np.array([99.0, 100.0, 101.0])
    >>> close = np.array([101.0, 102.0, 103.0])
    >>> typ = typprice(high, low, close)
    >>> typ
    array([100.666667, 101.666667, 102.666667])

    Notes
    -----
    - Simple arithmetic mean of high, low, and close
    - Excludes open price (unlike AVGPRICE)
    - Returns NaN if any input price is NaN
    - Often used in CCI (Commodity Channel Index) calculation
    - Also known as "Pivot Price" in some contexts
    """
    # Handle string inputs (Polars column names) or explicit polars request
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        return typprice_polars(high, low, close)
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
        return typprice_numba(high, low, close)
    # Pure NumPy fallback
    return (high + low + close) / 3.0


# Export the main function
__all__ = ["typprice"]
