# mypy: disable-error-code="no-any-return"
"""
Median Price (MEDPRICE) - TA-Lib compatible implementation.

MEDPRICE = (High + Low) / 2

The median price is simply the midpoint between the high and low close.
Also known as the "mid price" or "middle price".
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def medprice_numba(
    high: npt.NDArray[np.float64], low: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Median Price calculation using optimized Numba.

    MEDPRICE = (High + Low) / 2

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close

    Returns
    -------
    npt.NDArray
        Median price close
    """
    # Vectorized calculation - much faster than loop
    return (high + low) * 0.5


def medprice_polars(high: str, low_col: str) -> pl.Expr:
    """
    Median Price using Polars expressions.

    Parameters
    ----------
    high : str
        Name of high price column
    low_col : str
        Name of low price column

    Returns
    -------
    pl.Expr
        Polars expression for median price calculation
    """
    # Simple average of high and low
    return (pl.col(high) + pl.col(low_col)) / 2.0


@feature(
    name="medprice",
    category="price_transform",
    description="MEDPRICE - Median Price",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def medprice(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    implementation: Literal["auto", "numba", "polars"] = "auto",
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Median Price (MEDPRICE).

    Calculates the midpoint between the high and low close. This is the
    simplest price transformation, representing the middle of the trading
    range for each period.

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close

    Returns
    -------
    array or Polars expression
        Median price close

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([102.0, 103.0, 104.0])
    >>> low = np.array([98.0, 99.0, 100.0])
    >>> med = medprice(high, low)
    >>> med
    array([100.0, 101.0, 102.0])

    Notes
    -----
    - Simple arithmetic mean of high and low
    - Formula: (High + Low) / 2
    - Returns NaN if either input price is NaN
    - Represents the midpoint of the trading range
    - Often used as a simplified price representation
    """
    # Handle string inputs (Polars column names) or explicit polars request
    if isinstance(high, str) and isinstance(low, str):
        return medprice_polars(high, low)
    if implementation == "polars":
        raise ValueError(
            "Polars implementation requires all inputs to be column names (strings)",
        )

    # Convert to numpy arrays
    high = high.to_numpy() if isinstance(high, pl.Series) else np.asarray(high, dtype=np.float64)
    low = low.to_numpy() if isinstance(low, pl.Series) else np.asarray(low, dtype=np.float64)

    # Validate inputs
    if len(high) != len(low):
        raise ValueError("high and low must have the same length")

    # Choose implementation
    if implementation in {"numba", "auto"}:
        # Use Numba with proper caching
        return medprice_numba(high, low)
    # Pure NumPy fallback
    return (high + low) * 0.5


# Export the main function
__all__ = ["medprice"]
