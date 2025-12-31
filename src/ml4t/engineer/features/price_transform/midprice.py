# mypy: disable-error-code="no-any-return"
"""
MIDPRICE - Midpoint Price over period.

The midpoint price is the average of the highest high and lowest low over a given period.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit(cache=True)  # type: ignore[misc]
def midprice_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    timeperiod: int,
) -> npt.NDArray[np.float64]:
    """
    MIDPRICE calculation using dual monotonic deque algorithm.

    Time complexity: O(n) - each element is pushed/popped at most once
    Space complexity: O(timeperiod) for the deques

    Uses two monotonic deques simultaneously:
    - One for rolling max of high (decreasing order)
    - One for rolling min of low (increasing order)
    """
    n = len(high)
    result = np.full(n, np.nan)

    if n < timeperiod:
        return result

    # Monotonic deques for max (high) and min (low)
    max_deque = np.empty(n, dtype=np.int64)
    min_deque = np.empty(n, dtype=np.int64)
    max_front = 0
    max_back = 0
    min_front = 0
    min_back = 0

    # Process first window
    for i in range(timeperiod):
        # Max deque: remove smaller elements from back
        while max_back > max_front and high[max_deque[max_back - 1]] <= high[i]:
            max_back -= 1
        max_deque[max_back] = i
        max_back += 1

        # Min deque: remove larger elements from back
        while min_back > min_front and low[min_deque[min_back - 1]] >= low[i]:
            min_back -= 1
        min_deque[min_back] = i
        min_back += 1

    # First window result
    result[timeperiod - 1] = (high[max_deque[max_front]] + low[min_deque[min_front]]) * 0.5

    # Process remaining elements
    for i in range(timeperiod, n):
        # Remove elements outside window from both deques
        while max_front < max_back and max_deque[max_front] <= i - timeperiod:
            max_front += 1
        while min_front < min_back and min_deque[min_front] <= i - timeperiod:
            min_front += 1

        # Add new element to max deque
        while max_back > max_front and high[max_deque[max_back - 1]] <= high[i]:
            max_back -= 1
        max_deque[max_back] = i
        max_back += 1

        # Add new element to min deque
        while min_back > min_front and low[min_deque[min_back - 1]] >= low[i]:
            min_back -= 1
        min_deque[min_back] = i
        min_back += 1

        # Midprice is average of max high and min low
        result[i] = (high[max_deque[max_front]] + low[min_deque[min_front]]) * 0.5

    return result


def midprice_numpy(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    timeperiod: int = 14,
) -> npt.NDArray[np.float64]:
    """
    MIDPRICE calculation using optimized NumPy rolling max/min.

    This implementation leverages NumPy's highly optimized sliding_window_view
    to efficiently compute rolling max and min in O(n×k) time with excellent
    constants due to C-level optimizations.

    Parameters
    ----------
    high : npt.NDArray
        High values
    low : npt.NDArray
        Low values
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    npt.NDArray
        Midpoint price values
    """
    n = len(high)
    if n < timeperiod or len(low) != n:
        return np.full(n, np.nan)

    from numpy.lib.stride_tricks import sliding_window_view

    # Compute rolling max of high and rolling min of low using NumPy's optimized C code
    high_windows = sliding_window_view(high, window_shape=timeperiod)
    low_windows = sliding_window_view(low, window_shape=timeperiod)

    rolling_max_high = np.max(high_windows, axis=1)
    rolling_min_low = np.min(low_windows, axis=1)

    # Compute midpoint
    result = np.full(n, np.nan)
    result[timeperiod - 1 :] = (rolling_max_high + rolling_min_low) * 0.5

    return result


def midprice_polars(high: str, low_col: str, timeperiod: int = 14) -> pl.Expr:
    """
    MIDPRICE using Polars expressions.

    Parameters
    ----------
    high : str
        Name of the column containing high close
    low_col : str
        Name of the column containing low close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    pl.Expr
        Polars expression for MIDPRICE calculation
    """
    # Pure Polars expression optimization - much faster than map_batches
    # Calculate rolling max of high and rolling min of low, then take their average
    return (pl.col(high).rolling_max(timeperiod) + pl.col(low_col).rolling_min(timeperiod)) * 0.5


@feature(
    name="midprice",
    category="price_transform",
    description="MIDPRICE - Midpoint Price over period",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def midprice(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
    implementation: Literal["auto", "numba", "polars"] = "auto",
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    MIDPRICE - Midpoint Price over period.

    The midpoint price indicator calculates the middle point between the highest
    high and lowest low over a specified period. It's a price-based indicator
    that shows the center of the trading range.

    Formula:
    MIDPRICE = (Highest High + Lowest Low) / 2

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    array or Polars expression
        Midpoint price close

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([10, 12, 11, 13, 14, 12, 11, 15, 13, 12])
    >>> low = np.array([9, 10, 10, 11, 12, 10, 9, 13, 11, 10])
    >>> midprice_values = midprice(high, low, timeperiod=5)

    Notes
    -----
    - The midprice represents the center of the price range
    - Can be used as a simple support/resistance indicator
    - When timeperiod=1, MIDPRICE equals MEDPRICE
    - Useful for identifying the trading range center
    """
    # Handle string inputs (Polars column names) or explicit polars request
    if isinstance(high, str) and isinstance(low, str):
        return midprice_polars(high, low, timeperiod)
    if implementation == "polars":
        raise ValueError(
            "Polars implementation requires all inputs to be column names (strings)",
        )

    # Convert to numpy arrays
    high = high.to_numpy() if isinstance(high, pl.Series) else np.asarray(high, dtype=np.float64)
    low = low.to_numpy() if isinstance(low, pl.Series) else np.asarray(low, dtype=np.float64)

    # Validate inputs
    if len(high) != len(low):
        raise ValueError("high and low arrays must have the same length")
    if timeperiod < 2:
        raise ValueError(f"timeperiod must be >= 2, got {timeperiod}")

    # Choose implementation based on data size and window
    # For large datasets (>100k) with small windows (<20), Numba's dual monotonic
    # deque algorithm is O(n) and avoids NumPy's overhead
    # For smaller datasets, NumPy's C-optimized max/min is faster
    use_numba = implementation == "numba" or (
        implementation == "auto" and len(high) > 100_000 and timeperiod < 20
    )

    if use_numba:
        return midprice_numba(high, low, timeperiod)

    # Use NumPy's sliding window for smaller datasets
    return midprice_numpy(high, low, timeperiod)


# Export the main function
__all__ = ["midprice"]
