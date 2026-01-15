# mypy: disable-error-code="no-any-return"
"""
MAX - Highest value over a specified number of periods.

This function finds the maximum value in a rolling window of specified size.
It's a fundamental building block for many technical indicators.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit(cache=True)  # type: ignore[misc]
def max_numba(close: npt.NDArray[np.float64], timeperiod: int) -> npt.NDArray[np.float64]:
    """
    Calculate rolling maximum using monotonic deque algorithm.

    Time complexity: O(n) - each element is pushed and popped at most once
    Space complexity: O(timeperiod) for the deque

    Algorithm: Maintain a deque of indices in decreasing order of values.
    The front of the deque always contains the index of the maximum element.
    """
    n = len(close)
    result = np.full(n, np.nan)

    if n < timeperiod:
        return result

    # Monotonic deque to store indices
    # We'll use a simple array-based deque since Numba doesn't support collections.deque
    deque = np.empty(n, dtype=np.int64)
    front = 0  # Index of front of deque
    back = 0  # Index where next element will be added

    # Process first window
    for i in range(timeperiod):
        # Remove elements from back that are smaller than current
        # They'll never be the maximum
        while back > front and close[deque[back - 1]] <= close[i]:
            back -= 1

        deque[back] = i
        back += 1

    # First window's maximum
    result[timeperiod - 1] = close[deque[front]]

    # Process remaining elements
    for i in range(timeperiod, n):
        # Remove elements outside current window from front
        while front < back and deque[front] <= i - timeperiod:
            front += 1

        # Remove elements from back that are smaller than current
        while back > front and close[deque[back - 1]] <= close[i]:
            back -= 1

        deque[back] = i
        back += 1

        # Maximum is at the front of deque
        result[i] = close[deque[front]]

    return result


@feature(
    name="maximum",
    category="math",
    description="MAXIMUM - Highest value over period",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def maximum(
    close: npt.NDArray[np.float64] | pl.Expr | str,
    timeperiod: int = 30,
    implementation: Literal["auto", "numba", "polars"] = "auto",
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate the highest value over a specified number of periods.

    Parameters
    ----------
    close : array-like or Polars expression
        Input close
    timeperiod : int, default 30
        Number of periods for the calculation

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        Rolling maximum close

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>> close = np.array([1, 3, 2, 5, 4, 7, 6, 8])
    >>> result = qta.maximum(close, timeperiod=3)
    >>> print(result[2:])  # First 2 close are NaN
    [3. 5. 5. 7. 7. 8.]

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({"price": [1, 3, 2, 5, 4, 7, 6, 8]})
    >>> result = df.with_columns(qta.maximum("price", 3).alias("max"))
    """
    if isinstance(close, (pl.Expr, str)):
        # Return Polars expression
        if isinstance(close, str):
            close = pl.col(close)

        return close.rolling_max(window_size=timeperiod)

    # Handle numpy arrays
    close = np.asarray(close, dtype=np.float64)

    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

    if len(close) < timeperiod:
        return np.full(len(close), np.nan)

    # Choose implementation
    # For "auto", prefer numpy's optimized C code over Numba for small-medium datasets
    # Numba is better for very large datasets (>100k) with small windows
    use_numba = implementation == "numba" or (
        implementation == "auto" and len(close) > 100_000 and timeperiod < 20
    )

    if use_numba:
        return max_numba(close, timeperiod)

    # Use numpy.lib.stride_tricks for efficient rolling maximum
    # This uses highly optimized C code from NumPy
    from numpy.lib.stride_tricks import sliding_window_view

    result = np.full(len(close), np.nan)
    if len(close) >= timeperiod:
        # Create sliding windows and find max of each
        windows = sliding_window_view(close, window_shape=timeperiod)
        result[timeperiod - 1 :] = np.max(windows, axis=1)

    return result


# Make available at package level
__all__ = ["maximum"]
