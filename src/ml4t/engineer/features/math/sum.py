# mypy: disable-error-code="no-any-return"
"""
SUM - Summation over a specified number of periods.

This function calculates the rolling sum of close over a specified window.
It's a fundamental building block for many technical indicators.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit(cache=True)  # type: ignore[misc]
def sum_numba(close: npt.NDArray[np.float64], timeperiod: int) -> npt.NDArray[np.float64]:
    """
    Calculate rolling sum using sliding window algorithm.

    Time complexity: O(n) - each element is added and subtracted once
    Space complexity: O(1) besides output array

    Algorithm: Maintain a running sum, add new elements and subtract old ones.
    """
    n = len(close)
    result = np.full(n, np.nan)

    if n < timeperiod:
        return result

    # Calculate first window sum
    window_sum = 0.0
    for i in range(timeperiod):
        window_sum += close[i]
    result[timeperiod - 1] = window_sum

    # Slide the window: add new element, remove old element
    for i in range(timeperiod, n):
        window_sum = window_sum + close[i] - close[i - timeperiod]
        result[i] = window_sum

    return result


@feature(
    name="summation",
    category="math",
    description="SUMMATION - Sum of close over period",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def summation(
    close: npt.NDArray[np.float64] | pl.Expr | str,
    timeperiod: int = 30,
    implementation: Literal["auto", "numba", "polars"] = "auto",
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate the sum over a specified number of periods.

    Parameters
    ----------
    close : array-like or Polars expression
        Input close
    timeperiod : int, default 30
        Number of periods for the calculation

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        Rolling sum close

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> result = qta.summation(close, timeperiod=3)
    >>> print(result[2:])  # First 2 close are NaN
    [ 6.  9. 12. 15. 18. 21.]

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({"price": [1, 2, 3, 4, 5, 6, 7, 8]})
    >>> result = df.with_columns(qta.summation("price", 3).alias("sum"))
    """
    if isinstance(close, (pl.Expr, str)):
        # Return Polars expression
        if isinstance(close, str):
            close = pl.col(close)

        return close.rolling_sum(window_size=timeperiod)

    # Handle numpy arrays
    close = np.asarray(close, dtype=np.float64)

    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

    if len(close) < timeperiod:
        return np.full(len(close), np.nan)

    # Choose implementation
    # For "auto", prefer numpy's optimized methods
    use_numba = implementation == "numba" or (
        implementation == "auto" and len(close) > 100_000 and timeperiod < 20
    )

    if use_numba:
        return sum_numba(close, timeperiod)

    # Use numpy's sliding_window_view with sum - highly optimized C code
    from numpy.lib.stride_tricks import sliding_window_view

    result = np.full(len(close), np.nan)
    if len(close) >= timeperiod:
        windows = sliding_window_view(close, window_shape=timeperiod)
        result[timeperiod - 1 :] = np.sum(windows, axis=1)

    return result


# Make available at package level
__all__ = ["summation"]
