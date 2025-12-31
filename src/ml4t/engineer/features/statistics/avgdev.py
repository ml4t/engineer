# mypy: disable-error-code="no-any-return"
"""
AVGDEV - Average Deviation.

The average deviation is a measure of the variability or dispersion in a data set.
It's the average of the absolute deviations from the mean.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def avgdev_numba(close: npt.NDArray[np.float64], timeperiod: int = 14) -> npt.NDArray[np.float64]:
    """
    AVGDEV calculation using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    npt.NDArray
        Average deviation close
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Need at least timeperiod close
    if n < timeperiod:
        return result

    # Calculate average deviations using sliding window
    for i in range(timeperiod - 1, n):
        # Calculate mean for current window
        window_sum = 0.0
        for j in range(timeperiod):
            window_sum += close[i - j]
        mean = window_sum / timeperiod

        # Calculate average absolute deviation
        deviation_sum = 0.0
        for j in range(timeperiod):
            deviation_sum += abs(close[i - j] - mean)

        result[i] = deviation_sum / timeperiod

    return result


def avgdev_polars(col: str, timeperiod: int = 14) -> pl.Expr:
    """
    AVGDEV using Polars expressions.

    Parameters
    ----------
    col : str
        Name of the column containing close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    pl.Expr
        Polars expression for AVGDEV calculation
    """
    return pl.col(col).map_batches(
        lambda x: pl.Series(avgdev_numba(x.to_numpy(), timeperiod)),
        return_dtype=pl.Float64,
    )


@feature(
    name="avgdev",
    category="statistics",
    description="AVGDEV - Average Deviation",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=False,
)
def avgdev(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    AVGDEV - Average Deviation.

    Average deviation measures the average distance of data points from their mean.
    It's calculated as the mean of the absolute deviations from the arithmetic mean.

    Formula:
    AVGDEV = (1/n) * Σ|Xi - μ|

    Where:
    - n is the number of close
    - Xi is each value
    - μ is the mean of close

    Parameters
    ----------
    close : array-like or column name
        Input close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    array or Polars expression
        Average deviation close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> avgdev_values = avgdev(close, timeperiod=5)

    Notes
    -----
    - Average deviation is always non-negative
    - It's less sensitive to outliers than standard deviation
    - A value of 0 indicates all close in the period are identical
    - Unlike standard deviation, it doesn't square the deviations
    """
    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return avgdev_polars(close, timeperiod)

    # Convert to numpy array
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if timeperiod < 2:
        raise ValueError(f"timeperiod must be >= 2, got {timeperiod}")

    return avgdev_numba(close, timeperiod)


# Export the main function
__all__ = ["avgdev"]
