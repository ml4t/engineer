# mypy: disable-error-code="no-any-return"
"""
VAR - Variance over period.

The variance is a measure of dispersion of a set of data points around their mean value.
It is the square of the standard deviation.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def var_numba(
    close: npt.NDArray[np.float64],
    timeperiod: int = 5,
    nbdev: float = 1.0,  # noqa: ARG001 - TA-Lib compatibility parameter, not used in VAR
) -> npt.NDArray[np.float64]:
    """
    VAR calculation using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    timeperiod : int, default 5
        Number of periods
    nbdev : float, default 1.0
        Number of deviations (not used in VAR calculation)

    Returns
    -------
    npt.NDArray
        Variance close
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Need at least timeperiod close
    if n < timeperiod:
        return result

    # Initialize running sums
    sum_x = 0.0
    sum_x2 = 0.0

    # Calculate initial sums
    for i in range(timeperiod):
        val = close[i]
        sum_x += val
        sum_x2 += val * val

    # Calculate first variance
    mean = sum_x / timeperiod
    mean_of_squares = sum_x2 / timeperiod
    result[timeperiod - 1] = mean_of_squares - mean * mean

    # Calculate remaining variances using sliding window
    for i in range(timeperiod, n):
        # Add new value
        new_val = close[i]
        sum_x += new_val
        sum_x2 += new_val * new_val

        # Remove old value
        old_val = close[i - timeperiod]
        sum_x -= old_val
        sum_x2 -= old_val * old_val

        # Calculate variance
        mean = sum_x / timeperiod
        mean_of_squares = sum_x2 / timeperiod
        result[i] = mean_of_squares - mean * mean

    return result


def var_polars(col: str, timeperiod: int = 5, nbdev: float = 1.0) -> pl.Expr:  # noqa: ARG001 - TA-Lib compatibility
    """
    VAR using native Polars rolling variance for maximum performance.

    Parameters
    ----------
    col : str
        Name of the column containing close
    timeperiod : int, default 5
        Number of periods
    nbdev : float, default 1.0
        Number of deviations (not used in VAR calculation)

    Returns
    -------
    pl.Expr
        Polars expression for VAR calculation
    """
    # Use native Polars rolling variance - this is much faster than Numba
    # Polars uses the same population variance formula as TA-Lib
    return pl.col(col).rolling_var(window_size=timeperiod, ddof=0)


@feature(
    name="var",
    category="statistics",
    description="VAR - Variance",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def var(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 5,
    nbdev: float = 1.0,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    VAR - Variance over period.

    Variance measures the dispersion of a set of data points around their mean.
    It's calculated as the average of the squared deviations from the mean.

    Formula:
    VAR = E[X²] - (E[X])²

    Where:
    - E[X²] is the mean of squared close
    - E[X] is the mean of close

    Parameters
    ----------
    close : array-like or column name
        Input close
    timeperiod : int, default 5
        Number of periods
    nbdev : float, default 1.0
        Number of deviations (kept for TA-Lib compatibility but not used)

    Returns
    -------
    array or Polars expression
        Variance close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> var_values = var(close, timeperiod=5)

    Notes
    -----
    - Variance is always non-negative
    - A variance of 0 indicates all close in the period are identical
    - Standard deviation is the square root of variance
    - The nbdev parameter is kept for TA-Lib compatibility but is not used
    """
    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return var_polars(close, timeperiod, nbdev)

    # Convert to numpy array
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if timeperiod < 1:
        raise ValueError(f"timeperiod must be >= 1, got {timeperiod}")

    return var_numba(close, timeperiod, nbdev)


# Export the main function
__all__ = ["var"]
