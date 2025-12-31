# mypy: disable-error-code="no-any-return"
"""
LINEARREG - Linear Regression.

The linear regression indicator calculates the endpoint of a linear regression line
for each period. It fits a straight line to the data using the least squares method.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def linearreg_numba(
    close: npt.NDArray[np.float64], timeperiod: int = 14
) -> npt.NDArray[np.float64]:
    """
    LINEARREG calculation using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    npt.NDArray
        Linear regression close
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Need at least timeperiod close
    if n < timeperiod:
        return result

    # Pre-calculate constants for efficiency
    # SumX = sum of x close (0, 1, ..., period-1)
    SumX = timeperiod * (timeperiod - 1) * 0.5
    # SumXSqr = sum of x^2 close
    SumXSqr = timeperiod * (timeperiod - 1) * (2 * timeperiod - 1) / 6
    # Divisor for slope calculation
    Divisor = SumX * SumX - timeperiod * SumXSqr

    # Calculate linear regression for each window
    for i in range(timeperiod - 1, n):
        SumXY = 0.0
        SumY = 0.0

        # Calculate sums for regression
        # TA-Lib uses reversed x-axis: x=period-1 for oldest, x=0 for newest
        for j in range(timeperiod):
            k = timeperiod - 1 - j  # k goes from period-1 down to 0
            y = close[i - k]
            SumY += y
            SumXY += k * y

        # Calculate slope (m) and intercept (b)
        m = (timeperiod * SumXY - SumX * SumY) / Divisor
        b = (SumY - m * SumX) / timeperiod

        # Linear regression value at x = period-1
        result[i] = b + m * (timeperiod - 1)

    return result


def linearreg_polars(col: str, timeperiod: int = 14) -> pl.Expr:
    """
    LINEARREG using Polars expressions.

    Parameters
    ----------
    col : str
        Name of the column containing close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    pl.Expr
        Polars expression for LINEARREG calculation
    """
    return pl.col(col).map_batches(
        lambda x: pl.Series(linearreg_numba(x.to_numpy(), timeperiod)),
        return_dtype=pl.Float64,
    )


@feature(
    name="linearreg",
    category="statistics",
    description="LINEARREG - Linear Regression",
    lookback=14,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def linearreg(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    LINEARREG - Linear Regression.

    The linear regression indicator fits a straight line to the data over the
    specified period using the least squares method. It returns the y-value
    of the regression line at the current point (x = period-1).

    This is useful for identifying trend direction and strength. The value
    represents where the regression line ends at the current bar.

    Parameters
    ----------
    close : array-like or column name
        Input close
    timeperiod : int, default 14
        Number of periods for regression

    Returns
    -------
    array or Polars expression
        Linear regression close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> linearreg_values = linearreg(close, timeperiod=5)

    Notes
    -----
    - Returns the endpoint of the linear regression line
    - Different from TSF which projects one period forward
    - Useful for trend identification and smoothing
    - The regression line minimizes the sum of squared errors
    """
    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return linearreg_polars(close, timeperiod)

    # Convert to numpy array
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if timeperiod < 2:
        raise ValueError(f"timeperiod must be >= 2, got {timeperiod}")

    return linearreg_numba(close, timeperiod)


# Export the main function
__all__ = ["linearreg"]
