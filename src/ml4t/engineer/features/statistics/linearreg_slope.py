# mypy: disable-error-code="no-any-return"
"""
LINEARREG_SLOPE - Linear Regression Slope.

The linear regression slope indicator calculates the slope (m) of the
linear regression line for each period.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def linearreg_slope_numba(
    close: npt.NDArray[np.float64], timeperiod: int = 14
) -> npt.NDArray[np.float64]:
    """
    LINEARREG_SLOPE calculation using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    npt.NDArray
        Linear regression slope close
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
    Divisor = timeperiod * SumXSqr - SumX * SumX

    # Calculate linear regression slope for each window
    for i in range(timeperiod - 1, n):
        SumXY = 0.0
        SumY = 0.0

        # Calculate sums for regression
        # x coordinates go from 0 to period-1
        for j in range(timeperiod):
            x = j
            y = close[i - timeperiod + 1 + j]
            SumY += y
            SumXY += x * y

        # Calculate and return slope (m)
        result[i] = (timeperiod * SumXY - SumX * SumY) / Divisor

    return result


def linearreg_slope_polars(col: str, timeperiod: int = 14) -> pl.Expr:
    """
    LINEARREG_SLOPE using Polars expressions.

    Parameters
    ----------
    col : str
        Name of the column containing close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    pl.Expr
        Polars expression for LINEARREG_SLOPE calculation
    """
    return pl.col(col).map_batches(
        lambda x: pl.Series(linearreg_slope_numba(x.to_numpy(), timeperiod)),
        return_dtype=pl.Float64,
    )


@feature(
    name="linearreg_slope",
    category="statistics",
    description="LINEARREG_SLOPE - slope of linear regression",
    lookback=14,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def linearreg_slope(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    LINEARREG_SLOPE - Linear Regression Slope.

    The linear regression slope indicator calculates the slope (m) of the
    linear regression line equation y = mx + b. The slope represents the
    rate of change in the data over the specified period.

    A positive slope indicates an upward trend, while a negative slope
    indicates a downward trend. The magnitude indicates the strength of
    the trend.

    Parameters
    ----------
    close : array-like or column name
        Input close
    timeperiod : int, default 14
        Number of periods for regression

    Returns
    -------
    array or Polars expression
        Linear regression slope close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> slope = linearreg_slope(close, timeperiod=5)

    Notes
    -----
    - Returns the slope of the regression line
    - Positive close indicate upward trends
    - Negative close indicate downward trends
    - The magnitude indicates trend strength
    - For y = mx + b, this returns 'm'
    """
    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return linearreg_slope_polars(close, timeperiod)

    # Convert to numpy array
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if timeperiod < 2:
        raise ValueError(f"timeperiod must be >= 2, got {timeperiod}")

    return linearreg_slope_numba(close, timeperiod)


# Export the main function
__all__ = ["linearreg_slope"]
