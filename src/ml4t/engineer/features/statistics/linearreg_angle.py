# mypy: disable-error-code="no-any-return"
"""
LINEARREG_ANGLE - Linear Regression Angle.

The linear regression angle indicator calculates the angle (in degrees) of the
linear regression line's slope for each period.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def linearreg_angle_numba(
    close: npt.NDArray[np.float64], timeperiod: int = 14
) -> npt.NDArray[np.float64]:
    """
    LINEARREG_ANGLE calculation using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    npt.NDArray
        Linear regression angle close in degrees
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

    # Calculate linear regression angle for each window
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

        # Calculate slope (m)
        m = (timeperiod * SumXY - SumX * SumY) / Divisor

        # Convert slope to angle in degrees
        # angle = arctan(m) * (180 / pi)
        result[i] = np.arctan(m) * (180.0 / np.pi)

    return result


def linearreg_angle_polars(col: str, timeperiod: int = 14) -> pl.Expr:
    """
    LINEARREG_ANGLE using Polars expressions.

    Parameters
    ----------
    col : str
        Name of the column containing close
    timeperiod : int, default 14
        Number of periods

    Returns
    -------
    pl.Expr
        Polars expression for LINEARREG_ANGLE calculation
    """
    return pl.col(col).map_batches(
        lambda x: pl.Series(linearreg_angle_numba(x.to_numpy(), timeperiod)),
        return_dtype=pl.Float64,
    )


@feature(
    name="linearreg_angle",
    category="statistics",
    description="LINEARREG_ANGLE - angle of linear regression line",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def linearreg_angle(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    LINEARREG_ANGLE - Linear Regression Angle.

    The linear regression angle indicator calculates the angle (in degrees)
    of the linear regression line's slope. This represents the rate of change
    in the trend, with positive angles indicating upward trends and negative
    angles indicating downward trends.

    The angle is calculated as: arctan(slope) * (180/π)

    Parameters
    ----------
    close : array-like or column name
        Input close
    timeperiod : int, default 14
        Number of periods for regression

    Returns
    -------
    array or Polars expression
        Linear regression angle close in degrees

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> angle = linearreg_angle(close, timeperiod=5)

    Notes
    -----
    - Returns the angle of the regression line in degrees
    - Positive angles indicate upward trends
    - Negative angles indicate downward trends
    - The magnitude indicates the strength of the trend
    - A flat line has an angle of 0 degrees
    """
    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return linearreg_angle_polars(close, timeperiod)

    # Convert to numpy array
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if timeperiod < 2:
        raise ValueError(f"timeperiod must be >= 2, got {timeperiod}")

    return linearreg_angle_numba(close, timeperiod)


# Export the main function
__all__ = ["linearreg_angle"]
