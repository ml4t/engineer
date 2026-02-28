# mypy: disable-error-code="no-any-return"
"""
TSF - Time Series Forecast.

The Time Series Forecast is the value of a linear regression line projected one period into the future.
It's essentially Linear Regression + Linear Regression Slope, providing a forecast of the next value.

TSF[i] = LINEARREG[i] + LINEARREG_SLOPE[i]
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit  # type: ignore[misc]
def tsf_numba(close: npt.NDArray[np.float64], timeperiod: int) -> npt.NDArray[np.float64]:
    """Calculate Time Series Forecast using optimized sliding window algorithm."""
    n = len(close)
    result = np.full(n, np.nan)

    if n < timeperiod:
        return result

    # Pre-calculate constants that don't change
    x = np.arange(timeperiod, dtype=np.float64)
    sum_x: float = float(np.sum(x))
    sum_x2: float = float(np.sum(x * x))
    denominator = timeperiod * sum_x2 - sum_x * sum_x

    if abs(denominator) < 1e-10:
        return result

    # Initialize running sums for first window
    sum_y = 0.0
    sum_xy = 0.0

    for j in range(timeperiod):
        y_val = close[j]
        sum_y += y_val
        sum_xy += x[j] * y_val

    # Calculate first result
    slope = (timeperiod * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / timeperiod
    result[timeperiod - 1] = intercept + slope * timeperiod

    # Use sliding window for remaining calculations
    for i in range(timeperiod, n):
        # Remove oldest value and add newest value
        old_val = close[i - timeperiod]
        new_val = close[i]

        # Update running sums efficiently
        sum_y = sum_y - old_val + new_val
        # For sum_xy, we need to shift all x indices and recalculate
        # This is the bottleneck - can't avoid recalculation here
        sum_xy = 0.0
        for j in range(timeperiod):
            sum_xy += x[j] * close[i - timeperiod + 1 + j]

        # Calculate slope and intercept
        slope = (timeperiod * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / timeperiod

        # Time Series Forecast: project one period into the future
        result[i] = intercept + slope * timeperiod

    return result


@feature(
    name="tsf",
    category="statistics",
    description="TSF - Time Series Forecast",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def tsf(
    close: npt.NDArray[np.float64] | pl.Expr | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Calculate Time Series Forecast.

    The Time Series Forecast projects the linear regression line one period into the future.
    It's useful for trend analysis and simple forecasting.

    Parameters
    ----------
    close : array-like or Polars expression
        Input close (typically close close)
    timeperiod : int, default 14
        Number of periods for the calculation

    Returns
    -------
    npt.NDArray[np.float64] or pl.Expr
        Time Series Forecast close

    Notes
    -----
    TSF is calculated as:
    TSF[i] = LINEARREG[i] + LINEARREG_SLOPE[i]

    Where LINEARREG is the linear regression value at the current period,
    and LINEARREG_SLOPE projects it one period forward.

    Examples
    --------
    >>> import numpy as np
    >>> import ml4t.engineer.features.ta as ta as qta
    >>>
    >>> # Trending data
    >>> close = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> result = qta.tsf(close, timeperiod=3)
    >>> print(result[2:])  # First 2 close are NaN
    [4. 5. 6. 7. 8. 9. 10. 11.]

    Using with Polars:
    >>> import polars as pl
    >>> df = pl.DataFrame({"price": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    >>> result = df.with_columns(qta.tsf("price", 3).alias("tsf"))
    """
    if isinstance(close, pl.Expr | str):
        # Return Polars expression using map_batches for complex calculations
        if isinstance(close, str):
            close = pl.col(close)

        return close.map_batches(
            lambda x: tsf_numba(x.to_numpy(), timeperiod),
            return_dtype=pl.Float64,
        )

    # Handle numpy arrays
    close = np.asarray(close, dtype=np.float64)

    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

    if len(close) < timeperiod:
        return np.full(len(close), np.nan)

    return tsf_numba(close, timeperiod)


# Make available at package level
__all__ = ["tsf"]
