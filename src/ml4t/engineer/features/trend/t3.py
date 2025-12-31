# mypy: disable-error-code="no-any-return"
"""
T3 - Triple Exponential Moving Average (T3).

This is Tim Tillson's T3 moving average, which provides better smoothing than
traditional moving averages with less lag.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def t3_numba(
    close: npt.NDArray[np.float64],
    timeperiod: int = 5,
    vfactor: float = 0.7,
) -> npt.NDArray[np.float64]:
    """
    T3 calculation using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    timeperiod : int, default 5
        Number of periods
    vfactor : float, default 0.7
        Volume factor (0-1)

    Returns
    -------
    npt.NDArray
        T3 close
    """
    n = len(close)
    result = np.full(n, np.nan)

    # T3 uses 6 levels of EMA
    # The lookback is 6 * (period - 1)
    lookback = 6 * (timeperiod - 1)

    if n <= lookback:
        return result

    # EMA multiplier
    k = 2.0 / (timeperiod + 1.0)
    one_minus_k = 1.0 - k

    # Initialize the 6 EMAs
    # Start with SMA for each level
    today = 0

    # Initialize e1 (first EMA)
    temp_sum = 0.0
    for i in range(timeperiod):
        temp_sum += close[today + i]
    e1 = temp_sum / timeperiod
    today += timeperiod

    # Initialize e2 (EMA of e1)
    temp_sum = e1
    for _i in range(timeperiod - 1):
        e1 = k * close[today] + one_minus_k * e1
        temp_sum += e1
        today += 1
    e2 = temp_sum / timeperiod

    # Initialize e3 (EMA of e2)
    temp_sum = e2
    for _i in range(timeperiod - 1):
        e1 = k * close[today] + one_minus_k * e1
        e2 = k * e1 + one_minus_k * e2
        temp_sum += e2
        today += 1
    e3 = temp_sum / timeperiod

    # Initialize e4 (EMA of e3)
    temp_sum = e3
    for _i in range(timeperiod - 1):
        e1 = k * close[today] + one_minus_k * e1
        e2 = k * e1 + one_minus_k * e2
        e3 = k * e2 + one_minus_k * e3
        temp_sum += e3
        today += 1
    e4 = temp_sum / timeperiod

    # Initialize e5 (EMA of e4)
    temp_sum = e4
    for _i in range(timeperiod - 1):
        e1 = k * close[today] + one_minus_k * e1
        e2 = k * e1 + one_minus_k * e2
        e3 = k * e2 + one_minus_k * e3
        e4 = k * e3 + one_minus_k * e4
        temp_sum += e4
        today += 1
    e5 = temp_sum / timeperiod

    # Initialize e6 (EMA of e5)
    temp_sum = e5
    for _i in range(timeperiod - 1):
        e1 = k * close[today] + one_minus_k * e1
        e2 = k * e1 + one_minus_k * e2
        e3 = k * e2 + one_minus_k * e3
        e4 = k * e3 + one_minus_k * e4
        e5 = k * e4 + one_minus_k * e5
        temp_sum += e5
        today += 1
    e6 = temp_sum / timeperiod

    # Calculate T3 coefficients based on volume factor
    vf2 = vfactor * vfactor
    c1 = -(vf2 * vfactor)
    c2 = 3.0 * (vf2 - c1)
    c3 = -6.0 * vf2 - 3.0 * (vfactor - c1)
    c4 = 1.0 + 3.0 * vfactor - c1 + 3.0 * vf2

    # At this point, today = lookback + 1
    # TA-Lib outputs starting at lookback position
    # First output goes at position lookback (not lookback+1)

    # Calculate first T3 value
    result[lookback] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    # Calculate T3 for the remaining range
    while today < n:
        e1 = k * close[today] + one_minus_k * e1
        e2 = k * e1 + one_minus_k * e2
        e3 = k * e2 + one_minus_k * e3
        e4 = k * e3 + one_minus_k * e4
        e5 = k * e4 + one_minus_k * e5
        e6 = k * e5 + one_minus_k * e6

        # T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
        result[today] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
        today += 1

    return result


def t3_polars(col: str, timeperiod: int = 5, vfactor: float = 0.7) -> pl.Expr:
    """
    T3 using Polars expressions.

    Parameters
    ----------
    col : str
        Name of the column containing close
    timeperiod : int, default 5
        Number of periods
    vfactor : float, default 0.7
        Volume factor (0-1)

    Returns
    -------
    pl.Expr
        Polars expression for T3 calculation
    """
    return pl.col(col).map_batches(
        lambda x: pl.Series(t3_numba(x.to_numpy(), timeperiod, vfactor)),
        return_dtype=pl.Float64,
    )


@feature(
    name="t3",
    category="trend",
    description="T3 - Triple Exponential Moving Average (Tillson)",
    lookback="timeperiod",
    normalized=False,
    formula="Triple smoothed EMA with volume factor",
    ta_lib_compatible=True,
)
def t3(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 5,
    vfactor: float = 0.7,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    T3 - Triple Exponential Moving Average (T3).

    T3 is a moving average developed by Tim Tillson. It incorporates a
    smoothing technique which creates a moving average with better smoothing
    and less lag than traditional moving averages. The T3 is a weighted sum of
    six EMAs, each one applied to the previous EMA result.

    Parameters
    ----------
    close : array-like or column name
        Input close (typically close close)
    timeperiod : int, default 5
        Number of periods
    vfactor : float, default 0.7
        Volume factor (0-1). Controls the amount of smoothing

    Returns
    -------
    array or Polars expression
        T3 close

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([10, 11, 12, 11, 10, 11, 13, 14, 13, 12])
    >>> t3_values = t3(close, timeperiod=5, vfactor=0.7)

    Notes
    -----
    The T3 calculation involves:
    1. Calculate 6 levels of EMAs, each one on the previous
    2. Combine them using coefficients derived from the volume factor

    Formula:
    - c1 = -vfactor^3
    - c2 = 3*vfactor^2 + 3*vfactor^3
    - c3 = -6*vfactor^2 - 3*vfactor - 3*vfactor^3
    - c4 = 1 + 3*vfactor + vfactor^3 + 3*vfactor^2
    - T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3

    Reference: "Smoothing Techniques For More Accurate Signals"
    by Tim Tillson in Stocks & Commodities V16:1 (33-37)
    """
    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return t3_polars(close, timeperiod, vfactor)

    # Convert to numpy array
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if timeperiod < 2:
        raise ValueError(f"timeperiod must be >= 2, got {timeperiod}")
    if not 0 <= vfactor <= 1:
        raise ValueError(f"vfactor must be between 0 and 1, got {vfactor}")

    return t3_numba(close, timeperiod, vfactor)


# Export the main function
__all__ = ["t3"]
