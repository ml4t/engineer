# mypy: disable-error-code="no-any-return"
"""
Ultimate Oscillator (ULTOSC) - TA-Lib compatible implementation.

The Ultimate Oscillator is a momentum oscillator designed to capture momentum
across three different timeframes. It was developed by Larry Williams in 1976.

Formula:
1. Calculate Buying Pressure (BP): Close - Min(Low, Prior Close)
2. Calculate True Range (TR): Max(High, Prior Close) - Min(Low, Prior Close)
3. Calculate Average7 = Sum(BP, 7) / Sum(TR, 7)
4. Calculate Average14 = Sum(BP, 14) / Sum(TR, 14)
5. Calculate Average28 = Sum(BP, 28) / Sum(TR, 28)
6. UO = 100 * ((4 * Average7) + (2 * Average14) + Average28) / 7
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import njit

from ml4t.engineer.core.decorators import feature


@njit(cache=True, fastmath=True)  # type: ignore[misc]
def ultosc_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> npt.NDArray[np.float64]:
    """
    Ultimate Oscillator calculation using optimized Numba.

    Optimizations:
    - Vectorized BP/TR calculations using NumPy
    - Combined rolling sums in single pass
    - Minimal allocations

    Parameters
    ----------
    high : npt.NDArray
        High prices
    low : npt.NDArray
        Low prices
    close : npt.NDArray
        Close prices
    timeperiod1 : int, default 7
        First time period (shortest)
    timeperiod2 : int, default 14
        Second time period (medium)
    timeperiod3 : int, default 28
        Third time period (longest)

    Returns
    -------
    npt.NDArray
        Ultimate Oscillator values
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Need at least timeperiod3 + 1 data points
    if n <= timeperiod3:
        return result

    # Calculate Buying Pressure and True Range (Numba optimizes this loop well)
    bp = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        prior_close = close[i - 1]
        bp[i] = close[i] - min(low[i], prior_close)
        tr[i] = max(high[i], prior_close) - min(low[i], prior_close)

    # Pre-allocate rolling sum arrays
    bp_sum1 = np.zeros(n)
    bp_sum2 = np.zeros(n)
    bp_sum3 = np.zeros(n)
    tr_sum1 = np.zeros(n)
    tr_sum2 = np.zeros(n)
    tr_sum3 = np.zeros(n)

    # Initialize first sums (exactly matching TA-Lib algorithm)
    for i in range(1, timeperiod1 + 1):
        bp_sum1[timeperiod1] += bp[i]
        tr_sum1[timeperiod1] += tr[i]

    for i in range(1, timeperiod2 + 1):
        bp_sum2[timeperiod2] += bp[i]
        tr_sum2[timeperiod2] += tr[i]

    for i in range(1, timeperiod3 + 1):
        bp_sum3[timeperiod3] += bp[i]
        tr_sum3[timeperiod3] += tr[i]

    # Calculate rolling sums
    for i in range(timeperiod1 + 1, n):
        bp_sum1[i] = bp_sum1[i - 1] + bp[i] - bp[i - timeperiod1]
        tr_sum1[i] = tr_sum1[i - 1] + tr[i] - tr[i - timeperiod1]

    for i in range(timeperiod2 + 1, n):
        bp_sum2[i] = bp_sum2[i - 1] + bp[i] - bp[i - timeperiod2]
        tr_sum2[i] = tr_sum2[i - 1] + tr[i] - tr[i - timeperiod2]

    for i in range(timeperiod3 + 1, n):
        bp_sum3[i] = bp_sum3[i - 1] + bp[i] - bp[i - timeperiod3]
        tr_sum3[i] = tr_sum3[i - 1] + tr[i] - tr[i - timeperiod3]

    # Calculate Ultimate Oscillator
    for i in range(timeperiod3, n):
        if tr_sum1[i] != 0 and tr_sum2[i] != 0 and tr_sum3[i] != 0:
            avg1 = bp_sum1[i] / tr_sum1[i]
            avg2 = bp_sum2[i] / tr_sum2[i]
            avg3 = bp_sum3[i] / tr_sum3[i]
            result[i] = 100.0 * ((4.0 * avg1) + (2.0 * avg2) + avg3) / 7.0

    return result


def ultosc_polars(
    high: str,
    low_col: str,
    close_col: str,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> pl.Expr:
    """
    Ultimate Oscillator using Polars expressions.

    Note: Due to differences in how Polars and TA-Lib handle rolling operations,
    we use the Numba implementation through map_batches for exact accuracy.
    """
    # Use the Numba implementation through map_batches for exact TA-Lib compatibility
    return pl.struct([high, low_col, close_col]).map_batches(
        lambda x: pl.Series(
            ultosc_numba(
                x.struct.field(high).to_numpy(),
                x.struct.field(low_col).to_numpy(),
                x.struct.field(close_col).to_numpy(),
                timeperiod1,
                timeperiod2,
                timeperiod3,
            ),
        ),
        return_dtype=pl.Float64,
    )


@feature(
    name="ultosc",
    category="momentum",
    description="Ultimate Oscillator - weighted average of 3 stochastics",
    lookback=0,
    value_range=(0.0, 100.0),
    normalized=True,
    formula="",
    ta_lib_compatible=True,
)
def ultosc(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Ultimate Oscillator (ULTOSC).

    The Ultimate Oscillator is a momentum oscillator that incorporates three
    different time periods to reduce false signals and increase reliability.
    It combines short-term, intermediate-term, and long-term price action
    into one oscillator.

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    close : array-like or column name
        Close close
    timeperiod1 : int, default 7
        First time period (shortest)
    timeperiod2 : int, default 14
        Second time period (medium)
    timeperiod3 : int, default 28
        Third time period (longest)

    Returns
    -------
    array or Polars expression
        Ultimate Oscillator close (0-100 range)

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([10.5, 11.0, 11.5, 11.0, 10.5])
    >>> low = np.array([9.5, 10.0, 10.5, 10.0, 9.5])
    >>> close = np.array([10.0, 10.5, 11.0, 10.5, 10.0])
    >>> uo = ultosc(high, low, close, 2, 3, 4)

    Notes
    -----
    - Developed by Larry Williams in 1976
    - Combines three different timeframes
    - Values above 70 indicate overbought conditions
    - Values below 30 indicate oversold conditions
    - Divergences between price and indicator can signal reversals
    - Less prone to false signals than single-period oscillators
    """
    # Handle string inputs (Polars column names)
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        return ultosc_polars(high, low, close, timeperiod1, timeperiod2, timeperiod3)

    # Convert to numpy arrays
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, and close must have the same length")

    if timeperiod1 <= 0 or timeperiod2 <= 0 or timeperiod3 <= 0:
        raise ValueError("All time periods must be positive")

    if timeperiod1 >= timeperiod2 or timeperiod2 >= timeperiod3:
        raise ValueError(
            "Time periods must be in ascending order: timeperiod1 < timeperiod2 < timeperiod3",
        )

    return ultosc_numba(high, low, close, timeperiod1, timeperiod2, timeperiod3)


# Export the main function
__all__ = ["ultosc"]
