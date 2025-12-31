# mypy: disable-error-code="no-any-return"
"""
ATR (Average True Range) - TA-Lib compatible implementation.

ATR measures market volatility by calculating the average of True Range over a period.
Uses Wilder's smoothing method for exact TA-Lib compatibility.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.core.exceptions import InvalidParameterError


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def true_range_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate True Range exactly as TA-Lib does.

    True Range is the greatest of:
    - Today's high - today's low
    - |Yesterday's close - today's high|
    - |Yesterday's close - today's low|

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    close : npt.NDArray
        Close close

    Returns
    -------
    npt.NDArray
        True Range close (first value is NaN)
    """
    n = len(high)
    result = np.full(n, np.nan)

    # TA-Lib always skips the first bar for consistency
    for i in range(1, n):
        # Three possible ranges
        val1 = high[i] - low[i]  # Today's range
        val2 = abs(close[i - 1] - high[i])  # Yesterday's close to today's high
        val3 = abs(close[i - 1] - low[i])  # Yesterday's close to today's low

        # True Range is the maximum
        result[i] = max(val1, val2, val3)

    return result


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def atr_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period: int = 14,
) -> npt.NDArray[np.float64]:
    """
    ATR calculation exactly replicating TA-Lib algorithm.

    Uses Wilder's smoothing method:
    1. First ATR = SMA of first 'period' True Range close
    2. Subsequent ATR = ((prev_ATR * (period-1)) + current_TR) / period

    Parameters
    ----------
    high : npt.NDArray
        High close
    low : npt.NDArray
        Low close
    close : npt.NDArray
        Close close
    period : int, default 14
        Number of periods for ATR calculation

    Returns
    -------
    npt.NDArray
        ATR close exactly matching TA-Lib
    """
    n = len(high)
    result = np.full(n, np.nan)

    # Special case: period <= 1 just returns True Range
    if period <= 1:
        return true_range_numba(high, low, close)

    # Calculate True Range
    tr = true_range_numba(high, low, close)

    # Need at least period + 1 bars (1 for TR lookback + period for SMA)
    if n < period + 1:
        return result

    # First ATR value is SMA of first 'period' TR close
    # Start from index 1 (since TR[0] is NaN)
    sum_tr = 0.0
    count = 0

    for i in range(1, min(period + 1, n)):
        if not np.isnan(tr[i]):
            sum_tr += tr[i]
            count += 1

    if count < period:
        return result

    # First ATR at index period
    prev_atr = sum_tr / period
    result[period] = prev_atr

    # Subsequent close use Wilder's smoothing
    for i in range(period + 1, n):
        if not np.isnan(tr[i]):
            # ATR = ((prev_ATR * (period-1)) + current_TR) / period
            prev_atr = (prev_atr * (period - 1) + tr[i]) / period
            result[i] = prev_atr

    return result


def atr_polars(
    high: str,
    low_col: str,
    close_col: str,
    period: int = 14,
) -> pl.Expr:
    """
    ATR using Polars - delegates to Numba for exact compatibility.

    Parameters
    ----------
    high : str
        Column name for high close
    low_col : str
        Column name for low close
    close_col : str
        Column name for close close
    period : int, default 14
        Number of periods for ATR calculation

    Returns
    -------
    pl.Expr
        Polars expression for ATR calculation
    """
    return pl.struct([high, low_col, close_col]).map_batches(
        lambda s: pl.Series(
            atr_numba(
                s.struct.field(high).to_numpy(),
                s.struct.field(low_col).to_numpy(),
                s.struct.field(close_col).to_numpy(),
                period,
            ),
        ),
        return_dtype=pl.Float64,
    )


@feature(
    name="atr",
    category="volatility",
    description="ATR - Average True Range",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def atr(
    high: npt.NDArray[np.float64] | pl.Series,
    low: npt.NDArray[np.float64] | pl.Series,
    close: npt.NDArray[np.float64] | pl.Series,
    period: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    ATR (Average True Range) exactly matching TA-Lib.

    ATR measures volatility by calculating the average of True Range close
    using Wilder's smoothing method.

    Parameters
    ----------
    high : array-like
        High close
    low : array-like
        Low close
    close : array-like
        Close close
    period : int, default 14
        Number of periods for ATR calculation
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        ATR close exactly matching TA-Lib

    Examples
    --------
    >>> import numpy as np
    >>> high = np.array([48.70, 48.72, 48.90, 48.87, 48.82])
    >>> low = np.array([47.79, 48.14, 48.39, 48.37, 48.24])
    >>> close = np.array([48.16, 48.61, 48.75, 48.63, 48.74])
    >>> atr_values = atr(high, low, close, period=14)

    Notes
    -----
    - First value at index 0 is always NaN (TR needs previous close)
    - Second ATR value appears at index 'period' (e.g., 14)
    - Uses Wilder's smoothing: ATR = ((prev_ATR * (period-1)) + TR) / period
    - Exact replication of TA-Lib ta_ATR.c algorithm
    """
    # Validate parameters
    if period < 1:
        raise InvalidParameterError(f"period must be >= 1, got {period}")

    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        # Column names provided for Polars
        return atr_polars(high, low, close, period)

    # Convert to numpy if needed
    if isinstance(high, pl.Series):
        high = high.to_numpy()
    if isinstance(low, pl.Series):
        low = low.to_numpy()
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return atr_numba(high, low, close, period)


# Export all functions
__all__ = ["atr", "atr_numba", "atr_polars", "true_range_numba"]
