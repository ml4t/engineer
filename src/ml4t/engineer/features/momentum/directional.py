# mypy: disable-error-code="no-any-return"
"""
Directional Movement indicators - TA-Lib compatible implementation.

Includes PLUS_DI, MINUS_DI, and DX indicators which are components of the ADX system.

The directional movement concept measures the strength of price movements in positive
and negative directions.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def calculate_directional_movement_nb(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate Plus and Minus Directional Movement.

    Plus DM = Current High - Previous High (if positive and > Minus DM)
    Minus DM = Previous Low - Current Low (if positive and > Plus DM)
    """
    n = len(high)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]

        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        elif low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff

    return plus_dm, minus_dm


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def wilders_smoothing_nb(close: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    """
    Wilder's smoothing method matching TA-Lib implementation.

    TA-Lib approach (corrected based on DM analysis):
    - Sum close from index 1 to period-1 (exclusive end)
    - First output at index period-1 (not period)
    - Output sum directly (don't divide by period)
    """
    n = len(close)
    result = np.full(n, np.nan)

    if n <= period:
        return result

    # TA-Lib compatibility: Sum from index 1 to period-1 (exclusive end)
    sum_val = 0.0
    for i in range(1, period):
        if not np.isnan(close[i]):
            sum_val += close[i]

    # Store running sum for Wilder's calculation
    prev_sum = sum_val

    # First output at index period-1 (TA-Lib style)
    result[period - 1] = sum_val

    # Wilder's smoothing: new_sum = prev_sum - prev_sum/period + new_value
    for i in range(period, n):
        if not np.isnan(close[i]):
            prev_sum = prev_sum - prev_sum / period + close[i]
            result[i] = prev_sum

    return result


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def plus_di_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    timeperiod: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Plus Directional Indicator calculation using single-pass optimized algorithm.

    PLUS_DI = 100 * Smoothed Plus DM / Smoothed True Range
    """
    n = len(high)
    result = np.full(n, np.nan)

    if n <= timeperiod:
        return result

    # Calculate raw close in single pass
    plus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        # Directional Movement calculations
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]

        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff

        # True Range calculation - optimized
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Combined Wilder's smoothing to avoid duplicate loops
    if n <= timeperiod:
        return result

    # Initialize sums for Wilder's smoothing
    sum_plus_dm = 0.0
    sum_tr = 0.0

    # Calculate initial sums for first timeperiod close
    for i in range(1, timeperiod):
        sum_plus_dm += plus_dm[i]
        sum_tr += tr[i]

    # Apply Wilder's smoothing starting from index timeperiod
    for i in range(timeperiod, n):
        # Wilder's smoothing: new_sum = prev_sum - prev_sum/period + new_value
        sum_plus_dm = sum_plus_dm - sum_plus_dm / timeperiod + plus_dm[i]
        sum_tr = sum_tr - sum_tr / timeperiod + tr[i]

        if sum_tr != 0:
            result[i] = 100.0 * sum_plus_dm / sum_tr
        else:
            result[i] = 0.0

    return result


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def minus_di_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    timeperiod: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Minus Directional Indicator calculation using single-pass optimized algorithm.

    MINUS_DI = 100 * Smoothed Minus DM / Smoothed True Range
    """
    n = len(high)
    result = np.full(n, np.nan)

    if n <= timeperiod:
        return result

    # Calculate raw close in single pass
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        # Directional Movement calculations
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]

        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff

        # True Range calculation - optimized
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Combined Wilder's smoothing to avoid duplicate loops
    if n <= timeperiod:
        return result

    # Initialize sums for Wilder's smoothing
    sum_minus_dm = 0.0
    sum_tr = 0.0

    # Calculate initial sums for first timeperiod close
    for i in range(1, timeperiod):
        sum_minus_dm += minus_dm[i]
        sum_tr += tr[i]

    # Apply Wilder's smoothing starting from index timeperiod
    for i in range(timeperiod, n):
        # Wilder's smoothing: new_sum = prev_sum - prev_sum/period + new_value
        sum_minus_dm = sum_minus_dm - sum_minus_dm / timeperiod + minus_dm[i]
        sum_tr = sum_tr - sum_tr / timeperiod + tr[i]

        if sum_tr != 0:
            result[i] = 100.0 * sum_minus_dm / sum_tr
        else:
            result[i] = 0.0

    return result


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def dx_numba(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    timeperiod: int = 14,
) -> npt.NDArray[np.float64]:
    """
    Directional Movement Index calculation using Numba.

    DX = 100 * |Plus DI - Minus DI| / (Plus DI + Minus DI)
    """
    n = len(high)
    result = np.full(n, np.nan)

    if n <= timeperiod:
        return result

    # Calculate Plus DI and Minus DI
    plus_di = plus_di_numba(high, low, close, timeperiod)
    minus_di = minus_di_numba(high, low, close, timeperiod)

    # Calculate DX
    for i in range(timeperiod, n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            sum_di = plus_di[i] + minus_di[i]
            if sum_di != 0:
                result[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / sum_di
            else:
                result[i] = 0.0

    return result


def plus_di_polars(
    high: str,
    low_col: str,
    close_col: str,
    timeperiod: int = 14,
) -> pl.Expr:
    """Plus Directional Indicator using Polars expressions."""
    # This is a complex calculation that's better done with apply
    return pl.struct([high, low_col, close_col]).map_batches(
        lambda x: pl.Series(
            plus_di_numba(
                x.struct.field(high).to_numpy(),
                x.struct.field(low_col).to_numpy(),
                x.struct.field(close_col).to_numpy(),
                timeperiod,
            ),
        ),
    )


def minus_di_polars(
    high: str,
    low_col: str,
    close_col: str,
    timeperiod: int = 14,
) -> pl.Expr:
    """Minus Directional Indicator using Polars expressions."""
    return pl.struct([high, low_col, close_col]).map_batches(
        lambda x: pl.Series(
            minus_di_numba(
                x.struct.field(high).to_numpy(),
                x.struct.field(low_col).to_numpy(),
                x.struct.field(close_col).to_numpy(),
                timeperiod,
            ),
        ),
    )


def dx_polars(
    high: str,
    low_col: str,
    close_col: str,
    timeperiod: int = 14,
) -> pl.Expr:
    """Directional Movement Index using Polars expressions."""
    return pl.struct([high, low_col, close_col]).map_batches(
        lambda x: pl.Series(
            dx_numba(
                x.struct.field(high).to_numpy(),
                x.struct.field(low_col).to_numpy(),
                x.struct.field(close_col).to_numpy(),
                timeperiod,
            ),
        ),
    )


@feature(
    name="plus_di",
    category="momentum",
    description="Plus Directional Indicator - upward trend strength",
    lookback="period",
    normalized=True,
    value_range=(0.0, 100.0),
    formula="+DI = 100 * EMA(+DM) / ATR",
    ta_lib_compatible=True,
    input_type="HLC",
    parameters={"period": 14},
    tags=["directional", "trend"],
)
def plus_di(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Plus Directional Indicator (+DI).

    Measures the strength of upward price movement as part of the
    Directional Movement System developed by Welles Wilder.

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    close : array-like or column name
        Close close
    timeperiod : int, default 14
        Number of periods for smoothing

    Returns
    -------
    array or Polars expression
        Plus DI close (0-100 scale)
    """
    # Handle string inputs (Polars column names)
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        return plus_di_polars(high, low, close, timeperiod)

    # Validate period
    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

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

    return plus_di_numba(high, low, close, timeperiod)


@feature(
    name="minus_di",
    category="momentum",
    description="Minus Directional Indicator - downward trend strength",
    lookback="period",
    normalized=True,
    value_range=(0.0, 100.0),
    formula="-DI = 100 * EMA(-DM) / ATR",
    ta_lib_compatible=True,
    input_type="HLC",
    parameters={"period": 14},
    tags=["directional", "trend"],
)
def minus_di(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Minus Directional Indicator (-DI).

    Measures the strength of downward price movement as part of the
    Directional Movement System developed by Welles Wilder.

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    close : array-like or column name
        Close close
    timeperiod : int, default 14
        Number of periods for smoothing

    Returns
    -------
    array or Polars expression
        Minus DI close (0-100 scale)
    """
    # Handle string inputs (Polars column names)
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        return minus_di_polars(high, low, close, timeperiod)

    # Validate period
    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

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

    return minus_di_numba(high, low, close, timeperiod)


@feature(
    name="dx",
    category="momentum",
    description="Directional Movement Index - strength of directional movement",
    lookback=0,
    normalized=True,
    value_range=(0.0, 100.0),
    formula="",
    ta_lib_compatible=True,
)
def dx(
    high: npt.NDArray[np.float64] | pl.Series | str,
    low: npt.NDArray[np.float64] | pl.Series | str,
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 14,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Directional Movement Index (DX).

    Measures the strength of the directional movement regardless of direction.
    Used as a component in calculating ADX.

    Parameters
    ----------
    high : array-like or column name
        High close
    low : array-like or column name
        Low close
    close : array-like or column name
        Close close
    timeperiod : int, default 14
        Number of periods for smoothing

    Returns
    -------
    array or Polars expression
        DX close (0-100 scale)
    """
    # Handle string inputs (Polars column names)
    if isinstance(high, str) and isinstance(low, str) and isinstance(close, str):
        return dx_polars(high, low, close, timeperiod)

    # Validate period
    if timeperiod <= 0:
        raise ValueError("timeperiod must be > 0")

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

    return dx_numba(high, low, close, timeperiod)


# Export the functions
__all__ = ["dx", "minus_di", "plus_di"]
