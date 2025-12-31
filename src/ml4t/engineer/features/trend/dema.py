# mypy: disable-error-code="no-any-return"
"""
DEMA (Double Exponential Moving Average) - TA-Lib compatible implementation.

DEMA offers a moving average with less lag than the traditional EMA.
Formula: DEMA = 2 * EMA(close, period) - EMA(EMA(close, period), period)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature

from .ema import ema_numba


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def dema_numba(close: npt.NDArray[np.float64], period: int = 30) -> npt.NDArray[np.float64]:
    """
    DEMA calculation exactly replicating TA-Lib algorithm.

    Based on ta_DEMA.c with exact buffer management and timing.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    period : int, default 30
        Time period for DEMA calculation

    Returns
    -------
    npt.NDArray
        DEMA close exactly matching TA-Lib
    """
    n = len(close)

    # Calculate lookbacks exactly like TA-Lib
    lookback_ema = period - 1
    lookback_total = lookback_ema * 2

    # Adjust start index
    start_idx = lookback_total
    end_idx = n - 1

    if start_idx > end_idx:
        return np.full(n, np.nan)

    # Calculate first EMA for full data
    first_ema = ema_numba(close, period)

    # Extract portion of first EMA starting from where TA-Lib needs it
    # TA-Lib calls INT_EMA(startIdx-lookbackEMA, endIdx, ...)
    first_ema_start = start_idx - lookback_ema
    valid_start = max(first_ema_start, period - 1)

    # Create buffer for second EMA calculation matching TA-Lib's approach
    first_ema_for_second = first_ema[valid_start:n]

    # Calculate second EMA on the extracted portion
    second_ema_raw = ema_numba(first_ema_for_second, period)

    # Map second EMA back to full array
    second_ema = np.full(n, np.nan)
    second_ema_valid_start = period - 1  # relative to first_ema_for_second
    output_start = valid_start + second_ema_valid_start

    for i in range(len(second_ema_raw)):
        if not np.isnan(second_ema_raw[i]):
            idx = valid_start + i
            if idx < n:
                second_ema[idx] = second_ema_raw[i]

    # Calculate DEMA: 2 * firstEMA - secondEMA
    result = np.full(n, np.nan)
    for i in range(output_start, n):
        if not np.isnan(second_ema[i]) and not np.isnan(first_ema[i]):
            result[i] = 2.0 * first_ema[i] - second_ema[i]

    return result


def dema_polars(column: str, period: int = 30) -> pl.Expr:
    """
    DEMA using Polars - delegates to Numba for exact compatibility.

    Parameters
    ----------
    column : str
        Column name to apply DEMA to
    period : int, default 30
        Time period for DEMA calculation

    Returns
    -------
    pl.Expr
        Polars expression for DEMA calculation
    """
    return pl.col(column).map_batches(
        lambda s: pl.Series(dema_numba(s.to_numpy(), period)),
        return_dtype=pl.Float64,
    )


@feature(
    name="dema",
    category="trend",
    description="DEMA - Double Exponential Moving Average",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def dema(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 30,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    DEMA (Double Exponential Moving Average) exactly matching TA-Lib.

    DEMA offers a moving average with less lag than traditional EMA.

    Parameters
    ----------
    close : array-like or column name
        Input close
    period : int, default 30
        Time period for DEMA calculation
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        DEMA close exactly matching TA-Lib

    Examples
    --------
    >>> import numpy as np
    >>> close = np.random.randn(100).cumsum() + 100
    >>> dema_line = dema(close, 14)
    >>> # First ~56 close will be NaN due to double smoothing

    Notes
    -----
    - DEMA = 2 * EMA(close, period) - EMA(EMA(close, period), period)
    - Lookback period is 2 * (period - 1)
    - Exact replication of TA-Lib ta_DEMA.c algorithm
    """
    if isinstance(close, str):
        return dema_polars(close, period)

    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return dema_numba(close, period)


# Export all functions
__all__ = ["dema", "dema_numba", "dema_polars"]
