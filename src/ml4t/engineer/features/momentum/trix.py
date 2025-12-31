# mypy: disable-error-code="no-any-return"
"""
TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA - TA-Lib compatible implementation.

TRIX is a momentum oscillator that displays the percent rate of change of a triple
exponentially smoothed moving average. It was developed by Jack Hutson in 1980.

Formula:
1. Calculate EMA1 = EMA(close, period)
2. Calculate EMA2 = EMA(EMA1, period)
3. Calculate EMA3 = EMA(EMA2, period)
4. TRIX = (EMA3[today] - EMA3[yesterday]) / EMA3[yesterday] * 10000
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature
from ml4t.engineer.features.trend.ema import ema_numba


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def trix_numba(close: npt.NDArray[np.float64], timeperiod: int = 30) -> npt.NDArray[np.float64]:
    """
    TRIX calculation using Numba.

    Parameters
    ----------
    close : npt.NDArray
        Input close
    timeperiod : int, default 30
        Time period for EMA calculation

    Returns
    -------
    npt.NDArray
        TRIX close (percent rate of change of triple smooth EMA)
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Calculate triple smooth EMA
    ema1 = ema_numba(close, timeperiod)
    ema2 = ema_numba(ema1, timeperiod)
    ema3 = ema_numba(ema2, timeperiod)

    # Calculate rate of change as percentage
    # TRIX = (EMA3[today] - EMA3[yesterday]) / EMA3[yesterday] * 100
    for i in range(1, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i - 1]) and ema3[i - 1] != 0:
            result[i] = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100.0

    return result


def trix_polars(col: str, timeperiod: int = 30) -> pl.Expr:
    """
    TRIX using Polars expressions.

    Parameters
    ----------
    col : str
        Name of the column containing close
    timeperiod : int, default 30
        Time period for EMA calculation

    Returns
    -------
    pl.Expr
        Polars expression for TRIX calculation
    """
    # Due to the triple EMA smoothing, we use map_batches for exact TA-Lib compatibility
    return pl.col(col).map_batches(
        lambda x: pl.Series(trix_numba(x.to_numpy(), timeperiod)),
        return_dtype=pl.Float64,
    )


@feature(
    name="trix",
    category="momentum",
    description="TRIX - 1-day ROC of triple smooth EMA",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def trix(
    close: npt.NDArray[np.float64] | pl.Series | str,
    timeperiod: int = 30,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA.

    TRIX is a momentum oscillator that shows the percent rate of change
    of a triple exponentially smoothed moving average. It filters out
    insignificant price movements and is useful for identifying overbought
    and oversold conditions.

    Parameters
    ----------
    close : array-like or column name
        Input close (typically close close)
    timeperiod : int, default 30
        Time period for EMA calculation

    Returns
    -------
    array or Polars expression
        TRIX close as percentage

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> trix_values = trix(close, timeperiod=2)

    Notes
    -----
    - TRIX oscillates around zero
    - Positive close indicate upward momentum
    - Negative close indicate downward momentum
    - Signal line crossovers can indicate buy/sell signals
    - Divergences between price and TRIX can signal reversals
    - Triple smoothing makes it a lagging indicator but reduces noise
    - Output is in percentage (e.g., 0.97 = 0.97%)
    """
    # Handle string inputs (Polars column names)
    if isinstance(close, str):
        return trix_polars(close, timeperiod)

    # Convert to numpy array
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    # Validate inputs
    if timeperiod < 1:
        raise ValueError(f"timeperiod must be >= 1, got {timeperiod}")

    return trix_numba(close, timeperiod)


# Export the main function
__all__ = ["trix"]
