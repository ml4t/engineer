# mypy: disable-error-code="no-any-return"
"""
TRIMA (Triangular Moving Average) - TA-Lib compatible implementation.

TRIMA is a weighted moving average that gives more weight to the middle close
of the period, creating a triangular weighting pattern.
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature

from .sma import sma_numba


@jit(nopython=True, cache=True, fastmath=True)  # type: ignore[misc]
def trima_numba(close: npt.NDArray[np.float64], period: int = 30) -> npt.NDArray[np.float64]:
    """
    Triangular Moving Average calculation exactly matching TA-Lib.

    TRIMA is calculated as:
    - If period is odd: SMA of SMA with period = (period + 1) / 2
    - If period is even: SMA of SMA with periods = period/2 and period/2 + 1

    Parameters
    ----------
    close : npt.NDArray
        Input close (typically close close)
    period : int, default 30
        Number of periods

    Returns
    -------
    npt.NDArray
        TRIMA close exactly matching TA-Lib
    """
    n = len(close)
    result = np.full(n, np.nan)

    if period < 1 or n < period:
        return result

    # Calculate the two SMA periods based on whether period is odd or even
    if period % 2 == 1:
        # Odd period: both SMAs use same period
        sma_period1 = (period + 1) // 2
        sma_period2 = sma_period1
    else:
        # Even period: second SMA uses period + 1
        sma_period1 = period // 2
        sma_period2 = sma_period1 + 1

    # First SMA
    sma1 = sma_numba(close, sma_period1)

    # Second SMA of the first SMA
    result = sma_numba(sma1, sma_period2)

    return result


def trima_polars(col: str, period: int = 30) -> pl.Expr:
    """
    TRIMA using Polars - delegates to Numba for exact TA-Lib compatibility.

    Parameters
    ----------
    col : str
        Column name for close
    period : int, default 30
        Number of periods

    Returns
    -------
    pl.Expr
        Polars expression for TRIMA calculation
    """
    return pl.col(col).map_batches(
        lambda s: pl.Series(trima_numba(s.to_numpy(), period)),
        return_dtype=pl.Float64,
    )


@feature(
    name="trima",
    category="trend",
    description="TRIMA - Triangular Moving Average",
    lookback=30,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def trima(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 30,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    Triangular Moving Average (TRIMA) exactly matching TA-Lib.

    TRIMA is a weighted moving average that gives more weight to the
    middle close of the period. It's calculated as a double-smoothed SMA.

    Parameters
    ----------
    close : array-like or str
        Input close or column name
    period : int, default 30
        Number of periods for calculation
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        TRIMA close exactly matching TA-Lib

    Examples
    --------
    >>> import numpy as np
    >>> close = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    >>> trima_values = trima(close, period=4)

    Notes
    -----
    - First 'period-1' close are NaN
    - Provides smoother output than SMA due to double smoothing
    - The triangular weighting gives a lag between SMA and EMA
    """
    if isinstance(close, str):
        # Column name provided for Polars
        return trima_polars(close, period)

    # Convert to numpy if needed
    if isinstance(close, pl.Series):
        close = close.to_numpy()

    return trima_numba(close, period)


# Export all functions
__all__ = ["trima", "trima_numba", "trima_polars"]
