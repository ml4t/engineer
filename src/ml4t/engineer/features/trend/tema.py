# mypy: disable-error-code="no-any-return"
"""
TEMA (Triple Exponential Moving Average) - TA-Lib compatible implementation.

TEMA offers a moving average with less lag than the traditional EMA.
Formula: TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
where EMA1 = EMA(close), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
"""

import numpy as np
import numpy.typing as npt
import polars as pl
from numba import jit

from ml4t.engineer.core.decorators import feature

from .ema import ema_numba


@jit(nopython=True, cache=True, fastmath=False)  # type: ignore[misc]
def tema_numba(close: npt.NDArray[np.float64], period: int = 30) -> npt.NDArray[np.float64]:
    """
    TEMA calculation exactly replicating TA-Lib algorithm.

    TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
    where EMA1 = EMA(close), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)

    Parameters
    ----------
    close : npt.NDArray
        Input close
    period : int, default 30
        Time period for TEMA calculation

    Returns
    -------
    npt.NDArray
        TEMA close exactly matching TA-Lib
    """
    n = len(close)
    result = np.full(n, np.nan)

    # Calculate the three EMAs
    ema1 = ema_numba(close, period)
    ema2 = ema_numba(ema1, period)
    ema3 = ema_numba(ema2, period)

    # Calculate TEMA where all three EMAs are valid
    for i in range(n):
        if not np.isnan(ema1[i]) and not np.isnan(ema2[i]) and not np.isnan(ema3[i]):
            result[i] = 3.0 * ema1[i] - 3.0 * ema2[i] + ema3[i]

    return result


def tema_polars(column: str, period: int = 30) -> pl.Expr:
    """
    TEMA using Polars - delegates to Numba for exact compatibility.

    Parameters
    ----------
    column : str
        Column name to apply TEMA to
    period : int, default 30
        Time period for TEMA calculation

    Returns
    -------
    pl.Expr
        Polars expression for TEMA calculation
    """
    return pl.col(column).map_batches(
        lambda s: pl.Series(tema_numba(s.to_numpy(), period)),
        return_dtype=pl.Float64,
    )


@feature(
    name="tema",
    category="trend",
    description="TEMA - Triple Exponential Moving Average",
    lookback=0,
    normalized=False,
    formula="",
    ta_lib_compatible=True,
)
def tema(
    close: npt.NDArray[np.float64] | pl.Series | str,
    period: int = 30,
) -> npt.NDArray[np.float64] | pl.Expr:
    """
    TEMA (Triple Exponential Moving Average) exactly matching TA-Lib.

    TEMA offers a moving average with less lag than traditional EMA.

    Parameters
    ----------
    close : array-like or column name
        Input close
    period : int, default 30
        Time period for TEMA calculation
    implementation : str, default 'auto'
        Implementation to use: 'auto', 'numba', or 'polars'

    Returns
    -------
    array or Polars expression
        TEMA close exactly matching TA-Lib

    Examples
    --------
    >>> import numpy as np
    >>> close = np.random.randn(100).cumsum() + 100
    >>> tema_line = tema(close, 14)
    >>> # First ~39 close will be NaN due to triple smoothing

    Notes
    -----
    - TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
    - EMA1 = EMA(close), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
    - Lookback period is 3 * (period - 1)
    - Exact replication of TA-Lib ta_TEMA.c algorithm
    """
    if isinstance(close, str):
        return tema_polars(close, period)

    if isinstance(close, pl.Series):
        close = close.to_numpy()
    return tema_numba(close, period)


# Export all functions
__all__ = ["tema", "tema_numba", "tema_polars"]
